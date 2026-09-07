//! Neo4j graph storage ( +
//! `Neo4jDatabase.kt`).
//!
//! Uses the workspace name as node label for multi-tenant isolation.
//! All batch operations use UNWIND for single-roundtrip performance.
//! Nodes are identified by the `entity_id` property; edges use the
//! `DIRECTED` relationship type with undirected matching semantics.

use async_trait::async_trait;
use easyrag_core::concurrency::{with_retry as retry_core, IsRetryable, RetryConfig};
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::model::{KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode, TenantScopeRef};
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::OnceLock;

use crate::drop_result::{drop_error, drop_success};
use crate::scope_sql::{cypher_ws_label, TenantScopeCypher};

/// Query parameters expressed as JSON.
pub type Neo4jParams = serde_json::Map<String, JsonValue>;

/// One result record expressed as JSON. Nodes and relationships returned
/// as column values are encoded as JSON objects with reserved meta keys:
/// - `__kind__`: "node" or "rel"
/// - `__id__`: element id
/// - `__labels__`: node labels (nodes only)
/// - `__type__`: relationship type (relationships only)
pub type Neo4jRow = serde_json::Map<String, JsonValue>;

pub const META_KIND: &str = "__kind__";
pub const META_ID: &str = "__id__";
pub const META_LABELS: &str = "__labels__";
pub const META_TYPE: &str = "__type__";

/// Neo4j database operations abstraction.
/// Allows testing without a live Neo4j instance.
#[async_trait]
pub trait Neo4jOps: Send + Sync {
    /// Run a read query and return all records.
    async fn run(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<Vec<Neo4jRow>>;

    /// Run a write query within a transaction and return all records.
    async fn run_write(
        &self,
        query: &str,
        params: Neo4jParams,
    ) -> easyrag_core::Result<Vec<Neo4jRow>>;

    /// Run a write query without returning records (fire-and-forget style).
    async fn run_write_void(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<()>;
}

#[async_trait]
impl<T: Neo4jOps> Neo4jOps for std::sync::Arc<T> {
    async fn run(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<Vec<Neo4jRow>> {
        self.as_ref().run(query, params).await
    }

    async fn run_write(
        &self,
        query: &str,
        params: Neo4jParams,
    ) -> easyrag_core::Result<Vec<Neo4jRow>> {
        self.as_ref().run_write(query, params).await
    }

    async fn run_write_void(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<()> {
        self.as_ref().run_write_void(query, params).await
    }
}

// ---------------------------------------------------------------------------
// Cypher fragment library (Phase 1.6).
//
// Pure `fn -> String` builders (static dispatch, zero indirection) that
// single-source the recurring Cypher shapes hand-concatenated across ~24 query
// sites. Extracting them removes the duplication while the byte-level golden
// snapshots in `tests/neo4j_graph_storage.rs` prove the rendered Cypher is
// unchanged ([equivalent]). Shapes that occur once with a genuinely different
// form (the variable-length `-[..*0..N]-` traversal and the biz-conditional
// maps in `get_sub_graph`, the read-path `node_scope` wildcard, and the DDL
// `CREATE INDEX` statements) are deliberately NOT abstracted here — but they
// still interpolate `REL_TYPE`, since the relationship *name* is shared with
// every other pattern and with the edges handed back to callers, even where
// the pattern *shape* is unique.
// ---------------------------------------------------------------------------
mod cypher {
    use serde_json::Value as JsonValue;

    /// Relationship type every edge in the workspace graph carries.
    ///
    /// One type for all relations — the semantic type lives in the edge
    /// properties — so the same name appears in the match patterns below and in
    /// the `type` field of the `KnowledgeGraphEdge` values returned to callers.
    /// Declared once: a rename has to touch patterns and payloads together or
    /// reads silently stop matching writes.
    pub const REL_TYPE: &str = "DIRECTED";

    /// Workspace-scoped node pattern. `entity_key = Some(k)` renders the
    /// double-key form `({var}:{ws} {entity_id: k, biz_id: $biz})`; `None`
    /// renders the match-any side `({var}:{ws} {biz_id: $biz})`. `ws` is the
    /// already-escaped workspace label (see `ws_label`).
    pub fn node_pat(ws: &str, var: &str, entity_key: Option<&str>) -> String {
        match entity_key {
            Some(key) => format!("({var}:{ws} {{entity_id: {key}, biz_id: $biz}})"),
            None => format!("({var}:{ws} {{biz_id: $biz}})"),
        }
    }

    /// `DIRECTED` relationship connector: `-[{var}:DIRECTED]-` for undirected
    /// matching (read/degree paths) or `-[{var}:DIRECTED]->` for the directed
    /// MERGE/write paths.
    pub fn edge_pat(var: &str, directed: bool) -> String {
        if directed {
            format!("-[{var}:{REL_TYPE}]->")
        } else {
            format!("-[{var}:{REL_TYPE}]-")
        }
    }

    /// Degree-count pattern: an entity node joined to any biz-scoped node by an
    /// undirected `DIRECTED` edge —
    /// `({node_var} {entity_id: key})-[{edge_var}:DIRECTED]-({other_var} {biz_id})`.
    pub fn degree_pat(
        ws: &str,
        node_var: &str,
        entity_key: &str,
        edge_var: &str,
        other_var: &str,
    ) -> String {
        format!(
            "{}{}{}",
            node_pat(ws, node_var, Some(entity_key)),
            edge_pat(edge_var, false),
            node_pat(ws, other_var, None),
        )
    }

    /// `UNWIND ${list} AS {var}` batch prefix.
    pub fn unwind(list: &str, var: &str) -> String {
        format!("UNWIND ${list} AS {var}")
    }

    /// JSON string array for an UNWIND `$ids`/`$items` parameter.
    pub fn json_ids(ids: &[String]) -> Vec<JsonValue> {
        ids.iter().map(|id| JsonValue::String(id.clone())).collect()
    }
}

// ---------------------------------------------------------------------------
// Neo4jDatabase (live driver, neo4rs)
// ---------------------------------------------------------------------------

/// Connection settings (env variable names mirror the original).
#[derive(Clone, Debug)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub database: String,
    /// Aligns with NEO4J_MAX_POOL_SIZE (default 100).
    pub max_connections: usize,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            user: "neo4j".to_string(),
            password: String::new(),
            database: "neo4j".to_string(),
            max_connections: 100,
        }
    }
}

const MAX_RETRIES: u32 = 3;
const RETRY_DELAY_MS: u64 = 500;

/// Graph-driver retry policy: base 500ms, factor 2, 3 attempts, no jitter.
///
/// This reproduces the curve of the hand-rolled loop it replaces —
/// `RETRY_DELAY_MS * (attempt + 1)` slept 500ms then 1000ms, and
/// `500 * 2^(attempt-1)` sleeps exactly the same — so moving the loop onto
/// `core::with_retry` is [equivalent]. `max_delay` never binds at 3 attempts;
/// it is set so a future attempt-count bump cannot produce an unbounded wait.
const RETRY_CONFIG: RetryConfig = RetryConfig {
    max_attempts: MAX_RETRIES,
    base_delay: std::time::Duration::from_millis(RETRY_DELAY_MS),
    max_delay: std::time::Duration::from_secs(10),
    exponential_base: 2.0,
    jitter: false,
};

/// Adapter letting `neo4rs::Error` participate in `core::with_retry`.
///
/// `core::with_retry` decides retryability through the `IsRetryable` trait,
/// which cannot be implemented for the foreign `neo4rs::Error` (orphan rule).
/// The newtype carries the driver error plus the verdict `is_transient` already
/// computes, and lives only for the duration of one call. The `bool` is captured
/// up front rather than recomputed in `is_retryable` so the classification stays
/// the single `is_transient` definition.
#[derive(Debug)]
struct Neo4jRetryError {
    inner: neo4rs::Error,
    retryable: bool,
}

impl IsRetryable for Neo4jRetryError {
    fn is_retryable(&self) -> bool {
        self.retryable
    }
}

/// Map an exhausted retry back to the two messages callers have always seen.
///
/// The verdict is re-read rather than tracked separately: a non-transient error
/// means `core::with_retry` bailed on the first attempt (so the message names the
/// driver error), while a transient one means the budget ran out (so the message
/// names the attempt count). Free function, not a method, because constructing a
/// `Neo4jDatabase` needs a live connection — this way the mapping is unit-testable.
fn neo4j_retry_error(e: Neo4jRetryError) -> CoreError {
    if e.is_retryable() {
        CoreError::Internal(format!(
            "Neo4j operation failed after {MAX_RETRIES} retries: {}",
            e.inner
        ))
    } else {
        CoreError::Internal(format!("Neo4j error: {}", e.inner))
    }
}

/// Neo4j connection manager wrapping neo4rs with retry for transient
/// failures.
#[derive(Clone)]
pub struct Neo4jDatabase {
    graph: neo4rs::Graph,
    database: String,
}

impl Neo4jDatabase {
    pub async fn new(config: Neo4jConfig) -> easyrag_core::Result<Self> {
        let mut builder = neo4rs::ConfigBuilder::new()
            .uri(config.uri)
            .user(config.user)
            .password(config.password)
            .max_connections(config.max_connections);
        if !config.database.is_empty() {
            builder = builder.db(config.database.clone());
        }
        let conf = builder
            .build()
            .map_err(|e| CoreError::Internal(format!("Neo4j config error: {e}")))?;
        let graph = neo4rs::Graph::connect(conf)
            .await
            .map_err(|e| CoreError::Internal(format!("Neo4j connection failed: {e}")))?;
        Ok(Self {
            graph,
            database: config.database,
        })
    }

    /// Verify connectivity to the Neo4j server.
    pub async fn verify_connectivity(&self) -> bool {
        let q = neo4rs::query("RETURN 1 AS ok");
        match self.collect_read(q).await {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("Neo4j connectivity check failed: {e}");
                false
            }
        }
    }

    fn build_query(cypher: &str, params: Neo4jParams) -> neo4rs::Query {
        let mut q = neo4rs::query(cypher);
        for (key, value) in params {
            q = q.param(&key, json_to_bolt(&value));
        }
        q
    }

    fn is_transient(e: &neo4rs::Error) -> bool {
        match e {
            neo4rs::Error::IOError { .. } | neo4rs::Error::ConnectionError => true,
            neo4rs::Error::Neo4j(err) => err.code().starts_with("Neo.TransientError"),
            _ => false,
        }
    }

    /// Run `make` under the shared retry policy.
    ///
    /// Delegates the loop to `core::with_retry` (Phase 3.10) so the graph and
    /// SQL backends cannot drift on attempts, backoff or what counts as
    /// transient. `map_err` (not an `async` block) adapts the driver error:
    /// it yields an *owned* future type, which is what `with_retry`'s single
    /// `Fut` parameter requires — an async block would borrow `make` and could
    /// not be named as one type across calls.
    ///
    /// Both original outcome messages are preserved by re-reading the verdict on
    /// the way out: a non-transient failure surfaces immediately, an exhausted
    /// budget reports the attempt count.
    async fn with_retry<T, F, Fut>(&self, make: F) -> easyrag_core::Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, neo4rs::Error>>,
    {
        use futures::TryFutureExt;

        let outcome = retry_core(
            &RETRY_CONFIG,
            Some(|e: &Neo4jRetryError, attempt: u32, _delay: std::time::Duration| {
                tracing::warn!(
                    "Neo4j transient error (attempt {attempt}/{MAX_RETRIES}): {}",
                    e.inner
                );
            }),
            move || {
                make().map_err(|e| Neo4jRetryError {
                    retryable: Self::is_transient(&e),
                    inner: e,
                })
            },
        )
        .await;

        outcome.map_err(neo4j_retry_error)
    }

    async fn drain_rows<S>(
        mut stream: std::pin::Pin<Box<S>>,
    ) -> Result<Vec<Neo4jRow>, neo4rs::Error>
    where
        S: futures::TryStream<Ok = neo4rs::Row, Error = neo4rs::Error> + ?Sized,
    {
        // `try_poll_next` is a TryStream primitive that does not require the
        // `Stream::Item == Result<..>` normalization that `try_next` needs
        // (neo4rs 0.8 returns an opaque `impl TryStream`).
        let mut out = Vec::new();
        loop {
            match futures::future::poll_fn(|cx| stream.as_mut().try_poll_next(cx)).await {
                Some(Ok(row)) => out.push(row_to_json(&row)),
                Some(Err(e)) => return Err(e),
                None => return Ok(out),
            }
        }
    }

    async fn collect_read(&self, q: neo4rs::Query) -> Result<Vec<Neo4jRow>, neo4rs::Error> {
        let db = self.database.clone();
        let stream = if db.is_empty() {
            self.graph.execute(q).await?
        } else {
            self.graph.execute_on(&db, q).await?
        };
        Self::drain_rows(Box::pin(stream.into_stream())).await
    }

    async fn collect_write(
        &self,
        q: neo4rs::Query,
    ) -> Result<Vec<Neo4jRow>, neo4rs::Error> {
        let mut txn = self.graph.start_txn().await?;
        let stream = txn.execute(q).await?;
        // The stream borrows the transaction handle; draining consumes it so
        // the borrow ends before commit.
        let rows = Self::drain_rows(Box::pin(stream.into_stream(txn.handle()))).await?;
        txn.commit().await?;
        Ok(rows)
    }

    async fn exec_write_void(&self, q: neo4rs::Query) -> Result<(), neo4rs::Error> {
        let mut txn = self.graph.start_txn().await?;
        txn.run(q).await?;
        txn.commit().await?;
        Ok(())
    }
}

#[async_trait]
impl Neo4jOps for Neo4jDatabase {
    async fn run(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<Vec<Neo4jRow>> {
        let cypher = query.to_string();
        self.with_retry(|| {
            let q = Self::build_query(&cypher, params.clone());
            self.collect_read(q)
        })
        .await
    }

    async fn run_write(
        &self,
        query: &str,
        params: Neo4jParams,
    ) -> easyrag_core::Result<Vec<Neo4jRow>> {
        let cypher = query.to_string();
        self.with_retry(|| {
            let q = Self::build_query(&cypher, params.clone());
            self.collect_write(q)
        })
        .await
    }

    async fn run_write_void(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<()> {
        let cypher = query.to_string();
        self.with_retry(|| {
            let q = Self::build_query(&cypher, params.clone());
            self.exec_write_void(q)
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Bolt <-> JSON conversion
// ---------------------------------------------------------------------------

fn bolt_to_json(v: &neo4rs::BoltType) -> JsonValue {
    use neo4rs::BoltType;
    match v {
        BoltType::Null(_) => JsonValue::Null,
        BoltType::Boolean(b) => JsonValue::Bool(b.value),
        BoltType::Integer(i) => JsonValue::from(i.value),
        BoltType::Float(f) => JsonValue::from(f.value),
        BoltType::String(s) => JsonValue::String(s.value.clone()),
        BoltType::List(l) => {
            JsonValue::Array(l.value.iter().map(bolt_to_json).collect())
        }
        BoltType::Map(m) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in &m.value {
                obj.insert(k.value.clone(), bolt_to_json(v));
            }
            JsonValue::Object(obj)
        }
        BoltType::Node(n) => {
            let mut obj = serde_json::Map::new();
            obj.insert(META_KIND.to_string(), JsonValue::String("node".to_string()));
            obj.insert(META_ID.to_string(), JsonValue::from(n.id.value));
            let labels: Vec<JsonValue> = n
                .labels
                .value
                .iter()
                .map(|l| match l {
                    BoltType::String(s) => JsonValue::String(s.value.clone()),
                    other => JsonValue::String(other.to_string()),
                })
                .collect();
            obj.insert(META_LABELS.to_string(), JsonValue::Array(labels));
            for (k, v) in &n.properties.value {
                obj.insert(k.value.clone(), bolt_to_json(v));
            }
            JsonValue::Object(obj)
        }
        BoltType::Relation(r) => rel_to_json(r.id.value, &r.typ.value, &r.properties),
        BoltType::UnboundedRelation(r) => rel_to_json(r.id.value, &r.typ.value, &r.properties),
        other => JsonValue::String(format!("{other:?}")),
    }
}

fn rel_to_json(id: i64, typ: &str, props: &neo4rs::BoltMap) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert(META_KIND.to_string(), JsonValue::String("rel".to_string()));
    obj.insert(META_ID.to_string(), JsonValue::from(id));
    obj.insert(META_TYPE.to_string(), JsonValue::String(typ.to_string()));
    for (k, v) in &props.value {
        obj.insert(k.value.clone(), bolt_to_json(v));
    }
    JsonValue::Object(obj)
}

fn row_to_json(row: &neo4rs::Row) -> Neo4jRow {
    let mut out = Neo4jRow::new();
    // Row has no public key iterator; deserialize the whole attribute map.
    match row.to_strict::<std::collections::HashMap<String, neo4rs::BoltType>>() {
        Ok(attrs) => {
            for (key, v) in attrs {
                out.insert(key, bolt_to_json(&v));
            }
        }
        Err(_) => {
            // Single-column row: fall back to a positional "value" entry.
            if let Ok(v) = row.to::<neo4rs::BoltType>() {
                out.insert("value".to_string(), bolt_to_json(&v));
            }
        }
    }
    out
}

fn json_to_bolt(v: &JsonValue) -> neo4rs::BoltType {
    use neo4rs::BoltType;
    match v {
        JsonValue::Null => BoltType::Null(neo4rs::BoltNull),
        JsonValue::Bool(b) => BoltType::Boolean(neo4rs::BoltBoolean { value: *b }),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                BoltType::Integer(neo4rs::BoltInteger { value: i })
            } else {
                BoltType::Float(neo4rs::BoltFloat {
                    value: n.as_f64().unwrap_or(0.0),
                })
            }
        }
        JsonValue::String(s) => BoltType::String(neo4rs::BoltString::new(s)),
        JsonValue::Array(items) => BoltType::List(neo4rs::BoltList {
            value: items.iter().map(json_to_bolt).collect(),
        }),
        JsonValue::Object(m) => {
            let mut map = neo4rs::BoltMap::new();
            for (k, v) in m {
                map.put(neo4rs::BoltString::new(k), json_to_bolt(v));
            }
            BoltType::Map(map)
        }
    }
}

// ---------------------------------------------------------------------------
// Neo4jGraphStorage
// ---------------------------------------------------------------------------

/// Workspace name rule (identical to the postgres backend's
/// `validate_workspace`): the name is spliced into a backtick-quoted Cypher
/// label, and Cypher backquoted identifiers cannot contain backticks, so the
/// charset must be restricted up front to rule out Cypher injection.
fn validate_workspace(workspace: &str) -> easyrag_core::Result<()> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]{0,63}$").expect("valid regex")
    });
    if re.is_match(workspace) {
        Ok(())
    } else {
        Err(CoreError::Internal(format!(
            "invalid workspace name '{workspace}': must match \
             ^[a-zA-Z0-9_-][a-zA-Z0-9_-.]{{0,63}}$"
        )))
    }
}

/// Neo4j-backed GraphStorage implementation.
/// `biz_id` is a per-call filter value supplied to every data method; the
/// instance itself is bound to a workspace only.
pub struct Neo4jGraphStorage<O: Neo4jOps = Neo4jDatabase> {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    db: O,
}

impl<O: Neo4jOps> Neo4jGraphStorage<O> {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        db: O,
    ) -> easyrag_core::Result<Self> {
        Self::with_options(namespace, workspace, JsonMap::new(), None, db)
    }

    pub fn with_options(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
        db: O,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        validate_workspace(&workspace)?;
        Ok(Self {
            namespace: namespace.into(),
            workspace,
            global_config,
            embedding_func,
            db,
        })
    }

    /// Borrowed tenant scope for this backend (no allocation).
    fn scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a> {
        TenantScopeRef::new(&self.workspace, biz_id)
    }

    /// Escaped workspace label for Cypher queries (biz-independent).
    fn ws_label(&self) -> String {
        cypher_ws_label(&self.workspace)
    }

    /// Node label + biz-line scope fragment for Cypher MATCH: `` `ws` ``
    /// (read-path wildcard when `biz_id` is empty — lists every biz line of
    /// the workspace) or `` `ws` {biz_id: $biz} `` (exact biz line). Mirrors
    /// `PgScoped::list_scope_clause`: an empty biz_id is the read-path
    /// wildcard, never a literal match against the empty string. The rule
    /// itself lives in `TenantScopeCypher`, so the SQL and Cypher backends
    /// cannot drift apart on it.
    /// Write paths and single-node reads must NOT route through this helper
    /// — they keep the exact scope.
    fn node_scope(&self, biz_id: &str) -> String {
        self.scope(biz_id).node_label()
    }

    /// Fulltext index name (sanitized for Neo4j symbolic name rules).
    /// Derived per workspace; biz isolation comes from the `node.biz_id`
    /// post-filter applied on search results.
    fn fulltext_index_name(&self) -> String {
        let sanitize = |s: &str| -> String {
            s.chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
                .collect()
        };
        format!("`{}_entity_fulltext`", sanitize(&self.workspace))
    }

    fn params(entries: &[(&str, JsonValue)]) -> Neo4jParams {
        let mut map = Neo4jParams::new();
        for (k, v) in entries {
            map.insert((*k).to_string(), v.clone());
        }
        map
    }

    /// Cypher parameters pre-filled with the caller's `$biz`; every query
    /// filtering on biz_id must go through this helper.
    fn biz_params(biz_id: &str) -> Neo4jParams {
        Self::params(&[("biz", JsonValue::String(biz_id.to_string()))])
    }

    /// Merge `biz_params(biz_id)` with additional entries (entries win on
    /// collision).
    fn biz_params_with(biz_id: &str, entries: &[(&str, JsonValue)]) -> Neo4jParams {
        let mut map = Self::biz_params(biz_id);
        for (k, v) in entries {
            map.insert((*k).to_string(), v.clone());
        }
        map
    }

    /// Strip reserved meta keys from an encoded node/relationship object.
    fn strip_meta(obj: &JsonMap) -> JsonMap {
        obj.iter()
            .filter(|(k, _)| {
                !matches!(k.as_str(), META_KIND | META_ID | META_LABELS | META_TYPE)
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    fn as_object(v: &JsonValue) -> Option<&JsonMap> {
        match v {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    fn row_str(row: &Neo4jRow, key: &str) -> String {
        row.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    fn row_int(row: &Neo4jRow, key: &str) -> i64 {
        row.get(key).and_then(|v| v.as_i64()).unwrap_or(0)
    }

    fn row_bool(row: &Neo4jRow, key: &str) -> bool {
        row.get(key).and_then(|v| v.as_bool()).unwrap_or(false)
    }

    /// Node map = properties (with `entity_id` guaranteed present).
    fn node_to_map(node: &JsonValue) -> Option<JsonMap> {
        let obj = Self::as_object(node)?;
        let mut map = Self::strip_meta(obj);
        let entity_id = map
            .get("entity_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        map.insert("entity_id".to_string(), JsonValue::String(entity_id));
        Some(map)
    }

    fn node_labels(node: &JsonValue) -> Vec<String> {
        Self::as_object(node)
            .and_then(|o| o.get(META_LABELS))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn node_id(node: &JsonValue) -> String {
        Self::as_object(node)
            .and_then(|o| o.get("entity_id"))
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string()
    }

    /// Relationship properties (meta keys stripped).
    fn rel_props(rel: &JsonValue) -> JsonMap {
        Self::as_object(rel)
            .map(Self::strip_meta)
            .unwrap_or_default()
    }

    fn rel_element_id(rel: &JsonValue) -> String {
        Self::as_object(rel)
            .and_then(|o| o.get(META_ID))
            .map(|v| match v {
                JsonValue::Number(n) => n.to_string(),
                JsonValue::String(s) => s.clone(),
                _ => String::new(),
            })
            .unwrap_or_default()
    }

    async fn get_full_graph(&self, max_nodes: usize, biz_id: &str) -> easyrag_core::Result<KnowledgeGraph> {
        // Read-path: empty biz_id lists every biz line of the workspace
        // (see `node_scope`); the $biz param is still passed for the
        // non-wildcard branch.
        let node = self.node_scope(biz_id);
        // Effective LIMIT (guarded against 0 so is_truncated stays truthful).
        let limit = max_nodes.max(1);
        let node_records = self
            .db
            .run(
                &format!("MATCH (n:{node}) RETURN n LIMIT $limit"),
                Self::biz_params_with(biz_id, &[("limit", JsonValue::from(limit as i64))]),
            )
            .await?;
        let mut nodes = Vec::new();
        for record in &node_records {
            if let Some(n) = record.get("n") {
                nodes.push(KnowledgeGraphNode::new(
                    Self::node_id(n),
                    Self::node_labels(n),
                    Self::node_to_map(n).unwrap_or_default(),
                ));
            }
        }

        let node_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let edge_records = self
            .db
            .run(
                &format!(
                    "MATCH (a:{node}){}(b:{node}) \
                     RETURN a.entity_id AS src, b.entity_id AS tgt, r LIMIT $limit",
                    cypher::edge_pat("r", true)
                ),
                Self::biz_params_with(biz_id, &[(
                    "limit",
                    JsonValue::from(limit.saturating_mul(3).min(i64::MAX as usize) as i64),
                )]),
            )
            .await?;
        let mut edges = Vec::new();
        for record in &edge_records {
            let src = Self::row_str(record, "src");
            let tgt = Self::row_str(record, "tgt");
            if node_ids.contains(&src) && node_ids.contains(&tgt) {
                let rel = record.get("r").cloned().unwrap_or(JsonValue::Null);
                edges.push(KnowledgeGraphEdge::new(
                    Self::rel_element_id(&rel),
                    Some(cypher::REL_TYPE.to_string()),
                    src,
                    tgt,
                    Self::rel_props(&rel),
                ));
            }
        }

        Ok(KnowledgeGraph {
            is_truncated: nodes.len() >= limit,
            nodes,
            edges,
        })
    }

    async fn get_sub_graph(
        &self,
        start_label: &str,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        let safe_depth = max_depth.clamp(1, 10);
        let safe_max_nodes = max_nodes.clamp(1, 10_000);
        let ws = self.ws_label();
        // The variable-length traversal keeps its own shape (it is not an
        // `edge_pat` arm), but its relationship type is the same one every
        // other pattern matches on, so it interpolates `REL_TYPE` rather than
        // repeating the literal.
        let rel = cypher::REL_TYPE;
        // Read-path wildcard: an empty biz_id expands across every biz line
        // of the workspace (drops the biz_id predicate instead of matching
        // the literal empty string). See `node_scope`.
        let scope = self.scope(biz_id);
        let start_map = scope.entity_map("$label");
        let biz_where = scope
            .biz_predicate("n")
            .map(|p| format!("WHERE {p} "))
            .unwrap_or_default();
        // BFS-style expansion using variable-length path
        let records = self
            .db
            .run(
                &format!(
                    "MATCH path = (start:{ws} {start_map})-[:{rel}*0..{safe_depth}]-(connected:{ws}) \
                     UNWIND nodes(path) AS n \
                     {biz_where}WITH DISTINCT n LIMIT $maxNodes \
                     RETURN n"
                ),
                Self::biz_params_with(biz_id, &[
                    ("label", JsonValue::String(start_label.to_string())),
                    ("maxNodes", JsonValue::from(safe_max_nodes as i64)),
                ]),
            )
            .await?;

        let mut nodes = Vec::new();
        for record in &records {
            if let Some(n) = record.get("n") {
                nodes.push(KnowledgeGraphNode::new(
                    Self::node_id(n),
                    Self::node_labels(n),
                    Self::node_to_map(n).unwrap_or_default(),
                ));
            }
        }

        if nodes.is_empty() {
            return Ok(KnowledgeGraph {
                nodes: Vec::new(),
                edges: Vec::new(),
                is_truncated: false,
            });
        }

        // Get edges between collected nodes
        let node_ids: Vec<JsonValue> = nodes
            .iter()
            .map(|n| JsonValue::String(n.id.clone()))
            .collect();
        // Same read-path wildcard rule as `node_scope`, applied to the
        // `UNWIND`-bound source side of the match.
        let a_map = self.scope(biz_id).entity_map("srcId");
        let b_scope = self.node_scope(biz_id);
        let edge_records = self
            .db
            .run(
                &format!(
                    "{} \
                     MATCH (a:{ws} {a_map}){}(b:{b_scope}) \
                     WHERE b.entity_id IN $ids \
                     RETURN DISTINCT a.entity_id AS src, b.entity_id AS tgt, r",
                    cypher::unwind("ids", "srcId"),
                    cypher::edge_pat("r", false)
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(node_ids))]),
            )
            .await?;

        let mut edges = Vec::new();
        for record in &edge_records {
            let rel = record.get("r").cloned().unwrap_or(JsonValue::Null);
            edges.push(KnowledgeGraphEdge::new(
                Self::rel_element_id(&rel),
                Some(cypher::REL_TYPE.to_string()),
                Self::row_str(record, "src"),
                Self::row_str(record, "tgt"),
                Self::rel_props(&rel),
            ));
        }

        Ok(KnowledgeGraph {
            is_truncated: nodes.len() >= safe_max_nodes,
            nodes,
            edges,
        })
    }
}

/// Sanitize properties for Neo4j storage.
/// Neo4j only supports primitive types, strings, and lists of primitives.
/// Reserved META_* keys are dropped on the write side so that the read side
/// (`strip_meta`) and the wire encoding stay symmetric.
fn sanitize_props(data: JsonMap) -> JsonMap {
    data.into_iter()
        .filter(|(k, _)| !matches!(k.as_str(), META_KIND | META_ID | META_LABELS | META_TYPE))
        .map(|(k, v)| {
            let v = match v {
                JsonValue::String(_)
                | JsonValue::Bool(_)
                | JsonValue::Number(_)
                | JsonValue::Null => v,
                JsonValue::Array(items) => JsonValue::Array(
                    items
                        .into_iter()
                        .map(|it| match it {
                            JsonValue::Number(_) | JsonValue::Null => it,
                            other => JsonValue::String(other.to_string()),
                        })
                        .collect(),
                ),
                other => JsonValue::String(other.to_string()),
            };
            (k, v)
        })
        .collect()
}

#[async_trait]
impl<O: Neo4jOps> StorageNameSpace for Neo4jGraphStorage<O> {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    async fn initialize(&self) -> easyrag_core::Result<()> {
        let ws = self.ws_label();
        // Composite index on (entity_id, biz_id) backing the double-key MERGE.
        if let Err(e) = self
            .db
            .run_write_void(
                &format!("CREATE INDEX IF NOT EXISTS FOR (n:{ws}) ON (n.entity_id, n.biz_id)"),
                Neo4jParams::new(),
            )
            .await
        {
            tracing::warn!("Failed to create index (may already exist): {e}");
        } else {
            tracing::info!("Neo4j index ensured for workspace '{}'", self.workspace);
        }

        // Fulltext index per workspace: fulltext indexes cannot be
        // predicate-filtered on creation, so biz isolation comes from the
        // `node.biz_id` post-filter applied in `search_labels`.
        let ftn = self.fulltext_index_name();
        if let Err(e) = self
            .db
            .run_write_void(
                &format!(
                    "CREATE FULLTEXT INDEX {ftn} IF NOT EXISTS \
                     FOR (n:{ws}) ON EACH [n.entity_id]"
                ),
                Neo4jParams::new(),
            )
            .await
        {
            tracing::debug!("Fulltext index creation skipped: {e}");
        }
        Ok(())
    }

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let ws = self.ws_label();
        match self
            .db
            .run_write_void(
                &format!("MATCH {} DETACH DELETE n", cypher::node_pat(&ws, "n", None)),
                Self::biz_params(biz_id),
            )
            .await
        {
            Ok(()) => Ok(drop_success(format!(
                "All nodes in workspace '{}', biz_id '{}' deleted",
                self.workspace, biz_id
            ))),
            Err(e) => Ok(drop_error(e)),
        }
    }
}

#[async_trait]
impl<O: Neo4jOps> GraphStorage for Neo4jGraphStorage<O> {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn has_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<bool> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {} RETURN count(n) > 0 AS exists",
                    cypher::node_pat(&ws, "n", Some("$id"))
                ),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(node_id.to_string()))]),
            )
            .await?;
        Ok(records.first().map(|r| Self::row_bool(r, "exists")).unwrap_or(false))
    }

    async fn has_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<bool> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {}{}{} \
                     RETURN count(r) > 0 AS exists",
                    cypher::node_pat(&ws, "a", Some("$src")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", Some("$tgt"))
                ),
                Self::biz_params_with(biz_id, &[
                    ("src", JsonValue::String(source_node_id.to_string())),
                    ("tgt", JsonValue::String(target_node_id.to_string())),
                ]),
            )
            .await?;
        Ok(records.first().map(|r| Self::row_bool(r, "exists")).unwrap_or(false))
    }

    async fn node_degree(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {} RETURN count(r) AS degree",
                    cypher::degree_pat(&ws, "n", "$id", "r", "m")
                ),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(node_id.to_string()))]),
            )
            .await?;
        Ok(records
            .first()
            .map(|r| Self::row_int(r, "degree") as usize)
            .unwrap_or(0))
    }

    async fn edge_degree(&self, src_id: &str, tgt_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        Ok(self.node_degree(src_id, biz_id).await? + self.node_degree(tgt_id, biz_id).await?)
    }

    async fn get_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!("MATCH {} RETURN n", cypher::node_pat(&ws, "n", Some("$id"))),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(node_id.to_string()))]),
            )
            .await?;
        let Some(record) = records.first() else {
            return Ok(None);
        };
        let Some(node) = record.get("n") else {
            return Ok(None);
        };
        Ok(Self::node_to_map(node))
    }

    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {}{}{} \
                     RETURN r LIMIT 1",
                    cypher::node_pat(&ws, "a", Some("$src")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", Some("$tgt"))
                ),
                Self::biz_params_with(biz_id, &[
                    ("src", JsonValue::String(source_node_id.to_string())),
                    ("tgt", JsonValue::String(target_node_id.to_string())),
                ]),
            )
            .await?;
        let Some(record) = records.first() else {
            return Ok(None);
        };
        let Some(rel) = record.get("r") else {
            return Ok(None);
        };
        let mut props = Self::rel_props(rel);
        props.insert(
            "source".to_string(),
            JsonValue::String(source_node_id.to_string()),
        );
        props.insert(
            "target".to_string(),
            JsonValue::String(target_node_id.to_string()),
        );
        Ok(Some(props))
    }

    async fn get_node_edges(
        &self,
        source_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<Vec<(String, String)>>> {
        // None for a missing node, matching the InMemory/Postgres backends.
        if !self.has_node(source_node_id, biz_id).await? {
            return Ok(None);
        }
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {}{}{} \
                     RETURN a.entity_id AS src, b.entity_id AS tgt",
                    cypher::node_pat(&ws, "a", Some("$id")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", None)
                ),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(source_node_id.to_string()))]),
            )
            .await?;
        Ok(Some(
            records
                .iter()
                .map(|r| (Self::row_str(r, "src"), Self::row_str(r, "tgt")))
                .collect(),
        ))
    }

    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let ws = self.ws_label();
        let props = sanitize_props(node_data);
        self.db
            .run_write_void(
                &format!("MERGE {} SET n += $props", cypher::node_pat(&ws, "n", Some("$id"))),
                Self::biz_params_with(biz_id, &[
                    ("id", JsonValue::String(node_id.to_string())),
                    ("props", JsonValue::Object(props)),
                ]),
            )
            .await
    }

    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let ws = self.ws_label();
        let props = sanitize_props(edge_data);
        self.db
            .run_write_void(
                &format!(
                    "MATCH {}, {} \
                     MERGE (a){}(b) SET r += $props, r.biz_id = $biz",
                    cypher::node_pat(&ws, "a", Some("$src")),
                    cypher::node_pat(&ws, "b", Some("$tgt")),
                    cypher::edge_pat("r", true)
                ),
                Self::biz_params_with(biz_id, &[
                    ("src", JsonValue::String(source_node_id.to_string())),
                    ("tgt", JsonValue::String(target_node_id.to_string())),
                    ("props", JsonValue::Object(props)),
                ]),
            )
            .await
    }

    async fn delete_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<()> {
        let ws = self.ws_label();
        self.db
            .run_write_void(
                &format!("MATCH {} DETACH DELETE n", cypher::node_pat(&ws, "n", Some("$id"))),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(node_id.to_string()))]),
            )
            .await
    }

    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }
        let ws = self.ws_label();
        let ids = cypher::json_ids(node_ids);
        self.db
            .run_write_void(
                &format!(
                    "{} MATCH {} DETACH DELETE n",
                    cypher::unwind("ids", "id"),
                    cypher::node_pat(&ws, "n", Some("id"))
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(ids))]),
            )
            .await
    }

    async fn remove_edges(&self, edge_list: &[(String, String)], biz_id: &str) -> easyrag_core::Result<()> {
        if edge_list.is_empty() {
            return Ok(());
        }
        let ws = self.ws_label();
        let edges: Vec<JsonValue> = edge_list
            .iter()
            .map(|(src, tgt)| {
                JsonValue::Object(serde_json::Map::from_iter([
                    ("src".to_string(), JsonValue::String(src.clone())),
                    ("tgt".to_string(), JsonValue::String(tgt.clone())),
                ]))
            })
            .collect();
        self.db
            .run_write_void(
                &format!(
                    "{} \
                     MATCH {}{}{} \
                     DELETE r",
                    cypher::unwind("edges", "edge"),
                    cypher::node_pat(&ws, "a", Some("edge.src")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", Some("edge.tgt"))
                ),
                Self::biz_params_with(biz_id, &[("edges", JsonValue::Array(edges))]),
            )
            .await
    }

    async fn get_all_labels(&self, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let ws = self.ws_label();
        let records = self
            .db
            .run(
                &format!(
                    "MATCH {} RETURN n.entity_id AS label ORDER BY n.entity_id",
                    cypher::node_pat(&ws, "n", None)
                ),
                Self::biz_params(biz_id),
            )
            .await?;
        Ok(records.iter().map(|r| Self::row_str(r, "label")).collect())
    }

    async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        match node_label {
            None | Some("*") => self.get_full_graph(max_nodes, biz_id).await,
            Some(label) => self.get_sub_graph(label, max_depth, max_nodes, biz_id).await,
        }
    }

    async fn get_all_nodes(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        // Read-path wildcard: empty biz_id lists every biz line (node_scope).
        let node = self.node_scope(biz_id);
        let records = self
            .db
            .run(&format!("MATCH (n:{node}) RETURN n"), Self::biz_params(biz_id))
            .await?;
        Ok(records
            .iter()
            .filter_map(|r| r.get("n"))
            .filter_map(Self::node_to_map)
            .collect())
    }

    async fn get_all_edges(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let node = self.node_scope(biz_id);
        let records = self
            .db
            .run(
                &format!(
                    "MATCH (a:{node}){}(b:{node}) \
                     RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props",
                    cypher::edge_pat("r", true)
                ),
                Self::biz_params(biz_id),
            )
            .await?;
        Ok(records
            .iter()
            .map(|r| {
                let mut props = r
                    .get("props")
                    .and_then(Self::as_object)
                    .cloned()
                    .unwrap_or_default();
                props.insert(
                    "source".to_string(),
                    JsonValue::String(Self::row_str(r, "src")),
                );
                props.insert(
                    "target".to_string(),
                    JsonValue::String(Self::row_str(r, "tgt")),
                );
                props
            })
            .collect())
    }

    async fn get_popular_labels(&self, limit: usize, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let node = self.node_scope(biz_id);
        let records = self
            .db
            .run(
                &format!(
                    "MATCH (n:{node}){}() \
                     RETURN n.entity_id AS label, count(r) AS degree \
                     ORDER BY degree DESC LIMIT $limit",
                    cypher::edge_pat("r", false)
                ),
                Self::biz_params_with(biz_id, &[("limit", JsonValue::from(limit as i64))]),
            )
            .await?;
        Ok(records.iter().map(|r| Self::row_str(r, "label")).collect())
    }

    async fn search_labels(&self, query: &str, limit: usize, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let ftn = self.fulltext_index_name();
        // Read-path wildcard: an empty biz_id drops the biz post-filter so
        // search spans every biz line of the workspace.
        let scope = self.scope(biz_id);
        let biz_pred = scope
            .biz_predicate("node")
            .map(|p| format!(" AND {p}"))
            .unwrap_or_default();
        let node = scope.node_label();
        let params = Self::biz_params_with(biz_id, &[
            ("query", JsonValue::String(query.to_string())),
            ("limit", JsonValue::from(limit as i64)),
            // Tenant guard: the sanitized fulltext index name can collide
            // across workspaces; filter hits back to this workspace's label.
            ("ws", JsonValue::String(self.workspace.clone())),
        ]);
        // Try fulltext index first, fallback to CONTAINS. Both paths apply
        // the biz_id post-filter (fulltext indexes cannot be partitioned by
        // property at creation time); the filter is dropped on wildcard read.
        match self
            .db
            .run(
                &format!(
                    "CALL db.index.fulltext.queryNodes({ftn}, $query) \
                     YIELD node WHERE $ws IN labels(node){biz_pred} \
                     RETURN node.entity_id AS label LIMIT $limit"
                ),
                params.clone(),
            )
            .await
        {
            Ok(records) => Ok(records.iter().map(|r| Self::row_str(r, "label")).collect()),
            Err(e) => {
                tracing::debug!("Fulltext search failed, falling back to CONTAINS: {e}");
                let records = self
                    .db
                    .run(
                        &format!(
                            "MATCH (n:{node}) WHERE n.entity_id CONTAINS $query \
                             RETURN n.entity_id AS label LIMIT $limit"
                        ),
                        params,
                    )
                    .await?;
                Ok(records.iter().map(|r| Self::row_str(r, "label")).collect())
            }
        }
    }

    // --- Batch operations (UNWIND-based for performance) ---

    async fn get_nodes_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, JsonMap>> {
        if node_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ws = self.ws_label();
        let ids = cypher::json_ids(node_ids);
        let records = self
            .db
            .run(
                &format!(
                    "{} MATCH {} RETURN n",
                    cypher::unwind("ids", "id"),
                    cypher::node_pat(&ws, "n", Some("id"))
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(ids))]),
            )
            .await?;
        let mut result = HashMap::new();
        for record in &records {
            if let Some(node) = record.get("n") {
                let entity_id = Self::node_id(node);
                if let Some(map) = Self::node_to_map(node) {
                    result.insert(entity_id, map);
                }
            }
        }
        Ok(result)
    }

    async fn node_degrees_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, usize>> {
        if node_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ws = self.ws_label();
        let ids = cypher::json_ids(node_ids);
        let records = self
            .db
            .run(
                &format!(
                    "{} \
                     OPTIONAL MATCH {} \
                     RETURN id, count(r) AS degree",
                    cypher::unwind("ids", "id"),
                    cypher::degree_pat(&ws, "n", "id", "r", "m")
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(ids))]),
            )
            .await?;
        Ok(records
            .iter()
            .map(|r| (Self::row_str(r, "id"), Self::row_int(r, "degree") as usize))
            .collect())
    }

    async fn upsert_nodes_batch(&self, nodes: &[(String, JsonMap)], biz_id: &str) -> easyrag_core::Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        let ws = self.ws_label();
        let items: Vec<JsonValue> = nodes
            .iter()
            .map(|(id, data)| {
                JsonValue::Object(serde_json::Map::from_iter([
                    ("id".to_string(), JsonValue::String(id.clone())),
                    ("props".to_string(), JsonValue::Object(sanitize_props(data.clone()))),
                ]))
            })
            .collect();
        self.db
            .run_write_void(
                &format!(
                    "{} MERGE {} SET n += item.props",
                    cypher::unwind("items", "item"),
                    cypher::node_pat(&ws, "n", Some("item.id"))
                ),
                Self::biz_params_with(biz_id, &[("items", JsonValue::Array(items))]),
            )
            .await
    }

    async fn has_nodes_batch(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<HashSet<String>> {
        if node_ids.is_empty() {
            return Ok(HashSet::new());
        }
        let ws = self.ws_label();
        let ids = cypher::json_ids(node_ids);
        let records = self
            .db
            .run(
                &format!(
                    "{} MATCH {} RETURN n.entity_id AS id",
                    cypher::unwind("ids", "id"),
                    cypher::node_pat(&ws, "n", Some("id"))
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(ids))]),
            )
            .await?;
        Ok(records.iter().map(|r| Self::row_str(r, "id")).collect())
    }

    async fn upsert_edges_batch(
        &self,
        edges: &[(String, String, JsonMap)],
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        let ws = self.ws_label();
        let items: Vec<JsonValue> = edges
            .iter()
            .map(|(src, tgt, data)| {
                JsonValue::Object(serde_json::Map::from_iter([
                    ("src".to_string(), JsonValue::String(src.clone())),
                    ("tgt".to_string(), JsonValue::String(tgt.clone())),
                    ("props".to_string(), JsonValue::Object(sanitize_props(data.clone()))),
                ]))
            })
            .collect();
        self.db
            .run_write_void(
                &format!(
                    "{} \
                     MATCH {}, {} \
                     MERGE (a){}(b) SET r += item.props, r.biz_id = $biz",
                    cypher::unwind("items", "item"),
                    cypher::node_pat(&ws, "a", Some("item.src")),
                    cypher::node_pat(&ws, "b", Some("item.tgt")),
                    cypher::edge_pat("r", true)
                ),
                Self::biz_params_with(biz_id, &[("items", JsonValue::Array(items))]),
            )
            .await
    }

    async fn edge_degrees_batch(
        &self,
        edge_pairs: &[(String, String)],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<(String, String), usize>> {
        if edge_pairs.is_empty() {
            return Ok(HashMap::new());
        }
        let ws = self.ws_label();
        let items: Vec<JsonValue> = edge_pairs
            .iter()
            .map(|(src, tgt)| {
                JsonValue::Object(serde_json::Map::from_iter([
                    ("src".to_string(), JsonValue::String(src.clone())),
                    ("tgt".to_string(), JsonValue::String(tgt.clone())),
                ]))
            })
            .collect();
        let records = self
            .db
            .run(
                &format!(
                    "{} \
                     OPTIONAL MATCH {} \
                     WITH item, count(ra) AS srcDeg \
                     OPTIONAL MATCH {} \
                     RETURN item.src AS src, item.tgt AS tgt, srcDeg + count(rb) AS degree",
                    cypher::unwind("items", "item"),
                    cypher::degree_pat(&ws, "a", "item.src", "ra", "ma"),
                    cypher::degree_pat(&ws, "b", "item.tgt", "rb", "mb")
                ),
                Self::biz_params_with(biz_id, &[("items", JsonValue::Array(items))]),
            )
            .await?;
        Ok(records
            .iter()
            .map(|r| {
                (
                    (Self::row_str(r, "src"), Self::row_str(r, "tgt")),
                    Self::row_int(r, "degree") as usize,
                )
            })
            .collect())
    }

    async fn get_edges_batch(
        &self,
        pairs: &[(String, String)],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<(String, String), JsonMap>> {
        if pairs.is_empty() {
            return Ok(HashMap::new());
        }
        let ws = self.ws_label();
        let items: Vec<JsonValue> = pairs
            .iter()
            .map(|(src, tgt)| {
                JsonValue::Object(serde_json::Map::from_iter([
                    ("src".to_string(), JsonValue::String(src.clone())),
                    ("tgt".to_string(), JsonValue::String(tgt.clone())),
                ]))
            })
            .collect();
        let records = self
            .db
            .run(
                &format!(
                    "{} \
                     MATCH {}{}{} \
                     RETURN item.src AS src, item.tgt AS tgt, properties(r) AS props",
                    cypher::unwind("items", "item"),
                    cypher::node_pat(&ws, "a", Some("item.src")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", Some("item.tgt"))
                ),
                Self::biz_params_with(biz_id, &[("items", JsonValue::Array(items))]),
            )
            .await?;
        let mut result = HashMap::new();
        for r in &records {
            let src = Self::row_str(r, "src");
            let tgt = Self::row_str(r, "tgt");
            let mut props = r
                .get("props")
                .and_then(Self::as_object)
                .cloned()
                .unwrap_or_default();
            props.insert("source".to_string(), JsonValue::String(src.clone()));
            props.insert("target".to_string(), JsonValue::String(tgt.clone()));
            result.insert((src, tgt), props);
        }
        Ok(result)
    }

    async fn get_nodes_edges_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, Vec<(String, String)>>> {
        if node_ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ws = self.ws_label();
        let ids = cypher::json_ids(node_ids);
        let records = self
            .db
            .run(
                &format!(
                    "{} \
                     MATCH {}{}{} \
                     RETURN id, a.entity_id AS src, b.entity_id AS tgt",
                    cypher::unwind("ids", "id"),
                    cypher::node_pat(&ws, "a", Some("id")),
                    cypher::edge_pat("r", false),
                    cypher::node_pat(&ws, "b", None)
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(ids))]),
            )
            .await?;
        let mut collected: HashMap<String, Vec<(String, String)>> = HashMap::new();
        for r in &records {
            collected
                .entry(Self::row_str(r, "id"))
                .or_default()
                .push((Self::row_str(r, "src"), Self::row_str(r, "tgt")));
        }
        // Ensure all requested ids have an entry. Use get (not remove) so
        // duplicated ids in the input do not erase an earlier correct result.
        Ok(node_ids
            .iter()
            .map(|id| (id.clone(), collected.get(id).cloned().unwrap_or_default()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Phase 3.10 guardrails: the graph backend's retry loop moved onto
// `core::with_retry`. `Neo4jDatabase::with_retry` itself needs a live connection
// to reach, so the two things that could silently change are pinned directly —
// the backoff curve (via `RETRY_CONFIG`) and the two outcome messages (via
// `neo4j_retry_error`).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod retry_tests {
    use super::*;

    /// `500 * 2^(attempt-1)` for the attempt that just failed.
    fn delay_for(attempt: u32) -> u128 {
        (RETRY_CONFIG.base_delay.as_millis() as f64
            * RETRY_CONFIG.exponential_base.powi((attempt - 1) as i32)) as u128
    }

    #[test]
    fn policy_reproduces_the_delays_the_old_loop_slept() {
        assert_eq!(MAX_RETRIES, RETRY_CONFIG.max_attempts);
        assert_eq!(
            std::time::Duration::from_millis(RETRY_DELAY_MS),
            RETRY_CONFIG.base_delay
        );
        // Jitter off: the curve has to stay deterministic and match the old one.
        assert!(!RETRY_CONFIG.jitter);
        // The old loop slept `RETRY_DELAY_MS * (attempt + 1)` over a 0-based
        // attempt, i.e. 500ms then 1000ms across 3 tries (no sleep after the
        // last). Exponential 500 * 2^n lands on exactly the same two values.
        assert_eq!(500, delay_for(1), "first sleep must stay 500ms");
        assert_eq!(1000, delay_for(2), "second sleep must stay 1000ms");
        // The cap cannot bind at 3 attempts, so it must not shorten them.
        assert!(RETRY_CONFIG.max_delay >= std::time::Duration::from_millis(1000));
    }

    #[test]
    fn an_exhausted_budget_reports_the_attempt_count() {
        let err = Neo4jRetryError {
            inner: neo4rs::Error::ConnectionError,
            retryable: true,
        };
        let msg = neo4j_retry_error(err).to_string();
        assert!(
            msg.starts_with("Neo4j operation failed after 3 retries: "),
            "exhausted-budget message changed: {msg}"
        );
    }

    #[test]
    fn a_non_transient_failure_surfaces_the_driver_error_immediately() {
        let err = Neo4jRetryError {
            inner: neo4rs::Error::ConnectionError,
            retryable: false,
        };
        let msg = neo4j_retry_error(err).to_string();
        assert!(msg.starts_with("Neo4j error: "), "fatal message changed: {msg}");
        assert!(
            !msg.contains("retries"),
            "a fatal error must not claim a retry budget it never used: {msg}"
        );
    }
}
