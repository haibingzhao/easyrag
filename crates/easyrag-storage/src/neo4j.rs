//! Neo4j graph storage ( +
//! `Neo4jDatabase.kt`).
//!
//! Uses the workspace name as node label for multi-tenant isolation.
//! All batch operations use UNWIND for single-roundtrip performance.
//! Nodes are identified by the `entity_id` property; edges use the
//! `DIRECTED` relationship type with undirected matching semantics.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::model::{KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode};
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::OnceLock;

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

    async fn with_retry<T, F, Fut>(&self, make: F) -> easyrag_core::Result<T>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, neo4rs::Error>>,
    {
        let mut last: Option<neo4rs::Error> = None;
        for attempt in 0..MAX_RETRIES {
            match make().await {
                Ok(v) => return Ok(v),
                Err(e) if Self::is_transient(&e) => {
                    tracing::warn!(
                        "Neo4j transient error (attempt {}/{}): {e}",
                        attempt + 1,
                        MAX_RETRIES
                    );
                    if attempt < MAX_RETRIES - 1 {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            RETRY_DELAY_MS * (attempt + 1) as u64,
                        ))
                        .await;
                    }
                    last = Some(e);
                }
                Err(e) => {
                    return Err(CoreError::Internal(format!("Neo4j error: {e}")));
                }
            }
        }
        Err(CoreError::Internal(format!(
            "Neo4j operation failed after {MAX_RETRIES} retries: {}",
            last.map(|e| e.to_string()).unwrap_or_default()
        )))
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

    /// Escaped workspace label for Cypher queries.
    fn ws_label(&self) -> String {
        format!("`{}`", self.workspace)
    }

    /// Node label + biz-line scope fragment for Cypher MATCH: `` `ws` ``
    /// (read-path wildcard when `biz_id` is empty — lists every biz line of
    /// the workspace) or `` `ws` {biz_id: $biz} `` (exact biz line). Mirrors
    /// `PgDocStatusStorage::list_scope_clause`: an empty biz_id is the
    /// read-path wildcard, never a literal match against the empty string.
    /// Write paths and single-node reads must NOT route through this helper
    /// — they keep the exact scope.
    fn node_scope(&self, biz_id: &str) -> String {
        let ws = self.ws_label();
        if biz_id.is_empty() {
            ws
        } else {
            format!("{ws} {{biz_id: $biz}}")
        }
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
                nodes.push(KnowledgeGraphNode {
                    id: Self::node_id(n),
                    labels: Self::node_labels(n),
                    properties: Self::node_to_map(n).unwrap_or_default(),
                });
            }
        }

        let node_ids: HashSet<String> = nodes.iter().map(|n| n.id.clone()).collect();
        let edge_records = self
            .db
            .run(
                &format!(
                    "MATCH (a:{node})-[r:DIRECTED]->(b:{node}) \
                     RETURN a.entity_id AS src, b.entity_id AS tgt, r LIMIT $limit"
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
                edges.push(KnowledgeGraphEdge {
                    id: Self::rel_element_id(&rel),
                    edge_type: Some("DIRECTED".to_string()),
                    source: src,
                    target: tgt,
                    properties: Self::rel_props(&rel),
                });
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
        // Read-path wildcard: an empty biz_id expands across every biz line
        // of the workspace (drops the biz_id predicate instead of matching
        // the literal empty string). See `node_scope`.
        let start_map = if biz_id.is_empty() {
            "{entity_id: $label}".to_string()
        } else {
            "{entity_id: $label, biz_id: $biz}".to_string()
        };
        let biz_where = if biz_id.is_empty() {
            String::new()
        } else {
            "WHERE n.biz_id = $biz ".to_string()
        };
        // BFS-style expansion using variable-length path
        let records = self
            .db
            .run(
                &format!(
                    "MATCH path = (start:{ws} {start_map})-[:DIRECTED*0..{safe_depth}]-(connected:{ws}) \
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
                nodes.push(KnowledgeGraphNode {
                    id: Self::node_id(n),
                    labels: Self::node_labels(n),
                    properties: Self::node_to_map(n).unwrap_or_default(),
                });
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
        let a_map = if biz_id.is_empty() {
            "{entity_id: srcId}".to_string()
        } else {
            "{entity_id: srcId, biz_id: $biz}".to_string()
        };
        let b_scope = self.node_scope(biz_id);
        let edge_records = self
            .db
            .run(
                &format!(
                    "UNWIND $ids AS srcId \
                     MATCH (a:{ws} {a_map})-[r:DIRECTED]-(b:{b_scope}) \
                     WHERE b.entity_id IN $ids \
                     RETURN DISTINCT a.entity_id AS src, b.entity_id AS tgt, r"
                ),
                Self::biz_params_with(biz_id, &[("ids", JsonValue::Array(node_ids))]),
            )
            .await?;

        let mut edges = Vec::new();
        for record in &edge_records {
            let rel = record.get("r").cloned().unwrap_or(JsonValue::Null);
            edges.push(KnowledgeGraphEdge {
                id: Self::rel_element_id(&rel),
                edge_type: Some("DIRECTED".to_string()),
                source: Self::row_str(record, "src"),
                target: Self::row_str(record, "tgt"),
                properties: Self::rel_props(&rel),
            });
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

    async fn index_done_callback(&self) -> easyrag_core::Result<()> {
        // Neo4j auto-commits, nothing to flush
        Ok(())
    }

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let ws = self.ws_label();
        match self
            .db
            .run_write_void(
                &format!("MATCH (n:{ws} {{biz_id: $biz}}) DETACH DELETE n"),
                Self::biz_params(biz_id),
            )
            .await
        {
            Ok(()) => Ok(HashMap::from([
                ("status".to_string(), "success".to_string()),
                (
                    "message".to_string(),
                    format!(
                        "All nodes in workspace '{}', biz_id '{}' deleted",
                        self.workspace, biz_id
                    ),
                ),
            ])),
            Err(e) => Ok(HashMap::from([
                ("status".to_string(), "error".to_string()),
                ("message".to_string(), e.to_string()),
            ])),
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
                    "MATCH (n:{ws} {{entity_id: $id, biz_id: $biz}}) RETURN count(n) > 0 AS exists"
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
                    "MATCH (a:{ws} {{entity_id: $src, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{entity_id: $tgt, biz_id: $biz}}) \
                     RETURN count(r) > 0 AS exists"
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
                    "MATCH (n:{ws} {{entity_id: $id, biz_id: $biz}})-[r:DIRECTED]-(m:{ws} {{biz_id: $biz}}) RETURN count(r) AS degree"
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
                &format!("MATCH (n:{ws} {{entity_id: $id, biz_id: $biz}}) RETURN n"),
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
                    "MATCH (a:{ws} {{entity_id: $src, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{entity_id: $tgt, biz_id: $biz}}) \
                     RETURN r LIMIT 1"
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
                    "MATCH (a:{ws} {{entity_id: $id, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{biz_id: $biz}}) \
                     RETURN a.entity_id AS src, b.entity_id AS tgt"
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
                &format!("MERGE (n:{ws} {{entity_id: $id, biz_id: $biz}}) SET n += $props"),
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
                    "MATCH (a:{ws} {{entity_id: $src, biz_id: $biz}}), (b:{ws} {{entity_id: $tgt, biz_id: $biz}}) \
                     MERGE (a)-[r:DIRECTED]->(b) SET r += $props, r.biz_id = $biz"
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
                &format!("MATCH (n:{ws} {{entity_id: $id, biz_id: $biz}}) DETACH DELETE n"),
                Self::biz_params_with(biz_id, &[("id", JsonValue::String(node_id.to_string()))]),
            )
            .await
    }

    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        if node_ids.is_empty() {
            return Ok(());
        }
        let ws = self.ws_label();
        let ids: Vec<JsonValue> = node_ids
            .iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect();
        self.db
            .run_write_void(
                &format!(
                    "UNWIND $ids AS id MATCH (n:{ws} {{entity_id: id, biz_id: $biz}}) DETACH DELETE n"
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
                    "UNWIND $edges AS edge \
                     MATCH (a:{ws} {{entity_id: edge.src, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{entity_id: edge.tgt, biz_id: $biz}}) \
                     DELETE r"
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
                    "MATCH (n:{ws} {{biz_id: $biz}}) RETURN n.entity_id AS label ORDER BY n.entity_id"
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
                    "MATCH (a:{node})-[r:DIRECTED]->(b:{node}) \
                     RETURN a.entity_id AS src, b.entity_id AS tgt, properties(r) AS props"
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
                    "MATCH (n:{node})-[r:DIRECTED]-() \
                     RETURN n.entity_id AS label, count(r) AS degree \
                     ORDER BY degree DESC LIMIT $limit"
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
        let biz_pred = if biz_id.is_empty() {
            String::new()
        } else {
            " AND node.biz_id = $biz".to_string()
        };
        let node = self.node_scope(biz_id);
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
        let ids: Vec<JsonValue> = node_ids
            .iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect();
        let records = self
            .db
            .run(
                &format!("UNWIND $ids AS id MATCH (n:{ws} {{entity_id: id, biz_id: $biz}}) RETURN n"),
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
        let ids: Vec<JsonValue> = node_ids
            .iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect();
        let records = self
            .db
            .run(
                &format!(
                    "UNWIND $ids AS id \
                     OPTIONAL MATCH (n:{ws} {{entity_id: id, biz_id: $biz}})-[r:DIRECTED]-(m:{ws} {{biz_id: $biz}}) \
                     RETURN id, count(r) AS degree"
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
                    "UNWIND $items AS item MERGE (n:{ws} {{entity_id: item.id, biz_id: $biz}}) SET n += item.props"
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
        let ids: Vec<JsonValue> = node_ids
            .iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect();
        let records = self
            .db
            .run(
                &format!(
                    "UNWIND $ids AS id MATCH (n:{ws} {{entity_id: id, biz_id: $biz}}) RETURN n.entity_id AS id"
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
                    "UNWIND $items AS item \
                     MATCH (a:{ws} {{entity_id: item.src, biz_id: $biz}}), (b:{ws} {{entity_id: item.tgt, biz_id: $biz}}) \
                     MERGE (a)-[r:DIRECTED]->(b) SET r += item.props, r.biz_id = $biz"
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
                    "UNWIND $items AS item \
                     OPTIONAL MATCH (a:{ws} {{entity_id: item.src, biz_id: $biz}})-[ra:DIRECTED]-(ma:{ws} {{biz_id: $biz}}) \
                     WITH item, count(ra) AS srcDeg \
                     OPTIONAL MATCH (b:{ws} {{entity_id: item.tgt, biz_id: $biz}})-[rb:DIRECTED]-(mb:{ws} {{biz_id: $biz}}) \
                     RETURN item.src AS src, item.tgt AS tgt, srcDeg + count(rb) AS degree"
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
                    "UNWIND $items AS item \
                     MATCH (a:{ws} {{entity_id: item.src, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{entity_id: item.tgt, biz_id: $biz}}) \
                     RETURN item.src AS src, item.tgt AS tgt, properties(r) AS props"
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
        let ids: Vec<JsonValue> = node_ids
            .iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect();
        let records = self
            .db
            .run(
                &format!(
                    "UNWIND $ids AS id \
                     MATCH (a:{ws} {{entity_id: id, biz_id: $biz}})-[r:DIRECTED]-(b:{ws} {{biz_id: $biz}}) \
                     RETURN id, a.entity_id AS src, b.entity_id AS tgt"
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
