//! PostgreSQL storage backends.
//!
//!   `$N`-style placeholders (sqlx speaks PostgreSQL `$N` natively, so the
//!   JDBC `?` conversion layer is obsolete).
//! - Workspace names are validated up-front with the same regex as
//!   `AppConfig.validateWorkspace` before interpolation into identifiers.
//! - Performance option B (plan P2.2): batched multi-row `INSERT ... ON CONFLICT`
//!   upserts, one `embed(texts)` call per batch; falls back to row-by-row on
//!   batch failure. `PgConfig::upsert_batch_size = 1` restores exact
//!   row-by-row semantics.

use async_trait::async_trait;
use easyrag_core::concurrency::{with_retry, RetryConfig};
use easyrag_core::llm::{EmbeddingContext, EmbeddingFunc};
use easyrag_core::model::{
    DocProcessingStatus, DocStatus, KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode,
    TenantScopeRef,
};
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageNameSpace, VectorStorage,
};
use easyrag_core::storage::user::{UserRecord, UserStorage};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::tenant_config::{TenantConfig, TenantConfigStorage};
use easyrag_core::JsonMap;
use pgvector::Vector as PgVector;
use serde_json::Value as JsonValue;
use sqlx::postgres::{PgConnectOptions, PgPoolOptions, PgRow, PgValueRef};
use sqlx::{Column, Decode, Error as SqlxError, Pool, Postgres, Row, TypeInfo, ValueRef};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::drop_result::drop_result;
use crate::pg_upsert::{batched_with_fallback, PgUpsertBuilder, TENANT_CONFLICT_TARGET};
use crate::scope_sql::{sql_escape, sql_in_list, TenantScopeSql};

/// PostgreSQL configuration for EasyRAG storage.
#[derive(Clone, Debug)]
pub struct PgConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: String,
    pub database: String,
    pub workspace: String,
    pub max_connections: u32,
    pub enable_vector: bool,
    pub vector_index_type: String,
    pub hnsw_m: u32,
    pub hnsw_ef: u32,
    pub embedding_dim: usize,
    /// SSL/TLS: disable, allow, prefer, require, verify-ca, verify-full
    pub ssl_mode: Option<String>,
    pub ssl_cert: Option<String>,
    pub ssl_key: Option<String>,
    pub ssl_root_cert: Option<String>,
    pub ssl_crl: Option<String>,
    /// PostgreSQL text search config: simple, english, zhparser...
    pub fts_config: String,
    /// Batched upsert row count (plan P2.2 option B). 1 = legacy row-by-row.
    pub upsert_batch_size: usize,
}

impl Default for PgConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            user: "postgres".to_string(),
            password: "postgres".to_string(),
            database: "easyrag".to_string(),
            workspace: "default".to_string(),
            max_connections: 10,
            enable_vector: true,
            vector_index_type: "hnsw".to_string(),
            hnsw_m: 16,
            hnsw_ef: 64,
            embedding_dim: 1536,
            ssl_mode: None,
            ssl_cert: None,
            ssl_key: None,
            ssl_root_cert: None,
            ssl_crl: None,
            fts_config: "simple".to_string(),
            upsert_batch_size: 64,
        }
    }
}

/// Batch limits for PostgreSQL write operations.
pub struct PgBatchLimits;

impl PgBatchLimits {
    pub const UPSERT_MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024; // 16 MiB
    pub const UPSERT_MAX_RECORDS_PER_BATCH: usize = 200;
    pub const DELETE_MAX_RECORDS_PER_BATCH: usize = 1000;
}

/// Workspace validation regex.
const WORKSPACE_REGEX: &str = r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]{0,63}$";

/// Validate a workspace name before it is interpolated into SQL identifiers.
pub fn validate_workspace(workspace: &str) -> easyrag_core::Result<()> {
    let re = regex::Regex::new(WORKSPACE_REGEX).expect("static regex");
    if !re.is_match(workspace) {
        return Err(easyrag_core::CoreError::InvalidArgument(format!(
            "Invalid workspace name: {workspace}"
        )));
    }
    Ok(())
}

/// Table name derived from namespace (lowercased, dashes replaced by underscores).
fn namespace_table_name(namespace: &str) -> String {
    namespace.to_lowercase().replace('-', "_")
}

/// Stringify non-string property values.
fn stringify_props(props: &JsonMap) -> JsonMap {
    let mut out = JsonMap::new();
    for (k, v) in props {
        let s = match v {
            JsonValue::String(s) => s.clone(),
            JsonValue::Null => String::new(),
            other => other.to_string(),
        };
        out.insert(k.clone(), JsonValue::String(s));
    }
    out
}

// ---------------------------------------------------------------------------
// Database abstraction
// ---------------------------------------------------------------------------

/// Minimal database facade.
///
/// Generic so storage structs can run against mocks in unit tests
/// (mirrors the Mockito-based PgVectorStorageKeywordTest).
#[async_trait]
pub trait PgDb: Send + Sync {
    /// Execute a SQL statement (DDL or DML) without returning results.
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()>;

    /// Fetch a single row as a JSON map (columns -> values).
    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>>;

    /// Fetch all rows as JSON maps.
    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>>;

    /// Execute a DML statement and return the number of affected rows.
    /// Used for atomic conditional updates (e.g. claim-for-processing).
    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64>;
}

/// PostgreSQL connection manager using a sqlx connection pool
/// (replacement for HikariCP in `PgDatabase.kt`).
#[derive(Clone)]
pub struct PgDatabase {
    pool: Pool<Postgres>,
    config: PgConfig,
}

fn map_sqlx_error(e: SqlxError) -> easyrag_core::CoreError {
    easyrag_core::CoreError::Internal(format!("PostgreSQL error: {e}"))
}

/// Decode failures surface as boxed trait objects.
fn map_decode_error(e: Box<dyn std::error::Error + Send + Sync>) -> easyrag_core::CoreError {
    easyrag_core::CoreError::Internal(format!("PostgreSQL decode error: {e}"))
}

/// Decode a raw PG value with an explicit database type parameter.
fn decode_value<'r, T: Decode<'r, Postgres>>(
    raw: PgValueRef<'r>,
) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
    T::decode(raw)
}

/// Build a query from an audited dynamic SQL string.
/// Table names come from validated workspace/namespace identifiers and all
/// data values pass through `sql_escape`, so the string is injection-safe.
fn dynamic_query(
    sql: &str,
) -> sqlx::query::Query<'static, Postgres, <Postgres as sqlx::Database>::Arguments> {
    let builder = sqlx::QueryBuilder::<Postgres>::new(sql.to_string());
    sqlx::query(builder.into_sql())
}

/// Postgres retry policy: base 1s, factor 2, capped at 10s, 3 attempts, no
/// jitter — byte-for-byte the curve the hand-rolled loop used, so wiring it onto
/// `core::with_retry` is [equivalent]: attempt 1 waits 1000ms, attempt 2 waits
/// 2000ms, attempt 3 fails the operation.
///
/// Jitter stays off on purpose. The LLM clients enable it to de-synchronise a
/// fan-out of concurrent completions; storage retries are driven one statement
/// at a time and the fixed 1s/2s curve is what operators are told to expect.
const PG_RETRY_CONFIG: RetryConfig = RetryConfig {
    max_attempts: 3,
    base_delay: Duration::from_secs(1),
    max_delay: Duration::from_secs(10),
    exponential_base: 2.0,
    jitter: false,
};

/// Retry transient (connection-level) failures with exponential backoff.
///
/// *What counts as transient* lives in `CoreError::is_retryable` (core/error.rs)
/// and is shared with the Neo4j backend — Phase 3.10 moved the classification out
/// of this function precisely so the two backends cannot drift apart again.
///
/// The `on_retry` hook keeps the PG-specific warn line (`Transient PG failure
/// …`) that operations grep for; `core::with_retry` additionally emits its own
/// generic attempt line, so a retry now logs twice rather than once.
async fn retry<F, Fut, T>(op: F) -> easyrag_core::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = easyrag_core::Result<T>>,
{
    with_retry(
        &PG_RETRY_CONFIG,
        Some(|e: &easyrag_core::CoreError, attempt: u32, delay: Duration| {
            tracing::warn!(
                "Transient PG failure (attempt {attempt}), retrying in {}ms: {e}",
                delay.as_millis()
            );
        }),
        op,
    )
    .await
}

impl PgDatabase {
    /// Build the pool (lazy connect) after validating the scope names.
    pub fn new(config: PgConfig) -> easyrag_core::Result<Self> {
        validate_workspace(&config.workspace)?;
        let mut options = PgConnectOptions::new()
            .host(&config.host)
            .port(config.port)
            .username(&config.user)
            .password(&config.password)
            .database(&config.database)
            // Required for sql_escape's single-quote-doubling to be safe on
            // every server configuration (guards dynamically built literals).
            .options([("standard_conforming_strings", "on")]);
        // P2.14: SSL/TLS parameters.
        if let Some(mode) = &config.ssl_mode {
            let m = match mode.to_lowercase().as_str() {
                "disable" => sqlx::postgres::PgSslMode::Disable,
                "allow" => sqlx::postgres::PgSslMode::Allow,
                "prefer" => sqlx::postgres::PgSslMode::Prefer,
                "require" => sqlx::postgres::PgSslMode::Require,
                "verify-ca" => sqlx::postgres::PgSslMode::VerifyCa,
                "verify-full" => sqlx::postgres::PgSslMode::VerifyFull,
                other => {
                    return Err(easyrag_core::CoreError::InvalidArgument(format!(
                        "Unsupported sslmode: {other}"
                    )))
                }
            };
            options = options.ssl_mode(m);
        }
        if let Some(v) = &config.ssl_cert {
            options = options.ssl_client_cert(std::path::PathBuf::from(v));
        }
        if let Some(v) = &config.ssl_key {
            options = options.ssl_client_key(std::path::PathBuf::from(v));
        }
        if let Some(v) = &config.ssl_root_cert {
            options = options.ssl_root_cert(std::path::PathBuf::from(v));
        }
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(std::cmp::max(2, config.max_connections / 4))
            .acquire_timeout(Duration::from_secs(30))
            .idle_timeout(Duration::from_secs(600))
            .max_lifetime(Duration::from_secs(1800))
            .connect_lazy_with(options);
        Ok(Self { pool, config })
    }

    /// Initialize the pool (validates connectivity,).
    pub async fn initialize(&self) -> easyrag_core::Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(map_sqlx_error)?;
        tracing::info!(
            "PgDatabase initialized: host={}, port={}, db={}, maxConn={}",
            self.config.host,
            self.config.port,
            self.config.database,
            self.config.max_connections
        );
        Ok(())
    }

    /// Close the connection pool.
    pub async fn close(&self) {
        self.pool.close().await;
        tracing::info!("PgDatabase connection pool closed");
    }
}

#[async_trait]
impl PgDb for PgDatabase {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        let sql = sql.to_string();
        retry(|| async {
            dynamic_query(&sql)
                .execute(&self.pool)
                .await
                .map_err(map_sqlx_error)?;
            Ok(())
        })
        .await
    }

    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        let sql = sql.to_string();
        retry(|| async {
            let result = dynamic_query(&sql)
                .execute(&self.pool)
                .await
                .map_err(map_sqlx_error)?;
            Ok(result.rows_affected())
        })
        .await
    }

    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        let sql = sql.to_string();
        retry(|| async {
            let row = dynamic_query(&sql)
                .fetch_optional(&self.pool)
                .await
                .map_err(map_sqlx_error)?;
            match row {
                Some(r) => Ok(Some(row_to_map(&r)?)),
                None => Ok(None),
            }
        })
        .await
    }

    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let sql = sql.to_string();
        retry(|| async {
            let rows = dynamic_query(&sql)
                .fetch_all(&self.pool)
                .await
                .map_err(map_sqlx_error)?;
            rows.iter().map(row_to_map).collect::<easyrag_core::Result<Vec<_>>>()
        })
        .await
    }
}

/// Convert a result row into a JSON map.
fn row_to_map(row: &PgRow) -> easyrag_core::Result<JsonMap> {
    let mut map = JsonMap::new();
    for column in row.columns() {
        let name = column.name().to_string();
        let raw = row.try_get_raw(column.ordinal()).map_err(map_sqlx_error)?;
        let value = if raw.is_null() {
            JsonValue::Null
        } else {
            convert_pg_value(raw)?
        };
        map.insert(name, value);
    }
    Ok(map)
}

/// Map a PG wire value to JSON. jsonb -> parsed JSON (mirrors PGobject.value +
/// Jackson parsing), timestamptz -> string, vector -> "[f1,f2,...]" string.
fn convert_pg_value(raw: PgValueRef<'_>) -> easyrag_core::Result<JsonValue> {
    let type_name = raw.type_info().name().to_uppercase();
    match type_name.as_str() {
        "BOOL" => decode_value::<bool>(raw)
            .map(JsonValue::from)
            .map_err(map_decode_error),
        "INT2" => decode_value::<i16>(raw)
            .map(|v| JsonValue::from(v as i64))
            .map_err(map_decode_error),
        "INT4" => decode_value::<i32>(raw)
            .map(|v| JsonValue::from(v as i64))
            .map_err(map_decode_error),
        "INT8" => decode_value::<i64>(raw)
            .map(JsonValue::from)
            .map_err(map_decode_error),
        "FLOAT4" => decode_value::<f32>(raw)
            .map(|v| {
                serde_json::Number::from_f64(v as f64)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            })
            .map_err(map_decode_error),
        "FLOAT8" | "NUMERIC" => decode_value::<f64>(raw)
            .map(|v| {
                serde_json::Number::from_f64(v)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null)
            })
            .map_err(map_decode_error),
        "TIMESTAMPTZ" => decode_value::<chrono::DateTime<chrono::Utc>>(raw)
            .map(|dt| JsonValue::String(dt.to_rfc3339()))
            .map_err(map_decode_error),
        "JSONB" | "JSON" => {
            let json: serde_json::Value =
                decode_value::<serde_json::Value>(raw).map_err(map_decode_error)?;
            Ok(json)
        }
        "VECTOR" => {
            let v = decode_value::<PgVector>(raw).map_err(map_decode_error)?;
            let body = v
                .to_vec()
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",");
            Ok(JsonValue::String(format!("[{body}]")))
        }
        _ => {
            // Everything else (VARCHAR, TEXT, ...) as string.
            decode_value::<String>(raw)
                .map(JsonValue::String)
                .map_err(map_decode_error)
        }
    }
}

/// Shared tenant-scope rendering for every Pg backend.
///
/// Each backend supplies its workspace (as a borrow, so no query pays for a
/// clone); the filter text itself is defined once in
/// [`crate::scope_sql::TenantScopeSql`]. Both scope forms live here so a
/// backend cannot accidentally pick its own: `scope_clause` pins the exact
/// `(workspace, biz_id)` pair for single-record reads and writes, while
/// `list_scope_clause` treats an empty `biz_id` as the read-path wildcard over
/// every biz line of the workspace.
trait PgScoped {
    /// Borrow both halves of the scope without cloning: the workspace comes
    /// from the backend, the biz line from the caller.
    fn tenant_scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a>;

    fn scope_clause(&self, biz_id: &str) -> String {
        self.tenant_scope(biz_id).sql(None)
    }

    /// Alias-qualified scope for queries that join two scoped tables.
    /// Completes the {exact,list} × {plain,alias} matrix. Both joined list paths
    /// (`get_all_edges`, `get_popular_labels`) widen via
    /// [`Self::list_scope_clause_as`], and no joined query currently needs the
    /// exact scope, so this arm is retained for API symmetry.
    #[allow(dead_code)]
    fn scope_clause_as(&self, biz_id: &str, alias: &str) -> String {
        self.tenant_scope(biz_id).sql(Some(alias))
    }

    /// List/count scope: an empty biz_id is the read-path wildcard matching
    /// every biz line of the workspace. Single-document reads and writes keep
    /// the exact scope (status writes must never land on the wrong biz line).
    fn list_scope_clause(&self, biz_id: &str) -> String {
        self.tenant_scope(biz_id).list_sql(None)
    }

    /// Alias-qualified [`Self::list_scope_clause`] for joined list queries.
    fn list_scope_clause_as(&self, biz_id: &str, alias: &str) -> String {
        self.tenant_scope(biz_id).list_sql(Some(alias))
    }
}

// One workspace borrow per backend; everything else comes from the defaults
// above, so the four storages cannot drift apart on tenant filtering.
impl<D: PgDb> PgScoped for PgVectorStorage<D> {
    fn tenant_scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a> {
        TenantScopeRef::new(&self.workspace, biz_id)
    }
}

impl<D: PgDb> PgScoped for PgGraphStorage<D> {
    fn tenant_scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a> {
        TenantScopeRef::new(&self.workspace, biz_id)
    }
}

impl<D: PgDb> PgScoped for PgDocStatusStorage<D> {
    fn tenant_scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a> {
        TenantScopeRef::new(&self.workspace, biz_id)
    }
}

/// Render an optional string as a SQL literal (`NULL` when absent).
fn opt_sql_literal(value: Option<&str>) -> String {
    match value {
        Some(v) => format!("'{}'", sql_escape(v)),
        None => "NULL".to_string(),
    }
}

/// Escape LIKE pattern special characters so the prefix is matched literally.
fn escape_like_pattern(prefix: &str) -> String {
    prefix
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

/// Render a JSON value as a SQL literal string (single-quote escaped).
fn json_literal(value: &JsonMap) -> String {
    let json = serde_json::to_string(value).expect("JsonMap serializes");
    format!("'{}'", sql_escape(&json))
}

/// Format an f32 embedding as "[f1,f2,...]".
fn vector_literal(embedding: &[f32]) -> String {
    let body = embedding
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{body}]")
}

/// Parse a "[f1,f2,...]" vector string.
fn parse_vector_string(s: &str) -> Vec<f32> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
    if trimmed.is_empty() {
        return Vec::new();
    }
    trimmed
        .split(',')
        .filter_map(|part| part.trim().parse::<f32>().ok())
        .collect()
}

// ---------------------------------------------------------------------------
// Shared CRUD helpers
// ---------------------------------------------------------------------------

struct PgCrudOps<'a, D: PgDb> {
    db: &'a D,
    table_name: String,
    workspace: String,
    biz_id: String,
}

impl<D: PgDb> PgScoped for PgCrudOps<'_, D> {
    fn tenant_scope<'a>(&'a self, biz_id: &'a str) -> TenantScopeRef<'a> {
        TenantScopeRef::new(&self.workspace, biz_id)
    }
}

impl<'a, D: PgDb> PgCrudOps<'a, D> {
    // NOTE: PgCrudOps keeps workspace + biz_id as owned copies supplied by the
    // caller's `ops(biz_id)`; biz_id is a per-call filter, not instance state.

    async fn fetch_content_by_id(&self, id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        let sql = format!(
            "SELECT content FROM {} WHERE id = '{}' AND {}",
            self.table_name,
            sql_escape(id),
            self.scope_clause(&self.biz_id)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row.and_then(|r| read_content(&r)))
    }

    async fn fetch_contents_by_ids(&self, ids: &[String]) -> easyrag_core::Result<Vec<JsonMap>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let in_list = sql_in_list(ids.iter().map(String::as_str));
        let sql = format!(
            "SELECT content FROM {} WHERE {} AND id IN ({in_list})",
            self.table_name,
            self.scope_clause(&self.biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows.into_iter().filter_map(|r| read_content(&r)).collect())
    }

    async fn fetch_existing_ids(&self, keys: &HashSet<String>) -> easyrag_core::Result<HashSet<String>> {
        if keys.is_empty() {
            return Ok(HashSet::new());
        }
        let in_list = sql_in_list(keys.iter().map(String::as_str));
        let sql = format!(
            "SELECT id FROM {} WHERE {} AND id IN ({in_list})",
            self.table_name,
            self.scope_clause(&self.biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect())
    }

    async fn delete_by_ids(&self, ids: &[String]) -> easyrag_core::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        for batch in ids.chunks(PgBatchLimits::DELETE_MAX_RECORDS_PER_BATCH) {
            let in_list = sql_in_list(batch.iter().map(String::as_str));
            let sql = format!(
                "DELETE FROM {} WHERE {} AND id IN ({in_list})",
                self.table_name,
                self.scope_clause(&self.biz_id)
            );
            self.db.execute(&sql).await?;
        }
        Ok(())
    }

    async fn count(&self) -> easyrag_core::Result<i64> {
        let sql = format!(
            "SELECT COUNT(*) as cnt FROM {} WHERE {}",
            self.table_name,
            self.scope_clause(&self.biz_id)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row
            .and_then(|r| r.get("cnt").and_then(|v| v.as_i64()))
            .unwrap_or(0))
    }

    /// Delete only this (workspace, biz_id) slice: business tables are shared
    /// across tenants, so a bare DROP TABLE would destroy other tenants' data.
    async fn drop_scope(&self) -> easyrag_core::Result<()> {
        self.db
            .execute(&format!(
                "DELETE FROM {} WHERE {}",
                self.table_name,
                self.scope_clause(&self.biz_id)
            ))
            .await
    }
}

/// Parse the `content` jsonb column into a JsonMap.
fn read_content(row: &JsonMap) -> Option<JsonMap> {
    match row.get("content")? {
        JsonValue::Object(m) => Some(m.clone()),
        JsonValue::String(s) => serde_json::from_str(s).ok(),
        _ => None,
    }
}

/// Read a timestamp column that lives on the row itself rather than inside the
/// `content` payload. The driver normally renders these as JSON strings, but a
/// non-string value is stringified instead of being dropped, so the pre-existing
/// row-level shape is preserved verbatim.
fn row_timestamp(row: &JsonMap, key: &str) -> String {
    row.get(key)
        .map(|v| match v {
            JsonValue::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// PgKVStorage
// ---------------------------------------------------------------------------

/// PostgreSQL KV storage.
pub struct PgKVStorage<D: PgDb = PgDatabase> {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    db: D,
    table_name: String,
}

impl PgKVStorage<PgDatabase> {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        db: PgDatabase,
    ) -> easyrag_core::Result<Self> {
        Self::with_options(namespace, workspace, JsonMap::new(), None, db)
    }
}

impl<D: PgDb> PgKVStorage<D> {
    pub fn with_options(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
        db: D,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        validate_workspace(&workspace)?;
        let namespace = namespace.into();
        let table_name = format!("kv_{}", namespace_table_name(&namespace));
        Ok(Self {
            namespace,
            workspace,
            global_config,
            embedding_func,
            db,
            table_name,
        })
    }

    fn ops(&self, biz_id: &str) -> PgCrudOps<'_, D> {
        PgCrudOps {
            db: &self.db,
            table_name: self.table_name.clone(),
            workspace: self.workspace.clone(),
            biz_id: biz_id.to_string(),
        }
    }
}

#[async_trait]
impl<D: PgDb> StorageNameSpace for PgKVStorage<D> {
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
        let ws = sql_escape(&self.workspace);
        self.db
            .execute(&format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                    id VARCHAR(255) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{ws}',
                    biz_id VARCHAR(64) NOT NULL,
                    content JSONB NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (id, workspace, biz_id)
                )",
                t = self.table_name
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{t}_ws_biz ON {t}(workspace, biz_id)",
                t = self.table_name
            ))
            .await?;
        tracing::info!(
            "PgKVStorage initialized: table={}, workspace={}",
            self.table_name,
            self.workspace
        );
        Ok(())
    }

    // index_done_callback: PostgreSQL commits immediately, no buffering, so the
    // trait default no-op is the whole implementation.

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let message = format!(
            "Table {} cleared for workspace={}, biz_id={}",
            self.table_name, self.workspace, biz_id
        );
        let outcome = self.ops(biz_id).drop_scope().await;
        if outcome.is_ok() {
            tracing::info!(
                "PgKVStorage dropped scope: table={}, workspace={}, biz_id={}",
                self.table_name,
                self.workspace,
                biz_id
            );
        }
        Ok(drop_result(outcome, message))
    }
}

#[async_trait]
impl<D: PgDb> KVStorage for PgKVStorage<D> {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.ops(biz_id).fetch_content_by_id(id).await
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.ops(biz_id).fetch_contents_by_ids(ids).await
    }

    async fn filter_keys(
        &self,
        keys: &HashSet<String>,
        biz_id: &str,
    ) -> easyrag_core::Result<HashSet<String>> {
        let existing = self.ops(biz_id).fetch_existing_ids(keys).await?;
        Ok(keys.difference(&existing).cloned().collect())
    }

    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let entries: Vec<(&String, &JsonMap)> = data.iter().collect();
        let (table, workspace) = (self.table_name.as_str(), self.workspace.as_str());
        batched_with_fallback(
            &self.db,
            &entries,
            PgBatchLimits::UPSERT_MAX_RECORDS_PER_BATCH,
            // KV writes the entries themselves; there is nothing to prepare.
            |batch| async move { Ok(batch.to_vec()) },
            |rows| kv_upsert_sql(table, workspace, biz_id, rows),
        )
        .await?;
        tracing::debug!(
            "PgKVStorage upserted {} records to {}",
            data.len(),
            self.table_name
        );
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.ops(biz_id).delete_by_ids(ids).await
    }

    async fn is_empty(&self, biz_id: &str) -> easyrag_core::Result<bool> {
        Ok(self.ops(biz_id).count().await? == 0)
    }
}

/// Render the batched KV upsert (plan P2.2 option B; one statement per batch).
fn kv_upsert_sql(
    table_name: &str,
    workspace: &str,
    biz_id: &str,
    batch: &[(&String, &JsonMap)],
) -> String {
    let upsert = PgUpsertBuilder::new(
        table_name,
        "id, workspace, biz_id, content, updated_at",
        "content = EXCLUDED.content, updated_at = NOW()",
    );
    upsert.sql(batch.iter().map(|(id, content)| {
        format!(
            "('{}', '{}', '{}', {}::jsonb, NOW())",
            sql_escape(id),
            sql_escape(workspace),
            sql_escape(biz_id),
            json_literal(content)
        )
    }))
}


// ---------------------------------------------------------------------------
// PgVectorStorage
// ---------------------------------------------------------------------------

/// PostgreSQL vector storage using pgvector.
/// Also implements `KeywordSearchable` via tsvector + GIN + trigger.
pub struct PgVectorStorage<D: PgDb = PgDatabase> {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: EmbeddingFunc,
    db: D,
    config: PgConfig,
    table_name: String,
}

impl PgVectorStorage<PgDatabase> {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        embedding_func: EmbeddingFunc,
        db: PgDatabase,
        config: PgConfig,
    ) -> easyrag_core::Result<Self> {
        Self::with_options(namespace, workspace, JsonMap::new(), embedding_func, db, config)
    }
}

impl<D: PgDb> PgVectorStorage<D> {
    pub fn with_options(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: EmbeddingFunc,
        db: D,
        config: PgConfig,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        validate_workspace(&workspace)?;
        let namespace = namespace.into();
        let table_name = format!("vector_{}", namespace_table_name(&namespace));
        Ok(Self {
            namespace,
            workspace,
            global_config,
            embedding_func,
            db,
            config,
            table_name,
        })
    }

    fn ops(&self, biz_id: &str) -> PgCrudOps<'_, D> {
        PgCrudOps {
            db: &self.db,
            table_name: self.table_name.clone(),
            workspace: self.workspace.clone(),
            biz_id: biz_id.to_string(),
        }
    }

    /// tsvector + GIN index + auto-update trigger (non-fatal on failure).
    async fn initialize_full_text_search(&self) {
        let t = &self.table_name;
        let fts = sql_escape(&self.config.fts_config);
        let result: easyrag_core::Result<()> = async {
            self.db
                .execute(&format!("ALTER TABLE {t} ADD COLUMN IF NOT EXISTS content_tsv tsvector"))
                .await?;
            self.db
                .execute(&format!(
                    "CREATE INDEX IF NOT EXISTS idx_{t}_fts ON {t} USING gin(content_tsv)"
                ))
                .await?;
            self.db
                .execute(&format!(
                    "CREATE OR REPLACE FUNCTION {t}_tsv_update() RETURNS trigger AS $$
                    BEGIN
                        NEW.content_tsv := to_tsvector('{fts}', COALESCE(NEW.content->>'content', ''));
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql"
                ))
                .await?;
            self.db
                .execute(&format!("DROP TRIGGER IF EXISTS trg_{t}_tsv ON {t}"))
                .await?;
            self.db
                .execute(&format!(
                    "CREATE TRIGGER trg_{t}_tsv
                        BEFORE INSERT OR UPDATE OF content ON {t}
                        FOR EACH ROW EXECUTE FUNCTION {t}_tsv_update()"
                ))
                .await?;
            self.db
                .execute(&format!(
                    "UPDATE {t} SET content_tsv = to_tsvector('{fts}', COALESCE(content->>'content', ''))
                    WHERE content_tsv IS NULL"
                ))
                .await?;
            Ok(())
        }
        .await;
        if let Err(e) = result {
            tracing::warn!("Full-text search initialization failed (non-fatal): {e}");
        }
    }

    /// SQL fragment for time-range filtering.
    /// R1: compare against the business create_time stored in the JSONB content
    /// (epoch seconds). Rows lacking create_time (e.g. entity vectors, pre-R1
    /// records) fall back to the row-level created_at timestamp.
    fn time_filter(&self, start: Option<i64>, end: Option<i64>) -> String {
        let mut s = String::new();
        let expr = "COALESCE((content->>'create_time')::bigint, EXTRACT(EPOCH FROM created_at)::bigint)";
        if let Some(ts) = start {
            s.push_str(&format!(" AND {expr} >= {ts}"));
        }
        if let Some(ts) = end {
            s.push_str(&format!(" AND {expr} <= {ts}"));
        }
        s
    }

    /// Embed one batch and zip the vectors back onto its entries: the
    /// `prepare` step of [`batched_with_fallback`] for this backend.
    ///
    /// Entries with empty `content` get `None`, so the upsert leaves whatever
    /// embedding the row already has alone (`COALESCE(EXCLUDED.embedding, ..)`).
    /// The returned rows borrow the batch, which is why this runs once per
    /// batch instead of per record.
    async fn embed_batch<'e>(
        &self,
        batch: &'e [(String, JsonMap)],
    ) -> easyrag_core::Result<Vec<(&'e String, &'e JsonMap, Option<Vec<f32>>)>> {
        let texts: Vec<String> = batch
            .iter()
            .map(|(_, content)| {
                content
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        let embeddings = self
            .embedding_func
            .invoke(texts, Some(EmbeddingContext::Document))
            .await
            .map_err(|e| {
                // Propagate instead of silently inserting NULL vectors:
                // embedding failure must not leave documents invisible
                // to vector search.
                easyrag_core::CoreError::Internal(format!(
                    "PgVectorStorage embedding failed for table {}: {e}",
                    self.table_name
                ))
            })?;
        Ok(batch
            .iter()
            .enumerate()
            .map(|(i, (id, content))| {
                let text_empty = content
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.is_empty())
                    .unwrap_or(true);
                let emb = if text_empty {
                    None
                } else {
                    embeddings.get(i).cloned()
                };
                (id, content, emb)
            })
            .collect())
    }
}

#[async_trait]
impl<D: PgDb> StorageNameSpace for PgVectorStorage<D> {
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
        let t = &self.table_name;
        let ws = sql_escape(&self.workspace);
        if self.config.enable_vector {
            self.db.execute("CREATE EXTENSION IF NOT EXISTS vector").await?;
        }
        self.db
            .execute(&format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                    id VARCHAR(255) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{ws}',
                    biz_id VARCHAR(64) NOT NULL,
                    content JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({dim}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (id, workspace, biz_id)
                )",
                t = self.table_name,
                dim = self.config.embedding_dim
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{t}_ws_biz ON {t}(workspace, biz_id)"
            ))
            .await?;
        if self.config.enable_vector {
            self.db
                .execute(&format!(
                    "CREATE INDEX IF NOT EXISTS idx_{t}_hnsw
                    ON {t} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {}, ef_construction = {})",
                    self.config.hnsw_m, self.config.hnsw_ef
                ))
                .await?;
        }
        self.initialize_full_text_search().await;
        tracing::info!(
            "PgVectorStorage initialized: table={}, dim={}, workspace={}, fts={}",
            self.table_name,
            self.config.embedding_dim,
            self.workspace,
            self.config.fts_config
        );
        Ok(())
    }

    // index_done_callback: PostgreSQL commits immediately, no buffering, so the
    // trait default no-op is the whole implementation.

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let message = format!(
            "Table {} cleared for workspace={}, biz_id={}",
            self.table_name, self.workspace, biz_id
        );
        let outcome = self.ops(biz_id).drop_scope().await;
        if outcome.is_ok() {
            tracing::info!(
                "PgVectorStorage dropped scope: table={}, workspace={}, biz_id={}",
                self.table_name,
                self.workspace,
                biz_id
            );
        }
        Ok(drop_result(outcome, message))
    }
}

#[async_trait]
impl<D: PgDb> VectorStorage for PgVectorStorage<D> {
    fn embedding_func(&self) -> &EmbeddingFunc {
        &self.embedding_func
    }

    fn supports_time_range_filter(&self) -> bool {
        true
    }

    async fn query(
        &self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let embedding: Vec<f32> = match query_embedding {
            Some(e) => e.to_vec(),
            None => self
                .embedding_func
                .invoke(vec![query.to_string()], Some(EmbeddingContext::Query))
                .await?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    easyrag_core::CoreError::Internal("empty embedding result".to_string())
                })?,
        };
        let vector_str = vector_literal(&embedding);
        let time_filter = self.time_filter(time_range_start, time_range_end);
        let sql = format!(
            "SELECT id, content, 1 - (embedding <=> '{vector_str}'::vector) as similarity
             FROM {t}
             WHERE {scope} AND embedding IS NOT NULL{time_filter}
             ORDER BY embedding <=> '{vector_str}'::vector
             LIMIT {top_k}",
            t = self.table_name,
            // List path: an empty biz_id searches every biz line of the
            // workspace (BC-12/A-5), matching `list_scope_clause` on doc status.
            scope = self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        let threshold = self.cosine_better_than_threshold();
        let mut results = Vec::new();
        for row in rows {
            let mut content = read_content(&row).unwrap_or_default();
            let similarity = row
                .get("similarity")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            if let Some(id) = row.get("id") {
                content.insert("id".to_string(), id.clone());
            }
            content.insert(
                "score".to_string(),
                serde_json::Number::from_f64(similarity as f64)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            );
            if similarity >= threshold {
                results.push(content);
            }
        }
        Ok(results)
    }

    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let batch_size = self.config.upsert_batch_size.max(1);
        let entries: Vec<(String, JsonMap)> = data.into_iter().collect();
        let (table, workspace) = (self.table_name.as_str(), self.workspace.as_str());
        batched_with_fallback(
            &self.db,
            &entries,
            batch_size.min(PgBatchLimits::UPSERT_MAX_RECORDS_PER_BATCH),
            // One embed(texts) call per batch (option B), and never during the
            // row-by-row replay: a fallback must not re-embed (and re-pay for)
            // the records it is retrying.
            |batch| self.embed_batch(batch),
            |rows| vector_upsert_sql(table, workspace, biz_id, rows),
        )
        .await?;
        tracing::debug!(
            "PgVectorStorage upserted {} records to {}",
            entries.len(),
            self.table_name
        );
        Ok(())
    }

    async fn delete_entity(&self, entity_name: &str, biz_id: &str) -> easyrag_core::Result<()> {
        self.db
            .execute(&format!(
                "DELETE FROM {} WHERE {} AND id = '{}'",
                self.table_name,
                self.scope_clause(biz_id),
                sql_escape(entity_name)
            ))
            .await
    }

    async fn delete_entity_relation(&self, entity_name: &str, biz_id: &str) -> easyrag_core::Result<()> {
        self.db
            .execute(&format!(
                "DELETE FROM {} WHERE {}
                 AND (content->>'src_id' = '{}' OR content->>'tgt_id' = '{}')",
                self.table_name,
                self.scope_clause(biz_id),
                sql_escape(entity_name),
                sql_escape(entity_name)
            ))
            .await
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.ops(biz_id).fetch_content_by_id(id).await
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.ops(biz_id).fetch_contents_by_ids(ids).await
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.ops(biz_id).delete_by_ids(ids).await
    }

    async fn get_vectors_by_ids(
        &self,
        ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, Vec<f32>>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let in_list = sql_in_list(ids.iter().map(String::as_str));
        let sql = format!(
            "SELECT id, embedding FROM {} WHERE {} AND id IN ({in_list})",
            self.table_name,
            self.scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        let mut result = HashMap::new();
        for row in rows {
            let Some(id) = row.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let emb_str = row
                .get("embedding")
                .and_then(|v| v.as_str())
                .unwrap_or("[]");
            result.insert(id.to_string(), parse_vector_string(emb_str));
        }
        Ok(result)
    }
}

/// Render the batched vector upsert. Rows without an embedding keep their
/// existing embedding on conflict:
/// `embedding = COALESCE(EXCLUDED.embedding, table.embedding)`.
fn vector_upsert_sql(
    table_name: &str,
    workspace: &str,
    biz_id: &str,
    rows: &[(&String, &JsonMap, Option<Vec<f32>>)],
) -> String {
    let assignments = format!(
        "content = EXCLUDED.content, \
         embedding = COALESCE(EXCLUDED.embedding, {table_name}.embedding), updated_at = NOW()"
    );
    let upsert = PgUpsertBuilder::new(
        table_name,
        "id, workspace, biz_id, content, embedding, updated_at",
        &assignments,
    );
    upsert.sql(rows.iter().map(|(id, content, embedding)| {
        let emb_sql = match embedding {
            Some(e) => format!("'{}'::vector", vector_literal(e)),
            None => "NULL".to_string(),
        };
        format!(
            "('{}', '{}', '{}', {}::jsonb, {emb_sql}, NOW())",
            sql_escape(id),
            sql_escape(workspace),
            sql_escape(biz_id),
            json_literal(content)
        )
    }))
}

#[async_trait]
impl<D: PgDb> KeywordSearchable for PgVectorStorage<D> {
    async fn keyword_query(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        // tsquery: prefix-match each term with OR semantics.
        // Strip tsquery operator characters so user input cannot produce an
        // invalid tsquery (which would fail at runtime instead of matching).
        let ts_query = query
            .split_whitespace()
            .map(|term| {
                let cleaned: String = term
                    .chars()
                    .filter(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
                    .collect();
                format!("{cleaned}:*")
            })
            .filter(|t| t != ":*")
            .collect::<Vec<_>>()
            .join(" | ");
        if ts_query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let fts = sql_escape(&self.config.fts_config);
        let time_filter = self.time_filter(time_range_start, time_range_end);
        let sql = format!(
            "SELECT id, content, ts_rank_cd(content_tsv, to_tsquery('{fts}', '{tsq}')) as rank
             FROM {t}
             WHERE {scope} AND content_tsv @@ to_tsquery('{fts}', '{tsq}'){time_filter}
             ORDER BY rank DESC
             LIMIT {top_k}",
            tsq = sql_escape(&ts_query),
            t = self.table_name,
            // List path: see `query` (BC-12/A-5).
            scope = self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        let mut results = Vec::new();
        for row in rows {
            let mut content = read_content(&row).unwrap_or_default();
            if let Some(id) = row.get("id") {
                content.insert("id".to_string(), id.clone());
            }
            let rank = row.get("rank").and_then(|v| v.as_f64()).unwrap_or(0.0);
            content.insert(
                "score".to_string(),
                serde_json::Number::from_f64(rank)
                    .map(JsonValue::Number)
                    .unwrap_or(JsonValue::Null),
            );
            results.push(content);
        }
        Ok(results)
    }

    async fn index_documents(
        &self,
        _docs: HashMap<String, JsonMap>,
        _biz_id: &str,
    ) -> easyrag_core::Result<()> {
        // No-op: PostgreSQL trigger automatically maintains content_tsv.
        Ok(())
    }

    async fn remove_documents(&self, _ids: &[String], _biz_id: &str) -> easyrag_core::Result<()> {
        // No-op: DELETE removes the row (and its tsvector) entirely.
        Ok(())
    }

    async fn commit_index(&self) -> easyrag_core::Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PgGraphStorage
// ---------------------------------------------------------------------------

/// PostgreSQL graph storage with recursive CTE traversal.
pub struct PgGraphStorage<D: PgDb = PgDatabase> {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    db: D,
    nodes_table: String,
    edges_table: String,
}

impl PgGraphStorage<PgDatabase> {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        db: PgDatabase,
    ) -> easyrag_core::Result<Self> {
        Self::with_options(namespace, workspace, JsonMap::new(), None, db)
    }
}

impl<D: PgDb> PgGraphStorage<D> {
    pub fn with_options(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
        db: D,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        validate_workspace(&workspace)?;
        let namespace = namespace.into();
        let suffix = namespace_table_name(&namespace);
        Ok(Self {
            namespace,
            workspace,
            global_config,
            embedding_func,
            db,
            nodes_table: format!("graph_nodes_{suffix}"),
            edges_table: format!("graph_edges_{suffix}"),
        })
    }

    fn parse_props(row: &JsonMap, key: &str) -> JsonMap {
        match row.get(key) {
            Some(JsonValue::Object(m)) => m.clone(),
            Some(JsonValue::String(s)) => serde_json::from_str(s).unwrap_or_default(),
            _ => JsonMap::new(),
        }
    }

    /// Chunk-id lists longer than this are queried in shards and unioned, so a
    /// document with thousands of chunks cannot inflate one SQL literal without
    /// bound. `PgDb` has no bind parameters — every value is concatenated in.
    const SOURCE_ID_QUERY_SHARD: usize = 1000;

    /// `source_id` membership predicate, rendered identically for nodes and
    /// edges.
    ///
    /// Two arms, because both stored shapes are live until the backfill has run
    /// on every deployment:
    ///
    /// - `(properties -> 'source_id') ?| ARRAY[...]` — the canonical array. `?|`
    ///   is "has any of these keys", which on a jsonb *array* means element
    ///   equality, and it is the arm served by the GIN index built in
    ///   `migrate_source_id_to_array`. Note `jsonb_path_exists` would not be a
    ///   substitute: the function form is not index-resolvable, only `@?`/`@@`.
    /// - `string_to_array(...) && ARRAY[...]` — the legacy joined string. `?|`
    ///   is *false* for a jsonb string scalar, so without this arm every
    ///   pre-migration row (and anything an older pod writes during a rolling
    ///   deploy) would be invisible to cleanup and leak as an orphan forever.
    ///   `&&` is array overlap = element equality, so it is just as immune to the
    ///   `chunk_1`/`chunk_10` collision that a `LIKE '%chunk_1%'` would cause.
    ///
    /// The legacy arm does not `btrim` the split parts: the write side never
    /// emits padding, and the backfill trims every value it converts.
    fn source_id_predicate(chunk_ids: &[String]) -> String {
        let ids = sql_in_list(chunk_ids.iter().map(String::as_str));
        format!(
            "((properties -> 'source_id') ?| ARRAY[{ids}]::text[] \
             OR (jsonb_typeof(properties -> 'source_id') = 'string' \
             AND string_to_array(replace(properties ->> 'source_id', \
             '{sep}', ','), ',') && ARRAY[{ids}]::text[]))",
            ids = ids,
            sep = easyrag_core::constants::GRAPH_FIELD_SEP,
        )
    }

    /// Rows in `t` still holding the legacy joined-string `source_id`.
    /// Served by the partial index created in [`Self::migrate_source_id_to_array`],
    /// so once the backfill has run this is an empty index scan rather than a
    /// sequential one — which is what makes repeating it on every workspace
    /// rebuild affordable.
    async fn count_legacy_source_ids(&self, t: &str) -> easyrag_core::Result<i64> {
        let row = self
            .db
            .fetch_one(&format!(
                "SELECT COUNT(*) as cnt FROM {t} WHERE workspace = '{ws}' \
                 AND jsonb_typeof(properties -> 'source_id') = 'string'",
                ws = sql_escape(&self.workspace)
            ))
            .await?;
        Ok(row
            .and_then(|r| r.get("cnt").and_then(|v| v.as_i64()))
            .unwrap_or(0))
    }

    /// Rewrite every legacy `<SEP>`/`,`-joined `source_id` string into the
    /// canonical JSON array, and index the field so the cleanup filters stop
    /// scanning the table.
    ///
    /// Called from `initialize()`, which the workspace cache runs on *every*
    /// bundle rebuild (LRU eviction at 32 entries, or an explicit invalidate),
    /// so "cheap to repeat" is the design constraint rather than an
    /// afterthought — hence the partial index first.
    ///
    /// Non-fatal, exactly like `initialize_full_text_search`: a failed migration
    /// must not stop the server from serving. It is also not a correctness
    /// dependency, because `source_id_predicate` keeps its legacy
    /// `string_to_array(...) && ARRAY[...]` arm permanently — an unmigrated row
    /// is still cleaned up correctly, just without index assistance.
    async fn migrate_source_id_to_array(&self) {
        // Index names use `_srcid`, not `_src`: `initialize()` already creates
        // `idx_{edges}_src` as a btree over the edges' `source` *column*, and
        // `CREATE INDEX IF NOT EXISTS` would silently skip the GIN index on that
        // name, leaving every edge query a sequential scan with nothing in the
        // logs to say so.
        let result: easyrag_core::Result<()> = async {
            for t in [&self.nodes_table, &self.edges_table] {
                // (a) Partial index over exactly the rows that still need
                // converting; doubles as the migration-progress probe. After a
                // successful backfill it holds zero rows.
                self.db
                    .execute(&format!(
                        "CREATE INDEX IF NOT EXISTS idx_{t}_srcid_legacy ON {t} (id) \
                         WHERE jsonb_typeof(properties -> 'source_id') = 'string'"
                    ))
                    .await?;
                // (b) Expression GIN so the `?|` arm of `source_id_predicate` is
                // index-resolvable. Default `jsonb_ops` — `jsonb_path_ops` cannot
                // serve `?|`, only `@>`. Not CONCURRENTLY, matching the other
                // index DDL in `initialize()`; an operator who cannot afford the
                // build lock can create it by hand first and `IF NOT EXISTS`
                // short-circuits here.
                self.db
                    .execute(&format!(
                        "CREATE INDEX IF NOT EXISTS idx_{t}_srcid ON {t} \
                         USING gin ((properties -> 'source_id'))"
                    ))
                    .await?;

                let before = self.count_legacy_source_ids(t).await?;

                // (c) One set-based statement, idempotent by construction: the
                // WHERE matches only jsonb strings and the SET turns them into
                // arrays, so an interrupted run resumes where it stopped and a
                // finished run rewrites nothing. Splitting on `<SEP>` *and* `,`,
                // trimming and dropping empties is the rule in
                // `parse_source_ids`, so a migrated value reads back as the same
                // list the string did. `WITH ORDINALITY` + `ORDER BY` is what
                // keeps the element order: `jsonb_agg` does not promise one, and
                // FIFO capping in `merge_source_ids` drops from the front, so a
                // reordered array would evict the wrong chunk ids later.
                self.db
                    .execute(&format!(
                        "UPDATE {t} SET properties = jsonb_set(properties, '{{source_id}}', \
                         COALESCE((SELECT jsonb_agg(btrim(x.part) ORDER BY x.ord) \
                         FROM unnest(string_to_array(replace(properties ->> 'source_id', \
                         '{sep}', ','), ',')) WITH ORDINALITY AS x(part, ord) \
                         WHERE btrim(x.part) <> ''), '[]'::jsonb)), updated_at = NOW() \
                         WHERE workspace = '{ws}' \
                         AND jsonb_typeof(properties -> 'source_id') = 'string'",
                        sep = easyrag_core::constants::GRAPH_FIELD_SEP,
                        ws = sql_escape(&self.workspace)
                    ))
                    .await?;

                // (d) Probe again: zero is the migration-complete gate.
                let after = self.count_legacy_source_ids(t).await?;
                if after == 0 {
                    tracing::info!(
                        "Graph source_id backfill: {before} legacy value(s) in {t} now arrays"
                    );
                } else {
                    tracing::warn!(
                        "{after} row(s) in {t} still hold a string source_id after the backfill \
                         (concurrent writes, or a permissions problem). Cleanup stays correct via \
                         the legacy filter arm; the next initialize() retries."
                    );
                }
            }
            Ok(())
        }
        .await;
        if let Err(e) = result {
            tracing::warn!("Graph source_id migration failed (non-fatal): {e}");
        }
    }

    /// Read one node with an **already-rendered** scope clause, so a traversal
    /// and the per-node lookups it drives cannot disagree on tenant filtering
    /// (BC-12/A-5: `get_knowledge_graph` widens both, `get_node` widens neither).
    async fn fetch_node(&self, node_id: &str, scope: &str) -> easyrag_core::Result<Option<JsonMap>> {
        let sql = format!(
            "SELECT id, properties FROM {} WHERE id = '{}' AND {}",
            self.nodes_table,
            sql_escape(node_id),
            scope
        );
        let Some(row) = self.db.fetch_one(&sql).await? else {
            return Ok(None);
        };
        let mut properties = Self::parse_props(&row, "properties");
        properties.insert("id".to_string(), JsonValue::String(node_id.to_string()));
        Ok(Some(properties))
    }
}

#[async_trait]
impl<D: PgDb> StorageNameSpace for PgGraphStorage<D> {
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
        let ws = sql_escape(&self.workspace);
        self.db
            .execute(&format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                    id VARCHAR(512) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{ws}',
                    biz_id VARCHAR(64) NOT NULL,
                    properties JSONB NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (id, workspace, biz_id)
                )",
                t = self.nodes_table
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                    id VARCHAR(1024) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{ws}',
                    biz_id VARCHAR(64) NOT NULL,
                    source VARCHAR(512) NOT NULL,
                    target VARCHAR(512) NOT NULL,
                    properties JSONB NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (id, workspace, biz_id)
                )",
                t = self.edges_table
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{}_ws_biz ON {}(workspace, biz_id)",
                self.nodes_table, self.nodes_table
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{}_ws_biz ON {}(workspace, biz_id)",
                self.edges_table, self.edges_table
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{}_src ON {}(source, workspace, biz_id)",
                self.edges_table, self.edges_table
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{}_tgt ON {}(target, workspace, biz_id)",
                self.edges_table, self.edges_table
            ))
            .await?;
        tracing::info!(
            "PgGraphStorage initialized: nodes={}, edges={}, workspace={}",
            self.nodes_table,
            self.edges_table,
            self.workspace
        );
        self.migrate_source_id_to_array().await;
        Ok(())
    }

    // index_done_callback: trait default no-op (postgres commits eagerly).

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let message = format!(
            "Tables {}, {} cleared for workspace={}, biz_id={}",
            self.nodes_table, self.edges_table, self.workspace, biz_id
        );
        let outcome: easyrag_core::Result<()> = async {
            self.db
                .execute(&format!(
                    "DELETE FROM {} WHERE {}",
                    self.edges_table,
                    self.scope_clause(biz_id)
                ))
                .await?;
            self.db
                .execute(&format!(
                    "DELETE FROM {} WHERE {}",
                    self.nodes_table,
                    self.scope_clause(biz_id)
                ))
                .await?;
            Ok(())
        }
        .await;
        if outcome.is_ok() {
            tracing::info!(
                "PgGraphStorage dropped scope: nodes={}, edges={}, workspace={}, biz_id={}",
                self.nodes_table,
                self.edges_table,
                self.workspace,
                biz_id
            );
        }
        Ok(drop_result(outcome, message))
    }
}

/// Render the batched graph-node upsert: one statement per batch.
///
/// Same columns and same `DO UPDATE SET` as the single-row `upsert_node`, which
/// keeps its own hand-laid multi-line SQL because golden snapshots pin those
/// bytes — this is an *additional* batch statement, not a rewrite of it. The
/// graph cleanup trim writes every surviving node back, so without it a
/// 200-node trim was 200 round trips.
fn graph_nodes_upsert_sql(
    table_name: &str,
    workspace: &str,
    biz_id: &str,
    batch: &[&(String, JsonMap)],
) -> String {
    let upsert = PgUpsertBuilder::new(
        table_name,
        "id, workspace, biz_id, properties, updated_at",
        "properties = EXCLUDED.properties, updated_at = NOW()",
    );
    upsert.sql(batch.iter().map(|(node_id, node_data)| {
        format!(
            "('{}', '{}', '{}', {}::jsonb, NOW())",
            sql_escape(node_id),
            sql_escape(workspace),
            sql_escape(biz_id),
            json_literal(node_data)
        )
    }))
}

#[async_trait]
impl<D: PgDb> GraphStorage for PgGraphStorage<D> {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn has_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<bool> {
        let sql = format!(
            "SELECT 1 as exists_flag FROM {} WHERE id = '{}' AND {}",
            self.nodes_table,
            sql_escape(node_id),
            self.scope_clause(biz_id)
        );
        Ok(self.db.fetch_one(&sql).await?.is_some())
    }

    async fn has_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<bool> {
        let sql = format!(
            "SELECT 1 as exists_flag FROM {e}
             WHERE {scope} AND ((source = '{src}' AND target = '{tgt}') OR (source = '{tgt}' AND target = '{src}'))",
            e = self.edges_table,
            scope = self.scope_clause(biz_id),
            src = sql_escape(source_node_id),
            tgt = sql_escape(target_node_id)
        );
        Ok(self.db.fetch_one(&sql).await?.is_some())
    }

    async fn node_degree(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        let sql = format!(
            "SELECT COUNT(*) as degree FROM {e}
             WHERE {scope} AND (source = '{id}' OR target = '{id}')",
            e = self.edges_table,
            scope = self.scope_clause(biz_id),
            id = sql_escape(node_id)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row
            .and_then(|r| r.get("degree").and_then(|v| v.as_u64()))
            .unwrap_or(0) as usize)
    }

    async fn edge_degree(&self, src_id: &str, tgt_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        Ok(self.node_degree(src_id, biz_id).await? + self.node_degree(tgt_id, biz_id).await?)
    }

    async fn get_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        // Single-record read: exact scope, never the wildcard.
        self.fetch_node(node_id, &self.scope_clause(biz_id)).await
    }

    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        let sql = format!(
            "SELECT id, source, target, properties FROM {e}
             WHERE {scope} AND ((source = '{src}' AND target = '{tgt}') OR (source = '{tgt}' AND target = '{src}'))",
            e = self.edges_table,
            scope = self.scope_clause(biz_id),
            src = sql_escape(source_node_id),
            tgt = sql_escape(target_node_id)
        );
        let Some(row) = self.db.fetch_one(&sql).await? else {
            return Ok(None);
        };
        let mut properties = Self::parse_props(&row, "properties");
        if let Some(src) = row.get("source") {
            properties.insert("source".to_string(), src.clone());
        }
        if let Some(tgt) = row.get("target") {
            properties.insert("target".to_string(), tgt.clone());
        }
        Ok(Some(properties))
    }

    async fn get_node_edges(
        &self,
        source_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<Vec<(String, String)>>> {
        let sql = format!(
            "SELECT source, target FROM {e}
             WHERE {scope} AND (source = '{id}' OR target = '{id}')",
            e = self.edges_table,
            scope = self.scope_clause(biz_id),
            id = sql_escape(source_node_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        if rows.is_empty() {
            return Ok(None);
        }
        let pairs = rows
            .iter()
            .filter_map(|r| {
                let src = r.get("source").and_then(|v| v.as_str())?.to_string();
                let tgt = r.get("target").and_then(|v| v.as_str())?.to_string();
                Some((src, tgt))
            })
            .collect();
        Ok(Some(pairs))
    }

    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.db
            .execute(&format!(
                "INSERT INTO {t} (id, workspace, biz_id, properties, updated_at)
                 VALUES ('{id}', '{ws}', '{biz}', {props}::jsonb, NOW())
                 ON CONFLICT ({conflict}) DO UPDATE SET properties = {props}::jsonb, updated_at = NOW()",
                t = self.nodes_table,
                conflict = TENANT_CONFLICT_TARGET,
                id = sql_escape(node_id),
                ws = sql_escape(&self.workspace),
                biz = sql_escape(biz_id),
                props = json_literal(&node_data)
            ))
            .await
    }

    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let edge_id = format!("{source_node_id}-{target_node_id}");
        self.db
            .execute(&format!(
                "INSERT INTO {t} (id, workspace, biz_id, source, target, properties, updated_at)
                 VALUES ('{id}', '{ws}', '{biz}', '{src}', '{tgt}', {props}::jsonb, NOW())
                 ON CONFLICT ({conflict}) DO UPDATE SET properties = {props}::jsonb, updated_at = NOW()",
                t = self.edges_table,
                conflict = TENANT_CONFLICT_TARGET,
                id = sql_escape(&edge_id),
                ws = sql_escape(&self.workspace),
                biz = sql_escape(biz_id),
                src = sql_escape(source_node_id),
                tgt = sql_escape(target_node_id),
                props = json_literal(&edge_data)
            ))
            .await
    }

    async fn upsert_nodes_batch(
        &self,
        nodes: &[(String, JsonMap)],
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        // Borrowed like the KV path, so the property maps (full description
        // text) are never cloned just to be written.
        let entries: Vec<&(String, JsonMap)> = nodes.iter().collect();
        let (table, workspace) = (self.nodes_table.as_str(), self.workspace.as_str());
        batched_with_fallback(
            &self.db,
            &entries,
            PgBatchLimits::UPSERT_MAX_RECORDS_PER_BATCH,
            // Nothing to prepare: the properties are already in hand, unlike the
            // vector path which embeds per batch.
            |batch| async move { Ok(batch.to_vec()) },
            |rows| graph_nodes_upsert_sql(table, workspace, biz_id, rows),
        )
        .await
    }

    async fn delete_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<()> {
        self.db
            .execute(&format!(
                "DELETE FROM {e} WHERE {scope} AND (source = '{id}' OR target = '{id}')",
                e = self.edges_table,
                scope = self.scope_clause(biz_id),
                id = sql_escape(node_id)
            ))
            .await?;
        self.db
            .execute(&format!(
                "DELETE FROM {n} WHERE id = '{id}' AND {scope}",
                n = self.nodes_table,
                id = sql_escape(node_id),
                scope = self.scope_clause(biz_id)
            ))
            .await
    }

    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        // Two statements per shard instead of two per node: looping
        // `delete_node` turned a 200-node cleanup into 400 round trips. Within a
        // shard the edges still go first, exactly like `delete_node`, so an
        // interruption leaves no edge pointing at a node that is already gone.
        for batch in node_ids.chunks(PgBatchLimits::DELETE_MAX_RECORDS_PER_BATCH) {
            let ids = sql_in_list(batch.iter().map(String::as_str));
            self.db
                .execute(&format!(
                    "DELETE FROM {e} WHERE {scope} AND (source IN ({ids}) OR target IN ({ids}))",
                    e = self.edges_table,
                    scope = self.scope_clause(biz_id),
                ))
                .await?;
            self.db
                .execute(&format!(
                    "DELETE FROM {n} WHERE id IN ({ids}) AND {scope}",
                    n = self.nodes_table,
                    scope = self.scope_clause(biz_id),
                ))
                .await?;
        }
        Ok(())
    }

    async fn remove_edges(
        &self,
        edge_list: &[(String, String)],
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        // One statement per shard instead of one per edge. The two row-value
        // `IN (VALUES ...)` lists keep the undirected match the per-edge DELETE
        // had — stored direction and its reverse — without repeating the whole
        // predicate once per pair.
        for batch in edge_list.chunks(PgBatchLimits::DELETE_MAX_RECORDS_PER_BATCH) {
            let values = batch
                .iter()
                .map(|(src, tgt)| format!("('{}','{}')", sql_escape(src), sql_escape(tgt)))
                .collect::<Vec<_>>()
                .join(",");
            self.db
                .execute(&format!(
                    "DELETE FROM {e} WHERE {scope}
                     AND ((source, target) IN (VALUES {values}) OR (target, source) IN (VALUES {values}))",
                    e = self.edges_table,
                    scope = self.scope_clause(biz_id)
                ))
                .await?;
        }
        Ok(())
    }

    async fn get_all_labels(&self, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let sql = format!(
            "SELECT id FROM {} WHERE {} ORDER BY id",
            self.nodes_table,
            self.scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect())
    }

    async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        // No starting label (or the `*` wildcard):
        // return all nodes and edges.
        let Some(label) = node_label.filter(|l| *l != "*") else {
            let nodes = self.get_all_nodes(biz_id).await?;
            let edges = self.get_all_edges(biz_id).await?;
            // BC-1/A-6: `>=`, not `>`. A graph that exactly fills the budget is
            // reported truncated, matching the CTE path below, the in-memory
            // backend and Neo4j. `nodes` is the pre-`take` count, so this is the
            // total visible node count, not the number actually returned.
            let truncated = nodes.len() >= max_nodes;
            let graph_nodes: Vec<KnowledgeGraphNode> = nodes
                .into_iter()
                .take(max_nodes)
                .map(|mut props| {
                    let id = props
                        .remove("id")
                        .and_then(|v| v.as_str().map(|s| s.to_string()))
                        .unwrap_or_default();
                    KnowledgeGraphNode::self_labeled(id, stringify_props(&props))
                })
                .collect();
            let graph_edges: Vec<KnowledgeGraphEdge> = edges
                .into_iter()
                .map(|props| {
                    let src = props
                        .get("source")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let tgt = props
                        .get("target")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    KnowledgeGraphEdge::untyped(src, tgt, JsonMap::new())
                })
                .collect();
            return Ok(KnowledgeGraph {
                nodes: graph_nodes,
                edges: graph_edges,
                is_truncated: truncated,
            });
        };

        // Recursive CTE traversal from the starting node.
        let sql = format!(
            "WITH RECURSIVE graph_traversal AS (
                SELECT source, target, 1 as depth FROM {e}
                WHERE {scope} AND (source = '{label}' OR target = '{label}')
                UNION
                SELECT e.source, e.target, gt.depth + 1
                FROM {e} e
                JOIN graph_traversal gt ON (e.source = gt.target OR e.target = gt.source)
                WHERE {e_scope} AND gt.depth < {max_depth}
            )
            SELECT DISTINCT source, target FROM graph_traversal LIMIT {max_nodes}",
            e = self.edges_table,
            // List path: an empty biz_id spans every biz line (BC-12/A-5),
            // matching `Neo4jGraphStorage::get_sub_graph`. The per-node lookups
            // below render the SAME scope so traversal and properties agree.
            scope = self.list_scope_clause(biz_id),
            e_scope = self.list_scope_clause_as(biz_id, "e"),
            label = sql_escape(label)
        );
        let rows = self.db.fetch_all(&sql).await?;

        let mut edges = Vec::new();
        for row in rows {
            let Some(src) = row.get("source").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(tgt) = row.get("target").and_then(|v| v.as_str()) else {
                continue;
            };
            edges.push(KnowledgeGraphEdge::untyped(src, tgt, JsonMap::new()));
        }
        // Seed first, then endpoints in first-seen order; that ordering decides
        // which nodes survive the per-node fetch below (shared with the other
        // backends — see `KnowledgeGraph::expand_from_edges`).
        let node_ids = KnowledgeGraph::expand_from_edges(label, &edges);

        let mut nodes = Vec::with_capacity(node_ids.len());
        // Same (possibly widened) scope the traversal used — see the CTE above.
        let node_scope = self.list_scope_clause(biz_id);
        for node_id in &node_ids {
            let node_data = self.fetch_node(node_id, &node_scope).await?;
            let mut properties = node_data.unwrap_or_default();
            properties.remove("id");
            nodes.push(KnowledgeGraphNode::self_labeled(
                node_id.clone(),
                stringify_props(&properties),
            ));
        }

        Ok(KnowledgeGraph {
            is_truncated: node_ids.len() >= max_nodes,
            nodes,
            edges,
        })
    }

    async fn get_all_nodes(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let sql = format!(
            "SELECT id, properties FROM {} WHERE {}",
            self.nodes_table,
            // List path: an empty biz_id spans every biz line (BC-12/A-5),
            // matching `Neo4jGraphStorage::node_scope`.
            self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let mut properties = Self::parse_props(&row, "properties");
                if let Some(id) = row.get("id").and_then(|v| v.as_str()) {
                    properties.insert("id".to_string(), JsonValue::String(id.to_string()));
                }
                properties
            })
            .collect())
    }

    async fn get_all_edges(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let sql = format!(
            "SELECT source, target, properties FROM {} WHERE {}",
            self.edges_table,
            // List path: see `get_all_nodes`.
            self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .into_iter()
            .map(|row| {
                let mut properties = Self::parse_props(&row, "properties");
                if let Some(src) = row.get("source").and_then(|v| v.as_str()) {
                    properties.insert("source".to_string(), JsonValue::String(src.to_string()));
                }
                if let Some(tgt) = row.get("target").and_then(|v| v.as_str()) {
                    properties.insert("target".to_string(), JsonValue::String(tgt.to_string()));
                }
                properties
            })
            .collect())
    }

    async fn get_nodes_by_source_ids(
        &self,
        chunk_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if chunk_ids.is_empty() {
            // Never a full scan: "which nodes reference none of these chunks"
            // is an empty question, not a request for the whole graph.
            return Ok(Vec::new());
        }
        // List path, exactly like `get_all_nodes`: an empty biz_id widens to the
        // whole workspace (BC-12/A-5).
        let scope = self.list_scope_clause(biz_id);
        let mut out: Vec<JsonMap> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for shard in chunk_ids.chunks(Self::SOURCE_ID_QUERY_SHARD) {
            let sql = format!(
                "SELECT id, properties FROM {} WHERE {} AND {}",
                self.nodes_table,
                scope,
                Self::source_id_predicate(shard)
            );
            for row in self.db.fetch_all(&sql).await? {
                let id = row.get("id").and_then(|v| v.as_str());
                // A node referencing chunks in two shards comes back twice.
                if let Some(id) = id {
                    if !seen.insert(id.to_string()) {
                        continue;
                    }
                }
                let mut properties = Self::parse_props(&row, "properties");
                if let Some(id) = id {
                    properties.insert("id".to_string(), JsonValue::String(id.to_string()));
                }
                out.push(properties);
            }
        }
        Ok(out)
    }

    async fn get_edges_by_source_ids(
        &self,
        chunk_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let scope = self.list_scope_clause(biz_id);
        let mut out: Vec<JsonMap> = Vec::new();
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for shard in chunk_ids.chunks(Self::SOURCE_ID_QUERY_SHARD) {
            let sql = format!(
                "SELECT source, target, properties FROM {} WHERE {} AND {}",
                self.edges_table,
                scope,
                Self::source_id_predicate(shard)
            );
            for row in self.db.fetch_all(&sql).await? {
                let src = row.get("source").and_then(|v| v.as_str());
                let tgt = row.get("target").and_then(|v| v.as_str());
                if !seen.insert((
                    src.unwrap_or_default().to_string(),
                    tgt.unwrap_or_default().to_string(),
                )) {
                    continue;
                }
                let mut properties = Self::parse_props(&row, "properties");
                if let Some(src) = src {
                    properties.insert("source".to_string(), JsonValue::String(src.to_string()));
                }
                if let Some(tgt) = tgt {
                    properties.insert("target".to_string(), JsonValue::String(tgt.to_string()));
                }
                out.push(properties);
            }
        }
        Ok(out)
    }

    async fn get_popular_labels(&self, limit: usize, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let sql = format!(
            "SELECT n.id, COUNT(e.id) as degree
             FROM {n} n
             LEFT JOIN {e} e ON n.workspace = e.workspace AND n.biz_id = e.biz_id AND (n.id = e.source OR n.id = e.target)
             WHERE {scope}
             GROUP BY n.id
             ORDER BY degree DESC
             LIMIT {limit}",
            n = self.nodes_table,
            e = self.edges_table,
            // List path: the `/api/graph/labels` route reaches this through
            // `read_bundle`, which blanks an omitted biz_id to the read-path
            // wildcard. An empty biz_id must span every biz line (BC-12/A-5),
            // matching `Neo4jGraphStorage::get_popular_labels` (`node_scope`)
            // and the in-memory backend. The JOIN still links a node to edges
            // on its own biz line via `n.biz_id = e.biz_id`.
            scope = self.list_scope_clause_as(biz_id, "n")
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect())
    }

    async fn search_labels(
        &self,
        query: &str,
        limit: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        let sql = format!(
            "SELECT id FROM {} WHERE {} AND id ILIKE '%{}%' LIMIT {limit}",
            self.nodes_table,
            // List path: see `get_all_nodes`. Neo4j widens `search_labels` too.
            self.list_scope_clause(biz_id),
            sql_escape(query)
        );
        let rows = self.db.fetch_all(&sql).await?;
        Ok(rows
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// PgDocStatusStorage
// ---------------------------------------------------------------------------

/// PostgreSQL document status storage.
pub struct PgDocStatusStorage<D: PgDb = PgDatabase> {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    db: D,
    table_name: String,
}

impl PgDocStatusStorage<PgDatabase> {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        db: PgDatabase,
    ) -> easyrag_core::Result<Self> {
        Self::with_options(namespace, workspace, JsonMap::new(), None, db)
    }
}

impl<D: PgDb> PgDocStatusStorage<D> {
    pub fn with_options(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
        db: D,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        validate_workspace(&workspace)?;
        let namespace = namespace.into();
        let table_name = format!("doc_status_{}", namespace_table_name(&namespace));
        Ok(Self {
            namespace,
            workspace,
            global_config,
            embedding_func,
            db,
            table_name,
        })
    }

    fn ops(&self, biz_id: &str) -> PgCrudOps<'_, D> {
        PgCrudOps {
            db: &self.db,
            table_name: self.table_name.clone(),
            workspace: self.workspace.clone(),
            biz_id: biz_id.to_string(),
        }
    }

    /// Map a result row to a DocProcessingStatus.
    ///
    /// The payload fields come from the shared core mapper over the unwrapped
    /// `content` JSONB, which is what lets `chunks_list`/`error_msg`/`metadata`/
    /// `multimodal_processed` survive the round-trip instead of being hardcoded
    /// empty (A-1 / BC-9). Only the columns postgres stores *outside* `content`
    /// are read from the row and take precedence.
    fn map_row_to_doc_status(row: &JsonMap) -> DocProcessingStatus {
        let content = read_content(row).unwrap_or_default();
        let mut status = DocProcessingStatus::from_json(&content);
        // Envelope columns: real table columns, authoritative over `content`.
        status.status = DocStatus::from_value(
            row.get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("pending"),
        )
        .unwrap_or(DocStatus::Pending);
        status.created_at = row_timestamp(row, "created_at");
        status.updated_at = row_timestamp(row, "updated_at");
        status.biz_id = row
            .get("biz_id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        status
    }

    async fn fetch_docs_by_condition(
        &self,
        condition: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        let sql = format!(
            "SELECT id, biz_id, content, status, created_at, updated_at FROM {} WHERE {}{}",
            self.table_name,
            self.list_scope_clause(biz_id),
            condition
        );
        let rows = self.db.fetch_all(&sql).await?;
        let mut result = HashMap::new();
        for row in rows {
            let Some(id) = row.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            result.insert(id.to_string(), Self::map_row_to_doc_status(&row));
        }
        Ok(result)
    }
}

#[async_trait]
impl<D: PgDb> StorageNameSpace for PgDocStatusStorage<D> {
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
        let ws = sql_escape(&self.workspace);
        self.db
            .execute(&format!(
                "CREATE TABLE IF NOT EXISTS {t} (
                    id VARCHAR(255) NOT NULL,
                    workspace VARCHAR(255) NOT NULL DEFAULT '{ws}',
                    biz_id VARCHAR(64) NOT NULL,
                    content JSONB NOT NULL DEFAULT '{{}}',
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    track_id VARCHAR(255),
                    file_path TEXT,
                    content_hash VARCHAR(64),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (id, workspace, biz_id)
                )",
                t = self.table_name
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{t}_status ON {t}(workspace, biz_id, status)",
                t = self.table_name
            ))
            .await?;
        self.db
            .execute(&format!(
                "CREATE INDEX IF NOT EXISTS idx_{t}_track ON {t}(workspace, biz_id, track_id)",
                t = self.table_name
            ))
            .await?;
        tracing::info!(
            "PgDocStatusStorage initialized: table={}, workspace={}",
            self.table_name,
            self.workspace
        );
        Ok(())
    }

    // index_done_callback: trait default no-op (postgres commits eagerly).

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let message = format!(
            "Table {} cleared for workspace={}, biz_id={}",
            self.table_name, self.workspace, biz_id
        );
        let outcome = self.ops(biz_id).drop_scope().await;
        if outcome.is_ok() {
            tracing::info!(
                "PgDocStatusStorage dropped scope: table={}, workspace={}, biz_id={}",
                self.table_name,
                self.workspace,
                biz_id
            );
        }
        Ok(drop_result(outcome, message))
    }
}

#[async_trait]
impl<D: PgDb> KVStorage for PgDocStatusStorage<D> {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.ops(biz_id).fetch_content_by_id(id).await
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.ops(biz_id).fetch_contents_by_ids(ids).await
    }

    async fn filter_keys(
        &self,
        keys: &HashSet<String>,
        biz_id: &str,
    ) -> easyrag_core::Result<HashSet<String>> {
        let existing = self.ops(biz_id).fetch_existing_ids(keys).await?;
        Ok(keys.difference(&existing).cloned().collect())
    }

    async fn upsert(
        &self,
        data: HashMap<String, JsonMap>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        for (id, content) in &data {
            // Merge semantics: partial updates (e.g. status transitions) must not wipe
            // fields written at insert time (file_path, content_hash, external_id, ...).
            // Extracted columns keep their existing value when absent from the payload.
            let status = content.get("status").and_then(|v| v.as_str());
            let track_id = content.get("track_id").and_then(|v| v.as_str());
            let file_path = content.get("file_path").and_then(|v| v.as_str());
            let content_hash = content.get("content_hash").and_then(|v| v.as_str());
            self.db
                .execute(&format!(
                    "INSERT INTO {t} (id, workspace, biz_id, content, status, track_id, file_path, content_hash, updated_at)
                     VALUES ('{id}', '{ws}', '{biz}', {content}::jsonb, COALESCE({status}, 'pending'), {track}, {path}, {hash}, NOW())
                     ON CONFLICT ({conflict}) DO UPDATE
                     SET content = COALESCE({t}.content, '{{}}'::jsonb) || {content}::jsonb,
                         status = COALESCE({status}, {t}.status),
                         track_id = COALESCE({track}, NULLIF({t}.track_id, '')),
                         file_path = COALESCE({path}, NULLIF({t}.file_path, '')),
                         content_hash = COALESCE({hash}, NULLIF({t}.content_hash, '')),
                         updated_at = NOW()",
                    t = self.table_name,
                    conflict = TENANT_CONFLICT_TARGET,
                    id = sql_escape(id),
                    ws = sql_escape(&self.workspace),
                    biz = sql_escape(biz_id),
                    content = json_literal(content),
                    status = opt_sql_literal(status),
                    track = opt_sql_literal(track_id),
                    path = opt_sql_literal(file_path),
                    hash = opt_sql_literal(content_hash)
                ))
                .await?;
        }
        tracing::debug!(
            "PgDocStatusStorage upserted {} records to {}",
            data.len(),
            self.table_name
        );
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.ops(biz_id).delete_by_ids(ids).await
    }

    async fn is_empty(&self, biz_id: &str) -> easyrag_core::Result<bool> {
        Ok(self.ops(biz_id).count().await? == 0)
    }
}

#[async_trait]
impl<D: PgDb> DocStatusStorage for PgDocStatusStorage<D> {
    async fn get_status_counts(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, usize>> {
        let sql = format!(
            "SELECT status, COUNT(*) as cnt FROM {} WHERE {} GROUP BY status",
            self.table_name,
            self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        let mut result = HashMap::new();
        for row in rows {
            let Some(status) = row.get("status").and_then(|v| v.as_str()) else {
                continue;
            };
            let cnt = row.get("cnt").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            result.insert(status.to_string(), cnt);
        }
        Ok(result)
    }

    async fn get_docs_by_status(
        &self,
        status: DocStatus,
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        self.fetch_docs_by_condition(
            &format!(" AND status = '{}'", sql_escape(status.value())),
            biz_id,
        )
        .await
    }

    async fn get_docs_by_statuses(
        &self,
        statuses: &[DocStatus],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        if statuses.is_empty() {
            return Ok(HashMap::new());
        }
        let in_list = sql_in_list(statuses.iter().map(|s| s.value()));
        self.fetch_docs_by_condition(&format!(" AND status IN ({in_list})"), biz_id)
            .await
    }

    async fn get_docs_by_track_id(
        &self,
        track_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        self.fetch_docs_by_condition(
            &format!(" AND track_id = '{}'", sql_escape(track_id)),
            biz_id,
        )
        .await
    }

    async fn get_docs_paginated(
        &self,
        status_filter: Option<DocStatus>,
        status_filters: Option<&[DocStatus]>,
        page: usize,
        page_size: usize,
        sort_field: &str,
        sort_direction: &str,
        file_path_prefix: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<(Vec<(String, DocProcessingStatus)>, usize)> {
        let filter_values =
            easyrag_core::storage::traits::resolve_status_filter_values(status_filter, status_filters);
        let mut condition = match &filter_values {
            Some(values) if !values.is_empty() => {
                let mut sorted: Vec<&String> = values.iter().collect();
                sorted.sort();
                let in_list = sql_in_list(sorted.iter().map(|s| s.as_str()));
                format!(" AND status IN ({in_list})")
            }
            _ => String::new(),
        };
        if let Some(prefix) = file_path_prefix.map(str::trim).filter(|p| !p.is_empty()) {
            // R5: server-side prefix filter; escape LIKE wildcards in user input.
            condition.push_str(&format!(
                " AND file_path LIKE '{}%'",
                sql_escape(&escape_like_pattern(prefix))
            ));
        }

        // Total count.
        let count_sql = format!(
            "SELECT COUNT(*) as cnt FROM {} WHERE {}{}",
            self.table_name,
            self.list_scope_clause(biz_id),
            condition
        );
        let count_row = self.db.fetch_one(&count_sql).await?;
        let total_count = count_row
            .and_then(|r| r.get("cnt").and_then(|v| v.as_u64()))
            .unwrap_or(0) as usize;

        let offset = page.saturating_sub(1).saturating_mul(page_size);
        let safe_sort_field = if matches!(sort_field, "created_at" | "updated_at" | "status" | "id")
        {
            sort_field
        } else {
            "updated_at"
        };
        let safe_sort_dir = if sort_direction.eq_ignore_ascii_case("asc") {
            "ASC"
        } else {
            "DESC"
        };

        let sql = format!(
            "SELECT id, biz_id, content, status, created_at, updated_at FROM {t}
             WHERE {scope}{condition}
             ORDER BY {safe_sort_field} {safe_sort_dir}
             LIMIT {page_size} OFFSET {offset}",
            t = self.table_name,
            scope = self.list_scope_clause(biz_id)
        );
        let rows = self.db.fetch_all(&sql).await?;
        let results = rows
            .into_iter()
            .filter_map(|row| {
                let id = row.get("id").and_then(|v| v.as_str())?.to_string();
                Some((id, Self::map_row_to_doc_status(&row)))
            })
            .collect();
        Ok((results, total_count))
    }

    /// R6: Atomically claim a pending/failed document for single-document indexing.
    /// Conditional UPDATE returns affected row count: 1 = claimed, 0 = not claimable.
    async fn try_claim_for_processing(&self, doc_id: &str, biz_id: &str) -> easyrag_core::Result<bool> {
        let affected = self
            .db
            .execute_update(&format!(
                "UPDATE {t} SET status = 'processing', updated_at = NOW()
                 WHERE {scope} AND id = '{id}' AND status IN ('pending', 'failed')",
                t = self.table_name,
                scope = self.scope_clause(biz_id),
                id = sql_escape(doc_id)
            ))
            .await?;
        Ok(affected > 0)
    }

    async fn get_all_status_counts(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, usize>> {
        self.get_status_counts(biz_id).await
    }

    async fn get_doc_by_file_path(
        &self,
        file_path: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        let sql = format!(
            "SELECT content FROM {} WHERE {} AND file_path = '{}'",
            self.table_name,
            self.scope_clause(biz_id),
            sql_escape(file_path)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row.and_then(|r| read_content(&r)))
    }

    async fn get_doc_by_file_basename(
        &self,
        basename: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<(String, JsonMap)>> {
        let sql = format!(
            "SELECT id, content FROM {} WHERE {} AND file_path LIKE '%/{}' LIMIT 1",
            self.table_name,
            self.scope_clause(biz_id),
            sql_escape(basename)
        );
        let Some(row) = self.db.fetch_one(&sql).await? else {
            return Ok(None);
        };
        let Some(id) = row.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()) else {
            return Ok(None);
        };
        let content = read_content(&row).unwrap_or_default();
        Ok(Some((id, content)))
    }

    async fn get_doc_by_content_hash(
        &self,
        content_hash: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<(String, JsonMap)>> {
        let sql = format!(
            "SELECT id, content FROM {} WHERE {} AND content_hash = '{}' LIMIT 1",
            self.table_name,
            self.scope_clause(biz_id),
            sql_escape(content_hash)
        );
        let Some(row) = self.db.fetch_one(&sql).await? else {
            return Ok(None);
        };
        let Some(id) = row.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()) else {
            return Ok(None);
        };
        let content = read_content(&row).unwrap_or_default();
        Ok(Some((id, content)))
    }
}

// ---------------------------------------------------------------------------
// User storage
// ---------------------------------------------------------------------------

/// User credential storage backed by a global `users` table (no workspace
/// column: users are server-level and shared across workspaces).
///
/// Generic over `PgDb` so unit tests can run against mocks, mirroring
/// `PgDocStatusStorage`.
pub struct PgUserStorage<D: PgDb = PgDatabase> {
    db: D,
}

impl PgUserStorage<PgDatabase> {
    pub fn new(db: PgDatabase) -> Self {
        Self { db }
    }
}

impl<D: PgDb> PgUserStorage<D> {
    pub fn with_db(db: D) -> Self {
        Self { db }
    }

    fn row_to_record(row: &JsonMap) -> Option<UserRecord> {
        let username = row.get("username").and_then(|v| v.as_str())?.to_string();
        let password_hash = row
            .get("password_hash")
            .and_then(|v| v.as_str())?
            .to_string();
        let role = row
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("admin")
            .to_string();
        let parse_ts = |key: &str| {
            row.get(key)
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(chrono::Utc::now)
        };
        Some(UserRecord {
            username,
            password_hash,
            role,
            created_at: parse_ts("created_at"),
            updated_at: parse_ts("updated_at"),
        })
    }
}

#[async_trait]
impl<D: PgDb> UserStorage for PgUserStorage<D> {
    async fn initialize(&self) -> easyrag_core::Result<()> {
        self.db
            .execute(
                "CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(128) PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role VARCHAR(32) NOT NULL DEFAULT 'admin',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )",
            )
            .await
    }

    async fn find_by_username(&self, username: &str) -> easyrag_core::Result<Option<UserRecord>> {
        let sql = format!(
            "SELECT username, password_hash, role, created_at, updated_at
             FROM users WHERE username = '{}'",
            sql_escape(username)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row.as_ref().and_then(Self::row_to_record))
    }

    async fn upsert(&self, user: &UserRecord) -> easyrag_core::Result<()> {
        let sql = format!(
            "INSERT INTO users (username, password_hash, role)
             VALUES ('{username}', '{hash}', '{role}')
             ON CONFLICT (username) DO UPDATE SET
                 password_hash = EXCLUDED.password_hash,
                 role = EXCLUDED.role,
                 updated_at = NOW()",
            username = sql_escape(&user.username),
            hash = sql_escape(&user.password_hash),
            role = sql_escape(&user.role)
        );
        self.db.execute(&sql).await
    }

    async fn count(&self) -> easyrag_core::Result<u64> {
        let row = self.db.fetch_one("SELECT COUNT(*) as cnt FROM users").await?;
        Ok(row
            .and_then(|r| r.get("cnt").and_then(|v| v.as_u64()))
            .unwrap_or(0))
    }
}

// ---------------------------------------------------------------------------
// PgTenantConfigStorage
// ---------------------------------------------------------------------------

/// PostgreSQL-backed tenant configuration profile storage.
/// The `tenant_config` table is global (not workspace-suffixed): the
/// `workspace` column is the primary key.
pub struct PgTenantConfigStorage<D: PgDb = PgDatabase> {
    db: D,
}

impl PgTenantConfigStorage<PgDatabase> {
    pub fn new(db: PgDatabase) -> Self {
        Self { db }
    }
}

impl<D: PgDb> PgTenantConfigStorage<D> {
    pub fn with_db(db: D) -> Self {
        Self { db }
    }

    fn row_to_config(row: &JsonMap) -> Option<TenantConfig> {
        let workspace = row.get("workspace").and_then(|v| v.as_str())?.to_string();
        let parse_ts = |key: &str| {
            row.get(key)
                .and_then(|v| v.as_str())
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .unwrap_or_else(chrono::Utc::now)
        };
        Some(TenantConfig {
            workspace,
            llm_model: row.get("llm_model").and_then(|v| v.as_str()).map(String::from),
            llm_api_key: row.get("llm_api_key").and_then(|v| v.as_str()).map(String::from),
            llm_base_url: row.get("llm_base_url").and_then(|v| v.as_str()).map(String::from),
            llm_temperature: row.get("llm_temperature").and_then(|v| v.as_f64()).map(|v| v as f32),
            llm_max_tokens: row.get("llm_max_tokens").and_then(|v| v.as_u64()).map(|v| v as u32),
            embedding_model: row.get("embedding_model").and_then(|v| v.as_str()).map(String::from),
            embedding_api_key: row.get("embedding_api_key").and_then(|v| v.as_str()).map(String::from),
            embedding_base_url: row.get("embedding_base_url").and_then(|v| v.as_str()).map(String::from),
            embedding_dim: row.get("embedding_dim").and_then(|v| v.as_u64()).map(|v| v as usize),
            chunk_size: row.get("chunk_size").and_then(|v| v.as_u64()).map(|v| v as usize),
            chunk_overlap_size: row.get("chunk_overlap_size").and_then(|v| v.as_u64()).map(|v| v as usize),
            language: row.get("language").and_then(|v| v.as_str()).map(String::from),
            default_top_k: row.get("default_top_k").and_then(|v| v.as_u64()).map(|v| v as usize),
            rerank_enabled: row.get("rerank_enabled").and_then(|v| v.as_bool()),
            created_at: parse_ts("created_at"),
            updated_at: parse_ts("updated_at"),
        })
    }

    fn nullable_str_sql(value: &Option<String>) -> String {
        match value {
            Some(v) => format!("'{}'", sql_escape(v)),
            None => "NULL".to_string(),
        }
    }

    fn nullable_f32_sql(value: &Option<f32>) -> String {
        match value {
            Some(v) if v.is_finite() => v.to_string(),
            Some(_) => "NULL".to_string(),
            None => "NULL".to_string(),
        }
    }

    fn nullable_u64_sql(value: &Option<u64>) -> String {
        match value {
            Some(v) => v.to_string(),
            None => "NULL".to_string(),
        }
    }

    fn nullable_bool_sql(value: &Option<bool>) -> String {
        match value {
            Some(v) => v.to_string(),
            None => "NULL".to_string(),
        }
    }
}

#[async_trait]
impl<D: PgDb> TenantConfigStorage for PgTenantConfigStorage<D> {
    async fn initialize(&self) -> easyrag_core::Result<()> {
        self.db
            .execute(
                "CREATE TABLE IF NOT EXISTS tenant_config (
                    workspace TEXT PRIMARY KEY,
                    llm_model TEXT,
                    llm_api_key TEXT,
                    llm_base_url TEXT,
                    llm_temperature REAL,
                    llm_max_tokens INTEGER,
                    embedding_model TEXT,
                    embedding_api_key TEXT,
                    embedding_base_url TEXT,
                    embedding_dim INTEGER,
                    chunk_size INTEGER,
                    chunk_overlap_size INTEGER,
                    language TEXT,
                    default_top_k INTEGER,
                    rerank_enabled BOOLEAN,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )",
            )
            .await
    }

    async fn get_by_workspace(
        &self,
        workspace: &str,
    ) -> easyrag_core::Result<Option<TenantConfig>> {
        let sql = format!(
            "SELECT workspace, llm_model, llm_api_key, llm_base_url, llm_temperature,
                    llm_max_tokens, embedding_model, embedding_api_key, embedding_base_url,
                    embedding_dim, chunk_size, chunk_overlap_size, language, default_top_k,
                    rerank_enabled, created_at, updated_at
             FROM tenant_config WHERE workspace = '{}'",
            sql_escape(workspace)
        );
        let row = self.db.fetch_one(&sql).await?;
        Ok(row.as_ref().and_then(Self::row_to_config))
    }

    async fn upsert(&self, config: &TenantConfig) -> easyrag_core::Result<()> {
        let sql = format!(
            "INSERT INTO tenant_config (
                workspace, llm_model, llm_api_key, llm_base_url, llm_temperature,
                llm_max_tokens, embedding_model, embedding_api_key, embedding_base_url,
                embedding_dim, chunk_size, chunk_overlap_size, language, default_top_k,
                rerank_enabled, created_at, updated_at
            ) VALUES (
                {workspace}, {llm_model}, {llm_api_key}, {llm_base_url}, {llm_temperature},
                {llm_max_tokens}, {embedding_model}, {embedding_api_key}, {embedding_base_url},
                {embedding_dim}, {chunk_size}, {chunk_overlap_size}, {language}, {default_top_k},
                {rerank_enabled}, {created_at}, {updated_at}
            ) ON CONFLICT (workspace) DO UPDATE SET
                llm_model = EXCLUDED.llm_model,
                llm_api_key = EXCLUDED.llm_api_key,
                llm_base_url = EXCLUDED.llm_base_url,
                llm_temperature = EXCLUDED.llm_temperature,
                llm_max_tokens = EXCLUDED.llm_max_tokens,
                embedding_model = EXCLUDED.embedding_model,
                embedding_api_key = EXCLUDED.embedding_api_key,
                embedding_base_url = EXCLUDED.embedding_base_url,
                embedding_dim = EXCLUDED.embedding_dim,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap_size = EXCLUDED.chunk_overlap_size,
                language = EXCLUDED.language,
                default_top_k = EXCLUDED.default_top_k,
                rerank_enabled = EXCLUDED.rerank_enabled,
                updated_at = NOW()",
            workspace = Self::nullable_str_sql(&Some(config.workspace.clone())),
            llm_model = Self::nullable_str_sql(&config.llm_model),
            llm_api_key = Self::nullable_str_sql(&config.llm_api_key),
            llm_base_url = Self::nullable_str_sql(&config.llm_base_url),
            llm_temperature = Self::nullable_f32_sql(&config.llm_temperature),
            llm_max_tokens = Self::nullable_u64_sql(&config.llm_max_tokens.map(|v| v as u64)),
            embedding_model = Self::nullable_str_sql(&config.embedding_model),
            embedding_api_key = Self::nullable_str_sql(&config.embedding_api_key),
            embedding_base_url = Self::nullable_str_sql(&config.embedding_base_url),
            embedding_dim = Self::nullable_u64_sql(&config.embedding_dim.map(|v| v as u64)),
            chunk_size = Self::nullable_u64_sql(&config.chunk_size.map(|v| v as u64)),
            chunk_overlap_size = Self::nullable_u64_sql(&config.chunk_overlap_size.map(|v| v as u64)),
            language = Self::nullable_str_sql(&config.language),
            default_top_k = Self::nullable_u64_sql(&config.default_top_k.map(|v| v as u64)),
            rerank_enabled = Self::nullable_bool_sql(&config.rerank_enabled),
            created_at = format!("'{}'", config.created_at.to_rfc3339()),
            updated_at = format!("'{}'", config.updated_at.to_rfc3339()),
        );
        self.db.execute(&sql).await
    }

    async fn delete(&self, workspace: &str) -> easyrag_core::Result<()> {
        let sql = format!(
            "DELETE FROM tenant_config WHERE workspace = '{}'",
            sql_escape(workspace)
        );
        self.db.execute(&sql).await
    }

    async fn list_all(&self) -> easyrag_core::Result<Vec<TenantConfig>> {
        let rows = self
            .db
            .fetch_all(
                "SELECT workspace, llm_model, llm_api_key, llm_base_url, llm_temperature,
                        llm_max_tokens, embedding_model, embedding_api_key, embedding_base_url,
                        embedding_dim, chunk_size, chunk_overlap_size, language, default_top_k,
                        rerank_enabled, created_at, updated_at
                 FROM tenant_config ORDER BY workspace",
            )
            .await?;
        Ok(rows.iter().filter_map(Self::row_to_config).collect())
    }
}

// ---------------------------------------------------------------------------
// Phase 3.10 guardrails. The retry LOOP moved to `core::with_retry` and the
// retry CLASSIFICATION moved to `CoreError::is_retryable`; these pin both halves
// of the [equivalent] claim.
//
// `start_paused = true` makes the 1s/2s sleeps virtual, so each test asserts the
// exact backoff the old hand-rolled loop produced without waiting for it. Elapsed
// virtual time is the sum of the delays, which pins the curve: base 1000ms with
// factor 2 gives 1000 + 2000 = 3000, and any other curve lands elsewhere.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod retry_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Attempt counter shared with the closure under test.
    fn counter() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        let c = Arc::new(AtomicUsize::new(0));
        (c.clone(), c)
    }

    fn transient() -> easyrag_core::CoreError {
        easyrag_core::CoreError::Internal("error communicating with the server".to_string())
    }

    #[test]
    fn policy_is_the_curve_the_hand_rolled_loop_used() {
        assert_eq!(3, PG_RETRY_CONFIG.max_attempts, "3 attempts, as before");
        assert_eq!(Duration::from_secs(1), PG_RETRY_CONFIG.base_delay, "base 1s");
        assert_eq!(Duration::from_secs(10), PG_RETRY_CONFIG.max_delay, "capped at 10s");
        assert_eq!(2.0, PG_RETRY_CONFIG.exponential_base, "factor 2");
        // Jitter must stay off: the fixed curve is what the runbook documents,
        // and a random factor would make the delay assertions below flaky.
        assert!(!PG_RETRY_CONFIG.jitter);
    }

    #[tokio::test(start_paused = true)]
    async fn a_transient_failure_exhausts_three_attempts_sleeping_1s_then_2s() {
        let (bump, attempts) = counter();
        let start = tokio::time::Instant::now();

        let result: easyrag_core::Result<()> = retry(move || {
            bump.fetch_add(1, Ordering::SeqCst);
            async { Err(transient()) }
        })
        .await;

        assert!(result.is_err(), "the transient error must still surface");
        assert_eq!(3, attempts.load(Ordering::SeqCst));
        assert_eq!(
            Duration::from_millis(3000),
            start.elapsed(),
            "backoff must stay 1000ms then 2000ms, and never sleep after the last attempt"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_fatal_failure_is_returned_immediately_without_sleeping() {
        let (bump, attempts) = counter();
        let start = tokio::time::Instant::now();

        let result: easyrag_core::Result<()> = retry(move || {
            bump.fetch_add(1, Ordering::SeqCst);
            async {
                Err(easyrag_core::CoreError::Internal(
                    "duplicate key value violates unique constraint \"pk_docs\"".to_string(),
                ))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(1, attempts.load(Ordering::SeqCst), "no retry for a fatal error");
        assert_eq!(Duration::ZERO, start.elapsed(), "no sleep for a fatal error");
    }

    #[tokio::test(start_paused = true)]
    async fn recovery_on_the_second_attempt_stops_retrying() {
        let (bump, attempts) = counter();
        let start = tokio::time::Instant::now();

        let result = retry(move || {
            let seen = bump.fetch_add(1, Ordering::SeqCst);
            async move {
                if seen == 0 {
                    Err(easyrag_core::CoreError::Internal(
                        "connection closed".to_string(),
                    ))
                } else {
                    Ok(7u32)
                }
            }
        })
        .await;

        // `CoreError` is not `PartialEq`, so unwrap the success value instead of
        // comparing the whole `Result`.
        assert_eq!(7u32, result.expect("recovery on the second attempt must succeed"));
        assert_eq!(2, attempts.load(Ordering::SeqCst));
        assert_eq!(
            Duration::from_millis(1000),
            start.elapsed(),
            "exactly one 1000ms sleep before the successful attempt"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn api_timeout_is_still_retried() {
        // The old loop accepted `CoreError::ApiTimeout` alongside the six
        // `Internal` messages; that arm now lives in `CoreError::is_retryable`.
        let (bump, attempts) = counter();

        let result: easyrag_core::Result<()> = retry(move || {
            bump.fetch_add(1, Ordering::SeqCst);
            async { Err(easyrag_core::CoreError::ApiTimeout) }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(3, attempts.load(Ordering::SeqCst));
    }

    #[tokio::test(start_paused = true)]
    async fn bc3_a_deadlock_is_now_retried_where_it_used_to_fail_fast() {
        // [BC-3 widening] Before Phase 3.10 this message fell through to the
        // `_ => false` arm of the old inline `matches!`, so a deadlock aborted
        // the operation on attempt 1. It now consumes the full budget.
        let (bump, attempts) = counter();
        let start = tokio::time::Instant::now();

        let result: easyrag_core::Result<()> = retry(move || {
            bump.fetch_add(1, Ordering::SeqCst);
            async {
                Err(easyrag_core::CoreError::Internal(
                    "error returned from database: deadlock detected (SQLSTATE 40P01)".to_string(),
                ))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(
            3,
            attempts.load(Ordering::SeqCst),
            "BC-3: a deadlock must now be retried"
        );
        assert_eq!(Duration::from_millis(3000), start.elapsed());
    }
}
