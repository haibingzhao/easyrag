//! PostgreSQL storage tests.
//!
//! `PgDb` facade (replaces Mockito), plus workspace validation tests.
//! Integration tests requiring a live PostgreSQL live in
//! `postgres_integration.rs` behind `#[ignore]`.

#![cfg(feature = "postgres")]

use async_trait::async_trait;
use easyrag_core::llm::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::model::DocStatus;
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageNameSpace, VectorStorage,
};
use easyrag_core::JsonMap;
use easyrag_storage::postgres::{
    PgBatchLimits, PgConfig, PgDb, PgDocStatusStorage, PgGraphStorage, PgKVStorage, PgVectorStorage,
};
use parking_lot::Mutex;
use serde_json::json;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Mock PgDb (replaces Mockito mock of PgDatabase)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MockPgDb {
    fetch_all_rows: Vec<JsonMap>,
    fetch_all_calls: AtomicUsize,
    last_sql: Mutex<String>,
    // Phase 0.4 guardrail: full ordered statement log + total round-trip count
    // (including writes). `last_sql` alone cannot prove a refactor kept the same
    // number of DB round-trips, so the SQL golden tests assert BOTH exact text and
    // statement count (locks Phase 3.7 batched_with_fallback from adding trips).
    all_sql: Mutex<Vec<String>>,
    total_calls: AtomicUsize,
}

impl MockPgDb {
    fn with_rows(rows: Vec<JsonMap>) -> Self {
        Self {
            fetch_all_rows: rows,
            fetch_all_calls: AtomicUsize::new(0),
            last_sql: Mutex::new(String::new()),
            all_sql: Mutex::new(Vec::new()),
            total_calls: AtomicUsize::new(0),
        }
    }

    /// Record one DB round-trip: refresh last_sql, append to the ordered log,
    /// and bump the total call counter.
    fn record(&self, sql: &str) {
        *self.last_sql.lock() = sql.to_string();
        self.all_sql.lock().push(sql.to_string());
        self.total_calls.fetch_add(1, Ordering::SeqCst);
    }

    /// Ordered copy of every SQL statement issued so far.
    fn statements(&self) -> Vec<String> {
        self.all_sql.lock().clone()
    }

    /// Total DB round-trips (fetch_one + fetch_all + execute + execute_update).
    fn total_calls(&self) -> usize {
        self.total_calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl PgDb for MockPgDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        self.record(sql);
        Ok(())
    }

    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.record(sql);
        self.fetch_all_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.fetch_all_rows.first().cloned())
    }

    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.record(sql);
        self.fetch_all_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.fetch_all_rows.clone())
    }

    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        self.record(sql);
        Ok(0)
    }
}

/// Shared-ownership wrapper so tests can inspect the mock after moving it
/// into the storage.
struct ArcDb(Arc<MockPgDb>);

#[async_trait]
impl PgDb for ArcDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        self.0.execute(sql).await
    }

    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.0.fetch_one(sql).await
    }

    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.0.fetch_all(sql).await
    }

    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        self.0.execute_update(sql).await
    }
}

fn json_map(value: serde_json::Value) -> JsonMap {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

/// Test embedding function: always zero vectors.
struct ZeroEmbeddingFunc;

#[async_trait]
impl EmbeddingFunction for ZeroEmbeddingFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0f32; 4]).collect())
    }
}

fn create_storage(db: Arc<MockPgDb>) -> PgVectorStorage<ArcDb> {
    let config = PgConfig {
        host: "localhost".to_string(),
        database: "test".to_string(),
        embedding_dim: 4,
        fts_config: "simple".to_string(),
        ..Default::default()
    };
    let embedding_func = EmbeddingFunc::new(4, Arc::new(ZeroEmbeddingFunc));
    PgVectorStorage::with_options(
        "vectors",
        "test_ws",
        JsonMap::new(),
        embedding_func,
        ArcDb(db),
        config,
    )
    .unwrap()
}

// --- Case 1 ---
#[tokio::test]
async fn keyword_query_blank_query_returns_empty() {
    let storage = create_storage(Arc::new(MockPgDb::default()));

    let results = storage.keyword_query("", 10, None, None, BIZ).await.unwrap();
    assert!(results.is_empty());

    let results2 = storage.keyword_query("   ", 10, None, None, BIZ).await.unwrap();
    assert!(results2.is_empty());
}

// --- Case 2 ---
#[tokio::test]
async fn keyword_query_calls_db_with_ts_query() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());

    storage
        .keyword_query("hello world", 10, None, None, BIZ)
        .await
        .unwrap();

    // Verify fetchAll was called (the SQL contains ts_rank_cd and @@).
    assert!(
        db.fetch_all_calls.load(Ordering::SeqCst) >= 1,
        "fetchAll should be called"
    );
    let sql = db.last_sql.lock().clone();
    assert!(sql.contains("ts_rank_cd"), "SQL should use ts_rank_cd: {sql}");
    assert!(sql.contains("@@"), "SQL should use @@ match: {sql}");
    assert!(
        sql.contains("hello:*") && sql.contains("world:*"),
        "tsquery should prefix-match each term: {sql}"
    );
}

// --- Case 3 ---
#[tokio::test]
async fn keyword_query_parses_results() {
    let mock_row = json_map(json!({
        "id": "chunk-123",
        "content": {"content": "test document", "file_path": "test.txt"},
        "rank": 0.85
    }));
    let storage = create_storage(Arc::new(MockPgDb::with_rows(vec![mock_row])));

    let results = storage.keyword_query("test", 10, None, None, BIZ).await.unwrap();

    assert_eq!(1, results.len());
    assert_eq!("chunk-123", results[0]["id"].as_str().unwrap());
    assert_eq!("test document", results[0]["content"].as_str().unwrap());
    assert!((results[0]["score"].as_f64().unwrap() - 0.85).abs() < 1e-6);
}

// --- Case 4 ---
#[tokio::test]
async fn index_documents_is_no_op() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());

    // Should not throw; no DB interaction for indexDocuments (trigger handles it).
    storage
        .index_documents(std::collections::HashMap::from([(
            "id1".to_string(),
            json_map(json!({"content": "text"})),
        )]), BIZ)
        .await
        .unwrap();
    assert_eq!(0, db.fetch_all_calls.load(Ordering::SeqCst));
}

// --- Case 5 ---
#[tokio::test]
async fn remove_documents_is_no_op() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());

    storage
        .remove_documents(&["id1".to_string(), "id2".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(0, db.fetch_all_calls.load(Ordering::SeqCst));
}

// --- Case 6 ---
#[test]
fn pg_config_fts_config_default() {
    let config = PgConfig::default();
    assert_eq!("simple", config.fts_config);
}

// --- Case 7 ---
#[test]
fn pg_config_fts_config_custom() {
    let config = PgConfig {
        fts_config: "english".to_string(),
        ..Default::default()
    };
    assert_eq!("english", config.fts_config);
}

// --- Case 8 ---
#[tokio::test]
async fn keyword_query_multiple_results_ordered_by_rank() {
    let rows = vec![
        json_map(json!({"id": "doc1", "content": {"content": "best match"}, "rank": 0.9})),
        json_map(json!({"id": "doc2", "content": {"content": "second match"}, "rank": 0.5})),
    ];
    let storage = create_storage(Arc::new(MockPgDb::with_rows(rows)));

    let results = storage.keyword_query("match", 10, None, None, BIZ).await.unwrap();
    assert_eq!(2, results.len());
    assert_eq!("doc1", results[0]["id"].as_str().unwrap());
    assert!((results[0]["score"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!("doc2", results[1]["id"].as_str().unwrap());
    assert!((results[1]["score"].as_f64().unwrap() - 0.5).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Workspace validation
// ---------------------------------------------------------------------------

#[test]
fn validate_workspace_accepts_valid_names() {
    use easyrag_storage::postgres::validate_workspace;
    assert!(validate_workspace("default").is_ok());
    assert!(validate_workspace("my-workspace_1.x").is_ok());
}

#[test]
fn validate_workspace_rejects_invalid_names() {
    use easyrag_storage::postgres::validate_workspace;
    assert!(validate_workspace("").is_err());
    assert!(validate_workspace(".leading-dot").is_err());
    assert!(validate_workspace("has space").is_err());
    assert!(validate_workspace("sql'; DROP TABLE--").is_err());
    let long_name = "a".repeat(65);
    assert!(validate_workspace(&long_name).is_err());
}

// ---------------------------------------------------------------------------
// Phase 0.4 — SQL text golden + statement-count guardrails.
//
// SQL here is hand-built by string concatenation, so a single misplaced quote
// silently returns an empty set instead of erroring. These snapshots pin the
// EXACT SQL bytes AND the number of round-trips each operation issues, which is
// the only guardrail for the Phase 3 storage refactor (Scope value object 3.4,
// batched_with_fallback 3.7). Round-trip counts must not INCREASE.
//
// Snapshots land in `<workspace>/testdata/golden/sql/`. Regenerate deliberately:
// `INSTA_UPDATE=always cargo test -p easyrag-storage --features postgres --test postgres_storage`.
// ---------------------------------------------------------------------------

fn sql_golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-storage");
    root.join("testdata").join("golden").join("sql")
}

fn snap_sql(name: &str, sql: &str) {
    insta::with_settings!({
        snapshot_path => sql_golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, sql);
    });
}

fn create_doc_status_storage(db: Arc<MockPgDb>) -> PgDocStatusStorage<ArcDb> {
    PgDocStatusStorage::with_options("doc_status", "test_ws", JsonMap::new(), None, ArcDb(db))
        .unwrap()
}

fn create_graph_storage(db: Arc<MockPgDb>) -> PgGraphStorage<ArcDb> {
    PgGraphStorage::with_options("graph", "test_ws", JsonMap::new(), None, ArcDb(db)).unwrap()
}

#[tokio::test]
async fn sql_golden_doc_status_scope_filter() {
    // Exact scope: `workspace = '..' AND biz_id = '..'`.
    let db = Arc::new(MockPgDb::default());
    let storage = create_doc_status_storage(db.clone());
    storage
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "get_docs_by_status issues exactly 1 query");
    snap_sql("doc_status_scope_exact", &stmts[0]);

    // Empty biz_id = read-path wildcard (list_scope_clause): workspace only.
    let db2 = Arc::new(MockPgDb::default());
    let storage2 = create_doc_status_storage(db2.clone());
    storage2
        .get_docs_by_status(DocStatus::Processed, "")
        .await
        .unwrap();
    let stmts2 = db2.statements();
    assert_eq!(stmts2.len(), 1);
    snap_sql("doc_status_scope_wildcard", &stmts2[0]);
}

#[tokio::test]
async fn sql_golden_doc_status_upsert() {
    // Single record: exact INSERT ... ON CONFLICT (merge semantics) text.
    let db = Arc::new(MockPgDb::default());
    let storage = create_doc_status_storage(db.clone());
    let mut data = HashMap::new();
    data.insert(
        "doc1".to_string(),
        json_map(json!({
            "status": "processed",
            "file_path": "/data/doc1.txt",
            "content_hash": "h1",
            "track_id": "t1"
        })),
    );
    storage.upsert(data, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "single-record upsert = 1 statement");
    snap_sql("doc_status_upsert_single", &stmts[0]);

    // Per-record failure granularity: N records issue N statements (no batching).
    // Phase 3.7 must preserve this — postgres.rs deliberately keeps per-record
    // writes so one bad row cannot fail the whole batch.
    let db3 = Arc::new(MockPgDb::default());
    let storage3 = create_doc_status_storage(db3.clone());
    let mut data3 = HashMap::new();
    for i in 0..3 {
        data3.insert(
            format!("doc{i}"),
            json_map(json!({"status": "pending"})),
        );
    }
    storage3.upsert(data3, BIZ).await.unwrap();
    assert_eq!(
        db3.total_calls(),
        3,
        "doc_status upsert is per-record (3 records = 3 round-trips)"
    );
}

#[tokio::test]
async fn sql_golden_doc_status_paginated() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_doc_status_storage(db.clone());
    storage
        .get_docs_paginated(None, None, 2, 20, "updated_at", "desc", Some("/data/"), false, BIZ)
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 2, "paginated = 1 COUNT + 1 page query");
    snap_sql("doc_status_paginated_count", &stmts[0]);
    snap_sql("doc_status_paginated_page", &stmts[1]);
}

#[tokio::test]
async fn sql_golden_vector_upsert_batched() {
    // Single record: exact batched INSERT ... ON CONFLICT text (deterministic).
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());
    let mut data = HashMap::new();
    data.insert(
        "chunk1".to_string(),
        json_map(json!({"content": "hello world"})),
    );
    storage.upsert(data, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1);
    snap_sql("vector_upsert_single", &stmts[0]);

    // Batching: 3 records (within upsert_batch_size = 64) collapse into ONE
    // multi-row INSERT. Locks the round-trip count so Phase 3.7
    // batched_with_fallback cannot silently regress to per-row inserts.
    let db3 = Arc::new(MockPgDb::default());
    let storage3 = create_storage(db3.clone());
    let mut data3 = HashMap::new();
    for i in 0..3 {
        data3.insert(format!("chunk{i}"), json_map(json!({"content": "text"})));
    }
    storage3.upsert(data3, BIZ).await.unwrap();
    assert_eq!(
        db3.total_calls(),
        1,
        "3 records within batch size = 1 round-trip (no per-row regression)"
    );
}

#[tokio::test]
async fn sql_golden_graph_knowledge_star() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    // "*" wildcard => get_all_nodes + get_all_edges (2 queries).
    storage
        .get_knowledge_graph(Some("*"), 1, 100, BIZ)
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 2, "star path = all_nodes + all_edges");
    snap_sql("graph_knowledge_star_nodes", &stmts[0]);
    snap_sql("graph_knowledge_star_edges", &stmts[1]);
}

#[tokio::test]
async fn sql_golden_graph_knowledge_cte() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    // Labeled => recursive CTE traversal (statement 0), then a get_node per
    // discovered id. Empty rows => node_ids = [label], so 1 extra get_node fetch.
    storage
        .get_knowledge_graph(Some("Alice"), 2, 50, BIZ)
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 2, "CTE traversal + 1 get_node for the seed label");
    snap_sql("graph_knowledge_cte", &stmts[0]);
    snap_sql("graph_knowledge_get_node", &stmts[1]);
}

// ---------------------------------------------------------------------------
// Phase 3.6 — [BC-12] (A-5): the empty-biz_id read-path wildcard on the
// vector and graph LIST paths.
//
// Before this change only `PgDocStatusStorage` treated an empty biz_id as
// "every biz line of the workspace" (`list_scope_clause`); vector search and
// graph listing pinned `biz_id = ''` and so returned nothing. A caller that
// listed documents workspace-wide therefore could not retrieve their chunks.
// Neo4j has always widened exactly these paths (`node_scope`), which is the
// behaviour aligned to here — see `scope_sql::TenantScopeSql::list_sql`.
//
// Single-record reads and every write keep the EXACT scope: widening those
// would let a status write or a delete land on the wrong biz line.
// ---------------------------------------------------------------------------

/// Assert every statement a list path issued widened to the whole workspace.
fn assert_wildcard_scope(db: &MockPgDb, path: &str) {
    let stmts = db.statements();
    assert!(!stmts.is_empty(), "{path} issued no query");
    for sql in &stmts {
        assert!(
            sql.contains("workspace = 'test_ws'"),
            "{path} must stay workspace-scoped: {sql}"
        );
        assert!(
            !sql.contains("biz_id = ''"),
            "{path} must treat an empty biz_id as the wildcard, not a literal: {sql}"
        );
    }
}

/// Assert a single-record read or a write still pins the exact biz line.
fn assert_exact_scope(db: &MockPgDb, path: &str) {
    let stmts = db.statements();
    assert!(!stmts.is_empty(), "{path} issued no query");
    for sql in &stmts {
        assert!(
            sql.contains("workspace = 'test_ws' AND biz_id = ''"),
            "{path} must keep the exact scope (writes must never widen): {sql}"
        );
    }
}

#[tokio::test]
async fn bc12_vector_list_paths_widen_on_empty_biz_id() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());
    storage.query("q", 5, None, None, None, false, "").await.unwrap();
    assert_wildcard_scope(&db, "vector query");

    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());
    storage
        .keyword_query("hello", 10, None, None, "")
        .await
        .unwrap();
    assert_wildcard_scope(&db, "vector keyword_query");
}

#[tokio::test]
async fn bc12_vector_single_record_reads_stay_exact() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_storage(db.clone());
    storage
        .get_vectors_by_ids(&["v1".to_string()], "")
        .await
        .unwrap();
    assert_exact_scope(&db, "vector get_vectors_by_ids");
}

#[tokio::test]
async fn bc12_graph_list_paths_widen_on_empty_biz_id() {
    for (path, call) in [
        ("get_all_nodes", 0),
        ("get_all_edges", 1),
        ("search_labels", 2),
        ("get_knowledge_graph(*)", 3),
        ("get_knowledge_graph(Alice)", 4),
        ("get_popular_labels", 5),
        // Both are list paths: they widen exactly like `get_all_nodes`, whose
        // full-graph pull they replace during document deletion.
        ("get_nodes_by_source_ids", 6),
        ("get_edges_by_source_ids", 7),
    ] {
        let db = Arc::new(MockPgDb::default());
        let storage = create_graph_storage(db.clone());
        match call {
            0 => {
                storage.get_all_nodes("").await.unwrap();
            }
            1 => {
                storage.get_all_edges("").await.unwrap();
            }
            2 => {
                storage.search_labels("Al", 10, "").await.unwrap();
            }
            3 => {
                storage
                    .get_knowledge_graph(Some("*"), 1, 100, "")
                    .await
                    .unwrap();
            }
            4 => {
                storage
                    .get_knowledge_graph(Some("Alice"), 2, 50, "")
                    .await
                    .unwrap();
            }
            // The `/api/graph/labels` route reaches `get_popular_labels` through
            // `read_bundle` (blank biz_id = wildcard), so it must widen like the
            // other list paths and match Neo4j's `node_scope`.
            5 => {
                storage.get_popular_labels(10, "").await.unwrap();
            }
            6 => {
                storage
                    .get_nodes_by_source_ids(&["chunk-1".to_string()], "")
                    .await
                    .unwrap();
            }
            _ => {
                storage
                    .get_edges_by_source_ids(&["chunk-1".to_string()], "")
                    .await
                    .unwrap();
            }
        };
        assert_wildcard_scope(&db, path);
    }
}

#[tokio::test]
async fn bc12_graph_single_record_reads_and_writes_stay_exact() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_all_labels("").await.unwrap();
    assert_exact_scope(&db, "graph get_all_labels");

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_node("n1", "").await.unwrap();
    assert_exact_scope(&db, "graph get_node");

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.has_node("n1", "").await.unwrap();
    assert_exact_scope(&db, "graph has_node");

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.delete_node("n1", "").await.unwrap();
    assert_exact_scope(&db, "graph delete_node");
}

// ---------------------------------------------------------------------------
// Phase 3.7 — [等价] guardrails for `batched_with_fallback` + `PgUpsertBuilder`.
//
// The batch-then-replay-row-by-row loop existed twice (KV and vector) and
// NOTHING pinned either the KV statement bytes or the replay branch. These
// tests are the guardrail the 3.7 refactor must not move:
//
//   * KV/vector multi-row statement bytes (golden) and graph node/edge bytes,
//   * one round-trip per batch (no regression to per-row inserts),
//   * a failed batch replays record-by-record and still returns Ok,
//   * a record that fails during the replay aborts the upsert (the `?` stays),
//   * the replay must NOT re-embed: exactly one `embed(texts)` call per batch.
// ---------------------------------------------------------------------------

/// A `PgDb` that records into a [`MockPgDb`] and fails the statements matching
/// `fail_if`, which is the only way to reach the row-by-row fallback branch.
struct FailingPgDb {
    inner: Arc<MockPgDb>,
    fail_if: Box<dyn Fn(&str) -> bool + Send + Sync>,
    failures: Arc<AtomicUsize>,
}

impl FailingPgDb {
    fn new(
        inner: Arc<MockPgDb>,
        fail_if: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner,
            fail_if: Box::new(fail_if),
            failures: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Fail every multi-row INSERT (the batched statement), keep single rows Ok.
    fn failing_batches(inner: Arc<MockPgDb>) -> Self {
        Self::new(inner, |sql| sql.contains("), ("))
    }

    fn injected(&self, sql: &str) -> easyrag_core::Result<()> {
        self.inner.record(sql);
        if (self.fail_if)(sql) {
            self.failures.fetch_add(1, Ordering::SeqCst);
            return Err(easyrag_core::CoreError::Internal(format!(
                "injected failure: {sql}"
            )));
        }
        Ok(())
    }
}

#[async_trait]
impl PgDb for FailingPgDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        self.injected(sql)
    }

    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.injected(sql)?;
        Ok(None)
    }

    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.injected(sql)?;
        Ok(Vec::new())
    }

    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        self.injected(sql)?;
        Ok(0)
    }
}

/// Embedding stub that counts invocations and the text count of each one.
struct CountingEmbeddingFunc {
    calls: Arc<AtomicUsize>,
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl CountingEmbeddingFunc {
    fn new() -> Self {
        Self {
            calls: Arc::new(AtomicUsize::new(0)),
            batch_sizes: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl EmbeddingFunction for CountingEmbeddingFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.batch_sizes.lock().push(texts.len());
        Ok(texts.iter().map(|_| vec![0.0f32; 4]).collect())
    }
}

fn create_kv_storage<D: PgDb>(db: D) -> PgKVStorage<D> {
    PgKVStorage::with_options("cache", "test_ws", JsonMap::new(), None, db).unwrap()
}

fn create_vector_storage<D: PgDb>(db: D, embedder: CountingEmbeddingFunc) -> PgVectorStorage<D> {
    let config = PgConfig {
        host: "localhost".to_string(),
        database: "test".to_string(),
        embedding_dim: 4,
        fts_config: "simple".to_string(),
        ..Default::default()
    };
    PgVectorStorage::with_options(
        "vectors",
        "test_ws",
        JsonMap::new(),
        EmbeddingFunc::new(4, Arc::new(embedder)),
        db,
        config,
    )
    .unwrap()
}

fn three_records(prefix: &str) -> HashMap<String, JsonMap> {
    let mut data = HashMap::new();
    for i in 0..3 {
        data.insert(
            format!("{prefix}{i}"),
            json_map(json!({"content": "text"})),
        );
    }
    data
}

#[tokio::test]
async fn sql_golden_kv_upsert_batched() {
    // Single record: exact KV `INSERT ... ON CONFLICT` bytes.
    let db = Arc::new(MockPgDb::default());
    let storage = create_kv_storage(ArcDb(db.clone()));
    let mut data = HashMap::new();
    data.insert("k1".to_string(), json_map(json!({"content": "v1"})));
    storage.upsert(data, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "single-record KV upsert = 1 statement");
    snap_sql("kv_upsert_single", &stmts[0]);

    // 3 records inside the batch limit collapse into ONE multi-row INSERT.
    let db3 = Arc::new(MockPgDb::default());
    let storage3 = create_kv_storage(ArcDb(db3.clone()));
    storage3
        .upsert(three_records("k"), BIZ)
        .await
        .unwrap();
    assert_eq!(
        db3.total_calls(),
        1,
        "3 records within batch size = 1 round-trip (no per-row regression)"
    );
}

#[tokio::test]
async fn sql_golden_graph_upsert_node_and_edge() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage
        .upsert_node("Alice", json_map(json!({"entity_type": "person"})), BIZ)
        .await
        .unwrap();
    storage
        .upsert_edge("Alice", "Bob", json_map(json!({"weight": 1.0})), BIZ)
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 2, "node + edge = 2 statements");
    snap_sql("graph_upsert_node", &stmts[0]);
    snap_sql("graph_upsert_edge", &stmts[1]);
}

/// The `source_id` filters are what let document deletion stop pulling the whole
/// graph. The SQL is hand-concatenated, so a misplaced quote returns an empty set
/// instead of erroring and every orphaned node leaks silently — both the exact
/// bytes and the round-trip count are pinned.
///
/// Each query carries two arms: jsonb `?|` for the canonical array and
/// `string_to_array(...) && ...` for the legacy joined string, because `?|` is
/// false for a jsonb string scalar and pre-backfill rows would otherwise be
/// invisible to cleanup.
#[tokio::test]
async fn sql_golden_graph_source_id_filters() {
    let chunks = vec!["chunk-1".to_string(), "chunk-2".to_string()];

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_nodes_by_source_ids(&chunks, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "one targeted query, not a full-graph pull");
    snap_sql("graph_get_nodes_by_source_ids__default", &stmts[0]);

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_nodes_by_source_ids(&chunks, "").await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1);
    snap_sql("graph_get_nodes_by_source_ids__wildcard", &stmts[0]);

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_edges_by_source_ids(&chunks, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1);
    snap_sql("graph_get_edges_by_source_ids__default", &stmts[0]);

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.get_edges_by_source_ids(&chunks, "").await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1);
    snap_sql("graph_get_edges_by_source_ids__wildcard", &stmts[0]);

    // Empty chunk list -> empty result, zero statements. Degrading this into a
    // full scan would be worse than the code it replaces.
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    assert!(storage
        .get_nodes_by_source_ids(&[], BIZ)
        .await
        .unwrap()
        .is_empty());
    assert!(storage
        .get_edges_by_source_ids(&[], BIZ)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(db.statements().len(), 0);
}

/// The `source_id` backfill runs from `initialize()`, which the workspace cache
/// calls on every bundle rebuild. Its SQL is hand-concatenated *and* its failures
/// are deliberately swallowed into a `warn`, so a typo here would not error — it
/// would just leave every legacy row invisible to the array filter. Both the
/// bytes and the idempotence guards are pinned.
#[tokio::test]
async fn sql_golden_graph_source_id_migration() {
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage.initialize().await.unwrap();

    let stmts = db.statements();
    // 2 CREATE TABLE + 4 pre-existing indexes, then (2 indexes + probe + UPDATE +
    // probe) for each of the two tables.
    assert_eq!(stmts.len(), 16, "{stmts:#?}");
    let migration: Vec<&str> = stmts.iter().skip(6).map(String::as_str).collect();
    snap_sql("graph_migrate_source_id", &migration.join("\n"));

    // The trap this is really about: `initialize()` already owns
    // `idx_graph_edges_graph_src` as a btree over the edges' `source` *column*.
    // A GIN index under that same name would be skipped by `IF NOT EXISTS`, and
    // every edge filter would silently stay a sequential scan.
    assert!(
        stmts.iter().any(|s| s.contains("idx_graph_edges_graph_src ON")),
        "the pre-existing btree on the source column must still be created"
    );
    assert!(
        stmts.iter().any(|s| s.contains("idx_graph_edges_graph_srcid ON")),
        "the GIN index must not collide with it"
    );
    assert!(
        !stmts
            .iter()
            .any(|s| s.contains("USING gin") && s.contains("idx_graph_edges_graph_src ON")),
        "GIN must never reuse the btree name"
    );

    // Order gate: indexes exist before the rows are rewritten, so the UPDATE and
    // every later filter are index-assisted from the first run.
    let first_index = stmts.iter().position(|s| s.contains("_srcid_legacy ON")).unwrap();
    let first_update = stmts.iter().position(|s| s.starts_with("UPDATE ")).unwrap();
    assert!(first_index < first_update, "indexes must precede the backfill");

    // `jsonb_ops`, not `jsonb_path_ops`: only the former can serve `?|`.
    assert!(
        stmts.iter().any(|s| s.contains("USING gin ((properties -> 'source_id'))")),
        "{stmts:#?}"
    );
    assert!(!stmts.iter().any(|s| s.contains("jsonb_path_ops")));

    for t in ["graph_nodes_graph", "graph_edges_graph"] {
        let update = stmts
            .iter()
            .find(|s| s.starts_with(&format!("UPDATE {t} ")))
            .unwrap_or_else(|| panic!("no backfill for {t}: {stmts:#?}"));
        // Idempotent: only jsonb strings are rewritten, so a finished run matches
        // nothing and an interrupted one resumes.
        assert!(
            update.contains("jsonb_typeof(properties -> 'source_id') = 'string'"),
            "{update}"
        );
        // Both legacy delimiters, matching `parse_source_ids`.
        assert!(
            update.contains("replace(properties ->> 'source_id', '<SEP>', ',')"),
            "{update}"
        );
        // Element order must survive: FIFO capping drops from the front, so a
        // reordered array would evict the wrong chunk ids later.
        assert!(update.contains("ORDER BY x.ord"), "{update}");
        assert!(update.contains("btrim"), "{update}");
        // Workspace-wide (initialize has no biz context), and never unscoped.
        assert!(update.contains("workspace = 'test_ws'"), "{update}");
    }

    // Re-running is the normal case, not an edge case: every statement is guarded
    // so the second run is a no-op rather than a rewrite.
    let db2 = Arc::new(MockPgDb::default());
    let storage2 = create_graph_storage(db2.clone());
    storage2.initialize().await.unwrap();
    storage2.initialize().await.unwrap();
    assert_eq!(
        db2.statements().len(),
        32,
        "a second initialize() repeats the same guarded statements, no extra DDL"
    );
}

/// The cleanup write path. Deleting a document used to issue two statements per
/// node (`remove_nodes` looping `delete_node`) and one per edge, so a 200-node /
/// 600-edge cleanup was ~1000 round trips *after* the full-graph pull was already
/// gone. These pin the batched replacements — both the SQL bytes, which are
/// hand-concatenated so a misplaced quote silently deletes nothing, and the
/// round-trip counts, which must not go back up.
#[tokio::test]
async fn sql_golden_graph_cleanup_writes_are_set_based() {
    // remove_nodes: edges first, then nodes. Both statements live in ONE
    // snapshot because their order is the semantics — flipping them would leave
    // an edge pointing at a node that is already gone.
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    let ids: Vec<String> = ["Alice", "Bob", "Carol"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    storage.remove_nodes(&ids, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 2, "3 nodes = 2 statements, not 6");
    assert!(stmts[0].starts_with("DELETE FROM graph_edges_graph"), "{}", stmts[0]);
    assert!(stmts[1].starts_with("DELETE FROM graph_nodes_graph"), "{}", stmts[1]);
    snap_sql("graph_remove_nodes_set", &stmts.join("\n"));

    // remove_edges: one statement, still undirected (both column orders).
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    storage
        .remove_edges(
            &[
                ("Alice".to_string(), "Bob".to_string()),
                ("Bob".to_string(), "Carol".to_string()),
            ],
            BIZ,
        )
        .await
        .unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "2 edges = 1 statement, not 2");
    snap_sql("graph_remove_edges_set", &stmts[0]);

    // upsert_nodes_batch: the trim write-back, one statement per batch. The
    // single-row `upsert_node` above keeps its own golden — this is an added
    // batch statement, not a rewrite of it.
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    let nodes: Vec<(String, JsonMap)> = ["Alice", "Bob"]
        .iter()
        .map(|id| ((*id).to_string(), json_map(json!({"entity_type": "person"}))))
        .collect();
    storage.upsert_nodes_batch(&nodes, BIZ).await.unwrap();
    let stmts = db.statements();
    assert_eq!(stmts.len(), 1, "2 nodes = 1 statement, not 2");
    snap_sql("graph_upsert_nodes_batch", &stmts[0]);

    // Sharding keeps a huge delete from building one unbounded statement, but
    // only whole batches are added: one id over the limit = 2 shards.
    let over = PgBatchLimits::DELETE_MAX_RECORDS_PER_BATCH + 1;
    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    let many: Vec<String> = (0..over).map(|i| format!("n{i}")).collect();
    storage.remove_nodes(&many, BIZ).await.unwrap();
    assert_eq!(db.statements().len(), 4, "2 shards x (edges + nodes)");

    let db = Arc::new(MockPgDb::default());
    let storage = create_graph_storage(db.clone());
    let pairs: Vec<(String, String)> = (0..over).map(|i| (format!("a{i}"), format!("b{i}"))).collect();
    storage.remove_edges(&pairs, BIZ).await.unwrap();
    assert_eq!(db.statements().len(), 2, "2 shards x 1 statement");
}

/// The graph trim is the newest caller of `batched_with_fallback`, so it inherits
/// — and must not quietly lose — the per-record failure granularity the KV and
/// vector upserts promise: one bad row must not take the whole batch down.
#[tokio::test]
async fn graph_upsert_nodes_batch_replays_a_failed_batch_row_by_row() {
    let db = Arc::new(MockPgDb::default());
    let failing = FailingPgDb::failing_batches(db.clone());
    let failures = failing.failures.clone();
    let storage =
        PgGraphStorage::with_options("graph", "test_ws", JsonMap::new(), None, failing).unwrap();

    let nodes: Vec<(String, JsonMap)> = ["Alice", "Bob", "Carol"]
        .iter()
        .map(|id| ((*id).to_string(), json_map(json!({"entity_type": "person"}))))
        .collect();
    storage.upsert_nodes_batch(&nodes, BIZ).await.unwrap();

    assert_eq!(
        failures.load(Ordering::SeqCst),
        1,
        "exactly one batched statement failed"
    );
    assert_eq!(
        db.total_calls(),
        4,
        "1 failed batch + 3 single-row replays"
    );
}

#[tokio::test]
async fn kv_upsert_replays_a_failed_batch_row_by_row() {
    let db = Arc::new(MockPgDb::default());
    let failing = FailingPgDb::failing_batches(db.clone());
    let failures = failing.failures.clone();
    let storage = create_kv_storage(failing);

    storage.upsert(three_records("k"), BIZ).await.unwrap();

    assert_eq!(
        failures.load(Ordering::SeqCst),
        1,
        "exactly one batched statement failed"
    );
    let stmts = db.statements();
    assert_eq!(stmts.len(), 4, "1 batch attempt + 3 replayed rows");
    assert!(stmts[0].contains("), ("), "first attempt is multi-row");
    for sql in &stmts[1..] {
        assert!(!sql.contains("), ("), "replayed statement is single-row: {sql}");
    }
    assert_eq!(db.total_calls(), 4, "no extra round-trips beyond the replay");
}

#[tokio::test]
async fn kv_upsert_row_failure_still_aborts_the_upsert() {
    // Deterministic single record: the batch attempt fails and the replayed row
    // fails too, so the `?` inside the fallback must still propagate the error
    // instead of swallowing it (per-record failure granularity).
    let db = Arc::new(MockPgDb::default());
    let storage = create_kv_storage(FailingPgDb::new(db.clone(), |_| true));
    let mut data = HashMap::new();
    data.insert("k1".to_string(), json_map(json!({"content": "v1"})));

    let err = storage.upsert(data, BIZ).await.unwrap_err();
    assert!(err.to_string().contains("injected failure"), "{err}");
    assert_eq!(db.total_calls(), 2, "batch attempt + one replayed row");
}

#[tokio::test]
async fn vector_upsert_falls_back_without_re_embedding() {
    let db = Arc::new(MockPgDb::default());
    let failing = FailingPgDb::failing_batches(db.clone());
    let failures = failing.failures.clone();
    let embedder = CountingEmbeddingFunc::new();
    let calls = embedder.calls.clone();
    let sizes = embedder.batch_sizes.clone();
    let storage = create_vector_storage(failing, embedder);

    storage
        .upsert(three_records("chunk"), BIZ)
        .await
        .unwrap();

    assert_eq!(failures.load(Ordering::SeqCst), 1);
    assert_eq!(db.total_calls(), 4, "1 batch attempt + 3 replayed rows");
    // Embeddings are computed ONCE per batch and reused by the replay;
    // re-embedding per row would mean three extra paid model calls.
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "exactly one embed() call for the whole batch"
    );
    assert_eq!(*sizes.lock(), vec![3], "that call carried all three texts");
}

// ---------------------------------------------------------------------------
// Phase 3.8 guardrail: the CTE path derives its node set from the edge rows.
// The SQL golden locks the *statement text*; this locks the *result mapping* —
// seed first, then endpoints in first-seen order — because that order decides
// which nodes get a property fetch and so which survive `max_nodes`. Written
// against real edge rows (`MockPgDb::with_rows`) since every other graph test
// here runs on empty rows, where `node_ids` degenerates to `[label]`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn graph_knowledge_cte_derives_nodes_from_edges_in_first_seen_order() {
    // The seed appears as a *target* in the second row and as no source at all,
    // so the expected order is [Alice, Bob, Carol] — neither the raw row order
    // [Bob, Carol, Alice] nor source-before-seed [Bob, Alice, Carol].
    let rows = vec![
        json!({ "source": "Bob", "target": "Carol" }).as_object().unwrap().clone(),
        json!({ "source": "Carol", "target": "Alice" }).as_object().unwrap().clone(),
    ];
    let db = Arc::new(MockPgDb::with_rows(rows));
    let storage = create_graph_storage(db.clone());

    let graph = storage
        .get_knowledge_graph(Some("Alice"), 2, 50, BIZ)
        .await
        .unwrap();

    // Seed first, then endpoints in first-seen order (deduplicated).
    assert_eq!(
        vec!["Alice", "Bob", "Carol"],
        graph.nodes.iter().map(|n| n.id.as_str()).collect::<Vec<_>>()
    );
    // Relational backends have no label concept: the id doubles as the label.
    assert_eq!(
        vec![vec!["Alice"], vec!["Bob"], vec!["Carol"]],
        graph.nodes.iter().map(|n| n.labels.clone()).collect::<Vec<_>>()
    );
    // Edges keep the row order and the composite `source-target` id, and carry
    // no `edge_type` (serde skips it, so the wire shape has no `type` field).
    assert_eq!(
        vec!["Bob-Carol", "Carol-Alice"],
        graph.edges.iter().map(|e| e.id.as_str()).collect::<Vec<_>>()
    );
    assert!(graph.edges.iter().all(|e| e.edge_type.is_none()));
    // 3 endpoints < max_nodes, so nothing was dropped.
    assert!(!graph.is_truncated);
    // One CTE fetch + exactly one property fetch per node: deriving the node set
    // must not add round-trips.
    assert_eq!(1 + 3, db.total_calls());
}
