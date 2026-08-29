//! PostgreSQL storage tests.
//!
//! `PgDb` facade (replaces Mockito), plus workspace validation tests.
//! Integration tests requiring a live PostgreSQL live in
//! `postgres_integration.rs` behind `#[ignore]`.

#![cfg(feature = "postgres")]

use async_trait::async_trait;
use easyrag_core::llm::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::JsonMap;
use easyrag_storage::postgres::{PgConfig, PgDb, PgVectorStorage};
use parking_lot::Mutex;
use serde_json::json;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";
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
}

impl MockPgDb {
    fn with_rows(rows: Vec<JsonMap>) -> Self {
        Self {
            fetch_all_rows: rows,
            fetch_all_calls: AtomicUsize::new(0),
            last_sql: Mutex::new(String::new()),
        }
    }
}

#[async_trait]
impl PgDb for MockPgDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        *self.last_sql.lock() = sql.to_string();
        Ok(())
    }

    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        *self.last_sql.lock() = sql.to_string();
        self.fetch_all_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.fetch_all_rows.first().cloned())
    }

    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        *self.last_sql.lock() = sql.to_string();
        self.fetch_all_calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.fetch_all_rows.clone())
    }

    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        *self.last_sql.lock() = sql.to_string();
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
