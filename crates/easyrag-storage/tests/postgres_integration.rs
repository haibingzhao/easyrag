//! PostgreSQL integration tests (require a live PostgreSQL with pgvector).
//!
//! Run with:
//!   PG_URL=postgres://user:pass@localhost:5432/easyrag \
//!     cargo test -p easyrag-storage --features postgres -- --ignored
//!
//! Each test uses a unique workspace name to avoid cross-test interference
//! and drops its tables at the end.

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
    PgConfig, PgDb, PgDatabase, PgDocStatusStorage, PgGraphStorage, PgKVStorage, PgVectorStorage,
};
use serde_json::json;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn json_map(value: serde_json::Value) -> JsonMap {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

fn pg_url() -> String {
    std::env::var("PG_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5432/easyrag".to_string())
}

/// Parse postgres://user:pass@host:port/db (test helper only).
async fn connect(workspace: &str) -> PgDatabase {
    let url = pg_url();
    let rest = url
        .strip_prefix("postgres://")
        .or_else(|| url.strip_prefix("postgresql://"))
        .unwrap_or(&url);
    let (auth, hostpart) = rest.split_once('@').expect("user:pass@host in PG_URL");
    let (user, password) = match auth.split_once(':') {
        Some((u, p)) => (u.to_string(), p.to_string()),
        None => (auth.to_string(), String::new()),
    };
    let (hostport, database) = hostpart.split_once('/').expect("host/db in PG_URL");
    let (host, port) = match hostport.split_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().expect("port")),
        None => (hostport.to_string(), 5432),
    };
    let base = PgConfig {
        host,
        port,
        user: user.to_string(),
        password: password.to_string(),
        database: database.to_string(),
        workspace: workspace.to_string(),
        ..Default::default()
    };
    let db = PgDatabase::new(base).unwrap();
    db.initialize().await.unwrap();
    db
}

fn unique_workspace(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// Deterministic embedding: first dims are text bytes normalized.
struct DeterministicFunc;

#[async_trait]
impl EmbeddingFunction for DeterministicFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; 4];
                for (i, b) in t.bytes().take(4).enumerate() {
                    v[i] = b as f32 / 255.0;
                }
                v
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// KV storage round-trip
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_kv_storage_round_trip() {
    let ws = unique_workspace("kv");
    let db = connect(&ws).await;
    let storage =
        PgKVStorage::new("test_kv", &ws, db).unwrap();
    storage.initialize().await.unwrap();

    assert!(storage.is_empty(BIZ).await.unwrap());

    storage
        .upsert(HashMap::from([(
            "key1".to_string(),
            json_map(json!({"value": "hello", "n": 42})),
        )]), BIZ)
        .await
        .unwrap();
    storage.index_done_callback().await.unwrap();

    let got = storage.get_by_id("key1", BIZ).await.unwrap().unwrap();
    assert_eq!("hello", got["value"].as_str().unwrap());
    assert_eq!(42, got["n"].as_i64().unwrap());

    // filterKeys returns missing keys only.
    let missing = storage
        .filter_keys(&HashSet::from(["key1".to_string(), "key2".to_string()]), BIZ)
        .await
        .unwrap();
    assert_eq!(HashSet::from(["key2".to_string()]), missing);

    // Upsert overwrites.
    storage
        .upsert(HashMap::from([(
            "key1".to_string(),
            json_map(json!({"value": "world"})),
        )]), BIZ)
        .await
        .unwrap();
    let got = storage.get_by_id("key1", BIZ).await.unwrap().unwrap();
    assert_eq!("world", got["value"].as_str().unwrap());

    storage.delete(&["key1".to_string()], BIZ).await.unwrap();
    assert!(storage.get_by_id("key1", BIZ).await.unwrap().is_none());
    assert!(storage.is_empty(BIZ).await.unwrap());

    let result = storage.drop(BIZ).await.unwrap();
    assert_eq!("success", result["status"]);
}

// ---------------------------------------------------------------------------
// Vector storage round-trip (pgvector cosine + FTS keyword)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_vector_storage_round_trip() {
    let ws = unique_workspace("vec");
    let db = connect(&ws).await;
    let config = PgConfig {
        workspace: ws.clone(),
        embedding_dim: 4,
        fts_config: "simple".to_string(),
        ..Default::default()
    };
    let embedding = EmbeddingFunc::new(4, Arc::new(DeterministicFunc));
    let storage = PgVectorStorage::new(
        "test_vec",
        &ws,
        embedding,
        db,
        config,
    )
    .unwrap();
    storage.initialize().await.unwrap();

    storage
        .upsert(HashMap::from([
            (
                "chunk1".to_string(),
                json_map(json!({"content": "aaaa bbbb cccc", "file_path": "a.txt"})),
            ),
            (
                "chunk2".to_string(),
                json_map(json!({"content": "zzzz yyyy xxxx", "file_path": "b.txt"})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    // Vector query: identical text should return itself with high similarity.
    let results = storage.query("aaaa bbbb cccc", 5, None, None, None, BIZ).await.unwrap();
    assert!(!results.is_empty());
    assert_eq!("chunk1", results[0]["id"].as_str().unwrap());
    assert!(results[0]["score"].as_f64().unwrap() > 0.99);

    // get_vectors_by_ids parses embeddings back.
    let vectors = storage
        .get_vectors_by_ids(&["chunk1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(4, vectors["chunk1"].len());

    // Keyword query through the tsvector trigger.
    let kw = storage.keyword_query("zzzz", 5, None, None, BIZ).await.unwrap();
    assert_eq!(1, kw.len());
    assert_eq!("chunk2", kw[0]["id"].as_str().unwrap());

    storage.delete(&["chunk1".to_string(), "chunk2".to_string()], BIZ).await.unwrap();
    let result = storage.drop(BIZ).await.unwrap();
    assert_eq!("success", result["status"]);
}

// ---------------------------------------------------------------------------
// Graph storage round-trip (nodes, undirected edges, CTE subgraph)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_graph_storage_round_trip() {
    let ws = unique_workspace("graph");
    let db = connect(&ws).await;
    let storage = PgGraphStorage::new(
        "test_graph",
        &ws,
        db,
    )
    .unwrap();
    storage.initialize().await.unwrap();

    storage
        .upsert_node("Apple", json_map(json!({"entity_type": "ORG", "description": "A company"})), BIZ)
        .await
        .unwrap();
    storage
        .upsert_node("iPhone", json_map(json!({"entity_type": "PRODUCT"})), BIZ)
        .await
        .unwrap();
    storage
        .upsert_edge("Apple", "iPhone", json_map(json!({"keywords": "makes"})), BIZ)
        .await
        .unwrap();

    assert!(storage.has_node("Apple", BIZ).await.unwrap());
    assert!(storage.has_edge("Apple", "iPhone", BIZ).await.unwrap());
    // Undirected semantics.
    assert!(storage.has_edge("iPhone", "Apple", BIZ).await.unwrap());
    assert_eq!(1, storage.node_degree("Apple", BIZ).await.unwrap());
    assert_eq!(2, storage.edge_degree("Apple", "iPhone", BIZ).await.unwrap());

    let node = storage.get_node("Apple", BIZ).await.unwrap().unwrap();
    assert_eq!("ORG", node["entity_type"].as_str().unwrap());

    let edge = storage.get_edge("iPhone", "Apple", BIZ).await.unwrap().unwrap();
    assert_eq!("makes", edge["keywords"].as_str().unwrap());

    let labels = storage.get_all_labels(BIZ).await.unwrap();
    assert_eq!(vec!["Apple".to_string(), "iPhone".to_string()], labels);

    let kg = storage
        .get_knowledge_graph(Some("Apple"), 2, 100, BIZ)
        .await
        .unwrap();
    assert_eq!(2, kg.nodes.len());
    assert_eq!(1, kg.edges.len());

    let popular = storage.get_popular_labels(1, BIZ).await.unwrap();
    assert_eq!(vec!["Apple".to_string()], popular);

    let search = storage.search_labels("app", 10, BIZ).await.unwrap();
    assert!(search.contains(&"Apple".to_string()));

    storage.delete_node("Apple", BIZ).await.unwrap();
    assert!(!storage.has_node("Apple", BIZ).await.unwrap());
    assert!(!storage.has_edge("Apple", "iPhone", BIZ).await.unwrap());

    let result = storage.drop(BIZ).await.unwrap();
    assert_eq!("success", result["status"]);
}

// ---------------------------------------------------------------------------
// DocStatus storage round-trip (pagination, status filters, dedupe lookups)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_doc_status_storage_round_trip() {
    let ws = unique_workspace("doc");
    let db = connect(&ws).await;
    let storage = PgDocStatusStorage::new(
        "doc_status",
        &ws,
        db,
    )
    .unwrap();
    storage.initialize().await.unwrap();

    storage
        .upsert(HashMap::from([
            (
                "doc1".to_string(),
                json_map(json!({
                    "status": "processed",
                    "file_path": "/uploads/a.txt",
                    "content_hash": "hash1",
                    "track_id": "t1",
                    "content_summary": "summary a",
                    "content_length": 100
                })),
            ),
            (
                "doc2".to_string(),
                json_map(json!({
                    "status": "pending",
                    "file_path": "/uploads/b.txt",
                    "content_hash": "hash2",
                    "track_id": "t2",
                    "content_summary": "summary b",
                    "content_length": 200
                })),
            ),
        ]), BIZ)
        .await
        .unwrap();

    let counts = storage.get_status_counts(BIZ).await.unwrap();
    assert_eq!(1, counts["processed"]);
    assert_eq!(1, counts["pending"]);

    let processed = storage.get_docs_by_status(DocStatus::Processed, BIZ).await.unwrap();
    assert!(processed.contains_key("doc1"));

    let (page, total) = storage
        .get_docs_paginated(None, None, 1, 10, "updated_at", "desc", None, BIZ)
        .await
        .unwrap();
    assert_eq!(2, total);
    assert_eq!(2, page.len());

    // R5: server-side file_path prefix filter.
    let (page, total) = storage
        .get_docs_paginated(None, None, 1, 10, "updated_at", "desc", Some("/uploads/a"), BIZ)
        .await
        .unwrap();
    assert_eq!(1, total);
    assert_eq!("doc1", page[0].0);

    let by_track = storage.get_docs_by_track_id("t2", BIZ).await.unwrap();
    assert!(by_track.contains_key("doc2"));

    let by_path = storage.get_doc_by_file_path("/uploads/a.txt", BIZ).await.unwrap();
    assert!(by_path.is_some());

    let by_basename = storage.get_doc_by_file_basename("b.txt", BIZ).await.unwrap();
    assert_eq!("doc2", by_basename.unwrap().0);

    let by_hash = storage.get_doc_by_content_hash("hash1", BIZ).await.unwrap();
    assert_eq!("doc1", by_hash.unwrap().0);

    let result = storage.drop(BIZ).await.unwrap();
    assert_eq!("success", result["status"]);
}

// ---------------------------------------------------------------------------
// biz_id isolation: same workspace, two business lines
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn pg_biz_id_isolation() {
    let ws = unique_workspace("biz");
    let db = connect(&ws).await;
    const BIZ_A: &str = "biz_a";
    const BIZ_B: &str = "biz_b";

    // --- KV: same key in two biz lines stays separate ----------------
    // One shared instance; biz_id is a per-call filter argument.
    let kv = PgKVStorage::new("test_biz_kv", &ws, db.clone()).unwrap();
    kv.initialize().await.unwrap();
    kv.upsert(HashMap::from([(
        "k1".to_string(),
        json_map(json!({"owner": "a"})),
    )]), BIZ_A)
    .await
    .unwrap();
    kv.upsert(HashMap::from([(
        "k1".to_string(),
        json_map(json!({"owner": "b"})),
    )]), BIZ_B)
    .await
    .unwrap();
    assert_eq!(
        "a",
        kv.get_by_id("k1", BIZ_A).await.unwrap().unwrap()["owner"]
            .as_str()
            .unwrap()
    );
    assert_eq!(
        "b",
        kv.get_by_id("k1", BIZ_B).await.unwrap().unwrap()["owner"]
            .as_str()
            .unwrap()
    );
    // drop only clears the caller's own biz slice.
    kv.drop(BIZ_A).await.unwrap();
    assert!(kv.get_by_id("k1", BIZ_A).await.unwrap().is_none());
    assert!(kv.get_by_id("k1", BIZ_B).await.unwrap().is_some());
    kv.drop(BIZ_B).await.unwrap();

    // --- Vector: biz_b must not retrieve biz_a chunks ----------------
    let vec_config = PgConfig {
        workspace: ws.clone(),
        embedding_dim: 4,
        fts_config: "simple".to_string(),
        ..Default::default()
    };
    let vec = PgVectorStorage::new(
        "test_biz_vec",
        &ws,
        EmbeddingFunc::new(4, Arc::new(DeterministicFunc)),
        db.clone(),
        vec_config,
    )
    .unwrap();
    vec.initialize().await.unwrap();
    vec.upsert(HashMap::from([(
            "chunk1".to_string(),
            json_map(json!({"content": "aaaa bbbb cccc", "file_path": "a.txt"})),
        )]), BIZ_A)
        .await
        .unwrap();
    assert!(vec
        .get_vectors_by_ids(&["chunk1".to_string()], BIZ_B)
        .await
        .unwrap()
        .is_empty());
    assert!(vec
        .query("aaaa bbbb cccc", 5, None, None, None, BIZ_B)
        .await
        .unwrap()
        .is_empty());
    assert!(vec
        .keyword_query("aaaa", 5, None, None, BIZ_B)
        .await
        .unwrap()
        .is_empty());
    // biz_a still sees its own chunk.
    assert_eq!(
        1,
        vec.query("aaaa bbbb cccc", 5, None, None, None, BIZ_A)
            .await
            .unwrap()
            .len()
    );
    vec.drop(BIZ_A).await.unwrap();
    vec.drop(BIZ_B).await.unwrap();

    // --- Graph: same entity name is a distinct node per biz ----------
    let graph = PgGraphStorage::new("test_biz_graph", &ws, db.clone()).unwrap();
    graph.initialize().await.unwrap();
    graph
        .upsert_node("Apple", json_map(json!({"entity_type": "ORG"})), BIZ_A)
        .await
        .unwrap();
    graph
        .upsert_edge(
            "Apple",
            "iPhone",
            json_map(json!({"keywords": "makes"})),
            BIZ_A,
        )
        .await
        .unwrap();
    assert!(!graph.has_node("Apple", BIZ_B).await.unwrap());
    assert!(graph.get_node("Apple", BIZ_B).await.unwrap().is_none());
    assert!(graph
        .get_node_edges("Apple", BIZ_B)
        .await
        .unwrap()
        .unwrap_or_default()
        .is_empty());
    assert!(graph.has_node("Apple", BIZ_A).await.unwrap());
    graph.drop(BIZ_A).await.unwrap();
    graph.drop(BIZ_B).await.unwrap();

    // --- DocStatus: status counts are per-biz -------------------------
    let doc = PgDocStatusStorage::new("test_biz_doc", &ws, db.clone()).unwrap();
    doc.initialize().await.unwrap();
    doc.upsert(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({
                "status": "processed",
                "file_path": "/a.txt",
                "content_hash": "h1",
                "content_summary": "s",
                "content_length": 1
            })),
        )]), BIZ_A)
        .await
        .unwrap();
    let counts_a = doc.get_status_counts(BIZ_A).await.unwrap();
    assert_eq!(1, counts_a["processed"]);
    let counts_b = doc.get_status_counts(BIZ_B).await.unwrap();
    assert!(counts_b.get("processed").is_none() || counts_b["processed"] == 0);
    doc.drop(BIZ_A).await.unwrap();
    doc.drop(BIZ_B).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn pg_biz_id_not_null_enforced() {
    let ws = unique_workspace("biznull");
    let db = connect(&ws).await;
    let storage = PgKVStorage::new(
        "test_biz_null",
        &ws,
        db.clone(),
    )
    .unwrap();
    storage.initialize().await.unwrap();

    // Bypassing the storage layer, an INSERT without biz_id must be
    // rejected: NOT NULL with no DEFAULT makes omission a hard error.
    let sql = format!(
        "INSERT INTO kv_test_biz_null (id, workspace, content) VALUES ('x', '{}', '{{}}')",
        ws
    );
    assert!(db.execute(&sql).await.is_err());
    storage.drop(BIZ).await.unwrap();
}
