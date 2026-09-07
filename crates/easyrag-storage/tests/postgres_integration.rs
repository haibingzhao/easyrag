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

// ---------------------------------------------------------------------------
// source_id backfill (live)
// ---------------------------------------------------------------------------

/// Render an `EXPLAIN` result. The plan column's name is driver-dependent, so
/// take every string in the row instead of guessing the key.
fn plan_text(rows: &[JsonMap]) -> String {
    rows.iter()
        .flat_map(|row| row.values())
        .filter_map(|value| value.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn scalar(db_rows: Vec<JsonMap>, key: &str) -> i64 {
    db_rows
        .first()
        .and_then(|row| row.get(key))
        .and_then(|value| value.as_i64())
        .unwrap_or(-1)
}

/// The backfill's correctness contract, against a real server: both historical
/// delimiters convert, element order survives, a second `initialize()` is a
/// no-op, the probe reaches zero, and the GIN index is both present and actually
/// chosen for `?|`.
#[tokio::test]
#[ignore]
async fn pg_graph_source_id_backfill_converts_and_is_idempotent() {
    let ws = unique_workspace("srcid");
    let db = connect(&ws).await;
    // A namespace of its own: graph table names derive from the namespace, not
    // the workspace, so two live tests sharing one would race on `CREATE INDEX`
    // and on each other's rows.
    let graph = PgGraphStorage::new("srcid_migrate", &ws, db.clone()).unwrap();
    graph.initialize().await.unwrap();

    let (nodes, edges) = ("graph_nodes_srcid_migrate", "graph_edges_srcid_migrate");
    let probe = |table: &str| {
        let (db, ws, table) = (db.clone(), ws.clone(), table.to_string());
        async move {
            scalar(
                db.fetch_all(&format!(
                    "SELECT count(*) AS cnt FROM {table} WHERE workspace = '{ws}' \
                     AND jsonb_typeof(properties -> 'source_id') = 'string'"
                ))
                .await
                .unwrap(),
                "cnt",
            )
        }
    };

    // Seed legacy values *behind the storage layer's back*: an old pod leaves
    // exactly these. Both delimiter forms, plus the prefix pair that motivated
    // the whole refactor.
    for (id, sid) in [
        ("Alpha", "c1<SEP>c2"),
        ("Beta", "c1,c3"),
        ("chunk_1", "chunk_1"),
        ("chunk_10", "chunk_10"),
        ("Padded", " c7 <SEP>  "),
    ] {
        db.execute(&format!(
            "INSERT INTO {nodes} (id, workspace, biz_id, properties) \
             VALUES ('{id}', '{ws}', '{BIZ}', '{{\"source_id\": \"{sid}\"}}'::jsonb)"
        ))
        .await
        .unwrap();
    }
    db.execute(&format!(
        "INSERT INTO {edges} (id, workspace, biz_id, source, target, properties) \
         VALUES ('Alpha-Beta', '{ws}', '{BIZ}', 'Alpha', 'Beta', \
                 '{{\"source_id\": \"c1<SEP>c9\"}}'::jsonb)"
    ))
    .await
    .unwrap();
    assert_eq!(5, probe(nodes).await);
    assert_eq!(1, probe(edges).await);

    // Pre-migration the legacy arm is what makes cleanup correct: `?|` alone is
    // false against a string scalar, so without that arm these five nodes would
    // be silently skipped and never cleaned up.
    let legacy_hits = graph
        .get_nodes_by_source_ids(&["c1".to_string()], BIZ)
        .await
        .unwrap();
    let mut ids: Vec<&str> = legacy_hits
        .iter()
        .filter_map(|n| n.get("id").and_then(|v| v.as_str()))
        .collect();
    ids.sort_unstable();
    assert_eq!(vec!["Alpha", "Beta"], ids);

    graph.initialize().await.unwrap();

    // Both delimiters split, padding is trimmed, empties dropped, order kept.
    let alpha = graph.get_node("Alpha", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c2"]), alpha["source_id"]);
    let beta = graph.get_node("Beta", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c3"]), beta["source_id"]);
    let padded = graph.get_node("Padded", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c7"]), padded["source_id"]);
    let edge = graph.get_edge("Alpha", "Beta", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c9"]), edge["source_id"]);

    // The completion gate: zero legacy values left, on both tables.
    assert_eq!(0, probe(nodes).await);
    assert_eq!(0, probe(edges).await);

    // Second run is a no-op on the data: the partial index makes the WHERE an
    // index scan over zero rows, and the values are untouched.
    let before = graph.get_node("Alpha", BIZ).await.unwrap().unwrap();
    graph.initialize().await.unwrap();
    assert_eq!(before, graph.get_node("Alpha", BIZ).await.unwrap().unwrap());
    assert_eq!(0, probe(nodes).await);

    // Element-level equality is the point of the array model: a substring match
    // on the joined string would have dragged chunk_10 in with chunk_1.
    let hits = graph
        .get_nodes_by_source_ids(&["chunk_1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(1, hits.len());
    assert_eq!("chunk_1", hits[0]["id"].as_str().unwrap());
    let hits = graph
        .get_nodes_by_source_ids(&["c1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(2, hits.len(), "post-backfill the array arm still finds both");

    graph.drop(BIZ).await.unwrap();
}

/// R4 acceptance: `?|` has to be resolved by the expression GIN index, not a
/// sequential scan. The index is named `_srcid` because `initialize()` already
/// owns `idx_{edges}_src` for the btree over the edges' `source` *column* —
/// reusing that name would make `CREATE INDEX IF NOT EXISTS` skip ours silently
/// and leave every edge filter a seq scan with nothing in the logs to say so.
#[tokio::test]
#[ignore]
async fn pg_graph_source_id_filter_uses_the_gin_index() {
    let ws = unique_workspace("srcidx");
    let db = connect(&ws).await;
    let graph = PgGraphStorage::new("gin_probe", &ws, db.clone()).unwrap();
    graph.initialize().await.unwrap();
    let (nodes, edges) = ("graph_nodes_gin_probe", "graph_edges_gin_probe");

    // Definitions first: this is what makes `?|` *eligible* for an index, and it
    // is the assertion that catches the naming collision. Must be the default
    // `jsonb_ops` — `jsonb_path_ops` does not support `?|` at all. The strings
    // below are Postgres' own normalized rendering of the DDL, `::text` casts
    // and extra parens included.
    let defs = db
        .fetch_all(&format!(
            "SELECT indexdef FROM pg_indexes WHERE tablename IN ('{nodes}', '{edges}')"
        ))
        .await
        .unwrap();
    // Suffix-matched rather than `LIKE '%srcid%'`: the namespace is part of the
    // table name, so a substring filter also picks up the primary key and the
    // `_src`/`_tgt`/`_ws_biz` btrees.
    let rendered: Vec<String> = defs
        .iter()
        .filter_map(|row| row.get("indexdef").and_then(|v| v.as_str()).map(str::to_string))
        .filter(|def| def.contains("_srcid ON") || def.contains("_srcid_legacy ON"))
        .collect();
    assert_eq!(4, rendered.len(), "2 tables x (gin + legacy partial): {rendered:?}");
    let gin = "USING gin (((properties -> 'source_id'::text)))";
    assert_eq!(
        2,
        rendered.iter().filter(|d| d.contains(gin)).count(),
        "expression GIN on both tables: {rendered:?}"
    );
    assert!(
        rendered.iter().all(|d| !d.contains("jsonb_path_ops")),
        "jsonb_path_ops cannot serve ?|: {rendered:?}"
    );
    let partial = "WHERE (jsonb_typeof((properties -> 'source_id'::text)) = 'string'::text)";
    assert_eq!(
        2,
        rendered.iter().filter(|d| d.contains(partial)).count(),
        "legacy partial index on both tables: {rendered:?}"
    );
    // The collision this naming avoids, asserted against the live catalog: `_src`
    // is already taken by a btree on the edges' `source` *column*, so a GIN
    // called `_src` would have been skipped by IF NOT EXISTS.
    let clash = db
        .fetch_all(&format!(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_{edges}_src'"
        ))
        .await
        .unwrap();
    assert_eq!(
        format!(
            "CREATE INDEX idx_{edges}_src ON public.{edges} USING btree (source, workspace, biz_id)"
        ),
        plan_text(&clash),
        "the pre-existing btree still owns the _src name"
    );

    // Then the plan. A handful of rows would legitimately seq-scan, so seed
    // enough that the planner has a reason to prefer the index, and keep the
    // predicate selective.
    db.execute(&format!(
        "INSERT INTO {nodes} (id, workspace, biz_id, properties) \
         SELECT 'bulk_' || g, '{ws}', '{BIZ}', \
                jsonb_build_object('source_id', jsonb_build_array('chunk_' || g), \
                                   'description', repeat('x', 200)) \
         FROM generate_series(1, 20000) AS g"
    ))
    .await
    .unwrap();
    db.execute(&format!("ANALYZE {nodes}")).await.unwrap();

    let plan = plan_text(
        &db.fetch_all(&format!(
            "EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM {nodes} \
             WHERE workspace = '{ws}' AND (properties -> 'source_id') ?| ARRAY['chunk_7']::text[]"
        ))
        .await
        .unwrap(),
    );
    assert!(
        plan.contains(&format!("idx_{nodes}_srcid")),
        "expected a bitmap index scan on the expression GIN, got:\n{plan}"
    );
    assert!(
        !plan.contains("Seq Scan"),
        "the server-side filter must not degrade to a seq scan:\n{plan}"
    );

    graph.drop(BIZ).await.unwrap();
}

