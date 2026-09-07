//! Live Neo4j integration tests (require a running Neo4j instance).
//!
//! Run with:
//!   NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=secret \
//!     cargo test -p easyrag-storage --features neo4j --test neo4j_integration -- --ignored

#![cfg(feature = "neo4j")]

use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use easyrag_storage::neo4j::{
    Neo4jConfig, Neo4jDatabase, Neo4jGraphStorage, Neo4jOps, Neo4jParams,
};
use serde_json::{json, Value as JsonValue};

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

fn json_map(value: serde_json::Value) -> JsonMap {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

fn neo4j_config() -> Neo4jConfig {
    Neo4jConfig {
        uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into()),
        user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
        password: std::env::var("NEO4J_PASSWORD").unwrap_or_default(),
        ..Default::default()
    }
}

async fn connect(workspace: &str) -> Neo4jGraphStorage<Neo4jDatabase> {
    let db = Neo4jDatabase::new(neo4j_config()).await.unwrap();
    let storage = Neo4jGraphStorage::new(
        "graph",
        workspace,
        db,
    )
    .unwrap();
    storage.initialize().await.unwrap();
    storage
}

/// A raw driver plus the escaped workspace label, for seeding values the storage
/// layer would never write itself.
async fn raw(workspace: &str) -> (Neo4jDatabase, String) {
    let db = Neo4jDatabase::new(neo4j_config()).await.unwrap();
    (db, format!("`{workspace}`"))
}

fn biz_params(entries: &[(&str, JsonValue)]) -> Neo4jParams {
    let mut map = Neo4jParams::new();
    map.insert("biz".to_string(), JsonValue::String(BIZ.to_string()));
    for (k, v) in entries {
        map.insert((*k).to_string(), v.clone());
    }
    map
}

#[tokio::test]
#[ignore]
async fn neo4j_graph_storage_round_trip() {
    let ws = format!("itest_{}", uuid::Uuid::new_v4().simple());
    let storage = connect(&ws).await;

    storage
        .upsert_node(
            "Apple",
            json_map(json!({"entity_type": "ORG", "description": "A company"})),
        BIZ)
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
    storage.index_done_callback().await.unwrap();

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

    // Batch ops.
    let nodes = storage
        .get_nodes_batch(&["Apple".to_string(), "iPhone".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(2, nodes.len());
    let existing = storage
        .has_nodes_batch(&["Apple".to_string(), "Missing".to_string()], BIZ)
        .await
        .unwrap();
    assert!(existing.contains("Apple"));
    assert!(!existing.contains("Missing"));

    // Knowledge graph: full and sub.
    let full = storage.get_knowledge_graph(None, 2, 100, BIZ).await.unwrap();
    assert_eq!(2, full.nodes.len());
    assert_eq!(1, full.edges.len());
    let sub = storage
        .get_knowledge_graph(Some("Apple"), 2, 100, BIZ)
        .await
        .unwrap();
    assert_eq!(2, sub.nodes.len());
    assert_eq!(1, sub.edges.len());

    let search = storage.search_labels("App", 10, BIZ).await.unwrap();
    assert!(search.contains(&"Apple".to_string()));

    storage.delete_node("Apple", BIZ).await.unwrap();
    assert!(!storage.has_node("Apple", BIZ).await.unwrap());
    assert!(!storage.has_edge("Apple", "iPhone", BIZ).await.unwrap());

    let result = storage.drop(BIZ).await.unwrap();
    assert_eq!("success", result["status"]);
}

// ---------------------------------------------------------------------------
// source_id backfill (live)
// ---------------------------------------------------------------------------

/// The whole Neo4j side of the refactor, end to end against a real server:
/// legacy strings are matched before the marker exists (through the Rust
/// fallback), converted by the backfill, and still matched afterwards through
/// the server-side array query. That last step is what catches the
/// `sanitize_props` double-quoting bug — had the arrays been written as
/// `["\"c1\""]`, `c IN n.source_id` would never match again.
#[tokio::test]
#[ignore]
async fn neo4j_source_id_backfill_converts_and_publishes_the_marker() {
    let ws = format!("itest_srcid_{}", uuid::Uuid::new_v4().simple());
    let (db, label) = raw(&ws).await;

    // Seed pre-refactor values by hand: `upsert_node` writes arrays now, so the
    // only way to reproduce what an old pod left behind is to bypass it. Both
    // historical delimiters, plus the prefix pair that motivated the refactor.
    for (id, sid) in [
        ("Alpha", "c1<SEP>c2"),
        ("Beta", "c1,c3"),
        ("chunk_1", "chunk_1"),
        ("chunk_10", "chunk_10"),
    ] {
        db.run_write_void(
            &format!("MERGE (n:{label} {{entity_id: $id, biz_id: $biz}}) SET n.source_id = $sid"),
            biz_params(&[
                ("id", json!(id)),
                ("sid", json!(sid)),
            ]),
        )
        .await
        .unwrap();
    }
    db.run_write_void(
        &format!(
            "MATCH (a:{label} {{entity_id: 'Alpha', biz_id: $biz}}), \
                   (b:{label} {{entity_id: 'Beta', biz_id: $biz}}) \
             MERGE (a)-[r:DIRECTED]->(b) SET r.source_id = 'c1<SEP>c9'"
        ),
        biz_params(&[]),
    )
    .await
    .unwrap();

    let storage = Neo4jGraphStorage::new("graph", &ws, db).unwrap();
    assert!(!storage.source_id_arrays_only(), "no marker yet");

    // Before the backfill the Rust fallback is what keeps cleanup correct: the
    // array-only Cypher is false against a string scalar, so these would be
    // silently skipped and never cleaned up.
    let mut before: Vec<String> = storage
        .get_nodes_by_source_ids(&["c1".to_string()], BIZ)
        .await
        .unwrap()
        .iter()
        .filter_map(|n| n.get("entity_id").and_then(|v| v.as_str()).map(str::to_string))
        .collect();
    before.sort();
    assert_eq!(vec!["Alpha", "Beta"], before);
    // Element equality already holds on the fallback path too — it goes through
    // the same `source_ids_intersect` as every other backend: `chunk_1` must not
    // drag `chunk_10` in with it.
    let hits = storage
        .get_nodes_by_source_ids(&["chunk_1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(1, hits.len(), "{hits:?}");
    assert_eq!("chunk_1", hits[0]["entity_id"].as_str().unwrap());

    storage.initialize().await.unwrap();

    // Converted: both delimiters split, order kept, edges included.
    let alpha = storage.get_node("Alpha", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c2"]), alpha["source_id"]);
    let beta = storage.get_node("Beta", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c3"]), beta["source_id"]);
    let edge = storage.get_edge("Alpha", "Beta", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["c1", "c9"]), edge["source_id"]);

    // Marker published, so filtering moved server-side — and it must find
    // exactly the same rows as the fallback did.
    assert!(storage.source_id_arrays_only());
    let mut after: Vec<String> = storage
        .get_nodes_by_source_ids(&["c1".to_string()], BIZ)
        .await
        .unwrap()
        .iter()
        .filter_map(|n| n.get("entity_id").and_then(|v| v.as_str()).map(str::to_string))
        .collect();
    after.sort();
    assert_eq!(before, after);
    let edges = storage
        .get_edges_by_source_ids(&["c9".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(1, edges.len());
    assert_eq!("Alpha", edges[0]["source"].as_str().unwrap());
    assert_eq!("Beta", edges[0]["target"].as_str().unwrap());

    // A second `initialize()` short-circuits on the marker and changes nothing.
    storage.initialize().await.unwrap();
    assert!(storage.source_id_arrays_only());
    assert_eq!(json!(["c1", "c2"]), storage.get_node("Alpha", BIZ).await.unwrap().unwrap()["source_id"]);

    // The marker is workspace-level and outlives `drop(biz_id)`: an emptied
    // workspace still holds no legacy values. A brand-new instance on the same
    // database must therefore short-circuit rather than re-backfill.
    storage.drop(BIZ).await.unwrap();
    let revived = Neo4jGraphStorage::new(
        "graph",
        &ws,
        Neo4jDatabase::new(neo4j_config()).await.unwrap(),
    )
    .unwrap();
    revived.initialize().await.unwrap();
    assert!(
        revived.source_id_arrays_only(),
        "the marker must survive drop(biz_id)"
    );

    // Cleanup: the marker lives under its own label, so drop() leaves it behind.
    let (db, label) = raw(&ws).await;
    db.run_write_void(
        "MATCH (m:EasyRagGraphMeta {workspace: $ws}) DETACH DELETE m",
        {
            let mut p = Neo4jParams::new();
            p.insert("ws".to_string(), json!(ws));
            p
        },
    )
    .await
    .unwrap();
    db.run_write_void(
        &format!("MATCH (n:{label}) DETACH DELETE n"),
        Neo4jParams::new(),
    )
    .await
    .unwrap();
}

/// The migration must leave a workspace that is already all-arrays untouched,
/// and must still publish the marker so later runs short-circuit.
#[tokio::test]
#[ignore]
async fn neo4j_source_id_backfill_on_fresh_data_only_publishes_the_marker() {
    let ws = format!("itest_srcnew_{}", uuid::Uuid::new_v4().simple());
    // `connect` already ran `initialize()`, so the marker exists on an empty
    // workspace and the write path produces arrays anyway.
    let storage = connect(&ws).await;
    assert!(storage.source_id_arrays_only());

    storage
        .upsert_node(
            "Apple",
            json_map(json!({"entity_type": "ORG", "source_id": ["chunk-1"]})),
            BIZ,
        )
        .await
        .unwrap();
    storage
        .upsert_node(
            "iPhone",
            json_map(json!({"entity_type": "PRODUCT", "source_id": ["chunk-1", "chunk-2"]})),
            BIZ,
        )
        .await
        .unwrap();
    storage
        .upsert_edge(
            "Apple",
            "iPhone",
            json_map(json!({"keywords": "makes", "source_id": ["chunk-1"]})),
            BIZ,
        )
        .await
        .unwrap();

    // Round-trip through the server: a list property must come back as a list,
    // with plain string elements.
    let node = storage.get_node("iPhone", BIZ).await.unwrap().unwrap();
    assert_eq!(json!(["chunk-1", "chunk-2"]), node["source_id"]);

    let hits = storage
        .get_nodes_by_source_ids(&["chunk-1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(2, hits.len());
    // Prefix safety on real data: `chunk-1` must not match a hypothetical
    // `chunk-10`.
    storage
        .upsert_node(
            "Watch",
            json_map(json!({"entity_type": "PRODUCT", "source_id": ["chunk-10"]})),
            BIZ,
        )
        .await
        .unwrap();
    let hits = storage
        .get_nodes_by_source_ids(&["chunk-1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(2, hits.len(), "{hits:?}");
    let hits = storage
        .get_nodes_by_source_ids(&["chunk-10".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(1, hits.len());
    assert_eq!("Watch", hits[0]["entity_id"].as_str().unwrap());

    let edges = storage
        .get_edges_by_source_ids(&["chunk-1".to_string()], BIZ)
        .await
        .unwrap();
    assert_eq!(1, edges.len());

    // Empty chunk list never reaches the server.
    assert!(storage
        .get_nodes_by_source_ids(&[], BIZ)
        .await
        .unwrap()
        .is_empty());

    storage.drop(BIZ).await.unwrap();
    let (db, _) = raw(&ws).await;
    db.run_write_void(
        "MATCH (m:EasyRagGraphMeta {workspace: $ws}) DETACH DELETE m",
        {
            let mut p = Neo4jParams::new();
            p.insert("ws".to_string(), json!(ws));
            p
        },
    )
    .await
    .unwrap();
}

