//! Live Neo4j integration tests (require a running Neo4j instance).
//!
//! Run with:
//!   NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=secret \
//!     cargo test -p easyrag-storage --features neo4j --test neo4j_integration -- --ignored

#![cfg(feature = "neo4j")]

use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use easyrag_storage::neo4j::{Neo4jConfig, Neo4jDatabase, Neo4jGraphStorage};
use serde_json::json;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

fn json_map(value: serde_json::Value) -> JsonMap {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

async fn connect(workspace: &str) -> Neo4jGraphStorage<Neo4jDatabase> {
    let config = Neo4jConfig {
        uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".into()),
        user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".into()),
        password: std::env::var("NEO4J_PASSWORD").unwrap_or_default(),
        ..Default::default()
    };
    let db = Neo4jDatabase::new(config).await.unwrap();
    let storage = Neo4jGraphStorage::new(
        "graph",
        workspace,
        db,
    )
    .unwrap();
    storage.initialize().await.unwrap();
    storage
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
