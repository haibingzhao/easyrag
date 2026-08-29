//! Unit tests for Neo4jGraphStorage using a fake Neo4jOps.
//! Verifies Cypher query correctness without requiring a live Neo4j instance.
//!
//! cases for the Rust-side hardening (workspace validation, META key
//! filtering).

#![cfg(feature = "neo4j")]

use async_trait::async_trait;
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::{CoreError, JsonMap};
use easyrag_storage::neo4j::{Neo4jGraphStorage, Neo4jOps, Neo4jParams, Neo4jRow};
use serde_json::{json, Value as JsonValue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

type Handler = Box<dyn Fn(&str, &Neo4jParams) -> easyrag_core::Result<Vec<Neo4jRow>> + Send + Sync>;

/// Fake Neo4jOps that records queries and returns configurable results.
#[derive(Default)]
struct FakeNeo4jOps {
    executed: Mutex<Vec<(String, Neo4jParams)>>,
    run_results: Mutex<Vec<Neo4jRow>>,
    run_handler: Mutex<Option<Handler>>,
}

impl FakeNeo4jOps {
    fn set_results(&self, rows: Vec<Neo4jRow>) {
        *self.run_results.lock().unwrap() = rows;
    }

    fn set_handler(&self, h: Handler) {
        *self.run_handler.lock().unwrap() = Some(h);
    }

    fn executed_queries(&self) -> Vec<(String, Neo4jParams)> {
        self.executed.lock().unwrap().clone()
    }

    fn last_query(&self) -> String {
        self.executed
            .lock()
            .unwrap()
            .last()
            .map(|(q, _)| q.clone())
            .unwrap_or_default()
    }

    fn last_params(&self) -> Neo4jParams {
        self.executed
            .lock()
            .unwrap()
            .last()
            .map(|(_, p)| p.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl Neo4jOps for FakeNeo4jOps {
    async fn run(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<Vec<Neo4jRow>> {
        self.executed
            .lock()
            .unwrap()
            .push((query.to_string(), params.clone()));
        if let Some(h) = self.run_handler.lock().unwrap().as_ref() {
            return h(query, &params);
        }
        Ok(self.run_results.lock().unwrap().clone())
    }

    async fn run_write(
        &self,
        query: &str,
        params: Neo4jParams,
    ) -> easyrag_core::Result<Vec<Neo4jRow>> {
        self.executed
            .lock()
            .unwrap()
            .push((query.to_string(), params));
        Ok(self.run_results.lock().unwrap().clone())
    }

    async fn run_write_void(&self, query: &str, params: Neo4jParams) -> easyrag_core::Result<()> {
        self.executed
            .lock()
            .unwrap()
            .push((query.to_string(), params));
        Ok(())
    }
}

fn create_storage(
    workspace: &str,
) -> (Neo4jGraphStorage<Arc<FakeNeo4jOps>>, Arc<FakeNeo4jOps>) {
    let fake = Arc::new(FakeNeo4jOps::default());
    let storage = Neo4jGraphStorage::new(
        "graph",
        workspace,
        fake.clone(),
    )
    .unwrap();
    (storage, fake)
}

// --- Row construction helpers ---

fn bool_row(key: &str, value: bool) -> Neo4jRow {
    let mut m = Neo4jRow::new();
    m.insert(key.to_string(), JsonValue::Bool(value));
    m
}

fn int_row(key: &str, value: i64) -> Neo4jRow {
    let mut m = Neo4jRow::new();
    m.insert(key.to_string(), JsonValue::from(value));
    m
}

fn str_row(key: &str, value: &str) -> Neo4jRow {
    let mut m = Neo4jRow::new();
    m.insert(key.to_string(), JsonValue::String(value.to_string()));
    m
}

fn json_map(value: JsonValue) -> JsonMap {
    match value {
        JsonValue::Object(m) => m,
        _ => panic!("expected object"),
    }
}

// --- The 17 ported cases ---

#[tokio::test]
async fn initialize_creates_index() {
    let (storage, fake) = create_storage("test");
    storage.initialize().await.unwrap();

    assert!(fake
        .executed_queries()
        .iter()
        .any(|(q, _)| q.contains("CREATE INDEX")));
}

#[tokio::test]
async fn has_node_existing_returns_true() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![bool_row("exists", true)]);

    assert!(storage.has_node("entity1", BIZ).await.unwrap());
}

#[tokio::test]
async fn has_node_non_existing_returns_false() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![bool_row("exists", false)]);

    assert!(!storage.has_node("missing", BIZ).await.unwrap());
}

#[tokio::test]
async fn has_node_no_records_returns_false() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![]);
    assert!(!storage.has_node("anything", BIZ).await.unwrap());
}

#[tokio::test]
async fn node_degree_returns_count() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![int_row("degree", 5)]);

    assert_eq!(5, storage.node_degree("hub", BIZ).await.unwrap());
}

#[tokio::test]
async fn upsert_node_uses_merge_cypher() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_node(
            "node1",
            json_map(json!({"entity_type": "Person", "description": "A test"})),
        BIZ)
        .await
        .unwrap();

    assert!(fake.last_query().contains("MERGE"));
    assert!(fake.last_query().contains("entity_id"));
    assert_eq!(
        Some(&json!("node1")),
        fake.last_params().get("id")
    );
}

#[tokio::test]
async fn upsert_edge_uses_match_merge_cypher() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_edge("A", "B", json_map(json!({"relation": "knows"})), BIZ)
        .await
        .unwrap();

    assert!(fake.last_query().contains("MATCH"));
    assert!(fake.last_query().contains("MERGE"));
    assert!(fake.last_query().contains("DIRECTED"));
}

#[tokio::test]
async fn delete_node_uses_detach_delete() {
    let (storage, fake) = create_storage("test");
    storage.delete_node("victim", BIZ).await.unwrap();

    assert!(fake.last_query().contains("DETACH DELETE"));
}

#[tokio::test]
async fn remove_nodes_uses_unwind() {
    let (storage, fake) = create_storage("test");
    storage
        .remove_nodes(&["n1".to_string(), "n2".to_string(), "n3".to_string()], BIZ)
        .await
        .unwrap();

    assert!(fake.last_query().contains("UNWIND"));
    assert_eq!(
        Some(&json!(["n1", "n2", "n3"])),
        fake.last_params().get("ids")
    );
}

#[tokio::test]
async fn remove_nodes_empty_list_no_op() {
    let (storage, fake) = create_storage("test");
    storage.remove_nodes(&[], BIZ).await.unwrap();

    assert!(fake.executed_queries().is_empty());
}

#[tokio::test]
async fn get_all_labels_returns_list() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Apple"), str_row("label", "Zebra")]);

    let labels = storage.get_all_labels(BIZ).await.unwrap();
    assert_eq!(vec!["Apple".to_string(), "Zebra".to_string()], labels);
}

#[tokio::test]
async fn upsert_nodes_batch_uses_unwind() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_nodes_batch(&[
            ("a".to_string(), json_map(json!({"type": "X"}))),
            ("b".to_string(), json_map(json!({"type": "Y"}))),
        ], BIZ)
        .await
        .unwrap();

    assert!(fake.last_query().contains("UNWIND"));
    assert!(fake.last_query().contains("MERGE"));
}

#[tokio::test]
async fn upsert_edges_batch_uses_unwind() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_edges_batch(&[
            ("A".to_string(), "B".to_string(), json_map(json!({"rel": "r1"}))),
            ("C".to_string(), "D".to_string(), json_map(json!({"rel": "r2"}))),
        ], BIZ)
        .await
        .unwrap();

    assert!(fake.last_query().contains("UNWIND"));
    assert!(fake.last_query().contains("DIRECTED"));
}

#[tokio::test]
async fn drop_deletes_all_workspace_nodes() {
    let (storage, fake) = create_storage("myws");
    let result = storage.drop(BIZ).await.unwrap();

    assert_eq!("success", result["status"]);
    assert!(fake.last_query().contains("`myws`"));
    assert!(fake.last_query().contains("DETACH DELETE"));
}

#[tokio::test]
async fn search_labels_fallback_to_contains() {
    let (storage, fake) = create_storage("test");
    fake.set_handler(Box::new(|query, _params| {
        if query.contains("fulltext") {
            Err(CoreError::Internal("No fulltext index".to_string()))
        } else {
            Ok(vec![str_row("label", "Machine Learning")])
        }
    }));

    let results = storage.search_labels("Learning", 10, BIZ).await.unwrap();
    assert_eq!(vec!["Machine Learning".to_string()], results);
}

#[tokio::test]
async fn workspace_uses_escaped_label() {
    let (storage, fake) = create_storage("my-workspace");
    storage.has_node("test", BIZ).await.unwrap();

    assert!(fake.last_query().contains("`my-workspace`"));
}

#[tokio::test]
async fn get_popular_labels_uses_order_by_degree() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Hub")]);

    let popular = storage.get_popular_labels(10, BIZ).await.unwrap();
    assert!(fake.last_query().contains("ORDER BY degree DESC"));
    assert_eq!(vec!["Hub".to_string()], popular);
}

#[tokio::test]
async fn edge_degree_sum_of_node_degrees() {
    let (storage, fake) = create_storage("test");
    let call_count = Arc::new(AtomicUsize::new(0));
    let counter = call_count.clone();
    fake.set_handler(Box::new(move |_query, _params| {
        let n = counter.fetch_add(1, Ordering::SeqCst);
        // First call returns 3, second returns 2
        let degree = if n == 0 { 3 } else { 2 };
        Ok(vec![int_row("degree", degree)])
    }));

    let degree = storage.edge_degree("A", "B", BIZ).await.unwrap();
    assert_eq!(5, degree);
}

// --- Regression cases for Rust-side hardening ---

#[tokio::test]
async fn workspace_with_backtick_is_rejected() {
    let fake = Arc::new(FakeNeo4jOps::default());
    let result = Neo4jGraphStorage::new("graph", "a`) DETACH DELETE n //", fake);
    assert!(result.is_err());
}

#[tokio::test]
async fn workspace_empty_is_rejected() {
    let fake = Arc::new(FakeNeo4jOps::default());
    assert!(Neo4jGraphStorage::new("graph", "", fake).is_err());
}

#[tokio::test]
async fn upsert_node_drops_reserved_meta_keys() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_node(
            "node1",
            json_map(json!({"__kind__": "spoof", "__labels__": ["x"], "description": "kept"})),
        BIZ)
        .await
        .unwrap();

    // The props parameter must not carry reserved meta keys into Neo4j.
    let props = fake.last_params().get("props").cloned().unwrap();
    assert!(props.get("__kind__").is_none());
    assert!(props.get("__labels__").is_none());
    assert_eq!(Some(&json!("kept")), props.get("description"));
}

// --- biz_id isolation assertions ---

#[tokio::test]
async fn initialize_creates_composite_biz_index() {
    let (storage, fake) = create_storage("test");
    storage.initialize().await.unwrap();
    assert!(fake.executed_queries().iter().any(|(q, _)| {
        q.contains("CREATE INDEX") && q.contains("n.entity_id, n.biz_id")
    }));
}

#[tokio::test]
async fn node_merge_key_includes_biz_id() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_node("Alpha", json_map(json!({"entity_type": "ORG"})), BIZ)
        .await
        .unwrap();
    let (query, params) = fake
        .executed_queries()
        .into_iter()
        .find(|(q, _)| q.contains("MERGE"))
        .expect("MERGE query executed");
    assert!(query.contains("biz_id: $biz"), "MERGE key must carry biz_id: {query}");
    assert_eq!(
        Some(&json!("default")),
        params.get("biz"),
        "biz param must be bound"
    );
}

#[tokio::test]
async fn match_queries_filter_by_biz_id() {
    let (storage, fake) = create_storage("test");
    storage.upsert_edge("A", "B", json_map(json!({"relation": "knows"})), BIZ)
        .await
        .unwrap();
    let _ = storage.has_node("A", BIZ).await.unwrap();
    // Every read/write Cypher after construction must scope by biz_id,
    // either in the node pattern or the WHERE clause.
    for (query, _) in fake.executed_queries() {
        assert!(
            query.contains("biz_id"),
            "query missing biz_id scope: {query}"
        );
    }
}
