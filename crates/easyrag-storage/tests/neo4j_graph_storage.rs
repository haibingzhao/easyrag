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

// ---------------------------------------------------------------------------
// Phase 1.6 — Cypher text golden guardrail.
//
// Every Cypher string in neo4j.rs is hand-built by `format!` concatenation, so
// a single misplaced brace/backtick/space silently changes the query shape (and
// Neo4j returns an empty set instead of erroring). Phase 1.6 extracts a fragment
// library (node_pat/edge_pat/degree_pat/unwind_ids/json_ids); these snapshots pin
// the EXACT Cypher bytes AND the bound parameter map for every operation, across
// both scope branches (exact `$biz` vs empty-biz read-path wildcard) and the
// search_labels fulltext→CONTAINS fallback. The fragment refactor must leave
// every snapshot byte-identical ([equivalent] proof).
//
// Snapshots land in `<workspace>/testdata/golden/cypher/`. Regenerate
// deliberately:
// `INSTA_UPDATE=always cargo test -p easyrag-storage --features neo4j --test neo4j_graph_storage`.
// ---------------------------------------------------------------------------

fn cypher_golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-storage");
    root.join("testdata").join("golden").join("cypher")
}

/// Snapshot the exact Cypher bytes + bound params for every executed query.
/// Params are rendered as pretty JSON (deterministic key order) so a dropped or
/// renamed `$bind` is caught, not just a changed query string.
fn snap_cypher(name: &str, executed: &[(String, Neo4jParams)]) {
    let mut rendered = String::new();
    for (i, (q, p)) in executed.iter().enumerate() {
        rendered.push_str(&format!("--- query[{}] ---\n{}\n", i, q));
        let params_json = serde_json::to_string_pretty(&JsonValue::Object(p.clone()))
            .expect("params serialize");
        rendered.push_str(&format!("--- params[{}] ---\n{}\n", i, params_json));
    }
    insta::with_settings!({
        snapshot_path => cypher_golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, rendered);
    });
}

/// A node result row: `{"n": {"entity_id": id, ...}}` (matches `node_to_map`).
fn node_row(id: &str) -> Neo4jRow {
    let mut n = Neo4jRow::new();
    n.insert("entity_id".to_string(), JsonValue::String(id.to_string()));
    n.insert(
        "description".to_string(),
        JsonValue::String("desc".to_string()),
    );
    let mut m = Neo4jRow::new();
    m.insert("n".to_string(), JsonValue::Object(n));
    m
}

#[tokio::test]
async fn cypher_golden_initialize() {
    let (storage, fake) = create_storage("test");
    storage.initialize().await.unwrap();
    // Composite (entity_id, biz_id) index + per-workspace fulltext index.
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("initialize", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_drop() {
    let (storage, fake) = create_storage("myws");
    storage.drop(BIZ).await.unwrap();
    assert_eq!(fake.executed_queries().len(), 1);
    snap_cypher("drop", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_single_node_reads() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![bool_row("exists", true), int_row("degree", 3)]);
    storage.has_node("entity1", BIZ).await.unwrap();
    snap_cypher("has_node", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("entity1")]);
    storage.get_node("entity1", BIZ).await.unwrap();
    snap_cypher("get_node", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![int_row("degree", 5)]);
    storage.node_degree("hub", BIZ).await.unwrap();
    snap_cypher("node_degree", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_edge_degree() {
    // edge_degree delegates to node_degree twice → 2 identical queries.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![int_row("degree", 2)]);
    storage.edge_degree("A", "B", BIZ).await.unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("edge_degree", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_edge_reads() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![bool_row("exists", true)]);
    storage.has_edge("A", "B", BIZ).await.unwrap();
    snap_cypher("has_edge", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    let mut rel = Neo4jRow::new();
    rel.insert("relation".to_string(), JsonValue::String("knows".to_string()));
    let mut row = Neo4jRow::new();
    row.insert("r".to_string(), JsonValue::Object(rel));
    fake.set_results(vec![row]);
    storage.get_edge("A", "B", BIZ).await.unwrap();
    snap_cypher("get_edge", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_get_node_edges() {
    // has_node gate (exists=true) then the edge-list query → 2 queries.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![bool_row("exists", true)]);
    storage.get_node_edges("A", BIZ).await.unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("get_node_edges", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_single_writes() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_node(
            "node1",
            json_map(json!({"entity_type": "Person", "description": "A test"})),
            BIZ,
        )
        .await
        .unwrap();
    snap_cypher("upsert_node", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage
        .upsert_edge("A", "B", json_map(json!({"relation": "knows"})), BIZ)
        .await
        .unwrap();
    snap_cypher("upsert_edge", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.delete_node("victim", BIZ).await.unwrap();
    snap_cypher("delete_node", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_remove_bulk() {
    let (storage, fake) = create_storage("test");
    storage
        .remove_nodes(
            &["n1".to_string(), "n2".to_string(), "n3".to_string()],
            BIZ,
        )
        .await
        .unwrap();
    snap_cypher("remove_nodes", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage
        .remove_edges(
            &[("A".to_string(), "B".to_string())],
            BIZ,
        )
        .await
        .unwrap();
    snap_cypher("remove_edges", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_label_listing() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Apple")]);
    storage.get_all_labels(BIZ).await.unwrap();
    snap_cypher("get_all_labels", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_popular_labels_both_scopes() {
    // Exact biz scope.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Hub")]);
    storage.get_popular_labels(10, BIZ).await.unwrap();
    snap_cypher("get_popular_labels__default", &fake.executed_queries());

    // Empty biz_id = read-path wildcard (node_scope drops the biz predicate).
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Hub")]);
    storage.get_popular_labels(10, "").await.unwrap();
    snap_cypher("get_popular_labels__wildcard", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_all_nodes_edges_both_scopes() {
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage.get_all_nodes(BIZ).await.unwrap();
    snap_cypher("get_all_nodes__default", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage.get_all_nodes("").await.unwrap();
    snap_cypher("get_all_nodes__wildcard", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.get_all_edges(BIZ).await.unwrap();
    snap_cypher("get_all_edges__default", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.get_all_edges("").await.unwrap();
    snap_cypher("get_all_edges__wildcard", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_search_labels_paths() {
    // Fulltext success (exact biz): single query, biz post-filter present.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![str_row("label", "Machine Learning")]);
    storage.search_labels("Learning", 10, BIZ).await.unwrap();
    assert_eq!(fake.executed_queries().len(), 1);
    snap_cypher("search_labels__fulltext_default", &fake.executed_queries());

    // Fulltext failure → CONTAINS fallback (exact biz): 2 queries.
    let (storage, fake) = create_storage("test");
    fake.set_handler(Box::new(|query, _params| {
        if query.contains("fulltext") {
            Err(CoreError::Internal("No fulltext index".to_string()))
        } else {
            Ok(vec![str_row("label", "Machine Learning")])
        }
    }));
    storage.search_labels("Learning", 10, BIZ).await.unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("search_labels__fallback_default", &fake.executed_queries());

    // Fulltext failure → CONTAINS fallback (empty biz wildcard): 2 queries,
    // biz post-filter dropped on both, node_scope wildcard on the fallback.
    let (storage, fake) = create_storage("test");
    fake.set_handler(Box::new(|query, _params| {
        if query.contains("fulltext") {
            Err(CoreError::Internal("No fulltext index".to_string()))
        } else {
            Ok(vec![str_row("label", "Machine Learning")])
        }
    }));
    storage.search_labels("Learning", 10, "").await.unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("search_labels__fallback_wildcard", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_full_graph_both_scopes() {
    // get_knowledge_graph(None) → get_full_graph: node MATCH + edge MATCH.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage
        .get_knowledge_graph(None, 2, 10, BIZ)
        .await
        .unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("get_full_graph__default", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage
        .get_knowledge_graph(Some("*"), 2, 10, "")
        .await
        .unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("get_full_graph__wildcard", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_sub_graph_both_scopes() {
    // get_knowledge_graph(Some(label)) → get_sub_graph: variable-length path
    // query + (nodes non-empty) inter-node edge query.
    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage
        .get_knowledge_graph(Some("A"), 2, 10, BIZ)
        .await
        .unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("get_sub_graph__default", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("A")]);
    storage
        .get_knowledge_graph(Some("A"), 2, 10, "")
        .await
        .unwrap();
    assert_eq!(fake.executed_queries().len(), 2);
    snap_cypher("get_sub_graph__wildcard", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_batch_node_reads() {
    let ids = ["a".to_string(), "b".to_string()];

    let (storage, fake) = create_storage("test");
    fake.set_results(vec![node_row("a")]);
    storage.get_nodes_batch(&ids, BIZ).await.unwrap();
    snap_cypher("get_nodes_batch", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.node_degrees_batch(&ids, BIZ).await.unwrap();
    snap_cypher("node_degrees_batch", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.has_nodes_batch(&ids, BIZ).await.unwrap();
    snap_cypher("has_nodes_batch", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.get_nodes_edges_batch(&ids, BIZ).await.unwrap();
    snap_cypher("get_nodes_edges_batch", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_batch_writes() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_nodes_batch(
            &[
                ("a".to_string(), json_map(json!({"type": "X"}))),
                ("b".to_string(), json_map(json!({"type": "Y"}))),
            ],
            BIZ,
        )
        .await
        .unwrap();
    snap_cypher("upsert_nodes_batch", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage
        .upsert_edges_batch(
            &[
                (
                    "A".to_string(),
                    "B".to_string(),
                    json_map(json!({"rel": "r1"})),
                ),
                (
                    "C".to_string(),
                    "D".to_string(),
                    json_map(json!({"rel": "r2"})),
                ),
            ],
            BIZ,
        )
        .await
        .unwrap();
    snap_cypher("upsert_edges_batch", &fake.executed_queries());
}

#[tokio::test]
async fn cypher_golden_batch_edge_reads() {
    let pairs = [("A".to_string(), "B".to_string())];

    let (storage, fake) = create_storage("test");
    storage.edge_degrees_batch(&pairs, BIZ).await.unwrap();
    snap_cypher("edge_degrees_batch", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    storage.get_edges_batch(&pairs, BIZ).await.unwrap();
    snap_cypher("get_edges_batch", &fake.executed_queries());
}
