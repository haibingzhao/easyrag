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

/// A node row carrying an explicit `source_id`, in either stored shape.
fn node_row_sourced(id: &str, source_id: JsonValue) -> Neo4jRow {
    let mut n = Neo4jRow::new();
    n.insert("entity_id".to_string(), JsonValue::String(id.to_string()));
    n.insert("source_id".to_string(), source_id);
    let mut m = Neo4jRow::new();
    m.insert("n".to_string(), JsonValue::Object(n));
    m
}

/// An edge row: `properties(r)` under `props`, plus the two endpoints
/// `edge_row_to_map` injects as `source`/`target`.
fn edge_row_sourced(src: &str, tgt: &str, source_id: JsonValue) -> Neo4jRow {
    let mut props = Neo4jRow::new();
    props.insert("source_id".to_string(), source_id);
    let mut m = Neo4jRow::new();
    m.insert("src".to_string(), JsonValue::String(src.to_string()));
    m.insert("tgt".to_string(), JsonValue::String(tgt.to_string()));
    m.insert("props".to_string(), JsonValue::Object(props));
    m
}

fn entity_ids(rows: &[JsonMap]) -> Vec<String> {
    rows.iter()
        .map(|r| r["entity_id"].as_str().unwrap_or_default().to_string())
        .collect()
}

fn edge_pairs(rows: &[JsonMap]) -> Vec<(String, String)> {
    rows.iter()
        .map(|r| {
            (
                r["source"].as_str().unwrap_or_default().to_string(),
                r["target"].as_str().unwrap_or_default().to_string(),
            )
        })
        .collect()
}

#[tokio::test]
async fn cypher_golden_initialize() {
    let (storage, fake) = create_storage("test");
    storage.initialize().await.unwrap();
    // Two schema statements, then the source_id backfill on an empty database:
    // marker read, the marker's index — deliberately no `source_id` index, see
    // the measurement recorded in `migrate_source_id_with_batch` — one empty
    // keyset page, which is already the clean round that proves there is nothing
    // to convert, and the marker MERGE. Six in total; the backfill's own sequence
    // is pinned separately by `cypher_golden_source_id_migration`.
    assert_eq!(fake.executed_queries().len(), 6);
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

/// List-valued properties must reach Bolt verbatim.
///
/// `sanitize_props` used to route every non-numeric list element through
/// `Value::to_string()`, which is `Display` = JSON serialisation, turning
/// `["chunk-1"]` into `["\"chunk-1\""]`. The corrupted list still round-trips
/// as a list, so nothing failed loudly — but element-membership predicates
/// (`ANY(c IN $ids WHERE c IN n.source_id)`) never matched again, silently
/// disabling graph cleanup on document deletion. Scalars must survive at both
/// levels; only shapes Neo4j cannot represent (nested object / nested list)
/// are stringified.
#[tokio::test]
async fn array_props_keep_scalar_elements_verbatim() {
    let (storage, fake) = create_storage("test");
    storage
        .upsert_node(
            "node1",
            json_map(json!({
                "entity_type": "Person",
                "source_id": ["chunk-1", "chunk-2"],
                "flags": [true, false],
                "weights": [1.5, 2],
                "nullable": [null],
                "nested": [{"a": 1}, ["deep"]],
            })),
            BIZ,
        )
        .await
        .unwrap();

    let props = fake
        .last_params()
        .get("props")
        .and_then(|v| v.as_object())
        .cloned()
        .expect("props bound as an object");

    assert_eq!(
        props.get("source_id"),
        Some(&json!(["chunk-1", "chunk-2"])),
        "string elements must not be double-quoted"
    );
    assert_eq!(
        props.get("flags"),
        Some(&json!([true, false])),
        "bool elements must stay bools"
    );
    assert_eq!(props.get("weights"), Some(&json!([1.5, 2])));
    assert_eq!(props.get("nullable"), Some(&json!([null])));
    // Neo4j has no nested containers: these degrade to their JSON text.
    assert_eq!(
        props.get("nested"),
        Some(&json!(["{\"a\":1}", "[\"deep\"]"])),
        "nested containers are the only elements that may be stringified"
    );

    snap_cypher("upsert_node_array_props", &fake.executed_queries());
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

/// The `source_id` membership queries are the server-side half of document
/// deletion, so both their scope widening and the two backfill states are pinned
/// here.
///
/// Pre-backfill the recorded query is deliberately the *full pull*: no portable
/// Cypher can test `string|list` in a single WHERE without raising on one of the
/// two shapes, so the membership test moves into Rust and behaves exactly like
/// the trait's default implementation. That is slower but never wrong, which is
/// the trade the marker gate exists to make explicit.
#[tokio::test]
async fn cypher_golden_source_id_filters() {
    let chunks = || vec!["chunk-1".to_string(), "chunk-2".to_string()];
    // Both stored shapes must match, and matching must be whole-element: the
    // prefix pair is the bug the array model was introduced to remove.
    let node_rows = || {
        vec![
            node_row_sourced("legacy", json!("chunk-1<SEP>chunk-9")),
            node_row_sourced("legacy-comma", json!("chunk-2,chunk-8")),
            node_row_sourced("array", json!(["chunk-2"])),
            node_row_sourced("prefix", json!(["chunk-10", "chunk-100"])),
            node_row_sourced("unsourced", JsonValue::Null),
        ]
    };
    let edge_rows = || {
        vec![
            edge_row_sourced("A", "B", json!("chunk-1")),
            edge_row_sourced("C", "D", json!(["chunk-2"])),
            edge_row_sourced("E", "F", json!(["chunk-20"])),
        ]
    };

    // Pre-backfill (default scope): one full pull, no `$chunks` parameter, and
    // the Rust filter still rejects the prefix neighbour.
    let (storage, fake) = create_storage("test");
    fake.set_results(node_rows());
    let hits = storage
        .get_nodes_by_source_ids(&chunks(), BIZ)
        .await
        .unwrap();
    assert_eq!(vec!["legacy", "legacy-comma", "array"], entity_ids(&hits));
    assert_eq!(1, fake.executed_queries().len());
    assert!(!fake.last_params().contains_key("chunks"));
    snap_cypher("get_nodes_by_source_ids__default", &fake.executed_queries());

    // Same, with the read-path biz wildcard.
    let (storage, fake) = create_storage("test");
    fake.set_results(node_rows());
    let hits = storage.get_nodes_by_source_ids(&chunks(), "").await.unwrap();
    assert_eq!(vec!["legacy", "legacy-comma", "array"], entity_ids(&hits));
    snap_cypher(
        "get_nodes_by_source_ids__wildcard",
        &fake.executed_queries(),
    );

    let (storage, fake) = create_storage("test");
    fake.set_results(edge_rows());
    let hits = storage
        .get_edges_by_source_ids(&chunks(), BIZ)
        .await
        .unwrap();
    assert_eq!(
        vec![
            ("A".to_string(), "B".to_string()),
            ("C".to_string(), "D".to_string())
        ],
        edge_pairs(&hits)
    );
    snap_cypher("get_edges_by_source_ids__default", &fake.executed_queries());

    let (storage, fake) = create_storage("test");
    fake.set_results(edge_rows());
    let hits = storage.get_edges_by_source_ids(&chunks(), "").await.unwrap();
    assert_eq!(2, hits.len());
    snap_cypher("get_edges_by_source_ids__wildcard", &fake.executed_queries());

    // Post-backfill: array arm only, the form a list-property index could be
    // resolved against. `$chunks` is now a parameter, so the filtering is the
    // server's.
    let (storage, fake) = create_storage("test");
    storage.set_source_id_arrays_only(true);
    fake.set_results(node_rows());
    storage.get_nodes_by_source_ids(&chunks(), BIZ).await.unwrap();
    assert_eq!(1, fake.executed_queries().len());
    assert_eq!(
        Some(&json!(["chunk-1", "chunk-2"])),
        fake.last_params().get("chunks")
    );
    snap_cypher(
        "get_nodes_by_source_ids__arrays_only",
        &fake.executed_queries(),
    );

    let (storage, fake) = create_storage("test");
    storage.set_source_id_arrays_only(true);
    storage.get_edges_by_source_ids(&chunks(), BIZ).await.unwrap();
    snap_cypher(
        "get_edges_by_source_ids__arrays_only",
        &fake.executed_queries(),
    );

    // An empty chunk list must not touch the server at all: "which nodes
    // reference none of these chunks" is an empty question, and answering it
    // with a label scan would be worse than the full-graph pull it replaces.
    // Notably this holds pre-backfill too, where the fallback *is* a full pull.
    let (storage, fake) = create_storage("test");
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
    assert_eq!(fake.executed_queries().len(), 0);
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

// ---------------------------------------------------------------------------
// source_id backfill (`migrate_source_id_with_batch`)
// ---------------------------------------------------------------------------

/// One row of the keyset page read: `{id, biz, sid}`.
fn migration_node_row(id: &str, biz: &str, sid: JsonValue) -> Neo4jRow {
    let mut m = Neo4jRow::new();
    m.insert("id".to_string(), JsonValue::String(id.to_string()));
    m.insert("biz".to_string(), JsonValue::String(biz.to_string()));
    m.insert("sid".to_string(), sid);
    m
}

/// One row of the relationship read: `{src, biz, tgt, tgt_biz, sid}`. Both biz
/// halves are carried because a relationship is only uniquely addressed by all
/// four endpoint columns.
fn migration_edge_row(src: &str, tgt: &str, sid: JsonValue) -> Neo4jRow {
    let mut m = Neo4jRow::new();
    m.insert("src".to_string(), JsonValue::String(src.to_string()));
    m.insert("biz".to_string(), JsonValue::String(BIZ.to_string()));
    m.insert("tgt".to_string(), JsonValue::String(tgt.to_string()));
    m.insert(
        "tgt_biz".to_string(),
        JsonValue::String(BIZ.to_string()),
    );
    m.insert("sid".to_string(), sid);
    m
}

/// The `$items` payload of the first recorded write whose Cypher contains
/// `needle`, so a test can assert on what would be SET without a live server.
fn unwind_items(executed: &[(String, Neo4jParams)], needle: &str) -> JsonValue {
    let (_, params) = executed
        .iter()
        .find(|(q, _)| q.contains(needle))
        .unwrap_or_else(|| panic!("no write matching {needle:?}"));
    params.get("items").cloned().unwrap_or(JsonValue::Null)
}

/// Parameters of every keyset page read, in order — one entry per sweep round
/// the backfill actually performed.
fn page_reads(fake: &FakeNeo4jOps) -> Vec<Neo4jParams> {
    fake.executed_queries()
        .into_iter()
        .filter(|(q, _)| q.contains("LIMIT $batch"))
        .map(|(_, p)| p)
        .collect()
}

/// Script the reads of a backfill run: the marker lookup, the keyset node pages
/// and the per-page relationship reads. `node_pages`/`edge_pages` are consumed
/// one entry per call, so a run can be driven across several pages *and* across
/// verification rounds; an exhausted list yields an empty page, which is what
/// ends a round.
fn script_migration(
    fake: &FakeNeo4jOps,
    marker: Option<&str>,
    node_pages: Vec<Vec<Neo4jRow>>,
    edge_pages: Vec<Vec<Neo4jRow>>,
) {
    let node_page = Arc::new(AtomicUsize::new(0));
    let edge_page = Arc::new(AtomicUsize::new(0));
    let pages = Arc::new((node_pages, edge_pages));
    // Owned: the handler has to be 'static, so it cannot borrow the marker.
    let marker = marker.map(str::to_string);
    fake.set_handler(Box::new(move |query, _params| {
        if query.contains("RETURN m.meta_value") {
            return Ok(match marker.as_deref() {
                Some(v) => vec![str_row("v", v)],
                None => vec![],
            });
        }
        if query.contains("LIMIT $batch") {
            let n = node_page.fetch_add(1, Ordering::SeqCst);
            return Ok(pages.0.get(n).cloned().unwrap_or_default());
        }
        if query.contains("UNWIND $nodes AS nd") {
            let n = edge_page.fetch_add(1, Ordering::SeqCst);
            return Ok(pages.1.get(n).cloned().unwrap_or_default());
        }
        Ok(vec![])
    }));
}

#[tokio::test]
async fn cypher_golden_source_id_migration() {
    // Batch of 2 over 3 nodes: two pages, the second short, which is what ends
    // the round. Node A carries both legacy delimiters, node B is already a list
    // (so it must be skipped but still contribute its key to the edge read), and
    // the one relationship uses the comma form. Round 2 comes back empty and is
    // what publishes the marker.
    let (storage, fake) = create_storage("test");
    script_migration(
        &fake,
        None,
        vec![
            vec![
                migration_node_row("A", BIZ, json!("c1<SEP>c2")),
                migration_node_row("B", BIZ, json!(["c9"])),
            ],
            vec![migration_node_row("C", "other", json!("c3"))],
            vec![],
        ],
        vec![
            vec![migration_edge_row("A", "B", json!("c1,c2"))],
            vec![],
        ],
    );
    storage.migrate_source_id_with_batch(2).await.unwrap();
    snap_cypher("migrate_source_id", &fake.executed_queries());
}

#[tokio::test]
async fn source_id_migration_converts_both_delimiters_through_the_shared_parser() {
    let (storage, fake) = create_storage("test");
    script_migration(
        &fake,
        None,
        vec![
            vec![
                migration_node_row("A", BIZ, json!("c1<SEP>c2,c3")),
                migration_node_row("B", BIZ, json!("  c4  ")),
            ],
            vec![],
        ],
        vec![vec![migration_edge_row("A", "B", json!("c1,c2"))], vec![]],
    );
    storage.migrate_source_id_with_batch(10).await.unwrap();

    let executed = fake.executed_queries();
    // The conversion is `parse_source_ids` itself: `<SEP>` and `,` both split,
    // whitespace is trimmed. A backfill that disagreed with the reader about what
    // a legacy string meant would rewrite values into a different set.
    assert_eq!(
        json!([
            {"biz": BIZ, "id": "A", "sids": ["c1", "c2", "c3"]},
            {"biz": BIZ, "id": "B", "sids": ["c4"]},
        ]),
        unwind_items(&executed, "SET n.source_id = item.sids")
    );
    assert_eq!(
        json!([{
            "biz": BIZ, "src": "A", "tgt": "B", "tgt_biz": BIZ,
            "sids": ["c1", "c2"],
        }]),
        unwind_items(&executed, "SET r.source_id = item.sids")
    );
    // Verified clean, so the flag flips and the marker is published.
    assert!(storage.source_id_arrays_only());
    assert!(executed
        .iter()
        .any(|(q, _)| q.contains("MERGE (m:EasyRagGraphMeta") && q.contains("'array'")));
}

/// The converting round cannot vouch for itself: an old pod may write a legacy
/// string to a node the keyset cursor has already passed. Only a *fresh* round
/// that finds nothing to rewrite is evidence, so the marker waits for one.
#[tokio::test]
async fn source_id_migration_publishes_the_marker_only_after_a_clean_round() {
    let (storage, fake) = create_storage("test");
    script_migration(
        &fake,
        None,
        vec![vec![migration_node_row("A", BIZ, json!("c1"))], vec![]],
        vec![vec![], vec![]],
    );
    storage.migrate_source_id_with_batch(10).await.unwrap();

    assert_eq!(2, page_reads(&fake).len(), "convert round + verify round");
    assert!(storage.source_id_arrays_only());
    assert!(fake
        .executed_queries()
        .iter()
        .any(|(q, _)| q.starts_with("MERGE (m:EasyRagGraphMeta")));
}

#[tokio::test]
async fn source_id_migration_pages_on_the_full_node_key_not_entity_id_alone() {
    let (storage, fake) = create_storage("test");
    // Same entity id on two biz lines, split across the page boundary: paging on
    // `entity_id` alone would treat the tie as one position and drop the second
    // row forever. `A@other` is the page-1 tail, so page 2 must resume *after*
    // the (entity_id, biz_id) pair rather than after `A`.
    script_migration(
        &fake,
        None,
        vec![
            vec![
                migration_node_row("A", "aaa", json!("c1")),
                migration_node_row("A", "other", json!("c2")),
            ],
            vec![migration_node_row("A", "zzz", json!("c3"))],
            vec![],
        ],
        vec![vec![], vec![], vec![]],
    );
    storage.migrate_source_id_with_batch(2).await.unwrap();

    let executed = fake.executed_queries();
    let pages = page_reads(&fake);
    assert_eq!(3, pages.len(), "two data pages plus the verifying round");
    assert_eq!(Some(&json!("")), pages[0].get("cur_id"));
    assert_eq!(Some(&json!("")), pages[0].get("cur_biz"));
    assert_eq!(Some(&json!("A")), pages[1].get("cur_id"));
    assert_eq!(Some(&json!("other")), pages[1].get("cur_biz"));
    assert_eq!(Some(&json!(2)), pages[1].get("batch"));
    // All three rows converted: none was stranded behind the tie.
    let items = unwind_items(&executed, "SET n.source_id = item.sids");
    let ids: Vec<String> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| format!("{}@{}", i["id"].as_str().unwrap(), i["biz"].as_str().unwrap()))
        .collect();
    assert_eq!(vec!["A@aaa", "A@other"], ids, "page 1 converts its own rows");
}

#[tokio::test]
async fn source_id_migration_skips_rows_that_are_already_lists() {
    let (storage, fake) = create_storage("test");
    script_migration(
        &fake,
        None,
        vec![vec![
            migration_node_row("A", BIZ, json!(["c1"])),
            migration_node_row("B", BIZ, json!(["c2", "c3"])),
        ]],
        vec![vec![migration_edge_row("A", "B", json!(["c1"]))]],
    );
    storage.migrate_source_id_with_batch(10).await.unwrap();

    let executed = fake.executed_queries();
    // Reads still happen (they are what proves there is nothing left to do), but
    // no write is issued — an interrupted re-run must not churn the graph, and a
    // workspace that was already canonical verifies in the very first round.
    assert_eq!(1, page_reads(&fake).len());
    assert!(executed.iter().any(|(q, _)| q.contains("LIMIT $batch")));
    assert!(!executed
        .iter()
        .any(|(q, _)| q.contains("SET n.source_id") || q.contains("SET r.source_id")));
    assert!(storage.source_id_arrays_only());
}

/// A node with no `source_id` at all must still be paged, because the sweep
/// reaches relationships *through* their source node: filtering the page read on
/// `source_id IS NOT NULL` would skip this node's edges too, and a legacy value
/// left on one of them would let the marker be published over data the
/// array-only query can no longer see.
#[tokio::test]
async fn source_id_migration_sweeps_relationships_of_unsourced_nodes() {
    let (storage, fake) = create_storage("test");
    script_migration(
        &fake,
        None,
        vec![vec![migration_node_row("A", BIZ, JsonValue::Null)], vec![]],
        vec![vec![migration_edge_row("A", "B", json!("c1,c2"))], vec![]],
    );
    storage.migrate_source_id_with_batch(10).await.unwrap();

    let executed = fake.executed_queries();
    let page_query = &executed
        .iter()
        .find(|(q, _)| q.contains("LIMIT $batch"))
        .expect("keyset page read")
        .0;
    assert!(!page_query.contains("IS NOT NULL"), "{page_query}");
    // The node itself is left byte-identical — inventing `[]` for it would be a
    // data change the migration has no business making — while its relationship
    // is converted.
    assert!(!executed.iter().any(|(q, _)| q.contains("SET n.source_id")));
    assert_eq!(
        json!([{
            "biz": BIZ, "src": "A", "tgt": "B", "tgt_biz": BIZ,
            "sids": ["c1", "c2"],
        }]),
        unwind_items(&executed, "SET r.source_id = item.sids")
    );
    assert!(storage.source_id_arrays_only());
}

#[tokio::test]
async fn source_id_migration_short_circuits_on_the_marker() {
    let (storage, fake) = create_storage("test");
    script_migration(&fake, Some("array"), vec![], vec![]);
    storage.migrate_source_id_with_batch(10).await.unwrap();

    // `initialize()` reruns on every workspace LRU rebuild, so the steady-state
    // cost of the whole backfill has to be this one lookup: no index statements,
    // no scan, no writes.
    let executed = fake.executed_queries();
    assert_eq!(1, executed.len());
    assert!(executed[0].0.contains("EasyRagGraphMeta"));
    assert!(storage.source_id_arrays_only());
}

#[tokio::test]
async fn source_id_migration_withholds_the_marker_when_values_remain() {
    let (storage, fake) = create_storage("test");
    // A legacy value that survives every round — what an old pod still writing
    // strings looks like from in here.
    script_migration(
        &fake,
        None,
        vec![
            vec![migration_node_row("A", BIZ, json!("c1"))],
            vec![migration_node_row("A", BIZ, json!("c1"))],
            vec![migration_node_row("A", BIZ, json!("c1"))],
            vec![migration_node_row("A", BIZ, json!("c1"))],
        ],
        vec![vec![], vec![], vec![], vec![]],
    );
    storage.migrate_source_id_with_batch(10).await.unwrap();

    // A false "verified" would publish the marker, drop the Rust fallback from
    // every membership query and silently miss deletes. Staying false is merely
    // slower.
    assert!(!storage.source_id_arrays_only());
    assert!(!fake
        .executed_queries()
        .iter()
        .any(|(q, _)| q.starts_with("MERGE (m:EasyRagGraphMeta")));
    // Exactly the round budget was spent: a value the backfill cannot get rid of
    // must not turn the sweep into an infinite loop on every `initialize()`.
    assert_eq!(3, page_reads(&fake).len());
}

#[tokio::test]
async fn source_id_migration_never_uses_the_biz_wildcard_read_scope() {
    let (storage, fake) = create_storage("test");
    script_migration(&fake, None, vec![vec![]], vec![vec![]]);
    storage.migrate_source_id_with_batch(10).await.unwrap();

    // `initialize()` has no biz context, so the backfill is workspace-wide: the
    // bare label, never `node_scope`, which would bind `$biz` and silently leave
    // other biz lines on the old format.
    assert!(fake
        .executed_queries()
        .iter()
        .filter(|(q, _)| q.contains("MATCH"))
        .all(|(_, p)| !p.contains_key("biz")));
}

#[tokio::test]
async fn drop_leaves_the_source_id_marker_in_place() {
    let (storage, fake) = create_storage("test");
    storage.drop(BIZ).await.unwrap();

    // The marker is workspace-level and stays true after the data is gone: an
    // empty workspace has no legacy values either. Clearing it would only force a
    // pointless re-backfill on the next `initialize()`.
    let query = fake.last_query();
    assert!(!query.contains("EasyRagGraphMeta"), "{query}");
    assert!(query.contains("`test`"), "{query}");
}

#[tokio::test]
async fn initialize_reports_a_failed_migration_without_failing_startup() {
    let (storage, fake) = create_storage("test");
    // The marker read itself errors: the backfill must swallow it, leave the flag
    // false and still let `initialize()` succeed.
    fake.set_handler(Box::new(|query, _params| {
        if query.contains("RETURN m.meta_value") {
            Err(CoreError::Internal("neo4j unreachable".to_string()))
        } else {
            Ok(vec![])
        }
    }));
    storage.initialize().await.unwrap();
    assert!(!storage.source_id_arrays_only());
}

