//! Orchestration tests for [`cleanup_graph_by_chunks`] — the single copy of the
//! document-deletion graph trim that both `easyrag-server`'s cascade delete and
//! the [`EasyRAG`] facade delete now call.
//!
//! What is pinned here:
//! - the three outcomes (delete / trim / keep) and their exact counts;
//! - an edge is removed when *either* its own `source_id` references a deleted
//!   chunk *or* one of its endpoints is deleted, and an edge hit by both
//!   conditions is still one deletion;
//! - a deleted entity is also deleted from the vector store;
//! - legacy `<SEP>`/`,`-joined `source_id` strings are trimmed, not lost;
//! - the trimmed list keeps the stored order;
//! - an empty chunk set is a no-op.
//!
//! `DefaultFilterGraph` below is the reason this file exists separately from
//! `easyrag-storage`'s cross-backend tests: it deliberately does **not**
//! override `get_nodes_by_source_ids`/`get_edges_by_source_ids`, so cleanup
//! runs against the trait's default implementations — exactly what a future
//! backend gets for free. Filtering correctness per backend is covered by BC-13
//! in `crates/easyrag-storage/tests/cross_backend_consistency.rs`.

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::model::types::KnowledgeGraph;
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace, VectorStorage};
use easyrag_core::JsonMap;
use easyrag_pipeline::cleanup_graph_by_chunks;
use easyrag_storage::memory::{InMemoryGraphStorage, InMemoryVectorStorage};
use serde_json::{json, Value as JsonValue};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

const WS: &str = "cleanup_ws";

struct ConstFunc;

#[async_trait]
impl EmbeddingFunction for ConstFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1f32, 0f32, 0f32, 0f32]).collect())
    }
}

fn embedding() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ConstFunc))
}

/// A graph backend that overrides nothing: the two filtered reads fall through
/// to the trait defaults (full pull + in-Rust membership test), while every
/// other call forwards to a real in-memory graph so the deletes and trims under
/// test actually happen.
struct DefaultFilterGraph(InMemoryGraphStorage);

#[async_trait]
impl GraphStorage for DefaultFilterGraph {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.0.embedding_func()
    }
    async fn has_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<bool> {
        self.0.has_node(node_id, biz_id).await
    }
    async fn has_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<bool> {
        self.0.has_edge(source_node_id, target_node_id, biz_id).await
    }
    async fn node_degree(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        self.0.node_degree(node_id, biz_id).await
    }
    async fn edge_degree(
        &self,
        src_id: &str,
        tgt_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<usize> {
        self.0.edge_degree(src_id, tgt_id, biz_id).await
    }
    async fn get_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.0.get_node(node_id, biz_id).await
    }
    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        self.0.get_edge(source_node_id, target_node_id, biz_id).await
    }
    async fn get_node_edges(
        &self,
        source_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<Vec<(String, String)>>> {
        self.0.get_node_edges(source_node_id, biz_id).await
    }
    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.0.upsert_node(node_id, node_data, biz_id).await
    }
    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.0.upsert_edge(source_node_id, target_node_id, edge_data, biz_id).await
    }
    async fn delete_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<()> {
        self.0.delete_node(node_id, biz_id).await
    }
    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.0.remove_nodes(node_ids, biz_id).await
    }
    async fn remove_edges(
        &self,
        edge_list: &[(String, String)],
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.0.remove_edges(edge_list, biz_id).await
    }
    async fn get_all_labels(&self, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        self.0.get_all_labels(biz_id).await
    }
    async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        self.0
            .get_knowledge_graph(node_label, max_depth, max_nodes, biz_id)
            .await
    }
    async fn get_all_nodes(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.0.get_all_nodes(biz_id).await
    }
    async fn get_all_edges(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.0.get_all_edges(biz_id).await
    }
    async fn get_popular_labels(
        &self,
        limit: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        self.0.get_popular_labels(limit, biz_id).await
    }
    async fn search_labels(
        &self,
        query: &str,
        limit: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        self.0.search_labels(query, limit, biz_id).await
    }
}

#[async_trait]
impl StorageNameSpace for DefaultFilterGraph {
    fn namespace(&self) -> &str {
        self.0.namespace()
    }
    fn workspace(&self) -> &str {
        self.0.workspace()
    }
    fn global_config(&self) -> &JsonMap {
        self.0.global_config()
    }
    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        self.0.drop(biz_id).await
    }
}

fn graph() -> InMemoryGraphStorage {
    InMemoryGraphStorage::new("graph", WS)
}

fn vector() -> InMemoryVectorStorage {
    InMemoryVectorStorage::new("vdb", WS, embedding())
}

fn chunks(ids: &[&str]) -> HashSet<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

async fn node(graph: &InMemoryGraphStorage, id: &str, source_id: JsonValue) {
    graph
        .upsert_node(
            id,
            json_map(json!({"entity_type": "Person", "source_id": source_id})),
            BIZ,
        )
        .await
        .unwrap();
}

async fn edge(graph: &InMemoryGraphStorage, src: &str, tgt: &str, source_id: JsonValue) {
    graph
        .upsert_edge(
            src,
            tgt,
            json_map(json!({"keywords": "knows", "source_id": source_id})),
            BIZ,
        )
        .await
        .unwrap();
}

fn json_map(value: JsonValue) -> JsonMap {
    match value {
        JsonValue::Object(m) => m,
        other => panic!("expected object, got {other}"),
    }
}

/// Seed the fixture every test below reasons about. `c1` and `c5` are the
/// chunks being deleted.
///
/// | node          | source_id       | outcome                        |
/// |---------------|-----------------|--------------------------------|
/// | `gone`        | `["c1"]`        | deleted (nothing left)         |
/// | `legacy_gone` | `"c1<SEP>c5"`   | deleted (both delimiters)      |
/// | `trimmed`     | `["c1","c2","c3"]` | trimmed to `["c2","c3"]`    |
/// | `legacy_trim` | `"c1,c2"`       | trimmed to `["c2"]`            |
/// | `kept`        | `["c9"]`        | untouched                      |
/// | `kept2`       | `["c9"]`        | untouched                      |
/// | `other_biz`   | `["c1"]`        | untouched (different biz line) |
async fn seed(graph: &InMemoryGraphStorage) {
    node(graph, "gone", json!(["c1"])).await;
    node(graph, "legacy_gone", json!("c1<SEP>c5")).await;
    node(graph, "trimmed", json!(["c1", "c2", "c3"])).await;
    node(graph, "legacy_trim", json!("c1,c2")).await;
    node(graph, "kept", json!(["c9"])).await;
    node(graph, "kept2", json!(["c9"])).await;
    graph
        .upsert_node(
            "other_biz",
            json_map(json!({"entity_type": "Person", "source_id": ["c1"]})),
            "other",
        )
        .await
        .unwrap();

    // (a) own `source_id` references a deleted chunk, both endpoints survive.
    edge(graph, "kept", "trimmed", json!(["c1"])).await;
    // (a) *and* (b): references `c1` and its endpoint `gone` is deleted.
    edge(graph, "gone", "trimmed", json!(["c1"])).await;
    // (b) only: provenanced elsewhere, but endpoint `gone` is deleted.
    edge(graph, "gone", "kept", json!(["c9"])).await;
    // neither: survives.
    edge(graph, "kept", "kept2", json!(["c9"])).await;
}

fn ids(nodes: &[JsonMap]) -> Vec<String> {
    let mut out: Vec<String> = nodes
        .iter()
        .filter_map(|n| n.get("id").and_then(|v| v.as_str()).map(str::to_string))
        .collect();
    out.sort();
    out
}

fn source_id_of(nodes: &[JsonMap], id: &str) -> JsonValue {
    nodes
        .iter()
        .find(|n| n.get("id").and_then(|v| v.as_str()) == Some(id))
        .and_then(|n| n.get("source_id"))
        .cloned()
        .unwrap_or(JsonValue::Null)
}

fn edge_pairs(edges: &[JsonMap]) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = edges
        .iter()
        .filter_map(|e| {
            let s = e.get("source").and_then(|v| v.as_str())?.to_string();
            let t = e.get("target").and_then(|v| v.as_str())?.to_string();
            Some((s, t))
        })
        .collect();
    out.sort();
    out
}

#[tokio::test]
async fn cleanup_deletes_trims_and_keeps_the_three_states() {
    let graph = graph();
    let vector = vector();
    seed(&graph).await;

    let report = cleanup_graph_by_chunks(&graph, &vector, &chunks(&["c1", "c5"]), BIZ)
        .await
        .unwrap();

    // `gone` + `legacy_gone` deleted; three edges hit (see `seed`); `trimmed` +
    // `legacy_trim` survive with fewer chunks.
    assert_eq!(report.deleted_nodes, 2, "{report:?}");
    assert_eq!(report.deleted_edges, 3, "{report:?}");
    assert_eq!(report.trimmed_nodes, 2, "{report:?}");

    let nodes = graph.get_all_nodes(BIZ).await.unwrap();
    assert_eq!(
        ids(&nodes),
        ["kept", "kept2", "legacy_trim", "trimmed"],
        "deleted nodes must be gone, unrelated and other-biz nodes must stay"
    );

    // Trimmed in place, canonical array form, stored order preserved.
    assert_eq!(source_id_of(&nodes, "trimmed"), json!(["c2", "c3"]));
    assert_eq!(source_id_of(&nodes, "legacy_trim"), json!(["c2"]));

    // The survivor's own edges: only the untouched `kept`-`kept2` pair.
    let edges = graph.get_all_edges(BIZ).await.unwrap();
    let pairs = edge_pairs(&edges);
    assert_eq!(pairs.len(), 1, "{pairs:?}");
    assert!(
        pairs[0].0 == "kept" && pairs[0].1 == "kept2" || pairs[0].0 == "kept2" && pairs[0].1 == "kept",
        "the edge provenanced to a live chunk must survive: {pairs:?}"
    );

    // Scope isolation: the other biz line's node still references `c1`.
    let other = graph.get_all_nodes("other").await.unwrap();
    assert_eq!(ids(&other), ["other_biz"]);
    assert_eq!(source_id_of(&other, "other_biz"), json!(["c1"]));
}

#[tokio::test]
async fn cleanup_deletes_entity_vectors_with_the_graph_nodes() {
    let graph = graph();
    let vector = vector();
    seed(&graph).await;

    // Entity descriptions are embedded separately from the graph, so the
    // deleted entities have vector rows that must go too.
    let mut rows: HashMap<String, JsonMap> = HashMap::new();
    for id in ["gone", "trimmed", "kept"] {
        rows.insert(id.to_string(), json_map(json!({"content": format!("{id} desc")})));
    }
    vector.upsert(rows, BIZ).await.unwrap();
    assert_eq!(vector.size(), 3);

    cleanup_graph_by_chunks(&graph, &vector, &chunks(&["c1"]), BIZ)
        .await
        .unwrap();

    // Only the deleted entity's vector is dropped; trimmed and kept stay
    // retrievable.
    assert_eq!(vector.size(), 2);
    assert!(vector.get_by_ids(&["gone".to_string()], BIZ).await.unwrap().is_empty());
    assert_eq!(
        vector
            .get_by_ids(&["trimmed".to_string(), "kept".to_string()], BIZ)
            .await
            .unwrap()
            .len(),
        2
    );
}

#[tokio::test]
async fn cleanup_without_chunks_changes_nothing() {
    let graph = graph();
    let vector = vector();
    seed(&graph).await;
    let before_nodes = ids(&graph.get_all_nodes(BIZ).await.unwrap());
    let before_edges = edge_pairs(&graph.get_all_edges(BIZ).await.unwrap());

    let report = cleanup_graph_by_chunks(&graph, &vector, &HashSet::new(), BIZ)
        .await
        .unwrap();

    assert_eq!(report.deleted_nodes, 0);
    assert_eq!(report.deleted_edges, 0);
    assert_eq!(report.trimmed_nodes, 0);
    assert_eq!(ids(&graph.get_all_nodes(BIZ).await.unwrap()), before_nodes);
    assert_eq!(
        edge_pairs(&graph.get_all_edges(BIZ).await.unwrap()),
        before_edges
    );
}

#[tokio::test]
async fn cleanup_on_an_unrelated_chunk_changes_nothing() {
    let graph = graph();
    let vector = vector();
    seed(&graph).await;

    let report = cleanup_graph_by_chunks(&graph, &vector, &chunks(&["c999"]), BIZ)
        .await
        .unwrap();

    assert_eq!(report.deleted_nodes + report.deleted_edges + report.trimmed_nodes, 0);
    assert_eq!(graph.get_all_nodes(BIZ).await.unwrap().len(), 6);
}

/// The whole point of the trait defaults: a backend that overrides nothing must
/// still clean up correctly, just more slowly.
#[tokio::test]
async fn cleanup_works_against_the_trait_default_filters() {
    let vector = vector();

    // Two independent copies of the same fixture: one graph runs the in-memory
    // override of the filtered reads, the other runs the trait defaults.
    let overridden = graph();
    seed(&overridden).await;
    let wrapper = DefaultFilterGraph(graph());
    seed(&wrapper.0).await;

    let expected = cleanup_graph_by_chunks(&overridden, &vector, &chunks(&["c1", "c5"]), BIZ)
        .await
        .unwrap();
    let actual = cleanup_graph_by_chunks(&wrapper, &vector, &chunks(&["c1", "c5"]), BIZ)
        .await
        .unwrap();

    assert_eq!(actual, expected, "the defaults must agree with the override");
    assert_eq!(actual.deleted_nodes, 2);
    assert_eq!(actual.deleted_edges, 3);
    assert_eq!(actual.trimmed_nodes, 2);

    let nodes = wrapper.get_all_nodes(BIZ).await.unwrap();
    assert_eq!(ids(&nodes), ["kept", "kept2", "legacy_trim", "trimmed"]);
    assert_eq!(source_id_of(&nodes, "trimmed"), json!(["c2", "c3"]));
}

/// `chunk_1` must never take `chunk_10` down with it — the substring collision
/// that made a server-side `CONTAINS` filter unusable.
#[tokio::test]
async fn cleanup_does_not_confuse_prefixed_chunk_ids() {
    let graph = graph();
    let vector = vector();
    node(&graph, "one", json!(["chunk_1"])).await;
    node(&graph, "ten", json!(["chunk_10", "chunk_100"])).await;
    edge(&graph, "one", "ten", json!(["chunk_10"])).await;

    let report = cleanup_graph_by_chunks(&graph, &vector, &chunks(&["chunk_1"]), BIZ)
        .await
        .unwrap();

    assert_eq!(report.deleted_nodes, 1);
    assert_eq!(report.deleted_edges, 1, "the edge dies on its endpoint, not on a prefix match");
    let nodes = graph.get_all_nodes(BIZ).await.unwrap();
    assert_eq!(ids(&nodes), ["ten"]);
    assert_eq!(source_id_of(&nodes, "ten"), json!(["chunk_10", "chunk_100"]));
}
