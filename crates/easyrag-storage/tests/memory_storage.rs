//! Ported from InMemoryKVStorageTest.kt (11), InMemoryVectorStorageTest.kt (11),
//! InMemoryGraphStorageTest.kt (20) — 42 cases total.

use async_trait::async_trait;
use easyrag_core::llm::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::storage::traits::{GraphStorage, KVStorage, StorageNameSpace, VectorStorage};
use easyrag_storage::memory::{
    InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use serde_json::{json, Value};

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

fn json_map(value: Value) -> easyrag_core::JsonMap {
    match value {
        Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

// ===== InMemoryKVStorageTest (11 cases) =====

mod kv {
    use super::*;

    fn create_storage() -> InMemoryKVStorage {
        InMemoryKVStorage::new("kv", "test")
    }

    #[tokio::test]
    async fn upsert_and_get_by_id() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([(
                "key1".to_string(),
                json_map(json!({"name": "value1", "count": 42})),
            )]), BIZ)
            .await
            .unwrap();
        let result = storage.get_by_id("key1", BIZ).await.unwrap().unwrap();
        assert_eq!("value1", result["name"].as_str().unwrap());
        assert_eq!(42, result["count"].as_i64().unwrap());
    }

    #[tokio::test]
    async fn get_by_id_non_existent_returns_none() {
        let storage = create_storage();
        assert!(storage.get_by_id("missing", BIZ).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_by_ids_multiple_keys() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([
                ("a".to_string(), json_map(json!({"v": 1}))),
                ("b".to_string(), json_map(json!({"v": 2}))),
                ("c".to_string(), json_map(json!({"v": 3}))),
            ]), BIZ)
            .await
            .unwrap();
        let results = storage
            .get_by_ids(&["a".to_string(), "c".to_string()], BIZ)
            .await
            .unwrap();
        assert_eq!(2, results.len());
    }

    #[tokio::test]
    async fn get_by_ids_missing_keys_skipped() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([("a".to_string(), json_map(json!({"v": 1})))]), BIZ)
            .await
            .unwrap();
        let results = storage
            .get_by_ids(&["a".to_string(), "missing".to_string()], BIZ)
            .await
            .unwrap();
        assert_eq!(1, results.len());
    }

    #[tokio::test]
    async fn filter_keys_returns_non_existing_keys() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([(
                "existing".to_string(),
                json_map(json!({"v": 1})),
            )]), BIZ)
            .await
            .unwrap();
        let missing = storage
            .filter_keys(&HashSet::from([
                "existing".to_string(),
                "new1".to_string(),
                "new2".to_string(),
            ]), BIZ)
            .await
            .unwrap();
        assert_eq!(HashSet::from(["new1".to_string(), "new2".to_string()]), missing);
    }

    #[tokio::test]
    async fn filter_keys_all_exist_returns_empty() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([
                ("a".to_string(), json_map(json!({}))),
                ("b".to_string(), json_map(json!({}))),
            ]), BIZ)
            .await
            .unwrap();
        let missing = storage
            .filter_keys(&HashSet::from(["a".to_string(), "b".to_string()]), BIZ)
            .await
            .unwrap();
        assert!(missing.is_empty());
    }

    #[tokio::test]
    async fn delete_removes_entries() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([
                ("x".to_string(), json_map(json!({}))),
                ("y".to_string(), json_map(json!({}))),
            ]), BIZ)
            .await
            .unwrap();
        storage.delete(&["x".to_string()], BIZ).await.unwrap();
        assert!(storage.get_by_id("x", BIZ).await.unwrap().is_none());
        assert!(storage.get_by_id("y", BIZ).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn is_empty_initially_true() {
        let storage = create_storage();
        assert!(storage.is_empty(BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn is_empty_after_upsert_false() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([("k".to_string(), json_map(json!({})))]), BIZ)
            .await
            .unwrap();
        assert!(!storage.is_empty(BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn drop_clears_all() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([
                ("a".to_string(), json_map(json!({}))),
                ("b".to_string(), json_map(json!({}))),
            ]), BIZ)
            .await
            .unwrap();
        let result = storage.drop(BIZ).await.unwrap();
        assert_eq!("success", result["status"]);
        // Drop envelope guardrail: the shared helper must keep this text.
        assert_eq!("data dropped (2 records for biz default)", result["message"]);
        assert!(storage.is_empty(BIZ).await.unwrap());
        assert_eq!(0, storage.size());
    }

    #[tokio::test]
    async fn upsert_overwrites_existing() {
        let storage = create_storage();
        storage
            .upsert(HashMap::from([("k".to_string(), json_map(json!({"v": "old"})))]), BIZ)
            .await
            .unwrap();
        storage
            .upsert(HashMap::from([("k".to_string(), json_map(json!({"v": "new"})))]), BIZ)
            .await
            .unwrap();
        let result = storage.get_by_id("k", BIZ).await.unwrap().unwrap();
        assert_eq!("new", result["v"].as_str().unwrap());
    }
}

// ===== InMemoryVectorStorageTest (11 cases) =====

mod vector {
    use super::*;

    /// Simple deterministic embedding: normalize text length into vector.
    struct LengthBasedFunc {
        dim: usize,
    }

    #[async_trait]
    impl EmbeddingFunction for LengthBasedFunc {
        async fn embed(
            &self,
            texts: Vec<String>,
            _context: Option<EmbeddingContext>,
        ) -> easyrag_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| {
                    let base = text.len() as f32;
                    (0..self.dim).map(|i| base + i as f32 * 0.1).collect()
                })
                .collect())
        }
    }

    fn create_embedding_func(dim: usize) -> EmbeddingFunc {
        EmbeddingFunc::new(dim, Arc::new(LengthBasedFunc { dim })).with_supports_asymmetric(true)
    }

    fn create_storage(dim: usize) -> InMemoryVectorStorage {
        InMemoryVectorStorage::with_config(
            "vectors",
            "test",
            easyrag_core::JsonMap::new(),
            create_embedding_func(dim),
            0.0,
            HashSet::new(),
        )
    }

    #[tokio::test]
    async fn upsert_and_get_by_id() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([(
                "v1".to_string(),
                json_map(json!({"content": "hello world", "meta": "data"})),
            )]), BIZ)
            .await
            .unwrap();
        let result = storage.get_by_id("v1", BIZ).await.unwrap().unwrap();
        assert_eq!("hello world", result["content"].as_str().unwrap());
        assert_eq!("data", result["meta"].as_str().unwrap());
        assert_eq!("v1", result["id"].as_str().unwrap());
    }

    #[tokio::test]
    async fn get_by_id_non_existent_returns_none() {
        let storage = create_storage(4);
        assert!(storage.get_by_id("missing", BIZ).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn query_returns_similar_results() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([
                (
                    "v1".to_string(),
                    json_map(json!({"content": "machine learning algorithms"})),
                ),
                (
                    "v2".to_string(),
                    json_map(json!({"content": "deep learning neural networks"})),
                ),
                (
                    "v3".to_string(),
                    json_map(json!({"content": "cooking recipes for dinner"})),
                ),
            ]), BIZ)
            .await
            .unwrap();
        let results = storage
            .query("machine learning", 2, None, None, None, false, BIZ)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 2);
        // Results should have distance field.
        for r in &results {
            assert!(r.get("distance").is_some());
        }
    }

    #[tokio::test]
    async fn query_with_precomputed_embedding() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([(
                "v1".to_string(),
                json_map(json!({"content": "test content"})),
            )]), BIZ)
            .await
            .unwrap();
        let query_embedding = [1.0f32; 4];
        let results = storage
            .query("", 5, Some(&query_embedding), None, None, false, BIZ)
            .await
            .unwrap();
        // Should not throw, results depend on similarity.
        let _ = results;
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([(
                "v1".to_string(),
                json_map(json!({"content": "text"})),
            )]), BIZ)
            .await
            .unwrap();
        storage.delete(&["v1".to_string()], BIZ).await.unwrap();
        assert!(storage.get_by_id("v1", BIZ).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_entity_removes_by_name() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([(
                "entity1".to_string(),
                json_map(json!({"content": "data"})),
            )]), BIZ)
            .await
            .unwrap();
        storage.delete_entity("entity1", BIZ).await.unwrap();
        assert!(storage.get_by_id("entity1", BIZ).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn get_by_ids_multiple_ids() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([
                ("a".to_string(), json_map(json!({"content": "text a"}))),
                ("b".to_string(), json_map(json!({"content": "text b"}))),
            ]), BIZ)
            .await
            .unwrap();
        let results = storage
            .get_by_ids(&["a".to_string(), "b".to_string(), "missing".to_string()], BIZ)
            .await
            .unwrap();
        assert_eq!(2, results.len());
    }

    #[tokio::test]
    async fn get_vectors_by_ids_returns_vectors() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([(
                "v1".to_string(),
                json_map(json!({"content": "hello"})),
            )]), BIZ)
            .await
            .unwrap();
        let vectors = storage
            .get_vectors_by_ids(&["v1".to_string()], BIZ)
            .await
            .unwrap();
        assert_eq!(1, vectors.len());
        assert_eq!(4, vectors["v1"].len());
    }

    #[tokio::test]
    async fn query_multi_biz_filters_by_set_membership() {
        let storage = create_storage(4);
        storage
            .upsert(
                HashMap::from([
                    ("v1".to_string(), json_map(json!({"content": "aa"}))),
                    ("v2".to_string(), json_map(json!({"content": "bbbb"}))),
                    ("v3".to_string(), json_map(json!({"content": "cccccc"}))),
                ]),
                "b1",
            )
            .await
            .unwrap();
        storage
            .upsert(
                HashMap::from([("v2".to_string(), json_map(json!({"content": "bbbb"})))]),
                "b2",
            )
            .await
            .unwrap();
        storage
            .upsert(
                HashMap::from([("v3".to_string(), json_map(json!({"content": "cccccc"})))]),
                "b3",
            )
            .await
            .unwrap();

        // Only the requested set is scanned: b3 stays invisible, and the hits
        // come back globally ranked (descending distance) as one top_k.
        let hits = storage
            .query_multi_biz("x", 10, None, None, None, false, &["b1", "b2"])
            .await
            .unwrap();
        assert_eq!(2, hits.len());
        let ids: std::collections::HashSet<&str> = hits
            .iter()
            .filter_map(|r| r.get("id").and_then(|v| v.as_str()))
            .collect();
        assert!(ids.contains("v1"));
        assert!(ids.contains("v2"));
        assert!(!ids.contains("v3"));
        // Every hit echoes the biz line it was read from, so the data
        // endpoint's chunk projection can report per-record provenance.
        // (v2 was overwritten by the b2 upsert, so it now lives on b2.)
        for hit in &hits {
            let id = hit.get("id").and_then(|v| v.as_str()).unwrap();
            let biz = hit.get("biz_id").and_then(|v| v.as_str()).unwrap();
            let expected = match id {
                "v1" => "b1",
                "v2" => "b2",
                other => panic!("unexpected hit {other}"),
            };
            assert_eq!(expected, biz);
        }

        // A single-biz query keeps its exact-scope behavior — the multi-biz
        // predicate never leaks into it.
        let only_b1 = storage
            .query("x", 10, None, None, None, false, "b1")
            .await
            .unwrap();
        assert_eq!(1, only_b1.len());
        assert_eq!("v1", only_b1[0].get("id").and_then(|v| v.as_str()).unwrap());

        // Empty set returns nothing (callers never route an empty set here,
        // but the guard keeps it from widening into a whole-workspace scan).
        assert!(storage
            .query_multi_biz("x", 10, None, None, None, false, &[])
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn query_multi_biz_truncates_to_a_global_top_k() {
        let storage = create_storage(4);
        storage
            .upsert(
                HashMap::from([
                    ("a1".to_string(), json_map(json!({"content": "aa"}))),
                    ("a2".to_string(), json_map(json!({"content": "aaaa"}))),
                ]),
                "b1",
            )
            .await
            .unwrap();
        storage
            .upsert(
                HashMap::from([
                    ("z1".to_string(), json_map(json!({"content": "zz"}))),
                    ("z2".to_string(), json_map(json!({"content": "zzzz"}))),
                ]),
                "b2",
            )
            .await
            .unwrap();

        let hits = storage
            .query_multi_biz("x", 2, None, None, None, false, &["b1", "b2"])
            .await
            .unwrap();
        assert_eq!(2, hits.len());
        let distances: Vec<f64> = hits
            .iter()
            .filter_map(|r| r.get("distance").and_then(|v| v.as_f64()))
            .collect();
        assert_eq!(2, distances.len());
        // One globally-ordered cut across both biz lines.
        assert!(distances[0] >= distances[1]);
    }

    #[tokio::test]
    async fn drop_clears_all() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([
                ("v1".to_string(), json_map(json!({"content": "a"}))),
                ("v2".to_string(), json_map(json!({"content": "b"}))),
            ]), BIZ)
            .await
            .unwrap();
        let result = storage.drop(BIZ).await.unwrap();
        assert_eq!("success", result["status"]);
        assert_eq!("data dropped (2 vectors for biz default)", result["message"]);
        assert_eq!(0, storage.size());
    }

    #[tokio::test]
    async fn size_reflects_entry_count() {
        let storage = create_storage(4);
        assert_eq!(0, storage.size());
        storage
            .upsert(HashMap::from([(
                "a".to_string(),
                json_map(json!({"content": "x"})),
            )]), BIZ)
            .await
            .unwrap();
        assert_eq!(1, storage.size());
        storage
            .upsert(HashMap::from([(
                "b".to_string(),
                json_map(json!({"content": "y"})),
            )]), BIZ)
            .await
            .unwrap();
        assert_eq!(2, storage.size());
    }

    #[tokio::test]
    async fn delete_entity_relation_removes_matching_entries() {
        let storage = create_storage(4);
        storage
            .upsert(HashMap::from([
                (
                    "rel1".to_string(),
                    json_map(json!({"content": "relation", "source_id": "EntityA", "target_id": "EntityB"})),
                ),
                (
                    "rel2".to_string(),
                    json_map(json!({"content": "other", "source_id": "EntityC", "target_id": "EntityD"})),
                ),
            ]), BIZ)
            .await
            .unwrap();
        storage.delete_entity_relation("EntityA", BIZ).await.unwrap();
        assert!(storage.get_by_id("rel1", BIZ).await.unwrap().is_none());
        assert!(storage.get_by_id("rel2", BIZ).await.unwrap().is_some());
    }
}

// ===== InMemoryGraphStorageTest (20 cases) =====

mod graph {
    use super::*;

    fn create_storage() -> InMemoryGraphStorage {
        InMemoryGraphStorage::new("graph", "test")
    }

    #[tokio::test]
    async fn upsert_node_and_get_node() {
        let storage = create_storage();
        storage
            .upsert_node("node1", json_map(json!({"entity_type": "Person", "description": "A person"})), BIZ)
            .await
            .unwrap();
        let node = storage.get_node("node1", BIZ).await.unwrap().unwrap();
        assert_eq!("Person", node["entity_type"].as_str().unwrap());
        assert_eq!("A person", node["description"].as_str().unwrap());
    }

    #[tokio::test]
    async fn get_node_non_existent_returns_none() {
        let storage = create_storage();
        assert!(storage.get_node("nonexistent", BIZ).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn has_node_existing_returns_true() {
        let storage = create_storage();
        storage
            .upsert_node("n1", json_map(json!({"k": "v"})), BIZ)
            .await
            .unwrap();
        assert!(storage.has_node("n1", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn has_node_non_existing_returns_false() {
        let storage = create_storage();
        assert!(!storage.has_node("missing", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn upsert_node_merges_properties() {
        let storage = create_storage();
        storage
            .upsert_node("n1", json_map(json!({"a": 1})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("n1", json_map(json!({"b": 2})), BIZ)
            .await
            .unwrap();
        let node = storage.get_node("n1", BIZ).await.unwrap().unwrap();
        assert_eq!(1, node["a"].as_i64().unwrap());
        assert_eq!(2, node["b"].as_i64().unwrap());
    }

    #[tokio::test]
    async fn upsert_edge_and_get_edge() {
        let storage = create_storage();
        storage
            .upsert_node("A", json_map(json!({"type": "X"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("B", json_map(json!({"type": "Y"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("A", "B", json_map(json!({"relation": "knows"})), BIZ)
            .await
            .unwrap();

        let edge = storage.get_edge("A", "B", BIZ).await.unwrap().unwrap();
        assert_eq!("knows", edge["relation"].as_str().unwrap());
    }

    #[tokio::test]
    async fn has_edge_after_upsert_returns_true() {
        let storage = create_storage();
        storage
            .upsert_edge("X", "Y", json_map(json!({"r": "test"})), BIZ)
            .await
            .unwrap();
        assert!(storage.has_edge("X", "Y", BIZ).await.unwrap());
        assert!(storage.has_edge("Y", "X", BIZ).await.unwrap()); // undirected
    }

    #[tokio::test]
    async fn node_degree_counts_edges() {
        let storage = create_storage();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("A", "C", json_map(json!({})), BIZ)
            .await
            .unwrap();
        assert_eq!(2, storage.node_degree("A", BIZ).await.unwrap());
        assert_eq!(1, storage.node_degree("B", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn edge_degree_sum_of_node_degrees() {
        let storage = create_storage();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("A", "C", json_map(json!({})), BIZ)
            .await
            .unwrap();
        // A has degree 2, B has degree 1
        assert_eq!(3, storage.edge_degree("A", "B", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn get_node_edges_returns_connected_pairs() {
        let storage = create_storage();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("A", "C", json_map(json!({})), BIZ)
            .await
            .unwrap();
        let edges = storage.get_node_edges("A", BIZ).await.unwrap().unwrap();
        assert_eq!(2, edges.len());
    }

    #[tokio::test]
    async fn delete_node_removes_node_and_edges() {
        let storage = create_storage();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage.delete_node("A", BIZ).await.unwrap();
        assert!(!storage.has_node("A", BIZ).await.unwrap());
        assert!(!storage.has_edge("A", "B", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn remove_nodes_batch() {
        let storage = create_storage();
        for id in ["n1", "n2", "n3"] {
            storage
                .upsert_node(id, json_map(json!({})), BIZ)
                .await
                .unwrap();
        }
        storage
            .remove_nodes(&["n1".to_string(), "n2".to_string()], BIZ)
            .await
            .unwrap();
        assert!(!storage.has_node("n1", BIZ).await.unwrap());
        assert!(!storage.has_node("n2", BIZ).await.unwrap());
        assert!(storage.has_node("n3", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn remove_edges_batch() {
        let storage = create_storage();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("C", "D", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .remove_edges(&[("A".to_string(), "B".to_string())], BIZ)
            .await
            .unwrap();
        assert!(!storage.has_edge("A", "B", BIZ).await.unwrap());
        assert!(storage.has_edge("C", "D", BIZ).await.unwrap());
    }

    #[tokio::test]
    async fn get_all_labels_sorted() {
        let storage = create_storage();
        for label in ["Zebra", "Apple", "Mango"] {
            storage
                .upsert_node(label, json_map(json!({})), BIZ)
                .await
                .unwrap();
        }
        let labels = storage.get_all_labels(BIZ).await.unwrap();
        assert_eq!(vec!["Apple".to_string(), "Mango".to_string(), "Zebra".to_string()], labels);
    }

    #[tokio::test]
    async fn get_knowledge_graph_bfs() {
        let storage = create_storage();
        storage
            .upsert_node("Center", json_map(json!({"desc": "center"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("Neighbor1", json_map(json!({"desc": "n1"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("Neighbor2", json_map(json!({"desc": "n2"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("Center", "Neighbor1", json_map(json!({"rel": "r1"})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("Center", "Neighbor2", json_map(json!({"rel": "r2"})), BIZ)
            .await
            .unwrap();

        let graph = storage
            .get_knowledge_graph(Some("Center"), 2, 100, BIZ)
            .await
            .unwrap();
        assert_eq!(3, graph.nodes.len());
        assert!(graph.edges.len() >= 2);
    }

    #[tokio::test]
    async fn get_knowledge_graph_wildcard_returns_all() {
        let storage = create_storage();
        storage
            .upsert_node("A", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("B", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("A", "B", json_map(json!({})), BIZ)
            .await
            .unwrap();

        let graph = storage.get_knowledge_graph(Some("*"), 3, 100, BIZ).await.unwrap();
        assert_eq!(2, graph.nodes.len());
    }

    #[tokio::test]
    async fn drop_clears_all_data() {
        let storage = create_storage();
        storage
            .upsert_node("n1", json_map(json!({})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_edge("n1", "n2", json_map(json!({})), BIZ)
            .await
            .unwrap();
        let result = storage.drop(BIZ).await.unwrap();
        assert_eq!("success", result["status"]);
        assert_eq!("data dropped (2 nodes for biz default)", result["message"]);
        assert_eq!(0, storage.node_count());
        assert_eq!(0, storage.edge_count());
    }

    #[tokio::test]
    async fn batch_operations_get_nodes_batch() {
        let storage = create_storage();
        storage
            .upsert_node("a", json_map(json!({"v": 1})), BIZ)
            .await
            .unwrap();
        storage
            .upsert_node("b", json_map(json!({"v": 2})), BIZ)
            .await
            .unwrap();
        let result = storage
            .get_nodes_batch(&["a".to_string(), "b".to_string(), "c".to_string()], BIZ)
            .await
            .unwrap();
        assert_eq!(2, result.len());
        assert!(result.contains_key("a"));
        assert!(result.contains_key("b"));
    }

    #[tokio::test]
    async fn search_labels_fuzzy_match() {
        let storage = create_storage();
        for label in ["Machine Learning", "Deep Learning", "Cooking"] {
            storage
                .upsert_node(label, json_map(json!({})), BIZ)
                .await
                .unwrap();
        }
        let results = storage.search_labels("Learning", 100, BIZ).await.unwrap();
        assert_eq!(2, results.len());
    }

    #[tokio::test]
    async fn get_popular_labels_by_degree() {
        let storage = create_storage();
        for target in ["A", "B", "C"] {
            storage
                .upsert_edge("Hub", target, json_map(json!({})), BIZ)
                .await
                .unwrap();
        }
        storage
            .upsert_node("Lonely", json_map(json!({})), BIZ)
            .await
            .unwrap();
        let popular = storage.get_popular_labels(2, BIZ).await.unwrap();
        assert_eq!("Hub", popular[0]);
    }
}
