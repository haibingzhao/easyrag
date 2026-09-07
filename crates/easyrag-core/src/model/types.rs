//! Core type definitions.

use serde::{Deserialize, Serialize};

/// A single entity extracted from text by the LLM.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub entity_name: String,
    pub entity_type: String,
    pub entity_description: String,
}

/// A single relationship between two entities extracted from text.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractedRelationship {
    pub source_entity: String,
    pub target_entity: String,
    pub relationship_keywords: String,
    pub relationship_description: String,
}

/// Structured output format for entity and relationship extraction from text.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EntityExtractionResult {
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub relationships: Vec<ExtractedRelationship>,
}

/// A node in the knowledge graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphNode {
    pub id: String,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub properties: crate::JsonMap,
}

/// An edge in the knowledge graph.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeGraphEdge {
    pub id: String,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub properties: crate::JsonMap,
}

/// A knowledge graph consisting of nodes and edges.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    #[serde(default)]
    pub nodes: Vec<KnowledgeGraphNode>,
    #[serde(default)]
    pub edges: Vec<KnowledgeGraphEdge>,
    #[serde(default)]
    pub is_truncated: bool,
}

impl KnowledgeGraphNode {
    pub fn new(
        id: impl Into<String>,
        labels: Vec<String>,
        properties: crate::JsonMap,
    ) -> Self {
        Self {
            id: id.into(),
            labels,
            properties,
        }
    }

    /// Node whose only label is its own id.
    ///
    /// The relational and in-memory backends have no label concept, so they
    /// label a node with its id; Neo4j passes the real graph labels through
    /// [`Self::new`] instead. One definition, because a graph served with empty
    /// `labels` renders as unnamed nodes in the UI.
    pub fn self_labeled(id: impl Into<String>, properties: crate::JsonMap) -> Self {
        let id = id.into();
        Self {
            labels: vec![id.clone()],
            id,
            properties,
        }
    }
}

impl KnowledgeGraphEdge {
    pub fn new(
        id: impl Into<String>,
        edge_type: Option<String>,
        source: impl Into<String>,
        target: impl Into<String>,
        properties: crate::JsonMap,
    ) -> Self {
        Self {
            id: id.into(),
            edge_type,
            source: source.into(),
            target: target.into(),
            properties,
        }
    }

    /// The composite edge id convention: `source-target`.
    ///
    /// The Postgres edges table stores exactly this string in its `id` column
    /// and the in-memory backend keys edges by the same pair, so an edge built
    /// here has to stay addressable by the id it was written with.
    pub fn composite_id(source: &str, target: &str) -> String {
        format!("{source}-{target}")
    }

    /// Edge with no type, carrying the [`Self::composite_id`].
    ///
    /// `edge_type: None` is `skip_serializing_if`, so the wire shape has no
    /// `type` field at all — what the relational and in-memory backends have
    /// always emitted. Neo4j uses [`Self::new`] with its relationship type.
    pub fn untyped(
        source: impl Into<String>,
        target: impl Into<String>,
        properties: crate::JsonMap,
    ) -> Self {
        let (source, target) = (source.into(), target.into());
        let id = Self::composite_id(&source, &target);
        Self {
            id,
            edge_type: None,
            source,
            target,
            properties,
        }
    }
}

impl KnowledgeGraph {
    /// The node ids an edge list references: `seed` first, then every endpoint
    /// in first-seen order, deduplicated.
    ///
    /// Backend-agnostic on purpose. The three traversal mechanisms (BFS over an
    /// adjacency map, a recursive CTE, a Cypher variable-length path) are NOT
    /// unified — they are not semantically equivalent — but each one ends up
    /// needing "the node set these edges touch", and that ordering decides
    /// which nodes survive a `max_nodes` cut. Deriving it once keeps the cut
    /// from differing between backends.
    pub fn expand_from_edges(seed: &str, edges: &[KnowledgeGraphEdge]) -> Vec<String> {
        let mut ids: Vec<String> =
            Vec::with_capacity(edges.len().saturating_mul(2).saturating_add(1));
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        if seen.insert(seed) {
            ids.push(seed.to_string());
        }
        for edge in edges {
            for endpoint in [edge.source.as_str(), edge.target.as_str()] {
                if seen.insert(endpoint) {
                    ids.push(endpoint.to_string());
                }
            }
        }
        ids
    }
}

#[cfg(test)]
mod kg_tests {
    use super::*;
    use serde_json::json;

    fn edge(source: &str, target: &str) -> KnowledgeGraphEdge {
        KnowledgeGraphEdge::untyped(source, target, crate::JsonMap::new())
    }

    #[test]
    fn self_labeled_node_uses_its_id_as_the_only_label() {
        let node = KnowledgeGraphNode::self_labeled("Alice", crate::JsonMap::new());
        assert_eq!("Alice", node.id);
        assert_eq!(vec!["Alice".to_string()], node.labels);
    }

    #[test]
    fn untyped_edge_carries_the_composite_id_and_no_type() {
        let e = edge("Alice", "Bob");
        assert_eq!("Alice-Bob", e.id);
        assert_eq!(
            KnowledgeGraphEdge::composite_id("Alice", "Bob"),
            "Alice-Bob".to_string()
        );
        assert!(e.edge_type.is_none());
        // `type` is skipped when absent, so the wire shape stays as before.
        assert_eq!(
            r#"{"id":"Alice-Bob","source":"Alice","target":"Bob","properties":{}}"#,
            serde_json::to_string(&e).unwrap()
        );
    }

    #[test]
    fn expand_from_edges_seeds_first_then_keeps_first_seen_order() {
        let edges = vec![edge("a", "b"), edge("b", "c"), edge("c", "a"), edge("d", "e")];
        assert_eq!(
            vec!["seed", "a", "b", "c", "d", "e"],
            KnowledgeGraph::expand_from_edges("seed", &edges)
        );
    }

    #[test]
    fn expand_from_edges_does_not_duplicate_the_seed() {
        let edges = vec![edge("seed", "a"), edge("a", "seed")];
        assert_eq!(
            vec!["seed", "a"],
            KnowledgeGraph::expand_from_edges("seed", &edges)
        );
        assert!(KnowledgeGraph::expand_from_edges("solo", &[]).eq(&["solo".to_string()]));
    }

    #[test]
    fn a_typed_edge_serialises_its_type_field() {
        let e = KnowledgeGraphEdge::new(
            "42",
            Some("DIRECTED".to_string()),
            "a",
            "b",
            serde_json::Map::from_iter([("w".to_string(), json!(1))]),
        );
        assert_eq!(
            r#"{"id":"42","type":"DIRECTED","source":"a","target":"b","properties":{"w":1}}"#,
            serde_json::to_string(&e).unwrap()
        );
    }
}
