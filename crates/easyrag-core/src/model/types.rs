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
