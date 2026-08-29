//! Storage trait definitions.
//!
//! Data carriers use `JsonMap` (serde_json Map) to mirror the

use crate::llm::EmbeddingFunc;
use crate::model::{DocProcessingStatus, DocStatus, KnowledgeGraph};
use crate::JsonMap;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};

/// Base storage namespace interface.
#[async_trait]
pub trait StorageNameSpace: Send + Sync {
    fn namespace(&self) -> &str;
    fn workspace(&self) -> &str;

    fn global_config(&self) -> &JsonMap;

    /// Initialize the storage.
    async fn initialize(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Finalize the storage.
    async fn finalize(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Commit the storage operations after indexing.
    async fn index_done_callback(&self) -> crate::Result<()>;

    /// Discard any not-yet-flushed buffered index ops.
    /// Backends that defer writes to index_done_callback override this.
    async fn drop_pending_index_ops(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Drop this workspace's data for one business line and clean up
    /// resources. `biz_id` is a filter, not a structural dimension.
    /// Returns operation status: {"status": "success"/"error", "message": "..."}.
    async fn drop(&self, biz_id: &str) -> crate::Result<HashMap<String, String>>;
}

/// Vector storage interface.
#[async_trait]
pub trait VectorStorage: StorageNameSpace {
    fn embedding_func(&self) -> &EmbeddingFunc;

    fn cosine_better_than_threshold(&self) -> f32 {
        0.2
    }

    fn meta_fields(&self) -> HashSet<String> {
        HashSet::new()
    }

    /// Whether this backend applies time_range filtering at the DB level.
    fn supports_time_range_filter(&self) -> bool {
        false
    }

    /// Query the vector storage and retrieve top_k results.
    /// - `query_embedding`: optional pre-computed embedding for the query
    /// - `time_range_start`: Unix timestamp (seconds), create_time >= this
    /// - `time_range_end`: Unix timestamp (seconds), create_time <= this
    /// - `biz_id`: business-line filter
    async fn query(
        &self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> crate::Result<Vec<JsonMap>>;

    /// Insert or update vectors in the storage.
    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> crate::Result<()>;

    /// Delete a single entity by its name.
    async fn delete_entity(&self, entity_name: &str, biz_id: &str) -> crate::Result<()>;

    /// Delete relations for a given entity.
    async fn delete_entity_relation(&self, entity_name: &str, biz_id: &str) -> crate::Result<()>;

    /// Get vector data by its ID.
    async fn get_by_id(&self, id: &str, biz_id: &str) -> crate::Result<Option<JsonMap>>;

    /// Get multiple vector data by their IDs.
    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> crate::Result<Vec<JsonMap>>;

    /// Delete vectors with specified IDs.
    async fn delete(&self, ids: &[String], biz_id: &str) -> crate::Result<()>;

    /// Get vectors by their IDs, returning only ID and vector data.
    async fn get_vectors_by_ids(
        &self,
        ids: &[String],
        biz_id: &str,
    ) -> crate::Result<HashMap<String, Vec<f32>>>;

    /// Generate collection/table suffix from embedding_func model name:
    /// `{sanitized_model_name}_{dim}d`.
    fn generate_collection_suffix(&self) -> Option<String> {
        let model_name = self.embedding_func().model_name.as_ref()?.trim().to_string();
        if model_name.is_empty() {
            return None;
        }
        let safe: String = model_name
            .to_lowercase()
            .chars()
            .map(|c| {
                if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        Some(format!("{}_{}d", safe, self.embedding_func().embedding_dim))
    }
}

/// Key-Value storage interface.
#[async_trait]
pub trait KVStorage: StorageNameSpace {
    fn embedding_func(&self) -> Option<&EmbeddingFunc>;

    /// Get value by id.
    async fn get_by_id(&self, id: &str, biz_id: &str) -> crate::Result<Option<JsonMap>>;

    /// Get values by ids.
    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> crate::Result<Vec<JsonMap>>;

    /// Return keys that do NOT exist in storage.
    async fn filter_keys(
        &self,
        keys: &HashSet<String>,
        biz_id: &str,
    ) -> crate::Result<HashSet<String>>;

    /// Upsert data.
    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> crate::Result<()>;

    /// Delete specific records from storage by their IDs.
    async fn delete(&self, ids: &[String], biz_id: &str) -> crate::Result<()>;

    /// Check if the storage is empty.
    async fn is_empty(&self, biz_id: &str) -> crate::Result<bool>;
}

/// Graph storage interface. All edge operations are undirected.
#[async_trait]
pub trait GraphStorage: StorageNameSpace {
    fn embedding_func(&self) -> Option<&EmbeddingFunc>;

    /// Check if a node exists in the graph.
    async fn has_node(&self, node_id: &str, biz_id: &str) -> crate::Result<bool>;

    /// Check if an edge exists between two nodes.
    async fn has_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> crate::Result<bool>;

    /// Get the degree (number of connected edges) of a node.
    async fn node_degree(&self, node_id: &str, biz_id: &str) -> crate::Result<usize>;

    /// Get the total degree of an edge (sum of degrees of source and target).
    async fn edge_degree(&self, src_id: &str, tgt_id: &str, biz_id: &str) -> crate::Result<usize>;

    /// Get node by its ID, returning node properties.
    async fn get_node(&self, node_id: &str, biz_id: &str) -> crate::Result<Option<JsonMap>>;

    /// Get edge properties between two nodes.
    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> crate::Result<Option<JsonMap>>;

    /// Get all edges connected to a node as (source, target) pairs.
    async fn get_node_edges(
        &self,
        source_node_id: &str,
        biz_id: &str,
    ) -> crate::Result<Option<Vec<(String, String)>>>;

    /// Insert or update a node in the graph.
    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        biz_id: &str,
    ) -> crate::Result<()>;

    /// Insert or update an edge in the graph.
    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        biz_id: &str,
    ) -> crate::Result<()>;

    /// Delete a node from the graph.
    async fn delete_node(&self, node_id: &str, biz_id: &str) -> crate::Result<()>;

    /// Delete multiple nodes.
    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> crate::Result<()>;

    /// Delete multiple edges.
    async fn remove_edges(&self, edge_list: &[(String, String)], biz_id: &str)
        -> crate::Result<()>;

    /// Get all labels (entity names) in the graph, sorted alphabetically.
    async fn get_all_labels(&self, biz_id: &str) -> crate::Result<Vec<String>>;

    /// Retrieve a connected subgraph starting from node_label.
    async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> crate::Result<KnowledgeGraph>;

    /// Get all nodes in the graph.
    async fn get_all_nodes(&self, biz_id: &str) -> crate::Result<Vec<JsonMap>>;

    /// Get all edges in the graph.
    async fn get_all_edges(&self, biz_id: &str) -> crate::Result<Vec<JsonMap>>;

    /// Get popular labels by node degree (most connected entities).
    async fn get_popular_labels(&self, limit: usize, biz_id: &str) -> crate::Result<Vec<String>>;

    /// Search labels with fuzzy matching.
    async fn search_labels(&self, query: &str, limit: usize, biz_id: &str)
        -> crate::Result<Vec<String>>;

    // --- Batch operations with default implementations ---

    /// Get nodes as a batch.
    async fn get_nodes_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> crate::Result<HashMap<String, JsonMap>> {
        let mut result = HashMap::new();
        for node_id in node_ids {
            if let Some(node) = self.get_node(node_id, biz_id).await? {
                result.insert(node_id.clone(), node);
            }
        }
        Ok(result)
    }

    /// Node degrees as a batch.
    async fn node_degrees_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> crate::Result<HashMap<String, usize>> {
        let mut result = HashMap::new();
        for node_id in node_ids {
            result.insert(node_id.clone(), self.node_degree(node_id, biz_id).await?);
        }
        Ok(result)
    }

    /// Edge degrees as a batch.
    async fn edge_degrees_batch(
        &self,
        edge_pairs: &[(String, String)],
        biz_id: &str,
    ) -> crate::Result<HashMap<(String, String), usize>> {
        let mut result = HashMap::new();
        for (src, tgt) in edge_pairs {
            result.insert((src.clone(), tgt.clone()), self.edge_degree(src, tgt, biz_id).await?);
        }
        Ok(result)
    }

    /// Get edges as a batch.
    async fn get_edges_batch(
        &self,
        pairs: &[(String, String)],
        biz_id: &str,
    ) -> crate::Result<HashMap<(String, String), JsonMap>> {
        let mut result = HashMap::new();
        for (src, tgt) in pairs {
            if let Some(edge) = self.get_edge(src, tgt, biz_id).await? {
                result.insert((src.clone(), tgt.clone()), edge);
            }
        }
        Ok(result)
    }

    /// Get nodes' edges as a batch.
    async fn get_nodes_edges_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> crate::Result<HashMap<String, Vec<(String, String)>>> {
        let mut result = HashMap::new();
        for node_id in node_ids {
            result.insert(
                node_id.clone(),
                self.get_node_edges(node_id, biz_id).await?.unwrap_or_default(),
            );
        }
        Ok(result)
    }

    /// Upsert multiple nodes in batch.
    async fn upsert_nodes_batch(
        &self,
        nodes: &[(String, JsonMap)],
        biz_id: &str,
    ) -> crate::Result<()> {
        for (node_id, node_data) in nodes {
            self.upsert_node(node_id, node_data.clone(), biz_id).await?;
        }
        Ok(())
    }

    /// Check existence of multiple nodes.
    async fn has_nodes_batch(
        &self,
        node_ids: &[String],
        biz_id: &str,
    ) -> crate::Result<HashSet<String>> {
        let mut result = HashSet::new();
        for node_id in node_ids {
            if self.has_node(node_id, biz_id).await? {
                result.insert(node_id.clone());
            }
        }
        Ok(result)
    }

    /// Upsert multiple edges in batch.
    async fn upsert_edges_batch(
        &self,
        edges: &[(String, String, JsonMap)],
        biz_id: &str,
    ) -> crate::Result<()> {
        for (src, tgt, data) in edges {
            self.upsert_edge(src, tgt, data.clone(), biz_id).await?;
        }
        Ok(())
    }
}

/// Document status storage interface.
#[async_trait]
pub trait DocStatusStorage: KVStorage {
    /// Get counts of documents in each status.
    async fn get_status_counts(&self, biz_id: &str) -> crate::Result<HashMap<String, usize>>;

    /// Get all documents with a specific status.
    async fn get_docs_by_status(
        &self,
        status: DocStatus,
        biz_id: &str,
    ) -> crate::Result<HashMap<String, DocProcessingStatus>>;

    /// Get all documents matching any of the given statuses.
    async fn get_docs_by_statuses(
        &self,
        statuses: &[DocStatus],
        biz_id: &str,
    ) -> crate::Result<HashMap<String, DocProcessingStatus>>;

    /// Get all documents with a specific track_id.
    async fn get_docs_by_track_id(
        &self,
        track_id: &str,
        biz_id: &str,
    ) -> crate::Result<HashMap<String, DocProcessingStatus>>;

    /// Get documents with pagination support.
    /// Returns (list of (doc_id, DocProcessingStatus), total_count).
    /// `file_path_prefix`: optional server-side prefix filter on file_path (R5).
    async fn get_docs_paginated(
        &self,
        status_filter: Option<DocStatus>,
        status_filters: Option<&[DocStatus]>,
        page: usize,
        page_size: usize,
        sort_field: &str,
        sort_direction: &str,
        file_path_prefix: Option<&str>,
        biz_id: &str,
    ) -> crate::Result<(Vec<(String, DocProcessingStatus)>, usize)>;

    /// Get counts of documents in each status for all documents.
    async fn get_all_status_counts(&self, biz_id: &str) -> crate::Result<HashMap<String, usize>>;

    /// Get document by file path.
    async fn get_doc_by_file_path(
        &self,
        file_path: &str,
        biz_id: &str,
    ) -> crate::Result<Option<JsonMap>>;

    /// Get document by canonical file basename. Returns (doc_id, doc_data).
    async fn get_doc_by_file_basename(
        &self,
        basename: &str,
        biz_id: &str,
    ) -> crate::Result<Option<(String, JsonMap)>>;

    /// Get document by content_hash field. Returns (doc_id, doc_data).
    async fn get_doc_by_content_hash(
        &self,
        content_hash: &str,
        biz_id: &str,
    ) -> crate::Result<Option<(String, JsonMap)>>;

    /// Atomically claim a document for single-document indexing (R6).
    /// Transitions pending/failed -> processing; returns false if the document
    /// does not exist or is not in a claimable state (already processing/processed).
    ///
    /// Default implementation is a read-modify-write suitable for single-process
    /// backends (InMemory). Database backends should override with a conditional
    /// UPDATE for true atomicity.
    async fn try_claim_for_processing(&self, doc_id: &str, biz_id: &str) -> crate::Result<bool> {
        let doc = match self.get_by_id(doc_id, biz_id).await? {
            Some(doc) => doc,
            None => return Ok(false),
        };
        let status = doc
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or(DocStatus::Pending.value());
        if status != DocStatus::Pending.value() && status != DocStatus::Failed.value() {
            return Ok(false);
        }
        let mut updated = doc;
        updated.insert(
            "status".to_string(),
            serde_json::Value::String(DocStatus::Processing.value().to_string()),
        );
        updated.insert(
            "updated_at".to_string(),
            serde_json::Value::String(
                chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
            ),
        );
        let mut map = HashMap::new();
        map.insert(doc_id.to_string(), updated);
        self.upsert(map, biz_id).await?;
        Ok(true)
    }
}

/// A bound set of storage instances for one workspace (R2).
/// Default deployments use a single bundle backed by server-level config;
/// request-level workspace routing resolves a cached bundle per workspace.
/// biz_id is NOT a structural dimension: it flows as a filter parameter on
/// each storage call.
#[derive(Clone)]
pub struct StorageBundle {
    pub workspace: String,
    pub kv_storage: std::sync::Arc<dyn KVStorage>,
    pub vector_storage: std::sync::Arc<dyn VectorStorage>,
    pub graph_storage: std::sync::Arc<dyn GraphStorage>,
    pub doc_status_storage: std::sync::Arc<dyn DocStatusStorage>,
}

/// Normalize single- and multi-status filters into comparable values
///
pub fn resolve_status_filter_values(
    status_filter: Option<DocStatus>,
    status_filters: Option<&[DocStatus]>,
) -> Option<HashSet<String>> {
    if let Some(filters) = status_filters {
        if !filters.is_empty() {
            return Some(filters.iter().map(|s| s.value().to_string()).collect());
        }
    }
    if let Some(filter) = status_filter {
        return Some([filter.value().to_string()].into_iter().collect());
    }
    None
}
