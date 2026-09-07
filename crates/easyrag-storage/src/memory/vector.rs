//! In-memory vector storage with brute-force cosine similarity

use async_trait::async_trait;
use easyrag_core::llm::{EmbeddingContext, EmbeddingFunc};
use easyrag_core::storage::traits::{StorageNameSpace, VectorStorage};
use easyrag_core::util::text::cosine_similarity;
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};

use crate::drop_result::drop_success;

struct VectorEntry {
    vector: Vec<f32>,
    metadata: JsonMap,
    biz: String,
}

/// In-memory vector storage implementation with cosine similarity search.
/// Data is shared across biz lines within a workspace; `biz_id` is a
/// per-entry filter rather than instance state.
pub struct InMemoryVectorStorage {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: EmbeddingFunc,
    cosine_better_than_threshold: f32,
    meta_fields: HashSet<String>,
    data: Mutex<HashMap<String, VectorEntry>>,
}

impl InMemoryVectorStorage {
    pub fn new(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        embedding_func: EmbeddingFunc,
    ) -> Self {
        Self::with_config(
            namespace,
            workspace,
            JsonMap::new(),
            embedding_func,
            0.2,
            HashSet::new(),
        )
    }

    pub fn with_config(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: EmbeddingFunc,
        cosine_better_than_threshold: f32,
        meta_fields: HashSet<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            workspace: workspace.into(),
            global_config,
            embedding_func,
            cosine_better_than_threshold,
            meta_fields,
            data: Mutex::new(HashMap::new()),
        }
    }

    /// Get total count of vectors stored.
    pub fn size(&self) -> usize {
        self.data.lock().len()
    }

    async fn compute_embedding(&self, text: &str) -> easyrag_core::Result<Vec<f32>> {
        let results = self
            .embedding_func
            .invoke(vec![text.to_string()], Some(EmbeddingContext::Query))
            .await?;
        Ok(results.into_iter().next().unwrap_or_default())
    }

    async fn compute_document_embedding(&self, text: &str) -> easyrag_core::Result<Vec<f32>> {
        let results = self
            .embedding_func
            .invoke(vec![text.to_string()], Some(EmbeddingContext::Document))
            .await?;
        Ok(results.into_iter().next().unwrap_or_default())
    }
}

#[async_trait]
impl StorageNameSpace for InMemoryVectorStorage {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    // initialize / finalize / index_done_callback: no-op, data immediately visible.
    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let removed = {
            let mut data = self.data.lock();
            let before = data.len();
            data.retain(|_, entry| entry.biz != biz_id);
            before - data.len()
        };
        Ok(drop_success(format!(
            "data dropped ({removed} vectors for biz {biz_id})"
        )))
    }
}

#[async_trait]
impl VectorStorage for InMemoryVectorStorage {
    fn embedding_func(&self) -> &EmbeddingFunc {
        &self.embedding_func
    }

    fn cosine_better_than_threshold(&self) -> f32 {
        self.cosine_better_than_threshold
    }

    fn meta_fields(&self) -> HashSet<String> {
        self.meta_fields.clone()
    }

    async fn query(
        &self,
        query: &str,
        top_k: usize,
        query_embedding: Option<&[f32]>,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let owned_embedding;
        let embedding: &[f32] = match query_embedding {
            Some(e) => e,
            None => {
                owned_embedding = self.compute_embedding(query).await?;
                &owned_embedding
            }
        };

        let mut results: Vec<(f32, JsonMap)> = Vec::new();
        let data = self.data.lock();
        for (id, entry) in data.iter() {
            if entry.biz != biz_id {
                continue;
            }
            // Time range filtering.
            let create_time = entry
                .metadata
                .get("create_time")
                .and_then(|v| v.as_i64());
            if let (Some(start), Some(ct)) = (time_range_start, create_time) {
                if ct < start {
                    continue;
                }
            }
            if let (Some(end), Some(ct)) = (time_range_end, create_time) {
                if ct > end {
                    continue;
                }
            }

            let similarity = cosine_similarity(embedding, &entry.vector)?;
            if similarity > self.cosine_better_than_threshold {
                let mut result = entry.metadata.clone();
                result.insert("id".to_string(), serde_json::Value::String(id.clone()));
                result.insert("distance".to_string(), serde_json::Value::from(similarity));
                results.push((similarity, result));
            }
        }
        drop(data);

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().take(top_k).map(|(_, m)| m).collect())
    }

    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        for (id, metadata) in data {
            let content = metadata
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let vector = self.compute_document_embedding(content).await?;
            self.data.lock().insert(
                id,
                VectorEntry {
                    vector,
                    metadata,
                    biz: biz_id.to_string(),
                },
            );
        }
        Ok(())
    }

    async fn delete_entity(&self, entity_name: &str, biz_id: &str) -> easyrag_core::Result<()> {
        let mut data = self.data.lock();
        if data.get(entity_name).is_some_and(|e| e.biz == biz_id) {
            data.remove(entity_name);
        }
        Ok(())
    }

    async fn delete_entity_relation(&self, entity_name: &str, biz_id: &str) -> easyrag_core::Result<()> {
        // Remove all entries whose source_id or target_id contains the entity name.
        let mut data = self.data.lock();
        let to_remove: Vec<String> = data
            .iter()
            .filter(|(_, entry)| {
                if entry.biz != biz_id {
                    return false;
                }
                let source_id = entry
                    .metadata
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let target_id = entry
                    .metadata
                    .get("target_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                source_id.contains(entity_name) || target_id.contains(entity_name)
            })
            .map(|(k, _)| k.clone())
            .collect();
        for key in to_remove {
            data.remove(&key);
        }
        Ok(())
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .data
            .lock()
            .get(id)
            .filter(|entry| entry.biz == biz_id)
            .map(|entry| {
                let mut result = entry.metadata.clone();
                result.insert("id".to_string(), serde_json::Value::String(id.to_string()));
                result
            }))
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let mut out = Vec::new();
        for id in ids {
            if let Some(doc) = self.get_by_id(id, biz_id).await? {
                out.push(doc);
            }
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        let mut data = self.data.lock();
        for id in ids {
            if data.get(id).is_some_and(|e| e.biz == biz_id) {
                data.remove(id);
            }
        }
        Ok(())
    }

    async fn get_vectors_by_ids(
        &self,
        ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, Vec<f32>>> {
        let data = self.data.lock();
        let mut out = HashMap::new();
        for id in ids {
            if let Some(entry) = data.get(id).filter(|e| e.biz == biz_id) {
                out.insert(id.clone(), entry.vector.clone());
            }
        }
        Ok(out)
    }
}
