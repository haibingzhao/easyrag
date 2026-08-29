//! In-memory DocStatus storage.
//! For development and testing without external dependencies.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::model::{DocProcessingStatus, DocStatus};
use easyrag_core::storage::traits::{
    resolve_status_filter_values, DocStatusStorage, KVStorage, StorageNameSpace,
};
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Business-line tag stored inside each record.
fn record_biz(doc: &JsonMap) -> Option<&str> {
    doc.get("biz_id").and_then(|v| v.as_str())
}

pub struct InMemoryDocStatusStorage {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    data: Mutex<HashMap<String, JsonMap>>,
}

impl InMemoryDocStatusStorage {
    pub fn new(namespace: impl Into<String>, workspace: impl Into<String>) -> Self {
        Self::with_config(namespace, workspace, JsonMap::new(), None)
    }

    pub fn with_config(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            workspace: workspace.into(),
            global_config,
            embedding_func,
            data: Mutex::new(HashMap::new()),
        }
    }
}

fn map_to_doc_status(doc: &JsonMap) -> DocProcessingStatus {
    let status_str = doc
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("pending");
    DocProcessingStatus {
        file_path: doc
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        status: DocStatus::from_value(status_str).unwrap_or(DocStatus::Pending),
        content_summary: doc
            .get("content_summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        content_length: doc
            .get("content_length")
            .and_then(|v| v.as_i64())
            .unwrap_or(0),
        created_at: doc
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        updated_at: doc
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        chunks_count: doc.get("chunks_count").and_then(|v| v.as_i64()),
        chunks_list: doc
            .get("chunks_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default(),
        error_msg: doc
            .get("error_msg")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        metadata: doc
            .get("metadata")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default(),
        multimodal_processed: doc.get("multimodal_processed").and_then(|v| v.as_bool()),
        content_hash: doc
            .get("content_hash")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        track_id: doc
            .get("track_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        has_structure: doc.get("has_structure").and_then(|v| v.as_bool()) == Some(true),
        external_id: doc
            .get("external_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        biz_id: record_biz(doc).unwrap_or_default().to_string(),
    }
}

#[async_trait]
impl StorageNameSpace for InMemoryDocStatusStorage {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    async fn index_done_callback(&self) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let removed = {
            let mut data = self.data.lock();
            let before = data.len();
            data.retain(|_, v| record_biz(v) != Some(biz_id));
            before - data.len()
        };
        let mut result = HashMap::new();
        result.insert("status".to_string(), "success".to_string());
        result.insert(
            "message".to_string(),
            format!("data dropped ({removed} records for biz {biz_id})"),
        );
        Ok(result)
    }
}

#[async_trait]
impl KVStorage for InMemoryDocStatusStorage {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .data
            .lock()
            .get(id)
            .filter(|v| record_biz(v) == Some(biz_id))
            .cloned())
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let data = self.data.lock();
        Ok(ids
            .iter()
            .filter_map(|id| {
                data.get(id)
                    .filter(|v| record_biz(v) == Some(biz_id))
                    .cloned()
            })
            .collect())
    }

    async fn filter_keys(
        &self,
        keys: &HashSet<String>,
        biz_id: &str,
    ) -> easyrag_core::Result<HashSet<String>> {
        let data = self.data.lock();
        Ok(keys
            .iter()
            .filter(|k| !data.get(*k).is_some_and(|v| record_biz(v) == Some(biz_id)))
            .cloned()
            .collect())
    }

    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        // Merge semantics: partial updates (e.g. status transitions) must not wipe
        // fields written at insert time (file_path, content_hash, external_id, ...).
        let mut store = self.data.lock();
        for (id, mut record) in data {
            record.insert("biz_id".to_string(), Value::String(biz_id.to_string()));
            store
                .entry(id)
                .and_modify(|existing| {
                    if record_biz(existing) == Some(biz_id) {
                        existing.extend(record.clone());
                    }
                })
                .or_insert(record);
        }
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        let mut data = self.data.lock();
        for id in ids {
            if data.get(id).is_some_and(|v| record_biz(v) == Some(biz_id)) {
                data.remove(id);
            }
        }
        Ok(())
    }

    async fn is_empty(&self, biz_id: &str) -> easyrag_core::Result<bool> {
        Ok(!self
            .data
            .lock()
            .values()
            .any(|v| record_biz(v) == Some(biz_id)))
    }
}

#[async_trait]
impl DocStatusStorage for InMemoryDocStatusStorage {
    async fn get_status_counts(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, usize>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for doc in self
            .data
            .lock()
            .values()
            // Empty biz_id is the read-path wildcard: count every biz line.
            .filter(|doc| biz_id.is_empty() || record_biz(doc) == Some(biz_id))
        {
            let status = doc
                .get("status")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            *counts.entry(status).or_insert(0) += 1;
        }
        Ok(counts)
    }

    async fn get_docs_by_status(
        &self,
        status: DocStatus,
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        let data = self.data.lock();
        Ok(data
            .iter()
            .filter(|(_, doc)| {
                (biz_id.is_empty() || record_biz(doc) == Some(biz_id))
                    && doc.get("status").and_then(|v| v.as_str()) == Some(status.value())
            })
            .map(|(id, doc)| (id.clone(), map_to_doc_status(doc)))
            .collect())
    }

    async fn get_docs_by_statuses(
        &self,
        statuses: &[DocStatus],
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        let status_values: HashSet<&str> = statuses.iter().map(|s| s.value()).collect();
        let data = self.data.lock();
        Ok(data
            .iter()
            .filter(|(_, doc)| {
                (biz_id.is_empty() || record_biz(doc) == Some(biz_id))
                    && doc
                        .get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| status_values.contains(s))
                        .unwrap_or(false)
            })
            .map(|(id, doc)| (id.clone(), map_to_doc_status(doc)))
            .collect())
    }

    async fn get_docs_by_track_id(
        &self,
        track_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, DocProcessingStatus>> {
        let data = self.data.lock();
        Ok(data
            .iter()
            .filter(|(_, doc)| {
                (biz_id.is_empty() || record_biz(doc) == Some(biz_id))
                    && doc.get("track_id").and_then(|v| v.as_str()) == Some(track_id)
            })
            .map(|(id, doc)| (id.clone(), map_to_doc_status(doc)))
            .collect())
    }

    async fn get_docs_paginated(
        &self,
        status_filter: Option<DocStatus>,
        status_filters: Option<&[DocStatus]>,
        page: usize,
        page_size: usize,
        _sort_field: &str,
        sort_direction: &str,
        file_path_prefix: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<(Vec<(String, DocProcessingStatus)>, usize)> {
        let filter_values = resolve_status_filter_values(status_filter, status_filters);
        let prefix = file_path_prefix.map(str::trim).filter(|p| !p.is_empty());
        let data = self.data.lock();
        let mut filtered: Vec<(String, &JsonMap)> = data
            .iter()
            .filter(|(_, doc)| {
                // Empty biz_id is the read-path wildcard: list every biz line.
                let biz_ok = biz_id.is_empty() || record_biz(doc) == Some(biz_id);
                let status_ok = filter_values.as_ref().map_or(true, |values| {
                    doc.get("status")
                        .and_then(|v| v.as_str())
                        .map(|s| values.contains(s))
                        .unwrap_or(false)
                });
                let prefix_ok = prefix.map_or(true, |p| {
                    doc.get("file_path")
                        .and_then(|v| v.as_str())
                        .map(|fp| fp.starts_with(p))
                        .unwrap_or(false)
                });
                status_ok && prefix_ok && biz_ok
            })
            .map(|(id, doc)| (id.clone(), doc))
            .collect();
        let total = filtered.len();
        // Deterministic base ordering by id; "desc" reverses it
        //
        filtered.sort_by(|a, b| a.0.cmp(&b.0));
        if sort_direction == "desc" {
            filtered.reverse();
        }
        let paged: Vec<(String, DocProcessingStatus)> = filtered
            .into_iter()
            .skip((page.saturating_sub(1)) * page_size)
            .take(page_size)
            .map(|(id, doc)| (id, map_to_doc_status(doc)))
            .collect();
        Ok((paged, total))
    }

    async fn get_all_status_counts(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, usize>> {
        self.get_status_counts(biz_id).await
    }

    async fn get_doc_by_file_path(
        &self,
        file_path: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        let data = self.data.lock();
        Ok(data
            .values()
            .find(|doc| {
                record_biz(doc) == Some(biz_id)
                    && doc.get("file_path").and_then(|v| v.as_str()) == Some(file_path)
            })
            .cloned())
    }

    async fn get_doc_by_file_basename(
        &self,
        basename: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<(String, JsonMap)>> {
        let data = self.data.lock();
        Ok(data
            .iter()
            .find(|(_, doc)| {
                record_biz(doc) == Some(biz_id)
                    && doc
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .map(|p| p.ends_with(basename))
                        .unwrap_or(false)
            })
            .map(|(id, doc)| (id.clone(), doc.clone())))
    }

    async fn get_doc_by_content_hash(
        &self,
        content_hash: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<(String, JsonMap)>> {
        let data = self.data.lock();
        Ok(data
            .iter()
            .find(|(_, doc)| {
                record_biz(doc) == Some(biz_id)
                    && doc.get("content_hash").and_then(|v| v.as_str()) == Some(content_hash)
            })
            .map(|(id, doc)| (id.clone(), doc.clone())))
    }
}
