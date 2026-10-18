//! In-memory DocStatus storage.
//! For development and testing without external dependencies.
//!
//! The KV-family behaviour lives in [`MemKvTable`]; this storage embeds one and
//! delegates, overriding `upsert` alone so status transitions merge instead of
//! replacing the record.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::model::{DocProcessingStatus, DocStatus};
use easyrag_core::storage::traits::{
    resolve_status_filter_values, DocStatusStorage, KVStorage, StorageNameSpace,
};
use easyrag_core::JsonMap;
use std::collections::{HashMap, HashSet};

use super::kv_table::{record_biz, MemKvTable};

pub struct InMemoryDocStatusStorage {
    table: MemKvTable,
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
            table: MemKvTable::new(namespace, workspace, global_config, embedding_func),
        }
    }
}

/// Memory stores every status field flat at the record root, so the record IS
/// the unwrapped field map the shared core mapper expects (postgres passes its
/// nested `content` payload instead). Keeping one mapper for both backends is
/// what prevents per-field drift — see `DocProcessingStatus::from_json`.
fn map_to_doc_status(doc: &JsonMap) -> DocProcessingStatus {
    DocProcessingStatus::from_json(doc)
}

#[async_trait]
impl StorageNameSpace for InMemoryDocStatusStorage {
    fn namespace(&self) -> &str {
        self.table.namespace()
    }

    fn workspace(&self) -> &str {
        self.table.workspace()
    }

    fn global_config(&self) -> &JsonMap {
        self.table.global_config()
    }

    // initialize / finalize / index_done_callback: trait defaults (no-op).

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        StorageNameSpace::drop(&self.table, biz_id).await
    }
}

#[async_trait]
impl KVStorage for InMemoryDocStatusStorage {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.table.embedding_func()
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.table.get_by_id(id, biz_id).await
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.table.get_by_ids(ids, biz_id).await
    }

    async fn filter_keys(
        &self,
        keys: &HashSet<String>,
        biz_id: &str,
    ) -> easyrag_core::Result<HashSet<String>> {
        self.table.filter_keys(keys, biz_id).await
    }

    /// The one method that is NOT shared with `InMemoryKVStorage`.
    /// Merge semantics: partial updates (e.g. status transitions) must not wipe
    /// fields written at insert time (file_path, content_hash, external_id, ...).
    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        self.table.upsert_merge(data, biz_id);
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.table.delete(ids, biz_id).await
    }

    async fn is_empty(&self, biz_id: &str) -> easyrag_core::Result<bool> {
        self.table.is_empty(biz_id).await
    }
}

#[async_trait]
impl DocStatusStorage for InMemoryDocStatusStorage {
    async fn get_status_counts(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, usize>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for doc in self
            .table
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
        let data = self.table.data.lock();
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
        let data = self.table.data.lock();
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
        let data = self.table.data.lock();
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
        exclude_archived: bool,
        biz_id: &str,
    ) -> easyrag_core::Result<(Vec<(String, DocProcessingStatus)>, usize)> {
        let filter_values = resolve_status_filter_values(status_filter, status_filters);
        let prefix = file_path_prefix.map(str::trim).filter(|p| !p.is_empty());
        let data = self.table.data.lock();
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
                // R2: DB-level archive exclusion.
                let archived_ok = !exclude_archived
                    || !doc
                        .get("archived")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                status_ok && prefix_ok && biz_ok && archived_ok
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
        let data = self.table.data.lock();
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
        let data = self.table.data.lock();
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
        let data = self.table.data.lock();
        Ok(data
            .iter()
            .find(|(_, doc)| {
                record_biz(doc) == Some(biz_id)
                    && doc.get("content_hash").and_then(|v| v.as_str()) == Some(content_hash)
            })
            .map(|(id, doc)| (id.clone(), doc.clone())))
    }
}
