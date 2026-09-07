//! Shared in-memory record table behind the KV-family storages.
//!
//! `InMemoryKVStorage` and `InMemoryDocStatusStorage` used to carry
//! byte-identical copies of eleven `StorageNameSpace` / `KVStorage` methods.
//! Only `upsert` genuinely differs between them: plain KV replaces a record
//! wholesale, while doc status merges so that partial updates (status
//! transitions) cannot wipe fields written at insert time.
//!
//! `MemKvTable` owns the record map plus the shared behaviour. Each storage
//! embeds one table and delegates; `InMemoryDocStatusStorage` overrides
//! `upsert` alone (via `upsert_merge`).

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::storage::traits::{KVStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use crate::drop_result::drop_success;

/// Business-line tag stored inside each record. Data is shared across biz lines
/// within a workspace; `biz_id` is a per-record filter rather than instance
/// state.
pub(super) fn record_biz(record: &JsonMap) -> Option<&str> {
    record.get("biz_id").and_then(|v| v.as_str())
}

/// Mutex-guarded `id -> record` map plus the workspace/config metadata every
/// in-memory KV-family storage reports.
pub(super) struct MemKvTable {
    pub(super) namespace: String,
    pub(super) workspace: String,
    pub(super) global_config: JsonMap,
    pub(super) embedding_func: Option<EmbeddingFunc>,
    pub(super) data: Mutex<HashMap<String, JsonMap>>,
}

impl MemKvTable {
    pub(super) fn new(
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

    /// Replace-on-write upsert: the incoming record becomes the stored record.
    pub(super) fn upsert_replace(&self, data: HashMap<String, JsonMap>, biz_id: &str) {
        let mut store = self.data.lock();
        for (k, mut v) in data {
            v.insert("biz_id".to_string(), Value::String(biz_id.to_string()));
            store.insert(k, v);
        }
    }

    /// Merge-on-write upsert: partial updates (e.g. status transitions) must not
    /// wipe fields written at insert time (file_path, content_hash,
    /// external_id, ...). A record owned by a different biz line is left alone.
    pub(super) fn upsert_merge(&self, data: HashMap<String, JsonMap>, biz_id: &str) {
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
    }

    /// Get all keys in storage (utility for testing).
    pub(super) fn get_all_keys(&self) -> HashSet<String> {
        self.data.lock().keys().cloned().collect()
    }

    /// Get total count of entries.
    pub(super) fn size(&self) -> usize {
        self.data.lock().len()
    }
}

#[async_trait]
impl StorageNameSpace for MemKvTable {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    // initialize / finalize / index_done_callback: no-op for in-memory storage
    // (the trait default covers all three).

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let removed = {
            let mut data = self.data.lock();
            let before = data.len();
            data.retain(|_, v| record_biz(v) != Some(biz_id));
            before - data.len()
        };
        Ok(drop_success(format!(
            "data dropped ({removed} records for biz {biz_id})"
        )))
    }
}

#[async_trait]
impl KVStorage for MemKvTable {
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
        self.upsert_replace(data, biz_id);
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
