//! In-memory KV storage.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::storage::traits::{KVStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Business-line tag stored inside each record.
fn record_biz(record: &JsonMap) -> Option<&str> {
    record.get("biz_id").and_then(|v| v.as_str())
}

/// In-memory KV storage implementation.
/// Uses a mutex-guarded HashMap for thread-safe operations.
/// Data is shared across biz lines within a workspace; `biz_id` is a
/// per-record filter rather than instance state.
pub struct InMemoryKVStorage {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    data: Mutex<HashMap<String, JsonMap>>,
}

impl InMemoryKVStorage {
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

    /// Get all keys in storage (utility for testing).
    pub fn get_all_keys(&self) -> HashSet<String> {
        self.data.lock().keys().cloned().collect()
    }

    /// Get total count of entries.
    pub fn size(&self) -> usize {
        self.data.lock().len()
    }
}

#[async_trait]
impl StorageNameSpace for InMemoryKVStorage {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    // initialize / finalize / index_done_callback: no-op for in-memory storage.
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
impl KVStorage for InMemoryKVStorage {
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
        let mut store = self.data.lock();
        for (k, mut v) in data {
            v.insert("biz_id".to_string(), Value::String(biz_id.to_string()));
            store.insert(k, v);
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
