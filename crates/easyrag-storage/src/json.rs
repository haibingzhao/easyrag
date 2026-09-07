//! JSON file-based KV storage.
//! Uses serde serialization + atomic file writes (write temp file, rename).

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::storage::traits::{KVStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::drop_result::drop_success;

/// JSON file-based KV storage implementation.
///
/// One file per workspace; `biz_id` is a per-record field used as a
/// read/write filter rather than a structural partition.
pub struct JsonKVStorage {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    storage_dir: String,
    data: Mutex<serde_json::Map<String, Value>>,
}

impl JsonKVStorage {
    pub fn new(namespace: impl Into<String>, workspace: impl Into<String>) -> Self {
        Self::with_storage_dir(namespace, workspace, "./rag_storage")
    }

    pub fn with_storage_dir(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        storage_dir: impl Into<String>,
    ) -> Self {
        Self::with_config(namespace, workspace, JsonMap::new(), None, storage_dir)
    }

    pub fn with_config(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
        storage_dir: impl Into<String>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            workspace: workspace.into(),
            global_config,
            embedding_func,
            storage_dir: storage_dir.into(),
            data: Mutex::new(serde_json::Map::new()),
        }
    }

    fn storage_file(&self) -> PathBuf {
        Path::new(&self.storage_dir)
            .join(&self.workspace)
            .join(format!("kv_store_{}.json", self.namespace))
    }

    fn persist_locked(&self) {
        let file = self.storage_file();
        let json = Value::Object(self.data.lock().clone());
        if let Err(e) = write_atomic(&file, &json) {
            tracing::error!("Failed to persist JSON storage: {e}");
        } else {
            tracing::debug!(
                "Persisted JSON storage to {}",
                file.display()
            );
        }
    }
}

/// Atomic write: write to a temp file then rename over the target.
fn write_atomic(file: &Path, json: &Value) -> std::io::Result<()> {
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temp_file = file.with_extension("json.tmp");
    let pretty = serde_json::to_string_pretty(json)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&temp_file, pretty)?;
    std::fs::rename(&temp_file, file)?;
    Ok(())
}

/// Business-line tag stored inside each record.
fn record_biz(value: &Value) -> Option<&str> {
    value.get("biz_id").and_then(|v| v.as_str())
}

#[async_trait]
impl StorageNameSpace for JsonKVStorage {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    async fn initialize(&self) -> easyrag_core::Result<()> {
        let file = self.storage_file();
        if file.exists() {
            match std::fs::read_to_string(&file)
                .ok()
                .and_then(|content| serde_json::from_str::<Value>(&content).ok())
                .and_then(|value| match value {
                    Value::Object(map) => Some(map),
                    _ => None,
                }) {
                Some(loaded) => {
                    let count = loaded.len();
                    let mut data = self.data.lock();
                    for (k, v) in loaded {
                        data.insert(k, v);
                    }
                    tracing::info!("Loaded {count} entries from {}", file.display());
                }
                None => {
                    tracing::warn!(
                        "Failed to load JSON storage from {}",
                        file.display()
                    );
                }
            }
        }
        Ok(())
    }

    async fn finalize(&self) -> easyrag_core::Result<()> {
        self.persist_locked();
        Ok(())
    }

    async fn index_done_callback(&self) -> easyrag_core::Result<()> {
        self.persist_locked();
        Ok(())
    }

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        // Shared per-workspace file: only remove this biz slice, then persist.
        let removed = {
            let mut store = self.data.lock();
            let before = store.len();
            store.retain(|_, v| record_biz(v) != Some(biz_id));
            before - store.len()
        };
        self.persist_locked();
        Ok(drop_success(format!(
            "data dropped ({removed} records for biz {biz_id})"
        )))
    }
}

#[async_trait]
impl KVStorage for JsonKVStorage {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn get_by_id(&self, id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .data
            .lock()
            .get(id)
            .filter(|v| record_biz(v) == Some(biz_id))
            .and_then(|v| v.as_object())
            .cloned())
    }

    async fn get_by_ids(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let data = self.data.lock();
        Ok(ids
            .iter()
            .filter_map(|id| {
                data.get(id)
                    .filter(|v| record_biz(v) == Some(biz_id))
                    .and_then(|v| v.as_object())
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
            store.insert(k, Value::Object(v));
        }
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        let mut store = self.data.lock();
        for id in ids {
            if store.get(id).is_some_and(|v| record_biz(v) == Some(biz_id)) {
                store.remove(id);
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
