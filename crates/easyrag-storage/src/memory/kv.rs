//! In-memory KV storage.
//!
//! A thin wrapper over [`MemKvTable`]: the shared table owns the record map and
//! every `StorageNameSpace` / `KVStorage` behaviour, so this file only keeps the
//! public constructors and the testing utilities.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::storage::traits::{KVStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use std::collections::{HashMap, HashSet};

use super::kv_table::MemKvTable;

/// In-memory KV storage implementation.
/// Uses a mutex-guarded HashMap for thread-safe operations.
/// Data is shared across biz lines within a workspace; `biz_id` is a
/// per-record filter rather than instance state.
pub struct InMemoryKVStorage {
    table: MemKvTable,
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
            table: MemKvTable::new(namespace, workspace, global_config, embedding_func),
        }
    }

    /// Get all keys in storage (utility for testing).
    pub fn get_all_keys(&self) -> HashSet<String> {
        self.table.get_all_keys()
    }

    /// Get total count of entries.
    pub fn size(&self) -> usize {
        self.table.size()
    }
}

#[async_trait]
impl StorageNameSpace for InMemoryKVStorage {
    fn namespace(&self) -> &str {
        self.table.namespace()
    }

    fn workspace(&self) -> &str {
        self.table.workspace()
    }

    fn global_config(&self) -> &JsonMap {
        self.table.global_config()
    }

    // initialize / finalize / index_done_callback: no-op for in-memory storage
    // (the trait default covers all three).

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        StorageNameSpace::drop(&self.table, biz_id).await
    }
}

#[async_trait]
impl KVStorage for InMemoryKVStorage {
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

    /// Plain KV semantics: the incoming record replaces the stored one.
    async fn upsert(&self, data: HashMap<String, JsonMap>, biz_id: &str) -> easyrag_core::Result<()> {
        self.table.upsert_replace(data, biz_id);
        Ok(())
    }

    async fn delete(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        self.table.delete(ids, biz_id).await
    }

    async fn is_empty(&self, biz_id: &str) -> easyrag_core::Result<bool> {
        self.table.is_empty(biz_id).await
    }
}
