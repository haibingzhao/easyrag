//! In-memory tenant config storage for development and testing.

use async_trait::async_trait;
use easyrag_core::storage::tenant_config::{TenantConfig, TenantConfigStorage};
use parking_lot::Mutex;
use std::collections::HashMap;

pub struct InMemoryTenantConfigStorage {
    configs: Mutex<HashMap<String, TenantConfig>>,
}

impl InMemoryTenantConfigStorage {
    pub fn new() -> Self {
        Self {
            configs: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryTenantConfigStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TenantConfigStorage for InMemoryTenantConfigStorage {
    async fn get_by_workspace(
        &self,
        workspace: &str,
    ) -> easyrag_core::Result<Option<TenantConfig>> {
        Ok(self.configs.lock().get(workspace).cloned())
    }

    async fn upsert(&self, config: &TenantConfig) -> easyrag_core::Result<()> {
        self.configs
            .lock()
            .insert(config.workspace.clone(), config.clone());
        Ok(())
    }

    async fn delete(&self, workspace: &str) -> easyrag_core::Result<()> {
        self.configs.lock().remove(workspace);
        Ok(())
    }

    async fn list_all(&self) -> easyrag_core::Result<Vec<TenantConfig>> {
        Ok(self.configs.lock().values().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn sample_config(workspace: &str) -> TenantConfig {
        let now = Utc::now();
        TenantConfig {
            workspace: workspace.to_string(),
            llm_model: Some("gpt-4o-mini".to_string()),
            llm_api_key: Some("sk-test".to_string()),
            llm_base_url: None,
            llm_temperature: Some(0.7),
            llm_max_tokens: None,
            embedding_model: None,
            embedding_api_key: None,
            embedding_base_url: None,
            embedding_dim: Some(768),
            chunk_size: Some(800),
            chunk_overlap_size: None,
            language: None,
            default_top_k: Some(20),
            rerank_enabled: Some(true),
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn crud_round_trip() {
        let storage = InMemoryTenantConfigStorage::new();

        // Initially empty
        assert!(storage
            .get_by_workspace("ws1")
            .await
            .unwrap()
            .is_none());
        assert!(storage.list_all().await.unwrap().is_empty());

        // Upsert
        let config = sample_config("ws1");
        storage.upsert(&config).await.unwrap();

        // Get
        let fetched = storage
            .get_by_workspace("ws1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.workspace, "ws1");
        assert_eq!(fetched.llm_model.as_deref(), Some("gpt-4o-mini"));
        assert_eq!(fetched.llm_api_key.as_deref(), Some("sk-test"));
        assert_eq!(fetched.embedding_dim, Some(768));
        assert_eq!(fetched.chunk_size, Some(800));
        assert_eq!(fetched.default_top_k, Some(20));
        assert_eq!(fetched.rerank_enabled, Some(true));

        // List
        assert_eq!(storage.list_all().await.unwrap().len(), 1);

        // Upsert second workspace
        storage.upsert(&sample_config("ws2")).await.unwrap();
        assert_eq!(storage.list_all().await.unwrap().len(), 2);

        // Delete
        storage.delete("ws1").await.unwrap();
        assert!(storage
            .get_by_workspace("ws1")
            .await
            .unwrap()
            .is_none());
        assert_eq!(storage.list_all().await.unwrap().len(), 1);

        // Delete non-existent is a no-op
        storage.delete("ws1").await.unwrap();
        assert_eq!(storage.list_all().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn upsert_overwrites_by_workspace() {
        let storage = InMemoryTenantConfigStorage::new();
        storage.upsert(&sample_config("ws1")).await.unwrap();

        let mut updated = sample_config("ws1");
        updated.llm_model = Some("claude-3".to_string());
        storage.upsert(&updated).await.unwrap();

        let fetched = storage
            .get_by_workspace("ws1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(fetched.llm_model.as_deref(), Some("claude-3"));
        assert_eq!(storage.list_all().await.unwrap().len(), 1);
    }
}
