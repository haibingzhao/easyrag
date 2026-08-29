//! Tenant configuration profile storage (per-workspace config overrides).
//!
//! Each workspace may optionally have a [`TenantConfig`] row that overrides
//! specific `ServerConfig` fields (LLM model, API key, embedding model,
//! chunk size, etc.). Fields left as `None` fall back to the global `.env`
//! defaults when merged via `ServerConfig::merge_tenant_config`.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Per-workspace configuration profile. All override fields are `Option`;
/// `None` means "use the global default from `.env`".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Primary key — one-to-one with workspace name.
    pub workspace: String,

    // LLM overrides
    pub llm_model: Option<String>,
    pub llm_api_key: Option<String>,
    pub llm_base_url: Option<String>,
    pub llm_temperature: Option<f32>,
    pub llm_max_tokens: Option<u32>,

    // Embedding overrides
    pub embedding_model: Option<String>,
    pub embedding_api_key: Option<String>,
    pub embedding_base_url: Option<String>,
    pub embedding_dim: Option<usize>,

    // Chunking overrides
    pub chunk_size: Option<usize>,
    pub chunk_overlap_size: Option<usize>,

    // Query / language overrides
    pub language: Option<String>,
    pub default_top_k: Option<usize>,

    // Rerank override
    pub rerank_enabled: Option<bool>,

    // Timestamps
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Tenant configuration profile storage interface.
/// Global scope (no workspace column in the SQL sense — the `workspace`
/// field IS the primary key, not a filter column).
#[async_trait]
pub trait TenantConfigStorage: Send + Sync {
    /// Get the tenant config for a specific workspace. Returns `None` when
    /// no override row exists (the workspace uses global defaults).
    async fn get_by_workspace(
        &self,
        workspace: &str,
    ) -> crate::Result<Option<TenantConfig>>;

    /// Insert or update a tenant config (keyed by `workspace`).
    async fn upsert(&self, config: &TenantConfig) -> crate::Result<()>;

    /// Delete the tenant config for a workspace (revert to global defaults).
    async fn delete(&self, workspace: &str) -> crate::Result<()>;

    /// List all tenant configs across all workspaces.
    async fn list_all(&self) -> crate::Result<Vec<TenantConfig>>;

    /// Initialize the storage (e.g. create tables). Idempotent.
    async fn initialize(&self) -> crate::Result<()> {
        Ok(())
    }
}
