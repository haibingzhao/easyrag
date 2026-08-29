//! Tenant configuration profile management API.
//!
//! | Method | Path | Description |
//! |--------|------|-------------|
//! | GET | /api/admin/tenant-config | List all tenant configs |
//! | GET | /api/admin/tenant-config?workspace=xxx | Get config for a workspace |
//! | POST | /api/admin/tenant-config | Create/update a tenant config |
//! | DELETE | /api/admin/tenant-config?workspace=xxx | Delete a tenant config |
//!
//! API keys are masked in GET responses (first 4 chars + `****`).
//! POST with `"****"` or empty API key preserves the existing value.

use std::sync::Arc;

use axum::extract::Query;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use easyrag_core::storage::tenant_config::TenantConfig;

use crate::error::{AppError, AppResult};
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/admin/tenant-config", get(list_or_get_tenant_config))
        .route("/api/admin/tenant-config", post(upsert_tenant_config))
        .route("/api/admin/tenant-config", delete(delete_tenant_config))
}

// ---------------------------------------------------------------------------
// Query / DTO types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct WorkspaceQuery {
    workspace: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpsertTenantConfigRequest {
    workspace: String,
    llm_model: Option<String>,
    llm_api_key: Option<String>,
    llm_base_url: Option<String>,
    llm_temperature: Option<f32>,
    llm_max_tokens: Option<u32>,
    embedding_model: Option<String>,
    embedding_api_key: Option<String>,
    embedding_base_url: Option<String>,
    embedding_dim: Option<usize>,
    chunk_size: Option<usize>,
    chunk_overlap_size: Option<usize>,
    language: Option<String>,
    default_top_k: Option<usize>,
    rerank_enabled: Option<bool>,
}

/// Response DTO with API keys masked.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TenantConfigResponse {
    workspace: String,
    llm_model: Option<String>,
    llm_api_key: Option<String>,
    llm_base_url: Option<String>,
    llm_temperature: Option<f32>,
    llm_max_tokens: Option<u32>,
    embedding_model: Option<String>,
    embedding_api_key: Option<String>,
    embedding_base_url: Option<String>,
    embedding_dim: Option<usize>,
    chunk_size: Option<usize>,
    chunk_overlap_size: Option<usize>,
    language: Option<String>,
    default_top_k: Option<usize>,
    rerank_enabled: Option<bool>,
    created_at: String,
    updated_at: String,
}

// ---------------------------------------------------------------------------
// API key masking
// ---------------------------------------------------------------------------

/// Mask an API key: show first 4 characters + `****`.
/// Returns `None` when the key is `None` or shorter than 4 characters
/// (in which case we show `"****"` only).
fn mask_api_key(key: &Option<String>) -> Option<String> {
    key.as_ref().map(|k| {
        let char_count = k.chars().count();
        if char_count <= 4 {
            "****".to_string()
        } else {
            let prefix: String = k.chars().take(4).collect();
            format!("{prefix}****")
        }
    })
}

fn config_to_response(config: &TenantConfig) -> TenantConfigResponse {
    TenantConfigResponse {
        workspace: config.workspace.clone(),
        llm_model: config.llm_model.clone(),
        llm_api_key: mask_api_key(&config.llm_api_key),
        llm_base_url: config.llm_base_url.clone(),
        llm_temperature: config.llm_temperature,
        llm_max_tokens: config.llm_max_tokens,
        embedding_model: config.embedding_model.clone(),
        embedding_api_key: mask_api_key(&config.embedding_api_key),
        embedding_base_url: config.embedding_base_url.clone(),
        embedding_dim: config.embedding_dim,
        chunk_size: config.chunk_size,
        chunk_overlap_size: config.chunk_overlap_size,
        language: config.language.clone(),
        default_top_k: config.default_top_k,
        rerank_enabled: config.rerank_enabled,
        created_at: config.created_at.to_rfc3339(),
        updated_at: config.updated_at.to_rfc3339(),
    }
}

/// Returns `true` when the submitted API key should be treated as "keep
/// existing": the masked sentinel `"****"` or an empty string.
fn is_masked_or_empty(key: &Option<String>) -> bool {
    match key {
        None => true,
        Some(s) => s.is_empty() || s == "****",
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/admin/tenant-config
/// With `?workspace=xxx`: return a single config (404 if absent).
/// Without query param: return all configs.
async fn list_or_get_tenant_config(
    state: AppState,
    Query(query): Query<WorkspaceQuery>,
) -> AppResult<Json<Value>> {
    if let Some(workspace) = query.workspace.as_deref().filter(|w| !w.is_empty()) {
        let config = state
            .tenant_config_storage
            .get_by_workspace(workspace)
            .await?;
        match config {
            Some(c) => Ok(Json(json!(config_to_response(&c)))),
            None => Err(AppError::status(
                404,
                format!("No tenant config for workspace: {workspace}"),
            )),
        }
    } else {
        let all = state.tenant_config_storage.list_all().await?;
        let responses: Vec<_> = all.iter().map(config_to_response).collect();
        Ok(Json(json!(responses)))
    }
}

/// POST /api/admin/tenant-config
/// Create or update a tenant config. If the workspace already has a config,
/// the existing values are merged: fields not provided in the request keep
/// their current values; API keys sent as `"****"` or `""` also keep the
/// existing value.
async fn upsert_tenant_config(
    state: AppState,
    Json(body): Json<UpsertTenantConfigRequest>,
) -> AppResult<Json<Value>> {
    let workspace = body.workspace.trim().to_string();
    if workspace.is_empty() {
        return Err(AppError::bad_request("workspace is required"));
    }
    // The default workspace context is built once at startup from the global
    // config and is never rebuilt from tenant config rows; accepting writes
    // here would silently succeed with zero effect. Reject explicitly.
    if workspace == state.config.workspace {
        return Err(AppError::bad_request(format!(
            "Tenant config cannot be set for the default workspace '{workspace}'; \
             it always uses the global server configuration"
        )));
    }

    let now = Utc::now();

    // Load existing config for merge semantics
    let existing = state
        .tenant_config_storage
        .get_by_workspace(&workspace)
        .await?;

    let config = match existing {
        Some(old) => {
            // Merge: request fields override, masked/empty API keys preserve old
            TenantConfig {
                workspace: workspace.clone(),
                llm_model: body.llm_model.or(old.llm_model),
                llm_api_key: if is_masked_or_empty(&body.llm_api_key) {
                    old.llm_api_key
                } else {
                    body.llm_api_key
                },
                llm_base_url: body.llm_base_url.or(old.llm_base_url),
                llm_temperature: body.llm_temperature.or(old.llm_temperature),
                llm_max_tokens: body.llm_max_tokens.or(old.llm_max_tokens),
                embedding_model: body.embedding_model.or(old.embedding_model),
                embedding_api_key: if is_masked_or_empty(&body.embedding_api_key) {
                    old.embedding_api_key
                } else {
                    body.embedding_api_key
                },
                embedding_base_url: body.embedding_base_url.or(old.embedding_base_url),
                embedding_dim: body.embedding_dim.or(old.embedding_dim),
                chunk_size: body.chunk_size.or(old.chunk_size),
                chunk_overlap_size: body.chunk_overlap_size.or(old.chunk_overlap_size),
                language: body.language.or(old.language),
                default_top_k: body.default_top_k.or(old.default_top_k),
                rerank_enabled: body.rerank_enabled.or(old.rerank_enabled),
                created_at: old.created_at,
                updated_at: now,
            }
        }
        None => TenantConfig {
            workspace: workspace.clone(),
            llm_model: body.llm_model,
            llm_api_key: body.llm_api_key.filter(|k| !k.is_empty() && k != "****"),
            llm_base_url: body.llm_base_url,
            llm_temperature: body.llm_temperature,
            llm_max_tokens: body.llm_max_tokens,
            embedding_model: body.embedding_model,
            embedding_api_key: body.embedding_api_key.filter(|k| !k.is_empty() && k != "****"),
            embedding_base_url: body.embedding_base_url,
            embedding_dim: body.embedding_dim,
            chunk_size: body.chunk_size,
            chunk_overlap_size: body.chunk_overlap_size,
            language: body.language,
            default_top_k: body.default_top_k,
            rerank_enabled: body.rerank_enabled,
            created_at: now,
            updated_at: now,
        },
    };

    state.tenant_config_storage.upsert(&config).await?;

    // Invalidate cached workspace context so next request uses new config
    state.workspace_resolver.invalidate_workspace(&workspace).await;

    tracing::info!("Tenant config upserted for workspace: {workspace}");
    Ok(Json(json!(config_to_response(&config))))
}

/// DELETE /api/admin/tenant-config?workspace=xxx
/// Delete the tenant config for a workspace (revert to global defaults).
async fn delete_tenant_config(
    state: AppState,
    Query(query): Query<WorkspaceQuery>,
) -> AppResult<Json<Value>> {
    let workspace = query
        .workspace
        .as_deref()
        .filter(|w| !w.is_empty())
        .ok_or_else(|| AppError::bad_request("workspace query parameter is required"))?;

    if workspace == state.config.workspace {
        return Err(AppError::bad_request(format!(
            "Tenant config cannot be deleted for the default workspace '{workspace}'; \
             it always uses the global server configuration"
        )));
    }

    state.tenant_config_storage.delete(workspace).await?;

    // Invalidate cached workspace context
    state.workspace_resolver.invalidate_workspace(workspace).await;

    tracing::info!("Tenant config deleted for workspace: {workspace}");
    Ok(Json(json!({
        "status": "success",
        "message": format!("Tenant config for workspace '{workspace}' deleted")
    })))
}
