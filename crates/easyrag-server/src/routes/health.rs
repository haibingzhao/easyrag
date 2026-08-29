//! Health check and auth-status endpoints.

use std::sync::Arc;

use axum::routing::get;
use axum::{Json, Router};
use chrono::Utc;
use serde_json::{json, Value};

use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/health", get(health))
        .route("/api/auth-status", get(auth_status))
}

async fn health(state: AppState) -> Json<Value> {
    let config = &state.config;
    Json(json!({
        "status": "healthy",
        "timestamp": Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
        "version": "1.0.0",
        "configuration": {
            "llm_binding": "openai",
            "llm_model": config.llm_model,
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap_size,
        },
        "storage": {
            "type": config.storage_type,
            "workspace": config.workspace,
        },
        "pipeline": {
            "max_parallel": config.pipeline_max_parallel,
            "llm_max_async": config.llm_max_async,
        },
        "features": {
            "rerank_enabled": config.rerank_enabled,
            "cache_enabled": config.cache_enabled,
            "auth_enabled": config.auth_enabled,
        },
    }))
}

async fn auth_status(state: AppState) -> Json<Value> {
    let auth_enabled = state.config.auth_enabled;
    Json(json!({
        "auth_required": auth_enabled,
        "version": "1.0.0",
        "webui_title": "EasyRAG Knowledge Base",
        // `if (!authEnabled) "guest" else null` (null serialized).
        "guest_token": if auth_enabled { Value::Null } else { json!("guest") },
    }))
}
