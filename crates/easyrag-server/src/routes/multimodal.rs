//! Multimodal processing endpoints.
//!
//! The controller bean is conditional on `RAGAnything` being
//! configured; assembly mirrors that by merging this router only when
//! `AppServices.rag_anything` is present (otherwise routes 404).

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Multipart, Query};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::JsonMap;

use crate::error::{AppError, AppResult};
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/multimodal/process", post(process_upload))
        .route("/api/multimodal/process-path", post(process_path))
        .route("/api/multimodal/process-folder", post(process_folder))
        .route("/api/multimodal/query", post(multimodal_query))
        .route("/api/multimodal/status", get(status))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessPathRequest {
    file_path: String,
    #[serde(default)]
    parse_method: Option<String>,
    #[serde(default)]
    output_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessFolderRequest {
    folder_path: String,
    #[serde(default)]
    output_dir: Option<String>,
    #[serde(default)]
    parse_method: Option<String>,
    #[serde(default)]
    recursive: Option<bool>,
    #[serde(default)]
    max_workers: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MultimodalQueryRequest {
    query: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    multimodal_content: Option<Vec<JsonMap>>,
}

fn rag_anything(state: &AppServices) -> AppResult<&easyrag_pipeline::RAGAnything> {
    state
        .rag_anything
        .as_deref()
        .ok_or_else(|| AppError::status(404, "Multimodal processing is not configured"))
}

fn kwargs_from(parse_method: Option<&String>, output_dir: Option<&String>) -> JsonMap {
    let mut kwargs = JsonMap::new();
    if let Some(method) = parse_method {
        kwargs.insert("parse_method".into(), json!(method));
    }
    if let Some(dir) = output_dir {
        kwargs.insert("output_dir".into(), json!(dir));
    }
    kwargs
}

/// Upload and process a single file (multipart `file` part plus optional
/// `parseMethod` / `outputDir` query params).
async fn process_upload(
    state: AppState,
    Query(params): Query<HashMap<String, String>>,
    mut multipart: Multipart,
) -> AppResult<Json<Value>> {
    let rag = rag_anything(&state)?;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Invalid multipart request: {e}")))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let filename = field.file_name().unwrap_or("upload.bin").to_string();
        let bytes = field
            .bytes()
            .await
            .map_err(|e| AppError::bad_request(format!("Failed to read upload: {e}")))?;

        // temp file `mm_upload_<prefix>_<filename>`, deleted after.
        let temp_path = std::env::temp_dir().join(format!(
            "mm_upload_{}_{}",
            crate::error::short_uuid(),
            filename.replace(['/', '\\', '\0'], "_")
        ));
        tokio::fs::write(&temp_path, &bytes)
            .await
            .map_err(|e| AppError::Core(easyrag_core::CoreError::Internal(format!(
                "write temp file: {e}"
            ))))?;

        let kwargs = kwargs_from(params.get("parseMethod"), params.get("outputDir"));
        let result = rag
            .process_file(&temp_path.to_string_lossy(), &kwargs)
            .await;
        let _ = tokio::fs::remove_file(&temp_path).await;
        return Ok(Json(Value::Object(result?)));
    }
    Err(AppError::bad_request("Missing 'file' part"))
}

async fn process_path(
    state: AppState,
    Json(request): Json<ProcessPathRequest>,
) -> AppResult<Json<Value>> {
    let rag = rag_anything(&state)?;
    let kwargs = kwargs_from(request.parse_method.as_ref(), request.output_dir.as_ref());
    let result = rag.process_file(&request.file_path, &kwargs).await?;
    Ok(Json(Value::Object(result)))
}

async fn process_folder(
    state: AppState,
    Json(request): Json<ProcessFolderRequest>,
) -> AppResult<Json<Value>> {
    let rag = rag_anything(&state)?;
    let mut kwargs = kwargs_from(request.parse_method.as_ref(), request.output_dir.as_ref());
    if let Some(recursive) = request.recursive {
        kwargs.insert("recursive".into(), json!(recursive));
    }
    if let Some(max_workers) = request.max_workers {
        kwargs.insert("max_workers".into(), json!(max_workers));
    }
    let result = rag.process_folder(&request.folder_path, &kwargs).await?;
    Ok(Json(json!({
        "status": "success",
        "total_files": result.total_files,
        "successful": result.successful_files.len(),
        "failed": result.failed_files.len(),
        "skipped": result.skipped_files.len(),
        "processing_time_seconds": result.processing_time_ms as f64 / 1000.0,
        "errors": result.errors,
    })))
}

async fn multimodal_query(
    state: AppState,
    Json(request): Json<MultimodalQueryRequest>,
) -> AppResult<Json<Value>> {
    let rag = rag_anything(&state)?;
    let mode = request.mode.clone().unwrap_or_else(|| "mix".to_string());
    let kwargs = JsonMap::new();
    let has_content = request
        .multimodal_content
        .as_ref()
        .is_some_and(|c| !c.is_empty());
    let result = if has_content {
        rag.query_with_multimodal(
            &request.query,
            request.multimodal_content.as_deref(),
            &mode,
            &kwargs,
        )
        .await?
    } else {
        rag.query(&request.query, &mode, &kwargs).await?
    };
    Ok(Json(json!({
        "status": "success",
        "query": request.query,
        "mode": mode,
        "result": result,
    })))
}

async fn status(state: AppState) -> AppResult<Json<Value>> {
    let rag = rag_anything(&state)?;
    Ok(Json(Value::Object(rag.get_processor_info())))
}
