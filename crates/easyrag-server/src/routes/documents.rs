//! Document management endpoints.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::storage::traits::StorageNameSpace;
use easyrag_core::util::hash::compute_args_hash;
use easyrag_core::JsonMap;
use easyrag_pipeline::parser::DocumentProcessingOptions;
use easyrag_pipeline::PipelineService;

use crate::error::{AppError, AppResult};
use crate::pagination::Paginated;
use crate::service::{
    check_pipeline_busy, LifecycleOps, ResolvedScope, DEFAULT_SIMILAR_THRESHOLD,
    DEFAULT_SIMILAR_TOP_K,
};
use crate::state::{AppServices, AppState};
use crate::util::success_response;

const MAX_UPLOAD_SIZE: usize = 100 * 1024 * 1024; // 100 MB

/// Ceiling for `similarTopK`. Each returned document costs one KV read, so the
/// knob is bounded well below what the recall budget would allow.
const SIMILAR_TOP_K_MAX: usize = 50;

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/documents/text", post(insert_text))
        .route(
            "/api/documents/file",
            post(insert_file).layer(DefaultBodyLimit::max(MAX_UPLOAD_SIZE + 1024 * 1024)),
        )
        .route("/api/documents/process", post(process_documents))
        .route("/api/documents/status", get(document_status))
        .route("/api/documents/list", get(list_documents))
        .route(
            "/api/documents/by-external-id",
            get(get_by_external_id),
        )
        .route("/api/documents/by-ids", post(get_by_ids))
        .route("/api/documents/archive", patch(archive_documents))
        .route(
            "/api/documents/{docId}",
            get(get_document)
                .delete(delete_document)
                .put(update_document),
        )
        .route("/api/documents/{docId}/index", post(index_document))
        .route("/api/documents/{docId}/reprocess", post(reprocess_document))
        .route("/api/documents/batch_delete", post(batch_delete_documents))
        .route("/api/documents/pipeline_status", get(pipeline_status))
        .route(
            "/api/documents/auto_scan_status",
            get(auto_scan_status),
        )
        .route("/api/documents/status_counts", get(status_counts))
        .route("/api/documents/track_status/{trackId}", get(track_status))
        .route("/api/documents/reprocess_failed", post(reprocess_failed))
        .route("/api/documents/batch", post(insert_batch))
        .route("/api/documents/scan", post(scan_input_dir))
        .route("/api/documents/cancel_pipeline", post(cancel_pipeline))
        .route("/api/documents/clear_cache", post(clear_cache))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InsertTextRequest {
    text: String,
    #[serde(default)]
    file_path: Option<String>,
    // R3/R1/R4/R6 extension fields - all optional, legacy payloads unchanged
    #[serde(default)]
    external_id: Option<String>,
    #[serde(default)]
    create_time: Option<i64>,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    options: Option<ProcessingOptionsDto>,
    // R2: request-level workspace routing (optional)
    #[serde(default)]
    workspace: Option<String>,
    // biz_id: request-level business line routing (optional, default "default")
    #[serde(default)]
    biz_id: Option<String>,
    // Similar-document probe: report already-indexed documents of this biz
    // shard that resemble `text`. Absent/false leaves the response untouched.
    #[serde(default)]
    return_similar: Option<bool>,
    #[serde(default)]
    similar_top_k: Option<usize>,
    #[serde(default)]
    similar_threshold: Option<f32>,
    // R1 (archive layer): consolidation fields, all optional. Absent values
    // keep the legacy defaults (active, level 0, no refs).
    #[serde(default)]
    archived: Option<bool>,
    #[serde(default)]
    level: Option<i32>,
    #[serde(default)]
    ref_ids: Option<Vec<String>>,
    #[serde(default)]
    merged_into: Option<String>,
}

/// R6: per-request processing options mapped onto DocumentProcessingOptions.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProcessingOptionsDto {
    #[serde(default)]
    chunk_method: Option<String>,
    #[serde(default)]
    chunk_token_size: Option<usize>,
    #[serde(default)]
    chunk_overlap: Option<usize>,
    #[serde(default)]
    skip_kg: Option<bool>,
    #[serde(default)]
    build_structure: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchInsertRequest {
    documents: Vec<InsertTextRequest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BatchDeleteRequest {
    doc_ids: Vec<String>,
    #[serde(default)]
    delete_file: bool,
    /// Accepted for contract compatibility; passed through to
    /// `cascadeDeleteDocument`, which never acts on it.
    #[serde(default)]
    delete_llm_cache: bool,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ScanRequest {
    input_dir: String,
    #[serde(default = "default_scan_extensions")]
    extensions: Option<String>,
    #[serde(default)]
    max_files: Option<usize>,
}

fn default_scan_extensions() -> Option<String> {
    Some(".txt,.md,.pdf".to_string())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListQuery {
    #[serde(default)]
    page: Option<usize>,
    #[serde(default)]
    page_size: Option<usize>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    file_path_prefix: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
    /// R2: exclude archived documents from list results at DB level.
    #[serde(default)]
    exclude_archived: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DeleteQuery {
    #[serde(default)]
    delete_file: bool,
    /// Accepted for contract compatibility; passed through to
    /// `cascadeDeleteDocument`, which never acts on it.
    #[serde(default)]
    delete_llm_cache: bool,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateTextRequest {
    text: String,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    create_time: Option<i64>,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    options: Option<ProcessingOptionsDto>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkspaceQuery {
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ExternalIdQuery {
    external_id: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

/// R3 (archive layer): batch read by docId. Includes archived documents.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ByIdsRequest {
    doc_ids: Vec<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

/// R4 (archive layer): flag-only update, no re-index.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArchiveRequest {
    doc_ids: Vec<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
    #[serde(default)]
    archived: Option<bool>,
    #[serde(default)]
    level: Option<i32>,
    #[serde(default)]
    ref_ids: Option<Vec<String>>,
    #[serde(default)]
    merged_into: Option<String>,
}

/// Upper bound on one batch read/archive call. EasyAI batches on its side;
/// an unbounded list would fan out into unbounded storage round-trips.
const ARCHIVE_BATCH_MAX: usize = 200;

/// R2: resolve lifecycle ops for a request-level workspace/biz_id
/// (None = default for both). Returns the normalized biz_id that all
/// storage calls must carry.
pub(crate) async fn ops_for(
    state: &AppServices,
    workspace: Option<&str>,
    biz_id: Option<&str>,
) -> AppResult<(LifecycleOps, String)> {
    let resolved = state
        .workspace_resolver
        .resolve(workspace, biz_id)
        .await?;
    let ops = LifecycleOps {
        bundle: resolved.context.bundle,
        pipeline: resolved.context.pipeline,
    };
    Ok((ops, resolved.biz_id))
}

/// Scope-only query for handlers whose routing lives in the query
/// string (camelCase wire names).
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ScopeQuery {
    #[serde(default)]
    pub(crate) workspace: Option<String>,
    #[serde(default)]
    pub(crate) biz_id: Option<String>,
}

/// Resolve with server defaults for both workspace and biz_id; used by
/// endpoints whose contract carries no routing fields.
async fn default_scope(state: &AppServices) -> AppResult<ResolvedScope> {
    state.workspace_resolver.resolve(None, None).await
}

/// Resolve with optional query-string routing fields (None = defaults).
async fn query_scope(state: &AppServices, q: &ScopeQuery) -> AppResult<ResolvedScope> {
    state
        .workspace_resolver
        .resolve(q.workspace.as_deref(), q.biz_id.as_deref())
        .await
}

/// Read-path scope: an omitted/blank `bizId` selects every biz line of the
/// workspace (empty-string wildcard understood by the list/count storage
/// methods), so multi-biz workspaces keep their documents visible. Never
/// route writes through this helper — status writes must carry the real
/// per-document biz line.
pub(crate) async fn read_scope(state: &AppServices, q: &ScopeQuery) -> AppResult<ResolvedScope> {
    let mut scope = query_scope(state, q).await?;
    let explicit = q
        .biz_id
        .as_deref()
        .map(str::trim)
        .filter(|b| !b.is_empty());
    if explicit.is_none() {
        scope.biz_id = String::new();
    }
    Ok(scope)
}

/// Kick document processing for the given business line.
///
/// With streaming enabled (`pipeline_service` is `Some`) the persistent
/// discovery loop finds pending documents on its own, so this only makes sure
/// it is running (and nudges it). Otherwise it falls back to the one-shot batch
/// run — byte-for-byte the previous behaviour.
fn spawn_pipeline(
    pipeline_service: Option<Arc<PipelineService>>,
    pipeline: Arc<easyrag_pipeline::IngestionPipeline>,
    biz_id: String,
) {
    match pipeline_service {
        Some(service) => service.ensure_running(),
        None => {
            tokio::spawn(async move {
                pipeline.start(&biz_id).await;
            });
        }
    }
}

async fn insert_text(
    state: AppState,
    Json(request): Json<InsertTextRequest>,
) -> AppResult<Json<Value>> {
    let file_path = request.file_path.unwrap_or_else(|| "api_input".to_string());
    // R3: idempotent upsert when externalId is present; legacy insert otherwise
    let (ops, biz_id) =
        ops_for(&state, request.workspace.as_deref(), request.biz_id.as_deref()).await?;
    let options = request.options.map(|o| DocumentProcessingOptions {
        chunk_method: o.chunk_method,
        chunk_token_size: o.chunk_token_size,
        chunk_overlap: o.chunk_overlap,
        skip_kg: o.skip_kg.unwrap_or(false),
        parser_engine: None,
        build_structure: o.build_structure.unwrap_or(false),
    });
    let result = state
        .lifecycle_service
        .upsert_text(
            &request.text,
            &file_path,
            request.external_id.as_deref(),
            request.create_time,
            request.metadata.as_ref(),
            options.as_ref(),
            &crate::service::ArchiveFields {
                archived: request.archived,
                level: request.level,
                ref_ids: request.ref_ids.clone(),
                merged_into: request.merged_into.clone(),
            },
            &ops,
            &biz_id,
        )
        .await?;
    let message = if result.idempotent_skip {
        "Document unchanged, skipped reprocessing"
    } else if !result.created {
        "Document overwritten"
    } else {
        "Document inserted"
    };
    let doc_id = &result.doc_id;
    let mut body = json!({
        "status": "success",
        "docId": doc_id,
        "message": message,
    });
    // `similar` is written only when it was asked for: an absent key keeps the
    // legacy response byte-identical, and `[]` on an enabled request then means
    // "nothing similar" rather than "not requested".
    if request.return_similar.unwrap_or(false) {
        let (top_k, threshold) =
            resolve_similar_params(request.similar_top_k, request.similar_threshold);
        match state
            .lifecycle_service
            .find_similar_documents(&request.text, doc_id, top_k, threshold, &ops, &biz_id)
            .await
        {
            Ok(similar) => {
                body["similar"] = json!(similar
                    .iter()
                    .map(|doc| json!({
                        "docId": doc.doc_id,
                        "externalId": doc.external_id,
                        "filePath": doc.file_path,
                        "score": doc.score,
                        "contentSummary": doc.content_summary,
                    }))
                    .collect::<Vec<_>>());
            }
            Err(e) => {
                // The document is already stored; a failed probe must not fail
                // the ingest. The cause stays in the log — response bodies are
                // sanitized, so only a fixed marker reaches the caller.
                tracing::warn!("Similar-document probe failed for doc {doc_id}: {e:?}");
                body["similar"] = json!([]);
                body["message"] = json!(format!("{message}; similar search unavailable"));
            }
        }
    }
    Ok(Json(body))
}

/// Effective probe knobs: `similarTopK` converged into `1..=SIMILAR_TOP_K_MAX`
/// and `similarThreshold` into `0.0..=1.0`. Out-of-range values are clamped with
/// a warning instead of rejected — a probe returning slightly more or fewer
/// documents is not worth failing an ingest that already succeeded.
fn resolve_similar_params(top_k: Option<usize>, threshold: Option<f32>) -> (usize, f32) {
    let effective_top_k = top_k
        .unwrap_or(DEFAULT_SIMILAR_TOP_K)
        .clamp(1, SIMILAR_TOP_K_MAX);
    if top_k.is_some_and(|v| v != effective_top_k) {
        tracing::warn!("similarTopK {top_k:?} out of range; using {effective_top_k}");
    }
    let requested = threshold.unwrap_or(DEFAULT_SIMILAR_THRESHOLD);
    // NaN survives `clamp`, and a NaN threshold would reject every hit.
    let effective_threshold = if requested.is_finite() {
        requested.clamp(0.0, 1.0)
    } else {
        DEFAULT_SIMILAR_THRESHOLD
    };
    if effective_threshold != requested {
        tracing::warn!("similarThreshold {requested} out of range; using {effective_threshold}");
    }
    (effective_top_k, effective_threshold)
}

/// Filename validation.
fn is_valid_filename(filename: &str) -> bool {
    !filename.contains('\0')
        && !filename.contains('\n')
        && !filename.contains('\r')
        && !filename.contains("..")
        && !filename.contains('/')
        && !filename.contains('\\')
}

async fn insert_file(state: AppState, mut multipart: Multipart) -> AppResult<Json<Value>> {
    let registry = &state.parser_registry;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(format!("Invalid multipart request: {e}")))?
    {
        if field.name() != Some("file") {
            continue;
        }
        let filename = field
            .file_name()
            .unwrap_or_default()
            .to_string();

        // Security: path traversal protection.
        if !is_valid_filename(&filename) {
            return Err(AppError::bad_request("Invalid filename"));
        }

        // Security: extension whitelist.
        let ext = filename
            .rsplit_once('.')
            .map(|(_, e)| e.to_lowercase())
            .unwrap_or_default();
        if !registry.is_supported(&ext) {
            let supported = registry.supported_extensions().join(", ");
            return Err(AppError::bad_request(format!(
                "Unsupported file format: .{ext}. Supported: {supported}"
            )));
        }

        let bytes = field
            .bytes()
            .await
            .map_err(|e| AppError::bad_request(format!("Failed to read upload: {e}")))?;

        // Security: size limit.
        if bytes.len() > MAX_UPLOAD_SIZE {
            return Err(AppError::payload_too_large(format!(
                "File exceeds maximum upload size ({} MB)",
                MAX_UPLOAD_SIZE / 1024 / 1024
            )));
        }

        let parser = registry
            .get_parser_for_file_name(&filename)
            .ok_or_else(|| AppError::bad_request(format!("No parser available for: .{ext}")))?;
        let content = parser.parse_bytes(&bytes, &filename).await?;
        if content.trim().is_empty() {
            return Err(AppError::bad_request("File contains no extractable text"));
        }

        // Duplicate detection by content hash.
        let content_hash = compute_args_hash(std::iter::once(&content));
        let scope = default_scope(&state).await?;
        if let Some((existing_id, _)) = state
            .doc_status_storage
            .get_doc_by_content_hash(&content_hash, &scope.biz_id)
            .await?
        {
            return Err(AppError::conflict(format!(
                "Duplicate content detected (matches document: {existing_id})"
            )));
        }

        let doc_id = state
            .pipeline
            .insert_document(&content, &filename, None, None, &scope.biz_id)
            .await?;
        return Ok(Json(json!({
            "status": "success",
            "docId": doc_id,
            "message": "File uploaded and inserted",
        })));
    }
    Err(AppError::bad_request("Missing 'file' part"))
}

async fn process_documents(
    state: AppState,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    let scope = query_scope(&state, &q).await?;
    spawn_pipeline(
        state.pipeline_service.clone(),
        Arc::clone(&scope.context.pipeline),
        scope.biz_id,
    );
    Ok(Json(success_response("Pipeline started", None)))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DocumentStatusQuery {
    doc_id: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

async fn document_status(
    state: AppState,
    Query(query): Query<DocumentStatusQuery>,
) -> AppResult<Json<Value>> {
    let scope = state
        .workspace_resolver
        .resolve(query.workspace.as_deref(), query.biz_id.as_deref())
        .await?;
    match state
        .doc_status_storage
        .get_by_id(&query.doc_id, &scope.biz_id)
        .await?
    {
        Some(status) => Ok(Json(Value::Object(status))),
        None => Ok(Json(json!({ "error": "Document not found" }))),
    }
}

async fn list_documents(
    state: AppState,
    Query(query): Query<ListQuery>,
) -> AppResult<Json<Value>> {
    let page = query.page.unwrap_or(1);
    let page_size = query.page_size.unwrap_or(50);
    // DocStatus.fromValue throws on unknown values -> 400.
    let status_filter = match &query.status {
        Some(s) => Some(DocStatus::from_value(s)?),
        None => None,
    };
    let mut resolved = state
        .workspace_resolver
        .resolve(query.workspace.as_deref(), query.biz_id.as_deref())
        .await?;
    // Read path: blank bizId lists every biz line of the workspace.
    let explicit = query
        .biz_id
        .as_deref()
        .map(str::trim)
        .filter(|b| !b.is_empty());
    if explicit.is_none() {
        resolved.biz_id = String::new();
    }
    let exclude_archived = query.exclude_archived.unwrap_or(false);
    let (docs, total) = resolved
        .context
        .bundle
        .doc_status_storage
        .get_docs_paginated(
            status_filter,
            None,
            page,
            page_size,
            "updated_at",
            "desc",
            query.file_path_prefix.as_deref(),
            exclude_archived,
            &resolved.biz_id,
        )
        .await?;
    let documents: Vec<Value> = docs
        .iter()
        .map(|(id, doc)| {
            json!({
                "id": id,
                "filePath": doc.file_path,
                "status": doc.status.value(),
                "contentSummary": doc.content_summary,
                "contentLength": doc.content_length,
                "createdAt": doc.created_at,
                "updatedAt": doc.updated_at,
                "chunksCount": doc.chunks_count,
                "externalId": doc.external_id,
                "hasStructure": doc.has_structure,
                "bizId": doc.biz_id,
                "archived": doc.archived,
                "level": doc.level,
                "refIds": doc.ref_ids,
                "mergedInto": doc.merged_into,
            })
        })
        .collect();
    Ok(Json(serde_json::to_value(Paginated::new(
        documents, total, page, page_size,
    ))?))
}

/// R5: full document text plus status meta. Literal routes (/status,
/// /list, ...) take precedence over this `{docId}` wildcard in axum.
async fn get_document(
    state: AppState,
    AxumPath(doc_id): AxumPath<String>,
    Query(query): Query<WorkspaceQuery>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, query.workspace.as_deref(), query.biz_id.as_deref()).await?;
    let result = state
        .lifecycle_service
        .get_document_full(&doc_id, &ops, &biz_id)
        .await?;
    Ok(Json(result))
}

/// R3: look up a document by external business key
/// (query param: externalId may contain '/').
async fn get_by_external_id(
    state: AppState,
    Query(query): Query<ExternalIdQuery>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, query.workspace.as_deref(), query.biz_id.as_deref()).await?;
    let result = state
        .lifecycle_service
        .get_by_external_id(&query.external_id, &ops, &biz_id)
        .await?;
    Ok(Json(result))
}

/// R3 (archive layer): batch-read full documents by docId.
/// Archived documents are included by design (history drill-down).
async fn get_by_ids(
    state: AppState,
    Json(request): Json<ByIdsRequest>,
) -> AppResult<Json<Value>> {
    if request.doc_ids.len() > ARCHIVE_BATCH_MAX {
        return Err(AppError::bad_request(format!(
            "docIds exceeds the maximum of {ARCHIVE_BATCH_MAX} per request"
        )));
    }
    let (ops, biz_id) =
        ops_for(&state, request.workspace.as_deref(), request.biz_id.as_deref()).await?;
    let documents = state
        .lifecycle_service
        .get_documents_by_ids(&request.doc_ids, &ops, &biz_id)
        .await?;
    Ok(Json(json!({ "documents": documents })))
}

/// R4 (archive layer): update archived/level/ref_ids/merged_into only.
/// Content and vectors are untouched, so no re-index happens.
async fn archive_documents(
    state: AppState,
    Json(request): Json<ArchiveRequest>,
) -> AppResult<Json<Value>> {
    if request.doc_ids.len() > ARCHIVE_BATCH_MAX {
        return Err(AppError::bad_request(format!(
            "docIds exceeds the maximum of {ARCHIVE_BATCH_MAX} per request"
        )));
    }
    let (ops, biz_id) =
        ops_for(&state, request.workspace.as_deref(), request.biz_id.as_deref()).await?;
    let fields = crate::service::ArchiveFields {
        archived: request.archived,
        level: request.level,
        ref_ids: request.ref_ids,
        merged_into: request.merged_into,
    };
    let results = state
        .lifecycle_service
        .archive_documents(&request.doc_ids, &fields, &ops, &biz_id)
        .await?;
    Ok(Json(json!({ "results": results })))
}

/// R6: index a single document asynchronously, bypassing the global
/// pipeline queue.
async fn index_document(
    state: AppState,
    AxumPath(doc_id): AxumPath<String>,
    Query(query): Query<WorkspaceQuery>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, query.workspace.as_deref(), query.biz_id.as_deref()).await?;
    let result = state
        .lifecycle_service
        .index_single_document(&doc_id, &ops, &biz_id)
        .await?;
    Ok(Json(result))
}

/// Reprocess a single failed/pending document by id. The caller must pass the
/// `bizId` the document belongs to (echoed by `/api/documents/list`): doc ids
/// are content-derived and can repeat across biz lines, and every status write
/// is scoped by (workspace, biz_id) — writing with the server-default biz_id
/// would create a ghost status row instead of resetting the real one.
/// `index_single_document` guards the whole flow: 404 when the KV content is
/// missing under this scope, then an atomic claim (pending/failed ->
/// processing) whose storage-level UPDATE can never insert new rows.
async fn reprocess_document(
    state: AppState,
    AxumPath(doc_id): AxumPath<String>,
    Query(query): Query<WorkspaceQuery>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, query.workspace.as_deref(), query.biz_id.as_deref()).await?;
    let result = state
        .lifecycle_service
        .index_single_document(&doc_id, &ops, &biz_id)
        .await?;
    Ok(Json(result))
}

/// PUT /api/documents/{docId} — replace content of an existing document.
async fn update_document(
    state: AppState,
    AxumPath(doc_id): AxumPath<String>,
    Json(request): Json<UpdateTextRequest>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, request.workspace.as_deref(), request.biz_id.as_deref()).await?;
    let options = request.options.map(|o| DocumentProcessingOptions {
        chunk_method: o.chunk_method,
        chunk_token_size: o.chunk_token_size,
        chunk_overlap: o.chunk_overlap,
        skip_kg: o.skip_kg.unwrap_or(false),
        parser_engine: None,
        build_structure: o.build_structure.unwrap_or(false),
    });
    let result = state
        .lifecycle_service
        .update_document_by_id(
            &doc_id,
            &request.text,
            request.file_path.as_deref(),
            request.create_time,
            request.metadata.as_ref(),
            options.as_ref(),
            &ops,
            &biz_id,
        )
        .await?;
    let message = if result.idempotent_skip {
        "Document unchanged, skipped reprocessing"
    } else {
        "Document updated"
    };
    Ok(Json(json!({
        "status": "success",
        "docId": result.doc_id,
        "message": message,
    })))
}

async fn delete_document(
    state: AppState,
    AxumPath(doc_id): AxumPath<String>,
    Query(query): Query<DeleteQuery>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, query.workspace.as_deref(), query.biz_id.as_deref()).await?;
    check_pipeline_busy(&ops.pipeline)?;
    state
        .lifecycle_service
        .cascade_delete_document(&doc_id, query.delete_file, query.delete_llm_cache, &ops, &biz_id)
        .await?;
    Ok(Json(success_response(
        "Document deleted with cascade cleanup",
        None,
    )))
}

async fn batch_delete_documents(
    state: AppState,
    Json(request): Json<BatchDeleteRequest>,
) -> AppResult<Json<Value>> {
    let (ops, biz_id) =
        ops_for(&state, request.workspace.as_deref(), request.biz_id.as_deref()).await?;
    check_pipeline_busy(&ops.pipeline)?;
    let mut success_count = 0usize;
    let mut errors: Vec<String> = Vec::new();
    for doc_id in &request.doc_ids {
        match state
            .lifecycle_service
            .cascade_delete_document(
                doc_id,
                request.delete_file,
                request.delete_llm_cache,
                &ops,
                &biz_id,
            )
            .await
        {
            Ok(()) => success_count += 1,
            Err(e) => errors.push(format!("{doc_id}: {e:?}")),
        }
    }
    Ok(Json(json!({
        "status": if errors.is_empty() { "success" } else { "partial_success" },
        "deleted": success_count,
        "errors": errors,
    })))
}

async fn pipeline_status(
    state: AppState,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    let scope = read_scope(&state, &q).await?;
    let counts = scope
        .context
        .bundle
        .doc_status_storage
        .get_status_counts(&scope.biz_id)
        .await?;
    // Additive merge: keep `status`/`counts`, fold in the live pipeline state
    // (batch flags) plus the streaming counters. `active_count`/`channel_len`/
    // `total_processed` are atomic reads (no DB cost); `pending_count` derives
    // from `counts` so the webui's busy/pending indicators work — previously
    // `busy` was absent and the UI showed a permanent "Idle".
    let live = scope.context.pipeline.pipeline_status();
    let busy = live.get("busy").and_then(Value::as_bool).unwrap_or(false);
    let pending_count = counts
        .get(DocStatus::Pending.value())
        .copied()
        .unwrap_or(0);
    let (streaming, channel_len, total_processed) = match &state.pipeline_service {
        Some(service) => (
            service.is_active(),
            service.channel_len(),
            service.total_processed(),
        ),
        None => (false, 0, 0),
    };
    Ok(Json(json!({
        "status": if busy { "running" } else { "idle" },
        "counts": counts,
        "busy": busy,
        "cancel_requested": live.get("cancel_requested").and_then(Value::as_bool).unwrap_or(false),
        "docs_processed": live.get("docs_processed").and_then(Value::as_u64).unwrap_or(0),
        "last_run_at": live.get("last_run_at").cloned().unwrap_or(Value::Null),
        "active_count": live.get("active_count").and_then(Value::as_u64).unwrap_or(0),
        "channel_len": channel_len,
        "total_processed": total_processed,
        "pending_count": pending_count,
        "streaming": streaming,
    })))
}

/// Background auto-scanner observability (BC-11). Reports whether the scanner
/// is enabled, the effective task list (derived from config at startup), and
/// the latest cycle stats per task. Workspace-agnostic: the scanner is a
/// process-wide background service, so this reads global stats, not a scope.
async fn auto_scan_status(state: AppState) -> Json<Value> {
    Json(json!({
        "enabled": state.config.auto_scan_enabled,
        "tasks": state.auto_scan_stats.task_specs(),
        "cycles": state.auto_scan_stats.snapshot(),
    }))
}

async fn status_counts(
    state: AppState,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    let scope = read_scope(&state, &q).await?;
    let counts = scope
        .context
        .bundle
        .doc_status_storage
        .get_all_status_counts(&scope.biz_id)
        .await?;
    let total: usize = counts.values().sum();
    Ok(Json(json!({ "counts": counts, "total": total })))
}

async fn track_status(
    state: AppState,
    AxumPath(track_id): AxumPath<String>,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    let scope = query_scope(&state, &q).await?;
    let docs = state
        .doc_status_storage
        .get_docs_by_track_id(&track_id, &scope.biz_id)
        .await?;
    if docs.is_empty() {
        return Ok(Json(json!({
            "track_id": track_id,
            "status": "not_found",
            "documents": [],
        })));
    }
    let mut status_counts: HashMap<String, usize> = HashMap::new();
    for doc in docs.values() {
        *status_counts.entry(doc.status.value().to_string()).or_default() += 1;
    }
    let all_processed = docs
        .values()
        .all(|d| d.status == DocStatus::Processed || d.status == DocStatus::Failed);
    let documents: Vec<Value> = docs
        .iter()
        .map(|(id, doc)| {
            json!({ "id": id, "status": doc.status.value(), "file_path": doc.file_path })
        })
        .collect();
    Ok(Json(json!({
        "track_id": track_id,
        "status": if all_processed { "completed" } else { "processing" },
        "total": docs.len(),
        "status_counts": status_counts,
        "documents": documents,
    })))
}

async fn reprocess_failed(
    state: AppState,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    // Wildcard read: collect failed rows across every biz line of the
    // workspace, then reset each row under its own biz_id — an upsert with
    // the wrong line would create a ghost row instead of resetting this one.
    let scope = read_scope(&state, &q).await?;
    let doc_status_storage = &scope.context.bundle.doc_status_storage;
    let failed_docs = doc_status_storage
        .get_docs_by_status(DocStatus::Failed, &scope.biz_id)
        .await?;
    if failed_docs.is_empty() {
        return Ok(Json(json!({
            "status": "success",
            "message": "No failed documents to reprocess",
            "count": 0,
        })));
    }
    // Reset failed docs to pending.
    let now = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true);
    for (doc_id, doc) in failed_docs.iter() {
        let biz_id = if !doc.biz_id.is_empty() {
            doc.biz_id.clone()
        } else if !scope.biz_id.is_empty() {
            scope.biz_id.clone()
        } else {
            tracing::warn!("Skipping failed doc {doc_id}: no biz line recorded");
            continue;
        };
        let mut update = JsonMap::new();
        update.insert("status".into(), json!("pending"));
        update.insert("updated_at".into(), json!(now));
        update.insert("error_msg".into(), Value::Null);
        let mut data = HashMap::new();
        data.insert(doc_id.clone(), update);
        doc_status_storage.upsert(data, &biz_id).await?;
    }
    // Empty biz_id starts a wildcard pipeline run; per-document biz lines
    // are resolved inside run_pipeline.
    spawn_pipeline(
        state.pipeline_service.clone(),
        Arc::clone(&state.pipeline),
        scope.biz_id,
    );
    Ok(Json(json!({
        "status": "success",
        "message": format!("Reprocessing {} failed documents", failed_docs.len()),
        "count": failed_docs.len(),
    })))
}

async fn insert_batch(
    state: AppState,
    Json(request): Json<BatchInsertRequest>,
) -> AppResult<Json<Value>> {
    let mut doc_ids = Vec::new();
    for item in &request.documents {
        let file_path = item
            .file_path
            .clone()
            .unwrap_or_else(|| "batch_input".to_string());
        let scope = state
            .workspace_resolver
            .resolve(item.workspace.as_deref(), item.biz_id.as_deref())
            .await?;
        let doc_id = state
            .pipeline
            .insert_document(&item.text, &file_path, None, None, &scope.biz_id)
            .await?;
        doc_ids.push(doc_id);
    }
    // Auto-start pipeline.
    let scope = default_scope(&state).await?;
    spawn_pipeline(
        state.pipeline_service.clone(),
        Arc::clone(&state.pipeline),
        scope.biz_id,
    );
    let count = doc_ids.len();
    Ok(Json(json!({ "status": "success", "doc_ids": doc_ids, "count": count })))
}

async fn scan_input_dir(
    state: AppState,
    Json(request): Json<ScanRequest>,
) -> AppResult<Json<Value>> {
    let dir = Path::new(&request.input_dir);
    if !dir.exists() || !dir.is_dir() {
        return Ok(Json(json!({
            "status": "error",
            "message": format!("Directory not found: {}", request.input_dir),
        })));
    }
    let extensions: Vec<String> = request
        .extensions
        .clone()
        .unwrap_or_else(|| ".txt,.md,.pdf".to_string())
        .split(',')
        .map(|e| e.trim().to_lowercase())
        .collect();
    let max_files = request.max_files.unwrap_or(100);
    let scope = default_scope(&state).await?;
    // Shared scanner: identical content_hash for a known path is skipped,
    // changed content is re-inserted. This endpoint used to carry a private
    // copy of `collect_files` plus an insert loop with no dedup at all, so
    // every rescan re-ingested the whole directory.
    let outcome = easyrag_pipeline::scan_directory(
        &state.pipeline,
        &state.doc_status_storage,
        dir,
        &extensions,
        max_files,
        &scope.biz_id,
    )
    .await?;

    // Auto-start pipeline.
    let doc_ids = outcome.inserted_doc_ids;
    if !doc_ids.is_empty() {
        spawn_pipeline(
            state.pipeline_service.clone(),
            Arc::clone(&state.pipeline),
            scope.biz_id,
        );
    }
    let count = doc_ids.len();
    Ok(Json(json!({
        "status": "success",
        "message": format!("Scanned {} files, inserted {} documents", outcome.files_seen, count),
        "doc_ids": doc_ids,
        "count": count,
    })))
}

async fn cancel_pipeline(state: AppState) -> Json<Value> {
    match &state.pipeline_service {
        Some(service) => service.cancel(),
        None => state.pipeline.cancel(),
    }
    Json(success_response("Pipeline cancellation requested", None))
}

async fn clear_cache(state: AppState) -> AppResult<Json<Value>> {
    check_pipeline_busy(&state.pipeline)?;
    let scope = default_scope(&state).await?;
    let biz_id = scope.biz_id.as_str();
    // Full destructive clear: all storage + graph. Errors are propagated
    //
    StorageNameSpace::drop(&*state.vector_storage as &dyn StorageNameSpace, biz_id).await?;
    StorageNameSpace::drop(&*state.kv_storage as &dyn StorageNameSpace, biz_id).await?;
    StorageNameSpace::drop(&*state.graph_storage as &dyn StorageNameSpace, biz_id).await?;
    StorageNameSpace::drop(&*state.doc_status_storage as &dyn StorageNameSpace, biz_id).await?;
    Ok(Json(success_response("All data cleared successfully", None)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service::DEFAULT_SIMILAR_THRESHOLD;

    #[test]
    fn absent_probe_params_take_the_server_defaults() {
        assert_eq!(
            resolve_similar_params(None, None),
            (DEFAULT_SIMILAR_TOP_K, DEFAULT_SIMILAR_THRESHOLD)
        );
    }

    #[test]
    fn in_range_probe_params_pass_through() {
        assert_eq!(resolve_similar_params(Some(3), Some(0.55)), (3, 0.55));
        // Both bounds of each range are valid inputs, not off-by-ones.
        assert_eq!(resolve_similar_params(Some(1), Some(0.0)), (1, 0.0));
        assert_eq!(
            resolve_similar_params(Some(SIMILAR_TOP_K_MAX), Some(1.0)),
            (SIMILAR_TOP_K_MAX, 1.0)
        );
    }

    #[test]
    fn out_of_range_probe_params_are_clamped_not_rejected() {
        assert_eq!(resolve_similar_params(Some(0), Some(-0.2)), (1, 0.0));
        assert_eq!(
            resolve_similar_params(Some(9999), Some(1.5)),
            (SIMILAR_TOP_K_MAX, 1.0)
        );
    }

    #[test]
    fn a_non_finite_threshold_falls_back_to_the_default() {
        // `f32::clamp` hands NaN straight back, and every `score >= NaN`
        // comparison is false — an unguarded NaN would silently match nothing.
        assert_eq!(
            resolve_similar_params(None, Some(f32::NAN)).1,
            DEFAULT_SIMILAR_THRESHOLD
        );
        assert_eq!(
            resolve_similar_params(None, Some(f32::INFINITY)).1,
            DEFAULT_SIMILAR_THRESHOLD
        );
    }
}
