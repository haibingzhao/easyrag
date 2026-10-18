//! MCP tool surface for EasyRAG.
//!
//! Each `#[tool]` holds an in-process [`AppServices`] handle and calls the same
//! domain services the REST controllers do — never an HTTP loopback. The
//! request/response JSON projections are kept aligned with the keys the easyai
//! `EasyRagClient.kt` actually reads (snake_case for query/status/list bodies).

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, ErrorData, ServerCapabilities, ServerConfig,
};
use rmcp::{tool, tool_handler, tool_router, ServerHandler};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::model::doc_status::DocStatus;
use easyrag_pipeline::parser::DocumentProcessingOptions;

use crate::error::{AppError, AppResult};
use crate::routes::documents::{ops_for, read_scope, ScopeQuery};
use crate::routes::query::{build_query_param, normalize_biz_ids, QueryRequest};
use crate::state::AppServices;

/// Per-request body ceiling for the MCP endpoint (8 MB). File uploads stay on
/// the REST `/api/documents/file` route; text ingest payloads are far smaller.
pub(crate) const MCP_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Error mapping: AppError -> rmcp::ErrorData (preserve HTTP semantics in `data`)
// ---------------------------------------------------------------------------

fn to_error<E: Into<AppError>>(e: E) -> ErrorData {
    match e.into() {
        AppError::Status(code, message) => {
            let data = json!({ "http_status": code });
            if (500..600).contains(&code) {
                ErrorData::internal_error(message, Some(data))
            } else {
                ErrorData::invalid_params(message, Some(data))
            }
        }
        AppError::Core(err) => {
            // Mirror `IntoResponse`: raw core details stay server-side, not on the wire.
            tracing::error!("MCP handler core error: {err:?}");
            ErrorData::internal_error("Internal server error", None)
        }
    }
}

/// Wrap a JSON value as a successful tool result (single text content block).
fn json_result(value: Value) -> CallToolResult {
    CallToolResult::success(vec![ContentBlock::text(value.to_string())])
}

// ---------------------------------------------------------------------------
// Tool DTOs (schemars v1; field names are the MCP argument names)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchReq {
    #[schemars(description = "Natural-language query string")]
    query: String,
    #[schemars(description = "Retrieval mode; defaults to 'naive' (pure vector/graph data).")]
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    chunk_top_k: Option<usize>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
    #[serde(default)]
    biz_ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct IngestTextReq {
    text: String,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    external_id: Option<String>,
    #[serde(default)]
    create_time: Option<i64>,
    #[serde(default)]
    metadata: Option<HashMap<String, String>>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
    #[serde(default)]
    skip_kg: Option<bool>,
    #[serde(default)]
    build_structure: Option<bool>,
    /// When true, kick single-document indexing and poll doc_status to a
    /// terminal state before returning. Default false = submit-and-return docId.
    #[serde(default)]
    wait_for_index: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct DocRefReq {
    doc_id: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ExternalIdReq {
    external_id: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct DeleteDocReq {
    /// Delete by external business key (resolved to a docId first).
    #[serde(default)]
    external_id: Option<String>,
    /// Or delete directly by document id.
    #[serde(default)]
    doc_id: Option<String>,
    #[serde(default)]
    delete_file: bool,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct BatchDeleteReq {
    doc_ids: Vec<String>,
    #[serde(default)]
    delete_file: bool,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ListDocsReq {
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
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct TenantGetReq {
    #[serde(default)]
    workspace: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct TenantUpsertReq {
    workspace: String,
    #[serde(default)]
    llm_model: Option<String>,
    #[serde(default)]
    llm_api_key: Option<String>,
    #[serde(default)]
    llm_base_url: Option<String>,
    #[serde(default)]
    llm_temperature: Option<f32>,
    #[serde(default)]
    llm_max_tokens: Option<u32>,
    #[serde(default)]
    embedding_model: Option<String>,
    #[serde(default)]
    embedding_api_key: Option<String>,
    #[serde(default)]
    embedding_base_url: Option<String>,
    #[serde(default)]
    embedding_dim: Option<usize>,
    #[serde(default)]
    chunk_size: Option<usize>,
    #[serde(default)]
    chunk_overlap_size: Option<usize>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    default_top_k: Option<usize>,
    #[serde(default)]
    rerank_enabled: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct TenantDeleteReq {
    workspace: String,
}

/// Bridge the MCP upsert argument shape onto the shared REST tenant-config DTO
/// so both paths reuse `tenant_config::apply_upsert` (single merge truth).
impl From<TenantUpsertReq> for crate::routes::tenant_config::UpsertTenantConfigRequest {
    fn from(r: TenantUpsertReq) -> Self {
        Self {
            workspace: r.workspace,
            llm_model: r.llm_model,
            llm_api_key: r.llm_api_key,
            llm_base_url: r.llm_base_url,
            llm_temperature: r.llm_temperature,
            llm_max_tokens: r.llm_max_tokens,
            embedding_model: r.embedding_model,
            embedding_api_key: r.embedding_api_key,
            embedding_base_url: r.embedding_base_url,
            embedding_dim: r.embedding_dim,
            chunk_size: r.chunk_size,
            chunk_overlap_size: r.chunk_overlap_size,
            language: r.language,
            default_top_k: r.default_top_k,
            rerank_enabled: r.rerank_enabled,
        }
    }
}

/// Build a full [`QueryRequest`] from the subset the MCP tool exposes; every
/// other field keeps the REST boundary default.
fn base_query_request(
    query: String,
    mode: Option<String>,
    top_k: Option<usize>,
    chunk_top_k: Option<usize>,
    workspace: Option<String>,
    biz_id: Option<String>,
    biz_ids: Option<Vec<String>>,
) -> QueryRequest {
    QueryRequest {
        query,
        mode,
        top_k,
        chunk_top_k,
        response_type: None,
        language: None,
        user_prompt: None,
        stream: false,
        enable_rerank: None,
        time_range_start: None,
        time_range_end: None,
        hl_keywords: None,
        ll_keywords: None,
        conversation_history: None,
        only_need_context: None,
        only_need_prompt: None,
        include_references: None,
        doc_id: None,
        use_agent_mode: None,
        metadata_filters: None,
        structure_scope: None,
        include_sections: None,
        workspace,
        biz_id,
        biz_ids,
        enable_verification: None,
        enable_citation_validation: None,
        enable_evidence_scoring: None,
        enable_query_routing: None,
        exclude_archived: None,
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EasyRagMcp {
    tool_router: ToolRouter<Self>,
    services: Arc<AppServices>,
}

#[tool_router]
impl EasyRagMcp {
    pub fn new(services: Arc<AppServices>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            services,
        }
    }

    #[tool(description = "Health probe: reports the MCP service is live.")]
    async fn rag_health(&self) -> Result<CallToolResult, ErrorData> {
        Ok(json_result(json!({
            "status": "healthy",
            "service": "easyrag-mcp",
            "version": env!("CARGO_PKG_VERSION"),
        })))
    }

    #[tool(
        name = "rag_search",
        description = "Retrieve raw data (chunks/entities/relations) for a query, mirroring POST /api/query/data. Defaults to naive mode."
    )]
    async fn rag_search(
        &self,
        Parameters(req): Parameters<SearchReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let biz_ids = match &req.biz_ids {
            Some(raw) => normalize_biz_ids(raw).map_err(to_error)?,
            None => None,
        };
        let resolve_biz = biz_ids
            .as_ref()
            .and_then(|ids| ids.first().map(String::as_str))
            .or(req.biz_id.as_deref());
        let resolved = s
            .workspace_resolver
            .resolve(req.workspace.as_deref(), resolve_biz)
            .await
            .map_err(to_error)?;
        let engine = resolved.context.query_engine;
        let mut param = build_query_param(
            &base_query_request(
                req.query.clone(),
                Some(req.mode.clone().unwrap_or_else(|| "naive".to_string())),
                req.top_k,
                req.chunk_top_k,
                req.workspace.clone(),
                req.biz_id.clone(),
                req.biz_ids.clone(),
            ),
            &resolved.biz_id,
        );
        if let Some(ids) = biz_ids {
            param.biz_ids = ids;
        }
        let result = engine
            .query_data(&req.query, &param)
            .await
            .map_err(to_error)?;
        Ok(json_result(json!({
            "entities": result.entities,
            "relations": result.relations,
            "chunks": result.chunks,
            "degraded": result.degraded,
            "degradeReasons": result.degrade_reasons,
        })))
    }

    #[tool(
        name = "rag_ingest_text",
        description = "Insert or idempotently overwrite a text document. Returns the docId immediately; poll rag_document_status for indexing progress (or set waitForIndex)."
    )]
    async fn rag_ingest_text(
        &self,
        Parameters(req): Parameters<IngestTextReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let file_path = req.file_path.unwrap_or_else(|| "api_input".to_string());
        let (ops, biz_id) = ops_for(s, req.workspace.as_deref(), req.biz_id.as_deref())
            .await
            .map_err(to_error)?;
        let options = Some(DocumentProcessingOptions {
            chunk_method: None,
            chunk_token_size: None,
            chunk_overlap: None,
            skip_kg: req.skip_kg.unwrap_or(false),
            parser_engine: None,
            build_structure: req.build_structure.unwrap_or(false),
        });
        let result = s
            .lifecycle_service
            .upsert_text(
                &req.text,
                &file_path,
                req.external_id.as_deref(),
                req.create_time,
                req.metadata.as_ref(),
                options.as_ref(),
                &crate::service::ArchiveFields::default(),
                &ops,
                &biz_id,
            )
            .await
            .map_err(to_error)?;
        let message = if result.idempotent_skip {
            "Document unchanged, skipped reprocessing"
        } else if !result.created {
            "Document overwritten"
        } else {
            "Document inserted"
        };

        if req.wait_for_index.unwrap_or(false) {
            let _ = s
                .lifecycle_service
                .index_single_document(&result.doc_id, &ops, &biz_id)
                .await
                .map_err(to_error)?;
            let status =
                wait_for_terminal_status(s, &result.doc_id, &biz_id).await.map_err(to_error)?;
            return Ok(json_result(json!({
                "status": "success",
                "docId": result.doc_id,
                "message": message,
                "finalStatus": status,
            })));
        }

        Ok(json_result(json!({
            "status": "success",
            "docId": result.doc_id,
            "message": message,
        })))
    }

    #[tool(
        name = "rag_document_status",
        description = "Fetch a document's indexing status record (mirrors GET /api/documents/status)."
    )]
    async fn rag_document_status(
        &self,
        Parameters(req): Parameters<DocRefReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let resolved = s
            .workspace_resolver
            .resolve(req.workspace.as_deref(), req.biz_id.as_deref())
            .await
            .map_err(to_error)?;
        match s
            .doc_status_storage
            .get_by_id(&req.doc_id, &resolved.biz_id)
            .await
            .map_err(to_error)?
        {
            Some(status) => Ok(json_result(Value::Object(status))),
            None => Ok(json_result(json!({ "error": "Document not found" }))),
        }
    }

    #[tool(
        name = "rag_read_document",
        description = "Look up a document by external business key (mirrors GET /api/documents/by-external-id)."
    )]
    async fn rag_read_document(
        &self,
        Parameters(req): Parameters<ExternalIdReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let (ops, biz_id) = ops_for(s, req.workspace.as_deref(), req.biz_id.as_deref())
            .await
            .map_err(to_error)?;
        let result = s
            .lifecycle_service
            .get_by_external_id(&req.external_id, &ops, &biz_id)
            .await
            .map_err(to_error)?;
        Ok(json_result(result))
    }

    #[tool(
        name = "rag_list_documents",
        description = "List documents with pagination (mirrors GET /api/documents/list). A blank bizId spans every biz line."
    )]
    async fn rag_list_documents(
        &self,
        Parameters(req): Parameters<ListDocsReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let page = req.page.unwrap_or(1);
        let page_size = req.page_size.unwrap_or(50);
        let status_filter = match &req.status {
            Some(v) => Some(DocStatus::from_value(v).map_err(to_error)?),
            None => None,
        };
        let scope_query = ScopeQuery {
            workspace: req.workspace.clone(),
            biz_id: req.biz_id.clone(),
        };
        let scope = read_scope(s, &scope_query).await.map_err(to_error)?;
        let (docs, total) = scope
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
                req.file_path_prefix.as_deref(),
                false, // MCP does not expose archive exclusion
                &scope.biz_id,
            )
            .await
            .map_err(to_error)?;
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
                })
            })
            .collect();
        Ok(json_result(json!({
            "data": documents,
            "total": total,
            "page": page,
            "pageSize": page_size,
        })))
    }

    #[tool(
        name = "rag_delete_document",
        description = "Delete a document (by externalId or docId) with cascade cleanup."
    )]
    async fn rag_delete_document(
        &self,
        Parameters(req): Parameters<DeleteDocReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let (ops, biz_id) = ops_for(s, req.workspace.as_deref(), req.biz_id.as_deref())
            .await
            .map_err(to_error)?;
        crate::service::check_pipeline_busy(&ops.pipeline).map_err(to_error)?;
        let doc_id = match (req.doc_id.as_deref(), req.external_id.as_deref()) {
            (Some(id), _) => id.to_string(),
            (None, Some(ext)) => {
                let found = s
                    .lifecycle_service
                    .get_by_external_id(ext, &ops, &biz_id)
                    .await
                    .map_err(to_error)?;
                found
                    .get("doc_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ErrorData::internal_error("external id resolution failed", None)
                    })?
                    .to_string()
            }
            (None, None) => {
                return Err(ErrorData::invalid_params(
                    "either docId or externalId is required",
                    None,
                ))
            }
        };
        s.lifecycle_service
            .cascade_delete_document(&doc_id, req.delete_file, false, &ops, &biz_id)
            .await
            .map_err(to_error)?;
        Ok(json_result(json!({
            "status": "success",
            "message": "Document deleted with cascade cleanup",
        })))
    }

    #[tool(
        name = "rag_batch_delete",
        description = "Delete multiple documents by id with cascade cleanup (mirrors POST /api/documents/batch_delete)."
    )]
    async fn rag_batch_delete(
        &self,
        Parameters(req): Parameters<BatchDeleteReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let (ops, biz_id) = ops_for(s, req.workspace.as_deref(), req.biz_id.as_deref())
            .await
            .map_err(to_error)?;
        crate::service::check_pipeline_busy(&ops.pipeline).map_err(to_error)?;
        let mut success_count = 0usize;
        let mut errors: Vec<String> = Vec::new();
        for doc_id in &req.doc_ids {
            match s
                .lifecycle_service
                .cascade_delete_document(doc_id, req.delete_file, false, &ops, &biz_id)
                .await
            {
                Ok(()) => success_count += 1,
                Err(e) => errors.push(format!("{doc_id}: {e:?}")),
            }
        }
        Ok(json_result(json!({
            "status": if errors.is_empty() { "success" } else { "partial_success" },
            "deleted": success_count,
            "errors": errors,
        })))
    }

    #[tool(
        name = "rag_tenant_config_get",
        description = "Read per-workspace tenant configuration (API keys masked). Without a workspace, lists all."
    )]
    async fn rag_tenant_config_get(
        &self,
        Parameters(req): Parameters<TenantGetReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        if let Some(workspace) = req.workspace.as_deref().filter(|w| !w.is_empty()) {
            let config = s
                .tenant_config_storage
                .get_by_workspace(workspace)
                .await
                .map_err(to_error)?;
            match config {
                Some(c) => Ok(json_result(
                    serde_json::to_value(crate::routes::tenant_config::config_to_response(&c))
                        .map_err(to_error)?,
                )),
                None => Err(ErrorData::invalid_params(
                    format!("No tenant config for workspace: {workspace}"),
                    Some(json!({ "http_status": 404 })),
                )),
            }
        } else {
            let all = s.tenant_config_storage.list_all().await.map_err(to_error)?;
            let responses: Vec<Value> = all
                .iter()
                .map(|c| {
                    serde_json::to_value(
                        crate::routes::tenant_config::config_to_response(c),
                    )
                    .unwrap_or(Value::Null)
                })
                .collect();
            Ok(json_result(Value::Array(responses)))
        }
    }

    #[tool(
        name = "rag_tenant_config_upsert",
        description = "Create or update a workspace tenant config. Masked ('****')/empty API keys preserve the existing value."
    )]
    async fn rag_tenant_config_upsert(
        &self,
        Parameters(req): Parameters<TenantUpsertReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let body = crate::routes::tenant_config::UpsertTenantConfigRequest::from(req);
        let config = crate::routes::tenant_config::apply_upsert(&self.services, &body)
            .await
            .map_err(to_error)?;
        Ok(json_result(
            serde_json::to_value(crate::routes::tenant_config::config_to_response(&config))
                .map_err(to_error)?,
        ))
    }

    #[tool(
        name = "rag_tenant_config_delete",
        description = "Delete a workspace tenant config (revert to global defaults)."
    )]
    async fn rag_tenant_config_delete(
        &self,
        Parameters(req): Parameters<TenantDeleteReq>,
    ) -> Result<CallToolResult, ErrorData> {
        let s = &self.services;
        let workspace = req.workspace.trim().to_string();
        if workspace.is_empty() {
            return Err(ErrorData::invalid_params("workspace is required", None));
        }
        if workspace == s.config.workspace {
            return Err(ErrorData::invalid_params(
                format!(
                    "Tenant config cannot be deleted for the default workspace '{workspace}'"
                ),
                None,
            ));
        }
        s.tenant_config_storage
            .delete(&workspace)
            .await
            .map_err(to_error)?;
        s.workspace_resolver.invalidate_workspace(&workspace).await;
        Ok(json_result(json!({
            "status": "success",
            "message": format!("Tenant config for workspace '{workspace}' deleted"),
        })))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for EasyRagMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "EasyRAG knowledge-base tools: rag_search for retrieval, \
             rag_ingest_text/rag_document_status/rag_list_documents/\
             rag_read_document/rag_delete_document/rag_batch_delete for \
             lifecycle, rag_tenant_config_* for per-workspace config.",
        )
    }
}

/// Poll a document's status to a terminal state (`processed`/`failed`) or give
/// up after a bounded window, returning the last observed status string.
async fn wait_for_terminal_status(
    services: &Arc<AppServices>,
    doc_id: &str,
    biz_id: &str,
) -> AppResult<String> {
    const MAX_POLLS: usize = 40;
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
    let mut last = "processing".to_string();
    for _ in 0..MAX_POLLS {
        if let Some(status) = services
            .doc_status_storage
            .get_by_id(doc_id, biz_id)
            .await?
        {
            if let Some(value) = status.get("status").and_then(Value::as_str) {
                last = value.to_string();
                if value == DocStatus::Processed.value() || value == DocStatus::Failed.value() {
                    return Ok(last);
                }
            }
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    Ok(last)
}
