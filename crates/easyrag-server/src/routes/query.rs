//! Query endpoints.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::routing::post;
use axum::{Json, Router};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::model::query_param::{QueryMode, QueryParam};

use crate::error::{AppError, AppResult};
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/query", post(query))
        .route("/api/query/data", post(query_data))
        .route("/api/query/stream", post(query_stream))
        .route("/api/query/structure", post(structure_query))
}

/// Request DTO. Wire naming is camelCase except `doc_id` / `use_agent_mode`
///
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRequest {
    pub query: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub chunk_top_k: Option<usize>,
    #[serde(default)]
    pub response_type: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub user_prompt: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub enable_rerank: Option<bool>,
    #[serde(default)]
    pub time_range_start: Option<i64>,
    #[serde(default)]
    pub time_range_end: Option<i64>,
    #[serde(default)]
    pub hl_keywords: Option<Vec<String>>,
    #[serde(default)]
    pub ll_keywords: Option<Vec<String>>,
    #[serde(default)]
    pub conversation_history: Option<Vec<HashMap<String, String>>>,
    #[serde(default)]
    pub only_need_context: Option<bool>,
    #[serde(default)]
    pub only_need_prompt: Option<bool>,
    #[serde(default)]
    pub include_references: Option<bool>,
    #[serde(default, rename = "doc_id")]
    pub doc_id: Option<String>,
    #[serde(default, rename = "use_agent_mode")]
    pub use_agent_mode: Option<bool>,
    // R4: exact-match metadata filters applied to retrieved chunks
    #[serde(default)]
    pub metadata_filters: Option<HashMap<String, String>>,
    // M5: file path prefix scoping cross-document structure queries
    #[serde(default)]
    pub structure_scope: Option<String>,
    // R2: request-level workspace routing (optional)
    #[serde(default)]
    pub workspace: Option<String>,
    // biz_id: request-level business line routing (optional, default "default")
    #[serde(default)]
    pub biz_id: Option<String>,
    // Anti-hallucination guardrail per-request overrides
    #[serde(default)]
    pub enable_verification: Option<bool>,
    #[serde(default)]
    pub enable_citation_validation: Option<bool>,
    #[serde(default)]
    pub enable_evidence_scoring: Option<bool>,
    #[serde(default)]
    pub enable_query_routing: Option<bool>,
}

/// REST boundary defaults differ from the domain `QueryParam::default`
///: missing topK -> 40,
/// chunkTopK -> 5, responseType -> "Multiple Paragraphs",
/// unknown/missing mode -> Hybrid.
fn build_query_param(request: &QueryRequest, biz_id: &str) -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = parse_query_mode(request.mode.as_deref());
    param.biz_id = biz_id.to_string();
    param.top_k = request.top_k.unwrap_or(40);
    param.chunk_top_k = request.chunk_top_k.unwrap_or(5);
    param.response_type = request
        .response_type
        .clone()
        .unwrap_or_else(|| "Multiple Paragraphs".to_string());
    param.user_prompt = request.user_prompt.clone();
    param.enable_rerank = request.enable_rerank.unwrap_or(true);
    param.time_range_start = request.time_range_start;
    param.time_range_end = request.time_range_end;
    param.hl_keywords = request.hl_keywords.clone().unwrap_or_default();
    param.ll_keywords = request.ll_keywords.clone().unwrap_or_default();
    param.conversation_history = request.conversation_history.clone().unwrap_or_default();
    param.only_need_context = request.only_need_context.unwrap_or(false);
    param.only_need_prompt = request.only_need_prompt.unwrap_or(false);
    param.doc_id = request.doc_id.clone();
    param.use_agent_mode = request.use_agent_mode.unwrap_or(false);
    param.metadata_filters = request.metadata_filters.clone();
    param.structure_scope = request.structure_scope.clone();
    // Anti-hallucination guardrail per-request overrides
    param.enable_verification = request.enable_verification;
    param.enable_citation_validation = request.enable_citation_validation;
    param.enable_evidence_scoring = request.enable_evidence_scoring;
    param.enable_query_routing = request.enable_query_routing;
    param
}

/// Unknown values fall back to Hybrid.
fn parse_query_mode(mode: Option<&str>) -> QueryMode {
    match mode.map(|m| m.to_lowercase()) {
        Some(m) => match m.as_str() {
            "local" => QueryMode::Local,
            "global" => QueryMode::Global,
            "hybrid" => QueryMode::Hybrid,
            "naive" => QueryMode::Naive,
            "mix" => QueryMode::Mix,
            "bypass" => QueryMode::Bypass,
            "structure" => QueryMode::Structure,
            _ => QueryMode::Hybrid,
        },
        None => QueryMode::Hybrid,
    }
}

async fn query(state: AppState, Json(request): Json<QueryRequest>) -> AppResult<Json<Value>> {
    // R2: resolve the query engine for a request-level workspace.
    let resolved = state
        .workspace_resolver
        .resolve(request.workspace.as_deref(), request.biz_id.as_deref())
        .await?;
    let engine = resolved.context.query_engine;
    let param = build_query_param(&request, &resolved.biz_id);
    if request.include_references.unwrap_or(false) {
        let result = engine
            .query_with_references(&request.query, &param)
            .await?;
        let references: Vec<Value> = result
            .references
            .iter()
            .map(|r| {
                json!({
                    "reference_id": r.reference_id,
                    "file_path": r.file_path,
                    "content": r.content,
                    "score": r.score,
                    "section_path": r.section_path,
                })
            })
            .collect();
        Ok(Json(json!({ "response": result.response, "references": references })))
    } else {
        let response = engine.query(&request.query, &param).await?;
        // serializes the null field.
        Ok(Json(json!({ "response": response, "references": Value::Null })))
    }
}

async fn query_data(
    state: AppState,
    Json(request): Json<QueryRequest>,
) -> AppResult<Json<Value>> {
    let resolved = state
        .workspace_resolver
        .resolve(request.workspace.as_deref(), request.biz_id.as_deref())
        .await?;
    let engine = resolved.context.query_engine;
    let param = build_query_param(&request, &resolved.biz_id);
    let result = engine.query_data(&request.query, &param).await?;
    Ok(Json(json!({
        "entities": result.entities,
        "relations": result.relations,
        "chunks": result.chunks,
    })))
}

/// SSE endpoint emitting data-only frames
/// `ServerSentEvent.builder<String>.data(chunk)`).
async fn query_stream(
    state: AppState,
    Json(request): Json<QueryRequest>,
) -> AppResult<Sse<BoxStream<'static, Result<Event, Infallible>>>> {
    let resolved = state
        .workspace_resolver
        .resolve(request.workspace.as_deref(), request.biz_id.as_deref())
        .await?;
    let engine = resolved.context.query_engine;
    let param = build_query_param(&request, &resolved.biz_id);
    let chunks = engine.query_stream(&request.query, &param);
    let events = chunks.map(|chunk| Ok::<_, Infallible>(Event::default().data(chunk)));
    Ok(Sse::new(events.boxed()))
}

async fn structure_query(
    state: AppState,
    Json(request): Json<QueryRequest>,
) -> AppResult<Json<Value>> {
    let resolved = state
        .workspace_resolver
        .resolve(request.workspace.as_deref(), request.biz_id.as_deref())
        .await?;
    let strategy = resolved
        .context
        .structure_strategy
        .or_else(|| state.structure_strategy.clone())
        .ok_or_else(|| {
            // IllegalStateException -> GlobalExceptionHandler -> 500.
            AppError::Core(easyrag_core::CoreError::Internal(
                "Structure query strategy not configured".to_string(),
            ))
        })?;
    let param = build_query_param(&request, &resolved.biz_id);
    let result = strategy.query(&request.query, &param).await?;
    let reasoning_chain: Vec<Value> = result
        .reasoning_chain
        .iter()
        .map(|s| {
            json!({
                "level": s.level,
                "thinking": s.thinking,
                "considered": s.sections_considered,
                "selected": s.sections_selected,
                "confidence": s.confidence,
            })
        })
        .collect();
    Ok(Json(json!({
        "answer": result.answer,
        "reasoning_chain": reasoning_chain,
        "referenced_sections": result.referenced_sections,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request_from(json: &str) -> QueryRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn request_binds_mixed_naming_and_boundary_defaults() {
        let req = request_from(r#"{"query":"q"}"#);
        let param = build_query_param(&req, "default");
        assert_eq!(param.mode, QueryMode::Hybrid); // missing -> Hybrid
        assert_eq!(param.top_k, 40);
        assert_eq!(param.chunk_top_k, 5);
        assert_eq!(param.response_type, "Multiple Paragraphs");
        assert!(param.enable_rerank);
        assert!(!param.use_agent_mode);

        // snake_case aliases + unknown mode fallback
        let req = request_from(
            r#"{"query":"q","mode":"bogus","doc_id":"d1","use_agent_mode":true,"topK":7}"#,
        );
        assert_eq!(req.doc_id.as_deref(), Some("d1"));
        let param = build_query_param(&req, "default");
        assert_eq!(param.mode, QueryMode::Hybrid); // unknown -> Hybrid
        assert_eq!(param.top_k, 7);
        assert!(param.use_agent_mode);
        assert_eq!(param.doc_id.as_deref(), Some("d1"));

        let req = request_from(r#"{"query":"q","mode":"mix","enableRerank":false}"#);
        let param = build_query_param(&req, "default");
        assert_eq!(param.mode, QueryMode::Mix);
        assert!(!param.enable_rerank);
    }
}
