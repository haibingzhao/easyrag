//! MCP Streamable HTTP wire contract.
//!
//! Verifies the stateless `/api/mcp` endpoint end to end through the real
//! router (CORS + JWT layers included): JSON-RPC `tools/list` and `tools/call`
//! round-trips, the 2026-07-28 stateless guarantees (no `Mcp-Session-Id`
//! header, per-request `_meta`), and auth gating that reuses the existing JWT
//! middleware — proving the design adds a surface without touching auth code.

#![cfg(feature = "mcp")]

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use easyrag_core::constants::DEFAULT_BIZ_ID;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageBundle, VectorStorage,
};
use easyrag_pipeline::{
    EasyRAG, EasyRAGConfigBuilder, IngestionPipeline, QueryEngine, TextParserRegistry,
};
use easyrag_server::auth::{JwtAuthProvider, LoginRateLimiter};
use easyrag_server::service::{
    AuthService, DocumentLifecycleService, ResolvedScope, WorkspaceContext, WorkspaceResolver,
};
use easyrag_server::state::AppServices;
use easyrag_server::{build_router, AppResult, AutoScanStats, ServerConfig};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage,
    InMemoryTenantConfigStorage, InMemoryUserStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
use http_body_util::BodyExt;
use rmcp::model::{
    CallToolRequest, CallToolRequestParams, ClientCapabilities, ClientJsonRpcMessage, ClientRequest,
    Implementation, ListToolsRequest, ProtocolVersion, RequestId, RequestMetaObject,
};
use serde_json::{json, Map, Value};
use std::sync::Arc;
use tower::ServiceExt;

// ---------------------------------------------------------------------
// Mocks (mirror api_contract.rs)
// ---------------------------------------------------------------------

struct ConstFunc;

#[async_trait]
impl EmbeddingFunction for ConstFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1f32, 0f32, 0f32, 0f32]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ConstFunc))
}

struct MockLlm;

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok("MOCK-ANSWER".to_string())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

struct DefaultResolver {
    context: WorkspaceContext,
}

#[async_trait]
impl WorkspaceResolver for DefaultResolver {
    async fn resolve(
        &self,
        _workspace: Option<&str>,
        biz_id: Option<&str>,
    ) -> AppResult<ResolvedScope> {
        let normalized = biz_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_BIZ_ID)
            .to_string();
        Ok(ResolvedScope {
            context: self.context.clone(),
            biz_id: normalized,
        })
    }
}

// ---------------------------------------------------------------------
// Environment (always mounts MCP; auth toggle drives the gating assertions)
// ---------------------------------------------------------------------

struct Env {
    router: axum::Router,
    jwt: Arc<JwtAuthProvider>,
    _temp_dir: tempfile::TempDir,
}

async fn build_env(auth_enabled: bool) -> Env {
    let temp_dir = tempfile::tempdir().unwrap();
    let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
    let embedding = Arc::new(test_embedding_func());
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vdb",
        "mcp",
        test_embedding_func(),
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "mcp"));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "mcp"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "mcp"));

    let config = EasyRAGConfigBuilder::new()
        .working_dir(temp_dir.path().to_str().unwrap())
        .workspace("mcp")
        .build();
    let easy_rag = Arc::new(EasyRAG::new(
        config,
        llm.clone(),
        embedding.clone(),
        vector.clone() as Arc<dyn VectorStorage>,
        graph.clone() as Arc<dyn GraphStorage>,
        kv.clone() as Arc<dyn KVStorage>,
        doc_status.clone() as Arc<dyn DocStatusStorage>,
        None,
        None,
    ));
    easy_rag.initialize().await.unwrap();

    let query_engine = Arc::new(QueryEngine::new(
        llm.clone(),
        vector.clone() as Arc<dyn VectorStorage>,
        graph.clone() as Arc<dyn GraphStorage>,
        kv.clone() as Arc<dyn KVStorage>,
    ));
    let pipeline = Arc::new(IngestionPipeline::new(
        llm.clone(),
        kv.clone() as Arc<dyn KVStorage>,
        vector.clone() as Arc<dyn VectorStorage>,
        graph.clone() as Arc<dyn GraphStorage>,
        doc_status.clone() as Arc<dyn DocStatusStorage>,
        embedding,
    ));
    let lifecycle_service = Arc::new(DocumentLifecycleService::new(
        pipeline.clone(),
        doc_status.clone() as Arc<dyn DocStatusStorage>,
        kv.clone() as Arc<dyn KVStorage>,
        vector.clone() as Arc<dyn VectorStorage>,
        graph.clone() as Arc<dyn GraphStorage>,
    ));
    let resolver = Arc::new(DefaultResolver {
        context: WorkspaceContext {
            bundle: StorageBundle {
                workspace: "mcp".to_string(),
                kv_storage: kv.clone() as Arc<dyn KVStorage>,
                vector_storage: vector.clone() as Arc<dyn VectorStorage>,
                graph_storage: graph.clone() as Arc<dyn GraphStorage>,
                doc_status_storage: doc_status.clone() as Arc<dyn DocStatusStorage>,
            },
            pipeline: pipeline.clone(),
            query_engine: query_engine.clone(),
            structure_strategy: None,
        },
    });
    let workspace_resolver: Arc<dyn WorkspaceResolver> = resolver.clone();

    let mut config = ServerConfig::default();
    config.auth_enabled = auth_enabled;
    // The only knob that turns the MCP sub-router on inside `build_router`.
    config.mcp_enabled = true;
    let jwt_provider = Arc::new(JwtAuthProvider::new("test-secret", 24));
    let jwt = auth_enabled.then(|| Arc::clone(&jwt_provider));
    let auth_service = Arc::new(AuthService::new(
        Arc::new(InMemoryUserStorage::new()),
        Arc::clone(&jwt_provider),
    ));
    auth_service.ensure_default_admin().await.unwrap();

    let services = Arc::new(AppServices {
        config,
        easy_rag,
        query_engine,
        structure_strategy: None,
        pipeline,
        pipeline_service: None,
        vector_storage: vector as Arc<dyn VectorStorage>,
        graph_storage: graph as Arc<dyn GraphStorage>,
        kv_storage: kv as Arc<dyn KVStorage>,
        doc_status_storage: doc_status as Arc<dyn DocStatusStorage>,
        parser_registry: Arc::new(TextParserRegistry::default()),
        lifecycle_service,
        workspace_resolver,
        rag_anything: None,
        jwt,
        rate_limiter: Arc::new(LoginRateLimiter::new(5, 1000)),
        auth_service,
        tenant_config_storage: Arc::new(InMemoryTenantConfigStorage::new()),
        auto_scan_stats: Arc::new(AutoScanStats::new()),
    });
    Env {
        router: build_router(services),
        jwt: jwt_provider,
        _temp_dir: temp_dir,
    }
}

// ---------------------------------------------------------------------
// Stateless JSON-RPC request builders (2026-07-28 per-request `_meta`)
// ---------------------------------------------------------------------

fn stateless_meta() -> RequestMetaObject {
    RequestMetaObject::with_client_context(
        ProtocolVersion::V_2026_07_28,
        Implementation::new("mcp-contract-test", "1.0.0"),
        ClientCapabilities::default(),
    )
}

/// Serialize an rmcp message to the exact wire bytes the server expects, and
/// pair it with the mandatory headers the 2026-07-28 stateless path validates:
/// `MCP-Protocol-Version` (matches the per-request `_meta`), plus the SEP-2243
/// `Mcp-Method` / `Mcp-Name` standard headers that must agree with the body.
fn encode_request(msg: ClientJsonRpcMessage, method: &str, name: Option<&str>) -> (Vec<u8>, HeaderMap) {
    let body = serde_json::to_vec(&msg).unwrap();
    let mut headers = HeaderMap::new();
    // The Streamable HTTP tower service rejects requests without a Host header
    // (DNS-rebinding guard); `disable_allowed_hosts()` relaxes the value check
    // but the header itself is still mandatory. `oneshot` skips hyper's normal
    // Host synthesis, so provide one.
    headers.insert("host", "localhost:8020".parse().unwrap());
    headers.insert("content-type", "application/json".parse().unwrap());
    headers.insert(
        "accept",
        "application/json, text/event-stream".parse().unwrap(),
    );
    headers.insert(
        "mcp-protocol-version",
        ProtocolVersion::V_2026_07_28.as_str().parse().unwrap(),
    );
    headers.insert("mcp-method", method.parse().unwrap());
    if let Some(name) = name {
        headers.insert("mcp-name", name.parse().unwrap());
    }
    (body, headers)
}

fn list_tools_request() -> (Vec<u8>, HeaderMap) {
    let mut request = ListToolsRequest {
        method: Default::default(),
        params: None,
        extensions: Default::default(),
    };
    request.extensions.insert(stateless_meta());
    let msg = ClientJsonRpcMessage::request(
        ClientRequest::ListToolsRequest(request),
        RequestId::Number(1),
    );
    encode_request(msg, "tools/list", None)
}

fn call_tool_request(id: i64, name: &str, arguments: Map<String, Value>) -> (Vec<u8>, HeaderMap) {
    let params = CallToolRequestParams::new(name.to_string()).with_arguments(arguments);
    let mut request = CallToolRequest::new(params);
    request.extensions.insert(stateless_meta());
    let msg = ClientJsonRpcMessage::request(
        ClientRequest::CallToolRequest(request),
        RequestId::Number(id),
    );
    encode_request(msg, "tools/call", Some(name))
}

async fn post_mcp(env: &Env, body: Vec<u8>, headers: HeaderMap) -> (StatusCode, HeaderMap, Value) {
    let mut builder = Request::builder().method("POST").uri("/api/mcp");
    for (k, v) in headers.iter() {
        builder = builder.header(k, v);
    }
    let req = builder.body(Body::from(body)).unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, headers, json)
}

/// Unwrap a successful `tools/call` result into the embedded JSON projection
/// string (the tools return a single text content block holding JSON).
fn tool_result_json(response: &Value) -> Value {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("expected tool text content, got {response}"));
    serde_json::from_str(text).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------
// tools/list
// ---------------------------------------------------------------------

#[tokio::test]
async fn tools_list_exposes_full_surface_and_is_stateless() {
    let env = build_env(false).await;
    let (body, headers) = list_tools_request();
    let (status, resp_headers, json) = post_mcp(&env, body, headers).await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    let names: Vec<&str> = json["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    for expected in [
        "rag_health",
        "rag_search",
        "rag_ingest_text",
        "rag_document_status",
        "rag_read_document",
        "rag_list_documents",
        "rag_delete_document",
        "rag_batch_delete",
        "rag_tenant_config_get",
        "rag_tenant_config_upsert",
        "rag_tenant_config_delete",
    ] {
        assert!(names.contains(&expected), "missing tool {expected} in {names:?}");
    }

    // Stateless guarantee: the 2026-07-28 path never issues a session header.
    assert!(
        resp_headers.get("mcp-session-id").is_none(),
        "stateless response must not carry Mcp-Session-Id"
    );
}

// ---------------------------------------------------------------------
// tools/call: rag_search
// ---------------------------------------------------------------------

#[tokio::test]
async fn tools_call_rag_search_returns_projection() {
    let env = build_env(false).await;
    let mut args = Map::new();
    args.insert("query".into(), json!("hello world"));
    let (body, headers) = call_tool_request(2, "rag_search", args);
    let (status, resp_headers, json) = post_mcp(&env, body, headers).await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    let projection = tool_result_json(&json);
    // Empty knowledge base: naive retrieval yields an (empty) chunks array, not
    // an error, and the projection keys match what easyai's client reads.
    assert!(
        projection.get("chunks").is_some(),
        "expected chunks key, got {projection}"
    );
    assert!(resp_headers.get("mcp-session-id").is_none());
}

// ---------------------------------------------------------------------
// tools/call: rag_ingest_text
// ---------------------------------------------------------------------

#[tokio::test]
async fn tools_call_rag_ingest_text_returns_doc_id() {
    let env = build_env(false).await;
    let mut args = Map::new();
    args.insert("text".into(), json!("EasyRAG MCP contract document body."));
    args.insert("externalId".into(), json!("ext-mcp-1"));
    let (body, headers) = call_tool_request(3, "rag_ingest_text", args);
    let (status, _resp_headers, json) = post_mcp(&env, body, headers).await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    let projection = tool_result_json(&json);
    assert_eq!(projection["status"], "success");
    assert!(
        projection["docId"].is_string(),
        "expected string docId, got {projection}"
    );
}

// ---------------------------------------------------------------------
// tools/call: error path (missing required argument)
// ---------------------------------------------------------------------

#[tokio::test]
async fn tools_call_unknown_tool_is_rejected() {
    let env = build_env(false).await;
    let (body, headers) = call_tool_request(4, "no_such_tool", Map::new());
    let (_status, _resp_headers, json) = post_mcp(&env, body, headers).await;

    // An unknown tool surfaces as a JSON-RPC error (either a protocol error or
    // an isError tool result), never a success projection.
    let is_error = json.get("error").is_some()
        || json["result"]["isError"].as_bool().unwrap_or(false);
    assert!(is_error, "expected unknown tool to error, got {json}");
}

// ---------------------------------------------------------------------
// Auth gating: MCP reuses the JWT middleware because it lives under /api/
// ---------------------------------------------------------------------

#[tokio::test]
async fn auth_enabled_rejects_missing_bearer() {
    let env = build_env(true).await;
    let (body, headers) = list_tools_request();
    let (status, _resp_headers, _json) = post_mcp(&env, body, headers).await;

    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "unauthenticated /api/mcp must be gated by the shared JWT middleware"
    );
}

#[tokio::test]
async fn auth_enabled_accepts_valid_bearer() {
    let env = build_env(true).await;
    let token = env.jwt.generate_token("admin").unwrap();
    let (body, mut headers) = list_tools_request();
    headers.insert(
        "authorization",
        format!("Bearer {token}").parse().unwrap(),
    );
    let (status, resp_headers, json) = post_mcp(&env, body, headers).await;

    assert_eq!(status, StatusCode::OK, "body: {json}");
    assert!(json["result"]["tools"].is_array());
    assert!(resp_headers.get("mcp-session-id").is_none());
}
