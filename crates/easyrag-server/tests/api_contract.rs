//! End-to-end contract tests for the assembled router (tower oneshot).
//!
//! Verifies the wire contract against the controllers: response
//! shapes, error body `{status, message, correlation_id}`, auth
//! middleware gating, CORS headers and conditional route mounting.

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use easyrag_core::constants::{DEFAULT_BIZ_ID, STRUCTURE_INDEX_KEY_PREFIX};
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageBundle, VectorStorage,
};
use easyrag_core::util::hash::{compute_args_hash, compute_mdhash_id};
use easyrag_core::{JsonMap, JsonValue};
use easyrag_pipeline::{
    EasyRAG, EasyRAGConfigBuilder, IngestionPipeline, TextParserRegistry, QueryEngine,
    StructureQueryStrategy, TreeSearchEngine,
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
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tower::ServiceExt;

// ---------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------

struct ConstFunc;

/// Texts starting with this fail to embed, so a case can drive a degradation
/// path without a second mock embedding type.
const EMBED_FAIL_PREFIX: &str = "EMBED-FAIL";

#[async_trait]
impl EmbeddingFunction for ConstFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        if texts.iter().any(|t| t.starts_with(EMBED_FAIL_PREFIX)) {
            return Err(easyrag_core::CoreError::Internal(
                "embedding provider unavailable".to_string(),
            ));
        }
        Ok(texts.iter().map(|_| vec![1f32, 0f32, 0f32, 0f32]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ConstFunc))
}

struct MockLlm {
    default: String,
}

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok(self.default.clone())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

// ---------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------

/// Default-only workspace resolver for contract tests (no R2 routing).
/// Records every (workspace, biz_id) pair it receives for assertions and
/// returns the normalized biz_id alongside the context.
struct DefaultResolver {
    context: WorkspaceContext,
    scopes: std::sync::Mutex<Vec<(Option<String>, Option<String>)>>,
    /// Normalized biz_id returned to the handler, in call order.
    normalized: std::sync::Mutex<Vec<String>>,
}

#[async_trait]
impl WorkspaceResolver for DefaultResolver {
    async fn resolve(
        &self,
        workspace: Option<&str>,
        biz_id: Option<&str>,
    ) -> AppResult<ResolvedScope> {
        self.scopes
            .lock()
            .unwrap()
            .push((workspace.map(str::to_string), biz_id.map(str::to_string)));
        // Normalize like the real resolver: missing/blank -> "default".
        let normalized = biz_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_BIZ_ID)
            .to_string();
        self.normalized.lock().unwrap().push(normalized.clone());
        Ok(ResolvedScope {
            context: self.context.clone(),
            biz_id: normalized,
        })
    }
}

struct Env {
    router: axum::Router,
    resolver: Arc<DefaultResolver>,
    /// Storage handles the same router reads through, so a case can seed
    /// knowledge-base state (structure index, registry row, chunks) before
    /// issuing its request instead of reaching into `resolver.context`.
    kv: Arc<dyn KVStorage>,
    vector: Arc<dyn VectorStorage>,
    doc_status: Arc<dyn DocStatusStorage>,
    _temp_dir: tempfile::TempDir,
}

async fn build_env(auth_enabled: bool, with_structure: bool) -> Env {
    build_env_with_data_structure(auth_enabled, with_structure, false).await
}

/// `data_structure` opens the R8 rollout gate (`QUERY_DATA_STRUCTURE_ENABLED`)
/// on the engine, which is what lets `/api/query/data` actually run the tree
/// search instead of reporting `structure_disabled_by_config`.
async fn build_env_with_data_structure(
    auth_enabled: bool,
    with_structure: bool,
    data_structure: bool,
) -> Env {
    let temp_dir = tempfile::tempdir().unwrap();
    let llm: Arc<dyn LlmService> = Arc::new(MockLlm {
        default: "MOCK-ANSWER".to_string(),
    });
    let embedding = Arc::new(test_embedding_func());
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vdb",
        "contract",
        test_embedding_func(),
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "contract"));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "contract"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "contract"));

    let config = EasyRAGConfigBuilder::new()
        .working_dir(temp_dir.path().to_str().unwrap())
        .workspace("contract")
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

    // Built before the engine on purpose: production hands the engine the same
    // strategy the `/api/query/structure` handler resolves off the workspace
    // context. Assembling the engine first left `query_data` with no strategy at
    // all, so an R8 contract case could only ever reach the
    // `structure_strategy_unavailable` branch and pass for the wrong reason.
    //
    // `.with_doc_status_storage` matters for the same reason: without the
    // registry, the scoped/cross-document paths short-circuit before consulting
    // any seeded document.
    let structure_strategy = if with_structure {
        Some(Arc::new(
            StructureQueryStrategy::new(
                TreeSearchEngine::new(llm.clone()),
                None,
                kv.clone() as Arc<dyn KVStorage>,
                vector.clone() as Arc<dyn VectorStorage>,
                llm.clone(),
            )
            .with_doc_status_storage(doc_status.clone() as Arc<dyn DocStatusStorage>),
        ))
    } else {
        None
    };
    let mut query_engine = QueryEngine::new(
        llm.clone(),
        vector.clone() as Arc<dyn VectorStorage>,
        graph.clone() as Arc<dyn GraphStorage>,
        kv.clone() as Arc<dyn KVStorage>,
    )
    .with_data_structure(data_structure);
    if let Some(strategy) = &structure_strategy {
        query_engine = query_engine.with_structure_strategy(strategy.clone());
    }
    let query_engine = Arc::new(query_engine);
    // Handles for `Env`, cloned before the originals are moved into `AppServices`.
    let kv_handle: Arc<dyn KVStorage> = kv.clone();
    let vector_handle: Arc<dyn VectorStorage> = vector.clone();
    let doc_status_handle: Arc<dyn DocStatusStorage> = doc_status.clone();
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
                workspace: "contract".to_string(),
                kv_storage: kv.clone() as Arc<dyn KVStorage>,
                vector_storage: vector.clone() as Arc<dyn VectorStorage>,
                graph_storage: graph.clone() as Arc<dyn GraphStorage>,
                doc_status_storage: doc_status.clone() as Arc<dyn DocStatusStorage>,
            },
            pipeline: pipeline.clone(),
            query_engine: query_engine.clone(),
            structure_strategy: structure_strategy.clone(),
        },
        scopes: std::sync::Mutex::new(Vec::new()),
        normalized: std::sync::Mutex::new(Vec::new()),
    });
    let workspace_resolver: Arc<dyn WorkspaceResolver> = resolver.clone();

    let mut config = ServerConfig::default();
    config.auth_enabled = auth_enabled;
    config.query_data_structure_enabled = data_structure;
    // Provider is unconditional: the login endpoint issues
    // tokens even when the middleware filter chain is disabled.
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
        structure_strategy,
        pipeline,
        // Streaming ingestion is default-off; None keeps the batch path (and
        // every assertion in this contract suite) byte-for-byte unchanged.
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
        resolver,
        kv: kv_handle,
        vector: vector_handle,
        doc_status: doc_status_handle,
        _temp_dir: temp_dir,
    }
}

async fn get_json(env: &Env, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().uri(uri).body(Body::empty()).unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn post_json(env: &Env, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

// ---------------------------------------------------------------------
// Health + auth-status
// ---------------------------------------------------------------------

#[tokio::test]
async fn health_reports_status_and_configuration() {
    let env = build_env(false, false).await;
    let (status, body) = get_json(&env, "/api/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["version"], "1.0.0");
    assert!(body["timestamp"].is_string());
    assert_eq!(body["configuration"]["chunk_size"], 1200);
    assert_eq!(body["storage"]["type"], "json");
    assert_eq!(body["features"]["auth_enabled"], false);
}

#[tokio::test]
async fn auth_status_guest_token_rule() {
    // Auth disabled: guest_token "guest".
    let env = build_env(false, false).await;
    let (status, body) = get_json(&env, "/api/auth-status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["auth_required"], false);
    assert_eq!(body["guest_token"], "guest");

    // Auth enabled: guest_token null (serialized).
    let env = build_env(true, false).await;
    let (_status, body) = get_json(&env, "/api/auth-status").await;
    assert_eq!(body["auth_required"], true);
    assert!(body["guest_token"].is_null());
}

// ---------------------------------------------------------------------
// Ollama compatibility
// ---------------------------------------------------------------------

#[tokio::test]
async fn ollama_tags_version_chat_generate() {
    let env = build_env(false, false).await;

    let (status, body) = get_json(&env, "/api/version").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["version"], "1.0.0");

    let (status, body) = get_json(&env, "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["models"][0]["name"], "easyrag");

    let chat_req = json!({
        "model": "easyrag",
        "messages": [{"role": "user", "content": "hello"}]
    });
    let (status, body) = post_json(&env, "/api/chat", chat_req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["message"]["role"], "assistant");
    // Empty knowledge base: QueryEngine returns the no-context
    // fallback verbatim WITHOUT invoking the LLM.
    assert_eq!(
        body["message"]["content"],
        "Sorry, I'm not able to provide an answer to that question.[no-context]"
    );
    // Proof the LLM was not called: MockLlm would answer MOCK-ANSWER.
    assert_ne!(body["message"]["content"], "MOCK-ANSWER");
    assert_eq!(body["done"], true);
    assert_eq!(body["done_reason"], "stop");
    assert_eq!(body["total_duration"], 0);

    let gen_req = json!({"prompt": "hello"});
    let (status, body) = post_json(&env, "/api/generate", gen_req).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["response"].is_string());
    assert_eq!(body["done_reason"], "stop");
}

// ---------------------------------------------------------------------
// Query endpoints
// ---------------------------------------------------------------------

#[tokio::test]
async fn query_returns_response_with_null_references() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(&env, "/api/query", json!({"query": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
    // Empty knowledge base: exact fallback string, LLM untouched.
    assert_eq!(
        body["response"],
        "Sorry, I'm not able to provide an answer to that question.[no-context]"
    );
    assert_ne!(body["response"], "MOCK-ANSWER");
    // QueryResponse(references = null) serializes the null field.
    assert!(body["references"].is_null());

    // Bypass mode echoes the query verbatim without the LLM.
    let (status, body) = post_json(
        &env,
        "/api/query",
        json!({"query": "echo me", "mode": "bypass"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["response"], "echo me");

    // includeReferences=true -> references array instead of null.
    let (status, body) = post_json(
        &env,
        "/api/query",
        json!({"query": "hello", "includeReferences": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["references"].is_array());
}

#[tokio::test]
async fn query_data_returns_entities_relations_chunks() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(&env, "/api/query/data", json!({"query": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["entities"].is_array());
    assert!(body["relations"].is_array());
    assert!(body["chunks"].is_array());
}

/// Acceptance for the chunk `doc_id`: a document written through the ingest API
/// is addressable from a retrieval hit — the projected id is the same one the
/// registry lists, so citing what was retrieved costs no extra round trip.
#[tokio::test]
async fn query_data_chunk_doc_id_matches_the_registry_id() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "Deploy windows are frozen on Fridays and reopen on Mondays.",
            "filePath": "easyai/memory/deploy-windows.md",
            "externalId": "memory/deploy-windows.md",
            "options": {"skipKg": true},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let doc_id = body["docId"].as_str().unwrap_or_default().to_string();
    assert!(!doc_id.is_empty(), "{body}");

    // Inserting only registers the document; chunking and embedding run in the
    // R6 background task, so recall has to wait for it.
    let (status, body) =
        post_json(&env, &format!("/api/documents/{doc_id}/index"), json!({})).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let mut processed = false;
    for _ in 0..200 {
        let (status, body) = get_json(&env, &format!("/api/documents/status?docId={doc_id}")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_ne!(body["status"], "failed", "{body}");
        if body["status"] == "processed" {
            processed = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(processed, "document {doc_id} never reached a terminal status");

    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "frozen on Fridays", "mode": "naive", "chunkTopK": 5}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = body["chunks"].as_array().unwrap();
    assert!(!chunks.is_empty(), "{body}");
    for chunk in chunks {
        assert_eq!(chunk["doc_id"], doc_id, "{body}");
    }

    let (status, list) = get_json(&env, "/api/documents/list?page=1&pageSize=10").await;
    assert_eq!(status, StatusCode::OK, "{list}");
    assert!(
        list["documents"]
            .as_array()
            .unwrap()
            .iter()
            .any(|doc| doc["id"] == doc_id),
        "{list}"
    );
}

// ---------------------------------------------------------------------
// /api/query/data bizIds (multi-biz recall)
// ---------------------------------------------------------------------

#[tokio::test]
async fn query_data_biz_ids_reach_the_resolver_and_the_response_shape_holds() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(
        &env,
        "/api/query/data",
        // Bypass keeps the recall plan vector-only, so this assertion isolates
        // the bizIds signal from the keyword_search_unavailable marker the
        // fused plans fire on a deployment without a keyword index.
        json!({"query": "hello", "mode": "bypass", "bizIds": ["alpha", "beta"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["chunks"].is_array());
    // The first element rides the normal biz_id normalization path.
    assert_eq!(
        env.resolver.normalized.lock().unwrap().last().unwrap(),
        "alpha"
    );
    // Degradation signal hygiene: a valid multi-biz request without sections
    // is not a degradation, no matter which storage shape answers.
    assert_eq!(body["degraded"], false);
    assert_eq!(body["degradeReasons"], json!([]));
}

#[tokio::test]
async fn query_data_biz_ids_override_biz_id_after_trim_and_dedupe() {
    let env = build_env(false, false).await;
    let (status, _) = post_json(
        &env,
        "/api/query/data",
        json!({
            "query": "hello",
            "bizId": "legacy",
            "bizIds": [" alpha ", "beta", "alpha", ""]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Trimmed, deduped, blanks dropped; the first survivor replaces bizId.
    assert_eq!(
        env.resolver.normalized.lock().unwrap().last().unwrap(),
        "alpha"
    );
}

#[tokio::test]
async fn query_data_empty_biz_ids_falls_back_to_biz_id() {
    let env = build_env(false, false).await;
    let (status, _) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "hello", "bizId": "legacy", "bizIds": []}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        env.resolver.normalized.lock().unwrap().last().unwrap(),
        "legacy"
    );
}

#[tokio::test]
async fn query_data_invalid_biz_ids_element_is_a_400() {
    let env = build_env(false, false).await;
    // Space and quotes fail the scope-name charset rule: these values land in
    // SQL/Cypher predicates, so the boundary rejects before any storage call.
    let (status, _) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "hello", "bizIds": ["ok", "bad biz"]}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn query_data_biz_ids_with_sections_emit_the_structure_marker() {
    // The structure branch is out of scope for the multi-biz rollout: the
    // strategy still scopes by a single biz line, so the request must be told
    // the tree search did not run instead of silently answering from one line.
    let env = build_env_with_data_structure(false, true, true).await;
    seed_guide_document(&env).await;
    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "guide", "mode": "structure", "bizIds": ["a", "b"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sections"], json!([]));
    assert_eq!(body["degraded"], true);
    assert!(body["degradeReasons"]
        .as_array()
        .unwrap()
        .contains(&json!("structure_multi_biz_unsupported")));
}

// ---------------------------------------------------------------------
// /api/query/data structure retrieval (R8)
//
// The mode matrix and the degradation-marker enumeration these cases assert
// against live in `.docs/easyrag-data-endpoint-mode-matrix.md`.
// ---------------------------------------------------------------------

fn section_node(
    title: &str,
    level: usize,
    summary: Option<&str>,
    content_start: usize,
) -> SectionNode {
    SectionNode {
        title: title.to_string(),
        level,
        summary: summary.map(str::to_string),
        content_start,
        content_end: content_start + 100,
        children: Vec::new(),
    }
}

fn structure_index(doc_id: &str, sections: Vec<SectionNode>) -> DocumentStructureIndex {
    DocumentStructureIndex {
        doc_id: doc_id.to_string(),
        doc_title: Some("Guide".to_string()),
        doc_description: None,
        sections,
        total_chars: 1000,
        version: 1,
    }
}

/// Store a chunk under the content-hash id ingestion would have minted, so the
/// section join can reconstruct it on the way back.
async fn seed_chunk(env: &Env, doc_id: &str, content: &str, order: i64, path: &[&str]) -> String {
    let chunk_id = compute_mdhash_id(content, "chunk-");
    let mut metadata = JsonMap::new();
    metadata.insert(
        "section_path".into(),
        JsonValue::Array(
            path.iter()
                .map(|p| JsonValue::String((*p).to_string()))
                .collect(),
        ),
    );
    let mut record = JsonMap::new();
    record.insert("content".into(), JsonValue::String(content.to_string()));
    record.insert("full_doc_id".into(), JsonValue::String(doc_id.to_string()));
    record.insert("chunk_order_index".into(), JsonValue::from(order));
    record.insert("metadata".into(), JsonValue::Object(metadata));
    let mut batch = HashMap::new();
    batch.insert(chunk_id.clone(), record);
    env.kv.upsert(batch, DEFAULT_BIZ_ID).await.unwrap();
    chunk_id
}

/// Registry row for a processed document. `has_structure` and `chunks_list` are
/// the two fields structure retrieval reads; the rest satisfy deserialization.
async fn seed_registry_doc(env: &Env, doc_id: &str, file_path: &str, chunks_list: &[String]) {
    let mut row = JsonMap::new();
    row.insert(
        "content_summary".into(),
        JsonValue::String("summary".to_string()),
    );
    row.insert("content_length".into(), JsonValue::from(10i64));
    row.insert("file_path".into(), JsonValue::String(file_path.to_string()));
    row.insert(
        "status".into(),
        JsonValue::String(DocStatus::Processed.value().to_string()),
    );
    row.insert(
        "created_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert(
        "updated_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert("has_structure".into(), JsonValue::Bool(true));
    row.insert(
        "chunks_list".into(),
        JsonValue::Array(
            chunks_list
                .iter()
                .map(|c| JsonValue::String(c.clone()))
                .collect(),
        ),
    );
    let mut batch = HashMap::new();
    batch.insert(doc_id.to_string(), row);
    env.doc_status.upsert(batch, DEFAULT_BIZ_ID).await.unwrap();
}

/// A document that already went through the pipeline, as the similarity probe
/// sees one: the KV record it summarizes, the registry row it falls back to and
/// the chunk vector it recalls. `content_hash` matters — without it a re-write of
/// the same text takes the overwrite path (cascade delete) instead of the
/// idempotent skip, which would delete the very neighbour a case seeded.
/// Returns the docId the externalId derives to.
async fn seed_indexed_doc(
    env: &Env,
    external_id: &str,
    file_path: &str,
    content: &str,
    biz_id: &str,
) -> String {
    let doc_id = DocumentLifecycleService::derive_external_doc_id(external_id);
    let chunk_id = compute_mdhash_id(content, "chunk-");

    let mut doc = JsonMap::new();
    doc.insert("content".into(), JsonValue::String(content.to_string()));
    doc.insert("file_path".into(), JsonValue::String(file_path.to_string()));
    doc.insert(
        "external_id".into(),
        JsonValue::String(external_id.to_string()),
    );
    let mut kv_batch = HashMap::new();
    kv_batch.insert(doc_id.clone(), doc);
    env.kv.upsert(kv_batch, biz_id).await.unwrap();

    let mut chunk = JsonMap::new();
    chunk.insert("content".into(), JsonValue::String(content.to_string()));
    chunk.insert(
        "full_doc_id".into(),
        JsonValue::String(doc_id.clone()),
    );
    chunk.insert("file_path".into(), JsonValue::String(file_path.to_string()));
    let mut vector_batch = HashMap::new();
    vector_batch.insert(chunk_id.clone(), chunk);
    env.vector.upsert(vector_batch, biz_id).await.unwrap();

    let mut row = JsonMap::new();
    row.insert(
        "content_summary".into(),
        JsonValue::String(content.chars().take(100).collect()),
    );
    row.insert(
        "content_length".into(),
        JsonValue::from(content.chars().count() as u64),
    );
    row.insert("file_path".into(), JsonValue::String(file_path.to_string()));
    row.insert(
        "status".into(),
        JsonValue::String(DocStatus::Processed.value().to_string()),
    );
    row.insert(
        "created_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert(
        "updated_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert(
        "external_id".into(),
        JsonValue::String(external_id.to_string()),
    );
    row.insert(
        "content_hash".into(),
        JsonValue::String(compute_args_hash([content])),
    );
    row.insert(
        "chunks_list".into(),
        JsonValue::Array(vec![JsonValue::String(chunk_id)]),
    );
    let mut status_batch = HashMap::new();
    status_batch.insert(doc_id.clone(), row);
    env.doc_status.upsert(status_batch, biz_id).await.unwrap();

    doc_id
}

async fn seed_structure_index(env: &Env, doc_id: &str, index: &DocumentStructureIndex) {
    let mut record = JsonMap::new();
    record.insert("doc_id".into(), JsonValue::String(doc_id.to_string()));
    record.insert(
        "structure".into(),
        JsonValue::String(serde_json::to_string(index).unwrap()),
    );
    let mut batch = HashMap::new();
    batch.insert(format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"), record);
    env.kv.upsert(batch, DEFAULT_BIZ_ID).await.unwrap();
}

/// Seed a two-section document whose chunks are written out of order, so a body
/// assembled in lookup order rather than `chunk_order_index` order cannot pass.
async fn seed_guide_document(env: &Env) {
    let index = structure_index(
        "doc1",
        vec![
            section_node("Introduction", 1, Some("What this document is about"), 0),
            section_node("Installation", 1, Some("How to install"), 100),
        ],
    );
    seed_structure_index(env, "doc1", &index).await;
    let second = seed_chunk(env, "doc1", "intro body two", 1, &["Introduction"]).await;
    let first = seed_chunk(env, "doc1", "intro body one", 0, &["Introduction"]).await;
    seed_registry_doc(env, "doc1", "docs/guide.md", &[first, second]).await;
}

#[tokio::test]
async fn query_data_structure_mode_returns_sections() {
    let env = build_env_with_data_structure(false, true, true).await;
    seed_guide_document(&env).await;

    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "how do I install it", "mode": "structure"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let sections = body["sections"]
        .as_array()
        .expect("sections is always an array, empty when structure retrieval did not run");
    assert_eq!(sections.len(), 2);
    assert_eq!(sections[0]["docId"], "doc1");
    assert_eq!(sections[0]["filePath"], "docs/guide.md");
    assert_eq!(sections[0]["sectionPath"], "Introduction");
    assert_eq!(sections[0]["title"], "Introduction");
    assert_eq!(sections[0]["level"], 1);
    assert_eq!(sections[0]["summary"], "What this document is about");
    assert_eq!(sections[0]["content"], "intro body one\n\nintro body two");
    assert_eq!(sections[0]["contentTruncated"], false);
    assert_eq!(sections[0]["chunkIds"].as_array().unwrap().len(), 2);
    // `score` is an ordinal 1/(rank+1), not a similarity — pinning the exact
    // values is what stops it being silently redefined as one.
    assert_eq!(sections[0]["rank"], 0);
    assert_eq!(sections[0]["score"], 1.0);
    assert_eq!(sections[1]["rank"], 1);
    assert_eq!(sections[1]["score"], 0.5);
    assert_eq!(sections[1]["content"], "");

    assert_eq!(body["degraded"], false);
    assert_eq!(body["degradeReasons"], json!([]));
    assert!(body["reasoningChain"].is_array());
    // R8: this mode spends nothing on the KG pass and synthesises no answer.
    assert_eq!(body["entities"], json!([]));
    assert_eq!(body["relations"], json!([]));
    assert!(!body.as_object().unwrap().contains_key("answer"));
}

/// `includeSections` is what resolves the contradiction between the R8 text
/// (`mode=structure` retrieves structure) and the requirement's `mode=mix`
/// sample: the field defaults to "follow the mode" and can be forced on, so a
/// mix caller gets sections without every scoped query paying for a tree search.
#[tokio::test]
async fn query_data_include_sections_overrides_the_mode_default() {
    let env = build_env_with_data_structure(false, true, true).await;
    seed_guide_document(&env).await;

    // Forced on for a mode that would not ask for it.
    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "q", "mode": "mix", "includeSections": true}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sections"].as_array().unwrap().len(), 2);
    // The mix plan still fused recall and ran the KG pass — sections are added,
    // not substituted. This fixture wires no keyword index, so the fusion gap is
    // reported instead of silent.
    assert_eq!(body["degraded"], true);
    assert_eq!(
        body["degradeReasons"],
        json!(["keyword_search_unavailable"])
    );

    // Forced off for the mode that would ask for it.
    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "q", "mode": "structure", "includeSections": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sections"], json!([]));
    // Opting out is a choice, not a degradation.
    assert_eq!(body["degraded"], false);
    assert_eq!(body["degradeReasons"], json!([]));
}

#[tokio::test]
async fn query_data_structure_degrades_when_strategy_absent() {
    // Gate open, but the deployment shape wired no strategy: the caller asked
    // for structure retrieval and must be told it never ran.
    let env = build_env_with_data_structure(false, false, true).await;
    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "q", "mode": "structure"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sections"], json!([]));
    assert_eq!(body["degraded"], true);
    assert_eq!(
        body["degradeReasons"],
        json!(["structure_strategy_unavailable"])
    );
}

#[tokio::test]
async fn query_data_structure_disabled_by_default() {
    // The rollout gate defaults to closed, so a strategy being wired is not
    // enough: the request is answered without a tree search and says so.
    let env = build_env(false, true).await;
    seed_guide_document(&env).await;

    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "q", "mode": "structure"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["sections"], json!([]));
    assert_eq!(body["degraded"], true);
    assert_eq!(
        body["degradeReasons"],
        json!(["structure_disabled_by_config"])
    );

    // Signal hygiene: a mode that never asked for structure reports no
    // *structure* reason, even with the gate closed and a strategy present. It
    // does report the recall gap — `naive` recalls by fusion and this fixture
    // wires no keyword index — which is the marker doing its job, not noise.
    let (status, naive) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "q", "mode": "naive"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(naive["sections"], json!([]));
    assert_eq!(naive["reasoningChain"], json!([]));
    assert_eq!(naive["degraded"], true);
    assert_eq!(
        naive["degradeReasons"],
        json!(["keyword_search_unavailable"])
    );
}

#[tokio::test]
async fn query_data_default_mode_body_shape_unchanged() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(&env, "/api/query/data", json!({"query": "hello"})).await;
    assert_eq!(status, StatusCode::OK);
    // The three pre-R8 keys keep their shape, so an existing client is unaffected.
    assert!(body["entities"].is_array());
    assert!(body["relations"].is_array());
    assert!(body["chunks"].is_array());
    // The R8 keys are present but inert, so a client can read them
    // unconditionally. No mode was supplied, which REST normalizes to hybrid —
    // a plan that asks for keyword fusion this fixture cannot provide, so the
    // gap is reported rather than silently absorbed.
    assert_eq!(body["sections"], json!([]));
    assert_eq!(body["reasoningChain"], json!([]));
    assert_eq!(body["degraded"], true);
    assert_eq!(
        body["degradeReasons"],
        json!(["keyword_search_unavailable"])
    );
}

#[tokio::test]
async fn structure_endpoint_500_when_strategy_absent() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(&env, "/api/query/structure", json!({"query": "q"})).await;
    // IllegalStateException -> GlobalExceptionHandler -> 500 with
    // the fixed error body shape.
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(body["status"], 500);
    assert!(body["message"].is_string());
    assert!(body["correlation_id"].is_string());
}

#[tokio::test]
async fn structure_endpoint_success_with_strategy() {
    let env = build_env(false, true).await;
    let (status, body) = post_json(&env, "/api/query/structure", json!({"query": "q"})).await;
    // Empty knowledge base: no document can be located -> deterministic
    // fallback answer with the response shape.
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        body["answer"],
        "No document with structure index found for this query."
    );
    assert!(body["reasoning_chain"].is_array());
    assert!(body["referenced_sections"].is_array());
}

#[tokio::test]
async fn query_stream_is_sse() {
    let env = build_env(false, false).await;
    let req = Request::builder()
        .method("POST")
        .uri("/api/query/stream")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({"query": "hello", "mode": "bypass"})).unwrap(),
        ))
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(content_type.starts_with("text/event-stream"), "{content_type}");
}

// ---------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------

#[tokio::test]
async fn documents_insert_list_status_counts_delete() {
    let env = build_env(false, false).await;

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({"text": "Alice works at the lab."}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");
    let doc_id = body["docId"].as_str().unwrap().to_string();
    // docId shape is `doc-<32 hex>` (mdhash of content).
    assert!(doc_id.starts_with("doc-"), "{doc_id}");
    assert_eq!(doc_id.len(), 4 + 32);
    assert!(doc_id[4..].chars().all(|c| c.is_ascii_hexdigit()));

    // docId = mdhash(content), so re-inserting the same
    // text returns the same docId (idempotent upsert, still one pending).
    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({"text": "Alice works at the lab."}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["docId"], doc_id);

    let (status, body) = get_json(&env, "/api/documents/list?page=1&pageSize=10").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 1);
    assert_eq!(body["documents"].as_array().unwrap().len(), 1);
    assert_eq!(body["page"], 1);
    // PaginatedResponse wire shape is camelCase pageSize.
    assert_eq!(body["pageSize"], 10);

    let (status, body) = get_json(&env, "/api/documents/status_counts").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["counts"]["pending"], 1);
    assert_eq!(body["total"], 1);

    let req = Request::builder()
        .method("DELETE")
        .uri(format!("/api/documents/{doc_id}"))
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let (_status, body) = get_json(&env, "/api/documents/list?page=1&pageSize=10").await;
    assert_eq!(body["total"], 0);
    assert!(body["documents"].as_array().unwrap().is_empty());
}

/// The write path stores client metadata under `user_metadata`; a read-back must
/// echo it, otherwise easyai's `inspect` sees no remote checksum and re-ingests
/// every cycle.
#[tokio::test]
async fn documents_read_by_external_id_echoes_metadata() {
    let env = build_env(false, false).await;

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "Skill body used to check the metadata round trip.",
            "externalId": "ext-metadata-echo",
            "metadata": {"name": "demo", "checksum": "abc123"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = get_json(
        &env,
        "/api/documents/by-external-id?externalId=ext-metadata-echo",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["metadata"]["checksum"], "abc123");
    assert_eq!(body["metadata"]["name"], "demo");

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "Legacy payload without metadata.",
            "externalId": "ext-metadata-absent",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = get_json(
        &env,
        "/api/documents/by-external-id?externalId=ext-metadata-absent",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["metadata"], json!({}));
}

#[tokio::test]
async fn pipeline_status_reports_idle_with_counts() {
    let env = build_env(false, false).await;
    // Additive merge (BC-11): the endpoint keeps `status`/`counts` and folds in
    // the live batch flags (busy/cancel_requested/docs_processed/last_run_at),
    // the streaming counters (active_count/channel_len/total_processed) and a
    // `pending_count` derived from `counts`. With streaming default-off
    // (`pipeline_service` is None) the streaming counters read 0 and
    // `streaming` is false, but every field is present so the webui busy/pending
    // indicators work — previously `busy` was absent and the UI showed a
    // permanent "Idle".
    let (status, body) = get_json(&env, "/api/documents/pipeline_status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "idle");
    assert!(body["counts"].is_object());
    // Live batch flags (idle pipeline).
    assert_eq!(body["busy"], false);
    assert_eq!(body["cancel_requested"], false);
    assert_eq!(body["docs_processed"], 0);
    assert!(body["last_run_at"].is_null());
    // Streaming counters: present and zero with the flag off.
    assert_eq!(body["active_count"], 0);
    assert_eq!(body["channel_len"], 0);
    assert_eq!(body["total_processed"], 0);
    assert_eq!(body["streaming"], false);
    // pending_count derives from counts (no documents yet).
    assert_eq!(body["pending_count"], 0);
}

#[tokio::test]
async fn auto_scan_status_reports_disabled_with_empty_tasks() {
    let env = build_env(false, false).await;
    // BC-11 observability endpoint. The scanner is disabled by default in the
    // test config, so the always-available route reports enabled=false with
    // empty tasks/cycles. Also proves the literal path wins over the
    // `/api/documents/{docId}` wildcard (otherwise this would 404 as a doc id).
    let (status, body) = get_json(&env, "/api/documents/auto_scan_status").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["enabled"], false);
    assert!(body["tasks"].as_array().unwrap().is_empty());
    assert!(body["cycles"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn scan_dedups_unchanged_files_and_reinserts_changed_ones() {
    let env = build_env(false, false).await;
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("note.txt");
    std::fs::write(&file, "first version").unwrap();

    let scan = json!({"inputDir": dir.path().to_string_lossy(), "maxFiles": 10});
    let uri = "/api/documents/scan";

    // First scan ingests the only file.
    let (status, body) = post_json(&env, uri, scan.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");
    assert_eq!(body["count"], 1);
    assert_eq!(body["doc_ids"].as_array().unwrap().len(), 1);
    assert_eq!(body["message"], "Scanned 1 files, inserted 1 documents");

    // BC-8 (A-3): rescanning an unchanged directory must insert nothing.
    // Before the endpoint delegated to `pipeline::dir_scan::scan_directory`
    // this returned count=1 again, re-ingesting every file on every call.
    let (status, body) = post_json(&env, uri, scan.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");
    assert_eq!(body["count"], 0);
    assert!(body["doc_ids"].as_array().unwrap().is_empty());
    // `files_seen` still reports the directory contents; only inserts shrink.
    assert_eq!(body["message"], "Scanned 1 files, inserted 0 documents");

    // Changed content is a new version and must be re-inserted.
    std::fs::write(&file, "second version").unwrap();
    let (status, body) = post_json(&env, uri, scan).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["count"], 1);

    // Missing directory keeps its pre-existing error shape.
    let (status, body) = post_json(
        &env,
        uri,
        json!({"inputDir": dir.path().join("nope").to_string_lossy()}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "error");
    assert!(body["message"].as_str().unwrap().starts_with("Directory not found:"));
}

// ---------------------------------------------------------------------
// Graph CRUD
// ---------------------------------------------------------------------

#[tokio::test]
async fn graph_entity_relation_crud() {
    let env = build_env(false, false).await;

    let (status, body) = post_json(
        &env,
        "/api/graph/entity/create",
        json!({"name": "Alice", "entityType": "Person", "description": "A researcher"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");

    // Duplicate create -> status "error".
    let (status, body) = post_json(
        &env,
        "/api/graph/entity/create",
        json!({"name": "Alice"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "error");

    let (status, body) = post_json(
        &env,
        "/api/graph/entity/create",
        json!({"name": "The Lab", "entityType": "Organization"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");

    let (status, body) = get_json(&env, "/api/graph/entity/exists?name=Alice").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["exists"], true);

    let (status, body) = post_json(
        &env,
        "/api/graph/relation/create",
        json!({"source": "Alice", "target": "The Lab", "keywords": "works at"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "success");

    // Relation on missing entity -> status "error".
    let (status, body) = post_json(
        &env,
        "/api/graph/relation/create",
        json!({"source": "Alice", "target": "Nobody"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "error");

    let (status, body) = get_json(&env, "/api/graph/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["node_count"], 2);
    assert_eq!(body["edge_count"], 1);

    let (status, body) = get_json(&env, "/api/graph/node/Alice").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], "Alice");
    assert_eq!(body["degree"], 1);
    assert_eq!(body["edges"].as_array().unwrap().len(), 1);

    // Missing node -> {"error": "Node not found"}.
    let (status, body) = get_json(&env, "/api/graph/node/Unknown").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["error"], "Node not found");

    let (status, body) = get_json(&env, "/api/graph/labels").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 2);

    let req = Request::builder()
        .method("DELETE")
        .uri("/api/graph/relation/delete?source=Alice&target=The%20Lab")
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let req = Request::builder()
        .method("DELETE")
        .uri("/api/graph/entity/delete?name=Alice")
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let (_status, body) = get_json(&env, "/api/graph/stats").await;
    assert_eq!(body["node_count"], 1);
    assert_eq!(body["edge_count"], 0);
}

// ---------------------------------------------------------------------
// Conditional multimodal mounting
// ---------------------------------------------------------------------

#[tokio::test]
async fn multimodal_routes_absent_without_rag_anything() {
    let env = build_env(false, false).await;
    let req = Request::builder()
        .uri("/api/multimodal/status")
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------
// Auth middleware
// ---------------------------------------------------------------------

#[tokio::test]
async fn auth_middleware_blocks_protected_and_allows_whitelist() {
    let env = build_env(true, false).await;

    // Protected endpoint without token -> 401.
    let (status, body) = post_json(&env, "/api/query", json!({"query": "q"})).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["status"], 401);

    // Whitelisted endpoints stay open.
    let (status, _) = get_json(&env, "/api/health").await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = get_json(&env, "/api/auth-status").await;
    assert_eq!(status, StatusCode::OK);

    // Valid token passes the middleware.
    let jwt = JwtAuthProvider::new("test-secret", 24);
    let token = jwt.generate_token("admin").unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/api/query")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::from(
            serde_json::to_vec(&json!({"query": "q", "mode": "naive"})).unwrap(),
        ))
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------
// Login endpoint
// ---------------------------------------------------------------------

#[tokio::test]
async fn login_success_returns_token_usable_on_protected_routes() {
    let env = build_env(true, false).await;

    // Seeded admin/admin -> 200 + {token, username}.
    let (status, body) = post_json(
        &env,
        "/api/auth/login",
        json!({"username": "admin", "password": "admin"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["username"], "admin");
    let token = body["token"].as_str().expect("token string").to_string();

    // The issued token passes the auth middleware.
    let req = Request::builder()
        .method("POST")
        .uri("/api/query")
        .header("content-type", "application/json")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::from(
            serde_json::to_vec(&json!({"query": "q", "mode": "bypass"})).unwrap(),
        ))
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn login_rejects_bad_credentials_missing_fields_and_rate_limits() {
    let env = build_env(true, false).await;

    // Wrong password -> 401 with the fixed error body shape.
    let (status, body) = post_json(
        &env,
        "/api/auth/login",
        json!({"username": "admin", "password": "wrong"}),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["status"], 401);

    // Missing/blank fields -> 400 (does not consume rate-limit budget).
    let (status, _) = post_json(&env, "/api/auth/login", json!({"username": "admin"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = post_json(&env, "/api/auth/login", json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // build_env caps the limiter at 5 attempts/minute; without ConnectInfo
    // every attempt shares the "unknown:admin" key, so further attempts
    // must trip the 429 guard.
    let mut saw_429 = false;
    for _ in 0..10 {
        let (status, _) = post_json(
            &env,
            "/api/auth/login",
            json!({"username": "admin", "password": "wrong"}),
        )
        .await;
        if status == StatusCode::TOO_MANY_REQUESTS {
            saw_429 = true;
            break;
        }
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
    assert!(saw_429, "rate limiter must return 429 after the cap");
}

// ---------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------

#[tokio::test]
async fn cors_preflight_allows_any_origin_by_default() {
    let env = build_env(false, false).await;
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/api/query")
        .header("origin", "http://example.com")
        .header("access-control-request-method", "POST")
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("access-control-allow-origin").unwrap(),
        "*"
    );
}

// ---------------------------------------------------------------------
// biz_id routing
// ---------------------------------------------------------------------

#[tokio::test]
async fn omitted_biz_id_is_equivalent_to_default() {
    let env = build_env(false, false).await;
    let (status, _) = post_json(&env, "/api/query", json!({"query": "hello"})).await;
    assert_eq!(StatusCode::OK, status);
    // No workspace/biz_id on the wire -> resolver receives None/None and
    // answers from the default context with the normalized "default" biz.
    let scopes = env.resolver.scopes.lock().unwrap();
    assert!(scopes.contains(&(None, None)));
    drop(scopes);
    let normalized = env.resolver.normalized.lock().unwrap();
    assert!(
        normalized.iter().all(|b| b == DEFAULT_BIZ_ID),
        "omitted biz_id must normalize to default: {normalized:?}"
    );
}

#[tokio::test]
async fn biz_id_flows_from_query_and_graph_routes_to_resolver() {
    let env = build_env(false, false).await;
    let (status, _) = post_json(
        &env,
        "/api/query",
        json!({"query": "hello", "workspace": "wsX", "bizId": "bizX"}),
    )
    .await;
    assert_eq!(StatusCode::OK, status);
    // Graph routes accept request-level scope as well.
    let (status, _) =
        get_json(&env, "/api/graph/labels?workspace=wsY&bizId=bizY").await;
    assert_eq!(StatusCode::OK, status);
    let (status, _) =
        get_json(&env, "/api/graph/stats?workspace=wsY&bizId=bizY").await;
    assert_eq!(StatusCode::OK, status);

    let scopes = env.resolver.scopes.lock().unwrap();
    assert!(scopes.contains(&(Some("wsX".to_string()), Some("bizX".to_string()))));
    assert!(scopes.contains(&(Some("wsY".to_string()), Some("bizY".to_string()))));
    drop(scopes);
    // Explicit biz values pass through normalization unchanged.
    let normalized = env.resolver.normalized.lock().unwrap();
    assert!(normalized.contains(&"bizX".to_string()), "{normalized:?}");
    assert!(normalized.contains(&"bizY".to_string()), "{normalized:?}");
}

// ---------------------------------------------------------------------
// Tenant config admin API (Phase 6.8 guardrail)
//
// These lock the DTO layer in `routes/tenant_config.rs` — the camelCase wire,
// the api-key mask (first 4 chars + `****`, short keys fully masked), the
// upsert merge (`.or()` for plain fields, masked/empty sentinel preserves the
// stored key) and the create-time filter — so a `macro_rules!` regeneration of
// Request/Response/merge can be proven byte-for-byte equivalent.
// ---------------------------------------------------------------------

/// DELETE helper mirroring the inline style used elsewhere in this file.
async fn delete_json(env: &Env, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

#[tokio::test]
async fn tenant_config_create_masks_keys_and_uses_camel_case_wire() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({
            "workspace": "acme",
            "llmModel": "gpt-4o",
            "llmApiKey": "sk-secret-123456",
            "embeddingApiKey": "emb-key-999",
            "chunkSize": 800,
            "llmTemperature": 0.5,
            "llmMaxTokens": 2048,
            "rerankEnabled": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["workspace"], "acme");
    assert_eq!(body["llmModel"], "gpt-4o");
    // Mask = first 4 chars + "****".
    assert_eq!(body["llmApiKey"], "sk-s****");
    assert_eq!(body["embeddingApiKey"], "emb-****");
    assert_eq!(body["chunkSize"], 800);
    assert_eq!(body["llmTemperature"], 0.5);
    assert_eq!(body["llmMaxTokens"], 2048);
    assert_eq!(body["rerankEnabled"], true);
    // Timestamps are present as RFC3339 strings (values are non-deterministic).
    assert!(body["createdAt"].is_string());
    assert!(body["updatedAt"].is_string());
    // snake_case must not leak onto the wire.
    assert!(body.get("llm_api_key").is_none());

    // GET single returns the same masked projection.
    let (status, got) = get_json(&env, "/api/admin/tenant-config?workspace=acme").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(got["llmApiKey"], "sk-s****");
    assert_eq!(got["llmModel"], "gpt-4o");
}

#[tokio::test]
async fn tenant_config_merge_preserves_unset_fields_and_masked_key() {
    let env = build_env(false, false).await;
    let (status, _) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({
            "workspace": "acme",
            "llmModel": "gpt-4o",
            "llmApiKey": "sk-original-key",
            "chunkSize": 800
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Update sends the masked sentinel for the key (must preserve the stored
    // one), omits llmModel/chunkSize (must preserve via `.or`), adds a field.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({
            "workspace": "acme",
            "llmApiKey": "****",
            "embeddingModel": "text-embedding-3-large"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Stored key preserved (masks to the ORIGINAL prefix, not the sentinel).
    assert_eq!(body["llmApiKey"], "sk-o****");
    assert_eq!(body["llmModel"], "gpt-4o");
    assert_eq!(body["chunkSize"], 800);
    assert_eq!(body["embeddingModel"], "text-embedding-3-large");
}

#[tokio::test]
async fn tenant_config_update_with_real_key_overwrites_stored_key() {
    let env = build_env(false, false).await;
    // Create with an original key.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmApiKey": "sk-original-key"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["llmApiKey"], "sk-o****");

    // Update sends a REAL new key (not the masked sentinel): it must
    // OVERWRITE the stored one. This is the third merge branch — distinct
    // from create-filter (sentinel->null) and update-preserve
    // (sentinel/empty->keep old) — so a macro regeneration of the merge that
    // collapsed it into "always preserve" or "always overwrite" is caught.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmApiKey": "sk-rotated-999"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    // Mask now reflects the ROTATED prefix, proving overwrite not preserve.
    assert_eq!(body["llmApiKey"], "sk-r****");
}

#[tokio::test]
async fn tenant_config_short_key_fully_masked_and_empty_key_preserves() {
    let env = build_env(false, false).await;
    // A key of <= 4 chars masks to "****" with no prefix leak.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmApiKey": "abc"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["llmApiKey"], "****");

    // An empty-string key on update preserves the existing value.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmApiKey": ""}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["llmApiKey"], "****");
}

#[tokio::test]
async fn tenant_config_create_filters_sentinel_and_absent_keys_to_null() {
    let env = build_env(false, false).await;
    // On CREATE (no existing row) a "****"/"" key is filtered to None, and an
    // omitted key stays None — both serialize as null.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmApiKey": "****", "embeddingApiKey": ""}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["llmApiKey"].is_null());
    assert!(body["embeddingApiKey"].is_null());

    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "beta"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["llmApiKey"].is_null());
}

#[tokio::test]
async fn tenant_config_get_absent_is_404_and_list_returns_all() {
    let env = build_env(false, false).await;
    let (status, body) = get_json(&env, "/api/admin/tenant-config?workspace=ghost").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["status"], 404);

    // Empty list before any config exists.
    let (status, body) = get_json(&env, "/api/admin/tenant-config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 0);

    post_json(&env, "/api/admin/tenant-config", json!({"workspace": "acme"})).await;
    post_json(&env, "/api/admin/tenant-config", json!({"workspace": "beta"})).await;
    let (status, body) = get_json(&env, "/api/admin/tenant-config").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body.as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn tenant_config_delete_removes_config() {
    let env = build_env(false, false).await;
    post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "acme", "llmModel": "gpt-4o"}),
    )
    .await;
    let (status, _) = delete_json(&env, "/api/admin/tenant-config?workspace=acme").await;
    assert_eq!(status, StatusCode::OK);
    // Gone: a subsequent GET is 404.
    let (status, _) = get_json(&env, "/api/admin/tenant-config?workspace=acme").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn tenant_config_rejects_default_workspace() {
    let env = build_env(false, false).await;
    // The default workspace always uses the global config; writes are rejected
    // with 400 so a silent no-op write cannot happen.
    let (status, body) = post_json(
        &env,
        "/api/admin/tenant-config",
        json!({"workspace": "default"}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["status"], 400);

    let (status, body) =
        delete_json(&env, "/api/admin/tenant-config?workspace=default").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["status"], 400);
}

// ---------------------------------------------------------------------
// Similar-document probe (returnSimilar on /api/documents/text)
// ---------------------------------------------------------------------

/// A similar neighbour of the same shard is reported with everything the caller
/// needs to merge or delete it: identity, provenance, score and the body's head.
#[tokio::test]
async fn insert_text_similar_reports_an_indexed_neighbour() {
    let env = build_env(false, false).await;
    let neighbour = "User prefers terse answers and no trailing summaries during code review.";
    let doc_id = seed_indexed_doc(
        &env,
        "memory/feedback/terse-answers.md",
        "easyai/memory/feedback/terse-answers.md",
        neighbour,
        DEFAULT_BIZ_ID,
    )
    .await;

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "User wants terse answers, without summaries at the end of a review.",
            "filePath": "easyai/memory/feedback/terse.md",
            "externalId": "memory/feedback/terse.md",
            "returnSimilar": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "success");
    // The ingest itself is unchanged by the probe.
    assert_eq!(body["message"], "Document inserted");

    let similar = body["similar"]
        .as_array()
        .expect("similar is an array whenever it was requested");
    assert_eq!(similar.len(), 1, "{similar:?}");
    assert_eq!(similar[0]["docId"], doc_id);
    assert_eq!(similar[0]["externalId"], "memory/feedback/terse-answers.md");
    assert_eq!(
        similar[0]["filePath"],
        "easyai/memory/feedback/terse-answers.md"
    );
    // ConstFunc embeds every text to the same vector, so the cosine score is 1.
    assert!(similar[0]["score"].as_f64().unwrap() >= 0.999, "{similar:?}");
    assert_eq!(similar[0]["contentSummary"], neighbour);
}

/// Regression: the probe is opt-in, and an unrequested response carries no
/// `similar` key at all — not an empty one — so legacy callers see the exact
/// body they saw before.
#[tokio::test]
async fn insert_text_similar_absent_unless_requested() {
    let env = build_env(false, false).await;
    seed_indexed_doc(
        &env,
        "ext-neighbour",
        "neighbour.md",
        "An already indexed memory about deploy windows and freeze dates.",
        DEFAULT_BIZ_ID,
    )
    .await;

    for payload in [
        json!({"text": "A new memory about deploy windows and freeze dates."}),
        json!({
            "text": "Another memory about deploy windows and freeze dates.",
            "returnSimilar": false,
        }),
    ] {
        let (status, body) = post_json(&env, "/api/documents/text", payload).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "success");
        assert!(body.get("similar").is_none(), "{body}");
        assert_eq!(body["message"], "Document inserted");
        assert!(body["docId"].as_str().unwrap().starts_with("doc-"));
    }
}

/// Shard isolation: an identical text stored under another bizId must not
/// surface — the probe carries the resolved shard into the vector query.
#[tokio::test]
async fn insert_text_similar_stays_inside_the_biz_shard() {
    let env = build_env(false, false).await;
    let content = "A memory that lives in a completely different biz shard.";
    seed_indexed_doc(
        &env,
        "ext-other-shard",
        "other.md",
        content,
        "u_other_project",
    )
    .await;

    // Pin the reason an empty `similar` is the right answer: the chunk is
    // recallable, just not from this shard.
    assert_eq!(
        env.vector
            .query(content, 10, None, None, None, false, "u_other_project")
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(env
        .vector
        .query(content, 10, None, None, None, false, "u_my_project")
        .await
        .unwrap()
        .is_empty());

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            // Byte-identical to the seeded neighbour: only the shard keeps it out.
            "text": content,
            "externalId": "ext-mine",
            "bizId": "u_my_project",
            "returnSimilar": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["similar"], json!([]));
}

/// Self-exclusion on the idempotent-skip path, where the document's own chunks
/// are already indexed and would otherwise top its own list at 1.0.
#[tokio::test]
async fn insert_text_similar_excludes_the_document_itself() {
    let env = build_env(false, false).await;
    let content = "The staging deploy window is Tuesday 02:00 UTC, frozen before releases.";
    let doc_id = seed_indexed_doc(
        &env,
        "memory/project/deploy-window.md",
        "easyai/memory/project/deploy-window.md",
        content,
        DEFAULT_BIZ_ID,
    )
    .await;

    // The document's own chunk is in the store and recallable, so the empty
    // result below can only come from the self-exclusion.
    let recalled = env
        .vector
        .query(content, 10, None, None, None, false, DEFAULT_BIZ_ID)
        .await
        .unwrap();
    assert_eq!(recalled.len(), 1, "{recalled:?}");
    assert_eq!(
        recalled[0].get("full_doc_id").and_then(|v| v.as_str()),
        Some(doc_id.as_str())
    );

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": content,
            "externalId": "memory/project/deploy-window.md",
            "returnSimilar": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["docId"], doc_id);
    assert_eq!(body["message"], "Document unchanged, skipped reprocessing");
    assert_eq!(body["similar"], json!([]));
}

/// Degradation: a failing probe never fails the ingest, and the provider-side
/// cause stays out of the response body.
#[tokio::test]
async fn insert_text_similar_failure_degrades_to_an_empty_list() {
    let env = build_env(false, false).await;
    seed_indexed_doc(
        &env,
        "ext-neighbour",
        "neighbour.md",
        "An already indexed memory about deploy windows and freeze dates.",
        DEFAULT_BIZ_ID,
    )
    .await;

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            // Ingestion never embeds, so the sentinel only breaks the probe.
            "text": "EMBED-FAIL a memory whose probe cannot be embedded at all.",
            "externalId": "ext-broken-probe",
            "returnSimilar": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "success");
    assert!(body["docId"].as_str().unwrap().starts_with("doc-"));
    assert_eq!(body["similar"], json!([]));
    assert_eq!(body["message"], "Document inserted; similar search unavailable");
    assert!(!body.to_string().contains("embedding provider unavailable"), "{body}");
}

/// `similarTopK` bounds the result, and an out-of-range `similarThreshold` is
/// clamped instead of silently matching nothing.
#[tokio::test]
async fn insert_text_similar_honours_top_k_and_clamps_the_threshold() {
    let env = build_env(false, false).await;
    for (external_id, content) in [
        ("ext-1", "First indexed memory about the release checklist."),
        ("ext-2", "Second indexed memory about the release checklist."),
    ] {
        seed_indexed_doc(&env, external_id, "checklist.md", content, DEFAULT_BIZ_ID).await;
    }

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "A third memory about the release checklist for this project.",
            "externalId": "ext-3",
            "returnSimilar": true,
            "similarTopK": 1,
            // Above every possible cosine score: clamped to 1.0, not honored.
            "similarThreshold": 1.5,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let similar = body["similar"].as_array().unwrap();
    assert_eq!(similar.len(), 1, "{similar:?}");
    assert!(similar[0]["score"].as_f64().unwrap() >= 0.999, "{similar:?}");
}

/// A chunk whose document is gone must not become a `similar` entry with empty
/// identity — the caller cannot act on it.
#[tokio::test]
async fn insert_text_similar_drops_a_dangling_chunk() {
    let env = build_env(false, false).await;
    let mut chunk = JsonMap::new();
    chunk.insert(
        "content".into(),
        JsonValue::String("An orphaned chunk whose document was deleted.".to_string()),
    );
    chunk.insert(
        "full_doc_id".into(),
        JsonValue::String("doc-doesnotexist".to_string()),
    );
    let mut batch = HashMap::new();
    batch.insert(compute_mdhash_id("An orphaned chunk whose document was deleted.", "chunk-"), chunk);
    env.vector.upsert(batch, DEFAULT_BIZ_ID).await.unwrap();

    // Recall does find the chunk; what drops it is the missing document behind
    // it, not an empty result set.
    assert_eq!(
        env.vector
            .query(
                "An orphaned chunk whose document was deleted.",
                10,
                None,
                None,
                None,
                false,
                DEFAULT_BIZ_ID
            )
            .await
            .unwrap()
            .len(),
        1
    );

    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "A memory about an orphaned chunk whose document was deleted.",
            "externalId": "ext-probe",
            "returnSimilar": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["similar"], json!([]));
}

// ---------------------------------------------------------------------
// Memory archive layer (R1-R5)
// ---------------------------------------------------------------------

/// Insert the four archive fields in their canonical shape. Status rows always
/// carry all four; chunk rows mirror them so `query/data` can echo them.
fn put_archive_fields(
    map: &mut JsonMap,
    archived: bool,
    level: i64,
    ref_ids: &[&str],
    merged_into: Option<&str>,
) {
    map.insert("archived".into(), JsonValue::Bool(archived));
    map.insert("level".into(), JsonValue::from(level));
    map.insert(
        "ref_ids".into(),
        JsonValue::Array(
            ref_ids
                .iter()
                .map(|id| JsonValue::String((*id).to_string()))
                .collect(),
        ),
    );
    map.insert(
        "merged_into".into(),
        merged_into
            .map(|m| JsonValue::String(m.to_string()))
            .unwrap_or(JsonValue::Null),
    );
}

/// Seed one fully-indexed document carrying archive fields: KV doc, status row
/// and one chunk vector. Mirrors what the pipeline writes, without the async
/// queue. Returns the derived docId.
async fn seed_archive_doc(
    env: &Env,
    external_id: &str,
    content: &str,
    archived: bool,
    level: i64,
    ref_ids: &[&str],
    merged_into: Option<&str>,
) -> String {
    let doc_id = DocumentLifecycleService::derive_external_doc_id(external_id);
    let chunk_id = compute_mdhash_id(content, "chunk-");
    let file_path = format!("easyai/{external_id}.md");
    let create_time = 1_700_000_000i64;

    let mut doc = JsonMap::new();
    doc.insert("content".into(), JsonValue::String(content.to_string()));
    doc.insert("file_path".into(), JsonValue::String(file_path.clone()));
    doc.insert(
        "external_id".into(),
        JsonValue::String(external_id.to_string()),
    );
    doc.insert("create_time_epoch".into(), JsonValue::from(create_time));
    put_archive_fields(&mut doc, archived, level, ref_ids, merged_into);
    let mut kv_batch = HashMap::new();
    kv_batch.insert(doc_id.clone(), doc);
    env.kv.upsert(kv_batch, DEFAULT_BIZ_ID).await.unwrap();

    let mut chunk = JsonMap::new();
    chunk.insert("content".into(), JsonValue::String(content.to_string()));
    chunk.insert(
        "full_doc_id".into(),
        JsonValue::String(doc_id.clone()),
    );
    chunk.insert("file_path".into(), JsonValue::String(file_path.clone()));
    chunk.insert("create_time".into(), JsonValue::from(create_time));
    put_archive_fields(&mut chunk, archived, level, ref_ids, merged_into);
    let mut vector_batch = HashMap::new();
    vector_batch.insert(chunk_id.clone(), chunk);
    env.vector.upsert(vector_batch, DEFAULT_BIZ_ID).await.unwrap();

    let mut row = JsonMap::new();
    row.insert(
        "content_summary".into(),
        JsonValue::String(content.chars().take(100).collect()),
    );
    row.insert(
        "content_length".into(),
        JsonValue::from(content.chars().count() as u64),
    );
    row.insert("file_path".into(), JsonValue::String(file_path));
    row.insert(
        "status".into(),
        JsonValue::String(DocStatus::Processed.value().to_string()),
    );
    row.insert(
        "created_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert(
        "updated_at".into(),
        JsonValue::String("2026-01-01T00:00:00Z".to_string()),
    );
    row.insert(
        "external_id".into(),
        JsonValue::String(external_id.to_string()),
    );
    row.insert(
        "content_hash".into(),
        JsonValue::String(compute_args_hash([content])),
    );
    row.insert("chunks_count".into(), JsonValue::from(1u64));
    row.insert(
        "chunks_list".into(),
        JsonValue::Array(vec![JsonValue::String(chunk_id)]),
    );
    put_archive_fields(&mut row, archived, level, ref_ids, merged_into);
    let mut status_batch = HashMap::new();
    status_batch.insert(doc_id.clone(), row);
    env.doc_status.upsert(status_batch, DEFAULT_BIZ_ID).await.unwrap();

    doc_id
}

async fn patch_json(env: &Env, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("PATCH")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    let resp = env.router.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// R1: ingest accepts the four fields and every read path echoes them back.
#[tokio::test]
async fn ingest_accepts_archive_fields_and_echoes_them() {
    let env = build_env(false, false).await;
    let (status, body) = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "Merged memory body.",
            "externalId": "ext-archive-fields",
            "archived": true,
            "level": 2,
            "refIds": ["doc-a", "doc-b"],
            "mergedInto": "doc-parent",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let doc_id = body["docId"].as_str().unwrap().to_string();

    let (status, body) = get_json(
        &env,
        "/api/documents/by-external-id?externalId=ext-archive-fields",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["doc_id"], doc_id);
    assert_eq!(body["archived"], true);
    assert_eq!(body["level"], 2);
    assert_eq!(body["ref_ids"], json!(["doc-a", "doc-b"]));
    assert_eq!(body["merged_into"], "doc-parent");

    let (status, body) = get_json(&env, "/api/documents/list?page=1&pageSize=10").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["total"], 1);
    let listed = &body["documents"][0];
    assert_eq!(listed["archived"], true);
    assert_eq!(listed["level"], 2);
    assert_eq!(listed["refIds"], json!(["doc-a", "doc-b"]));
    assert_eq!(listed["mergedInto"], "doc-parent");
}

/// R2: `excludeArchived` is a storage-level filter on both recall and listing,
/// and the default (absent) keeps the pre-R2 result set byte-for-byte.
#[tokio::test]
async fn query_and_list_exclude_archived_at_storage_level() {
    let env = build_env(false, false).await;
    let _active_1 = seed_archive_doc(
        &env,
        "ext-active-1",
        "alpha memory about deploys",
        false,
        0,
        &[],
        None,
    )
    .await;
    let _active_2 = seed_archive_doc(
        &env,
        "ext-active-2",
        "beta memory about releases",
        false,
        0,
        &[],
        None,
    )
    .await;
    let archived = seed_archive_doc(
        &env,
        "ext-archived-1",
        "gamma memory about deploys",
        true,
        0,
        &[],
        Some("doc-parent"),
    )
    .await;

    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({"query": "memory about deploys", "mode": "naive", "chunkTopK": 10}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let all_chunks = body["chunks"].as_array().unwrap();
    assert_eq!(all_chunks.len(), 3, "{body}");
    // R1: every hit echoes the four archive fields; the legacy defaults are
    // materialized rather than absent.
    let archived_hit = all_chunks
        .iter()
        .find(|c| c["content"] == "gamma memory about deploys")
        .expect("archived chunk should be recallable without the flag");
    assert_eq!(archived_hit["archived"], true);
    assert_eq!(archived_hit["merged_into"], "doc-parent");
    for c in all_chunks.iter().filter(|c| c["archived"] != true) {
        assert_eq!(c["level"], 0);
        assert_eq!(c["ref_ids"], json!([]));
        assert!(c["merged_into"].is_null());
    }

    let (status, body) = post_json(
        &env,
        "/api/query/data",
        json!({
            "query": "memory about deploys",
            "mode": "naive",
            "chunkTopK": 10,
            "excludeArchived": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = body["chunks"].as_array().unwrap();
    assert_eq!(chunks.len(), 2, "{body}");
    let hit_contents: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c["content"].as_str())
        .collect();
    assert!(hit_contents.contains(&"alpha memory about deploys"));
    assert!(hit_contents.contains(&"beta memory about releases"));
    assert!(!hit_contents.contains(&"gamma memory about deploys"));

    let (status, body) =
        get_json(&env, "/api/documents/list?page=1&pageSize=10&excludeArchived=true").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["total"], 2, "{body}");
    assert_eq!(body["documents"].as_array().unwrap().len(), 2, "{body}");
    let listed_ids: Vec<&str> = body["documents"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|d| d["id"].as_str())
        .collect();
    assert!(!listed_ids.contains(&archived.as_str()));

    // Default: no exclusion, same as before the flag existed.
    let (status, body) = get_json(&env, "/api/documents/list?page=1&pageSize=10").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["total"], 3, "{body}");
}

/// R3: by-ids reads archived documents too, and a missing id is omitted rather
/// than failing the batch.
#[tokio::test]
async fn by_ids_reads_archived_docs_and_omits_missing_ids() {
    let env = build_env(false, false).await;
    let archived = seed_archive_doc(
        &env,
        "ext-byids-archived",
        "The archived source memory body.",
        true,
        1,
        &["doc-ref-1", "doc-ref-2"],
        Some("doc-merged"),
    )
    .await;

    let (status, body) = post_json(
        &env,
        "/api/documents/by-ids",
        json!({"docIds": [archived, "doc-does-not-exist"]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let docs = body["documents"].as_array().unwrap();
    assert_eq!(docs.len(), 1, "{body}");
    let doc = &docs[0];
    assert_eq!(doc["doc_id"], archived);
    assert_eq!(doc["external_id"], "ext-byids-archived");
    assert_eq!(doc["content"], "The archived source memory body.");
    assert_eq!(doc["status"], "processed");
    assert_eq!(doc["chunks_count"], 1);
    assert_eq!(doc["archived"], true);
    assert_eq!(doc["level"], 1);
    assert_eq!(doc["ref_ids"], json!(["doc-ref-1", "doc-ref-2"]));
    assert_eq!(doc["merged_into"], "doc-merged");
    assert!(doc["error_msg"].is_null());
    assert_eq!(doc["has_structure"], false);

    // Oversized batches fail fast instead of fanning out.
    let too_many: Vec<String> = (0..201).map(|i| format!("doc-{i}")).collect();
    let (status, _) = post_json(
        &env,
        "/api/documents/by-ids",
        json!({"docIds": too_many}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// R4: archive flips the flags only — no re-chunk/re-embedding — and the change
/// is visible to by-ids, recall and listing immediately.
#[tokio::test]
async fn archive_updates_flags_without_reindexing() {
    let env = build_env(false, false).await;
    let doc_id = seed_archive_doc(
        &env,
        "ext-archive-source",
        "A source memory absorbed by the merged one.",
        false,
        0,
        &[],
        None,
    )
    .await;

    // One chunk row before the archive call; the same row must survive it.
    assert_eq!(
        env.vector
            .query(
                "source memory",
                100,
                Some(&[1f32, 0f32, 0f32, 0f32]),
                None,
                None,
                false,
                DEFAULT_BIZ_ID,
            )
            .await
            .unwrap()
            .len(),
        1
    );

    let (status, body) = patch_json(
        &env,
        "/api/documents/archive",
        json!({
            "docIds": [doc_id, "doc-missing"],
            "archived": true,
            "mergedInto": "doc-merged",
            "level": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "{body}");
    assert_eq!(results[0]["docId"], doc_id);
    assert_eq!(results[0]["status"], "success");
    assert_eq!(results[0]["updated"], true);
    assert_eq!(results[1]["status"], "not_found");

    // The doc KV row is the source of truth for reprocessing; it must mirror
    // the archive update too.
    let kv_doc = env
        .kv
        .get_by_id(&doc_id, DEFAULT_BIZ_ID)
        .await
        .unwrap()
        .expect("kv doc exists");
    assert_eq!(kv_doc.get("archived").and_then(|v| v.as_bool()), Some(true));
    assert_eq!(
        kv_doc.get("merged_into").and_then(|v| v.as_str()),
        Some("doc-merged")
    );

    // Oversized archive batches fail fast.
    let too_many: Vec<String> = (0..201).map(|i| format!("doc-{i}")).collect();
    let (status, _) = patch_json(
        &env,
        "/api/documents/archive",
        json!({"docIds": too_many, "archived": true}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // Flags changed; content and chunk count did not.
    let (status, body) = post_json(
        &env,
        "/api/documents/by-ids",
        json!({"docIds": [doc_id]}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let doc = &body["documents"][0];
    assert_eq!(doc["archived"], true);
    assert_eq!(doc["level"], 1);
    assert_eq!(doc["merged_into"], "doc-merged");
    assert_eq!(doc["chunks_count"], 1);
    assert_eq!(doc["content"], "A source memory absorbed by the merged one.");

    // Vector row still exists (nothing re-embedded or deleted), but recall now
    // skips it when asked.
    assert_eq!(
        env.vector
            .query(
                "source memory",
                100,
                Some(&[1f32, 0f32, 0f32, 0f32]),
                None,
                None,
                false,
                DEFAULT_BIZ_ID,
            )
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(env
        .vector
        .query(
            "source memory",
            100,
            Some(&[1f32, 0f32, 0f32, 0f32]),
            None,
            None,
            true,
            DEFAULT_BIZ_ID,
        )
        .await
        .unwrap()
        .is_empty());
    let (status, body) =
        get_json(&env, "/api/documents/list?page=1&pageSize=10&excludeArchived=true").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["total"], 0, "{body}");

    // Unarchiving restores recall; still no reindex.
    let (status, body) = patch_json(
        &env,
        "/api/documents/archive",
        json!({"docIds": [doc_id], "archived": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["results"][0]["status"], "success");
    assert_eq!(
        env.vector
            .query(
                "source memory",
                100,
                Some(&[1f32, 0f32, 0f32, 0f32]),
                None,
                None,
                true,
                DEFAULT_BIZ_ID,
            )
            .await
            .unwrap()
            .len(),
        1
    );
}

/// R5: the docId derived from an externalId survives content changes and
/// archiving (it is a pure function of the key, never of the content).
#[tokio::test]
async fn external_id_doc_id_is_stable_across_upserts_and_archive() {
    let env = build_env(false, false).await;
    let first = post_json(
        &env,
        "/api/documents/text",
        json!({"text": "version one", "externalId": "ext-stable"}),
    )
    .await;
    assert_eq!(first.0, StatusCode::OK, "{}", first.1);
    let doc_id = first.1["docId"].as_str().unwrap().to_string();

    let second = post_json(
        &env,
        "/api/documents/text",
        json!({"text": "version two with different content", "externalId": "ext-stable"}),
    )
    .await;
    assert_eq!(second.0, StatusCode::OK, "{}", second.1);
    assert_eq!(second.1["docId"], doc_id);

    let archive = patch_json(
        &env,
        "/api/documents/archive",
        json!({"docIds": [doc_id], "archived": true}),
    )
    .await;
    assert_eq!(archive.0, StatusCode::OK, "{}", archive.1);

    let (status, body) = get_json(
        &env,
        "/api/documents/by-external-id?externalId=ext-stable",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["doc_id"], doc_id);
    assert_eq!(body["archived"], true);
}

/// Regression: an external-id content overwrite must inherit existing archive
/// flags when the caller does not resupply them.
#[tokio::test]
async fn external_id_content_update_preserves_archive_fields() {
    let env = build_env(false, false).await;
    let first = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "version one",
            "externalId": "ext-stable-archive",
            "archived": true,
            "level": 2,
            "refIds": ["doc-a"],
            "mergedInto": "doc-parent",
        }),
    )
    .await;
    assert_eq!(first.0, StatusCode::OK, "{}", first.1);
    let doc_id = first.1["docId"].as_str().unwrap().to_string();

    let second = post_json(
        &env,
        "/api/documents/text",
        json!({
            "text": "version two with completely different content",
            "externalId": "ext-stable-archive",
        }),
    )
    .await;
    assert_eq!(second.0, StatusCode::OK, "{}", second.1);
    assert_eq!(second.1["docId"], doc_id);

    let (status, body) = get_json(
        &env,
        "/api/documents/by-external-id?externalId=ext-stable-archive",
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["doc_id"], doc_id);
    assert_eq!(body["archived"], true);
    assert_eq!(body["level"], 2);
    assert_eq!(body["ref_ids"], json!(["doc-a"]));
    assert_eq!(body["merged_into"], "doc-parent");
}
