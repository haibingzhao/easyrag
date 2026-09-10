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
use easyrag_core::util::hash::compute_mdhash_id;
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
