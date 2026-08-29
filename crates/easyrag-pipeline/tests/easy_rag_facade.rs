//! Contract tests for the EasyRAG facade (`EasyRAG.kt` port).

use async_trait::async_trait;
use easyrag_core::constants::GRAPH_FIELD_SEP;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DeletionStatus;
use easyrag_core::model::{QueryParam, QueryMode};
use easyrag_core::storage::traits::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::JsonMap;
use easyrag_pipeline::multimodal::TextQueryBackend;
use easyrag_pipeline::{EasyRAG, EasyRAGConfig, EasyRAGConfigBuilder};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
use serde_json::json;
/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------
// Mocks (same shapes as ingest_pipeline.rs)
// ---------------------------------------------------------------------

struct ConstFunc;

#[async_trait]
impl EmbeddingFunction for ConstFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        // Non-zero constant vector: cosine similarity 1.0 so chunks
        // always pass the storage threshold.
        Ok(texts.iter().map(|_| vec![1f32, 0f32, 0f32, 0f32]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ConstFunc))
}

/// Scripted LLM: returns queued responses in order, then the default.
struct MockLlm {
    responses: Mutex<VecDeque<String>>,
    default: String,
}

impl MockLlm {
    fn with_responses<I: IntoIterator<Item = String>>(default: &str, responses: I) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            default: default.to_string(),
        }
    }
}

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default.clone()))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

const EXTRACTION_JSON: &str = r#"{"entities":[
    {"name":"Alice","type":"Person","description":"A researcher"},
    {"name":"The Lab","type":"Organization","description":"A research lab"}
],"relationships":[
    {"source":"Alice","target":"The Lab","keywords":"works at","description":"Alice works at the lab"}
]}"#;

struct FacadeEnv {
    rag: Arc<EasyRAG>,
    kv: Arc<InMemoryKVStorage>,
    vector: Arc<InMemoryVectorStorage>,
    graph: Arc<InMemoryGraphStorage>,
    doc_status: Arc<InMemoryDocStatusStorage>,
}

async fn setup(llm: Arc<dyn LlmService>) -> FacadeEnv {
    let kv = Arc::new(InMemoryKVStorage::new("kv", "facade"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vdb",
        "facade",
        test_embedding_func(),
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "facade"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "facade"));
    let config = EasyRAGConfigBuilder::new()
        .working_dir("./target/test-rag-storage-facade")
        .workspace("facade")
        .build();
    let rag = Arc::new(EasyRAG::new(
        config,
        llm,
        Arc::new(test_embedding_func()),
        vector.clone(),
        graph.clone(),
        kv.clone(),
        doc_status.clone(),
        None,
        None,
    ));
    FacadeEnv {
        rag,
        kv,
        vector,
        graph,
        doc_status,
    }
}

fn default_llm() -> Arc<dyn LlmService> {
    // One extraction call consumes the queued JSON; every later call
    // (query answering) gets the default.
    Arc::new(MockLlm::with_responses(
        "FACADE-ANSWER",
        vec![EXTRACTION_JSON.to_string()],
    ))
}

fn naive_param() -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;
    param
}

// ---------------------------------------------------------------------
// Lifecycle gating
// ---------------------------------------------------------------------

#[tokio::test]
async fn operations_require_initialization() {
    let env = setup(default_llm()).await;
    let err = env.rag.query("q", &naive_param()).await.unwrap_err();
    assert!(err.to_string().contains("not initialized"));
    let err = env.rag.insert("content", None, &JsonMap::new(), BIZ).await.unwrap_err();
    assert!(err.to_string().contains("not initialized"));

    // Idempotent initialize.
    env.rag.initialize().await.unwrap();
    env.rag.initialize().await.unwrap();
}

#[tokio::test]
async fn finalize_is_idempotent_and_reinitialize_works() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();
    env.rag.finalize().await.unwrap();
    env.rag.finalize().await.unwrap();
    // Operations gated again after finalize.
    assert!(env.rag.query("q", &naive_param()).await.is_err());
    env.rag.initialize().await.unwrap();
    assert!(env.rag.query("q", &naive_param()).await.is_ok());
}

// ---------------------------------------------------------------------
// insert
// ---------------------------------------------------------------------

#[tokio::test]
async fn insert_generates_content_hash_doc_id_and_processes() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();

    let content = "Alice works at the lab studying proteins.";
    let doc_id = env.rag.insert(content, None, &JsonMap::new(), BIZ).await.unwrap();
    // `docId ?: computeMdhashId(content)` with empty prefix.
    assert_eq!(doc_id, compute_mdhash_id(content, ""));

    let status = env.rag.get_doc_status(&doc_id, BIZ).await.unwrap().expect("status");
    assert_eq!(status.status.value(), "processed");
    assert!(status.chunks_count.unwrap_or(0) > 0);
    assert!(!status.chunks_list.is_empty());

    // Chunks landed in KV storage.
    let first_chunk = &status.chunks_list[0];
    assert!(env.kv.get_by_id(first_chunk, BIZ).await.unwrap().is_some());

    // Graph entities extracted from the scripted LLM response.
    let labels = env.rag.get_graph_labels(BIZ).await.unwrap();
    assert!(!labels.is_empty(), "entities must be extracted");
}

#[tokio::test]
async fn insert_batch_uses_explicit_and_hashed_ids() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();

    let docs = vec![
        ("first document".to_string(), JsonMap::new()),
        ("second document".to_string(), JsonMap::new()),
    ];
    let ids = env.rag.insert_batch(&docs, BIZ).await.unwrap();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids[0], compute_mdhash_id("first document", ""));
    assert_eq!(ids[1], compute_mdhash_id("second document", ""));
}

// ---------------------------------------------------------------------
// query
// ---------------------------------------------------------------------

#[tokio::test]
async fn query_returns_llm_answer_and_text_backend_bridge_matches() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();
    env.rag
        .insert("Alice works at the lab.", None, &JsonMap::new(), BIZ)
        .await
        .unwrap();

    let answer = env.rag.query("who works at the lab?", &naive_param()).await.unwrap();
    assert_eq!(answer, "FACADE-ANSWER");

    // The facade is the concrete TextQueryBackend of the multimodal
    // layer; the trait call must behave identically (cache hit here).
    let backend: Arc<dyn TextQueryBackend> = env.rag.clone();
    let bridged = backend.query("who works at the lab?", &naive_param()).await.unwrap();
    assert_eq!(bridged, answer);
}

// ---------------------------------------------------------------------
// delete_document
// ---------------------------------------------------------------------

#[tokio::test]
async fn delete_document_cleans_chunks_graph_and_status() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();

    let doc_id = env
        .rag
        .insert("Alice works at the lab studying proteins.", None, &JsonMap::new(), BIZ)
        .await
        .unwrap();
    let chunks_list: Vec<String> = env
        .doc_status
        .get_by_id(&doc_id, BIZ)
        .await
        .unwrap()
        .expect("doc status")
        .get("chunks_list")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    assert!(!chunks_list.is_empty());
    assert!(!env.graph.get_all_nodes(BIZ).await.unwrap().is_empty());

    let result = env.rag.delete_document(&doc_id, BIZ).await.unwrap();
    assert_eq!(result.status, DeletionStatus::Success);
    assert_eq!(result.status_code, 200);

    // Chunks removed from KV and vector storage.
    for chunk_id in &chunks_list {
        assert!(env.kv.get_by_id(chunk_id, BIZ).await.unwrap().is_none());
        assert!(env.vector.get_by_id(chunk_id, BIZ).await.unwrap().is_none());
    }
    // Graph fully cleaned (all sources belonged to this document).
    assert!(env.graph.get_all_nodes(BIZ).await.unwrap().is_empty());
    assert!(env.graph.get_all_edges(BIZ).await.unwrap().is_empty());
    // Document record and status gone.
    assert!(env.kv.get_by_id(&doc_id, BIZ).await.unwrap().is_none());
    assert!(env.doc_status.get_by_id(&doc_id, BIZ).await.unwrap().is_none());

    // Second delete reports NotFound with 404.
    let again = env.rag.delete_document(&doc_id, BIZ).await.unwrap();
    assert_eq!(again.status, DeletionStatus::NotFound);
    assert_eq!(again.status_code, 404);
}

#[tokio::test]
async fn delete_updates_shared_node_preserving_source_order() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();

    // Seed a shared node referenced by chunks of multiple documents.
    let sep_sources = ["x", "c1", "y"].join(GRAPH_FIELD_SEP);
    env.graph
        .upsert_node(
            "E1",
            json!({"id": "E1", "source_id": sep_sources}).as_object().unwrap().clone(),
        BIZ)
        .await
        .unwrap();
    env.graph
        .upsert_node(
            "E2",
            json!({"id": "E2", "source_id": "z"}).as_object().unwrap().clone(),
        BIZ)
        .await
        .unwrap();
    // Edge referencing a deleted chunk must be removed entirely
    //
    env.graph
        .upsert_edge(
            "E1",
            "E2",
            json!({"source": "E1", "target": "E2", "source_id": "c1"})
                .as_object()
                .unwrap()
                .clone(),
        BIZ)
        .await
        .unwrap();

    // Seed a doc status record whose chunks_list contains only c1.
    let mut record = JsonMap::new();
    record.insert("status".to_string(), json!("processed"));
    record.insert("chunks_list".to_string(), json!(["c1"]));
    let mut batch = HashMap::new();
    batch.insert("doc-seeded".to_string(), record);
    env.doc_status.upsert(batch, BIZ).await.unwrap();

    let result = env.rag.delete_document("doc-seeded", BIZ).await.unwrap();
    assert_eq!(result.status, DeletionStatus::Success);

    // Node survives with remaining sources in original order
    //
    let nodes = env.graph.get_all_nodes(BIZ).await.unwrap();
    let e1 = nodes.iter().find(|n| n.get("id").and_then(|v| v.as_str()) == Some("E1"));
    let e1 = e1.expect("shared node must survive");
    assert_eq!(
        e1.get("source_id").and_then(|v| v.as_str()),
        Some(["x", "y"].join(GRAPH_FIELD_SEP).as_str())
    );
    // Edge referencing the deleted chunk is gone.
    assert!(env.graph.get_all_edges(BIZ).await.unwrap().is_empty());
}

// ---------------------------------------------------------------------
// drop
// ---------------------------------------------------------------------

#[tokio::test]
async fn drop_reports_all_four_backends_and_clears_data() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();
    env.rag
        .insert("Alice works at the lab.", None, &JsonMap::new(), BIZ)
        .await
        .unwrap();

    let results = EasyRAG::drop(&env.rag, BIZ).await.unwrap();
    assert_eq!(results.len(), 4);
    for key in ["vector", "graph", "kv", "docStatus"] {
        assert_eq!(results.get(key).map(|s| s.as_str()), Some("success"), "key={key}");
    }
    assert!(env.graph.get_all_nodes(BIZ).await.unwrap().is_empty());
    assert!(env.doc_status.get_by_id("anything", BIZ).await.unwrap().is_none());
}

// ---------------------------------------------------------------------
// knowledge graph defaults
// ---------------------------------------------------------------------

#[tokio::test]
async fn knowledge_graph_default_uses_defaults() {
    let env = setup(default_llm()).await;
    env.rag.initialize().await.unwrap();
    env.rag
        .insert("Alice works at the lab.", None, &JsonMap::new(), BIZ)
        .await
        .unwrap();

    let kg = env.rag.get_knowledge_graph_default(Some("Alice"), BIZ).await.unwrap();
    assert!(!kg.nodes.is_empty());
}

// ---------------------------------------------------------------------
// config builder
// ---------------------------------------------------------------------

#[test]
fn config_builder_overrides_defaults() {
    let config = EasyRAGConfigBuilder::new()
        .working_dir("/tmp/rag")
        .workspace("ws")
        .llm_model("qwen-max")
        .embedding_dim(1024)
        .chunk_size(500)
        .storage_type("postgres")
        .language("Chinese")
        .build();
    assert_eq!(config.working_dir, "/tmp/rag");
    assert_eq!(config.workspace, "ws");
    assert_eq!(config.llm_model, "qwen-max");
    assert_eq!(config.embedding_dim, 1024);
    assert_eq!(config.chunk_size, 500);
    assert_eq!(config.storage_type, "postgres");
    assert_eq!(config.language, "Chinese");
    // Untouched fields keep defaults.
    assert_eq!(config.chunk_overlap_size, 100);
    assert_eq!(config.default_top_k, 40);
    let default = EasyRAGConfig::default();
    assert_eq!(default.llm_timeout_seconds, 120);
}
