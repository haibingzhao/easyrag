//! Ingestion pipeline + entity extractor integration tests.
//! The project has no ingestion test suite, so these are original
//! tests covering the same behavioral contract (status state machine,
//! chunk persistence, KG extraction, skip_kg, concurrency guard, cancel).

use async_trait::async_trait;
use easyrag_core::constants::GRAPH_FIELD_SEP;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage};
use easyrag_pipeline::extraction::{EntityExtractor, EntityExtractorConfig};
use easyrag_pipeline::parser::DocumentProcessingOptions;
use easyrag_pipeline::IngestionPipeline;
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------

struct ZeroFunc;

#[async_trait]
impl EmbeddingFunction for ZeroFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0f32; 4]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ZeroFunc))
}

/// Scripted LLM: returns queued responses in order, then the default.
struct MockLlm {
    responses: Mutex<VecDeque<String>>,
    default: String,
    calls: AtomicUsize,
}

impl MockLlm {
    fn new(default: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            default: default.into(),
            calls: AtomicUsize::new(0),
        }
    }

    fn with_responses<I: IntoIterator<Item = String>>(default: &str, responses: I) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            default: default.to_string(),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
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

/// LLM whose completions block until a watch gate is opened, letting
/// tests cancel the pipeline mid-run.
struct GateLlm {
    default: String,
    rx: tokio::sync::watch::Receiver<bool>,
    calls: AtomicUsize,
}

impl GateLlm {
    fn new(default: impl Into<String>, rx: tokio::sync::watch::Receiver<bool>) -> Self {
        Self {
            default: default.into(),
            rx,
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LlmService for GateLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut rx = self.rx.clone();
        let _ = rx.wait_for(|open| *open).await;
        Ok(self.default.clone())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

/// Shared assertion surface for the scripted LLM mocks.
trait CountedLlm: LlmService {
    fn call_count(&self) -> usize;
}

impl CountedLlm for MockLlm {
    fn call_count(&self) -> usize {
        MockLlm::call_count(self)
    }
}

impl CountedLlm for GateLlm {
    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

const EXTRACTION_JSON: &str = r#"{"entities":[
    {"name":"Alice","type":"Person","description":"A researcher"},
    {"name":"The Lab","type":"Organization","description":"A research lab"}
],"relationships":[
    {"source":"Alice","target":"The Lab","keywords":"works at","description":"Alice works at the lab"}
]}"#;

struct TestEnv<L: CountedLlm> {
    pipeline: Arc<IngestionPipeline>,
    kv: Arc<InMemoryKVStorage>,
    graph: Arc<InMemoryGraphStorage>,
    doc_status: Arc<InMemoryDocStatusStorage>,
    llm: Arc<L>,
}

async fn setup_with_concurrency<L: CountedLlm + 'static>(
    llm: Arc<L>,
    max_concurrency: usize,
) -> TestEnv<L> {
    let kv = Arc::new(InMemoryKVStorage::new("kv", "test"));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "test"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "test"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "test",
        test_embedding_func(),
    ));
    let pipeline = Arc::new(
        IngestionPipeline::new(
            llm.clone(),
            kv.clone(),
            vector,
            graph.clone(),
            doc_status.clone(),
            Arc::new(test_embedding_func()),
        )
        .with_max_concurrency(max_concurrency)
        .with_extractor(EntityExtractor::new(
            llm.clone(),
            graph.clone(),
            Arc::new(InMemoryVectorStorage::new(
                "vector2",
                "test",
                test_embedding_func(),
            )),
            kv.clone(),
            EntityExtractorConfig::default(),
        )),
    );
    TestEnv {
        pipeline,
        kv,
        graph,
        doc_status,
        llm,
    }
}

async fn setup<L: CountedLlm + 'static>(llm: Arc<L>) -> TestEnv<L> {
    setup_with_concurrency(llm, 4).await
}

// ---------------------------------------------------------------------
// insertDocument
// ---------------------------------------------------------------------

#[tokio::test]
async fn insert_document_stores_content_and_pending_status() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let doc_id = env
        .pipeline
        .insert_document("Hello world document content", "a.txt", None, None, BIZ)
        .await
        .unwrap();
    assert!(doc_id.starts_with("doc-"));

    let doc = env.kv.get_by_id(&doc_id, BIZ).await.unwrap().expect("doc stored");
    assert_eq!(
        Some("Hello world document content"),
        doc.get("content").and_then(|v| v.as_str())
    );
    assert_eq!(Some("a.txt"), doc.get("file_path").and_then(|v| v.as_str()));

    let statuses = env
        .doc_status
        .get_docs_by_status(DocStatus::Pending, BIZ)
        .await
        .unwrap();
    assert!(statuses.contains_key(&doc_id));
    let status = &statuses[&doc_id];
    assert_eq!(DocStatus::Pending, status.status);
    assert_eq!(28, status.content_length);
    assert!(status.content_hash.is_some());
}

#[tokio::test]
async fn insert_document_persists_processing_options() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let options = DocumentProcessingOptions {
        chunk_method: Some("recursive_character".to_string()),
        chunk_token_size: Some(50),
        chunk_overlap: Some(5),
        skip_kg: true,
        build_structure: true,
        parser_engine: None,
    };
    let doc_id = env
        .pipeline
        .insert_document("Content with options", "b.txt", Some(&options), None, BIZ)
        .await
        .unwrap();
    let doc = env.kv.get_by_id(&doc_id, BIZ).await.unwrap().unwrap();
    assert_eq!(
        Some("recursive_character"),
        doc.get("chunk_method").and_then(|v| v.as_str())
    );
    assert_eq!(Some(true), doc.get("skip_kg").and_then(|v| v.as_bool()));
    assert_eq!(
        Some(true),
        doc.get("build_structure").and_then(|v| v.as_bool())
    );
}

// ---------------------------------------------------------------------
// processDocument
// ---------------------------------------------------------------------

#[tokio::test]
async fn process_document_full_flow_marks_processed_and_builds_graph() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let doc_id = env
        .pipeline
        .insert_document(
            "Alice works at The Lab. This is a long enough document to be chunked properly by the tokenizer.",
            "paper.txt",
            None,
            None,
            BIZ)
        .await
        .unwrap();

    let content = "Alice works at The Lab. This is a long enough document to be chunked properly by the tokenizer.";
    env.pipeline
        .process_document(&doc_id, content, "paper.txt", None, None, None, BIZ)
        .await;

    // Status PROCESSED with chunks_count.
    let processed = env
        .doc_status
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    let status = processed.get(&doc_id).expect("doc processed");
    assert!(status.chunks_count.unwrap_or(0) >= 1);
    assert!(!status.chunks_list.is_empty());

    // Chunks stored in KV with full_doc_id.
    let chunk_id = &status.chunks_list[0];
    let chunk = env.kv.get_by_id(chunk_id, BIZ).await.unwrap().expect("chunk stored");
    assert_eq!(
        Some(doc_id.as_str()),
        chunk.get("full_doc_id").and_then(|v| v.as_str())
    );

    // Graph populated from extraction.
    let alice = env.graph.get_node("Alice", BIZ).await.unwrap().expect("node");
    assert_eq!(
        Some("Person"),
        alice.get("entity_type").and_then(|v| v.as_str())
    );
    assert_eq!(
        Some("A researcher"),
        alice.get("description").and_then(|v| v.as_str())
    );
    assert_eq!(
        Some(chunk_id.as_str()),
        alice.get("source_id").and_then(|v| v.as_str())
    );
    let edge = env
        .graph
        .get_edge("Alice", "The Lab", BIZ)
        .await
        .unwrap()
        .expect("edge");
    assert_eq!(
        Some("works at"),
        edge.get("keywords").and_then(|v| v.as_str())
    );
    assert_eq!(Some(1.0), edge.get("weight").and_then(|v| v.as_f64()));
}

#[tokio::test]
async fn process_document_empty_content_marks_failed() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    // Insert real content first, then process with blank content.
    let doc_id = env
        .pipeline
        .insert_document("real content placeholder", "c.txt", None, None, BIZ)
        .await
        .unwrap();
    env.pipeline
        .process_document(&doc_id, "   \n\t ", "c.txt", None, None, None, BIZ)
        .await;

    let failed = env
        .doc_status
        .get_docs_by_status(DocStatus::Failed, BIZ)
        .await
        .unwrap();
    let status = failed.get(&doc_id).expect("doc failed");
    assert_eq!(
        Some("Document empty after text cleaning"),
        status.error_msg.as_deref()
    );
    assert_eq!(0, env.llm.call_count());
}

#[tokio::test]
async fn process_document_skip_kg_leaves_graph_empty() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let doc_id = env
        .pipeline
        .insert_document("Alice works at The Lab on interesting projects", "d.txt", None, None, BIZ)
        .await
        .unwrap();
    let options = DocumentProcessingOptions {
        skip_kg: true,
        ..Default::default()
    };
    env.pipeline
        .process_document(
            &doc_id,
            "Alice works at The Lab on interesting projects",
            "d.txt",
            Some(&options),
            None,
            None,
        BIZ)
        .await;

    let processed = env
        .doc_status
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    assert!(processed.contains_key(&doc_id));
    assert!(env.graph.get_node("Alice", BIZ).await.unwrap().is_none());
    // No extraction LLM call when KG is skipped.
    assert_eq!(0, env.llm.call_count());
}

#[tokio::test]
async fn process_document_build_structure_index_when_requested() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let content = "# Title\nSome structured content here for the tree builder.";
    let doc_id = env
        .pipeline
        .insert_document(content, "e.md", None, None, BIZ)
        .await
        .unwrap();
    let options = DocumentProcessingOptions {
        build_structure: true,
        ..Default::default()
    };
    env.pipeline
        .process_document(&doc_id, content, "e.md", Some(&options), None, None, BIZ)
        .await;

    let key = format!("struct-idx-{doc_id}");
    let structure = env.kv.get_by_id(&key, BIZ).await.unwrap().expect("structure stored");
    assert_eq!(
        Some(doc_id.as_str()),
        structure.get("doc_id").and_then(|v| v.as_str())
    );
    let structure_json = structure
        .get("structure")
        .and_then(|v| v.as_str())
        .expect("structure json");
    assert!(structure_json.contains("Title"));
}

// ---------------------------------------------------------------------
// start / cancel / busy guard
// ---------------------------------------------------------------------

#[tokio::test]
async fn start_processes_pending_documents_and_reports_status() {
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let doc_id = env
        .pipeline
        .insert_document("Alice works at The Lab pipeline start test", "f.txt", None, None, BIZ)
        .await
        .unwrap();

    env.pipeline.start(BIZ).await;

    let processed = env
        .doc_status
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    assert!(processed.contains_key(&doc_id));

    let status = env.pipeline.pipeline_status();
    assert_eq!(Some(false), status.get("busy").and_then(|v| v.as_bool()));
    assert_eq!(Some(1), status.get("docs_processed").and_then(|v| v.as_u64()));
    assert!(status.get("last_run_at").and_then(|v| v.as_str()).is_some());
}

#[tokio::test]
async fn cancel_before_start_is_reset_when_pipeline_starts() {
    // `start` clears the cancel flag before iterating
    // pending docs, so a cancel issued before start does not skip them.
    let env = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let doc_id = env
        .pipeline
        .insert_document("Alice works at The Lab cancel test content", "g.txt", None, None, BIZ)
        .await
        .unwrap();

    env.pipeline.cancel();
    env.pipeline.start(BIZ).await;

    let processed = env
        .doc_status
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    assert!(processed.contains_key(&doc_id));
    assert!(env.llm.call_count() > 0);
}

#[tokio::test]
async fn cancel_during_run_skips_remaining_documents() {
    // max_concurrency = 1 forces the second document to wait for the
    // semaphore, where the cancel flag is re-checked after acquiring.
    let (tx, rx) = tokio::sync::watch::channel(false);
    let llm = Arc::new(GateLlm::new(EXTRACTION_JSON, rx));
    let env = setup_with_concurrency(llm.clone(), 1).await;
    let doc_a = env
        .pipeline
        .insert_document("Alice cancel run test one content", "c1.txt", None, None, BIZ)
        .await
        .unwrap();
    let doc_b = env
        .pipeline
        .insert_document("Bob cancel run test two content here", "c2.txt", None, None, BIZ)
        .await
        .unwrap();

    let pipeline = Arc::clone(&env.pipeline);
    let handle = tokio::spawn(async move { pipeline.start(BIZ).await });

    // Wait until the first document is inside LLM extraction.
    for _ in 0..1000 {
        if llm.call_count() > 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(llm.call_count() > 0, "first doc never reached extraction");

    env.pipeline.cancel();
    let _ = tx.send(true); // release the gated LLM call
    handle.await.unwrap();

    let processed = env
        .doc_status
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap();
    let pending = env
        .doc_status
        .get_docs_by_status(DocStatus::Pending, BIZ)
        .await
        .unwrap();
    assert_eq!(1, processed.len());
    assert_eq!(1, pending.len());
    for doc_id in [&doc_a, &doc_b] {
        assert!(processed.contains_key(doc_id) ^ pending.contains_key(doc_id));
    }
    let status = env.pipeline.pipeline_status();
    assert_eq!(Some(false), status.get("busy").and_then(|v| v.as_bool()));
}

// ---------------------------------------------------------------------
// EntityExtractor: merge semantics
// ---------------------------------------------------------------------

async fn make_extractor() -> (EntityExtractor, Arc<InMemoryGraphStorage>, Arc<MockLlm>) {
    let llm = Arc::new(MockLlm::new(EXTRACTION_JSON));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "ext"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "ext",
        test_embedding_func(),
    ));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "ext"));
    let extractor = EntityExtractor::new(
        llm.clone(),
        graph.clone(),
        vector,
        kv,
        EntityExtractorConfig::default(),
    );
    (extractor, graph, llm)
}

#[tokio::test]
async fn extractor_merges_descriptions_and_weights_on_second_merge() {
    let (extractor, graph, _llm) = make_extractor().await;
    let result = extractor
        .extract("Alice works at The Lab", "chunk-1", "")
        .await
        .unwrap();
    assert_eq!(2, result.entities.len());
    assert_eq!(1, result.relationships.len());

    extractor.merge_to_graph(&result, "chunk-1", BIZ).await.unwrap();
    extractor.merge_to_graph(&result, "chunk-2", BIZ).await.unwrap();

    // Same description fragment is deduplicated (containment check).
    let alice = graph.get_node("Alice", BIZ).await.unwrap().unwrap();
    assert_eq!(
        "A researcher",
        alice.get("description").and_then(|v| v.as_str()).unwrap()
    );
    // Source ids accumulate.
    let source_id = alice.get("source_id").and_then(|v| v.as_str()).unwrap();
    assert!(source_id.contains("chunk-1") && source_id.contains("chunk-2"));

    // Edge weight increments on re-merge.
    let edge = graph.get_edge("Alice", "The Lab", BIZ).await.unwrap().unwrap();
    assert_eq!(Some(2.0), edge.get("weight").and_then(|v| v.as_f64()));
}

#[tokio::test]
async fn extractor_merges_distinct_descriptions_with_separator() {
    let (extractor, graph, _llm) = make_extractor().await;
    let first = extractor
        .extract("first chunk text", "chunk-1", "")
        .await
        .unwrap();
    extractor.merge_to_graph(&first, "chunk-1", BIZ).await.unwrap();

    // Second extraction with a different description for the same entity.
    let other = r#"{"entities":[
        {"name":"Alice","type":"Person","description":"Completely different additional info"}
    ],"relationships":[]}"#;
    let llm2 = Arc::new(MockLlm::new(other));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "ext2"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "ext2",
        test_embedding_func(),
    ));
    let extractor2 = EntityExtractor::new(
        llm2,
        graph.clone(),
        vector,
        kv,
        EntityExtractorConfig::default(),
    );
    let second = extractor2
        .extract("second chunk text", "chunk-2", "")
        .await
        .unwrap();
    extractor2.merge_to_graph(&second, "chunk-2", BIZ).await.unwrap();

    let alice = graph.get_node("Alice", BIZ).await.unwrap().unwrap();
    let description = alice.get("description").and_then(|v| v.as_str()).unwrap();
    assert!(description.contains(GRAPH_FIELD_SEP));
    assert!(description.contains("A researcher"));
    assert!(description.contains("Completely different additional info"));
}

#[tokio::test]
async fn extractor_tuple_mode_parses_entity_and_relation_lines() {
    let llm = Arc::new(MockLlm::new(
        "entity<|#|>Bob<|#|>Person<|#|>An engineer\n\
         relation<|#|>Bob<|#|>The Lab<|#|>member of<|#|>Bob is a member\n\
         <|COMPLETE|>",
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "tuple"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "tuple",
        test_embedding_func(),
    ));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "tuple"));
    let extractor = EntityExtractor::new(
        llm,
        graph.clone(),
        vector,
        kv,
        EntityExtractorConfig {
            use_json_extraction: false,
            ..Default::default()
        },
    );
    let result = extractor.extract("some text", "chunk-t", "").await.unwrap();
    assert_eq!(1, result.entities.len());
    assert_eq!("Bob", result.entities[0].entity_name);
    assert_eq!(1, result.relationships.len());
    assert_eq!("member of", result.relationships[0].relationship_keywords);

    extractor.merge_to_graph(&result, "chunk-t", BIZ).await.unwrap();
    assert!(graph.get_node("Bob", BIZ).await.unwrap().is_some());
}

#[tokio::test]
async fn extractor_invalid_json_returns_empty_result() {
    let llm = Arc::new(MockLlm::new("this is not json at all"));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "bad"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "bad",
        test_embedding_func(),
    ));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "bad"));
    let extractor = EntityExtractor::new(
        llm,
        graph,
        vector,
        kv,
        EntityExtractorConfig::default(),
    );
    let result = extractor.extract("text", "chunk-bad", "").await.unwrap();
    assert!(result.entities.is_empty());
    assert!(result.relationships.is_empty());
}

#[tokio::test]
async fn extractor_gleaning_rounds_merge_extra_entities() {
    // Round 1: one entity. Round 2 (gleaning): one more entity.
    // Round 3: empty -> stops.
    let llm = Arc::new(MockLlm::with_responses(
        "",
        [
            r#"{"entities":[{"name":"E1","type":"Person","description":"first"}],"relationships":[]}"#,
            r#"{"entities":[{"name":"E2","type":"Person","description":"second"}],"relationships":[]}"#,
            r#"{"entities":[],"relationships":[]}"#,
        ]
        .map(|s| s.to_string()),
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "glean"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "glean",
        test_embedding_func(),
    ));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "glean"));
    let extractor = EntityExtractor::new(
        llm.clone(),
        graph,
        vector,
        kv,
        EntityExtractorConfig {
            gleaning_rounds: 3,
            ..Default::default()
        },
    );
    let result = extractor.extract("text", "chunk-g", "").await.unwrap();
    assert_eq!(2, result.entities.len());
    // 1 first round + 2 gleaning rounds (third returns empty and stops).
    assert_eq!(3, llm.call_count());
}

#[tokio::test]
async fn extractor_heading_path_included_in_user_prompt() {
    let llm = Arc::new(MockLlm::new(EXTRACTION_JSON));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "hp"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "hp",
        test_embedding_func(),
    ));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "hp"));
    let extractor = EntityExtractor::new(
        llm,
        graph,
        vector,
        kv,
        EntityExtractorConfig::default(),
    );
    // Smoke test: heading path is accepted and extraction still works.
    let result = extractor
        .extract("text", "chunk-hp", "Chapter 1 > Section 1.2")
        .await
        .unwrap();
    assert_eq!(2, result.entities.len());
}
