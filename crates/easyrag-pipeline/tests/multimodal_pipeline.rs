//! RAGAnything multimodal pipeline contract tests (P4.6e).
//!
//! Covers: content-based doc id generation, text insertion contract,
//! parse-failure error, custom parser roundtrip with cache, multimodal
//! query fallback / enhancement / cache, folder discovery, batch
//! processing, and the RAGAnything facade lifecycle.

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::storage::traits::{KVStorage, VectorStorage};
use easyrag_core::JsonMap;
use easyrag_pipeline::multimodal::context::{ContextConfig, ContextExtractor};
use easyrag_pipeline::multimodal::document_processor::generate_content_based_doc_id;
use easyrag_pipeline::multimodal::model::{ContentItem, ParseResult};
use easyrag_pipeline::multimodal::parser::{MultimodalParser, ExternalParserRegistry};
use easyrag_pipeline::multimodal::{
    BatchProcessor, DocumentProcessor, MultimodalQueryEngine, ParseCache,
    RAGAnything, RagAnythingConfig, TextQueryBackend,
};
use easyrag_storage::memory::{InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage};
use futures::stream::BoxStream;
use serde_json::Value as JsonValue;
/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{Arc, Mutex};

// QueryParam lives in core.
use easyrag_core::model::QueryParam;

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
}

impl MockLlm {
    fn new(default: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            default: default.into(),
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

/// Recording text-query backend standing in for the EasyRAG facade.
struct RecordingBackend {
    queries: Mutex<Vec<(String, String)>>,
    response: String,
}

impl RecordingBackend {
    fn new(response: impl Into<String>) -> Self {
        Self {
            queries: Mutex::new(Vec::new()),
            response: response.into(),
        }
    }
}

#[async_trait]
impl TextQueryBackend for RecordingBackend {
    async fn query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        self.queries
            .lock()
            .unwrap()
            .push((query.to_string(), format!("{:?}", param.mode)));
        Ok(self.response.clone())
    }
}

/// Deterministic in-memory parser for roundtrip tests.
struct TestParser;

#[async_trait]
impl MultimodalParser for TestParser {
    fn name(&self) -> &str {
        "test-parser"
    }

    fn supported_formats(&self) -> Vec<String> {
        vec![".txt".to_string(), ".md".to_string()]
    }

    async fn parse(
        &self,
        file: &Path,
        _parse_method: Option<&str>,
        _kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
        let text = std::fs::read_to_string(file).unwrap_or_default();
        let mut text_meta = JsonMap::new();
        text_meta.insert("type".to_string(), JsonValue::from("text"));
        text_meta.insert("text".to_string(), JsonValue::from(text.trim().to_string()));
        let mut image_meta = JsonMap::new();
        image_meta.insert("type".to_string(), JsonValue::from("image"));
        image_meta.insert("img_path".to_string(), JsonValue::from("/tmp/figure1.png"));
        let content_list = vec![
            ContentItem::from_raw_map(text_meta, 0),
            ContentItem::from_raw_map(image_meta, 1),
        ];
        let mut metadata = JsonMap::new();
        metadata.insert("parser".to_string(), JsonValue::from("test-parser"));
        Ok(ParseResult::new(
            content_list,
            file.to_string_lossy().to_string(),
            metadata,
        ))
    }

    fn check_installation(&self) -> bool {
        true
    }
}

fn test_config(parser: &str) -> RagAnythingConfig {
    let mut config = RagAnythingConfig::from_env();
    config.parser = parser.to_string();
    config.parse_method = "auto".to_string();
    config.working_dir = std::env::temp_dir().to_string_lossy().to_string();
    config.parser_output_dir = std::env::temp_dir().to_string_lossy().to_string();
    config
}

struct Storages {
    kv: Arc<InMemoryKVStorage>,
    vector: Arc<InMemoryVectorStorage>,
    graph: Arc<InMemoryGraphStorage>,
}

fn test_storages() -> Storages {
    Storages {
        kv: Arc::new(InMemoryKVStorage::new("kv", "mm")),
        vector: Arc::new(InMemoryVectorStorage::new(
            "vdb",
            "mm",
            test_embedding_func(),
        )),
        graph: Arc::new(InMemoryGraphStorage::new("graph", "mm")),
    }
}

// ---------------------------------------------------------------------
// doc id generation
// ---------------------------------------------------------------------

#[test]
fn doc_id_is_content_based_and_stable() {
    let mut text_meta = JsonMap::new();
    text_meta.insert("type".to_string(), JsonValue::from("text"));
    text_meta.insert("text".to_string(), JsonValue::from("  hello world  "));
    let mut table_meta = JsonMap::new();
    table_meta.insert("type".to_string(), JsonValue::from("table"));
    table_meta.insert("table_body".to_string(), JsonValue::from("<table>data</table>"));
    let items = vec![
        ContentItem::from_raw_map(text_meta, 0),
        ContentItem::from_raw_map(table_meta, 1),
    ];
    let id1 = generate_content_based_doc_id(&items);
    let id2 = generate_content_based_doc_id(&items);
    assert_eq!(id1, id2);
    assert!(id1.starts_with("doc-"));
    assert_eq!(id1.len(), "doc-".len() + 32);

    // Different content → different id
    let mut other_meta = JsonMap::new();
    other_meta.insert("type".to_string(), JsonValue::from("text"));
    other_meta.insert("text".to_string(), JsonValue::from("different"));
    let other = vec![ContentItem::from_raw_map(other_meta, 0)];
    assert_ne!(generate_content_based_doc_id(&other), id1);
}

// ---------------------------------------------------------------------
// text insertion contract
// ---------------------------------------------------------------------

#[tokio::test]
async fn insert_text_content_writes_kv_and_vector() {
    let storages = test_storages();
    let llm = Arc::new(MockLlm::new("unused"));
    let processor = DocumentProcessor::new(
        test_config("test-parser"),
        llm,
        storages.kv.clone(),
        storages.vector.clone(),
        storages.graph.clone(),
        None,
        None,
        BIZ.to_string(),
    );

    let mut meta = JsonMap::new();
    meta.insert("type".to_string(), JsonValue::from("text"));
    meta.insert("text".to_string(), JsonValue::from("para one"));
    let item1 = ContentItem::from_raw_map(meta, 0);
    let mut meta2 = JsonMap::new();
    meta2.insert("type".to_string(), JsonValue::from("text"));
    meta2.insert("text".to_string(), JsonValue::from("para two"));
    let item2 = ContentItem::from_raw_map(meta2, 1);

    let chunk_id = processor
        .insert_text_content(&[&item1, &item2], "doc-x", "/tmp/f.txt")
        .await
        .unwrap();
    assert!(chunk_id.starts_with("chunk-"));

    let stored = storages.kv.get_by_id(&chunk_id, BIZ).await.unwrap().unwrap();
    assert_eq!(stored.get("content").unwrap().as_str().unwrap(), "para one\n\npara two");
    assert_eq!(stored.get("full_doc_id").unwrap().as_str().unwrap(), "doc-x");
    assert_eq!(stored.get("chunk_order_index").unwrap().as_i64().unwrap(), 0);
    // tokens = chars / 4 (project convention)
    let expected_tokens = "para one\n\npara two".chars().count() / 4;
    assert_eq!(stored.get("tokens").unwrap().as_i64().unwrap(), expected_tokens as i64);

    let vector_row = storages.vector.get_by_id(&chunk_id, BIZ).await.unwrap().unwrap();
    assert_eq!(vector_row.get("full_doc_id").unwrap().as_str().unwrap(), "doc-x");

    // Empty text list → empty chunk id, no writes
    let empty = processor
        .insert_text_content(&[], "doc-x", "/tmp/f.txt")
        .await
        .unwrap();
    assert_eq!(empty, "");
}

// ---------------------------------------------------------------------
// parse error contract
// ---------------------------------------------------------------------

/// Parser returning no content must trigger the "Parsing failed" error.
struct EmptyParser;

#[async_trait]
impl MultimodalParser for EmptyParser {
    fn name(&self) -> &str {
        "empty-parser"
    }
    fn supported_formats(&self) -> Vec<String> {
        vec![".txt".to_string()]
    }
    async fn parse(
        &self,
        file: &Path,
        _parse_method: Option<&str>,
        _kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
        Ok(ParseResult::new(
            Vec::new(),
            file.to_string_lossy().to_string(),
            JsonMap::new(),
        ))
    }
    fn check_installation(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn parse_document_errors_on_empty_extraction() {
    ExternalParserRegistry::register_parser(
        "empty-parser",
        Arc::new(|| Arc::new(EmptyParser) as Arc<dyn MultimodalParser>),
    );

    let dir = std::env::temp_dir().join("easyrag_mm_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let file_path = dir.join("empty.txt");
    std::fs::write(&file_path, "anything").unwrap();

    let storages = test_storages();
    let processor = DocumentProcessor::new(
        test_config("empty-parser"),
        Arc::new(MockLlm::new("unused")),
        storages.kv,
        storages.vector,
        storages.graph,
        None,
        None,
        BIZ.to_string(),
    );
    let err = processor
        .parse_document(&file_path.to_string_lossy(), &JsonMap::new())
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Parsing failed"), "unexpected: {msg}");

    // Missing file → InvalidArgument
    let err = processor
        .parse_document("/nonexistent/nowhere.txt", &JsonMap::new())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("File not found"));
}

// ---------------------------------------------------------------------
// full roundtrip with cache
// ---------------------------------------------------------------------

#[tokio::test]
async fn process_document_complete_roundtrip_with_cache() {
    ExternalParserRegistry::register_parser(
        "test-parser",
        Arc::new(|| Arc::new(TestParser) as Arc<dyn MultimodalParser>),
    );

    let dir = std::env::temp_dir().join("easyrag_mm_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let file_path = dir.join("doc.txt");
    std::fs::write(&file_path, "Hello multimodal world").unwrap();

    let storages = test_storages();
    let parse_cache = Arc::new(ParseCache::new(Some(storages.kv.clone()), BIZ.to_string()));
    let processor = DocumentProcessor::new(
        test_config("test-parser"),
        Arc::new(MockLlm::new(
            r#"{"entity_name":"Figure 1","entity_type":"IMAGE","summary":"A figure"}"#,
        )),
        storages.kv.clone(),
        storages.vector.clone(),
        storages.graph.clone(),
        None,
        Some(parse_cache.clone()),
        BIZ.to_string(),
    );

    // Register a no-op image processor so the image item is processed
    // deterministically without LLM variance.
    easyrag_pipeline::multimodal::processor::ModalProcessorRegistry::register(
        easyrag_pipeline::multimodal::model::ModalType::Image,
        Arc::new(easyrag_pipeline::multimodal::processor::GenericModalProcessor::new(
            easyrag_pipeline::multimodal::processor::BaseModalProcessor::new(
                Arc::new(MockLlm::new(
                    r#"{"entity_name":"Figure 1","entity_type":"IMAGE","summary":"A figure"}"#,
                )),
                storages.kv.clone(),
                storages.vector.clone(),
                storages.graph.clone(),
                Arc::new(ContextExtractor::new(ContextConfig::default())),
                BIZ.to_string(),
            ),
        )),
    );

    let result = processor
        .process_document_complete(&file_path.to_string_lossy(), &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(result.get("total_blocks").unwrap().as_i64().unwrap(), 2);
    assert_eq!(result.get("text_blocks").unwrap().as_i64().unwrap(), 1);
    assert_eq!(result.get("multimodal_blocks").unwrap().as_i64().unwrap(), 1);
    assert_eq!(result.get("status").unwrap().as_str().unwrap(), "completed");
    let doc_id = result.get("doc_id").unwrap().as_str().unwrap().to_string();
    assert!(doc_id.starts_with("doc-"));

    // Second run must hit the parse cache (same content blocks).
    let result2 = processor
        .process_document_complete(&file_path.to_string_lossy(), &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(result2.get("doc_id").unwrap().as_str().unwrap(), doc_id);
    assert_eq!(result2.get("total_blocks").unwrap().as_i64().unwrap(), 2);

    easyrag_pipeline::multimodal::processor::ModalProcessorRegistry::clear();
}

// ---------------------------------------------------------------------
// multimodal query engine
// ---------------------------------------------------------------------

#[tokio::test]
async fn query_with_multimodal_fallback_and_enhancement() {
    let backend = Arc::new(RecordingBackend::new("base answer"));
    let engine = MultimodalQueryEngine::new(
        backend.clone(),
        Arc::new(MockLlm::new("unused")),
        test_config("test-parser"),
        None,
        BIZ.to_string(),
    );

    // Empty content → pure text query passthrough
    let out = engine
        .query_with_multimodal("what is X?", None, "mix", &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(out, "base answer");
    {
        let queries = backend.queries.lock().unwrap();
        assert_eq!(queries.len(), 1);
        assert_eq!(queries[0].0, "what is X?");
    }

    // With multimodal content → enhanced query includes context section
    let mut image_item = JsonMap::new();
    image_item.insert("type".to_string(), JsonValue::from("image"));
    image_item.insert("img_path".to_string(), JsonValue::from("/some/dir/figure.png"));
    let mut table_item = JsonMap::new();
    table_item.insert("type".to_string(), JsonValue::from("table"));
    table_item.insert("table_data".to_string(), JsonValue::from("a,b\n1,2"));
    let content = vec![image_item, table_item];

    let out = engine
        .query_with_multimodal("what is X?", Some(&content), "mix", &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(out, "base answer");
    {
        let queries = backend.queries.lock().unwrap();
        assert_eq!(queries.len(), 2);
        let enhanced = &queries[1].0;
        assert!(enhanced.starts_with("what is X?\n"));
        assert!(enhanced.contains("Additional multimodal context:"));
        assert!(enhanced.contains("Image: /some/dir/figure.png"));
        assert!(enhanced.contains("Table data:\na,b\n1,2"));
    }
}

#[tokio::test]
async fn query_with_multimodal_cache_hit() {
    let backend = Arc::new(RecordingBackend::new("cached answer"));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "mmcache"));
    let engine = MultimodalQueryEngine::new(
        backend.clone(),
        Arc::new(MockLlm::new("unused")),
        test_config("test-parser"),
        Some(kv),
        BIZ.to_string(),
    );

    let mut image_item = JsonMap::new();
    image_item.insert("type".to_string(), JsonValue::from("image"));
    image_item.insert("img_path".to_string(), JsonValue::from("/a/figure.png"));
    let content = vec![image_item];

    let first = engine
        .query_with_multimodal("q", Some(&content), "mix", &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(first, "cached answer");

    // Same inputs → cache hit; backend must not be called again.
    let second = engine
        .query_with_multimodal("q", Some(&content), "mix", &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(second, "cached answer");
    assert_eq!(backend.queries.lock().unwrap().len(), 1);

    // Different mode → different cache key → new backend call.
    let third = engine
        .query_with_multimodal("q", Some(&content), "naive", &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(third, "cached answer");
    assert_eq!(backend.queries.lock().unwrap().len(), 2);
}

// ---------------------------------------------------------------------
// batch processor: folder discovery + processing
// ---------------------------------------------------------------------

#[tokio::test]
async fn batch_processor_discovers_and_processes_files() {
    ExternalParserRegistry::register_parser(
        "test-parser",
        Arc::new(|| Arc::new(TestParser) as Arc<dyn MultimodalParser>),
    );

    let root = std::env::temp_dir().join("easyrag_mm_batch");
    let sub = root.join("sub");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(root.join("a.txt"), "alpha content").unwrap();
    std::fs::write(sub.join("b.txt"), "beta content").unwrap();
    std::fs::write(root.join("skip.bin"), "binary").unwrap();

    let storages = test_storages();
    let processor = Arc::new(DocumentProcessor::new(
        test_config("test-parser"),
        Arc::new(MockLlm::new(
            r#"{"entity_name":"Figure 1","entity_type":"IMAGE","summary":"A figure"}"#,
        )),
        storages.kv.clone(),
        storages.vector.clone(),
        storages.graph.clone(),
        None,
        None,
        BIZ.to_string(),
    ));
    let mut config = test_config("test-parser");
    config.supported_file_extensions = vec![".txt".to_string()];
    let batch = BatchProcessor::new(config, processor);

    // Non-recursive: only a.txt
    let result = batch
        .process_folder(
            &root.to_string_lossy(),
            None,
            None,
            None,
            false,
            2,
        )
        .await
        .unwrap();
    assert_eq!(result.total_files, 1);
    assert_eq!(result.successful_files.len(), 1);
    assert!(result.failed_files.is_empty());

    // Recursive: a.txt + sub/b.txt
    let result = batch
        .process_folder(
            &root.to_string_lossy(),
            None,
            None,
            None,
            true,
            2,
        )
        .await
        .unwrap();
    assert_eq!(result.total_files, 2);
    assert_eq!(result.successful_files.len(), 2);

    // Missing folder → error
    let err = batch
        .process_folder("/nonexistent/folder", None, None, None, true, 1)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Folder not found"));
}

// ---------------------------------------------------------------------
// RAGAnything facade lifecycle
// ---------------------------------------------------------------------

#[tokio::test]
async fn rag_anything_lifecycle_and_info() {
    ExternalParserRegistry::register_parser(
        "test-parser",
        Arc::new(|| Arc::new(TestParser) as Arc<dyn MultimodalParser>),
    );

    let storages = test_storages();
    let backend = Arc::new(RecordingBackend::new("facade answer"));
    let rag = RAGAnything::new(
        test_config("test-parser"),
        backend.clone(),
        Arc::new(MockLlm::new(
            r#"{"entity_name":"Figure 1","entity_type":"IMAGE","summary":"A figure"}"#,
        )),
        storages.kv.clone(),
        storages.vector.clone(),
        storages.graph.clone(),
        None,
        BIZ.to_string(),
    );

    // Before init: queries must fail
    let err = rag.query("q", "mix", &JsonMap::new()).await.unwrap_err();
    assert!(err.to_string().contains("not initialized"));

    let info = rag.get_processor_info();
    assert_eq!(info.get("initialized").unwrap().as_bool().unwrap(), false);

    rag.initialize();
    let info = rag.get_processor_info();
    assert_eq!(info.get("initialized").unwrap().as_bool().unwrap(), true);
    assert_eq!(info.get("parser").unwrap().as_str().unwrap(), "test-parser");
    let registered = info.get("registered_processors").unwrap().as_array().unwrap();
    assert!(registered.len() >= 4);

    // Query delegates to the backend
    let answer = rag.query("hello", "mix", &JsonMap::new()).await.unwrap();
    assert_eq!(answer, "facade answer");

    // Process a file through the facade
    let dir = std::env::temp_dir().join("easyrag_mm_facade");
    std::fs::create_dir_all(&dir).unwrap();
    let file_path = dir.join("facade.txt");
    std::fs::write(&file_path, "facade document text").unwrap();
    let result = rag
        .process_file(&file_path.to_string_lossy(), &JsonMap::new())
        .await
        .unwrap();
    assert_eq!(result.get("status").unwrap().as_str().unwrap(), "completed");

    // Finalize releases the components
    rag.finalize();
    let info = rag.get_processor_info();
    assert_eq!(info.get("initialized").unwrap().as_bool().unwrap(), false);
    let err = rag.query("q", "mix", &JsonMap::new()).await.unwrap_err();
    assert!(err.to_string().contains("not initialized"));

    easyrag_pipeline::multimodal::processor::ModalProcessorRegistry::clear();
}
