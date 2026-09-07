//! Query engine behavioral contract tests.
//! Behavioral contract tests
//! covering: mode dispatch, query cache, keyword extraction, KG context
//! assembly (entity-related chunks via source_id split), rerank filtering,
//! tree search (small-doc optimization / navigate / __self__ / early stop),
//! structure strategy fallbacks, and the agentic tool loop.

use async_trait::async_trait;
use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{
    ChatMessage, LlmRequest, LlmService, LlmToolResponse, RerankResult, RerankService, ToolCall,
    ToolCallingLlmService, ToolDefinition,
};
use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};
use easyrag_core::model::query_param::{QueryParam, QueryMode};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::JsonMap;
use easyrag_pipeline::query::{
    ClaimVerifier, EvidenceScorer, QueryRouter, StructureAgentStrategy, StructureQueryStrategy,
    TreeSearchEngine,
};
use easyrag_pipeline::QueryEngine;
use easyrag_storage::memory::{InMemoryGraphStorage, InMemoryKVStorage};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value as JsonValue;
/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

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
    stream_tokens: Mutex<VecDeque<Vec<String>>>,
    default: String,
    calls: AtomicUsize,
}

impl MockLlm {
    fn new(default: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            stream_tokens: Mutex::new(VecDeque::new()),
            default: default.into(),
            calls: AtomicUsize::new(0),
        }
    }

    fn with_responses<I: IntoIterator<Item = String>>(default: &str, responses: I) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            stream_tokens: Mutex::new(VecDeque::new()),
            default: default.to_string(),
            calls: AtomicUsize::new(0),
        }
    }

    /// Script the token list for the next `complete_stream` call.
    fn push_stream_tokens(&self, tokens: Vec<String>) {
        self.stream_tokens.lock().unwrap().push_back(tokens);
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
        self.calls.fetch_add(1, Ordering::SeqCst);
        let tokens = self.stream_tokens.lock().unwrap().pop_front();
        match tokens {
            Some(tokens) => Box::pin(futures::stream::iter(tokens)),
            // Legacy default: empty stream (pre-existing tests rely on it).
            None => Box::pin(futures::stream::empty()),
        }
    }
}

/// Vector storage returning scripted results for every query.
struct MockVectorStorage {
    results: Vec<JsonMap>,
    embedding_func: EmbeddingFunc,
    global_config: JsonMap,
    /// Every `top_k` this store was queried with, in order. `query` already
    /// honours `take(top_k)`, so recording it makes the budget each retrieval
    /// path reads directly assertable.
    requested_top_k: Mutex<Vec<usize>>,
}

impl MockVectorStorage {
    fn new(results: Vec<JsonMap>) -> Self {
        Self {
            results,
            embedding_func: test_embedding_func(),
            global_config: JsonMap::new(),
            requested_top_k: Mutex::new(Vec::new()),
        }
    }

    fn top_k_requests(&self) -> Vec<usize> {
        self.requested_top_k.lock().unwrap().clone()
    }
}

#[async_trait]
impl VectorStorage for MockVectorStorage {
    fn embedding_func(&self) -> &EmbeddingFunc {
        &self.embedding_func
    }

    async fn query(
        &self,
        _query: &str,
        top_k: usize,
        _query_embedding: Option<&[f32]>,
        _time_range_start: Option<i64>,
        _time_range_end: Option<i64>,
        _biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        self.requested_top_k.lock().unwrap().push(top_k);
        Ok(self.results.iter().take(top_k).cloned().collect())
    }

    async fn upsert(&self, _data: HashMap<String, JsonMap>, _biz_id: &str) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn delete_entity(&self, _entity_name: &str, _biz_id: &str) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn delete_entity_relation(&self, _entity_name: &str, _biz_id: &str) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn get_by_id(&self, _id: &str, _biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(None)
    }

    async fn get_by_ids(&self, _ids: &[String], _biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        Ok(Vec::new())
    }

    async fn delete(&self, _ids: &[String], _biz_id: &str) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn get_vectors_by_ids(
        &self,
        _ids: &[String],
        _biz_id: &str,
    ) -> easyrag_core::Result<HashMap<String, Vec<f32>>> {
        Ok(HashMap::new())
    }
}

#[async_trait]
impl easyrag_core::storage::traits::StorageNameSpace for MockVectorStorage {
    fn namespace(&self) -> &str {
        "test"
    }
    fn workspace(&self) -> &str {
        "test"
    }
    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    async fn index_done_callback(&self) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn drop(&self, _biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        Ok(HashMap::new())
    }
}

/// Rerank service returning scripted results.
struct ScriptedReranker {
    results: Vec<RerankResult>,
}

#[async_trait]
impl RerankService for ScriptedReranker {
    async fn rerank(
        &self,
        _query: &str,
        _documents: Vec<String>,
        top_n: Option<usize>,
    ) -> easyrag_core::Result<Vec<RerankResult>> {
        Ok(self
            .results
            .iter()
            .take(top_n.unwrap_or(self.results.len()))
            .cloned()
            .collect())
    }
}

/// Tool-calling LLM with scripted LlmToolResponse queue.
struct ScriptedToolLlm {
    tool_responses: Mutex<VecDeque<LlmToolResponse>>,
    complete_responses: Mutex<VecDeque<String>>,
    complete_default: String,
}

impl ScriptedToolLlm {
    fn new(
        complete_default: impl Into<String>,
        tool_responses: Vec<LlmToolResponse>,
    ) -> Self {
        Self {
            tool_responses: Mutex::new(tool_responses.into_iter().collect()),
            complete_responses: Mutex::new(VecDeque::new()),
            complete_default: complete_default.into(),
        }
    }

    fn with_complete_responses(mut self, responses: Vec<String>) -> Self {
        self.complete_responses = Mutex::new(responses.into_iter().collect());
        self
    }
}

#[async_trait]
impl LlmService for ScriptedToolLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok(self
            .complete_responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.complete_default.clone()))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

#[async_trait]
impl ToolCallingLlmService for ScriptedToolLlm {
    async fn complete_with_tools(
        &self,
        _messages: Vec<ChatMessage>,
        _tools: Vec<ToolDefinition>,
    ) -> easyrag_core::Result<LlmToolResponse> {
        Ok(self
            .tool_responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| LlmToolResponse {
                content: Some("No answer generated.".to_string()),
                tool_calls: Vec::new(),
            }))
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn chunk(content: &str, full_doc_id: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("content".into(), JsonValue::String(content.to_string()));
    m.insert(
        "full_doc_id".into(),
        JsonValue::String(full_doc_id.to_string()),
    );
    m.insert("file_path".into(), JsonValue::String("doc.md".to_string()));
    m
}

fn chunk_with_section_path(content: &str, full_doc_id: &str, path: &[&str]) -> JsonMap {
    let mut m = chunk(content, full_doc_id);
    let mut metadata = JsonMap::new();
    metadata.insert(
        "section_path".into(),
        JsonValue::Array(path.iter().map(|p| JsonValue::String((*p).to_string())).collect()),
    );
    m.insert("metadata".into(), JsonValue::Object(metadata));
    m
}

fn leaf(title: &str, content_start: usize) -> SectionNode {
    SectionNode {
        title: title.to_string(),
        level: 1,
        summary: None,
        content_start,
        content_end: content_start + 100,
        children: Vec::new(),
    }
}

fn engine_with(llm: Arc<MockLlm>, vector_results: Vec<JsonMap>) -> QueryEngine {
    QueryEngine::new(
        llm,
        Arc::new(MockVectorStorage::new(vector_results)),
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    )
}

// ---------------------------------------------------------------------
// QueryEngine: mode dispatch + cache
// ---------------------------------------------------------------------

#[tokio::test]
async fn bypass_returns_query_verbatim_without_llm() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), Vec::new());
    let mut param = QueryParam::default();
    param.mode = QueryMode::Bypass;

    let answer = engine.query("hello world", &param).await.unwrap();
    assert_eq!(answer, "hello world");
    // Bypass is neither cached nor LLM-backed.
    let answer2 = engine.query("hello world", &param).await.unwrap();
    assert_eq!(answer2, "hello world");
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn naive_query_cache_hit_skips_second_llm_call() {
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("some content", "doc1")]);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let a1 = engine.query("q", &param).await.unwrap();
    let a2 = engine.query("q", &param).await.unwrap();
    assert_eq!(a1, "the answer");
    assert_eq!(a2, "the answer");
    assert_eq!(llm.call_count(), 1, "second call must hit the query cache");
}

#[tokio::test]
async fn naive_query_empty_chunks_returns_fail_response() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), Vec::new());
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let answer = engine.query("q", &param).await.unwrap();
    assert_eq!(answer, easyrag_core::prompt::FAIL_RESPONSE);
    assert_eq!(llm.call_count(), 0);
}

// ---------------------------------------------------------------------
// QueryEngine: keyword extraction
// ---------------------------------------------------------------------

#[tokio::test]
async fn extract_keywords_parses_json_and_falls_back_on_garbage() {
    let llm = Arc::new(MockLlm::with_responses(
        "fallback",
        vec![
            r#"{"high_level_keywords":["concept A"],"low_level_keywords":["term b"]}"#.to_string(),
            "not json at all".to_string(),
        ],
    ));
    let engine = engine_with(Arc::clone(&llm), Vec::new());

    let k1 = engine.extract_keywords("q", "English", None).await.unwrap();
    assert_eq!(k1.high_level, vec!["concept A"]);
    assert_eq!(k1.low_level, vec!["term b"]);

    let k2 = engine.extract_keywords("q", "English", None).await.unwrap();
    assert!(k2.high_level.is_empty());
    assert!(k2.low_level.is_empty());
}

#[tokio::test]
async fn extract_keywords_manual_short_circuits_llm() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), Vec::new());
    let mut param = QueryParam::default();
    param.hl_keywords = vec!["hl".to_string()];
    param.ll_keywords = vec!["ll".to_string()];

    let k = engine
        .extract_keywords("q", "English", Some(&param))
        .await
        .unwrap();
    assert_eq!(k.high_level, vec!["hl"]);
    assert_eq!(k.low_level, vec!["ll"]);
    assert_eq!(llm.call_count(), 0);
}

// ---------------------------------------------------------------------
// QueryEngine: KG context assembly
// ---------------------------------------------------------------------

#[tokio::test]
async fn kg_query_only_need_context_with_manual_keywords_makes_no_llm_call() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let mut entity = JsonMap::new();
    entity.insert("entity_name".into(), JsonValue::String("Alice".into()));
    entity.insert("entity_type".into(), JsonValue::String("Person".into()));
    entity.insert("description".into(), JsonValue::String("An engineer".into()));
    let engine = engine_with(Arc::clone(&llm), vec![entity]);

    let mut param = QueryParam::default();
    param.mode = QueryMode::Hybrid;
    param.hl_keywords = vec!["alice".to_string()];
    param.only_need_context = true;

    let context = engine.query("who is alice", &param).await.unwrap();
    assert!(context.contains("Alice"), "context must contain the entity");
    assert_eq!(llm.call_count(), 0, "manual keywords skip extraction; only_need_context skips answer");
}

#[tokio::test]
async fn kg_query_entity_related_chunks_split_sep_and_comma() {
    // Entity with source_id using both <SEP> and "," delimiters; related
    // chunks live in KV storage and must be merged into the context.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let mut entity = JsonMap::new();
    entity.insert("entity_name".into(), JsonValue::String("Alice".into()));
    entity.insert("entity_type".into(), JsonValue::String("Person".into()));
    entity.insert("description".into(), JsonValue::String("An engineer".into()));
    entity.insert(
        "source_id".into(),
        JsonValue::String("c1<SEP>c2,c3".into()),
    );
    let engine = QueryEngine::new(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::new(MockVectorStorage::new(vec![entity])),
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        {
            let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
            let mut batch = HashMap::new();
            batch.insert("c1".to_string(), chunk("chunk one content", "doc1"));
            batch.insert("c2".to_string(), chunk("chunk two content", "doc1"));
            batch.insert("c3".to_string(), chunk("chunk three content", "doc1"));
            kv.upsert(batch, BIZ).await.map_err(|_| ()).unwrap();
            kv as Arc<dyn KVStorage>
        },
    );

    let mut param = QueryParam::default();
    param.mode = QueryMode::Hybrid;
    param.hl_keywords = vec!["alice".to_string()];
    param.only_need_context = true;

    let context = engine.query("q", &param).await.unwrap();
    assert!(context.contains("chunk one content"));
    assert!(context.contains("chunk two content"));
    assert!(context.contains("chunk three content"));
}

#[tokio::test]
async fn kg_query_only_need_prompt_returns_prompt_without_answer_call() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let mut entity = JsonMap::new();
    entity.insert("entity_name".into(), JsonValue::String("Alice".into()));
    entity.insert("entity_type".into(), JsonValue::String("Person".into()));
    entity.insert("description".into(), JsonValue::String("An engineer".into()));
    let engine = engine_with(Arc::clone(&llm), vec![entity]);

    let mut param = QueryParam::default();
    param.mode = QueryMode::Local;
    param.ll_keywords = vec!["alice".to_string()];
    param.only_need_prompt = true;

    let prompt = engine.query("who is alice", &param).await.unwrap();
    assert!(!prompt.is_empty());
    assert_ne!(prompt, "SHOULD_NOT_APPEAR");
    assert_eq!(llm.call_count(), 0);
}

// ---------------------------------------------------------------------
// QueryEngine: rerank
// ---------------------------------------------------------------------

#[tokio::test]
async fn naive_rerank_filters_below_min_score() {
    // The
    // rerank filtering is observed through the prompt sent to the LLM.
    struct PromptCapture {
        inner: MockLlm,
        prompts: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmService for PromptCapture {
        async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
            self.prompts.lock().unwrap().push(request.prompt.clone());
            self.inner.complete(request).await
        }

        fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
            Box::pin(futures::stream::empty())
        }
    }

    let llm = Arc::new(PromptCapture {
        inner: MockLlm::new("the answer"),
        prompts: Mutex::new(Vec::new()),
    });
    let chunks = vec![chunk("good content", "doc1"), chunk("noisy content", "doc1")];
    let engine = QueryEngine::new(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::new(MockVectorStorage::new(chunks)),
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    )
    .with_rerank(Arc::new(ScriptedReranker {
        results: vec![
            RerankResult {
                index: 0,
                score: 0.9,
                document: "good content".into(),
            },
            RerankResult {
                index: 1,
                score: 0.1,
                document: "noisy content".into(),
            },
        ],
    }))
    .with_min_rerank_score(0.5);

    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let answer = engine.query("q", &param).await.unwrap();
    assert_eq!(answer, "the answer");
    let prompts = llm.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("good content"));
    assert!(
        !prompts[0].contains("noisy content"),
        "chunks below min_rerank_score must be filtered out of the context"
    );
}

// ---------------------------------------------------------------------
// Tree search
// ---------------------------------------------------------------------

fn structure_index(sections: Vec<SectionNode>) -> DocumentStructureIndex {
    DocumentStructureIndex {
        doc_id: "doc1".to_string(),
        doc_title: Some("Doc".to_string()),
        doc_description: None,
        sections,
        total_chars: 1000,
        version: 1,
    }
}

#[tokio::test]
async fn tree_search_small_document_selects_all_without_llm() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = TreeSearchEngine::new(llm.clone());
    let index = structure_index(vec![
        leaf("Intro", 0),
        leaf("Body", 100),
        leaf("End", 200),
    ]);

    let result = engine.search("q", &index).await.unwrap();
    assert_eq!(result.total_llm_calls, 0);
    assert_eq!(result.selected_sections.len(), 3);
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn tree_search_navigate_then_drill_self() {
    // 9 top-level sections (exceeds small-doc cap of 8); one has children.
    let llm = Arc::new(MockLlm::with_responses(
        "{}",
        vec![
            // Navigate: pick the section with children.
            r#"{"thinking":"t","confidence":"medium","selected_sections":["Chapter 2"]}"#
                .to_string(),
            // Drill-down: parent itself is sufficient.
            r#"{"thinking":"t","need_deeper":false,"selected_sections":["__self__"]}"#.to_string(),
        ],
    ));
    let mut ch2 = leaf("Chapter 2", 100);
    ch2.children = vec![leaf("2.1 Details", 110), leaf("2.2 More", 150)];
    let mut sections: Vec<SectionNode> = (0..8)
        .map(|i| leaf(&format!("Filler {i}"), i as usize * 10))
        .collect();
    sections.push(ch2);

    let engine = TreeSearchEngine::with_limits(llm.clone(), 2, 3);
    let result = engine.search("q", &structure_index(sections)).await.unwrap();

    assert_eq!(result.total_llm_calls, 2);
    // __self__ kept the parent; final leaves are its children.
    let titles: Vec<&str> = result
        .selected_sections
        .iter()
        .map(|s| s.title.as_str())
        .collect();
    assert_eq!(titles, vec!["2.1 Details", "2.2 More"]);
}

#[tokio::test]
async fn tree_search_high_confidence_early_stop_collects_leaves() {
    let llm = Arc::new(MockLlm::with_responses(
        "{}",
        vec![
            r#"{"thinking":"sure","confidence":"high","selected_sections":["A","B"]}"#.to_string(),
        ],
    ));
    let mut a = leaf("A", 0);
    a.children = vec![leaf("A.1", 1), leaf("A.2", 2)];
    let mut b = leaf("B", 10);
    b.children = vec![leaf("B.1", 11)];
    let mut sections: Vec<SectionNode> = (0..8)
        .map(|i| leaf(&format!("S{i}"), 100 + i as usize))
        .collect();
    sections.push(a);
    sections.push(b);

    let engine = TreeSearchEngine::new(llm.clone());
    let result = engine.search("q", &structure_index(sections)).await.unwrap();

    assert_eq!(result.total_llm_calls, 1, "high confidence ≤2 selections stops early");
    let titles: Vec<&str> = result
        .selected_sections
        .iter()
        .map(|s| s.title.as_str())
        .collect();
    assert_eq!(titles, vec!["A.1", "A.2", "B.1"]);
}

// ---------------------------------------------------------------------
// Structure strategy
// ---------------------------------------------------------------------

async fn seed_structure_index(kv: &InMemoryKVStorage, doc_id: &str, index: &DocumentStructureIndex) {
    let mut structure_data = JsonMap::new();
    structure_data.insert("doc_id".into(), JsonValue::String(doc_id.to_string()));
    structure_data.insert(
        "structure".into(),
        JsonValue::String(serde_json::to_string(index).unwrap()),
    );
    let mut batch = HashMap::new();
    batch.insert(format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"), structure_data);
    kv.upsert(batch, BIZ).await.unwrap();
}

#[tokio::test]
async fn structure_strategy_missing_index_returns_fallback_message() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let strategy = StructureQueryStrategy::new(
        TreeSearchEngine::new(llm.clone()),
        None,
        kv,
        Arc::new(MockVectorStorage::new(Vec::new())),
        llm.clone(),
    );

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc-missing".to_string());

    let result = strategy.query("q", &param).await.unwrap();
    assert!(result.answer.contains("no structure index"));
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn structure_strategy_full_flow_small_document() {
    let llm = Arc::new(MockLlm::new("FINAL STRUCTURED ANSWER"));
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let index = structure_index(vec![leaf("Intro", 0), leaf("Methods", 100)]);
    seed_structure_index(&kv, "doc1", &index).await;

    let chunks = vec![
        chunk_with_section_path("intro body", "doc1", &["Intro"]),
        chunk_with_section_path("methods body", "doc1", &["Methods"]),
        chunk_with_section_path("other doc chunk", "doc2", &["Intro"]),
    ];
    let strategy = StructureQueryStrategy::new(
        TreeSearchEngine::new(llm.clone()),
        None,
        kv,
        Arc::new(MockVectorStorage::new(chunks)),
        llm.clone(),
    );

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc1".to_string());

    let result = strategy.query("q", &param).await.unwrap();
    assert_eq!(result.answer, "FINAL STRUCTURED ANSWER");
    assert_eq!(result.referenced_sections, vec!["Intro", "Methods"]);
    // Small document optimization: tree search used no LLM; only the final
    // answer generation did.
    assert_eq!(llm.call_count(), 1);
}

// ---------------------------------------------------------------------
// Agent strategy
// ---------------------------------------------------------------------

fn tool_call_response(name: &str, args: &[(&str, &str)]) -> LlmToolResponse {
    LlmToolResponse {
        content: Some("thinking".to_string()),
        tool_calls: vec![ToolCall {
            id: None,
            name: name.to_string(),
            arguments: args
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        }],
    }
}

#[tokio::test]
async fn agent_tool_loop_tracks_referenced_sections() {
    let chunks = vec![chunk_with_section_path(
        "section body text",
        "doc1",
        &["Ch1", "S1"],
    )];
    let llm = Arc::new(
        ScriptedToolLlm::new(
            "unused",
            vec![
                tool_call_response(
                    "get_section_content",
                    &[("doc_id", "doc1"), ("section_path", "Ch1 > S1")],
                ),
                // Second response: final answer, no tool calls.
                LlmToolResponse {
                    content: Some("Agent answer".to_string()),
                    tool_calls: Vec::new(),
                },
            ],
        )
        .with_complete_responses(vec!["final".to_string()]),
    );

    let agent = StructureAgentStrategy::new(
        llm,
        Arc::new(InMemoryKVStorage::new("test", "test")),
        Arc::new(MockVectorStorage::new(chunks)),
    );
    let mut param = QueryParam::default();
    param.doc_id = Some("doc1".to_string());

    let result = agent.query("what is in S1?", &param).await.unwrap();
    assert_eq!(result.answer, "Agent answer");
    assert_eq!(result.tool_call_trace.len(), 1);
    assert_eq!(result.referenced_sections, vec!["Ch1 > S1"]);
}

#[tokio::test]
async fn agent_max_tool_calls_falls_back_to_final_completion() {
    // LLM always issues tool calls; the strategy must stop at the limit and
    // request a final answer via plain completion.
    let always_call = tool_call_response("list_documents", &[]);
    let llm = Arc::new(
        ScriptedToolLlm::new("unused", vec![always_call.clone(), always_call])
            .with_complete_responses(vec!["FINAL ANSWER".to_string()]),
    );

    let agent = StructureAgentStrategy::with_max_tool_calls(
        llm,
        Arc::new(InMemoryKVStorage::new("test", "test")),
        Arc::new(MockVectorStorage::new(Vec::new())),
        2,
    );
    let param = QueryParam::default();

    let result = agent.query("q", &param).await.unwrap();
    assert_eq!(result.answer, "FINAL ANSWER");
    assert_eq!(result.tool_call_trace.len(), 2);
}

// ---------------------------------------------------------------------
// Streaming + query_data
// ---------------------------------------------------------------------

#[tokio::test]
async fn query_stream_bypass_yields_query() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = Arc::new(engine_with(Arc::clone(&llm), Vec::new()));
    let mut param = QueryParam::default();
    param.mode = QueryMode::Bypass;

    let mut stream = engine.query_stream("streamed text", &param);
    let mut collected = String::new();
    while let Some(token) = stream.next().await {
        collected.push_str(&token);
    }
    assert_eq!(collected, "streamed text");
}

#[tokio::test]
async fn query_stream_naive_streams_llm_tokens() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    llm.push_stream_tokens(vec!["Hello ".to_string(), "world".to_string()]);
    let engine = Arc::new(engine_with(Arc::clone(&llm), vec![chunk("context", "doc1")]));
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let mut stream = engine.query_stream("q", &param);
    let mut collected = String::new();
    while let Some(token) = stream.next().await {
        collected.push_str(&token);
    }
    assert_eq!(collected, "Hello world");
}

#[tokio::test]
async fn query_stream_naive_empty_retrieval_yields_fail_response() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = Arc::new(engine_with(Arc::clone(&llm), Vec::new()));
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let mut stream = engine.query_stream("q", &param);
    let mut collected = String::new();
    while let Some(token) = stream.next().await {
        collected.push_str(&token);
    }
    assert_eq!(collected, easyrag_core::prompt::FAIL_RESPONSE);
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn query_stream_structure_emits_sync_result_as_single_token() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let strategy = StructureQueryStrategy::new(
        TreeSearchEngine::new(llm.clone()),
        None,
        kv,
        Arc::new(MockVectorStorage::new(Vec::new())),
        llm.clone(),
    );
    let engine = Arc::new(
        QueryEngine::new(
            llm.clone(),
            Arc::new(MockVectorStorage::new(Vec::new())),
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_structure_strategy(Arc::new(strategy)),
    );
    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc-missing".to_string());

    let mut stream = engine.query_stream("q", &param);
    let mut collected = Vec::new();
    while let Some(token) = stream.next().await {
        collected.push(token);
    }
    // Structure mode executes synchronously then emits one result.
    assert_eq!(collected.len(), 1);
    assert!(collected[0].contains("no structure index"));
}

#[tokio::test]
async fn cache_key_ignores_manual_keywords() {
    // Blueprint cache key omits
    // hl/ll keywords, so changing them must NOT invalidate the cache entry.
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("some content", "doc1")]);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let a1 = engine.query("q", &param).await.unwrap();
    assert_eq!(llm.call_count(), 1);

    param.hl_keywords = vec!["changed".to_string()];
    param.ll_keywords = vec!["also-changed".to_string()];
    let a2 = engine.query("q", &param).await.unwrap();
    assert_eq!(a2, a1);
    assert_eq!(llm.call_count(), 1);
}

#[tokio::test]
async fn query_data_naive_mode_returns_chunks_without_entities() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("data chunk", "doc1")]);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let data = engine.query_data("q", &param).await.unwrap();
    assert!(data.entities.is_empty());
    assert!(data.relations.is_empty());
    assert_eq!(data.chunks.len(), 1);
    assert_eq!(
        data.chunks[0].get("content").and_then(|v| v.as_str()),
        Some("data chunk")
    );
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn query_with_references_maps_reference_ids() {
    let llm = Arc::new(MockLlm::new("the answer"));
    let chunks = vec![
        chunk_with_section_path("first", "doc1", &["A", "B"]),
        chunk("second", "doc1"),
    ];
    let engine = engine_with(Arc::clone(&llm), chunks);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;

    let result = engine.query_with_references("q", &param).await.unwrap();
    assert_eq!(result.response, "the answer");
    assert_eq!(result.references.len(), 2);
    assert_eq!(result.references[0].reference_id, 1);
    assert_eq!(result.references[0].section_path.as_deref(), Some("A > B"));
    assert_eq!(result.references[1].reference_id, 2);
    assert_eq!(result.references[1].section_path, None);
}

// ---------------------------------------------------------------------
// Phase 0.3 — Query engine prompt/context golden snapshots.
//
// query/engine.rs is the largest single duplicate source (~500 lines) and the
// Phase 5 main battlefield. These snapshots pin the EXACT prompt and context
// bytes each QueryMode produces through the only_need_prompt / only_need_context
// debug hooks, so the Phase 5 four-step refactor (gate_* 5.1, retrieve_chunks
// 5.2, naive/kg merge 5.3, RetrievalPipeline 5.4) is provably behavior-preserving.
//
// Fixtures are deliberately minimal and deterministic: manual hl/ll keywords skip
// LLM keyword extraction, a single entity avoids ordering ambiguity, and chunks
// come back in scripted Vec order. The debug hooks must never call the answer LLM.
//
// Snapshots land in `<workspace>/testdata/golden/query/`. Regenerate deliberately:
// `INSTA_UPDATE=always cargo test -p easyrag-pipeline --test query_engine`.
// ---------------------------------------------------------------------

fn query_golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-pipeline");
    root.join("testdata").join("golden").join("query")
}

fn snap_query(name: &str, text: &str) {
    insta::with_settings!({
        snapshot_path => query_golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, text);
    });
}

const GOLDEN_QUERY: &str = "What does Alice do?";

/// Single KG entity fixture (one entity => deterministic context ordering).
fn golden_entity() -> JsonMap {
    let mut entity = JsonMap::new();
    entity.insert("entity_name".into(), JsonValue::String("Alice".into()));
    entity.insert("entity_type".into(), JsonValue::String("Person".into()));
    entity.insert(
        "description".into(),
        JsonValue::String("A researcher who studies retrieval-augmented generation.".into()),
    );
    entity
}

/// Build a param with manual keywords (skip LLM extraction) and one debug hook.
fn golden_param(mode: QueryMode, prompt: bool) -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = mode;
    param.hl_keywords = vec!["alice".to_string()];
    param.ll_keywords = vec!["alice".to_string()];
    if prompt {
        param.only_need_prompt = true;
    } else {
        param.only_need_context = true;
    }
    param
}

/// Naive mode deliberately ignores `only_need_prompt` / `only_need_context`
/// (blueprint behavior, engine.rs:533-534): `naive_query` always calls the
/// answer LLM and returns its output. So the naive golden captures the EXACT
/// prompt bytes sent to the LLM — the rendered `NAIVE_RAG_RESPONSE` with the
/// retrieved context embedded inline. That single snapshot pins both the
/// template rendering and the context assembly (`build_naive_context`) for the
/// Phase 5.3 naive/kg merge, which is exactly what must stay byte-identical.
#[tokio::test]
async fn golden_naive_prompt_captured_from_llm_request() {
    struct PromptCapture {
        inner: MockLlm,
        prompts: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmService for PromptCapture {
        async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
            self.prompts.lock().unwrap().push(request.prompt.clone());
            self.inner.complete(request).await
        }

        fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
            Box::pin(futures::stream::empty())
        }
    }

    let capture = Arc::new(PromptCapture {
        inner: MockLlm::new("SHOULD_NOT_APPEAR"),
        prompts: Mutex::new(Vec::new()),
    });
    let engine = QueryEngine::new(
        Arc::clone(&capture) as Arc<dyn LlmService>,
        Arc::new(MockVectorStorage::new(vec![
            chunk("Alice is a researcher.", "doc1"),
            chunk("She studies retrieval-augmented generation.", "doc1"),
        ])),
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    );

    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;
    param.hl_keywords = vec!["alice".to_string()];
    param.ll_keywords = vec!["alice".to_string()];

    let answer = engine.query(GOLDEN_QUERY, &param).await.unwrap();
    assert_eq!(
        answer, "SHOULD_NOT_APPEAR",
        "naive returns the LLM answer; debug hooks are intentionally not honored"
    );

    let prompts = capture.prompts.lock().unwrap();
    assert_eq!(prompts.len(), 1, "naive calls the answer LLM exactly once");
    snap_query("naive_prompt", &prompts[0]);
}

#[tokio::test]
async fn golden_kg_modes_prompt_and_context() {
    // Local (ll_keywords), Global (hl_keywords), Hybrid (both) — the three KG
    // dispatch paths Phase 5.3 merges. Each gets a fresh engine so the query
    // cache never leaks between modes.
    for (mode, name) in [
        (QueryMode::Local, "local"),
        (QueryMode::Global, "global"),
        (QueryMode::Hybrid, "hybrid"),
    ] {
        let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
        let engine = engine_with(Arc::clone(&llm), vec![golden_entity()]);

        let prompt = engine
            .query(GOLDEN_QUERY, &golden_param(mode, true))
            .await
            .unwrap();
        snap_query(&format!("{name}_prompt"), &prompt);

        let context = engine
            .query(GOLDEN_QUERY, &golden_param(mode, false))
            .await
            .unwrap();
        snap_query(&format!("{name}_context"), &context);

        assert_eq!(llm.call_count(), 0, "{name}: debug hooks skip the answer LLM");
    }
}

/// The two streaming prompt builders (`build_naive_query_prompt` /
/// `build_kg_query_prompt`) render through the same vars block as the
/// synchronous paths, but the goldens above cannot see them: streaming hands the
/// prompt to `complete_stream` instead of returning it. These snapshots pin
/// those bytes directly, one per template.
///
/// Only Local is snapshotted for the graph template — the three KG modes share
/// one rendering path and differ solely in which keywords become search terms,
/// and `MockVectorStorage` ignores the query text, so Global/Hybrid would
/// produce identical bytes.
///
/// The streaming KG path always runs LLM keyword extraction and ignores manual
/// hl/ll keywords (unlike the synchronous one), so `complete` answers with
/// keyword JSON to keep it deterministic.
#[tokio::test]
async fn golden_streaming_prompts() {
    struct StreamPromptCapture {
        prompts: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmService for StreamPromptCapture {
        async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
            Ok(
                r#"{"high_level_keywords":["Alice"],"low_level_keywords":["researcher"]}"#
                    .to_string(),
            )
        }

        fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
            self.prompts.lock().unwrap().push(request.prompt.clone());
            Box::pin(futures::stream::empty())
        }
    }

    for (mode, name) in [(QueryMode::Naive, "naive"), (QueryMode::Local, "kg")] {
        let capture = Arc::new(StreamPromptCapture {
            prompts: Mutex::new(Vec::new()),
        });
        let engine = Arc::new(QueryEngine::new(
            capture.clone() as Arc<dyn LlmService>,
            Arc::new(MockVectorStorage::new(vec![
                chunk("Alice is a researcher.", "doc1"),
                chunk("She studies retrieval-augmented generation.", "doc1"),
            ])),
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        ));

        let mut param = QueryParam::default();
        param.mode = mode;
        let mut stream = engine.query_stream(GOLDEN_QUERY, &param);
        while stream.next().await.is_some() {}

        let prompts = capture.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 1, "{name}: exactly one streaming prompt");
        snap_query(&format!("stream_{name}_prompt"), &prompts[0]);
    }
}

// ---------------------------------------------------------------------
// QueryEngine: hallucination guardrails (Features 1 & 2)
//
// Characterization of the two gates the naive and graph paths share. They have
// always been the same gate run twice, differing only in a `" (kg)"` log suffix,
// so each test drives *both* paths and pins that the suffix cannot leak into the
// decision. Written for the `gate_evidence`/`gate_claims` extraction (Phase 5.1)
// — before it, nothing covered these branches at engine level.
// ---------------------------------------------------------------------

/// LLM whose every call fails. Proves a broken guardrail component degrades to
/// "proceed" instead of refusing to answer, and — where a gate is expected to
/// decide from metadata alone — that it never reached for the LLM at all.
struct ErrLlm;

#[async_trait]
impl LlmService for ErrLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Err(easyrag_core::CoreError::Internal(
            "guardrail llm down".to_string(),
        ))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

/// `rerank_score` is the only field `EvidenceScorer::score_from_rerank` trusts,
/// so a chunk carrying one makes the gate decide without any LLM call.
fn scored_chunk(content: &str, rerank_score: f64) -> JsonMap {
    let mut m = chunk(content, "doc1");
    m.insert("rerank_score".into(), JsonValue::from(rerank_score));
    m
}

/// Manual keywords keep the graph path off the keyword-extraction LLM call, so
/// `call_count()` measures generation (and verification) only.
fn guardrail_param(mode: QueryMode) -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = mode;
    param.hl_keywords = vec!["manual keyword".to_string()];
    param
}

#[tokio::test]
async fn weak_evidence_abstains_identically_on_both_paths() {
    for mode in [QueryMode::Naive, QueryMode::Local] {
        let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
        // The scorer's own LLM fails, so a green run also proves the decision
        // came from the rerank fast path.
        let engine = engine_with(Arc::clone(&llm), vec![scored_chunk("weak", 0.05)])
            .with_evidence_scorer(Arc::new(EvidenceScorer::new(Arc::new(ErrLlm))));

        let answer = engine.query("q", &guardrail_param(mode)).await.unwrap();

        assert_eq!(answer, easyrag_core::prompt::FAIL_RESPONSE, "{mode:?}");
        assert_eq!(
            llm.call_count(),
            0,
            "{mode:?}: abstain must happen before generation"
        );
    }
}

#[tokio::test]
async fn strong_evidence_proceeds_on_both_paths() {
    for mode in [QueryMode::Naive, QueryMode::Local] {
        let llm = Arc::new(MockLlm::new("the answer"));
        let engine = engine_with(Arc::clone(&llm), vec![scored_chunk("strong", 0.95)])
            .with_evidence_scorer(Arc::new(EvidenceScorer::new(Arc::new(ErrLlm))));

        let answer = engine.query("q", &guardrail_param(mode)).await.unwrap();

        assert_eq!(answer, "the answer", "{mode:?}");
        assert_eq!(llm.call_count(), 1, "{mode:?}: generation only");
    }
}

#[tokio::test]
async fn evidence_scorer_failure_is_logged_and_ignored() {
    // No rerank_score: the fast path stays neutral (Refine), so scoring falls
    // through to the LLM path, which fails here.
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")])
        .with_evidence_scorer(Arc::new(EvidenceScorer::new(Arc::new(ErrLlm))));

    let answer = engine
        .query("q", &guardrail_param(QueryMode::Naive))
        .await
        .unwrap();

    assert_eq!(answer, "the answer", "a failing scorer must not abstain");
}

#[tokio::test]
async fn evidence_scoring_requested_without_a_scorer_proceeds() {
    for mode in [QueryMode::Naive, QueryMode::Local] {
        let llm = Arc::new(MockLlm::new("the answer"));
        let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")]);
        let mut param = guardrail_param(mode);
        param.enable_evidence_scoring = Some(true);

        let answer = engine.query("q", &param).await.unwrap();

        assert_eq!(
            answer, "the answer",
            "{mode:?}: a missing scorer only warns, it does not abstain"
        );
    }
}

#[tokio::test]
async fn gate_rejection_is_not_replayed_from_cache() {
    // Scoring goes through the LLM path here, which makes every gate invocation
    // observable as a call on the scorer's own mock.
    let scorer_llm = Arc::new(MockLlm::new(r#"{"score": 0.05}"#));
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")])
        .with_evidence_scorer(Arc::new(EvidenceScorer::new(
            scorer_llm.clone() as Arc<dyn LlmService>,
        )));
    let param = guardrail_param(QueryMode::Naive);

    let first = engine.query("q", &param).await.unwrap();
    let second = engine.query("q", &param).await.unwrap();

    assert_eq!(first, easyrag_core::prompt::FAIL_RESPONSE);
    assert_eq!(second, easyrag_core::prompt::FAIL_RESPONSE);
    assert_eq!(
        scorer_llm.call_count(),
        2,
        "an abstain is a one-off verdict and must never be cached"
    );
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn unsupported_claims_abstain_identically_on_both_paths() {
    for mode in [QueryMode::Naive, QueryMode::Local] {
        let llm = Arc::new(MockLlm::with_responses(
            "SHOULD_NOT_APPEAR",
            vec![
                "Paris is in Germany".to_string(), // generation
                r#"{"claims": ["Paris is in Germany"]}"#.to_string(), // decomposition
                r#"{"verdicts": [{"claim": "Paris is in Germany", "supported": false, "confidence": 0.05}]}"#
                    .to_string(), // verification
            ],
        ));
        let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")])
            .with_claim_verifier(Arc::new(ClaimVerifier::new(
                llm.clone() as Arc<dyn LlmService>
            )));

        let answer = engine.query("q", &guardrail_param(mode)).await.unwrap();

        assert_eq!(answer, easyrag_core::prompt::FAIL_RESPONSE, "{mode:?}");
        assert_eq!(
            llm.call_count(),
            3,
            "{mode:?}: generation + decompose + verify, then abstain"
        );
    }
}

#[tokio::test]
async fn well_supported_claims_pass_the_claim_gate() {
    let llm = Arc::new(MockLlm::with_responses(
        "SHOULD_NOT_APPEAR",
        vec![
            "Paris is in France".to_string(),
            r#"{"claims": ["Paris is in France"]}"#.to_string(),
            r#"{"verdicts": [{"claim": "Paris is in France", "supported": true, "confidence": 0.95}]}"#
                .to_string(),
        ],
    ));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")])
        .with_claim_verifier(Arc::new(ClaimVerifier::new(
            llm.clone() as Arc<dyn LlmService>
        )));

    let answer = engine
        .query("q", &guardrail_param(QueryMode::Naive))
        .await
        .unwrap();

    assert_eq!(answer, "Paris is in France");
}

#[tokio::test]
async fn claim_verification_requested_without_a_verifier_proceeds() {
    for mode in [QueryMode::Naive, QueryMode::Local] {
        let llm = Arc::new(MockLlm::new("the answer"));
        let engine = engine_with(Arc::clone(&llm), vec![chunk("c", "doc1")]);
        let mut param = guardrail_param(mode);
        param.enable_verification = Some(true);

        let answer = engine.query("q", &param).await.unwrap();

        assert_eq!(
            answer, "the answer",
            "{mode:?}: a missing verifier only warns, it does not abstain"
        );
    }
}

// ---------------------------------------------------------------------
// QueryEngine: per-path retrieval budgets (Phase 5.2)
//
// All six recall sites now go through `retrieve_chunks(Recall, TopKField)`.
// Which `QueryParam` field a path reads is the one part of that folding a
// prompt/context snapshot cannot see, so it is pinned here instead: the mock
// store has more hits than any budget, and reports every `top_k` it was asked
// for.
// ---------------------------------------------------------------------

/// 40 scripted hits — more than any budget below, so `take(top_k)` shows the
/// requested size rather than the corpus size.
fn recording_engine() -> (Arc<QueryEngine>, Arc<MockVectorStorage>) {
    let results: Vec<JsonMap> = (0..40)
        .map(|i| chunk(&format!("chunk {i}"), "doc1"))
        .collect();
    let vector = Arc::new(MockVectorStorage::new(results));
    let engine = Arc::new(QueryEngine::new(
        Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
        vector.clone() as Arc<dyn VectorStorage>,
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    ));
    (engine, vector)
}

/// `top_k` and `chunk_top_k` are deliberately different so a path reading the
/// wrong one cannot pass by coincidence; manual keywords keep the KG paths off
/// the keyword-extraction LLM call.
fn budgeted_param(mode: QueryMode) -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = mode;
    param.top_k = 3;
    param.chunk_top_k = 7;
    param.hl_keywords = vec!["manual keyword".to_string()];
    param
}

#[tokio::test]
async fn naive_answer_path_recalls_with_top_k() {
    let (engine, vector) = recording_engine();
    engine
        .query("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    assert_eq!(vector.top_k_requests(), vec![3]);
}

#[tokio::test]
async fn kg_answer_path_recalls_with_chunk_top_k() {
    let (engine, vector) = recording_engine();
    engine
        .query("q", &budgeted_param(QueryMode::Local))
        .await
        .unwrap();
    // Two recalls, in this order: the entity search inside
    // `search_entities_and_relations` splits `top_k` across the search terms
    // (one manual keyword here, so 3), then the chunk retrieval reads
    // `chunk_top_k`. The entity search is a separate shape — per term, no
    // metadata filter, no rerank — and stays outside `retrieve_chunks`.
    assert_eq!(vector.top_k_requests(), vec![3, 7]);
}

#[tokio::test]
async fn pure_data_path_recalls_with_chunk_top_k() {
    let (engine, vector) = recording_engine();
    engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    assert_eq!(vector.top_k_requests(), vec![7]);
}

#[tokio::test]
async fn reference_path_recalls_for_references_then_for_the_answer() {
    let (engine, vector) = recording_engine();
    engine
        .query_with_references("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    // The reference list is vector-only on `chunk_top_k`; the answer it then
    // runs is the naive path, hybrid on `top_k`.
    assert_eq!(vector.top_k_requests(), vec![7, 3]);
}

#[tokio::test]
async fn streaming_prompt_paths_keep_their_own_budgets() {
    let (engine, vector) = recording_engine();
    let mut naive = engine.query_stream("q", &budgeted_param(QueryMode::Naive));
    while naive.next().await.is_some() {}
    assert_eq!(vector.top_k_requests(), vec![3], "streaming naive prompt");

    let (engine, vector) = recording_engine();
    let mut kg = engine.query_stream("q", &budgeted_param(QueryMode::Local));
    while kg.next().await.is_some() {}
    assert_eq!(vector.top_k_requests(), vec![7], "streaming kg prompt");
}

/// Keyword search returning a hit vector search never produces, so RRF fusion
/// is visible in what comes back, plus a call counter.
struct MockKeywordSearch {
    hits: Vec<JsonMap>,
    calls: AtomicUsize,
}

impl MockKeywordSearch {
    fn new(hits: Vec<JsonMap>) -> Self {
        Self {
            hits,
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl KeywordSearchable for MockKeywordSearch {
    async fn keyword_query(
        &self,
        _query: &str,
        top_k: usize,
        _time_range_start: Option<i64>,
        _time_range_end: Option<i64>,
        _biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.hits.iter().take(top_k).cloned().collect())
    }

    async fn index_documents(
        &self,
        _docs: HashMap<String, JsonMap>,
        _biz_id: &str,
    ) -> easyrag_core::Result<()> {
        Ok(())
    }

    async fn remove_documents(&self, _ids: &[String], _biz_id: &str) -> easyrag_core::Result<()> {
        Ok(())
    }
}

/// `chunk` plus an `id`. `rrf_fusion` keys (and deduplicates) on `id` and
/// skips entries that lack one, so hybrid-recall fixtures must carry it —
/// every real backend does.
fn identified_chunk(content: &str, full_doc_id: &str, id: &str) -> JsonMap {
    let mut m = chunk(content, full_doc_id);
    m.insert("id".into(), JsonValue::String(id.to_string()));
    m
}

/// One vector hit plus one keyword-only hit, so which recall strategy a path
/// uses shows up in what it retrieves.
fn fused_engine() -> (Arc<QueryEngine>, Arc<MockKeywordSearch>) {
    let keyword = Arc::new(MockKeywordSearch::new(vec![identified_chunk(
        "keyword-only hit",
        "docK",
        "chunkK",
    )]));
    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            Arc::new(MockVectorStorage::new(vec![identified_chunk(
                "vector hit", "docV", "chunkV",
            )])) as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_keyword_search(keyword.clone() as Arc<dyn KeywordSearchable>),
    );
    (engine, keyword)
}

#[tokio::test]
async fn only_the_answer_paths_fuse_keyword_hits() {
    let (engine, keyword) = fused_engine();

    // Graph answer path: RRF-fused, so the context carries both hits.
    let mut param = budgeted_param(QueryMode::Local);
    param.only_need_context = true;
    let context = engine.query("q", &param).await.unwrap();
    assert!(context.contains("vector hit"), "{context}");
    assert!(context.contains("keyword-only hit"), "{context}");
    assert_eq!(keyword.call_count(), 1);

    // Pure-data path: vector recall only.
    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(contents, vec!["vector hit"]);
    assert_eq!(
        keyword.call_count(),
        1,
        "query_data must not fuse keyword hits"
    );

    // Streaming naive prompt: vector recall only, like the sync data path.
    let mut stream = engine.query_stream("q", &budgeted_param(QueryMode::Naive));
    while stream.next().await.is_some() {}
    assert_eq!(
        keyword.call_count(),
        1,
        "the streaming prompt path must not fuse keyword hits"
    );

    // Naive answer path: hybrid, so the keyword store is consulted again.
    engine
        .query("q2", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    assert_eq!(keyword.call_count(), 2);
}

/// Reranker that records `(document_count, top_n)` per call and reverses the
/// order, so both "did this path rerank at all" and "with what budget" are
/// observable, and a reranked list is distinguishable from a recalled one.
struct RecordingReranker {
    calls: Mutex<Vec<(usize, Option<usize>)>>,
}

impl RecordingReranker {
    fn new() -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
        }
    }

    fn calls(&self) -> Vec<(usize, Option<usize>)> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl RerankService for RecordingReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: Vec<String>,
        top_n: Option<usize>,
    ) -> easyrag_core::Result<Vec<RerankResult>> {
        self.calls.lock().unwrap().push((documents.len(), top_n));
        Ok(documents
            .iter()
            .enumerate()
            .rev()
            .map(|(index, document)| RerankResult {
                index,
                score: 1.0,
                document: document.clone(),
            })
            .collect())
    }
}

/// Four chunks of recall, a reranker wired in, and the two budgets set apart
/// (`top_k = 3`, `chunk_top_k = 7`) so the rerank `top_n` cannot be confused
/// with the recall budget.
fn reranking_engine() -> (Arc<QueryEngine>, Arc<RecordingReranker>) {
    let reranker = Arc::new(RecordingReranker::new());
    let chunks: Vec<JsonMap> = (0..4)
        .map(|i| identified_chunk(&format!("chunk {i}"), "doc1", &format!("id{i}")))
        .collect();
    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            Arc::new(MockVectorStorage::new(chunks)) as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_rerank(reranker.clone() as Arc<dyn RerankService>),
    );
    (engine, reranker)
}

/// `rerank_if_enabled` folds the four reranking call sites; this pins what all
/// four had in common before the fold.
#[tokio::test]
async fn rerank_gate_runs_on_the_answer_paths_with_chunk_top_k() {
    let (engine, reranker) = reranking_engine();

    // Naive answer path: recalls with `top_k`, reranks with `chunk_top_k`.
    engine
        .query("naive", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    // Graph answer path (context-only keeps it out of the cache).
    let mut kg = budgeted_param(QueryMode::Local);
    kg.only_need_context = true;
    engine.query("kg", &kg).await.unwrap();
    // Both streaming prompt paths.
    let mut naive_stream = engine.query_stream("s-naive", &budgeted_param(QueryMode::Naive));
    while naive_stream.next().await.is_some() {}
    let mut kg_stream = engine.query_stream("s-kg", &budgeted_param(QueryMode::Local));
    while kg_stream.next().await.is_some() {}

    // One rerank per answer path, each sized by `chunk_top_k` — never by the
    // recall budget. The document counts restate the recall budgets: the naive
    // paths recall `top_k = 3` of the 4 available chunks, the graph ones
    // `chunk_top_k = 7` (all 4).
    assert_eq!(
        reranker.calls(),
        vec![
            (3, Some(7)), // naive answer
            (4, Some(7)), // graph answer
            (3, Some(7)), // streaming naive prompt
            (4, Some(7)), // streaming kg prompt
        ],
        "every reranking path must pass chunk_top_k as top_n"
    );
}

/// The two paths that never reranked must still not: the reference list and the
/// pure-data result are built from raw recall order.
#[tokio::test]
async fn reference_and_data_paths_do_not_rerank() {
    let (engine, reranker) = reranking_engine();

    let result = engine
        .query_with_references("refs", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let reference_contents: Vec<&str> = result
        .references
        .iter()
        .map(|r| r.content.as_str())
        .collect();
    assert_eq!(
        reference_contents,
        vec!["chunk 0", "chunk 1", "chunk 2", "chunk 3"],
        "the reference list must keep recall order, not reranked order"
    );
    // The single call comes from the inner naive answer (3 recalled chunks),
    // not from the reference list.
    assert_eq!(reranker.calls(), vec![(3, Some(7))]);

    let data = engine
        .query_data("data", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let data_contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        data_contents,
        vec!["chunk 0", "chunk 1", "chunk 2", "chunk 3"],
        "query_data must keep recall order"
    );
    assert_eq!(
        reranker.calls().len(),
        1,
        "query_data must not rerank"
    );
}

// ---------------------------------------------------------------------
// QueryEngine: pre-retrieval routing gate (Feature 6)
//
// Characterization of the gate `query` and `query_stream` share. Written for the
// `route_query` extraction (Phase 5.4) — before it nothing covered these
// branches at engine level, and the `no_retrieval` reply was a string literal
// duplicated in both entry points.
// ---------------------------------------------------------------------

/// Engine with a router wired to the same mock LLM. `"hello"` hits the router's
/// LLM-free fast path to `no_retrieval`, so a diverted route is observable
/// without scripting a routing response.
fn routed_engine(llm: Arc<MockLlm>) -> (Arc<QueryEngine>, Arc<MockVectorStorage>) {
    let vector = Arc::new(MockVectorStorage::new(vec![chunk(
        "Alice is a researcher.",
        "doc1",
    )]));
    let engine = Arc::new(
        QueryEngine::new(
            llm.clone() as Arc<dyn LlmService>,
            vector.clone() as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_query_router(Arc::new(QueryRouter::new(llm as Arc<dyn LlmService>))),
    );
    (engine, vector)
}

fn naive_param() -> QueryParam {
    let mut param = QueryParam::default();
    param.mode = QueryMode::Naive;
    param
}

#[tokio::test]
async fn no_retrieval_route_short_circuits_both_entry_points() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let (engine, vector) = routed_engine(llm.clone());
    let param = naive_param();

    let answer = engine.query("hello", &param).await.unwrap();
    assert!(
        answer.starts_with("I'm here to help with questions about your documents."),
        "{answer}"
    );
    assert_eq!(
        vector.top_k_requests(),
        Vec::<usize>::new(),
        "routing is pre-retrieval"
    );
    assert_eq!(llm.call_count(), 0, "the fast path never reaches the LLM");

    let mut tokens: Vec<String> = Vec::new();
    let mut stream = engine.query_stream("hello", &param);
    while let Some(token) = stream.next().await {
        tokens.push(token);
    }
    assert_eq!(
        tokens,
        vec![answer.clone()],
        "the stream must yield the routed answer as its single token"
    );
    assert_eq!(vector.top_k_requests(), Vec::<usize>::new());
    assert_eq!(llm.call_count(), 0);
}

#[tokio::test]
async fn routing_is_skipped_for_bypass_and_debug_hooks() {
    let llm = Arc::new(MockLlm::new("the answer"));
    let (engine, vector) = routed_engine(llm);

    // Bypass keeps its echo contract on both entry points.
    let mut bypass = QueryParam::default();
    bypass.mode = QueryMode::Bypass;
    assert_eq!(engine.query("hello", &bypass).await.unwrap(), "hello");
    let mut echoed: Vec<String> = Vec::new();
    let mut stream = engine.query_stream("hello", &bypass);
    while let Some(token) = stream.next().await {
        echoed.push(token);
    }
    assert_eq!(echoed, vec!["hello".to_string()]);
    assert_eq!(vector.top_k_requests(), Vec::<usize>::new());

    // The debug hooks must stay deterministic, so routing is skipped for them
    // and retrieval runs instead. Local, not Naive: `naive_query` deliberately
    // ignores `only_need_context` and always returns the LLM answer.
    let mut ctx = QueryParam::default();
    ctx.mode = QueryMode::Local;
    ctx.only_need_context = true;
    let context = engine.query("hello", &ctx).await.unwrap();
    assert!(context.contains("Alice is a researcher."), "{context}");
    assert_eq!(vector.top_k_requests(), vec![ctx.chunk_top_k]);
}

#[tokio::test]
async fn routed_answer_is_not_replayed_from_cache() {
    let llm = Arc::new(MockLlm::new(
        r#"{"query_type":"no_retrieval","sub_queries":[],"has_false_premise":false}"#,
    ));
    let (engine, vector) = routed_engine(llm.clone());
    let param = naive_param();

    let first = engine.query("What does Alice do?", &param).await.unwrap();
    let second = engine.query("What does Alice do?", &param).await.unwrap();
    assert_eq!(first, second);
    assert!(first.starts_with("I'm here to help"), "{first}");
    // Routing returns before the cache store, so the router runs again.
    assert_eq!(
        llm.call_count(),
        2,
        "a routed answer must be re-decided, not cached"
    );
    assert_eq!(vector.top_k_requests(), Vec::<usize>::new());
}

#[tokio::test]
async fn a_broken_router_does_not_cost_the_answer() {
    let engine = QueryEngine::new(
        Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
        Arc::new(MockVectorStorage::new(vec![chunk("Alice is a researcher.", "doc1")]))
            as Arc<dyn VectorStorage>,
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    )
    .with_query_router(Arc::new(QueryRouter::new(
        Arc::new(ErrLlm) as Arc<dyn LlmService>
    )));

    assert_eq!(
        engine
            .query("What does Alice do?", &naive_param())
            .await
            .unwrap(),
        "the answer",
        "a routing failure is logged and the query proceeds"
    );
}

#[tokio::test]
async fn routing_requested_without_a_router_proceeds() {
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(llm, vec![chunk("Alice is a researcher.", "doc1")]);

    let mut param = naive_param();
    param.enable_query_routing = Some(true);
    assert_eq!(
        engine.query("What does Alice do?", &param).await.unwrap(),
        "the answer"
    );
}

// Keep the trait imports used.
#[allow(dead_code)]
fn _assert_traits(g: Arc<InMemoryGraphStorage>, k: Arc<InMemoryKVStorage>) {
    let _g: Arc<dyn GraphStorage> = g;
    let _k: Arc<dyn KVStorage> = k;
    let _set: HashSet<String> = HashSet::new();
}
