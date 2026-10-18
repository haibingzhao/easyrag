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
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};
use easyrag_core::model::query_param::{QueryParam, QueryMode};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::JsonMap;
use easyrag_pipeline::query::degrade;
use easyrag_pipeline::query::structure::{MAX_SECTION_CONTENT_CHARS, STRUCTURE_DATA_MAX_DOCS};
use easyrag_pipeline::query::{
    ClaimVerifier, EvidenceScorer, QueryRouter, StructureAgentStrategy, StructureQueryStrategy,
    TreeSearchEngine,
};
use easyrag_pipeline::repository::DocChunkRepository;
use easyrag_pipeline::QueryEngine;
use easyrag_storage::memory::{InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage};
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
    /// Rows served by `get_by_ids` (archive-aware keyword filtering reads
    /// these). Empty by default, matching the pre-R2 behaviour.
    by_ids: Vec<JsonMap>,
    embedding_func: EmbeddingFunc,
    global_config: JsonMap,
    /// Every `top_k` this store was queried with, in order. `query` already
    /// honours `take(top_k)`, so recording it makes the budget each retrieval
    /// path reads directly assertable.
    requested_top_k: Mutex<Vec<usize>>,
    /// Every biz set the multi-biz variant was called with, in order. Empty
    /// means no retrieval path took the multi-biz branch.
    requested_biz_sets: Mutex<Vec<Vec<String>>>,
}

impl MockVectorStorage {
    fn new(results: Vec<JsonMap>) -> Self {
        Self {
            results,
            by_ids: Vec::new(),
            embedding_func: test_embedding_func(),
            global_config: JsonMap::new(),
            requested_top_k: Mutex::new(Vec::new()),
            requested_biz_sets: Mutex::new(Vec::new()),
        }
    }

    fn with_by_ids(mut self, rows: Vec<JsonMap>) -> Self {
        self.by_ids = rows;
        self
    }

    fn top_k_requests(&self) -> Vec<usize> {
        self.requested_top_k.lock().unwrap().clone()
    }

    fn biz_set_requests(&self) -> Vec<Vec<String>> {
        self.requested_biz_sets.lock().unwrap().clone()
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
        _exclude_archived: bool,
        _biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        self.requested_top_k.lock().unwrap().push(top_k);
        Ok(self.results.iter().take(top_k).cloned().collect())
    }

    async fn query_multi_biz(
        &self,
        _query: &str,
        top_k: usize,
        _query_embedding: Option<&[f32]>,
        _time_range_start: Option<i64>,
        _time_range_end: Option<i64>,
        _exclude_archived: bool,
        biz_ids: &[&str],
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        self.requested_biz_sets
            .lock()
            .unwrap()
            .push(biz_ids.iter().map(|s| s.to_string()).collect());
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

    async fn get_by_ids(&self, ids: &[String], _biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        Ok(self
            .by_ids
            .iter()
            .filter(|row| {
                row.get("id")
                    .and_then(|v| v.as_str())
                    .map(|id| ids.iter().any(|wanted| wanted == id))
                    .unwrap_or(false)
            })
            .cloned()
            .collect())
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
async fn query_data_chunk_biz_id_backfills_the_single_biz_param() {
    // The projection always carries `biz_id`; a backend that does not echo it
    // (the mock below returns raw chunk maps) falls back to the normalized
    // single-biz `param.biz_id`.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("data chunk", "doc1")]);
    let param = QueryParam {
        mode: QueryMode::Naive,
        biz_id: "single".to_string(),
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert_eq!(data.chunks.len(), 1);
    assert_eq!(
        data.chunks[0].get("biz_id").and_then(|v| v.as_str()),
        Some("single")
    );
}

#[tokio::test]
async fn query_data_chunk_biz_id_prefers_the_storage_echoed_biz_line() {
    // Multi-biz recall: a record's own `biz_id` from storage must reach the
    // projection verbatim — never the first requested biz (`param.biz_id`).
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let mut tagged = chunk("data chunk", "doc1");
    tagged.insert("biz_id".into(), JsonValue::String("b2".into()));
    let store = Arc::new(MockVectorStorage::new(vec![tagged]));
    let engine = engine_with_shared_vector(Arc::clone(&llm), Arc::clone(&store));
    let param = QueryParam {
        mode: QueryMode::Naive,
        biz_id: "single".to_string(),
        biz_ids: vec!["b1".to_string(), "b2".to_string()],
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert_eq!(data.chunks.len(), 1);
    assert_eq!(
        data.chunks[0].get("biz_id").and_then(|v| v.as_str()),
        Some("b2")
    );
}

/// The chunk projection addresses the document a hit came from: `doc_id` is the
/// record's `full_doc_id`, i.e. the same id `/api/documents/list` returns as
/// `id`. Without it a retrieval hit is only a body of text, and a caller that
/// needs to reference the source entry has to list every document first.
#[tokio::test]
async fn query_data_chunk_doc_id_is_the_records_full_doc_id() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("data chunk", "doc1")]);
    let param = QueryParam {
        mode: QueryMode::Naive,
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert_eq!(data.chunks[0]["doc_id"], "doc1");
}

#[tokio::test]
async fn query_data_chunk_doc_id_is_null_for_a_record_without_one() {
    // Backward compatibility mirrors the archive fields: the key is always
    // present, but a pre-`full_doc_id` record must not get a fabricated id.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let mut legacy = chunk("data chunk", "doc1");
    legacy.remove("full_doc_id");
    let engine = engine_with(Arc::clone(&llm), vec![legacy]);
    let param = QueryParam {
        mode: QueryMode::Naive,
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert!(data.chunks[0]["doc_id"].is_null(), "{:?}", data.chunks[0]);
}

// ---------------------------------------------------------------------
// query_data bizIds (multi-biz recall)
// ---------------------------------------------------------------------

/// Engine wired to a shared MockVectorStorage handle so the biz-set recording
/// survives the Arc into the engine.
fn engine_with_shared_vector(llm: Arc<MockLlm>, store: Arc<MockVectorStorage>) -> QueryEngine {
    QueryEngine::new(
        llm,
        store as Arc<dyn VectorStorage>,
        Arc::new(InMemoryGraphStorage::new("test", "test")),
        Arc::new(InMemoryKVStorage::new("test", "test")),
    )
}

#[tokio::test]
async fn query_data_biz_ids_recall_via_the_multi_biz_variant() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let store = Arc::new(MockVectorStorage::new(vec![chunk("data chunk", "doc1")]));
    let engine = engine_with_shared_vector(Arc::clone(&llm), Arc::clone(&store));
    let param = QueryParam {
        mode: QueryMode::Naive,
        biz_id: "single".to_string(),
        biz_ids: vec!["b1".to_string(), "b2".to_string()],
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert_eq!(data.chunks.len(), 1);
    // The exact requested set reaches the store; `biz_id` is ignored, so the
    // single-biz variant is never consulted.
    assert_eq!(
        store.biz_set_requests(),
        vec![vec!["b1".to_string(), "b2".to_string()]]
    );
    assert!(store.top_k_requests().is_empty());
}

#[tokio::test]
async fn query_data_single_element_biz_ids_still_takes_the_multi_biz_path() {
    // bizIds=["b1"] is a request to recall from b1 **instead of** the resolved
    // `biz_id`, so it must ride the multi-biz filter (IN ('b1')), not fall
    // back to whatever `biz_id` was normalized to.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let store = Arc::new(MockVectorStorage::new(vec![chunk("data chunk", "doc1")]));
    let engine = engine_with_shared_vector(Arc::clone(&llm), Arc::clone(&store));
    let param = QueryParam {
        mode: QueryMode::Naive,
        biz_id: "single".to_string(),
        biz_ids: vec!["b1".to_string()],
        ..Default::default()
    };

    engine.query_data("q", &param).await.unwrap();
    assert_eq!(store.biz_set_requests(), vec![vec!["b1".to_string()]]);
    assert!(store.top_k_requests().is_empty());
}

#[tokio::test]
async fn query_data_without_biz_ids_keeps_the_single_biz_recall_path() {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let store = Arc::new(MockVectorStorage::new(vec![chunk("data chunk", "doc1")]));
    let engine = engine_with_shared_vector(Arc::clone(&llm), Arc::clone(&store));
    let param = QueryParam {
        mode: QueryMode::Naive,
        ..Default::default()
    };

    engine.query_data("q", &param).await.unwrap();
    assert!(store.biz_set_requests().is_empty());
    assert_eq!(store.top_k_requests().len(), 1);
}

#[tokio::test]
async fn query_data_biz_ids_with_sections_emit_the_structure_marker() {
    // Signal hygiene: the marker fires only when sections were actually asked
    // for, and then INSTEAD of running a tree search scoped to one biz line.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("data chunk", "doc1")]);
    let param = QueryParam {
        mode: QueryMode::Naive,
        biz_ids: vec!["b1".to_string(), "b2".to_string()],
        include_sections: Some(true),
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert!(data.sections.is_empty());
    assert!(data
        .degrade_reasons
        .contains(&"structure_multi_biz_unsupported".to_string()));
}

#[tokio::test]
async fn query_data_biz_ids_without_sections_emit_no_structure_marker() {
    // Bypass keeps the signal surface clean: the fused modes flag
    // `keyword_search_unavailable` in this fixture (no keyword index), which
    // would mask the structure marker's absence.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("data chunk", "doc1")]);
    let param = QueryParam {
        mode: QueryMode::Bypass,
        biz_ids: vec!["b1".to_string(), "b2".to_string()],
        ..Default::default()
    };

    let data = engine.query_data("q", &param).await.unwrap();
    assert!(!data.degraded);
    assert!(data.degrade_reasons.is_empty());
}

#[tokio::test]
async fn cache_key_distinguishes_distinct_biz_id_sets() {
    // The cache key carries the bizIds join, so two different biz sets over
    // the same query must never share a cache entry — while a repeated call
    // with the same set still hits it.
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(Arc::clone(&llm), vec![chunk("some content", "doc1")]);

    let a = QueryParam {
        mode: QueryMode::Naive,
        biz_ids: vec!["b1".to_string(), "b2".to_string()],
        ..Default::default()
    };
    let b = QueryParam {
        mode: QueryMode::Naive,
        biz_ids: vec!["b1".to_string(), "b3".to_string()],
        ..Default::default()
    };

    engine.query("q", &a).await.unwrap();
    engine.query("q", &b).await.unwrap();
    engine.query("q", &a).await.unwrap();
    // Calls: set {b1,b2} miss, set {b1,b3} miss, repeat of {b1,b2} hit.
    assert_eq!(
        llm.call_count(),
        2,
        "distinct biz sets must not share a cache entry"
    );
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
async fn keyword_fusion_follows_the_plan_not_the_endpoint() {
    let (engine, keyword) = fused_engine();

    // Graph answer path: RRF-fused, so the context carries both hits.
    let mut param = budgeted_param(QueryMode::Local);
    param.only_need_context = true;
    let context = engine.query("q", &param).await.unwrap();
    assert!(context.contains("vector hit"), "{context}");
    assert!(context.contains("keyword-only hit"), "{context}");
    assert_eq!(keyword.call_count(), 1);

    // Pure-data path on a fused plan: same two hits as the answer path above.
    // Whether a path touches the keyword store is decided by its `Recall`, not
    // by which endpoint it serves.
    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(contents, vec!["vector hit", "keyword-only hit"]);
    assert_eq!(
        keyword.call_count(),
        2,
        "the naive data plan fuses, exactly like the naive answer path"
    );

    // Streaming naive prompt: still vector-only, the one answer path that is.
    let mut stream = engine.query_stream("q", &budgeted_param(QueryMode::Naive));
    while stream.next().await.is_some() {}
    assert_eq!(
        keyword.call_count(),
        2,
        "the streaming prompt path must not fuse keyword hits"
    );

    // Naive answer path: hybrid, so the keyword store is consulted again.
    engine
        .query("q2", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    assert_eq!(keyword.call_count(), 3);
}

/// R2 (archive layer): the keyword index has no `archived` column, so hybrid
/// recall must consult the authoritative vector rows and drop archived
/// keyword-only hits *before* fusion — otherwise the keyword leg would put
/// back exactly what the storage-level vector filter removed.
#[tokio::test]
async fn hybrid_recall_drops_archived_keyword_hits_only_when_asked() {
    let mut archived_hit = identified_chunk("archived keyword hit", "docA", "chunkA");
    archived_hit.insert("archived".into(), JsonValue::Bool(true));
    let keyword = Arc::new(MockKeywordSearch::new(vec![archived_hit.clone()]));
    let vector = Arc::new(MockVectorStorage::new(vec![identified_chunk(
        "vector hit", "docV", "chunkV",
    )])
    .with_by_ids(vec![archived_hit]));
    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            vector as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_keyword_search(keyword as Arc<dyn KeywordSearchable>),
    );

    // Default request keeps the pre-R2 hit set: the archived keyword hit is
    // still echoable when the caller did not ask to exclude archived docs.
    let all = engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let contents: Vec<&str> = all
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert!(contents.contains(&"archived keyword hit"), "{contents:?}");
    let hydrated = all
        .chunks
        .iter()
        .find(|c| c["content"] == "archived keyword hit")
        .expect("keyword hit present");
    assert_eq!(
        hydrated["archived"], true,
        "keyword hits echo the archive flags of their authoritative row"
    );

    let mut param = budgeted_param(QueryMode::Naive);
    param.exclude_archived = true;
    let filtered = engine.query_data("q", &param).await.unwrap();
    let contents: Vec<&str> = filtered
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert!(
        !contents.contains(&"archived keyword hit"),
        "archived keyword hit must not survive exclude_archived: {contents:?}"
    );
    assert!(contents.contains(&"vector hit"), "{contents:?}");
}

/// Keyword-only hits address the same document as the vector leg: the tantivy
/// schema stores no `full_doc_id`, so the projection depends on the
/// authoritative-row read `hydrate_keyword_hits` already performs.
#[tokio::test]
async fn keyword_hits_hydrate_full_doc_id_for_the_chunk_projection() {
    let mut keyword_hit = identified_chunk("keyword-only hit", "docK", "chunkK");
    keyword_hit.remove("full_doc_id");
    let keyword = Arc::new(MockKeywordSearch::new(vec![keyword_hit]));
    let vector = Arc::new(
        MockVectorStorage::new(vec![identified_chunk("vector hit", "docV", "chunkV")])
            .with_by_ids(vec![identified_chunk(
                "keyword-only hit", "docK", "chunkK",
            )]),
    );
    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            vector as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_keyword_search(keyword as Arc<dyn KeywordSearchable>),
    );

    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let doc_ids: Vec<(&str, Option<&str>)> = data
        .chunks
        .iter()
        .map(|c| {
            (
                c["content"].as_str().unwrap_or_default(),
                c["doc_id"].as_str(),
            )
        })
        .collect();
    assert!(
        doc_ids.contains(&("vector hit", Some("docV"))),
        "{doc_ids:?}"
    );
    assert!(
        doc_ids.contains(&("keyword-only hit", Some("docK"))),
        "{doc_ids:?}"
    );
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

// ---------------------------------------------------------------------
// Pure-data endpoint: mode matrix (R9), manual keywords (R10), raw
// structure hits (R8) and cross-document scope (R11)
// ---------------------------------------------------------------------

/// `leaf` with an explicit level, summary and children — the structure-index
/// fixtures below need all four, `leaf` fixes them at level 1 / no summary.
fn node(
    title: &str,
    level: usize,
    summary: Option<&str>,
    content_start: usize,
    children: Vec<SectionNode>,
) -> SectionNode {
    SectionNode {
        title: title.to_string(),
        level,
        summary: summary.map(|s| s.to_string()),
        content_start,
        content_end: content_start + 100,
        children,
    }
}

/// Store one chunk record under the content-hash id ingestion would have given
/// it, so `DocChunkRepository` can reconstruct the same id on the way back.
async fn seed_chunk(
    kv: &InMemoryKVStorage,
    content: &str,
    doc_id: &str,
    order: i64,
    section_path: &[&str],
) -> String {
    let chunk_id = compute_mdhash_id(content, "chunk-");
    let mut record = chunk_with_section_path(content, doc_id, section_path);
    record.insert("chunk_order_index".into(), JsonValue::from(order));
    let mut batch = HashMap::new();
    batch.insert(chunk_id.clone(), record);
    kv.upsert(batch, BIZ).await.unwrap();
    chunk_id
}

/// Registry row for a processed document. `has_structure` and `chunks_list` are
/// the two fields structure retrieval reads; the rest satisfy deserialization.
async fn seed_registry_doc(
    doc_status: &InMemoryDocStatusStorage,
    doc_id: &str,
    file_path: &str,
    chunks_list: &[String],
    has_structure: bool,
) {
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
    row.insert("has_structure".into(), JsonValue::Bool(has_structure));
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
    doc_status.upsert(batch, BIZ).await.unwrap();
}

/// Engine with a registry-backed structure strategy and the R8 gate open, so
/// `query_data` can actually reach `search_sections`.
fn structure_data_engine() -> (
    Arc<QueryEngine>,
    Arc<MockLlm>,
    Arc<InMemoryKVStorage>,
    Arc<InMemoryDocStatusStorage>,
) {
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "test"));
    let strategy = StructureQueryStrategy::new(
        TreeSearchEngine::new(llm.clone()),
        None,
        kv.clone(),
        Arc::new(MockVectorStorage::new(Vec::new())),
        llm.clone(),
    )
    .with_doc_status_storage(doc_status.clone());
    let engine = Arc::new(
        QueryEngine::new(
            llm.clone(),
            Arc::new(MockVectorStorage::new(Vec::new())),
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            kv.clone(),
        )
        .with_structure_strategy(Arc::new(strategy))
        .with_data_structure(true),
    );
    (engine, llm, kv, doc_status)
}

/// One section of `data.sections` as a string, for readable assertions.
fn section_field(sections: &[JsonMap], index: usize, key: &str) -> Option<String> {
    sections[index]
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// R9/R8: every mode's data-path behaviour, read off the two observable side
/// effects — which recall budget the vector store was asked for, and whether
/// the keyword store was consulted at all. This covers the private `data_plan`
/// without an inline test module in `engine.rs`, which has never had one.
#[tokio::test]
async fn data_plan_truth_table() {
    // (mode, vector top_k requests in order, keyword-store calls).
    //
    // `budgeted_param` sets top_k=3 / chunk_top_k=7 and one manual keyword, so
    // a chunk recall reads 7 and a KG entity search reads 3 — a path reading the
    // wrong budget cannot pass by coincidence.
    let expectations: [(QueryMode, Vec<usize>, usize); 7] = [
        // naive: fused recall — the same chunks the naive answer path sees —
        // but no KG pass, which is what still separates it from `hybrid`.
        (QueryMode::Naive, vec![7], 1),
        // hybrid: fused recall (one vector + one keyword call), KG pass, no
        // entity backfill.
        (QueryMode::Hybrid, vec![7, 3], 1),
        // mix: fused recall + KG pass + entity→chunk backfill (KV, not vector).
        (QueryMode::Mix, vec![7, 3], 1),
        // structure: vector-only recall and no KG pass — the mode is about the
        // structure index, and paying for KG hits it discards was the R8 gap.
        (QueryMode::Structure, vec![7], 0),
        // Unchanged from before R9.
        (QueryMode::Local, vec![7, 3], 0),
        (QueryMode::Global, vec![7, 3], 0),
        (QueryMode::Bypass, vec![7, 3], 0),
    ];

    for (mode, expected_top_k, expected_keyword_calls) in expectations {
        let (engine, keyword) = fused_engine();
        engine.query_data("q", &budgeted_param(mode)).await.unwrap();
        assert_eq!(
            keyword.call_count(),
            expected_keyword_calls,
            "mode {mode:?}: keyword-store calls"
        );
        // The vector store lives inside the engine, so read its budget back out
        // through a second engine built the same way.
        let (engine, vector) = recording_engine_with_keyword(keyword.clone());
        engine.query_data("q", &budgeted_param(mode)).await.unwrap();
        assert_eq!(
            vector.top_k_requests(),
            expected_top_k,
            "mode {mode:?}: vector recall budgets"
        );
    }
}

/// `recording_engine` plus the keyword store, so the truth table can assert
/// recall budgets on the hybrid modes too.
fn recording_engine_with_keyword(
    keyword: Arc<MockKeywordSearch>,
) -> (Arc<QueryEngine>, Arc<MockVectorStorage>) {
    let results: Vec<JsonMap> = (0..40)
        .map(|i| identified_chunk(&format!("chunk {i}"), "doc1", &format!("id {i}")))
        .collect();
    let vector = Arc::new(MockVectorStorage::new(results));
    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            vector.clone() as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            Arc::new(InMemoryKVStorage::new("test", "test")),
        )
        .with_keyword_search(keyword as Arc<dyn KeywordSearchable>),
    );
    (engine, vector)
}

/// R9: `hybrid` on the data endpoint fuses keyword hits, as the answer paths
/// always have.
#[tokio::test]
async fn query_data_hybrid_mode_fuses_keyword_hits() {
    let (engine, keyword) = fused_engine();
    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Hybrid))
        .await
        .unwrap();
    let contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(contents, vec!["vector hit", "keyword-only hit"]);
    assert_eq!(keyword.call_count(), 1);
    assert!(
        !data.degraded,
        "a configured keyword store means no recall degradation: {:?}",
        data.degrade_reasons
    );
}

/// R9: `mix` on the data endpoint additionally folds in the chunks its KG
/// entities point at, projected exactly like the recalled ones.
#[tokio::test]
async fn query_data_mix_mode_appends_entity_related_chunks() {
    // The vector store answers every query with one entity-shaped hit that
    // carries a `source_id`; the keyword store contributes a second hit; KV
    // holds the chunk that `source_id` names.
    let mut entity_hit = identified_chunk("vector hit", "docV", "chunkV");
    entity_hit.insert("source_id".into(), JsonValue::String("chunkE".to_string()));
    let keyword = Arc::new(MockKeywordSearch::new(vec![identified_chunk(
        "keyword-only hit",
        "docK",
        "chunkK",
    )]));
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let mut related = chunk("entity-related body", "docE");
    related.insert("chunk_order_index".into(), JsonValue::from(0i64));
    let mut batch = HashMap::new();
    batch.insert("chunkE".to_string(), related);
    kv.upsert(batch, BIZ).await.unwrap();

    let engine = Arc::new(
        QueryEngine::new(
            Arc::new(MockLlm::new("the answer")) as Arc<dyn LlmService>,
            Arc::new(MockVectorStorage::new(vec![entity_hit])) as Arc<dyn VectorStorage>,
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            kv.clone(),
        )
        .with_keyword_search(keyword.clone() as Arc<dyn KeywordSearchable>),
    );

    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Mix))
        .await
        .unwrap();
    let contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(
        contents,
        vec!["vector hit", "keyword-only hit", "entity-related body"],
        "mix must append the entity-related chunk after the fused recall"
    );
    // Appended *before* the projection, so it is numbered like any other hit
    // and carries the same `score: null` a KV record has always produced.
    let ids: Vec<i64> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("reference_id").and_then(|v| v.as_i64()))
        .collect();
    assert_eq!(ids, vec![1, 2, 3]);
    assert!(data.chunks[2].get("score").unwrap().is_null());
    assert_eq!(data.entities.len(), 1);
}

/// `naive` recalls exactly like `hybrid`, which is the point: the two endpoints
/// must not disagree about what the corpus holds. What still separates it from
/// `hybrid` is everything *after* recall — no KG pass, no entity backfill.
#[tokio::test]
async fn query_data_naive_mode_fuses_but_skips_the_kg_pass() {
    let (engine, keyword) = fused_engine();
    let data = engine
        .query_data("q", &budgeted_param(QueryMode::Naive))
        .await
        .unwrap();
    let contents: Vec<&str> = data
        .chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(contents, vec!["vector hit", "keyword-only hit"]);
    assert_eq!(keyword.call_count(), 1, "naive recalls by fusion");
    assert!(data.entities.is_empty(), "but runs no KG pass");
    assert!(data.relations.is_empty());
    assert!(data.sections.is_empty());
    assert!(
        !data.degraded,
        "naive with no structure request is not degraded: {:?}",
        data.degrade_reasons
    );
}

/// R10 acceptance: a request carrying manual keywords makes **zero** LLM calls
/// on the data path. (The requirement's own suggested wording — "entities and
/// relations are identical with and without manual keywords" — only holds when
/// the client's keywords happen to equal the LLM's, so the call count is the
/// assertion that actually means something.)
#[tokio::test]
async fn query_data_with_manual_keywords_makes_zero_llm_calls() {
    let llm = Arc::new(MockLlm::new("the answer"));
    let engine = engine_with(llm.clone(), vec![chunk("data chunk", "doc1")]);

    // `budgeted_param` already carries one manual high-level keyword.
    engine
        .query_data("q", &budgeted_param(QueryMode::Mix))
        .await
        .unwrap();
    assert_eq!(
        llm.call_count(),
        0,
        "manual keywords must short-circuit keyword extraction"
    );

    // Contrast: the same request without them pays exactly one extraction call.
    let mut extracted = budgeted_param(QueryMode::Mix);
    extracted.hl_keywords.clear();
    extracted.ll_keywords.clear();
    engine.query_data("q", &extracted).await.unwrap();
    assert_eq!(llm.call_count(), 1);
}

/// R8: `mode=structure` returns the raw hits — path, level, summary, exact body,
/// chunk ids, ordinal rank/score — and never synthesises an answer.
#[tokio::test]
async fn query_data_structure_mode_returns_raw_sections_without_llm_answer() {
    let (engine, llm, kv, doc_status) = structure_data_engine();
    let index = structure_index(vec![
        node(
            "Introduction",
            1,
            Some("What this document is about"),
            0,
            Vec::new(),
        ),
        node("Installation", 1, Some("How to install"), 100, Vec::new()),
    ]);
    seed_structure_index(&kv, "doc1", &index).await;
    // Deliberately out of insertion order: `get_by_ids` is a hash lookup, so the
    // join must sort by `chunk_order_index` rather than trust it.
    let second = seed_chunk(&kv, "intro body two", "doc1", 1, &["Introduction"]).await;
    let first = seed_chunk(&kv, "intro body one", "doc1", 0, &["Introduction"]).await;
    seed_registry_doc(
        &doc_status,
        "doc1",
        "docs/guide.md",
        &[first.clone(), second.clone()],
        true,
    )
    .await;

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc1".to_string());

    let data = engine.query_data("q", &param).await.unwrap();

    // Two top-level leaves ≤ 8 → the tree search takes its zero-LLM path, and
    // no answer is synthesised on top of it.
    assert_eq!(
        llm.call_count(),
        0,
        "raw structure hits must cost no LLM call on a small document"
    );
    assert_eq!(data.sections.len(), 2);
    assert_eq!(
        section_field(&data.sections, 0, "sectionPath").as_deref(),
        Some("Introduction")
    );
    assert_eq!(
        section_field(&data.sections, 0, "docId").as_deref(),
        Some("doc1")
    );
    assert_eq!(
        section_field(&data.sections, 0, "filePath").as_deref(),
        Some("docs/guide.md")
    );
    assert_eq!(
        section_field(&data.sections, 0, "summary").as_deref(),
        Some("What this document is about")
    );
    assert_eq!(
        data.sections[0].get("level").and_then(|v| v.as_i64()),
        Some(1)
    );
    assert_eq!(
        section_field(&data.sections, 0, "content").as_deref(),
        Some("intro body one\n\nintro body two"),
        "the body must follow chunk_order_index, not lookup order"
    );
    let chunk_ids: Vec<&str> = data.sections[0]
        .get("chunkIds")
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(chunk_ids, vec![first.as_str(), second.as_str()]);
    // Ordinal score: rank 0 → 1.0, rank 1 → 0.5.
    assert_eq!(
        data.sections[0].get("rank").and_then(|v| v.as_i64()),
        Some(0)
    );
    assert_eq!(
        data.sections[0].get("score").and_then(|v| v.as_f64()),
        Some(1.0)
    );
    assert_eq!(
        data.sections[1].get("rank").and_then(|v| v.as_i64()),
        Some(1)
    );
    assert_eq!(
        data.sections[1].get("score").and_then(|v| v.as_f64()),
        Some(0.5)
    );
    assert!(!data.degraded, "{:?}", data.degrade_reasons);
    // R8: the structure mode spends nothing on the KG pass.
    assert!(data.entities.is_empty());
    assert!(data.relations.is_empty());
}

/// R8/D1: prefix matching is mandatory, not an optimisation. The structure tree
/// builder drops headings deeper than its max depth while the chunker does not,
/// so a chunk can sit *below* the deepest indexed node — equality matching would
/// return an empty body for exactly those documents.
#[tokio::test]
async fn section_content_uses_prefix_match_for_deep_headings() {
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    // Index stops at depth 4 ("Deep"); the chunk carries a fifth-level path.
    let index = structure_index(vec![node(
        "A",
        1,
        None,
        0,
        vec![node(
            "B",
            2,
            None,
            10,
            vec![node(
                "C",
                3,
                None,
                20,
                vec![node("Deep", 4, None, 30, Vec::new())],
            )],
        )],
    )]);
    seed_structure_index(&kv, "doc1", &index).await;
    let long_body = "x".repeat(MAX_SECTION_CONTENT_CHARS + 500);
    let deep_id = seed_chunk(
        &kv,
        &long_body,
        "doc1",
        0,
        &["A", "B", "C", "Deep", "Deeper"],
    )
    .await;
    seed_registry_doc(&doc_status, "doc1", "docs/deep.md", &[deep_id], true).await;

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc1".to_string());
    let data = engine.query_data("q", &param).await.unwrap();

    assert_eq!(data.sections.len(), 1);
    assert_eq!(
        section_field(&data.sections, 0, "sectionPath").as_deref(),
        Some("A > B > C > Deep")
    );
    let content = section_field(&data.sections, 0, "content").unwrap();
    assert_eq!(
        content.chars().count(),
        MAX_SECTION_CONTENT_CHARS,
        "the deep chunk's body must be found by prefix match, then capped"
    );
    assert_eq!(
        data.sections[0]
            .get("contentTruncated")
            .and_then(|v| v.as_bool()),
        Some(true)
    );
}

/// D1 dedupe: one chunk belongs to exactly one section, so no body is ever
/// emitted twice.
///
/// The parent-and-child-both-selected form of this is unreachable —
/// `TreeSearchEngine` always expands a non-leaf into its leaves before
/// returning — so the reachable form is pinned instead: duplicate titles, which
/// `section_paths` resolves first-wins, put two selected nodes on the *same*
/// path. Longest-match-then-first-wins keeps the body on one of them.
#[tokio::test]
async fn section_bodies_are_never_duplicated_across_sections() {
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    let index = structure_index(vec![
        node("Intro", 1, None, 0, Vec::new()),
        node("Details", 1, None, 100, Vec::new()),
        node("Intro", 1, None, 200, Vec::new()),
    ]);
    seed_structure_index(&kv, "doc1", &index).await;
    let id = seed_chunk(&kv, "shared body", "doc1", 0, &["Intro"]).await;
    seed_registry_doc(&doc_status, "doc1", "docs/dup.md", &[id], true).await;

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc1".to_string());
    let data = engine.query_data("q", &param).await.unwrap();

    assert_eq!(
        data.sections.len(),
        3,
        "the small-document path selects all"
    );
    let bodies: Vec<String> = data
        .sections
        .iter()
        .map(|s| section_field(&[s.clone()], 0, "content").unwrap_or_default())
        .collect();
    assert_eq!(
        bodies.iter().filter(|b| *b == "shared body").count(),
        1,
        "a chunk body must be reported once, not once per matching section: {bodies:?}"
    );
    assert_eq!(bodies[0], "shared body", "first-wins on duplicate titles");
}

/// D1: an empty body is a legitimate result, not a degradation. The chunker
/// skips headings that carry no text while the structure index still records the
/// heading, so `content: ""` must not set `degraded`.
#[tokio::test]
async fn empty_body_section_yields_empty_content_without_degrade() {
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    let index = structure_index(vec![
        node("Has Body", 1, None, 0, Vec::new()),
        node("Heading Only", 1, None, 100, Vec::new()),
    ]);
    seed_structure_index(&kv, "doc1", &index).await;
    let id = seed_chunk(&kv, "body text", "doc1", 0, &["Has Body"]).await;
    // A preamble chunk (empty section path) belongs to no section at all.
    let preamble = seed_chunk(&kv, "preamble text", "doc1", 1, &[]).await;
    seed_registry_doc(&doc_status, "doc1", "docs/x.md", &[id, preamble], true).await;

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.doc_id = Some("doc1".to_string());
    let data = engine.query_data("q", &param).await.unwrap();

    assert_eq!(data.sections.len(), 2);
    assert_eq!(
        section_field(&data.sections, 0, "content").as_deref(),
        Some("body text")
    );
    assert_eq!(
        section_field(&data.sections, 1, "content").as_deref(),
        Some(""),
        "a heading with no body text is a valid empty section"
    );
    assert!(data.sections[1]
        .get("chunkIds")
        .and_then(|v| v.as_array())
        .unwrap()
        .is_empty());
    assert!(
        !data.degraded,
        "an empty body must not be reported as degradation: {:?}",
        data.degrade_reasons
    );
}

/// R8 additional requirement: every way structure retrieval can fail to run is
/// reported, so a client can tell "nothing matched" from "never attempted".
#[tokio::test]
async fn query_data_structure_degrades_explicitly() {
    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;

    // (a) No strategy wired at all.
    let llm = Arc::new(MockLlm::new("SHOULD_NOT_APPEAR"));
    let bare = engine_with(llm.clone(), Vec::new());
    let data = bare.query_data("q", &param).await.unwrap();
    assert!(data.degraded);
    assert_eq!(data.degrade_reasons, vec![degrade::STRUCTURE_UNAVAILABLE]);
    assert!(data.sections.is_empty());

    // (b) Strategy present but the deployment gate is off.
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    let gated = Arc::new(
        QueryEngine::new(
            llm.clone(),
            Arc::new(MockVectorStorage::new(Vec::new())),
            Arc::new(InMemoryGraphStorage::new("test", "test")),
            kv.clone(),
        )
        .with_structure_strategy(Arc::new(
            StructureQueryStrategy::new(
                TreeSearchEngine::new(llm.clone()),
                None,
                kv.clone(),
                Arc::new(MockVectorStorage::new(Vec::new())),
                llm.clone(),
            )
            .with_doc_status_storage(doc_status.clone()),
        )),
    );
    let data = gated.query_data("q", &param).await.unwrap();
    assert!(data.degraded);
    assert_eq!(data.degrade_reasons, vec![degrade::STRUCTURE_DISABLED]);

    // (c) Gate on, but nothing in scope carries a structure index.
    let mut scoped = param.clone();
    scoped.structure_scope = Some("docs/none/".to_string());
    let data = engine.query_data("q", &scoped).await.unwrap();
    assert!(data.degraded);
    assert_eq!(
        data.degrade_reasons,
        vec![degrade::NO_DOCUMENT_IN_SCOPE],
        "an empty scope must say so"
    );

    // (d) Gate on, a document in scope, but no structure index for it.
    seed_registry_doc(&doc_status, "doc-noindex", "docs/plain.md", &[], true).await;
    let mut located = param.clone();
    located.structure_scope = Some("docs/".to_string());
    let data = engine.query_data("q", &located).await.unwrap();
    assert!(data.degraded);
    assert_eq!(data.degrade_reasons, vec![degrade::NO_INDEX]);

    // Signal hygiene: not asking for sections is not *structure* degradation.
    // This engine wires no keyword index and `naive` now recalls by fusion, so
    // the recall gap is legitimately reported — what must never happen is a
    // structure reason appearing for a request that asked for no sections.
    let mut naive = QueryParam::default();
    naive.mode = QueryMode::Naive;
    let data = engine.query_data("q", &naive).await.unwrap();
    assert!(
        !data
            .degrade_reasons
            .iter()
            .any(|reason| reason.starts_with("structure_")),
        "a mode that never requested structure must not report structure reasons: {:?}",
        data.degrade_reasons
    );
}

/// R9/D3 signal hygiene: the missing-keyword-index marker tracks the plan's
/// recall strategy, not the mode's name. Every fused mode reports it — `naive`
/// included, which is this endpoint's most common mode — and the vector-only
/// ones stay clean, so the marker still means exactly "you asked for fusion and
/// did not get it".
#[tokio::test]
async fn keyword_search_unavailable_follows_the_recall_plan() {
    let llm = Arc::new(MockLlm::new("the answer"));
    // No `.with_keyword_search`, so hybrid recall silently degrades to vector.
    let engine = engine_with(llm, vec![chunk("only vector", "doc1")]);

    for mode in [QueryMode::Naive, QueryMode::Hybrid] {
        let mut param = QueryParam::default();
        param.mode = mode;
        param.hl_keywords = vec!["manual".to_string()];
        let data = engine.query_data("q", &param).await.unwrap();
        assert_eq!(
            data.degrade_reasons,
            vec![degrade::KEYWORD_SEARCH_UNAVAILABLE],
            "mode {mode:?} recalls by fusion, so the missing index is reported"
        );
        // The recall still happened — degraded, not failed.
        assert_eq!(data.chunks.len(), 1);
    }

    // A vector-only plan never asked for a keyword index, so it must not be
    // told one is missing. (`structure` reports its own gap instead: this
    // engine has no structure strategy wired.)
    let mut structure = QueryParam::default();
    structure.mode = QueryMode::Structure;
    let data = engine.query_data("q", &structure).await.unwrap();
    assert!(
        !data
            .degrade_reasons
            .contains(&degrade::KEYWORD_SEARCH_UNAVAILABLE.to_string()),
        "a vector-only plan must not report a missing keyword index: {:?}",
        data.degrade_reasons
    );
}

/// R11: the registry walk is no longer capped at one page. Only the 120th
/// document (last in `updated_at desc` order) carries a structure index, so a
/// single-page read finds nothing and a full walk finds it.
#[tokio::test]
async fn list_structure_documents_paginates_beyond_one_page() {
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    // 120 registry rows; ids sort descending, so `doc-000` lands on page 2.
    for i in 0..120 {
        let doc_id = format!("doc-{i:03}");
        let structured = i == 0;
        seed_registry_doc(
            &doc_status,
            &doc_id,
            &format!("docs/{doc_id}.md"),
            &[],
            structured,
        )
        .await;
    }
    let index = structure_index(vec![node("Only Section", 1, None, 0, Vec::new())]);
    seed_structure_index(&kv, "doc-000", &index).await;

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.structure_scope = Some("docs/".to_string());
    let data = engine.query_data("q", &param).await.unwrap();

    assert!(
        !data
            .degrade_reasons
            .contains(&degrade::NO_DOCUMENT_IN_SCOPE.to_string()),
        "the second page must be reachable: {:?}",
        data.degrade_reasons
    );
    assert_eq!(data.sections.len(), 1);
    assert_eq!(
        section_field(&data.sections, 0, "docId").as_deref(),
        Some("doc-000")
    );
    assert_eq!(
        section_field(&data.sections, 0, "filePath").as_deref(),
        Some("docs/doc-000.md"),
        "filePath comes from the registry row the walk already read"
    );
}

/// R11: a scope wider than the fan-out limit is searched up to the limit and
/// says it was truncated, rather than silently dropping documents.
#[tokio::test]
async fn structure_scope_truncation_is_reported() {
    let (engine, _llm, kv, doc_status) = structure_data_engine();
    let index = structure_index(vec![node("S", 1, None, 0, Vec::new())]);
    for i in 0..(STRUCTURE_DATA_MAX_DOCS + 3) {
        let doc_id = format!("doc-{i}");
        seed_registry_doc(
            &doc_status,
            &doc_id,
            &format!("docs/{doc_id}.md"),
            &[],
            true,
        )
        .await;
        seed_structure_index(&kv, &doc_id, &index).await;
    }

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.structure_scope = Some("docs/".to_string());
    let data = engine.query_data("q", &param).await.unwrap();

    assert!(
        data.degrade_reasons
            .contains(&degrade::SCOPE_TRUNCATED.to_string()),
        "{:?}",
        data.degrade_reasons
    );
    // Every hit still carries its own document, so a caller can see which
    // documents made the cut.
    let doc_ids: HashSet<String> = data
        .sections
        .iter()
        .filter_map(|s| section_field(&[s.clone()], 0, "docId"))
        .collect();
    assert_eq!(doc_ids.len(), STRUCTURE_DATA_MAX_DOCS);
}

/// D5/R-8: the LLM bill of a scoped structure query is bounded by the fan-out
/// limit times the per-document tree-search ceiling — never by corpus size.
#[tokio::test]
async fn structure_data_llm_calls_bounded_by_fanout() {
    let (engine, llm, kv, doc_status) = structure_data_engine();
    // Three-level chain per document: with the default limits (depth 3,
    // fan-out 3) and an unparseable LLM reply, tree search spends exactly one
    // navigate call plus one drill per level = 3 calls.
    let index = structure_index(vec![node(
        "P",
        1,
        None,
        0,
        vec![node(
            "C",
            2,
            None,
            10,
            vec![node("D", 3, None, 20, Vec::new())],
        )],
    )]);
    let seeded = STRUCTURE_DATA_MAX_DOCS + 4;
    for i in 0..seeded {
        let doc_id = format!("doc-{i}");
        seed_registry_doc(
            &doc_status,
            &doc_id,
            &format!("docs/{doc_id}.md"),
            &[],
            true,
        )
        .await;
        seed_structure_index(&kv, &doc_id, &index).await;
    }

    let mut param = QueryParam::default();
    param.mode = QueryMode::Structure;
    param.structure_scope = Some("docs/".to_string());
    engine.query_data("q", &param).await.unwrap();

    const PER_DOC_CEILING: usize = 13; // 1 navigate + 3 drills + 9 sub-drills
    assert_eq!(
        llm.call_count(),
        STRUCTURE_DATA_MAX_DOCS * 3,
        "only the fan-out limit's worth of documents may be searched"
    );
    assert!(
        llm.call_count() <= STRUCTURE_DATA_MAX_DOCS * PER_DOC_CEILING,
        "the worst-case budget must stay bounded: {}",
        llm.call_count()
    );
    assert_eq!(
        llm.call_count(),
        STRUCTURE_DATA_MAX_DOCS * 3,
        "no answer synthesis on top of the tree search"
    );
}

/// R-6: `chunks_list` must survive a write/read round trip on the backend CI
/// actually runs. The cross-backend guard for this is `#[cfg(feature =
/// "postgres")]`-gated and therefore never executes in the default-feature CI
/// job, so the in-memory half is pinned here.
#[tokio::test]
async fn chunks_list_round_trip_on_memory_backend() {
    let kv = Arc::new(InMemoryKVStorage::new("test", "test"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "test"));
    // Written out of order on purpose: the repository must sort, because
    // `get_by_ids` order is a backend detail.
    let c2 = seed_chunk(&kv, "second", "doc1", 2, &["S"]).await;
    let c0 = seed_chunk(&kv, "first", "doc1", 0, &["S"]).await;
    let c1 = seed_chunk(&kv, "middle", "doc1", 1, &["S"]).await;
    let unordered = vec![c2.clone(), c0.clone(), c1.clone()];
    seed_registry_doc(&doc_status, "doc1", "docs/x.md", &unordered, true).await;

    let doc_status: Arc<dyn DocStatusStorage> = doc_status;
    let kv_store: Arc<dyn KVStorage> = kv.clone();
    let repo = DocChunkRepository::new(&doc_status, &kv_store);
    let chunks = repo.chunks_of("doc1", BIZ).await.unwrap();
    let contents: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(contents, vec!["first", "middle", "second"]);
    // A KV record has no self-identifying field; the repository restores it so
    // `sections[].chunkIds` can be reported.
    let ids: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c.get("id").and_then(|v| v.as_str()))
        .collect();
    assert_eq!(ids, vec![c0.as_str(), c1.as_str(), c2.as_str()]);

    // An unknown document is an empty result, not an error.
    assert!(repo.chunks_of("missing", BIZ).await.unwrap().is_empty());
}

// Keep the trait imports used.
#[allow(dead_code)]
fn _assert_traits(g: Arc<InMemoryGraphStorage>, k: Arc<InMemoryKVStorage>) {
    let _g: Arc<dyn GraphStorage> = g;
    let _k: Arc<dyn KVStorage> = k;
    let _set: HashSet<String> = HashSet::new();
}
