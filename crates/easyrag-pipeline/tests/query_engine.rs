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
use easyrag_core::storage::traits::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::JsonMap;
use easyrag_pipeline::query::{
    StructureAgentStrategy, StructureQueryStrategy, TreeSearchEngine,
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
}

impl MockVectorStorage {
    fn new(results: Vec<JsonMap>) -> Self {
        Self {
            results,
            embedding_func: test_embedding_func(),
            global_config: JsonMap::new(),
        }
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

// Keep the trait imports used.
#[allow(dead_code)]
fn _assert_traits(g: Arc<InMemoryGraphStorage>, k: Arc<InMemoryKVStorage>) {
    let _g: Arc<dyn GraphStorage> = g;
    let _k: Arc<dyn KVStorage> = k;
    let _set: HashSet<String> = HashSet::new();
}
