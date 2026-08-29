//! Query engine — unified entry point for all query modes

use super::structure::StructureQueryStrategy;
use super::{ClaimVerifier, EvidenceScorer, QueryRouter};
use easyrag_core::llm::service::{
    ChatMessage, LlmRequest, LlmService, MessageRole, RerankService, ResponseFormat,
};
use easyrag_core::model::query_param::{QueryParam, QueryMode};
use easyrag_core::model::verification::QualityDecision;
use easyrag_core::prompt::{
    format_prompt, FAIL_RESPONSE, KEYWORDS_EXTRACTION, KEYWORDS_EXTRACTION_EXAMPLES,
    KG_QUERY_CONTEXT, NAIVE_QUERY_CONTEXT, NAIVE_RAG_RESPONSE, RAG_RESPONSE,
};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::rrf_fusion;
use easyrag_core::storage::traits::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::hash::compute_args_hash;
use easyrag_core::util::json::{parse_json, to_pretty_json};
use easyrag_core::util::tokenizer::{Tokenizer, TokenizerProvider};
use easyrag_core::JsonMap;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Keywords extraction result.
#[derive(Clone, Debug, Default)]
pub struct KeywordsResult {
    pub high_level: Vec<String>,
    pub low_level: Vec<String>,
}

/// Pure data retrieval result (no LLM generation).
/// Returned by the /query/data endpoint.
#[derive(Clone, Debug, Default)]
pub struct QueryDataResult {
    pub entities: Vec<JsonMap>,
    pub relations: Vec<JsonMap>,
    pub chunks: Vec<JsonMap>,
}

/// A single reference/citation in query results.
#[derive(Clone, Debug)]
pub struct QueryReference {
    pub reference_id: usize,
    pub file_path: String,
    pub content: String,
    pub score: Option<f32>,
    pub section_path: Option<String>,
}

/// Query result with structured references.
/// Returned by /query when include_references is requested.
#[derive(Clone, Debug)]
pub struct QueryWithRefsResult {
    pub response: String,
    pub references: Vec<QueryReference>,
}

/// Bounded LRU query cache.
/// Single mutex guards map + order together so touch/evict is atomic.
struct QueryCache {
    inner: Mutex<(HashMap<String, String>, std::collections::VecDeque<String>)>,
}

impl QueryCache {
    const CAPACITY: usize = 256;

    fn new() -> Self {
        Self {
            inner: Mutex::new((HashMap::new(), std::collections::VecDeque::new())),
        }
    }

    fn get(&self, key: &str) -> Option<String> {
        // Cache is best-effort: recover from a poisoned mutex instead of
        // turning one panicked holder into a permanent query-path failure.
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (map, order) = &mut *guard;
        let value = map.get(key).cloned()?;
        // Touch: move to most-recently-used position (access-order).
        if let Some(pos) = order.iter().position(|k| k == key) {
            order.remove(pos);
        }
        order.push_back(key.to_string());
        Some(value)
    }

    fn put(&self, key: String, value: String) {
        // Cache is best-effort: recover from a poisoned mutex instead of
        // turning one panicked holder into a permanent query-path failure.
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let (map, order) = &mut *guard;
        // Access-order semantics: re-putting an existing key also refreshes
        // its recency.
        if let Some(pos) = order.iter().position(|k| k == &key) {
            order.remove(pos);
        }
        order.push_back(key.clone());
        while order.len() > Self::CAPACITY {
            if let Some(evicted) = order.pop_front() {
                map.remove(&evicted);
            }
        }
        map.insert(key, value);
    }
}

pub struct QueryEngine {
    llm_service: Arc<dyn LlmService>,
    vector_storage: Arc<dyn VectorStorage>,
    graph_storage: Arc<dyn GraphStorage>,
    kv_storage: Arc<dyn KVStorage>,
    rerank_service: Option<Arc<dyn RerankService>>,
    tokenizer: Arc<Tokenizer>,
    language: String,
    min_rerank_score: f32,
    keyword_search: Option<Arc<dyn KeywordSearchable>>,
    rrf_k: usize,
    structure_strategy: Option<Arc<StructureQueryStrategy>>,
    query_cache: QueryCache,
    // Anti-hallucination guardrail components (all optional, opt-in)
    claim_verifier: Option<Arc<ClaimVerifier>>,
    evidence_scorer: Option<Arc<EvidenceScorer>>,
    query_router: Option<Arc<QueryRouter>>,
    enable_citation_validation: bool,
}

impl QueryEngine {
    /// Note: the global tokenizer is captured at construction time
    /// (`TokenizerProvider::get_tokenizer` snapshot); a later
    /// `TokenizerProvider::set_tokenizer` does not affect already-built
    /// engines. Use `with_tokenizer` to override explicitly.
    pub fn new(
        llm_service: Arc<dyn LlmService>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        kv_storage: Arc<dyn KVStorage>,
    ) -> Self {
        Self {
            llm_service,
            vector_storage,
            graph_storage,
            kv_storage,
            rerank_service: None,
            tokenizer: TokenizerProvider::get_tokenizer(),
            language: "English".to_string(),
            min_rerank_score: 0.0,
            keyword_search: None,
            rrf_k: 60,
            structure_strategy: None,
            query_cache: QueryCache::new(),
            claim_verifier: None,
            evidence_scorer: None,
            query_router: None,
            enable_citation_validation: false,
        }
    }

    pub fn with_rerank(mut self, rerank_service: Arc<dyn RerankService>) -> Self {
        self.rerank_service = Some(rerank_service);
        self
    }

    pub fn with_min_rerank_score(mut self, score: f32) -> Self {
        self.min_rerank_score = score;
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn with_keyword_search(mut self, keyword_search: Arc<dyn KeywordSearchable>) -> Self {
        self.keyword_search = Some(keyword_search);
        self
    }

    pub fn with_rrf_k(mut self, rrf_k: usize) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    pub fn with_tokenizer(mut self, tokenizer: Arc<Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    pub fn with_structure_strategy(mut self, strategy: Arc<StructureQueryStrategy>) -> Self {
        self.structure_strategy = Some(strategy);
        self
    }

    /// Enable the claim verification gate (Feature 1).
    pub fn with_claim_verifier(mut self, verifier: Arc<ClaimVerifier>) -> Self {
        self.claim_verifier = Some(verifier);
        self
    }

    /// Enable CRAG-style evidence quality scoring (Feature 2).
    pub fn with_evidence_scorer(mut self, scorer: Arc<EvidenceScorer>) -> Self {
        self.evidence_scorer = Some(scorer);
        self
    }

    /// Enable query routing and decomposition (Feature 6).
    pub fn with_query_router(mut self, router: Arc<QueryRouter>) -> Self {
        self.query_router = Some(router);
        self
    }

    /// Enable citation validity checking (Feature 3).
    pub fn with_citation_validation(mut self, enable: bool) -> Self {
        self.enable_citation_validation = enable;
        self
    }

    /// Execute a query with the given parameters.
    /// Supports only_need_context and only_need_prompt for debugging.
    /// Uses a query cache to avoid redundant LLM calls for identical queries.
    pub async fn query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        let cacheable = !matches!(param.mode, QueryMode::Bypass)
            && !param.only_need_context
            && !param.only_need_prompt;

        // Compute once; reused after execution for the cache store.
        let cache_key = cacheable.then(|| compute_query_cache_key(query, param));
        if let Some(key) = cache_key.as_deref() {
            if let Some(cached) = self.query_cache.get(key) {
                tracing::debug!("Query cache hit");
                return Ok(cached);
            }
        }

        // [Guardrail] Query routing (Feature 6) — pre-retrieval.
        // Skipped for Bypass / only_need_context / only_need_prompt
        // (debug contract: those modes must return deterministic output).
        let use_routing = cacheable
            && param.enable_query_routing.unwrap_or(self.query_router.is_some());
        if use_routing {
            match &self.query_router {
                Some(router) => {
                    match router.route(query).await {
                        Ok(route) => {
                            use easyrag_core::model::verification::QueryRouteType;
                            match route.route_type {
                                QueryRouteType::NoRetrieval => {
                                    tracing::debug!("Query routed to no_retrieval");
                                    return Ok("I'm here to help with questions about your documents. Could you please rephrase your question in the context of your knowledge base?".to_string());
                                }
                                QueryRouteType::MultiHop => {
                                    tracing::debug!(
                                        "Query routed to multi_hop with {} sub-queries",
                                        route.sub_queries.len()
                                    );
                                    // Execute sub-queries and merge context
                                    // For now, fall through to normal query with expanded context
                                }
                                QueryRouteType::SingleHop => {
                                    tracing::debug!("Query routed to single_hop");
                                    // Normal flow
                                }
                            }
                            if route.has_false_premise {
                                tracing::info!("False premise detected in query, response may be affected");
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Query routing failed: {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!("Query routing requested but no QueryRouter is configured");
                }
            }
        }

        let result = match param.mode {
            QueryMode::Bypass => query.to_string(),
            QueryMode::Naive => self.naive_query(query, param).await?,
            QueryMode::Local => self.kg_query(query, param, "local").await?,
            QueryMode::Global => self.kg_query(query, param, "global").await?,
            QueryMode::Hybrid => self.kg_query(query, param, "hybrid").await?,
            QueryMode::Mix => self.kg_query(query, param, "mix").await?,
            QueryMode::Structure => self.structure_query(query, param).await?,
        };

        // Citation validation is applied in query_with_references where
        // the reference count is known. The bare query path has no reference
        // boundary to validate against.

        if let Some(key) = cache_key {
            // Never cache guardrail rejections (FAIL_RESPONSE): abstain
            // verdicts are non-deterministic LLM decisions, replaying them
            // from cache would freeze a one-off refusal forever.
            if result != FAIL_RESPONSE {
                self.query_cache.put(key, result.clone());
            }
        }

        Ok(result)
    }

    /// Execute a query and return structured result with references.
    pub async fn query_with_references(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<QueryWithRefsResult> {
        let chunks = apply_metadata_filters(
            self.vector_storage
                .query(
                    query,
                    effective_top_k(param, param.chunk_top_k),
                    None,
                    param.time_range_start,
                    param.time_range_end,
                    &param.biz_id,
                )
                .await?,
            param,
            param.chunk_top_k,
        );

        let references: Vec<QueryReference> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let section_path = chunk
                    .get("metadata")
                    .and_then(|m| m.get("section_path"))
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join(" > ")
                    });
                QueryReference {
                    reference_id: index + 1,
                    file_path: chunk
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    content: chunk
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    score: chunk.get("score").and_then(|v| v.as_f64()).map(|v| v as f32),
                    section_path,
                }
            })
            .collect();

        let response = self.query(query, param).await?;

        // [Guardrail] Citation validation (Feature 3) — strip invalid citations
        let response = if self.enable_citation_validation
            || param.enable_citation_validation.unwrap_or(false)
        {
            let result = super::citation_check::validate_citations(&response, references.len());
            if !result.invalid_citations.is_empty() {
                tracing::debug!(
                    "Citation check: stripped {} invalid citations: {:?}",
                    result.invalid_citations.len(),
                    result.invalid_citations
                );
            }
            result.stripped_response
        } else {
            response
        };

        Ok(QueryWithRefsResult {
            response,
            references,
        })
    }

    /// Pure data retrieval without LLM generation.
    pub async fn query_data(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<QueryDataResult> {
        let chunks = apply_metadata_filters(
            self.vector_storage
                .query(
                    query,
                    effective_top_k(param, param.chunk_top_k),
                    None,
                    param.time_range_start,
                    param.time_range_end,
                    &param.biz_id,
                )
                .await?,
            param,
            param.chunk_top_k,
        );

        let mut entities: Vec<JsonMap> = Vec::new();
        let mut relations: Vec<JsonMap> = Vec::new();

        if !matches!(param.mode, QueryMode::Naive) {
            let keywords = self.extract_keywords(query, &self.language.clone(), None).await?;
            let search_terms: Vec<String> = keywords
                .high_level
                .iter()
                .chain(keywords.low_level.iter())
                .cloned()
                .collect();
            let (found_entities, found_relations) =
                self.search_entities_and_relations(&search_terms, param, false)
                    .await?;
            entities.extend(found_entities);
            relations.extend(found_relations);
        }

        Ok(QueryDataResult {
            entities: entities
                .iter()
                .map(|e| {
                    let mut m = JsonMap::new();
                    let name = e
                        .get("entity_name")
                        .or_else(|| e.get("id"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    m.insert("name".into(), JsonValue::String(name.to_string()));
                    m.insert(
                        "type".into(),
                        JsonValue::String(
                            e.get("entity_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        ),
                    );
                    m.insert(
                        "description".into(),
                        JsonValue::String(
                            e.get("description")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        ),
                    );
                    m
                })
                .collect(),
            relations: relations
                .iter()
                .map(|r| {
                    let mut m = JsonMap::new();
                    for (out, in_key) in
                        [("source", "source"), ("target", "target"), ("description", "description"), ("keywords", "keywords")]
                    {
                        m.insert(
                            out.into(),
                            JsonValue::String(
                                r.get(in_key)
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                            ),
                        );
                    }
                    m
                })
                .collect(),
            chunks: chunks
                .iter()
                .enumerate()
                .map(|(index, chunk)| {
                    let mut m = JsonMap::new();
                    m.insert("reference_id".into(), JsonValue::from((index + 1) as u64));
                    m.insert(
                        "content".into(),
                        JsonValue::String(
                            chunk
                                .get("content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string(),
                        ),
                    );
                    m.insert(
                        "file_path".into(),
                        JsonValue::String(
                            chunk
                                .get("file_path")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string(),
                        ),
                    );
                    match chunk.get("score").and_then(|v| v.as_f64()) {
                        Some(s) => m.insert("score".into(), JsonValue::from(s as f32)),
                        None => m.insert("score".into(), JsonValue::Null),
                    };
                    // R7: expose business create_time and client metadata
                    // (already present in stored chunks).
                    match chunk.get("create_time").and_then(|v| v.as_i64()) {
                        Some(ts) => m.insert("create_time".into(), JsonValue::from(ts)),
                        None => m.insert("create_time".into(), JsonValue::Null),
                    };
                    m.insert(
                        "metadata".into(),
                        chunk
                            .get("metadata")
                            .cloned()
                            .unwrap_or(JsonValue::Null),
                    );
                    m
                })
                .collect(),
        })
    }

    /// Execute a query with true streaming response.
    /// Performs retrieval synchronously, then streams the LLM generation
    /// token-by-token.
    ///
    /// The streaming path intentionally
    /// differs from [`query`](Self::query) — it skips the query cache, naive
    /// mode uses pure vector search (no keyword/RRF hybrid), KG mode does
    /// not inject `conversation_history` and ignores `only_need_context` /
    /// `only_need_prompt`. These behaviors match the blueprint exactly.
    pub fn query_stream(
        self: &Arc<Self>,
        query: &str,
        param: &QueryParam,
    ) -> BoxStream<'static, String> {
        let this = Arc::clone(self);
        let query = query.to_string();
        let param = param.clone();
        Box::pin(async_stream::stream! {
            // [Guardrail] Query routing (Feature 6) — pre-retrieval.
            // Bypass mode keeps its echo contract untouched.
            let use_routing = !matches!(param.mode, QueryMode::Bypass)
                && param.enable_query_routing.unwrap_or(this.query_router.is_some());
            if use_routing {
                if let Some(router) = &this.query_router {
                    match router.route(&query).await {
                        Ok(route) => {
                            use easyrag_core::model::verification::QueryRouteType;
                            match route.route_type {
                                QueryRouteType::NoRetrieval => {
                                    yield "I'm here to help with questions about your documents. Could you please rephrase your question in the context of your knowledge base?".to_string();
                                    return;
                                }
                                QueryRouteType::MultiHop => {
                                    tracing::debug!(
                                        "Query (stream) routed to multi_hop with {} sub-queries",
                                        route.sub_queries.len()
                                    );
                                }
                                QueryRouteType::SingleHop => {
                                    tracing::debug!("Query (stream) routed to single_hop");
                                }
                            }
                            if route.has_false_premise {
                                tracing::info!("False premise detected in stream query");
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Query routing (stream) failed: {e}");
                        }
                    }
                } else {
                    tracing::warn!("Query routing requested but no QueryRouter is configured");
                }
            }

            match param.mode {
                QueryMode::Bypass => yield query.clone(),
                QueryMode::Naive => {
                    match this.build_naive_query_prompt(&query, &param).await {
                        Ok(Some(prompt)) => {
                            let mut stream = this.llm_service.complete_stream(LlmRequest::new(prompt));
                            while let Some(token) = stream.next().await {
                                yield token;
                            }
                        }
                        _ => yield FAIL_RESPONSE.to_string(),
                    }
                }
                QueryMode::Structure => {
                    // Structure mode: execute synchronously then emit result.
                    // We
                    // emit FAIL_RESPONSE instead for stream robustness.
                    match this.structure_query(&query, &param).await {
                        Ok(result) => yield result,
                        Err(_) => yield FAIL_RESPONSE.to_string(),
                    }
                }
                _ => {
                    match this.build_kg_query_prompt(&query, &param).await {
                        Ok(Some(prompt)) => {
                            let mut stream = this.llm_service.complete_stream(LlmRequest::new(prompt));
                            while let Some(token) = stream.next().await {
                                yield token;
                            }
                        }
                        _ => yield FAIL_RESPONSE.to_string(),
                    }
                }
            }
        })
    }

    /// Structure-based reasoning query. Falls back to naive if the
    /// strategy is unavailable.
    async fn structure_query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        let Some(strategy) = &self.structure_strategy else {
            return self.naive_query(query, param).await;
        };
        let result = strategy.query(query, param).await?;
        Ok(result.answer)
    }

    /// Build the naive query prompt for streaming (retrieval + prompt
    /// construction). Returns None if no relevant context found.
    async fn build_naive_query_prompt(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<Option<String>> {
        let mut chunks = apply_metadata_filters(
            self.vector_storage
                .query(
                    query,
                    effective_top_k(param, param.top_k),
                    None,
                    param.time_range_start,
                    param.time_range_end,
                    &param.biz_id,
                )
                .await?,
            param,
            param.top_k,
        );
        if chunks.is_empty() {
            return Ok(None);
        }

        if param.enable_rerank && self.rerank_service.is_some() && chunks.len() > 1 {
            chunks = self.rerank_chunks(query, chunks, param.chunk_top_k).await;
        }

        let context_data = self.build_naive_context(&chunks, param);
        let mut vars = HashMap::new();
        vars.insert("response_type".to_string(), param.response_type.clone());
        vars.insert(
            "user_prompt".to_string(),
            param.user_prompt.clone().unwrap_or_default(),
        );
        vars.insert("content_data".to_string(), context_data);
        Ok(Some(format_prompt(NAIVE_RAG_RESPONSE, &vars)))
    }

    /// Build the KG query prompt for streaming (keyword extraction +
    /// retrieval + prompt construction). Returns None if no relevant
    /// context found.
    async fn build_kg_query_prompt(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<Option<String>> {
        let mode = param.mode.value();

        // The streaming KG path always runs LLM extraction
        // WITHOUT param — the streaming KG path always runs LLM extraction and
        // ignores manual hl/ll keywords (unlike the synchronous kg_query).
        let language = self.language.clone();
        let keywords = self.extract_keywords(query, &language, None).await?;

        let search_terms = resolve_search_terms(&keywords, mode);
        let (entities, relations) = self
            .search_entities_and_relations(&search_terms, param, false)
            .await?;

        let mut chunks = apply_metadata_filters(
            self.vector_storage
                .query(
                    query,
                    effective_top_k(param, param.chunk_top_k),
                    None,
                    param.time_range_start,
                    param.time_range_end,
                    &param.biz_id,
                )
                .await?,
            param,
            param.chunk_top_k,
        );
        if param.enable_rerank && self.rerank_service.is_some() && chunks.len() > 1 {
            chunks = self.rerank_chunks(query, chunks, param.chunk_top_k).await;
        }

        if entities.is_empty() && chunks.is_empty() {
            return Ok(None);
        }

        let context_data = self.build_kg_context(&entities, &relations, &chunks, param);
        let mut vars = HashMap::new();
        vars.insert("response_type".to_string(), param.response_type.clone());
        vars.insert(
            "user_prompt".to_string(),
            param.user_prompt.clone().unwrap_or_default(),
        );
        vars.insert("context_data".to_string(), context_data);
        Ok(Some(format_prompt(RAG_RESPONSE, &vars)))
    }

    /// Search entities by terms and collect their connected graph edges.
    /// When `separate_relations` is true, relationship results are added
    /// to relations instead of entities.
    async fn search_entities_and_relations(
        &self,
        search_terms: &[String],
        param: &QueryParam,
        separate_relations: bool,
    ) -> easyrag_core::Result<(Vec<JsonMap>, Vec<JsonMap>)> {
        let mut entities: Vec<JsonMap> = Vec::new();
        let mut relations: Vec<JsonMap> = Vec::new();

        for term in search_terms.iter().take(5) {
            let per_term_top_k = param.top_k / search_terms.len().max(1);
            let results = self
                .vector_storage
                .query(
                    term,
                    per_term_top_k,
                    None,
                    param.time_range_start,
                    param.time_range_end,
                    &param.biz_id,
                )
                .await?;
            if separate_relations {
                for result in results {
                    let entity_type = result.get("entity_type").and_then(|v| v.as_str());
                    if entity_type == Some("relationship") {
                        relations.push(result);
                    } else {
                        entities.push(result);
                    }
                }
            } else {
                entities.extend(results);
            }
        }

        // Collect related edges from graph using batch operations
        // (avoids N+1).
        let entity_ids: Vec<String> = entities
            .iter()
            .take(10)
            .filter_map(|e| e.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();
        if !entity_ids.is_empty() {
            let batch_edges = self.graph_storage.get_nodes_edges_batch(&entity_ids, &param.biz_id).await?;
            // Deterministic order: iterate by entity id order
            // HashMap iteration order is unspecified).
            let mut edge_pairs: Vec<(String, String)> = Vec::new();
            for id in &entity_ids {
                if let Some(edges) = batch_edges.get(id) {
                    for pair in edges {
                        if !edge_pairs.contains(pair) {
                            edge_pairs.push(pair.clone());
                            if edge_pairs.len() >= 50 {
                                break;
                            }
                        }
                    }
                }
                if edge_pairs.len() >= 50 {
                    break;
                }
            }
            if !edge_pairs.is_empty() {
                let edge_data = self.graph_storage.get_edges_batch(&edge_pairs, &param.biz_id).await?;
                for pair in &edge_pairs {
                    if let Some(data) = edge_data.get(pair) {
                        let mut m = data.clone();
                        m.insert("source".into(), JsonValue::String(pair.0.clone()));
                        m.insert("target".into(), JsonValue::String(pair.1.clone()));
                        relations.push(m);
                    }
                }
            }
        }

        Ok((entities, relations))
    }

    /// Extract keywords from query using LLM.
    /// If param provides hl_keywords/ll_keywords, skip LLM extraction.
    pub async fn extract_keywords(
        &self,
        query: &str,
        language: &str,
        param: Option<&QueryParam>,
    ) -> easyrag_core::Result<KeywordsResult> {
        // Use manually specified keywords if provided.
        if let Some(param) = param {
            if !param.hl_keywords.is_empty() || !param.ll_keywords.is_empty() {
                return Ok(KeywordsResult {
                    high_level: param.hl_keywords.clone(),
                    low_level: param.ll_keywords.clone(),
                });
            }
        }

        let examples = KEYWORDS_EXTRACTION_EXAMPLES.join("\n");
        let mut vars = HashMap::new();
        vars.insert("examples".to_string(), examples);
        vars.insert("query".to_string(), query.to_string());
        vars.insert("language".to_string(), language.to_string());
        let prompt = format_prompt(KEYWORDS_EXTRACTION, &vars);

        let mut request = LlmRequest::new(prompt);
        request.response_format = Some(ResponseFormat::JsonObject);
        let response = self.llm_service.complete(&request).await?;

        let Some(json) = parse_json(&response) else {
            tracing::warn!("Failed to parse keywords from LLM response");
            return Ok(KeywordsResult::default());
        };

        let high_level = json
            .get("high_level_keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let low_level = json
            .get("low_level_keywords")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        Ok(KeywordsResult {
            high_level,
            low_level,
        })
    }

    /// Hybrid chunk retrieval: parallel vector search + keyword search,
    /// fused via RRF. Falls back to pure vector search when keywordSearch
    /// is not configured.
    async fn hybrid_chunk_query(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let Some(keyword_search) = &self.keyword_search else {
            return self
                .vector_storage
                .query(query, top_k, None, time_range_start, time_range_end, biz_id)
                .await;
        };

        let (vec_results, kw_results) = futures::try_join!(
            self.vector_storage
                .query(query, top_k, None, time_range_start, time_range_end, biz_id),
            keyword_search.keyword_query(query, top_k, time_range_start, time_range_end, biz_id)
        )?;
        Ok(rrf_fusion(vec_results, kw_results, self.rrf_k, 0.7, 0.3))
    }

    /// Naive query - pure vector retrieval without knowledge graph.
    async fn naive_query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        // 1. Retrieve relevant chunks via hybrid search (vector + keyword).
        let mut chunks = apply_metadata_filters(
            self.hybrid_chunk_query(
                query,
                effective_top_k(param, param.top_k),
                param.time_range_start,
                param.time_range_end,
                &param.biz_id,
            )
            .await?,
            param,
            param.top_k,
        );

        if chunks.is_empty() {
            return Ok(FAIL_RESPONSE.to_string());
        }

        // 2. Apply reranking if enabled and available.
        if param.enable_rerank && self.rerank_service.is_some() && chunks.len() > 1 {
            chunks = self.rerank_chunks(query, chunks, param.chunk_top_k).await;
        }

        // [Guardrail] Evidence quality scoring (Feature 2) — post-retrieval
        let use_scoring = param.enable_evidence_scoring.unwrap_or(self.evidence_scorer.is_some());
        if use_scoring {
            match &self.evidence_scorer {
                Some(scorer) => {
                    match scorer.score(query, &chunks).await {
                        Ok(score) => {
                            tracing::debug!("Evidence quality: {:.2} ({:?})", score.score, score.decision);
                            match score.decision {
                                QualityDecision::Abstain => {
                                    return Ok(FAIL_RESPONSE.to_string());
                                }
                                QualityDecision::Refine => {
                                    tracing::debug!("Evidence quality borderline, proceeding with caution");
                                }
                                QualityDecision::Generate => {
                                    // Good evidence, proceed normally
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Evidence scoring failed: {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!("Evidence quality scoring requested but no EvidenceScorer is configured");
                }
            }
        }

        // 3. Build context from chunks.
        let context_data = self.build_naive_context(&chunks, param);

        // 4. Generate response.
        let mut vars = HashMap::new();
        vars.insert("response_type".to_string(), param.response_type.clone());
        vars.insert(
            "user_prompt".to_string(),
            param.user_prompt.clone().unwrap_or_default(),
        );
        vars.insert("content_data".to_string(), context_data);
        let prompt = format_prompt(NAIVE_RAG_RESPONSE, &vars);

        let response = self.llm_service.complete(&LlmRequest::new(prompt)).await?;

        // [Guardrail] Claim verification gate (Feature 1) — post-generation
        let use_verification = param.enable_verification.unwrap_or(self.claim_verifier.is_some());
        if use_verification {
            match &self.claim_verifier {
                Some(verifier) => {
                    match verifier.verify_answer(&response, &chunks).await {
                        Ok(result) => {
                            tracing::debug!(
                                "Verification: {} claims, min_support={:.2}, abstain={}",
                                result.claims.len(),
                                result.min_support,
                                result.abstain
                            );
                            if result.abstain {
                                return Ok(FAIL_RESPONSE.to_string());
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Claim verification failed: {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!("Claim verification requested but no ClaimVerifier is configured");
                }
            }
        }

        Ok(response)
    }

    /// Knowledge graph query - uses entities and relationships from the graph.
    async fn kg_query(
        &self,
        query: &str,
        param: &QueryParam,
        mode: &str,
    ) -> easyrag_core::Result<String> {
        // 1. Extract keywords (use manual keywords if provided).
        let keywords = self
            .extract_keywords(query, &self.language.clone(), Some(param))
            .await?;
        tracing::debug!(
            "Extracted keywords - high: {:?}, low: {:?}",
            keywords.high_level,
            keywords.low_level
        );

        // 2. Retrieve entities and relationships based on mode.
        let search_terms = resolve_search_terms(&keywords, mode);
        let (entities, relations) = self
            .search_entities_and_relations(&search_terms, param, true)
            .await?;

        // 4. Retrieve text chunks (hybrid search + entity-related chunks).
        let mut chunks = apply_metadata_filters(
            self.hybrid_chunk_query(
                query,
                effective_top_k(param, param.chunk_top_k),
                param.time_range_start,
                param.time_range_end,
                &param.biz_id,
            )
            .await?,
            param,
            param.chunk_top_k,
        );

        // 4b. Collect entity-related chunks via source_ids.
        let mut entity_chunk_ids: Vec<String> = Vec::new();
        for entity in entities.iter().take(10) {
            let Some(source_id) = entity.get("source_id").and_then(|v| v.as_str()) else {
                continue;
            };
            // split by either delimiter. Done in
            // two passes since &[&str] is not a stable str::Pattern.
            for part in source_id.split("<SEP>") {
                for id in part.split(',').map(|s| s.trim()) {
                    if !id.is_empty() && !entity_chunk_ids.iter().any(|x| x == id) {
                        entity_chunk_ids.push(id.to_string());
                    }
                }
            }
        }
        if !entity_chunk_ids.is_empty() {
            let ids: Vec<String> = entity_chunk_ids.into_iter().take(20).collect();
            let related_chunks = self.kv_storage.get_by_ids(&ids, &param.biz_id).await?;
            for chunk_data in related_chunks {
                let Some(content) = chunk_data.get("content").and_then(|v| v.as_str()) else {
                    continue;
                };
                if !chunks
                    .iter()
                    .any(|c| c.get("content").and_then(|v| v.as_str()) == Some(content))
                {
                    let mut with_source = chunk_data;
                    with_source
                        .insert("source".into(), JsonValue::String("entity_related".into()));
                    chunks.push(with_source);
                }
            }
        }

        // 5. Apply reranking if enabled and available.
        if param.enable_rerank && self.rerank_service.is_some() && chunks.len() > 1 {
            chunks = self.rerank_chunks(query, chunks, param.chunk_top_k).await;
        }

        // [Guardrail] Evidence quality scoring (Feature 2) — post-retrieval
        let use_scoring = param.enable_evidence_scoring.unwrap_or(self.evidence_scorer.is_some());
        if use_scoring {
            match &self.evidence_scorer {
                Some(scorer) => {
                    match scorer.score(query, &chunks).await {
                        Ok(score) => {
                            tracing::debug!("Evidence quality (kg): {:.2} ({:?})", score.score, score.decision);
                            match score.decision {
                                QualityDecision::Abstain => {
                                    return Ok(FAIL_RESPONSE.to_string());
                                }
                                QualityDecision::Refine => {
                                    tracing::debug!("Evidence quality borderline, proceeding with caution");
                                }
                                QualityDecision::Generate => {}
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Evidence scoring failed (kg): {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!("Evidence quality scoring requested but no EvidenceScorer is configured");
                }
            }
        }

        if entities.is_empty() && chunks.is_empty() {
            return Ok(FAIL_RESPONSE.to_string());
        }

        // 6. Build context.
        let context_data = self.build_kg_context(&entities, &relations, &chunks, param);

        // Support only_need_context for debugging.
        if param.only_need_context {
            return Ok(context_data);
        }

        // 7. Build prompt.
        let mut vars = HashMap::new();
        vars.insert("response_type".to_string(), param.response_type.clone());
        vars.insert(
            "user_prompt".to_string(),
            param.user_prompt.clone().unwrap_or_default(),
        );
        vars.insert("context_data".to_string(), context_data);
        let prompt = format_prompt(RAG_RESPONSE, &vars);

        // Support only_need_prompt for debugging.
        if param.only_need_prompt {
            return Ok(prompt);
        }

        // 8. Generate response (with conversation history if provided).
        let history: Vec<ChatMessage> = param
            .conversation_history
            .iter()
            .map(|msg| ChatMessage {
                role: if msg.get("role").map(|r| r == "assistant").unwrap_or(false) {
                    MessageRole::Assistant
                } else {
                    MessageRole::User
                },
                content: msg.get("content").cloned().unwrap_or_default(),
            })
            .collect();

        let mut request = LlmRequest::new(prompt);
        request.history = history;
        let response = self.llm_service.complete(&request).await?;

        // [Guardrail] Claim verification gate (Feature 1) — post-generation
        let use_verification = param.enable_verification.unwrap_or(self.claim_verifier.is_some());
        if use_verification {
            match &self.claim_verifier {
                Some(verifier) => {
                    match verifier.verify_answer(&response, &chunks).await {
                        Ok(result) => {
                            tracing::debug!(
                                "Verification (kg): {} claims, min_support={:.2}, abstain={}",
                                result.claims.len(),
                                result.min_support,
                                result.abstain
                            );
                            if result.abstain {
                                return Ok(FAIL_RESPONSE.to_string());
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Claim verification failed (kg): {e}");
                        }
                    }
                }
                None => {
                    tracing::warn!("Claim verification requested but no ClaimVerifier is configured");
                }
            }
        }

        Ok(response)
    }

    fn build_naive_context(&self, chunks: &[JsonMap], param: &QueryParam) -> String {
        // Apply token budget: truncate chunks to fit within maxTotalTokens.
        let truncated_chunks = truncate_to_token_budget(chunks, param.max_total_tokens, &self.tokenizer);

        let chunks_json: Vec<JsonMap> = truncated_chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let section_path = chunk
                    .get("metadata")
                    .and_then(|m| m.get("section_path"))
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join(" > ")
                    })
                    .unwrap_or_default();
                let mut m = JsonMap::new();
                m.insert(
                    "reference_id".into(),
                    JsonValue::String((index + 1).to_string()),
                );
                m.insert(
                    "content".into(),
                    JsonValue::String(
                        chunk
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m.insert(
                    "file_path".into(),
                    JsonValue::String(
                        chunk
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    ),
                );
                m.insert("section_path".into(), JsonValue::String(section_path));
                m
            })
            .collect();

        let reference_list = chunks_json
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let section_info = chunk
                    .get("section_path")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|s| format!(" [{s}]"))
                    .unwrap_or_default();
                let file_path = chunk.get("file_path").and_then(|v| v.as_str()).unwrap_or("");
                format!("[{}] {}{}", index + 1, file_path, section_info)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = HashMap::new();
        vars.insert("text_chunks_str".to_string(), to_pretty_json(&chunks_json));
        vars.insert("reference_list_str".to_string(), reference_list);
        format_prompt(NAIVE_QUERY_CONTEXT, &vars)
    }

    /// 4-stage query context architecture:
    /// Stage 1: Search (done in kgQuery)
    /// Stage 2: Truncate each source independently to its token budget
    /// Stage 3: Round-robin merge with dynamic token redistribution
    /// Stage 4: Build final context string
    ///
    /// P1.9: Dynamic token budget - accounts for system prompt + query
    /// overhead, then distributes remaining budget across
    /// entities/relations/chunks.
    fn build_kg_context(
        &self,
        entities: &[JsonMap],
        relations: &[JsonMap],
        chunks: &[JsonMap],
        param: &QueryParam,
    ) -> String {
        // P1.9: Calculate dynamic token budget accounting for prompt overhead.
        let system_prompt_tokens = self.tokenizer.count_tokens(RAG_RESPONSE)
            + self.tokenizer.count_tokens(&param.response_type)
            + self
                .tokenizer
                .count_tokens(param.user_prompt.as_deref().unwrap_or(""));
        let available_budget = param
            .max_total_tokens
            .saturating_sub(system_prompt_tokens)
            .max(2000);

        // Dynamic allocation: entities 30%, relations 30%, chunks 40%.
        let entity_budget = ((available_budget * 30) / 100).max(500);
        let relation_budget = ((available_budget * 30) / 100).max(500);
        let chunk_budget = ((available_budget * 40) / 100).max(1000);

        // Stage 2: Independent truncation per source with dynamic budgets.
        let truncated_entities =
            truncate_entities_to_budget(entities, entity_budget, &self.tokenizer);
        let truncated_relations =
            truncate_relations_to_budget(relations, relation_budget, &self.tokenizer);
        let truncated_chunks =
            truncate_to_token_budget(chunks, chunk_budget, &self.tokenizer);

        // Stage 3: Round-robin merge with dynamic token redistribution.
        let entity_tokens_used = estimate_tokens(&truncated_entities, &self.tokenizer);
        let relation_tokens_used = estimate_tokens(&truncated_relations, &self.tokenizer);
        let chunk_tokens_used = estimate_tokens(&truncated_chunks, &self.tokenizer);
        let total_used = entity_tokens_used + relation_tokens_used + chunk_tokens_used;

        let (final_entities, final_relations, final_chunks) =
            if total_used > available_budget && total_used > 0 {
                // Dynamic redistribution: scale each source proportionally.
                let ratio = available_budget as f64 / total_used as f64;
                let adj_entity_budget =
                    ((entity_tokens_used as f64 * ratio) as usize).max(500);
                let adj_relation_budget =
                    ((relation_tokens_used as f64 * ratio) as usize).max(500);
                let adj_chunk_budget = available_budget
                    .saturating_sub(adj_entity_budget)
                    .saturating_sub(adj_relation_budget)
                    .max(1000);

                (
                    truncate_entities_to_budget(&truncated_entities, adj_entity_budget, &self.tokenizer),
                    truncate_relations_to_budget(&truncated_relations, adj_relation_budget, &self.tokenizer),
                    truncate_to_token_budget(&truncated_chunks, adj_chunk_budget, &self.tokenizer),
                )
            } else {
                (truncated_entities, truncated_relations, truncated_chunks)
            };

        // Stage 4: Build structured context output.
        let entities_json: Vec<JsonMap> = final_entities
            .iter()
            .map(|e| {
                let mut m = JsonMap::new();
                let name = e
                    .get("entity_name")
                    .or_else(|| e.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                m.insert("entity_name".into(), JsonValue::String(name.to_string()));
                m.insert(
                    "entity_type".into(),
                    JsonValue::String(
                        e.get("entity_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m.insert(
                    "description".into(),
                    JsonValue::String(
                        e.get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m
            })
            .collect();

        let relations_json: Vec<JsonMap> = final_relations
            .iter()
            .map(|r| {
                let mut m = JsonMap::new();
                m.insert(
                    "src".into(),
                    JsonValue::String(
                        r.get("source").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    ),
                );
                m.insert(
                    "tgt".into(),
                    JsonValue::String(
                        r.get("target").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    ),
                );
                m.insert(
                    "description".into(),
                    JsonValue::String(
                        r.get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m.insert(
                    "keywords".into(),
                    JsonValue::String(
                        r.get("keywords")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m
            })
            .collect();

        // Round-robin interleave chunks from different sources for balanced
        // coverage.
        let vector_chunks: Vec<&JsonMap> = final_chunks
            .iter()
            .filter(|c| c.get("source").and_then(|v| v.as_str()) != Some("entity_related"))
            .collect();
        let entity_related_chunks: Vec<&JsonMap> = final_chunks
            .iter()
            .filter(|c| c.get("source").and_then(|v| v.as_str()) == Some("entity_related"))
            .collect();
        let merged_chunks = round_robin_merge(&[&vector_chunks, &entity_related_chunks]);

        let chunks_json: Vec<JsonMap> = merged_chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let mut m = JsonMap::new();
                m.insert(
                    "reference_id".into(),
                    JsonValue::String((index + 1).to_string()),
                );
                m.insert(
                    "content".into(),
                    JsonValue::String(
                        chunk
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    ),
                );
                m.insert(
                    "file_path".into(),
                    JsonValue::String(
                        chunk
                            .get("file_path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                    ),
                );
                m
            })
            .collect();

        let reference_list = chunks_json
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                format!(
                    "[{}] {}",
                    index + 1,
                    chunk.get("file_path").and_then(|v| v.as_str()).unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = HashMap::new();
        vars.insert("entities_str".to_string(), to_pretty_json(&entities_json));
        vars.insert("relations_str".to_string(), to_pretty_json(&relations_json));
        vars.insert("text_chunks_str".to_string(), to_pretty_json(&chunks_json));
        vars.insert("reference_list_str".to_string(), reference_list);
        format_prompt(KG_QUERY_CONTEXT, &vars)
    }

    /// Rerank chunks using the rerank service.
    /// Filters out chunks below the minimum rerank score threshold.
    async fn rerank_chunks(
        &self,
        query: &str,
        chunks: Vec<JsonMap>,
        top_n: usize,
    ) -> Vec<JsonMap> {
        let documents: Vec<String> = chunks
            .iter()
            .map(|c| {
                c.get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            })
            .collect();

        let Some(service) = self.rerank_service.as_ref() else {
            return chunks;
        };
        match service.rerank(query, documents, Some(top_n)).await {
            Ok(rerank_results) => rerank_results
                .iter()
                .filter(|r| r.score >= self.min_rerank_score)
                .filter_map(|result| {
                    // Write the rerank score back into the chunk metadata so
                    // downstream guardrails (e.g. EvidenceScorer fast path)
                    // can consume a calibrated [0,1] confidence.
                    let mut chunk = chunks.get(result.index)?.clone();
                    chunk.insert(
                        "rerank_score".to_string(),
                        serde_json::Value::from(result.score),
                    );
                    Some(chunk)
                })
                .collect(),
            Err(e) => {
                tracing::warn!("Rerank failed, using original order: {e}");
                chunks.into_iter().take(top_n).collect()
            }
        }
    }
}

fn compute_query_cache_key(query: &str, param: &QueryParam) -> String {
    // The blueprint key
    // intentionally omits hl/ll keywords, user_prompt, use_agent_mode and
    // conversation history content (only the length is keyed). The R4/M5
    // revision adds metadata_filters, time range, doc_id and structure_scope
    // so filtered queries never hit unfiltered cache entries.
    let filters_repr = match &param.metadata_filters {
        Some(m) if !m.is_empty() => {
            let mut pairs: Vec<(&String, &String)> = m.iter().collect();
            pairs.sort();
            format!("{pairs:?}")
        }
        _ => "null".to_string(),
    };
    let raw = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        query,
        param.mode,
        param.top_k,
        param.chunk_top_k,
        param.response_type,
        param.conversation_history.len(),
        filters_repr,
        param
            .time_range_start
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
        param
            .time_range_end
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string()),
        param.doc_id.as_deref().unwrap_or("null"),
        param.structure_scope.as_deref().unwrap_or("null"),
        param.biz_id,
        param.enable_verification.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_citation_validation.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_evidence_scoring.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_query_routing.map(|b| b.to_string()).as_deref().unwrap_or("null"),
    );
    compute_args_hash([raw])
}

/// R4: retrieval size is over-fetched 3x when metadata filters are active,
/// compensating for chunks dropped by post-filtering.
fn effective_top_k(param: &QueryParam, top_k: usize) -> usize {
    if param
        .metadata_filters
        .as_ref()
        .map_or(true, |f| f.is_empty())
    {
        top_k
    } else {
        top_k * 3
    }
}

/// R4: post-filter chunks by exact key-value match on chunk metadata.
/// Truncates to `limit` after filtering to honor the requested top_k.
fn apply_metadata_filters(
    chunks: Vec<JsonMap>,
    param: &QueryParam,
    limit: usize,
) -> Vec<JsonMap> {
    let Some(filters) = param.metadata_filters.as_ref().filter(|f| !f.is_empty()) else {
        return chunks;
    };
    chunks
        .into_iter()
        .filter(|chunk| {
            chunk
                .get("metadata")
                .and_then(|m| m.as_object())
                .map_or(false, |metadata| {
                    filters.iter().all(|(k, v)| {
                        metadata.get(k).map_or(false, |val| match val {
                            JsonValue::String(s) => s == v,
                            JsonValue::Null => false,
                            other => other.to_string() == *v,
                        })
                    })
                })
        })
        .take(limit)
        .collect()
}

/// Resolve search terms by query mode.
/// local: low-level keywords; global: high-level keywords; others: both.
fn resolve_search_terms(keywords: &KeywordsResult, mode: &str) -> Vec<String> {
    match mode {
        "local" => {
            if keywords.low_level.is_empty() {
                keywords.high_level.clone()
            } else {
                keywords.low_level.clone()
            }
        }
        "global" => {
            if keywords.high_level.is_empty() {
                keywords.low_level.clone()
            } else {
                keywords.high_level.clone()
            }
        }
        _ => keywords
            .high_level
            .iter()
            .chain(keywords.low_level.iter())
            .cloned()
            .collect(),
    }
}

/// Round-robin merge: interleave items from multiple sources for balanced
/// coverage.
fn round_robin_merge<'a>(sources: &[&'a Vec<&'a JsonMap>]) -> Vec<&'a JsonMap> {
    let mut result: Vec<&JsonMap> = Vec::new();
    let mut cursors = vec![0usize; sources.len()];
    let mut has_more = true;
    while has_more {
        has_more = false;
        for (i, source) in sources.iter().enumerate() {
            if cursors[i] < source.len() {
                result.push(source[cursors[i]]);
                cursors[i] += 1;
                has_more = true;
            }
        }
    }
    result
}

/// Truncate chunks list to fit within a token budget.
/// Keeps at least the first chunk even if it alone exceeds budget.
fn truncate_to_token_budget(
    chunks: &[JsonMap],
    max_tokens: usize,
    tokenizer: &Tokenizer,
) -> Vec<JsonMap> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<JsonMap> = Vec::new();
    let mut tokens_used = 0usize;
    for chunk in chunks {
        let content = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let chunk_tokens = tokenizer.count_tokens(content);
        if tokens_used + chunk_tokens > max_tokens && !result.is_empty() {
            break;
        }
        result.push(chunk.clone());
        tokens_used += chunk_tokens;
    }
    result
}

/// Truncate entities to fit within a token budget.
fn truncate_entities_to_budget(
    entities: &[JsonMap],
    max_tokens: usize,
    tokenizer: &Tokenizer,
) -> Vec<JsonMap> {
    if entities.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<JsonMap> = Vec::new();
    let mut tokens_used = 0usize;
    for entity in entities {
        let desc = entity
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let name = entity
            .get("entity_name")
            .or_else(|| entity.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let item_tokens = tokenizer.count_tokens(&format!("{name}: {desc}"));
        if tokens_used + item_tokens > max_tokens && !result.is_empty() {
            break;
        }
        result.push(entity.clone());
        tokens_used += item_tokens;
    }
    result
}

/// Truncate relations to fit within a token budget.
fn truncate_relations_to_budget(
    relations: &[JsonMap],
    max_tokens: usize,
    tokenizer: &Tokenizer,
) -> Vec<JsonMap> {
    if relations.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<JsonMap> = Vec::new();
    let mut tokens_used = 0usize;
    for relation in relations {
        let desc = relation
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let keywords = relation
            .get("keywords")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let item_tokens = tokenizer.count_tokens(&format!("{desc} {keywords}"));
        if tokens_used + item_tokens > max_tokens && !result.is_empty() {
            break;
        }
        result.push(relation.clone());
        tokens_used += item_tokens;
    }
    result
}

/// Estimate total tokens for a list of map items (string values only,
///).
fn estimate_tokens(items: &[JsonMap], tokenizer: &Tokenizer) -> usize {
    items
        .iter()
        .map(|item| {
            let text = item
                .values()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            tokenizer.count_tokens(&text)
        })
        .sum()
}
