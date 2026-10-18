//! Query engine — unified entry point for all query modes

use super::structure::{SectionHit, StructureQueryStrategy};
use super::tree_search::ReasoningStep;
use super::{ClaimVerifier, EvidenceScorer, QueryRouter};
use easyrag_core::llm::service::{
    ChatMessage, LlmRequest, LlmService, MessageRole, RerankService, ResponseFormat,
};
use easyrag_core::model::document_structure::section_path_of;
use easyrag_core::model::query_param::{QueryParam, QueryMode};
use easyrag_core::model::source_id::parse_source_ids;
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
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Keywords extraction result.
#[derive(Clone, Debug, Default)]
pub struct KeywordsResult {
    pub high_level: Vec<String>,
    pub low_level: Vec<String>,
}

/// Pure data retrieval result (no LLM generation).
/// Returned by the /query/data endpoint.
///
/// The last four fields are the R8 additions: raw structure-index hits plus
/// explicit degradation signalling. They stay empty on every request that did
/// not ask for structure retrieval, so a structure-disabled deployment returns
/// exactly the pre-R8 body plus three inert keys.
#[derive(Clone, Debug, Default)]
pub struct QueryDataResult {
    pub entities: Vec<JsonMap>,
    pub relations: Vec<JsonMap>,
    pub chunks: Vec<JsonMap>,
    /// Raw structure hits (R8), camelCase-keyed to match the request DTO.
    /// Unlike `chunks` — whose keys are snake_case for historical reasons and
    /// stay that way — this field is new, so it follows the wire convention the
    /// caller already sees on the way in.
    pub sections: Vec<JsonMap>,
    /// Tree-search reasoning steps behind `sections`; empty when structure
    /// retrieval did not run or took the zero-LLM small-document path.
    pub reasoning_chain: Vec<ReasoningStep>,
    /// Mirror of `!degrade_reasons.is_empty()`, so a client that only cares
    /// whether the answer is complete does not have to inspect the list.
    pub degraded: bool,
    /// Stable machine-readable degradation markers. Enumerated in
    /// `.docs/easyrag-data-endpoint-mode-matrix.md`.
    pub degrade_reasons: Vec<String>,
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
    /// R8 gradual-rollout gate (`QUERY_DATA_STRUCTURE_ENABLED`). `false` keeps
    /// `query_data` on its pre-R8 behaviour: no structure retrieval, and
    /// `structure_disabled_by_config` is reported when a caller asks for it.
    enable_data_structure: bool,
}

/// Recall strategy for a chunk retrieval.
///
/// The two variants are not interchangeable: `Hybrid` fuses keyword hits in via
/// RRF, so it can return chunks `Vector` never would, and it orders them by
/// fusion rank rather than by similarity. Which paths use which is pre-existing
/// and stays that way — with one R9 exception: the pure-data path no longer
/// hardcodes `Vector` but takes it from [`data_plan`], so `naive`, `hybrid` and
/// `mix` requests recall the same way the synchronous answer paths always did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Recall {
    /// `vector_storage.query` alone — the reference list, both streaming-prompt
    /// paths, and whichever pure-data modes [`data_plan`] maps here.
    Vector,
    /// `hybrid_chunk_query`: vector + keyword fused by RRF — the synchronous
    /// naive answer path, the KG answer path, and the pure-data
    /// `naive`/`hybrid`/`mix` modes.
    Hybrid,
}

/// Which `QueryParam` budget sizes a retrieval. The same field feeds both the
/// recall `top_k` and the post-filter limit, so a path cannot over-fetch and
/// then truncate to a different number by accident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKField {
    /// `param.top_k` — only the naive paths.
    TopK,
    /// `param.chunk_top_k` — every other path.
    ChunkTopK,
}

impl TopKField {
    fn of(self, param: &QueryParam) -> usize {
        match self {
            TopKField::TopK => param.top_k,
            TopKField::ChunkTopK => param.chunk_top_k,
        }
    }
}

/// The router's own reply to a query no corpus could answer (Feature 6). Both
/// entry points returned this string verbatim; it lives here once.
const NO_RETRIEVAL_RESPONSE: &str = "I'm here to help with questions about your documents. Could you please rephrase your question in the context of your knowledge base?";

/// Outcome of the pre-retrieval routing gate.
enum Routing {
    /// Routing did not divert the query — carry on with retrieval.
    Proceed,
    /// The router answered outright; return this text without retrieving.
    Answer(String),
}

/// Per-caller log wording of the routing gate. `scope` slots into
/// `Query{scope} routed to …` and `Query routing{scope} failed: …`; the
/// false-premise line differs wholesale between the two callers, so it is
/// carried verbatim. Log text only, never control flow — which is what lets one
/// implementation serve both entry points.
struct RoutingLogs {
    scope: &'static str,
    false_premise: &'static str,
}

const ROUTING_LOGS_SYNC: RoutingLogs = RoutingLogs {
    scope: "",
    false_premise: "False premise detected in query, response may be affected",
};

const ROUTING_LOGS_STREAM: RoutingLogs = RoutingLogs {
    scope: " (stream)",
    false_premise: "False premise detected in stream query",
};

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
            enable_data_structure: false,
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

    /// Enable raw structure retrieval on the pure-data path (R8).
    ///
    /// Off by default so the endpoint's behaviour — and its LLM bill — is
    /// unchanged until an operator opts in, mirroring the streaming-ingestion
    /// rollout gate.
    pub fn with_data_structure(mut self, enable: bool) -> Self {
        self.enable_data_structure = enable;
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
        if let Routing::Answer(text) = self
            .route_query(query, use_routing, &ROUTING_LOGS_SYNC)
            .await
        {
            return Ok(text);
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
        let chunks = self
            .retrieve_chunks(query, param, Recall::Vector, TopKField::ChunkTopK)
            .await?;

        let references: Vec<QueryReference> = chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                let section_path = section_path_of(chunk);
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
    ///
    /// What each `mode` does here is decided in exactly one place —
    /// [`data_plan`] — and every branch below reads a field off it rather than
    /// re-matching on the mode. The caller-facing matrix, the `sections[]`
    /// field dictionary and the degradation-marker enumeration live in
    /// `.docs/easyrag-data-endpoint-mode-matrix.md`; they are deliberately not
    /// restated here, because a second copy would drift.
    ///
    /// Structure retrieval (R8) returns the raw hits and never synthesises an
    /// answer: the caller is an Agent tool, and an LLM retelling would cost an
    /// extra round trip while destroying the provenance the Agent needs.
    pub async fn query_data(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<QueryDataResult> {
        let plan = data_plan(param.mode);
        let mut degrade_reasons: Vec<String> = Vec::new();

        // R9 signalling. `hybrid_chunk_query` has always fallen back to
        // vector-only when no keyword index is configured; only the reporting is
        // new. It fires on the modes whose plan actually asks for fusion, so the
        // vector-only ones (`structure`/`local`/`global`/`bypass`) stay clean.
        //
        // `naive` is a fused mode, so a deployment without a keyword index
        // reports this on the endpoint's most common mode. That is the honest
        // signal rather than a noisy one: the recall really is narrower than the
        // plan asked for, and a client comparing `/api/query/data` against
        // `/api/query` needs to know the two are not seeing the same corpus.
        if plan.recall == Recall::Hybrid && self.keyword_search.is_none() {
            degrade_reasons.push(super::degrade::KEYWORD_SEARCH_UNAVAILABLE.to_string());
        }

        let mut chunks = self
            .retrieve_chunks(query, param, plan.recall, TopKField::ChunkTopK)
            .await?;

        let mut entities: Vec<JsonMap> = Vec::new();
        let mut relations: Vec<JsonMap> = Vec::new();

        if plan.run_kg {
            // R10: hand `param` over so caller-supplied hl/ll keywords
            // short-circuit the extraction call, exactly as `kg_query` does.
            let keywords = self
                .extract_keywords(query, &self.language.clone(), Some(param))
                .await?;
            let search_terms: Vec<String> = keywords
                .high_level
                .iter()
                .chain(keywords.low_level.iter())
                .cloned()
                .collect();
            let (found_entities, found_relations) = self
                .search_entities_and_relations(&search_terms, param)
                .await?;

            // R9 `mix`: fold in the chunks reachable from the entities just hit.
            // Appended before the projection below so they carry the same
            // `reference_id` / `score` / `metadata` shape as the recalled ones.
            //
            // Deliberate deviation, matching this endpoint's long-standing
            // contract: vector hits are all entity/chunk rows (ingest never
            // writes `rel-` rows to the vector store), so `relations` stays
            // empty. Those rows carry `source_id` too, so letting them
            // drive the backfill is correct, not merely tolerated.
            if plan.backfill_entity_chunks {
                let related = self
                    .entity_related_chunks(&found_entities, &chunks, param)
                    .await?;
                chunks.extend(related);
            }

            entities.extend(found_entities);
            relations.extend(found_relations);
        }

        // R8: raw structure hits, opt-in per request and gated per deployment.
        // A bizIds override opts out of the tree search entirely: the structure
        // strategy still scopes by the single normalized `biz_id`, so running
        // it would silently answer from one line of the requested set — say so
        // with a machine-readable marker instead (signal hygiene: the marker
        // only ever fires when sections were actually asked for).
        let mut sections: Vec<JsonMap> = Vec::new();
        let mut reasoning_chain: Vec<ReasoningStep> = Vec::new();
        if wants_sections(param) {
            if !param.biz_ids.is_empty() {
                degrade_reasons.push(super::degrade::STRUCTURE_MULTI_BIZ_UNSUPPORTED.to_string());
            } else {
                match &self.structure_strategy {
                    Some(strategy) if self.enable_data_structure => {
                        let found = strategy.search_sections(query, param).await?;
                        reasoning_chain = found.reasoning_chain;
                        degrade_reasons.extend(found.degrade_reasons);
                        sections = found
                            .sections
                            .iter()
                            .enumerate()
                            .map(|(rank, hit)| section_json(hit, rank))
                            .collect();
                    }
                    Some(_) => {
                        degrade_reasons.push(super::degrade::STRUCTURE_DISABLED.to_string());
                    }
                    None => {
                        degrade_reasons.push(super::degrade::STRUCTURE_UNAVAILABLE.to_string());
                    }
                }
            }
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
                    // Per-chunk biz provenance: the multi-biz recall paths
                    // echo the biz line each record was read from; the
                    // single-biz path backfills the normalized `param.biz_id`
                    // so the field is always present (additive, backward
                    // compatible).
                    m.insert(
                        "biz_id".into(),
                        JsonValue::String(
                            chunk
                                .get("biz_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or(&param.biz_id)
                                .to_string(),
                        ),
                    );
                    m.insert(
                        "metadata".into(),
                        chunk
                            .get("metadata")
                            .cloned()
                            .unwrap_or(JsonValue::Null),
                    );
                    // R1 (archive layer): echo the four consolidation fields on
                    // every hit. Chunks written before R1 carry none of them,
                    // so the defaults here are also the backward-compatible
                    // contract (active, level 0, no refs, not merged).
                    m.insert(
                        "archived".into(),
                        JsonValue::Bool(
                            chunk
                                .get("archived")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false),
                        ),
                    );
                    m.insert(
                        "level".into(),
                        JsonValue::from(
                            chunk.get("level").and_then(|v| v.as_i64()).unwrap_or(0),
                        ),
                    );
                    m.insert(
                        "ref_ids".into(),
                        chunk
                            .get("ref_ids")
                            .cloned()
                            .unwrap_or_else(|| JsonValue::Array(Vec::new())),
                    );
                    m.insert(
                        "merged_into".into(),
                        chunk.get("merged_into").cloned().unwrap_or(JsonValue::Null),
                    );
                    // Addressable provenance: the record's `full_doc_id`, i.e. the
                    // same id `/api/documents/list` returns. A missing id projects
                    // `null`, never an empty string that could pass for an id.
                    m.insert(
                        "doc_id".into(),
                        match chunk.get("full_doc_id").and_then(|v| v.as_str()) {
                            Some(id) => JsonValue::String(id.to_string()),
                            None => JsonValue::Null,
                        },
                    );
                    m
                })
                .collect(),
            sections,
            reasoning_chain,
            degraded: !degrade_reasons.is_empty(),
            degrade_reasons,
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
            if let Routing::Answer(text) = this
                .route_query(&query, use_routing, &ROUTING_LOGS_STREAM)
                .await
            {
                yield text;
                return;
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
        let chunks = self.naive_recall(query, param, Recall::Vector).await?;
        if chunks.is_empty() {
            return Ok(None);
        }

        let context_data = self.build_naive_context(&chunks, param);
        Ok(Some(self.render_answer_prompt(
            &NAIVE_RAG_RESPONSE,
            "content_data",
            context_data,
            param,
        )))
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

        // Three deliberate deviations from the synchronous `kg_query`, all
        // long-standing: extraction runs WITHOUT `param`, so manual hl/ll
        // keywords are ignored; and there is no entity-related-chunk pass (4b),
        // no evidence gate and no conversation history.
        let language = self.language.clone();
        let keywords = self.extract_keywords(query, &language, None).await?;

        let search_terms = resolve_search_terms(&keywords, mode);
        let (entities, relations) = self
            .search_entities_and_relations(&search_terms, param)
            .await?;

        let chunks = self
            .retrieve_chunks(query, param, Recall::Vector, TopKField::ChunkTopK)
            .await?;
        let chunks = self.rerank_if_enabled(query, chunks, param).await;

        if entities.is_empty() && chunks.is_empty() {
            return Ok(None);
        }

        let context_data = self.build_kg_context(&entities, &relations, &chunks, param);
        Ok(Some(self.render_answer_prompt(
            &RAG_RESPONSE,
            "context_data",
            context_data,
            param,
        )))
    }

    /// Search entities by terms and collect their connected graph edges.
    async fn search_entities_and_relations(
        &self,
        search_terms: &[String],
        param: &QueryParam,
    ) -> easyrag_core::Result<(Vec<JsonMap>, Vec<JsonMap>)> {
        let mut entities: Vec<JsonMap> = Vec::new();
        let mut relations: Vec<JsonMap> = Vec::new();

        // bizIds override (data endpoint only): entity recall reads the same
        // multi-biz corpus as the chunk recall, so the two cannot disagree
        // about what exists.
        let multi_biz_ids: Vec<String> = if param.biz_ids.is_empty() {
            Vec::new()
        } else {
            param.effective_biz_ids()
        };
        let multi_biz_refs: Vec<&str> = multi_biz_ids.iter().map(String::as_str).collect();

        for term in search_terms.iter().take(5) {
            let per_term_top_k = param.top_k / search_terms.len().max(1);
            let results = if multi_biz_refs.is_empty() {
                self.vector_storage
                    .query(
                        term,
                        per_term_top_k,
                        None,
                        param.time_range_start,
                        param.time_range_end,
                        param.exclude_archived,
                        &param.biz_id,
                    )
                    .await?
            } else {
                self.vector_storage
                    .query_multi_biz(
                        term,
                        per_term_top_k,
                        None,
                        param.time_range_start,
                        param.time_range_end,
                        param.exclude_archived,
                        &multi_biz_refs,
                    )
                    .await?
            };
            // Ingest never writes `rel-` rows to the vector store, so every
            // hit here is an entity/chunk row; there is nothing to separate.
            entities.extend(results);
        }

        // Collect related edges from graph using batch operations
        // (avoids N+1).
        let entity_ids: Vec<String> = entities
            .iter()
            .take(10)
            .filter_map(|e| e.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();
        if !entity_ids.is_empty() {
            let batch_edges = if multi_biz_refs.is_empty() {
                self.graph_storage
                    .get_nodes_edges_batch(&entity_ids, &param.biz_id)
                    .await?
            } else {
                self.graph_storage
                    .get_nodes_edges_batch_multi_biz(&entity_ids, &multi_biz_refs)
                    .await?
            };
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
                let edge_data = if multi_biz_refs.is_empty() {
                    self.graph_storage
                        .get_edges_batch(&edge_pairs, &param.biz_id)
                        .await?
                } else {
                    self.graph_storage
                        .get_edges_batch_multi_biz(&edge_pairs, &multi_biz_refs)
                        .await?
                };
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
        exclude_archived: bool,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let Some(keyword_search) = &self.keyword_search else {
            return self
                .vector_storage
                .query(
                    query,
                    top_k,
                    None,
                    time_range_start,
                    time_range_end,
                    exclude_archived,
                    biz_id,
                )
                .await;
        };

        let (vec_results, kw_results) = futures::try_join!(
            self.vector_storage.query(
                query,
                top_k,
                None,
                time_range_start,
                time_range_end,
                exclude_archived,
                biz_id
            ),
            keyword_search.keyword_query(query, top_k, time_range_start, time_range_end, biz_id)
        )?;
        let kw_results = self
            .hydrate_keyword_hits(kw_results, &[biz_id], exclude_archived)
            .await?;
        Ok(rrf_fusion(vec_results, kw_results, self.rrf_k, 0.7, 0.3))
    }

    /// R1/R2 (archive layer): the keyword index schema is fixed and carries no
    /// archive fields. Batch-read the authoritative vector rows *before*
    /// fusion so every keyword hit (a) echoes the same four fields as vector
    /// hits and (b) is dropped when `exclude_archived` is set. The vector leg
    /// itself stays filtered in SQL; a hit whose row is gone keeps the pre-R2
    /// behaviour (it is not known to be archived).
    async fn hydrate_keyword_hits(
        &self,
        hits: Vec<JsonMap>,
        biz_ids: &[&str],
        exclude_archived: bool,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if hits.is_empty() {
            return Ok(hits);
        }
        // Group by the biz each hit came from so every read stays inside one
        // shard; a hit with no provenance is left unhydrated rather than
        // guessing a shard, and single-biz callers fall back to their only shard.
        let mut by_biz: HashMap<String, Vec<String>> = HashMap::new();
        for hit in &hits {
            let Some(id) = hit.get("id").and_then(|v| v.as_str()) else {
                continue;
            };
            let Some(biz) = hit
                .get("biz_id")
                .and_then(|v| v.as_str())
                .filter(|b| !b.is_empty())
                .or_else(|| if biz_ids.len() == 1 { biz_ids.first().copied() } else { None })
            else {
                continue;
            };
            by_biz.entry(biz.to_string()).or_default().push(id.to_string());
        }
        let mut rows_by_id: HashMap<String, JsonMap> = HashMap::new();
        for (biz, ids) in by_biz {
            for row in self.vector_storage.get_by_ids(&ids, &biz).await? {
                if let Some(id) = row.get("id").and_then(|v| v.as_str()) {
                    rows_by_id.insert(id.to_string(), row);
                }
            }
        }
        let mut out = Vec::with_capacity(hits.len());
        for mut hit in hits {
            let Some(id) = hit
                .get("id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
            else {
                out.push(hit);
                continue;
            };
            let Some(row) = rows_by_id.get(&id) else {
                out.push(hit);
                continue;
            };
            let archived = row.get("archived").and_then(|v| v.as_bool()).unwrap_or(false);
            if exclude_archived && archived {
                continue;
            }
            hit.insert("archived".into(), JsonValue::Bool(archived));
            // Tantivy's schema stores no `full_doc_id`, so a keyword-only hit
            // would project a null `doc_id` where the pg leg (whole content
            // column) does not. The authoritative row is already in hand.
            if let Some(full_doc_id) = row.get("full_doc_id") {
                hit.insert("full_doc_id".into(), full_doc_id.clone());
            }
            hit.insert(
                "level".into(),
                JsonValue::from(row.get("level").and_then(|v| v.as_i64()).unwrap_or(0)),
            );
            hit.insert(
                "ref_ids".into(),
                row.get("ref_ids")
                    .cloned()
                    .unwrap_or_else(|| JsonValue::Array(Vec::new())),
            );
            hit.insert(
                "merged_into".into(),
                row.get("merged_into").cloned().unwrap_or(JsonValue::Null),
            );
            out.push(hit);
        }
        Ok(out)
    }

    /// Multi-biz twin of [`Self::hybrid_chunk_query`] (bizIds): both recall
    /// legs run with the native set filter and fuse once, so the RRF ranking
    /// sees one corpus instead of per-biz partitions.
    async fn hybrid_chunk_query_multi_biz(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        exclude_archived: bool,
        biz_ids: &[&str],
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let Some(keyword_search) = &self.keyword_search else {
            return self
                .vector_storage
                .query_multi_biz(
                    query,
                    top_k,
                    None,
                    time_range_start,
                    time_range_end,
                    exclude_archived,
                    biz_ids,
                )
                .await;
        };

        let (vec_results, kw_results) = futures::try_join!(
            self.vector_storage.query_multi_biz(
                query,
                top_k,
                None,
                time_range_start,
                time_range_end,
                exclude_archived,
                biz_ids
            ),
            keyword_search.keyword_query_multi_biz(
                query,
                top_k,
                time_range_start,
                time_range_end,
                biz_ids
            )
        )?;
        let kw_results = self
            .hydrate_keyword_hits(kw_results, biz_ids, exclude_archived)
            .await?;
        Ok(rrf_fusion(vec_results, kw_results, self.rrf_k, 0.7, 0.3))
    }

    /// Recall + metadata filter — the first two steps of every chunk retrieval
    /// in the engine.
    ///
    /// Reranking is deliberately *not* folded in: three of the four paths that
    /// rerank do something between the filter and the rerank (an emptiness check
    /// that must return `FAIL_RESPONSE`/`None`, or `kg_query` appending
    /// entity-related chunks so the combined set is ranked together), and
    /// reordering those would change results. Callers follow up with
    /// [`Self::rerank_if_enabled`] at the point where they always did.
    ///
    /// The per-term entity recall inside `search_entities_and_relations` is not
    /// a caller: it splits `top_k` across search terms and applies neither the
    /// metadata filter nor reranking, so it is a different shape that happens to
    /// touch the same store.
    async fn retrieve_chunks(
        &self,
        query: &str,
        param: &QueryParam,
        recall: Recall,
        top_k_field: TopKField,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let top_k = top_k_field.of(param);
        // bizIds override (data endpoint only): the multi-biz set filter keeps
        // the recall a single global top_k over the requested corpus. An empty
        // `param.biz_ids` takes the legacy single-biz path unchanged.
        let multi_biz_ids: Vec<String> = if param.biz_ids.is_empty() {
            Vec::new()
        } else {
            param.effective_biz_ids()
        };
        let multi_biz_refs: Vec<&str> = multi_biz_ids.iter().map(String::as_str).collect();
        let recalled = match recall {
            Recall::Hybrid => {
                if multi_biz_refs.is_empty() {
                    self.hybrid_chunk_query(
                        query,
                        effective_top_k(param, top_k),
                        param.time_range_start,
                        param.time_range_end,
                        param.exclude_archived,
                        &param.biz_id,
                    )
                    .await?
                } else {
                    self.hybrid_chunk_query_multi_biz(
                        query,
                        effective_top_k(param, top_k),
                        param.time_range_start,
                        param.time_range_end,
                        param.exclude_archived,
                        &multi_biz_refs,
                    )
                    .await?
                }
            }
            Recall::Vector => {
                if multi_biz_refs.is_empty() {
                    self.vector_storage
                        .query(
                            query,
                            effective_top_k(param, top_k),
                            None,
                            param.time_range_start,
                            param.time_range_end,
                            param.exclude_archived,
                            &param.biz_id,
                        )
                        .await?
                } else {
                    self.vector_storage
                        .query_multi_biz(
                            query,
                            effective_top_k(param, top_k),
                            None,
                            param.time_range_start,
                            param.time_range_end,
                            param.exclude_archived,
                            &multi_biz_refs,
                        )
                        .await?
                }
            }
        };
        Ok(apply_metadata_filters(recalled, param, top_k))
    }

    /// Rerank when the caller asked for it, a service is configured, and there
    /// is more than one candidate worth ordering; otherwise return `chunks`
    /// untouched, so a caller can always assign the result back.
    ///
    /// `top_n` is `chunk_top_k` on every path, including the naive ones that
    /// size their recall from `top_k` — a pre-existing asymmetry, kept as is.
    async fn rerank_if_enabled(
        &self,
        query: &str,
        chunks: Vec<JsonMap>,
        param: &QueryParam,
    ) -> Vec<JsonMap> {
        if param.enable_rerank && self.rerank_service.is_some() && chunks.len() > 1 {
            return self.rerank_chunks(query, chunks, param.chunk_top_k).await;
        }
        chunks
    }

    /// Recall + rerank for the naive path — everything `naive_query` and
    /// `build_naive_query_prompt` share before they diverge. An empty result
    /// means "no context", which each caller signals its own way. Emptiness is
    /// checked after this call rather than before the rerank as it used to be,
    /// which is the same check: `rerank_if_enabled` needs more than one
    /// candidate, so it hands an empty list straight back.
    async fn naive_recall(
        &self,
        query: &str,
        param: &QueryParam,
        recall: Recall,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        let chunks = self
            .retrieve_chunks(query, param, recall, TopKField::TopK)
            .await?;
        Ok(self.rerank_if_enabled(query, chunks, param).await)
    }

    /// Fill the vars all four answer prompts share and render `template`. The
    /// retrieved-context var is `content_data` in the naive template and
    /// `context_data` in the graph one, so the key stays a parameter.
    fn render_answer_prompt(
        &self,
        template: &str,
        context_key: &str,
        context_data: String,
        param: &QueryParam,
    ) -> String {
        let mut vars = HashMap::new();
        vars.insert("response_type".to_string(), param.response_type.clone());
        vars.insert(
            "user_prompt".to_string(),
            param.user_prompt.clone().unwrap_or_default(),
        );
        vars.insert(context_key.to_string(), context_data);
        format_prompt(template, &vars)
    }

    /// Pre-retrieval query routing (Feature 6), shared by [`Self::query`] and
    /// [`Self::query_stream`].
    ///
    /// Each caller decides *whether* to route — they gate on different things,
    /// the synchronous one also skipping the debug hooks — and passes that in as
    /// `enabled`. A routing failure is logged and proceeds, so a broken router
    /// cannot cost the user their answer.
    async fn route_query(&self, query: &str, enabled: bool, logs: &RoutingLogs) -> Routing {
        if !enabled {
            return Routing::Proceed;
        }
        let Some(router) = &self.query_router else {
            tracing::warn!("Query routing requested but no QueryRouter is configured");
            return Routing::Proceed;
        };
        let RoutingLogs {
            scope,
            false_premise,
        } = logs;
        match router.route(query).await {
            Ok(route) => {
                use easyrag_core::model::verification::QueryRouteType;
                match route.route_type {
                    QueryRouteType::NoRetrieval => {
                        tracing::debug!("Query{scope} routed to no_retrieval");
                        return Routing::Answer(NO_RETRIEVAL_RESPONSE.to_string());
                    }
                    QueryRouteType::MultiHop => {
                        tracing::debug!(
                            "Query{scope} routed to multi_hop with {} sub-queries",
                            route.sub_queries.len()
                        );
                        // Execute sub-queries and merge context
                        // For now, fall through to normal query with expanded context
                    }
                    QueryRouteType::SingleHop => {
                        tracing::debug!("Query{scope} routed to single_hop");
                        // Normal flow
                    }
                }
                if route.has_false_premise {
                    tracing::info!("{false_premise}");
                }
            }
            Err(e) => {
                tracing::warn!("Query routing{scope} failed: {e}");
            }
        }
        Routing::Proceed
    }

    /// Post-retrieval hallucination gate (Feature 2). Returns `true` when the
    /// evidence is too weak to answer and the caller should return
    /// [`FAIL_RESPONSE`]; a scoring *failure* is logged and ignored, so a broken
    /// scorer cannot turn a good answer into a refusal.
    ///
    /// `log_scope` is the suffix each caller puts in its tracing messages — `""`
    /// on the naive path, `" (kg)"` on the graph one. Log text only, never control
    /// flow, which is what lets one implementation serve both.
    async fn gate_evidence(
        &self,
        query: &str,
        chunks: &[JsonMap],
        param: &QueryParam,
        log_scope: &str,
    ) -> bool {
        let use_scoring = param
            .enable_evidence_scoring
            .unwrap_or(self.evidence_scorer.is_some());
        if !use_scoring {
            return false;
        }
        let Some(scorer) = &self.evidence_scorer else {
            tracing::warn!("Evidence quality scoring requested but no EvidenceScorer is configured");
            return false;
        };
        match scorer.score(query, chunks).await {
            Ok(score) => {
                tracing::debug!("Evidence quality{log_scope}: {:.2} ({:?})", score.score, score.decision);
                match score.decision {
                    QualityDecision::Abstain => true,
                    QualityDecision::Refine => {
                        tracing::debug!("Evidence quality borderline, proceeding with caution");
                        false
                    }
                    // Good evidence, proceed normally.
                    QualityDecision::Generate => false,
                }
            }
            Err(e) => {
                tracing::warn!("Evidence scoring failed{log_scope}: {e}");
                false
            }
        }
    }

    /// Post-generation hallucination gate (Feature 1).
    ///
    /// Returns `true` when the answer rests on insufficiently supported claims and
    /// the caller should return [`FAIL_RESPONSE`]. `log_scope` means exactly what
    /// it does in [`Self::gate_evidence`]: log text only.
    async fn gate_claims(
        &self,
        response: &str,
        chunks: &[JsonMap],
        param: &QueryParam,
        log_scope: &str,
    ) -> bool {
        let use_verification = param
            .enable_verification
            .unwrap_or(self.claim_verifier.is_some());
        if !use_verification {
            return false;
        }
        let Some(verifier) = &self.claim_verifier else {
            tracing::warn!("Claim verification requested but no ClaimVerifier is configured");
            return false;
        };
        match verifier.verify_answer(response, chunks).await {
            Ok(result) => {
                tracing::debug!(
                    "Verification{log_scope}: {} claims, min_support={:.2}, abstain={}",
                    result.claims.len(),
                    result.min_support,
                    result.abstain
                );
                result.abstain
            }
            Err(e) => {
                tracing::warn!("Claim verification failed{log_scope}: {e}");
                false
            }
        }
    }

    /// Naive query - pure vector retrieval without knowledge graph.
    async fn naive_query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        // 1./2. Retrieve relevant chunks via hybrid search (vector + keyword)
        // and rerank them.
        let chunks = self.naive_recall(query, param, Recall::Hybrid).await?;

        if chunks.is_empty() {
            return Ok(FAIL_RESPONSE.to_string());
        }

        // [Guardrail] Evidence quality scoring (Feature 2) — post-retrieval
        if self.gate_evidence(query, &chunks, param, "").await {
            return Ok(FAIL_RESPONSE.to_string());
        }

        // 3. Build context from chunks.
        let context_data = self.build_naive_context(&chunks, param);

        // 4. Generate response.
        let prompt = self.render_answer_prompt(
            &NAIVE_RAG_RESPONSE,
            "content_data",
            context_data,
            param,
        );

        let response = self.llm_service.complete(&LlmRequest::new(prompt)).await?;

        // [Guardrail] Claim verification gate (Feature 1) — post-generation
        if self.gate_claims(&response, &chunks, param, "").await {
            return Ok(FAIL_RESPONSE.to_string());
        }

        Ok(response)
    }

    /// Chunks reachable from a set of KG entities via their `source_id` lists.
    ///
    /// Returns **only** the chunks not already in `already_recalled`, each
    /// tagged `"source": "entity_related"` so downstream prompt building can
    /// tell them apart from directly recalled ones. Extracted verbatim from
    /// `kg_query` step 4b so the `mix` data path (R9) and the KG answer path
    /// share one implementation instead of drifting copies.
    ///
    /// Dedup happens on two levels: source ids that match a recalled row's
    /// `id` are dropped *before* the KV query (recall rows carry the record
    /// key, and ingest keys KV/vector rows with the same
    /// `compute_mdhash_id(content, "chunk-")` value that entity `source_id`
    /// points at — one identifier, all stores), and the surviving rows are
    /// compared by content after the fetch, as this backfill always has.
    ///
    /// The two `take` caps are pre-existing budget guards, not tuning knobs:
    /// 10 entities keep the id fan-out bounded, and 20 chunk ids keep the KV
    /// `IN` query off the deep end on long documents.
    async fn entity_related_chunks(
        &self,
        entities: &[JsonMap],
        already_recalled: &[JsonMap],
        param: &QueryParam,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        // `parse_source_ids` owns the split rule (array, or legacy `<SEP>`/`,`
        // joined string); the old inline `split("<SEP>")` literal was a fourth
        // copy of that rule and a drift source.
        let recalled_ids: HashSet<&str> = already_recalled
            .iter()
            .filter_map(|c| c.get("id").and_then(|v| v.as_str()))
            .collect();

        let mut entity_chunk_ids: Vec<String> = Vec::new();
        let mut seen_chunk_ids: HashSet<String> = HashSet::new();
        for entity in entities.iter().take(10) {
            for id in parse_source_ids(entity.get("source_id")) {
                if !seen_chunk_ids.insert(id.clone()) || recalled_ids.contains(id.as_str()) {
                    continue;
                }
                entity_chunk_ids.push(id);
            }
        }
        if entity_chunk_ids.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<String> = entity_chunk_ids.into_iter().take(20).collect();
        // bizIds override: the chunk ids come from entities already scoped by
        // the multi-biz recall, so the KV backfill reads the same set.
        let multi_biz_ids: Vec<String> = if param.biz_ids.is_empty() {
            Vec::new()
        } else {
            param.effective_biz_ids()
        };
        let multi_biz_refs: Vec<&str> = multi_biz_ids.iter().map(String::as_str).collect();
        let kv_hits = if multi_biz_refs.is_empty() {
            self.kv_storage.get_by_ids(&ids, &param.biz_id).await?
        } else {
            self.kv_storage
                .get_by_ids_multi_biz(&ids, &multi_biz_refs)
                .await?
        };
        let mut extra: Vec<JsonMap> = Vec::new();
        for chunk_data in kv_hits {
            let Some(content) = chunk_data.get("content").and_then(|v| v.as_str()) else {
                continue;
            };
            let duplicate = already_recalled
                .iter()
                .chain(extra.iter())
                .any(|c| c.get("content").and_then(|v| v.as_str()) == Some(content));
            if duplicate {
                continue;
            }
            let mut with_source = chunk_data;
            with_source.insert("source".into(), JsonValue::String("entity_related".into()));
            extra.push(with_source);
        }
        Ok(extra)
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
            .search_entities_and_relations(&search_terms, param)
            .await?;

        // 4. Retrieve text chunks (hybrid search + entity-related chunks).
        let mut chunks = self
            .retrieve_chunks(query, param, Recall::Hybrid, TopKField::ChunkTopK)
            .await?;

        // 4b. Collect entity-related chunks via source_ids.
        let related = self
            .entity_related_chunks(&entities, &chunks, param)
            .await?;
        chunks.extend(related);

        // 5. Apply reranking if enabled and available — ranks the recall and
        // the entity-related chunks appended in 4b as one set.
        let chunks = self.rerank_if_enabled(query, chunks, param).await;

        // [Guardrail] Evidence quality scoring (Feature 2) — post-retrieval.
        // Deliberately ahead of the "nothing retrieved" check below, as it has
        // always been: the scorer runs (and logs) even when `chunks` came back empty.
        if self.gate_evidence(query, &chunks, param, " (kg)").await {
            return Ok(FAIL_RESPONSE.to_string());
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
        let prompt =
            self.render_answer_prompt(&RAG_RESPONSE, "context_data", context_data, param);

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
        if self.gate_claims(&response, &chunks, param, " (kg)").await {
            return Ok(FAIL_RESPONSE.to_string());
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
                let section_path = section_path_of(chunk).unwrap_or_default();
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
        format_prompt(&NAIVE_QUERY_CONTEXT, &vars)
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
        let system_prompt_tokens = self.tokenizer.count_tokens(&RAG_RESPONSE)
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
        format_prompt(&KG_QUERY_CONTEXT, &vars)
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
        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
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
        // bizIds override: distinct biz sets must never share a cache entry.
        if param.biz_ids.is_empty() {
            "null".to_string()
        } else {
            param.biz_ids.join(",")
        },
        param.enable_verification.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_citation_validation.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_evidence_scoring.map(|b| b.to_string()).as_deref().unwrap_or("null"),
        param.enable_query_routing.map(|b| b.to_string()).as_deref().unwrap_or("null"),
    );
    compute_args_hash([raw])
}

/// What the pure-data path does for a given [`QueryMode`] (R8/R9).
///
/// This struct is the single source of truth for the `/api/query/data`
/// behaviour matrix — every branch of `query_data` reads a field off it rather
/// than re-matching on the mode, so the matrix cannot disagree with itself.
/// Mirrored in `.docs/easyrag-data-endpoint-mode-matrix.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DataPlan {
    /// Chunk recall strategy.
    recall: Recall,
    /// Whether to run the KG entity/relationship pass at all.
    run_kg: bool,
    /// Whether to additionally fold in the chunks those entities point at.
    backfill_entity_chunks: bool,
}

/// Resolve a mode into the data path's behaviour (see [`DataPlan`]).
///
/// `naive`, `hybrid` and `mix` all recall by fusion, the way the synchronous
/// answer paths always have, so a caller asking for plain retrieval gets the
/// same chunks here as it does from `/api/query` and the two endpoints cannot
/// disagree about what the corpus holds. `local`/`global`/`bypass` stay
/// vector-recall + KG pass, which is what `query_data` did before R9; the R8
/// change is that `structure` skips the KG pass, which is the mode's whole
/// point.
///
/// A fused plan needs a keyword index. A deployment without one still recalls
/// by vector — `hybrid_chunk_query` falls back rather than failing — and says so
/// via [`degrade::KEYWORD_SEARCH_UNAVAILABLE`].
///
/// `bypass` keeping `run_kg: true` is a pre-existing oddity (it means "echo the
/// query" on the answer path but still retrieves on the data path). It is left
/// alone deliberately — fixing it here would smuggle an unrelated behaviour
/// change into R9.
fn data_plan(mode: QueryMode) -> DataPlan {
    match mode {
        QueryMode::Naive => DataPlan {
            recall: Recall::Hybrid,
            run_kg: false,
            backfill_entity_chunks: false,
        },
        QueryMode::Hybrid => DataPlan {
            recall: Recall::Hybrid,
            run_kg: true,
            backfill_entity_chunks: false,
        },
        QueryMode::Mix => DataPlan {
            recall: Recall::Hybrid,
            run_kg: true,
            backfill_entity_chunks: true,
        },
        QueryMode::Structure => DataPlan {
            recall: Recall::Vector,
            run_kg: false,
            backfill_entity_chunks: false,
        },
        QueryMode::Local | QueryMode::Global | QueryMode::Bypass => DataPlan {
            recall: Recall::Vector,
            run_kg: true,
            backfill_entity_chunks: false,
        },
    }
}

/// Whether this request asked for raw structure hits (R8).
///
/// `includeSections` is opt-in per request and defaults to
/// "`mode == structure`". That default is what lets the two shapes the
/// requirement asked for coexist: `mode=structure` retrieves structure, and
/// `mode=mix` can still carry sections when the caller says so — without every
/// scoped `mix` query silently paying the tree-search LLM bill.
fn wants_sections(param: &QueryParam) -> bool {
    param
        .include_sections
        .unwrap_or(matches!(param.mode, QueryMode::Structure))
}

/// Project one [`SectionHit`] onto the `sections[]` wire shape.
///
/// `score` is an **ordinal** score — `1.0 / (rank + 1)`, i.e. a value in
/// `(0, 1]` derived purely from position — and not a similarity. There is no
/// similarity to report: hits come from a persisted chunk join, and the
/// tree-search confidence that could stand in for one is absent on the zero-LLM
/// small-document path. `rank` is returned alongside it so a caller can order
/// without having to invert the formula, and must never be threshold-compared
/// against `chunks[].score`.
fn section_json(hit: &SectionHit, rank: usize) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("docId".into(), JsonValue::String(hit.doc_id.clone()));
    m.insert("filePath".into(), JsonValue::String(hit.file_path.clone()));
    m.insert(
        "sectionPath".into(),
        JsonValue::String(hit.section_path.clone()),
    );
    m.insert("title".into(), JsonValue::String(hit.title.clone()));
    m.insert("level".into(), JsonValue::from(hit.level as u64));
    m.insert(
        "summary".into(),
        match &hit.summary {
            Some(summary) => JsonValue::String(summary.clone()),
            None => JsonValue::Null,
        },
    );
    m.insert("content".into(), JsonValue::String(hit.content.clone()));
    m.insert(
        "contentTruncated".into(),
        JsonValue::Bool(hit.content_truncated),
    );
    m.insert(
        "chunkIds".into(),
        JsonValue::Array(
            hit.chunk_ids
                .iter()
                .map(|id| JsonValue::String(id.clone()))
                .collect(),
        ),
    );
    m.insert("rank".into(), JsonValue::from(rank as u64));
    m.insert(
        "score".into(),
        JsonValue::from(1.0f64 / (rank as f64 + 1.0)),
    );
    m
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

/// Truncate a list of items to fit within a token budget, where each item's token
/// cost is computed from a projection `key`. Items are kept in order until adding
/// the next one would exceed `max_tokens`; at least one item is always kept when
/// the input is non-empty. Generic over `T` with an `impl Fn` key so every call
/// site monomorphizes to zero-cost code (no dyn/Box indirection).
fn truncate_by<T: Clone>(
    items: &[T],
    max_tokens: usize,
    tokenizer: &Tokenizer,
    key: impl Fn(&T) -> String,
) -> Vec<T> {
    if items.is_empty() {
        return Vec::new();
    }
    let mut result: Vec<T> = Vec::new();
    let mut tokens_used = 0usize;
    for item in items {
        let item_tokens = tokenizer.count_tokens(&key(item));
        if tokens_used + item_tokens > max_tokens && !result.is_empty() {
            break;
        }
        result.push(item.clone());
        tokens_used += item_tokens;
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
    truncate_by(chunks, max_tokens, tokenizer, |chunk| {
        chunk
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    })
}

/// Truncate entities to fit within a token budget.
fn truncate_entities_to_budget(
    entities: &[JsonMap],
    max_tokens: usize,
    tokenizer: &Tokenizer,
) -> Vec<JsonMap> {
    truncate_by(entities, max_tokens, tokenizer, |entity| {
        let desc = entity
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let name = entity
            .get("entity_name")
            .or_else(|| entity.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        format!("{name}: {desc}")
    })
}

/// Truncate relations to fit within a token budget.
fn truncate_relations_to_budget(
    relations: &[JsonMap],
    max_tokens: usize,
    tokenizer: &Tokenizer,
) -> Vec<JsonMap> {
    truncate_by(relations, max_tokens, tokenizer, |relation| {
        let desc = relation
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let keywords = relation
            .get("keywords")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        format!("{desc} {keywords}")
    })
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
