//! Entity and relationship extraction from text chunks using the LLM

use easyrag_core::concurrency::keyed_mutex::KeyedMutex;
use easyrag_core::constants::{
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, DEFAULT_SUMMARY_MAX_TOKENS, GRAPH_FIELD_SEP,
};
use easyrag_core::llm::service::{
    ChatMessage, LlmRequest, LlmService, MessageRole, ResponseFormat,
};
use easyrag_core::model::types::{
    EntityExtractionResult, ExtractedEntity, ExtractedRelationship,
};
use easyrag_core::prompt::{
    format_prompt, DEFAULT_COMPLETION_DELIMITER, DEFAULT_ENTITY_TYPES_GUIDANCE,
    DEFAULT_TUPLE_DELIMITER, ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT,
    ENTITY_CONTINUE_EXTRACTION_USER_PROMPT, ENTITY_EXTRACTION_EXAMPLES,
    ENTITY_EXTRACTION_JSON_EXAMPLES, ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_JSON_USER_PROMPT, ENTITY_EXTRACTION_SECTION_CONTEXT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_PROMPT,
    SUMMARIZE_ENTITY_DESCRIPTIONS,
};
use easyrag_core::storage::traits::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::json::{parse_json, to_json};
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Result of entity/relationship extraction from a text chunk.
pub type ExtractionResult = EntityExtractionResult;

/// EntityExtractor configuration.
#[derive(Clone, Debug)]
pub struct EntityExtractorConfig {
    pub language: String,
    pub entity_types_guidance: String,
    pub use_json_extraction: bool,
    pub max_total_records: usize,
    pub max_entity_records: usize,
    pub gleaning_rounds: usize,
    pub force_llm_summary_on_merge: usize,
    pub summary_max_tokens: usize,
    pub max_source_ids_per_entity: usize,
    pub max_extract_input_tokens: usize,
}

impl EntityExtractorConfig {
    /// Shared host-level overrides for every ingestion assembly site
    /// (`EasyRAG`, the global pipeline, and per-workspace pipelines):
    /// language and gleaning pass through directly; empty guidance means
    /// "unset" and keeps the built-in entity-type guidance instead of
    /// blanking the extraction prompt.
    pub fn apply_ingest_settings(
        &mut self,
        language: impl Into<String>,
        gleaning_rounds: usize,
        entity_types_guidance: &str,
    ) {
        self.language = language.into();
        self.gleaning_rounds = gleaning_rounds;
        if !entity_types_guidance.is_empty() {
            self.entity_types_guidance = entity_types_guidance.to_string();
        }
    }
}

impl Default for EntityExtractorConfig {
    fn default() -> Self {
        Self {
            language: "English".to_string(),
            entity_types_guidance: DEFAULT_ENTITY_TYPES_GUIDANCE.to_string(),
            use_json_extraction: true,
            max_total_records: 50,
            max_entity_records: 30,
            gleaning_rounds: 0,
            force_llm_summary_on_merge: DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
            summary_max_tokens: DEFAULT_SUMMARY_MAX_TOKENS,
            max_source_ids_per_entity: 30,
            max_extract_input_tokens: 20000,
        }
    }
}

/// Which wire format the extractor asks the model for.
///
/// Collapses the `if cfg.use_json_extraction` branches that used to be
/// scattered through [`EntityExtractor::extract`]: the examples block, the
/// three prompt templates, the per-mode prompt variables, `response_format`,
/// and the parser. Both arms stay reachable — `use_json_extraction` is a
/// public config field — even though every production assembly site leaves it
/// at the `true` default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractionMode {
    /// Structured JSON output, requested through `response_format`.
    Json,
    /// Delimiter-separated tuples, parsed positionally.
    Tuple,
}

impl ExtractionMode {
    fn of(cfg: &EntityExtractorConfig) -> Self {
        if cfg.use_json_extraction {
            Self::Json
        } else {
            Self::Tuple
        }
    }

    fn examples(&self) -> String {
        match self {
            Self::Json => ENTITY_EXTRACTION_JSON_EXAMPLES.join("\n"),
            Self::Tuple => ENTITY_EXTRACTION_EXAMPLES.join("\n"),
        }
    }

    fn system_prompt_template(&self) -> &'static str {
        match self {
            Self::Json => ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT.as_str(),
            Self::Tuple => ENTITY_EXTRACTION_SYSTEM_PROMPT.as_str(),
        }
    }

    fn user_prompt_template(&self) -> &'static str {
        match self {
            Self::Json => ENTITY_EXTRACTION_JSON_USER_PROMPT.as_str(),
            Self::Tuple => ENTITY_EXTRACTION_USER_PROMPT.as_str(),
        }
    }

    fn gleaning_template(&self) -> &'static str {
        match self {
            Self::Json => ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT,
            Self::Tuple => ENTITY_CONTINUE_EXTRACTION_USER_PROMPT,
        }
    }

    /// Only JSON mode constrains the response shape; tuple mode leaves it
    /// unset so the model emits free-form delimited lines.
    fn response_format(&self) -> Option<ResponseFormat> {
        match self {
            Self::Json => Some(ResponseFormat::JsonObject),
            Self::Tuple => None,
        }
    }

    fn parse(&self, response: &str) -> ExtractionResult {
        match self {
            Self::Json => EntityExtractor::parse_json_extraction(response),
            Self::Tuple => EntityExtractor::parse_tuple_extraction(
                response,
                DEFAULT_TUPLE_DELIMITER,
                DEFAULT_COMPLETION_DELIMITER,
            ),
        }
    }

    /// System-prompt variables. Both modes supply the same seven keys: the
    /// delimiters are inert in JSON mode (its template never mentions them)
    /// but must still be present, because `format_prompt` substitutes by
    /// iteration and a missing key would leave the placeholder behind.
    fn system_vars(&self, cfg: &EntityExtractorConfig) -> HashMap<String, String> {
        let mut vars = base_vars(cfg);
        vars.insert(
            "entity_types_guidance".to_string(),
            cfg.entity_types_guidance.clone(),
        );
        vars.insert("examples".to_string(), self.examples());
        vars.insert(
            "tuple_delimiter".to_string(),
            DEFAULT_TUPLE_DELIMITER.to_string(),
        );
        vars.insert(
            "completion_delimiter".to_string(),
            DEFAULT_COMPLETION_DELIMITER.to_string(),
        );
        vars
    }

    /// User-prompt variables. The per-mode extra is not cosmetic: the JSON
    /// user template carries `{entity_types_guidance}` (its system template
    /// does not), while the tuple user template inlines
    /// `{completion_delimiter}`.
    fn user_vars(
        &self,
        cfg: &EntityExtractorConfig,
        input_text: &str,
        heading_context_block: String,
    ) -> HashMap<String, String> {
        let mut vars = base_vars(cfg);
        vars.insert("input_text".to_string(), input_text.to_string());
        match self {
            Self::Json => vars.insert(
                "entity_types_guidance".to_string(),
                cfg.entity_types_guidance.clone(),
            ),
            Self::Tuple => vars.insert(
                "completion_delimiter".to_string(),
                DEFAULT_COMPLETION_DELIMITER.to_string(),
            ),
        };
        vars.insert("heading_context_block".to_string(), heading_context_block);
        vars
    }

    /// Gleaning-prompt variables. The JSON gleaning template mentions neither
    /// the guidance nor a delimiter, so JSON adds nothing here; only the tuple
    /// template inlines `{completion_delimiter}`.
    fn gleaning_vars(&self, cfg: &EntityExtractorConfig) -> HashMap<String, String> {
        let mut vars = base_vars(cfg);
        if let Self::Tuple = self {
            vars.insert(
                "completion_delimiter".to_string(),
                DEFAULT_COMPLETION_DELIMITER.to_string(),
            );
        }
        vars
    }
}

/// The prompt variables every extraction template shares verbatim. The
/// remaining keys are mode- or role-specific and stay with their callers.
fn base_vars(cfg: &EntityExtractorConfig) -> HashMap<String, String> {
    let mut vars = HashMap::new();
    vars.insert(
        "max_total_records".to_string(),
        cfg.max_total_records.to_string(),
    );
    vars.insert(
        "max_entity_records".to_string(),
        cfg.max_entity_records.to_string(),
    );
    vars.insert("language".to_string(), cfg.language.clone());
    vars
}

/// A graph payload, built once and shared by the four
/// `upsert_node`/`upsert_edge` sites in [`EntityExtractor::merge_to_graph`].
///
/// Insertion order is observable: `serde_json` compiles with `preserve_order`,
/// so `JsonMap` is an `IndexMap` and every backend sees the keys in the order
/// written here. `testdata/golden/extraction/*_payload.snap` pins it.
struct GraphUpsert {
    description: String,
    source_id: String,
}

impl GraphUpsert {
    /// Node payload, key order `[entity_type, description, source_id]`.
    fn node(self, entity_type: impl Into<String>) -> JsonMap {
        let mut data = JsonMap::new();
        data.insert("entity_type".into(), JsonValue::String(entity_type.into()));
        data.insert("description".into(), JsonValue::String(self.description));
        data.insert("source_id".into(), JsonValue::String(self.source_id));
        data
    }

    /// Edge payload, key order `[keywords, description, source_id, weight]`.
    ///
    /// `weight` is `f32` deliberately. The bump in `merge_to_graph` is
    /// computed in `f32`, and widening the result to `f64` would change the
    /// serialised number (`1.1` instead of `1.100000023841858`) — pinned by
    /// `existing_relation_weight_precision_payload.snap`.
    fn edge(self, keywords: impl Into<String>, weight: f32) -> JsonMap {
        let mut data = JsonMap::new();
        data.insert("keywords".into(), JsonValue::String(keywords.into()));
        data.insert("description".into(), JsonValue::String(self.description));
        data.insert("source_id".into(), JsonValue::String(self.source_id));
        data.insert("weight".into(), serde_json::json!(weight));
        data
    }
}

/// Entity and relationship extraction from text chunks using LLM.
pub struct EntityExtractor {
    llm_service: Arc<dyn LlmService>,
    graph_storage: Arc<dyn GraphStorage>,
    #[allow(dead_code)]
    vector_storage: Arc<dyn VectorStorage>,
    #[allow(dead_code)]
    kv_storage: Arc<dyn KVStorage>,
    keyed_mutex: KeyedMutex,
    config: EntityExtractorConfig,
}

impl EntityExtractor {
    pub fn new(
        llm_service: Arc<dyn LlmService>,
        graph_storage: Arc<dyn GraphStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        kv_storage: Arc<dyn KVStorage>,
        config: EntityExtractorConfig,
    ) -> Self {
        Self {
            llm_service,
            graph_storage,
            vector_storage,
            kv_storage,
            keyed_mutex: KeyedMutex::new(),
            config,
        }
    }

    pub fn with_keyed_mutex(mut self, keyed_mutex: KeyedMutex) -> Self {
        self.keyed_mutex = keyed_mutex;
        self
    }

    /// Extract entities and relationships from a text chunk.
    /// Supports multi-round gleaning to recover missed entities.
    /// `heading_path`: optional section path context
    /// (e.g. "Chapter 1 > Section 1.2").
    pub async fn extract(
        &self,
        chunk_content: &str,
        chunk_id: &str,
        heading_path: &str,
    ) -> easyrag_core::Result<ExtractionResult> {
        let cfg = &self.config;
        let mode = ExtractionMode::of(cfg);

        let system_prompt =
            format_prompt(mode.system_prompt_template(), &mode.system_vars(cfg));

        let heading_context_block = if !heading_path.is_empty() {
            let mut vars = HashMap::new();
            vars.insert("heading_path".to_string(), heading_path.to_string());
            format_prompt(ENTITY_EXTRACTION_SECTION_CONTEXT, &vars)
        } else {
            String::new()
        };

        let user_prompt = format_prompt(
            mode.user_prompt_template(),
            &mode.user_vars(cfg, chunk_content, heading_context_block),
        );

        // First round extraction.
        let (response, mut result) = self
            .run_round(mode, user_prompt.clone(), &system_prompt, None)
            .await?;

        // Gleaning: multi-round extraction to recover missed
        // entities/relationships.
        if cfg.gleaning_rounds > 0 {
            let mut history = vec![
                ChatMessage {
                    role: MessageRole::User,
                    content: user_prompt,
                },
                ChatMessage {
                    role: MessageRole::Assistant,
                    content: response.clone(),
                },
            ];

            // writes `for (round in 1 downTo gleaningRounds)`,
            // which is an empty descending range whenever
            // gleaningRounds > 1 (a latent bug that silently disables
            // gleaning). We implement the intended semantics: run
            // `gleaning_rounds` rounds, counting down.
            for round in (1..=cfg.gleaning_rounds).rev() {
                // Token safety guard: skip gleaning if context would
                // exceed limit (rough chars/4 estimate).
                let history_tokens: usize =
                    history.iter().map(|m| m.content.len() / 4).sum();
                let system_tokens = system_prompt.len() / 4;
                if history_tokens + system_tokens > cfg.max_extract_input_tokens {
                    tracing::warn!(
                        "Gleaning round {round} skipped: estimated tokens {} exceeds limit {}",
                        history_tokens + system_tokens,
                        cfg.max_extract_input_tokens
                    );
                    break;
                }

                let gleaning_prompt =
                    format_prompt(mode.gleaning_template(), &mode.gleaning_vars(cfg));

                let (gleaning_response, gleaning_result) = self
                    .run_round(mode, gleaning_prompt, &system_prompt, Some(history.clone()))
                    .await?;

                // Stop if no new entities/relationships found.
                if gleaning_result.entities.is_empty()
                    && gleaning_result.relationships.is_empty()
                {
                    tracing::debug!(
                        "Gleaning round {round} produced no new results, stopping"
                    );
                    break;
                }

                // Merge gleaning results: keep longer descriptions per
                // entity name.
                let merged_entities =
                    Self::merge_gleaning_entities(&result.entities, &gleaning_result.entities);
                let mut relationships = result.relationships;
                relationships.extend(gleaning_result.relationships.clone());
                result = ExtractionResult {
                    entities: merged_entities,
                    relationships,
                };

                history.push(ChatMessage {
                    role: MessageRole::Assistant,
                    content: gleaning_response,
                });
                tracing::debug!(
                    "Gleaning round {round} added {} entities, {} relationships",
                    gleaning_result.entities.len(),
                    gleaning_result.relationships.len()
                );
            }
        }

        tracing::info!(
            "Extracted {} entities and {} relationships from chunk {chunk_id} (gleaning={})",
            result.entities.len(),
            result.relationships.len(),
            cfg.gleaning_rounds
        );
        Ok(result)
    }

    /// One extraction round: build the request, constrain the response shape
    /// when the mode asks for it, call the model, then parse.
    ///
    /// The first round and every gleaning round are these same four steps and
    /// differ only in `history`, which gleaning carries and the first round
    /// leaves empty. The raw response is returned alongside the parsed result
    /// because gleaning appends it to the conversation history verbatim.
    async fn run_round(
        &self,
        mode: ExtractionMode,
        prompt: String,
        system_prompt: &str,
        history: Option<Vec<ChatMessage>>,
    ) -> easyrag_core::Result<(String, ExtractionResult)> {
        let mut request = LlmRequest::new(prompt).with_system_prompt(system_prompt);
        if let Some(history) = history {
            request.history = history;
        }
        request.response_format = mode.response_format();
        let response = self.llm_service.complete(&request).await?;
        let result = mode.parse(&response);
        Ok((response, result))
    }

    /// Merge extracted entities and relationships into the knowledge graph.
    /// Uses KeyedMutex for entity-level locking to prevent concurrent
    /// modifications. Entity type uses stability voting; source IDs are
    /// capped at `max_source_ids_per_entity` (FIFO).
    pub async fn merge_to_graph(
        &self,
        result: &ExtractionResult,
        chunk_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        for entity in &result.entities {
            let entity_name = entity.entity_name.clone();
            let entity = entity.clone();
            let chunk_id = chunk_id.to_string();
            self.keyed_mutex
                .with_lock(&entity_name, || async {
                    let existing = self.graph_storage.get_node(&entity_name, biz_id).await?;
                    if let Some(existing) = existing {
                        // Merge descriptions using GRAPH_FIELD_SEP.
                        let existing_desc = existing
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let merged_desc =
                            Self::merge_description(existing_desc, &entity.entity_description);

                        // Check if LLM summary is needed.
                        let final_desc = self
                            .maybe_summarize_descriptions(&merged_desc, &entity_name, "Entity")
                            .await;

                        // Entity type voting: prefer the more specific type.
                        let existing_type = existing
                            .get("entity_type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Other");
                        let voted_type =
                            Self::vote_entity_type(existing_type, &entity.entity_type);

                        let data = GraphUpsert {
                            description: final_desc,
                            source_id: merge_source_ids_str(
                                existing
                                    .get("source_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(""),
                                &chunk_id,
                                self.config.max_source_ids_per_entity,
                            ),
                        }
                        .node(voted_type);
                        self.graph_storage.upsert_node(&entity_name, data, biz_id).await?;
                    } else {
                        let data = GraphUpsert {
                            description: entity.entity_description.clone(),
                            source_id: chunk_id.clone(),
                        }
                        .node(entity.entity_type.clone());
                        self.graph_storage.upsert_node(&entity_name, data, biz_id).await?;
                    }
                    Ok::<(), easyrag_core::CoreError>(())
                })
                .await?;
        }

        for rel in &result.relationships {
            let edge_key = format!("{}<->{}", rel.source_entity, rel.target_entity);
            let rel = rel.clone();
            let chunk_id = chunk_id.to_string();
            self.keyed_mutex
                .with_lock(&edge_key, || async {
                    let existing = self
                        .graph_storage
                        .get_edge(&rel.source_entity, &rel.target_entity, biz_id)
                        .await?;
                    if let Some(existing) = existing {
                        let existing_desc = existing
                            .get("description")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let merged_desc = Self::merge_description(
                            existing_desc,
                            &rel.relationship_description,
                        );

                        // Check if LLM summary is needed.
                        let pair_name =
                            format!("{} <-> {}", rel.source_entity, rel.target_entity);
                        let final_desc = self
                            .maybe_summarize_descriptions(&merged_desc, &pair_name, "Relation")
                            .await;

                        let existing_weight = existing
                            .get("weight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0) as f32;

                        let data = GraphUpsert {
                            description: final_desc,
                            source_id: merge_source_ids_str(
                                existing
                                    .get("source_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(""),
                                &chunk_id,
                                self.config.max_source_ids_per_entity,
                            ),
                        }
                        .edge(rel.relationship_keywords.clone(), existing_weight + 1.0);
                        self.graph_storage
                            .upsert_edge(&rel.source_entity, &rel.target_entity, data, biz_id)
                            .await?;
                    } else {
                        let data = GraphUpsert {
                            description: rel.relationship_description.clone(),
                            source_id: chunk_id.clone(),
                        }
                        .edge(rel.relationship_keywords.clone(), 1.0_f32);
                        self.graph_storage
                            .upsert_edge(&rel.source_entity, &rel.target_entity, data, biz_id)
                            .await?;
                    }
                    Ok::<(), easyrag_core::CoreError>(())
                })
                .await?;
        }
        Ok(())
    }

    /// Merge a new description fragment into the existing description using
    /// GRAPH_FIELD_SEP, skipping duplicates (containment in either direction).
    fn merge_description(existing: &str, new_fragment: &str) -> String {
        if new_fragment.trim().is_empty() {
            return existing.to_string();
        }
        if existing.trim().is_empty() {
            return new_fragment.to_string();
        }
        let fragments: Vec<&str> = existing
            .split(GRAPH_FIELD_SEP)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        // Skip if this fragment is already contained.
        if fragments
            .iter()
            .any(|f| f.contains(new_fragment) || new_fragment.contains(*f))
        {
            return existing.to_string();
        }
        format!("{existing}{GRAPH_FIELD_SEP}{new_fragment}")
    }

    /// Entity type voting: when existing and new types differ, prefer the
    /// more specific one; "Other"/"Unknown"/"entity"/"" are generic.
    /// Otherwise keep existing (stability).
    fn vote_entity_type(existing_type: &str, new_type: &str) -> String {
        if existing_type.eq_ignore_ascii_case(new_type) {
            return existing_type.to_string();
        }
        let generic = ["other", "unknown", "entity", ""];
        let existing_lower = existing_type.to_lowercase();
        let new_lower = new_type.to_lowercase();
        let existing_generic = generic.contains(&existing_lower.as_str());
        let new_generic = generic.contains(&new_lower.as_str());
        if existing_generic && !new_generic {
            new_type.to_string()
        } else if new_generic {
            existing_type.to_string()
        } else {
            existing_type.to_string()
        }
    }

    /// Merge gleaning entities: for same-named entities, keep the one with
    /// the longer description.
    fn merge_gleaning_entities(
        existing: &[ExtractedEntity],
        gleaned: &[ExtractedEntity],
    ) -> Vec<ExtractedEntity> {
        let mut by_name: HashMap<String, ExtractedEntity> = HashMap::new();
        // Preserve first-seen order,
        // since downstream graph merge order is observable.
        let mut order: Vec<String> = Vec::new();
        for entity in existing {
            let key = entity.entity_name.to_lowercase();
            if !by_name.contains_key(&key) {
                order.push(key.clone());
            }
            by_name.insert(key, entity.clone());
        }
        for entity in gleaned {
            let key = entity.entity_name.to_lowercase();
            let replace = match by_name.get(&key) {
                None => true,
                Some(prev) => entity.entity_description.len() > prev.entity_description.len(),
            };
            if replace {
                if !by_name.contains_key(&key) {
                    order.push(key.clone());
                }
                by_name.insert(key, entity.clone());
            }
        }
        order
            .into_iter()
            .filter_map(|key| by_name.remove(&key))
            .collect()
    }

    /// If description fragments exceed `force_llm_summary_on_merge`, use
    /// iterative map-reduce to generate a cohesive summary. Falls back to
    /// the raw merged description on LLM failure.
    async fn maybe_summarize_descriptions(
        &self,
        merged_desc: &str,
        name: &str,
        description_type: &str,
    ) -> String {
        let cfg = &self.config;
        let fragments: Vec<&str> = merged_desc
            .split(GRAPH_FIELD_SEP)
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        if fragments.len() < cfg.force_llm_summary_on_merge {
            return merged_desc.to_string();
        }

        tracing::info!(
            "Triggering LLM summary for {description_type} '{name}' ({} fragments)",
            fragments.len()
        );

        // Estimate total tokens; if within budget, single-pass summary.
        let total_chars: usize = fragments.iter().map(|f| f.len()).sum();
        let estimated_tokens = total_chars / 4;
        // Guard against a degenerate config (0 budget) that would panic on
        // integer division below; try/catch swallows the error
        // instead, so clamping keeps the graceful-degradation contract.
        let summary_budget = cfg.summary_max_tokens.max(1);

        let outcome = async {
            if estimated_tokens <= summary_budget {
                self.summarize_fragment_batch(&fragments, name, description_type)
                    .await
            } else {
                // Map-reduce: split into groups, summarize each, then merge.
                let group_size = (fragments.len()
                    / ((estimated_tokens / summary_budget) + 1))
                    .max(1);
                let groups: Vec<&[&str]> = fragments.chunks(group_size).collect();

                // Map phase: summarize each group.
                let mut group_summaries = Vec::new();
                for group in groups {
                    let summary = self
                        .summarize_fragment_batch(group, name, description_type)
                        .await?;
                    if !summary.trim().is_empty() {
                        group_summaries.push(summary);
                    }
                }

                // Reduce phase: if multiple summaries, merge them.
                if group_summaries.len() <= 1 {
                    Ok(group_summaries
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| merged_desc.to_string()))
                } else {
                    let description_list = group_summaries
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            format!("{{\"index\": {}, \"description\": {}}}", i + 1, to_json(s))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let mut vars = HashMap::new();
                    vars.insert("description_type".to_string(), description_type.to_string());
                    vars.insert("description_name".to_string(), name.to_string());
                    vars.insert("description_list".to_string(), description_list);
                    vars.insert(
                        "summary_length".to_string(),
                        cfg.summary_max_tokens.to_string(),
                    );
                    vars.insert("language".to_string(), cfg.language.clone());
                    let reduce_prompt = format_prompt(SUMMARIZE_ENTITY_DESCRIPTIONS, &vars);
                    let final_summary = self
                        .llm_service
                        .complete(&LlmRequest::new(reduce_prompt))
                        .await?;
                    if !final_summary.trim().is_empty() {
                        Ok(final_summary.trim().to_string())
                    } else {
                        Ok(group_summaries.join(" "))
                    }
                }
            }
        }
        .await;

        match outcome {
            Ok(summary) => summary,
            Err(e) => {
                tracing::warn!(
                    "LLM summary failed for {description_type} '{name}', keeping raw descriptions: {e}"
                );
                merged_desc.to_string()
            }
        }
    }

    /// Summarize a single batch of description fragments via LLM.
    async fn summarize_fragment_batch(
        &self,
        fragments: &[&str],
        name: &str,
        description_type: &str,
    ) -> easyrag_core::Result<String> {
        let cfg = &self.config;
        let description_list = fragments
            .iter()
            .enumerate()
            .map(|(i, desc)| format!("{{\"index\": {}, \"description\": {}}}", i + 1, to_json(desc)))
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = HashMap::new();
        vars.insert("description_type".to_string(), description_type.to_string());
        vars.insert("description_name".to_string(), name.to_string());
        vars.insert("description_list".to_string(), description_list);
        vars.insert(
            "summary_length".to_string(),
            cfg.summary_max_tokens.to_string(),
        );
        vars.insert("language".to_string(), cfg.language.clone());
        let prompt = format_prompt(SUMMARIZE_ENTITY_DESCRIPTIONS, &vars);

        let response = self.llm_service.complete(&LlmRequest::new(prompt)).await?;
        Ok(response.trim().to_string())
    }

    fn parse_json_extraction(response: &str) -> ExtractionResult {
        let json = match parse_json(response) {
            Some(json) => json,
            None => return ExtractionResult::default(),
        };

        let entities = json
            .get("entities")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|node| {
                        let name = node.get("name")?.as_str()?.trim().to_string();
                        if name.is_empty() {
                            return None;
                        }
                        // falls back to "Other" only when the field
                        // is missing/null; an explicit empty string is kept
                        // as-is (it is treated as generic by type voting).
                        let type_ = node
                            .get("type")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .unwrap_or_else(|| "Other".to_string());
                        let description = node
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        Some(ExtractedEntity {
                            entity_name: name,
                            entity_type: type_,
                            entity_description: description,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let relationships = json
            .get("relationships")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|node| {
                        let source = node.get("source")?.as_str()?.trim().to_string();
                        let target = node.get("target")?.as_str()?.trim().to_string();
                        if source.is_empty() || target.is_empty() {
                            return None;
                        }
                        let keywords = node
                            .get("keywords")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        let description = node
                            .get("description")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim().to_string())
                            .unwrap_or_default();
                        Some(ExtractedRelationship {
                            source_entity: source,
                            target_entity: target,
                            relationship_keywords: keywords,
                            relationship_description: description,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        ExtractionResult {
            entities,
            relationships,
        }
    }

    fn parse_tuple_extraction(
        response: &str,
        tuple_delimiter: &str,
        completion_delimiter: &str,
    ) -> ExtractionResult {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        let lines: Vec<&str> = response
            .split('\n')
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && *l != completion_delimiter)
            .collect();

        for line in lines {
            let parts: Vec<&str> = line.split(tuple_delimiter).map(|p| p.trim()).collect();
            if parts.len() >= 4 && parts[0].eq_ignore_ascii_case("entity") {
                entities.push(ExtractedEntity {
                    entity_name: parts[1].to_string(),
                    entity_type: parts[2].to_string(),
                    entity_description: parts[3].to_string(),
                });
            } else if parts.len() >= 5 && parts[0].eq_ignore_ascii_case("relation") {
                relationships.push(ExtractedRelationship {
                    source_entity: parts[1].to_string(),
                    target_entity: parts[2].to_string(),
                    relationship_keywords: parts[3].to_string(),
                    relationship_description: parts[4].to_string(),
                });
            }
        }

        ExtractionResult {
            entities,
            relationships,
        }
    }
}

/// Merge source IDs, deduplicating and respecting the FIFO limit
/// (oldest IDs dropped when exceeding `max_source_ids`).
fn merge_source_ids_str(existing: &str, new_id: &str, max_source_ids: usize) -> String {
    if existing.trim().is_empty() {
        return new_id.to_string();
    }
    let mut ids: Vec<&str> = existing
        .split(GRAPH_FIELD_SEP)
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if !ids.contains(&new_id) {
        ids.push(new_id);
    }
    // FIFO: keep only the most recent `max_source_ids` entries.
    let trimmed = if ids.len() > max_source_ids {
        ids.split_off(ids.len() - max_source_ids)
    } else {
        ids
    };
    trimmed.join(GRAPH_FIELD_SEP)
}
