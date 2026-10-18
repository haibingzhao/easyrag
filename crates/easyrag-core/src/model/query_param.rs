//! Query parameters and modes.

use crate::constants;
use serde::{Deserialize, Serialize};

/// Query mode enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryMode {
    Local,
    Global,
    Hybrid,
    Naive,
    #[default]
    Mix,
    Bypass,
    Structure,
}

impl QueryMode {
    pub fn value(&self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Global => "global",
            Self::Hybrid => "hybrid",
            Self::Naive => "naive",
            Self::Mix => "mix",
            Self::Bypass => "bypass",
            Self::Structure => "structure",
        }
    }

    pub fn from_value(value: &str) -> Result<Self, crate::CoreError> {
        match value.to_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "global" => Ok(Self::Global),
            "hybrid" => Ok(Self::Hybrid),
            "naive" => Ok(Self::Naive),
            "mix" => Ok(Self::Mix),
            "bypass" => Ok(Self::Bypass),
            "structure" => Ok(Self::Structure),
            other => Err(crate::CoreError::InvalidArgument(format!(
                "Unknown query mode: {other}"
            ))),
        }
    }
}

impl std::fmt::Display for QueryMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Configuration parameters for query execution.
/// All wire names are snake_case.
///
/// NOTE: the serde defaults here are *domain* defaults. The REST boundary
/// defaults differ (missing
/// `chunkTopK` → 5, missing/unknown `mode` → hybrid via `parseQueryMode`).
/// The server layer (P5) must apply those boundary defaults on request
/// binding instead of reusing `QueryParam::default`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryParam {
    #[serde(default)]
    pub mode: QueryMode,
    #[serde(default)]
    pub only_need_context: bool,
    #[serde(default)]
    pub only_need_prompt: bool,
    #[serde(default = "default_response_type")]
    pub response_type: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_chunk_top_k")]
    pub chunk_top_k: usize,
    #[serde(default = "default_max_entity_tokens")]
    pub max_entity_tokens: usize,
    #[serde(default = "default_max_relation_tokens")]
    pub max_relation_tokens: usize,
    #[serde(default = "default_max_total_tokens")]
    pub max_total_tokens: usize,
    #[serde(default)]
    pub hl_keywords: Vec<String>,
    #[serde(default)]
    pub ll_keywords: Vec<String>,
    #[serde(default)]
    pub conversation_history: Vec<std::collections::HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_prompt: Option<String>,
    #[serde(default = "default_true")]
    pub enable_rerank: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_range_start: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_range_end: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_id: Option<String>,
    #[serde(default)]
    pub use_agent_mode: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_filters: Option<std::collections::HashMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_scope: Option<String>,
    /// Return raw structure-index hits alongside the chunk/entity payload (R8).
    ///
    /// `None` means "follow the mode": on for `structure`, off everywhere else.
    /// The override exists so a caller can ask for sections without giving up
    /// hybrid recall (`mode=mix` + `includeSections=true`), and so a scoped
    /// `mix` query does not silently pay the tree-search LLM cost it never asked
    /// for. Only the pure-data endpoint reads this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_sections: Option<bool>,
    /// Business-line filter; normalized to `DEFAULT_BIZ_ID` at the REST
    /// boundary when omitted.
    #[serde(default = "default_biz_id")]
    pub biz_id: String,
    /// Multi-biz filter override for the pure-data endpoint. Non-empty means
    /// recall treats these biz lines as one corpus (single global top_k via
    /// native set filtering) and `biz_id` is ignored; empty keeps the
    /// single-biz contract byte-for-byte. Only `/api/query/data` reads this.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub biz_ids: Vec<String>,

    // Anti-hallucination guardrail overrides (per-request).
    // None = use config default.

    /// Enable claim verification gate for this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_verification: Option<bool>,
    /// Enable citation validity check for this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_citation_validation: Option<bool>,
    /// Enable CRAG-style evidence quality scoring for this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_evidence_scoring: Option<bool>,
    /// Enable query routing and decomposition for this query.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_query_routing: Option<bool>,
    /// R2: exclude archived documents from retrieval at DB level.
    #[serde(default)]
    pub exclude_archived: bool,
}

fn default_response_type() -> String {
    "Multiple Paragraphs".to_string()
}
fn default_top_k() -> usize {
    constants::DEFAULT_TOP_K
}
fn default_chunk_top_k() -> usize {
    constants::DEFAULT_CHUNK_TOP_K
}
fn default_max_entity_tokens() -> usize {
    constants::DEFAULT_MAX_ENTITY_TOKENS
}
fn default_max_relation_tokens() -> usize {
    constants::DEFAULT_MAX_RELATION_TOKENS
}
fn default_max_total_tokens() -> usize {
    constants::DEFAULT_MAX_TOTAL_TOKENS
}
fn default_true() -> bool {
    true
}
fn default_biz_id() -> String {
    constants::DEFAULT_BIZ_ID.to_string()
}

impl QueryParam {
    /// The biz-line set this query filters on: the multi-biz override when
    /// non-empty (deduplicated, order-preserving), otherwise the single
    /// normalized `biz_id`. Callers pass the result to the storage layer's
    /// multi-biz read variants — a single-element result is the legacy
    /// single-biz path.
    pub fn effective_biz_ids(&self) -> Vec<String> {
        if self.biz_ids.is_empty() {
            return vec![self.biz_id.clone()];
        }
        let mut seen = std::collections::HashSet::new();
        self.biz_ids
            .iter()
            .filter(|b| seen.insert(b.as_str()))
            .cloned()
            .collect()
    }
}

impl Default for QueryParam {
    fn default() -> Self {
        Self {
            mode: QueryMode::Mix,
            only_need_context: false,
            only_need_prompt: false,
            response_type: default_response_type(),
            stream: false,
            top_k: constants::DEFAULT_TOP_K,
            chunk_top_k: constants::DEFAULT_CHUNK_TOP_K,
            max_entity_tokens: constants::DEFAULT_MAX_ENTITY_TOKENS,
            max_relation_tokens: constants::DEFAULT_MAX_RELATION_TOKENS,
            max_total_tokens: constants::DEFAULT_MAX_TOTAL_TOKENS,
            hl_keywords: Vec::new(),
            ll_keywords: Vec::new(),
            conversation_history: Vec::new(),
            user_prompt: None,
            enable_rerank: true,
            time_range_start: None,
            time_range_end: None,
            doc_id: None,
            use_agent_mode: false,
            metadata_filters: None,
            structure_scope: None,
            include_sections: None,
            biz_id: default_biz_id(),
            biz_ids: Vec::new(),
            enable_verification: None,
            enable_citation_validation: None,
            enable_evidence_scoring: None,
            enable_query_routing: None,
            exclude_archived: false,
        }
    }
}

#[cfg(test)]
mod biz_ids_tests {
    use super::*;

    #[test]
    fn empty_biz_ids_falls_back_to_the_single_biz_id() {
        let param = QueryParam {
            biz_id: "default".to_string(),
            ..Default::default()
        };
        assert_eq!(vec!["default".to_string()], param.effective_biz_ids());
    }

    #[test]
    fn non_empty_biz_ids_override_and_deduplicate_in_order() {
        let param = QueryParam {
            biz_id: "ignored".to_string(),
            biz_ids: vec![
                "b2".to_string(),
                "b1".to_string(),
                "b2".to_string(),
                "b1".to_string(),
            ],
            ..Default::default()
        };
        // First-seen order wins; `biz_id` is not mixed back in.
        assert_eq!(
            vec!["b2".to_string(), "b1".to_string()],
            param.effective_biz_ids()
        );
    }
}
