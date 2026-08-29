//! Query routing and decomposition (Feature 6).
//!
//! Classifies queries into no_retrieval / single_hop / multi_hop, decomposes
//! multi-hop queries into sub-questions, and detects false premises.
//! Includes a fast-path heuristic for trivial queries.

use easyrag_core::llm::service::{LlmRequest, LlmService, ResponseFormat};
use easyrag_core::model::verification::{QueryRoute, QueryRouteType};
use easyrag_core::prompt::{format_prompt, QUERY_ROUTING_PROMPT};
use std::collections::HashMap;
use std::sync::Arc;

/// Query router that classifies and optionally decomposes queries.
pub struct QueryRouter {
    llm_service: Arc<dyn LlmService>,
}

impl QueryRouter {
    pub fn new(llm_service: Arc<dyn LlmService>) -> Self {
        Self { llm_service }
    }

    /// Classify a query and optionally decompose it into sub-questions.
    ///
    /// Fast path: very short queries without question words are classified as
    /// `NoRetrieval` without an LLM call.
    pub async fn route(&self, query: &str) -> easyrag_core::Result<QueryRoute> {
        // Fast-path heuristic: skip LLM for trivial non-queries
        if let Some(fast) = Self::fast_path_classify(query) {
            return Ok(fast);
        }

        let mut vars = HashMap::new();
        vars.insert("query".to_string(), query.to_string());
        let prompt = format_prompt(QUERY_ROUTING_PROMPT, &vars);

        let request = LlmRequest::new(&prompt)
            .with_max_tokens(128)
            .with_response_format(ResponseFormat::JsonObject);

        let response = self.llm_service.complete(&request).await?;
        parse_route_response(&response)
    }

    /// Fast-path heuristic classification without LLM.
    /// Returns `Some` for clearly non-retrieval queries, `None` otherwise.
    ///
    /// Only an explicit greeting/acknowledgment whitelist is matched. A
    /// previous "short query without English question word" heuristic was
    /// removed: `str::len` counts bytes (non-ASCII queries shrink the
    /// budget) and keyword-style queries such as "deployment guide" or
    /// Chinese queries like "量子计算原理" were misrouted to NoRetrieval.
    fn fast_path_classify(query: &str) -> Option<QueryRoute> {
        let trimmed = query.trim();

        // Very short greetings / non-queries
        let lower = trimmed.to_lowercase();
        if matches!(
            lower.as_str(),
            "hi" | "hello" | "hey" | "thanks" | "thank you" | "bye" | "goodbye" | "ok" | "yes" | "no"
        ) {
            return Some(QueryRoute {
                route_type: QueryRouteType::NoRetrieval,
                sub_queries: Vec::new(),
                has_false_premise: false,
            });
        }

        None // Need LLM classification
    }
}

/// Parse the LLM JSON response into a QueryRoute.
fn parse_route_response(response: &str) -> easyrag_core::Result<QueryRoute> {
    // Try to extract JSON from the response (may be wrapped in markdown)
    let json_str = extract_json(response);

    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap_or_else(|_| {
        tracing::warn!("Failed to parse query routing response as JSON: {response}");
        serde_json::json!({"query_type": "single_hop", "sub_queries": [], "has_false_premise": false})
    });

    let route_type = json
        .get("query_type")
        .and_then(|v| v.as_str())
        .map(|s| match s {
            "no_retrieval" => QueryRouteType::NoRetrieval,
            "multi_hop" => QueryRouteType::MultiHop,
            _ => QueryRouteType::SingleHop,
        })
        .unwrap_or(QueryRouteType::SingleHop);

    let sub_queries = json
        .get("sub_queries")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let has_false_premise = json
        .get("has_false_premise")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(QueryRoute {
        route_type,
        sub_queries,
        has_false_premise,
    })
}

/// Extract JSON from a response that may be wrapped in markdown code blocks.
fn extract_json(response: &str) -> String {
    let trimmed = response.trim();
    // Handle ```json ... ``` wrapping
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_path_greeting() {
        let result = QueryRouter::fast_path_classify("hello");
        assert!(result.is_some());
        assert_eq!(result.unwrap().route_type, QueryRouteType::NoRetrieval);
    }

    #[test]
    fn fast_path_question() {
        let result = QueryRouter::fast_path_classify("What is the capital of France?");
        assert!(result.is_none()); // Needs LLM
    }

    #[test]
    fn fast_path_keyword_style_query_needs_llm() {
        // Keyword-style and non-English short queries must not be
        // misrouted to NoRetrieval.
        assert!(QueryRouter::fast_path_classify("deployment guide").is_none());
        assert!(QueryRouter::fast_path_classify("量子计算原理").is_none());
    }

    #[test]
    fn parse_route_single_hop() {
        let response = r#"{"query_type": "single_hop", "sub_queries": [], "has_false_premise": false}"#;
        let route = parse_route_response(response).unwrap();
        assert_eq!(route.route_type, QueryRouteType::SingleHop);
        assert!(route.sub_queries.is_empty());
        assert!(!route.has_false_premise);
    }

    #[test]
    fn parse_route_multi_hop() {
        let response = r#"{"query_type": "multi_hop", "sub_queries": ["What is X?", "What is Y?"], "has_false_premise": false}"#;
        let route = parse_route_response(response).unwrap();
        assert_eq!(route.route_type, QueryRouteType::MultiHop);
        assert_eq!(route.sub_queries.len(), 2);
    }

    #[test]
    fn parse_route_with_markdown() {
        let response = "```json\n{\"query_type\": \"no_retrieval\", \"sub_queries\": [], \"has_false_premise\": false}\n```";
        let route = parse_route_response(response).unwrap();
        assert_eq!(route.route_type, QueryRouteType::NoRetrieval);
    }

    #[test]
    fn parse_route_invalid_json() {
        let response = "I'm not sure what you mean";
        let route = parse_route_response(response).unwrap();
        // Should default to single_hop
        assert_eq!(route.route_type, QueryRouteType::SingleHop);
    }
}
