//! CRAG-style evidence quality scoring (Feature 2).
//!
//! Scores retrieval quality before generation to decide whether to proceed,
//! refine the query, or abstain. Provides a zero-cost fast path using existing
//! rerank scores, and an optional LLM-based scoring path.

use easyrag_core::llm::service::{LlmRequest, LlmService, ResponseFormat};
use easyrag_core::model::verification::{EvidenceQualityScore, QualityDecision};
use easyrag_core::prompt::{format_prompt, EVIDENCE_QUALITY_PROMPT};
use easyrag_core::JsonMap;
#[cfg(test)]
use easyrag_core::JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

/// Evidence quality scorer with fast-path (rerank scores) and LLM path.
pub struct EvidenceScorer {
    llm_service: Arc<dyn LlmService>,
    /// Threshold above which evidence is considered good (default: 0.7).
    good_threshold: f32,
    /// Threshold below which evidence is considered bad (default: 0.4).
    bad_threshold: f32,
}

impl EvidenceScorer {
    pub fn new(llm_service: Arc<dyn LlmService>) -> Self {
        Self {
            llm_service,
            good_threshold: 0.7,
            bad_threshold: 0.4,
        }
    }

    pub fn with_thresholds(mut self, good: f32, bad: f32) -> Self {
        self.good_threshold = good;
        self.bad_threshold = bad;
        self
    }

    /// Fast path: score evidence quality using existing rerank scores.
    ///
    /// Extracts rerank scores from the chunk metadata (if available) and
    /// computes an aggregate quality score. Zero additional LLM calls.
    ///
    /// Only the `rerank_score` field (written back by `rerank_chunks`) is
    /// trusted: the retrieval `score` field has a different scale (RRF
    /// fusion scores are ~0.01, cosine similarity is uncalibrated) and
    /// would mis-trigger the thresholds. If no rerank scores are present,
    /// returns a neutral score of 0.5 (Refine decision), signaling the
    /// caller to use the LLM path.
    pub fn score_from_rerank(&self, chunks: &[JsonMap]) -> EvidenceQualityScore {
        let scores: Vec<f32> = chunks
            .iter()
            .filter_map(|chunk| {
                chunk
                    .get("rerank_score")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
            })
            .collect();

        if scores.is_empty() {
            // No rerank scores available — neutral score, suggest LLM path
            return EvidenceQualityScore {
                score: 0.5,
                decision: QualityDecision::Refine,
            };
        }

        // Use a weighted combination: top-3 average (precision signal) +
        // overall mean (recall signal). Top-3 gets 60% weight.
        let top_n = scores.len().min(3);
        let top_avg: f32 = scores[..top_n].iter().sum::<f32>() / top_n as f32;
        let overall_avg: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
        let combined = 0.6 * top_avg + 0.4 * overall_avg;

        EvidenceQualityScore::with_thresholds(combined, self.good_threshold, self.bad_threshold)
    }

    /// LLM path: score evidence quality using the language model.
    ///
    /// Sends the query and top context passages to the LLM for quality
    /// assessment. Adds 1 LLM call (~300-500ms).
    pub async fn score_with_llm(
        &self,
        query: &str,
        chunks: &[JsonMap],
    ) -> easyrag_core::Result<EvidenceQualityScore> {
        // Build context preview from top chunks
        let context = chunks
            .iter()
            .take(8)
            .filter_map(|chunk| {
                chunk
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.chars().take(200).collect::<String>())
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let mut vars = HashMap::new();
        vars.insert("query".to_string(), query.to_string());
        vars.insert("context".to_string(), context);
        let prompt = format_prompt(EVIDENCE_QUALITY_PROMPT, &vars);

        let request = LlmRequest::new(&prompt)
            .with_max_tokens(64)
            .with_response_format(ResponseFormat::JsonObject);

        let response = self.llm_service.complete(&request).await?;

        // Parse the JSON response for the score
        let score = parse_evidence_score(&response, self.good_threshold, self.bad_threshold);
        Ok(score)
    }

    /// Score evidence using the best available method.
    ///
    /// Prefers the fast path (rerank scores) if available, falls back to LLM.
    pub async fn score(
        &self,
        query: &str,
        chunks: &[JsonMap],
    ) -> easyrag_core::Result<EvidenceQualityScore> {
        let fast = self.score_from_rerank(chunks);
        // If fast path returned a definitive decision (not Refine/neutral), use it
        if fast.decision != QualityDecision::Refine {
            return Ok(fast);
        }
        // Fast path was neutral — try LLM path for better signal
        match self.score_with_llm(query, chunks).await {
            Ok(llm_score) => Ok(llm_score),
            Err(e) => {
                tracing::warn!("LLM evidence scoring failed, using fast path: {e}");
                Ok(fast)
            }
        }
    }
}

/// Regex for extracting a float from free-text LLM responses.
static SCORE_REGEX: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(r"(\d+\.\d+|\d+)").expect("valid regex"));

/// Parse the LLM response for an evidence quality score.
fn parse_evidence_score(response: &str, good: f32, bad: f32) -> EvidenceQualityScore {
    // Try to parse as JSON first
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(score) = json.get("score").and_then(|v| v.as_f64()) {
            // Clamp: the LLM may return out-of-range values (e.g. 2.5).
            let score = (score as f32).clamp(0.0, 1.0);
            return EvidenceQualityScore::with_thresholds(score, good, bad);
        }
    }

    // Fallback: try to extract a float from the response text
    if let Some(m) = SCORE_REGEX.find(response) {
        if let Ok(score) = m.as_str().parse::<f32>() {
            if score >= 0.0 && score <= 1.0 {
                return EvidenceQualityScore::with_thresholds(score, good, bad);
            }
        }
    }

    // Default: neutral score
    EvidenceQualityScore {
        score: 0.5,
        decision: QualityDecision::Refine,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_from_rerank_high_quality() {
        let scorer = EvidenceScorer::new(Arc::new(MockLlm));
        let chunks = vec![
            json_map_with_score(0.95),
            json_map_with_score(0.90),
            json_map_with_score(0.85),
        ];
        let result = scorer.score_from_rerank(&chunks);
        assert!(result.score > 0.7);
        assert_eq!(result.decision, QualityDecision::Generate);
    }

    #[test]
    fn score_from_rerank_low_quality() {
        let scorer = EvidenceScorer::new(Arc::new(MockLlm));
        let chunks = vec![
            json_map_with_score(0.2),
            json_map_with_score(0.1),
            json_map_with_score(0.3),
        ];
        let result = scorer.score_from_rerank(&chunks);
        assert!(result.score < 0.4);
        assert_eq!(result.decision, QualityDecision::Abstain);
    }

    #[test]
    fn score_from_rerank_no_scores() {
        let scorer = EvidenceScorer::new(Arc::new(MockLlm));
        let chunks = vec![JsonMap::new()];
        let result = scorer.score_from_rerank(&chunks);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.decision, QualityDecision::Refine);
    }

    #[test]
    fn score_from_rerank_ignores_retrieval_score() {
        // Raw retrieval scores (RRF/cosine) must not feed the thresholds.
        let scorer = EvidenceScorer::new(Arc::new(MockLlm));
        let mut map = JsonMap::new();
        map.insert("score".to_string(), JsonValue::from(0.01_f64));
        let result = scorer.score_from_rerank(&[map]);
        assert_eq!(result.score, 0.5);
        assert_eq!(result.decision, QualityDecision::Refine);
    }

    #[test]
    fn parse_evidence_score_json_out_of_range_is_clamped() {
        let result = parse_evidence_score(r#"{"score": 2.5}"#, 0.7, 0.4);
        assert!((result.score - 1.0).abs() < 0.01);
        assert_eq!(result.decision, QualityDecision::Generate);
    }

    #[test]
    fn parse_evidence_score_json() {
        let result = parse_evidence_score(r#"{"score": 0.85, "reason": "good"}"#, 0.7, 0.4);
        assert!((result.score - 0.85).abs() < 0.01);
        assert_eq!(result.decision, QualityDecision::Generate);
    }

    #[test]
    fn parse_evidence_score_fallback() {
        let result = parse_evidence_score("The score is 0.3", 0.7, 0.4);
        assert!((result.score - 0.3).abs() < 0.01);
        assert_eq!(result.decision, QualityDecision::Abstain);
    }

    // Test helpers
    struct MockLlm;

    #[async_trait::async_trait]
    impl LlmService for MockLlm {
        async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
            Ok(r#"{"score": 0.5, "reason": "mock"}"#.to_string())
        }

        fn complete_stream(&self, _request: LlmRequest) -> futures::stream::BoxStream<'static, String> {
            unimplemented!()
        }
    }

    fn json_map_with_score(score: f32) -> JsonMap {
        let mut map = JsonMap::new();
        map.insert("rerank_score".to_string(), JsonValue::from(score as f64));
        map.insert(
            "content".to_string(),
            JsonValue::String("test content".to_string()),
        );
        map
    }
}
