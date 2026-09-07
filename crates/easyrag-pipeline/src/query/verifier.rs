//! Claim verification gate (Feature 1).
//!
//! Splits generated answers into atomic claims, verifies each against the
//! cited context, and decides whether to abstain. Uses a two-step process:
//! 1. Decompose the answer into atomic claims (1 LLM call)
//! 2. Verify claims against context (batch LLM call or embedding fast-path)

use easyrag_core::llm::service::{LlmRequest, LlmService, ResponseFormat};
use easyrag_core::model::verification::{ClaimVerification, VerificationResult};
use easyrag_core::prompt::{format_prompt, CLAIM_DECOMPOSITION_PROMPT, CLAIM_VERIFICATION_PROMPT};
use easyrag_core::util::json::parse_json;
use easyrag_core::JsonMap;
use std::collections::HashMap;
use std::sync::Arc;

/// Claim verification gate that checks answer faithfulness.
pub struct ClaimVerifier {
    llm_service: Arc<dyn LlmService>,
    /// Minimum support threshold. Claims below this are unsupported.
    abstain_threshold: f32,
}

impl ClaimVerifier {
    pub fn new(llm_service: Arc<dyn LlmService>) -> Self {
        Self {
            llm_service,
            abstain_threshold: 0.3,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.abstain_threshold = threshold;
        self
    }

    /// Full verification pipeline: decompose claims, verify, decide.
    ///
    /// Returns a `VerificationResult` indicating whether the answer should
    /// be accepted or the system should abstain.
    pub async fn verify_answer(
        &self,
        answer: &str,
        context_chunks: &[JsonMap],
    ) -> easyrag_core::Result<VerificationResult> {
        // Step 1: Decompose answer into atomic claims
        let claims = self.decompose_claims(answer).await?;

        if claims.is_empty() {
            // No extractable claims — accept the answer as-is
            return Ok(VerificationResult {
                claims: Vec::new(),
                all_verified: true,
                min_support: 1.0,
                abstain: false,
            });
        }

        // Step 2: Verify claims against context
        let verdicts = self.verify_claims(&claims, context_chunks).await?;

        // Fail-closed: when claims were extracted but the verification LLM
        // returned nothing parseable, abstain instead of accepting the
        // answer with an implicit max-confidence (empty-fold) verdict.
        if verdicts.is_empty() {
            tracing::warn!(
                "Claim verification returned no verdicts for {} claims; abstaining",
                claims.len()
            );
            return Ok(VerificationResult {
                claims: Vec::new(),
                all_verified: false,
                min_support: 0.0,
                abstain: true,
            });
        }

        // Step 3: Aggregate — use weakest-claim rule
        let min_support = verdicts
            .iter()
            .map(|v| v.confidence)
            .fold(1.0f32, f32::min);
        let all_verified = verdicts.iter().all(|v| v.supported);
        let abstain = min_support < self.abstain_threshold;

        Ok(VerificationResult {
            claims: verdicts,
            all_verified,
            min_support,
            abstain,
        })
    }

    /// Step 1: Decompose an answer into atomic, independently verifiable claims.
    pub async fn decompose_claims(&self, answer: &str) -> easyrag_core::Result<Vec<String>> {
        let mut vars = HashMap::new();
        vars.insert("answer".to_string(), answer.to_string());
        let prompt = format_prompt(CLAIM_DECOMPOSITION_PROMPT, &vars);

        let request = LlmRequest::new(&prompt)
            .with_max_tokens(300)
            .with_response_format(ResponseFormat::JsonObject);

        let response = self.llm_service.complete(&request).await?;
        parse_claims_response(&response)
    }

    /// Step 2: Verify a list of claims against the provided context.
    ///
    /// Uses a single batch LLM call to verify all claims at once.
    pub async fn verify_claims(
        &self,
        claims: &[String],
        context_chunks: &[JsonMap],
    ) -> easyrag_core::Result<Vec<ClaimVerification>> {
        if claims.is_empty() {
            return Ok(Vec::new());
        }

        // Build context string from chunks
        let context = context_chunks
            .iter()
            .filter_map(|chunk| {
                chunk
                    .get("content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.chars().take(1000).collect::<String>())
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // Format claims as numbered list
        let claims_text = claims
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{}. {}", i + 1, c))
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = HashMap::new();
        vars.insert("context".to_string(), context);
        vars.insert("claims".to_string(), claims_text);
        let prompt = format_prompt(CLAIM_VERIFICATION_PROMPT, &vars);

        let request = LlmRequest::new(&prompt)
            .with_max_tokens(512)
            .with_response_format(ResponseFormat::JsonObject);

        let response = self.llm_service.complete(&request).await?;
        parse_verification_response(&response)
    }
}

/// Parse the LLM response for claim decomposition.
fn parse_claims_response(response: &str) -> easyrag_core::Result<Vec<String>> {
    // Shared multi-strategy parser (lenient JSON, thinking-tag/markdown fence
    // stripping, quote repair, brace-matched extraction).
    let json: serde_json::Value = match parse_json(response) {
        Some(v) => v,
        None => {
            tracing::warn!("Failed to parse claim decomposition response: {response}");
            return Ok(Vec::new());
        }
    };

    let claims = json
        .get("claims")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.trim().to_string()))
                .filter(|s| s.len() > 3)
                .collect()
        })
        .unwrap_or_default();

    Ok(claims)
}

/// Parse the LLM response for claim verification.
fn parse_verification_response(response: &str) -> easyrag_core::Result<Vec<ClaimVerification>> {
    let json: serde_json::Value = match parse_json(response) {
        Some(v) => v,
        None => {
            tracing::warn!("Failed to parse verification response: {response}");
            return Ok(Vec::new());
        }
    };

    let verdicts = json
        .get("verdicts")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let claim = v.get("claim")?.as_str()?.to_string();
                    let supported = v.get("supported")?.as_bool()?;
                    let confidence = v
                        .get("confidence")
                        .and_then(|c| c.as_f64())
                        .unwrap_or(if supported { 0.8 } else { 0.2 })
                        as f32;
                    Some(ClaimVerification {
                        claim,
                        supported,
                        confidence,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(verdicts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_claims_basic() {
        let response = r#"{"claims": ["Paris is in France", "The Louvre is in Paris"]}"#;
        let claims = parse_claims_response(response).unwrap();
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0], "Paris is in France");
    }

    #[test]
    fn parse_claims_empty() {
        let response = r#"{"claims": []}"#;
        let claims = parse_claims_response(response).unwrap();
        assert!(claims.is_empty());
    }

    #[test]
    fn parse_claims_invalid_json() {
        let response = "I cannot decompose this";
        let claims = parse_claims_response(response).unwrap();
        assert!(claims.is_empty());
    }

    #[test]
    fn parse_verdicts_basic() {
        let response = r#"{
            "verdicts": [
                {"claim": "Paris is in France", "supported": true, "confidence": 0.95},
                {"claim": "Paris is in Germany", "supported": false, "confidence": 0.1}
            ]
        }"#;
        let verdicts = parse_verification_response(response).unwrap();
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts[0].supported);
        assert!(!verdicts[1].supported);
    }

    #[test]
    fn parse_claims_from_markdown_fence() {
        let response = "Here is the result:\n```json\n{\"claims\": [\"Paris is in France\"]}\n```";
        let claims = parse_claims_response(response).unwrap();
        assert_eq!(claims, vec!["Paris is in France".to_string()]);
    }

    #[test]
    fn parse_claims_tolerates_trailing_brace_prose() {
        // BC-13 (A-11): the removed local extractor sliced from the first `{`
        // to the *last* `}` of the whole response, so prose containing braces
        // after the payload yielded invalid JSON and every claim was silently
        // dropped. The shared parser strips the fence before parsing.
        let response =
            "```json\n{\"claims\": [\"Paris is in France\"]}\n```\nDone {see the fence above}.";
        let claims = parse_claims_response(response).unwrap();
        assert_eq!(claims, vec!["Paris is in France".to_string()]);
    }

    #[test]
    fn parse_verdicts_from_markdown_fence() {
        let response = "```json\n{\"verdicts\": [{\"claim\": \"Paris is in France\", \"supported\": true, \"confidence\": 0.9}]}\n```";
        let verdicts = parse_verification_response(response).unwrap();
        assert_eq!(verdicts.len(), 1);
        assert!(verdicts[0].supported);
    }

    #[test]
    fn legacy_brace_slicing_is_invalid_where_shared_parser_succeeds() {
        // Red-first proof for BC-13: replays the removed extractor's algorithm
        // (first `{` ..= last `}`) on the fixture above. That slice is not
        // valid JSON, so the old code silently returned no claims — exactly the
        // A-11 defect. The shared parser recovers the payload.
        let response =
            "```json\n{\"claims\": [\"Paris is in France\"]}\n```\nDone {see the fence above}.";
        let trimmed = response.trim();
        let legacy = &trimmed[trimmed.find('{').unwrap()..=trimmed.rfind('}').unwrap()];
        assert!(
            serde_json::from_str::<serde_json::Value>(legacy).is_err(),
            "legacy slice should not parse: {legacy}"
        );
        assert!(parse_json(response).is_some());
        assert_eq!(
            parse_claims_response(response).unwrap(),
            vec!["Paris is in France".to_string()]
        );
    }
}
