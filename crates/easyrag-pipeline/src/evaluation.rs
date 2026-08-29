//! Evaluation framework (Feature 7).
//!
//! Provides golden set evaluation, risk-coverage analysis, and reporting
//! for measuring the effectiveness of anti-hallucination guardrails.

use easyrag_core::model::evaluation::{
    ConfusionMatrix, EvalReport, GoldenItem, GoldenItemResult, RiskCoveragePoint,
};
use easyrag_core::model::QueryParam;
use easyrag_core::prompt::FAIL_RESPONSE;
use std::sync::Arc;

/// Trait for running queries during evaluation.
///
/// This allows the evaluator to work with any query backend (EasyRAG facade,
/// QueryEngine, or a mock for testing).
#[async_trait::async_trait]
pub trait Queryable: Send + Sync {
    async fn eval_query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String>;
}

/// Evaluator that runs a golden set of queries and computes metrics.
pub struct GoldenSetEvaluator {
    queryable: Arc<dyn Queryable>,
    param: QueryParam,
}

impl GoldenSetEvaluator {
    pub fn new(queryable: Arc<dyn Queryable>, param: QueryParam) -> Self {
        Self { queryable, param }
    }

    /// Evaluate a golden set and produce a report.
    ///
    /// For each item:
    /// - If `expected_answer` is `None`, the system should abstain.
    /// - If the response matches `FAIL_RESPONSE`, the system abstained.
    /// - Classification: tp (should answer + answered), fp (should abstain + answered),
    ///   tn (should abstain + abstained), fn (should answer + abstained).
    pub async fn evaluate(&self, golden_set: &[GoldenItem]) -> easyrag_core::Result<EvalReport> {
        let mut matrix = ConfusionMatrix::default();
        let mut item_results = Vec::with_capacity(golden_set.len());

        for item in golden_set {
            let response = match self.queryable.eval_query(&item.query, &self.param).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Query failed for '{}': {}", item.query, e);
                    FAIL_RESPONSE.to_string()
                }
            };

            let abstained = is_abstain_response(&response);
            let should_abstain = item.expected_answer.is_none();

            let (classification, correct) = match (should_abstain, abstained) {
                (false, false) => {
                    // Should answer, did answer — check if roughly correct
                    let correct = is_roughly_correct(&response, item.expected_answer.as_deref());
                    matrix.true_positive += 1;
                    ("tp".to_string(), correct)
                }
                (true, true) => {
                    // Should abstain, did abstain — correct
                    matrix.true_negative += 1;
                    ("tn".to_string(), true)
                }
                (true, false) => {
                    // Should abstain, but answered — hallucination
                    matrix.false_positive += 1;
                    ("fp".to_string(), false)
                }
                (false, true) => {
                    // Should answer, but abstained — missed
                    matrix.false_negative += 1;
                    ("fn".to_string(), false)
                }
            };

            item_results.push(GoldenItemResult {
                query: item.query.clone(),
                expected_answer: item.expected_answer.clone(),
                actual_response: response,
                abstained,
                correct,
                classification,
            });
        }

        let total = matrix.total();
        let answered = matrix.true_positive + matrix.false_positive;
        let coverage = if total > 0 {
            answered as f32 / total as f32
        } else {
            0.0
        };
        let hallucination_rate = if answered > 0 {
            matrix.false_positive as f32 / answered as f32
        } else {
            0.0
        };
        let faithfulness = if matrix.true_positive > 0 {
            let correct_answers = item_results
                .iter()
                .filter(|r| r.classification == "tp" && r.correct)
                .count();
            correct_answers as f32 / matrix.true_positive as f32
        } else {
            0.0
        };

        // Build risk-coverage curve
        let risk_coverage_curve = compute_risk_coverage_curve(&item_results);

        Ok(EvalReport {
            confusion_matrix: matrix,
            hallucination_rate,
            coverage,
            faithfulness,
            risk_coverage_curve,
            item_results,
        })
    }
}

/// Check if a response is an abstention (matches FAIL_RESPONSE pattern).
fn is_abstain_response(response: &str) -> bool {
    let trimmed = response.trim();
    trimmed == FAIL_RESPONSE
        || trimmed.contains("not able to provide an answer")
        || trimmed.contains("I don't know")
        || trimmed.contains("I cannot answer")
}

/// Rough correctness check: does the response contain key terms from the expected answer?
///
/// This is a simple heuristic. A proper evaluation would use LLM-as-judge or
/// exact-match with normalization.
fn is_roughly_correct(actual: &str, expected: Option<&str>) -> bool {
    let Some(expected) = expected else {
        return false;
    };

    // Normalize both strings
    let actual_lower = actual.to_lowercase();
    let expected_lower = expected.to_lowercase();

    // Check if the expected answer's key words appear in the actual response
    let expected_words: Vec<&str> = expected_lower
        .split_whitespace()
        .filter(|w| w.len() > 3) // Skip short words
        .collect();

    if expected_words.is_empty() {
        // If no significant words, check for substring match
        return actual_lower.contains(&expected_lower) || expected_lower.contains(&actual_lower);
    }

    // Count how many expected words appear in the actual response
    let matching = expected_words
        .iter()
        .filter(|w| actual_lower.contains(*w))
        .count();

    // Consider correct if >50% of key words match
    matching as f32 / expected_words.len() as f32 > 0.5
}

/// Compute the risk-coverage curve by sweeping thresholds.
///
/// For each threshold, we simulate what would happen if we only answer
/// queries where the model's confidence exceeds the threshold.
fn compute_risk_coverage_curve(item_results: &[GoldenItemResult]) -> Vec<RiskCoveragePoint> {
    let thresholds: Vec<f32> = (0..=20).map(|i| i as f32 * 0.05).collect();
    let mut curve = Vec::new();

    for threshold in thresholds {
        // Simulate: answer if the item was correct with confidence > threshold
        // For simplicity, we use the actual results and vary coverage by
        // randomly dropping answers (in practice, this would use model confidence)
        let total = item_results.len();
        if total == 0 {
            continue;
        }

        // Use a simple heuristic: items classified as fp are "risky"
        // Coverage decreases as threshold increases (more abstentions)
        let answered_count = item_results
            .iter()
            .filter(|r| {
                if r.abstained {
                    return false;
                }
                // Simulate threshold: tp items have high confidence, fp items have lower
                match r.classification.as_str() {
                    "tp" => threshold < 0.8,
                    "fp" => threshold < 0.3,
                    _ => false,
                }
            })
            .count();

        let risky_count = item_results
            .iter()
            .filter(|r| {
                if r.abstained {
                    return false;
                }
                match r.classification.as_str() {
                    "tp" => threshold < 0.8 && !r.correct,
                    "fp" => threshold < 0.3,
                    _ => false,
                }
            })
            .count();

        let coverage = answered_count as f32 / total as f32;
        let risk = if answered_count > 0 {
            risky_count as f32 / answered_count as f32
        } else {
            0.0
        };

        curve.push(RiskCoveragePoint {
            threshold,
            coverage,
            risk,
        });
    }

    curve
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_abstain_response_detects_fail() {
        assert!(is_abstain_response(FAIL_RESPONSE));
        assert!(is_abstain_response(
            "Sorry, I'm not able to provide an answer to that question.[no-context]"
        ));
        assert!(!is_abstain_response("The answer is 42."));
    }

    #[test]
    fn is_roughly_correct_basic() {
        assert!(is_roughly_correct(
            "The capital of France is Paris.",
            Some("Paris is the capital of France.")
        ));
        assert!(!is_roughly_correct(
            "The capital of Germany is Berlin.",
            Some("Paris is the capital of France.")
        ));
    }

    #[test]
    fn confusion_matrix_metrics() {
        let cm = ConfusionMatrix {
            true_positive: 80,
            false_positive: 10,
            true_negative: 5,
            false_negative: 5,
        };
        assert_eq!(cm.total(), 100);
        assert!((cm.precision() - 0.888).abs() < 0.01);
        assert!((cm.recall() - 0.941).abs() < 0.01);
    }
}
