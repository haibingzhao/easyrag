//! Evaluation types for measuring RAG pipeline quality.
//!
//! Provides golden set evaluation, confusion matrix, and risk-coverage
//! analysis for anti-hallucination guardrail tuning.

use serde::{Deserialize, Serialize};

/// A single item in a golden evaluation set.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldenItem {
    /// The query to evaluate.
    pub query: String,
    /// The expected answer. `None` means the system should abstain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_answer: Option<String>,
    /// Optional query type classification (single_hop, multi_hop, etc.).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
}

/// Confusion matrix for binary classification: answer vs. abstain.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Correctly answered (should answer, did answer).
    pub true_positive: usize,
    /// Hallucination (should abstain, but answered).
    pub false_positive: usize,
    /// Correctly abstained (should abstain, did abstain).
    pub true_negative: usize,
    /// Missed answer (should answer, but abstained).
    pub false_negative: usize,
}

impl ConfusionMatrix {
    /// Total number of evaluated items.
    pub fn total(&self) -> usize {
        self.true_positive + self.false_positive + self.true_negative + self.false_negative
    }

    /// Precision: TP / (TP + FP).
    pub fn precision(&self) -> f32 {
        let denom = self.true_positive + self.false_positive;
        if denom == 0 {
            0.0
        } else {
            self.true_positive as f32 / denom as f32
        }
    }

    /// Recall: TP / (TP + FN).
    pub fn recall(&self) -> f32 {
        let denom = self.true_positive + self.false_negative;
        if denom == 0 {
            0.0
        } else {
            self.true_positive as f32 / denom as f32
        }
    }

    /// F1 score: harmonic mean of precision and recall.
    pub fn f1(&self) -> f32 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }
}

/// A point on the risk-coverage curve.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskCoveragePoint {
    /// The threshold value used.
    pub threshold: f32,
    /// Coverage: fraction of queries that were answered (not abstained).
    pub coverage: f32,
    /// Risk: fraction of answered queries that were incorrect (hallucinated).
    pub risk: f32,
}

/// Complete evaluation report for a golden set run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvalReport {
    /// Confusion matrix (answer vs. abstain classification).
    pub confusion_matrix: ConfusionMatrix,
    /// Hallucination rate: FP / (TP + FP) — fraction of answers that are unsupported.
    pub hallucination_rate: f32,
    /// Coverage: fraction of queries that received an answer.
    pub coverage: f32,
    /// Faithfulness: fraction of answers that are factually correct (estimated).
    pub faithfulness: f32,
    /// Risk-coverage curve points (sorted by threshold).
    pub risk_coverage_curve: Vec<RiskCoveragePoint>,
    /// Per-item results for debugging.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub item_results: Vec<GoldenItemResult>,
}

/// Result of evaluating a single golden item.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GoldenItemResult {
    /// The query.
    pub query: String,
    /// The expected answer (if any).
    pub expected_answer: Option<String>,
    /// The actual response from the pipeline.
    pub actual_response: String,
    /// Whether the pipeline abstained.
    pub abstained: bool,
    /// Whether the result is correct.
    pub correct: bool,
    /// Classification: tp/fp/tn/fn.
    pub classification: String,
}
