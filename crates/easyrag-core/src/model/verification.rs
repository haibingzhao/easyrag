//! Verification, evidence quality, query routing, and citation check types
//! for the anti-hallucination guardrail layer.

use serde::{Deserialize, Serialize};

// ===== Claim Verification (Feature 1) =====

/// Result of verifying a single atomic claim against its cited context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClaimVerification {
    /// The atomic claim text.
    pub claim: String,
    /// Whether the claim is supported by the context.
    pub supported: bool,
    /// Confidence score (0.0–1.0) for the verification decision.
    pub confidence: f32,
}

/// Aggregate result of the verification gate over all claims in an answer.
#[derive(Clone, Debug)]
pub struct VerificationResult {
    /// Individual claim verifications.
    pub claims: Vec<ClaimVerification>,
    /// True only if every claim was verified as supported.
    pub all_verified: bool,
    /// The minimum support score across all claims (weakest link).
    pub min_support: f32,
    /// Whether the system should abstain from answering.
    pub abstain: bool,
}

// ===== Evidence Quality Scoring (Feature 2) =====

/// Decision from the CRAG-style evidence quality scorer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityDecision {
    /// Evidence is strong enough to proceed with generation.
    Generate,
    /// Evidence is borderline; refine the query and re-retrieve.
    Refine,
    /// Evidence is too weak; abstain without generating.
    Abstain,
}

/// Score and decision from the evidence quality scorer.
#[derive(Clone, Debug)]
pub struct EvidenceQualityScore {
    /// Quality score (0.0–1.0).
    pub score: f32,
    /// Decision based on configured thresholds.
    pub decision: QualityDecision,
}

impl EvidenceQualityScore {
    /// Create a score with automatic decision based on good/bad thresholds.
    pub fn with_thresholds(score: f32, good: f32, bad: f32) -> Self {
        let decision = if score >= good {
            QualityDecision::Generate
        } else if score < bad {
            QualityDecision::Abstain
        } else {
            QualityDecision::Refine
        };
        Self { score, decision }
    }
}

// ===== Query Routing (Feature 6) =====

/// Query route classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryRouteType {
    /// Greetings, chitchat, or questions no document corpus could answer.
    NoRetrieval,
    /// Answerable by finding one fact.
    SingleHop,
    /// Needs combining facts from multiple documents.
    MultiHop,
}

/// Result of query routing and decomposition.
#[derive(Clone, Debug)]
pub struct QueryRoute {
    /// The classified route type.
    pub route_type: QueryRouteType,
    /// Sub-queries for multi-hop decomposition (empty for other types).
    pub sub_queries: Vec<String>,
    /// Whether the query contains a false premise.
    pub has_false_premise: bool,
}

// ===== Citation Check (Feature 3) =====

/// Result of citation validity checking.
#[derive(Clone, Debug)]
pub struct CitationCheckResult {
    /// Citation IDs that are valid (within range).
    pub valid_citations: Vec<usize>,
    /// Citation IDs that are invalid (out of range or fabricated).
    pub invalid_citations: Vec<usize>,
    /// Response text with invalid citation markers stripped.
    pub stripped_response: String,
}
