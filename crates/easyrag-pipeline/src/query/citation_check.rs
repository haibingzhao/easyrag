//! Citation validity checking (Feature 3).
//!
//! Validates that citation markers `[n]` in LLM responses reference valid
//! chunk IDs. Strips fabricated citation markers that don't correspond to
//! any retrieved chunk. Pure CPU operation — no LLM calls required.

use easyrag_core::model::verification::CitationCheckResult;
use regex::Regex;
use std::sync::LazyLock;

/// Regex matching citation markers like `[1]`, `[2]`, `[12]`.
/// Captures the numeric ID inside brackets.
static CITATION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d+)\]").expect("citation regex must compile")
});

/// Validate all citation markers in a response against the known reference range.
///
/// `max_reference_id` is the highest valid reference ID (typically the number
/// of retrieved chunks, since references are 1-based sequential IDs).
///
/// Returns a `CitationCheckResult` with valid/invalid citation lists and the
/// cleaned response with invalid markers stripped.
pub fn validate_citations(response: &str, max_reference_id: usize) -> CitationCheckResult {
    if max_reference_id == 0 {
        // No references provided — strip all citation markers
        let stripped = CITATION_RE.replace_all(response, "").to_string();
        return CitationCheckResult {
            valid_citations: Vec::new(),
            invalid_citations: Vec::new(),
            stripped_response: stripped,
        };
    }

    let mut valid = Vec::new();
    let mut invalid = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for cap in CITATION_RE.captures_iter(response) {
        if let Ok(n) = cap[1].parse::<usize>() {
            if seen.insert(n) {
                if n >= 1 && n <= max_reference_id {
                    valid.push(n);
                } else {
                    invalid.push(n);
                }
            }
        }
    }

    // Strip invalid citation markers from the response
    let stripped = if invalid.is_empty() {
        response.to_string()
    } else {
        let result = CITATION_RE.replace_all(response, |caps: &regex::Captures| {
            if let Ok(n) = caps[1].parse::<usize>() {
                if n >= 1 && n <= max_reference_id {
                    // Valid citation — keep as-is
                    return caps[0].to_string();
                }
            }
            // Invalid citation — remove the marker entirely
            String::new()
        });
        result.to_string()
    };

    CitationCheckResult {
        valid_citations: valid,
        invalid_citations: invalid,
        stripped_response: stripped,
    }
}

/// Extract all citation IDs from a response (without validation).
pub fn extract_citations(response: &str) -> Vec<usize> {
    CITATION_RE
        .captures_iter(response)
        .filter_map(|cap| cap[1].parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_citations_all_valid() {
        let response = "Paris is the capital of France [1]. The Louvre opened in 1793 [2].";
        let result = validate_citations(response, 3);
        assert_eq!(result.valid_citations, vec![1, 2]);
        assert!(result.invalid_citations.is_empty());
        assert_eq!(result.stripped_response, response);
    }

    #[test]
    fn validate_citations_strips_invalid() {
        let response =
            "Paris is the capital of France [1]. The Louvre opened in 1793 [99].";
        let result = validate_citations(response, 3);
        assert_eq!(result.valid_citations, vec![1]);
        assert_eq!(result.invalid_citations, vec![99]);
        assert!(!result.stripped_response.contains("[99]"));
        assert!(result.stripped_response.contains("[1]"));
    }

    #[test]
    fn validate_citations_zero_max() {
        let response = "Some text with [1] citation.";
        let result = validate_citations(response, 0);
        assert!(result.valid_citations.is_empty());
        assert!(!result.stripped_response.contains("[1]"));
    }

    #[test]
    fn validate_citations_deduplicates() {
        let response = "Fact [1] and again [1] and [2].";
        let result = validate_citations(response, 5);
        assert_eq!(result.valid_citations, vec![1, 2]);
    }

    #[test]
    fn extract_citations_basic() {
        let response = "See [1] and [3] and [10].";
        assert_eq!(extract_citations(response), vec![1, 3, 10]);
    }
}
