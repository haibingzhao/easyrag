//! Stable machine-readable degradation markers for the pure-data endpoint (R8/R9).
//!
//! Before this module the data endpoint could only fail loudly: a structure
//! request that could not be served silently fell through to plain vector
//! recall, and a deployment without a keyword index silently recalled by vector
//! alone. The caller had no way to tell "the index answered nothing" from "the
//! index was never consulted".
//!
//! These constants are a **wire contract**, not log text: they are surfaced
//! verbatim in `QueryDataResult::degrade_reasons` and therefore in the
//! `degradeReasons` JSON array. Renaming or removing one is a breaking change —
//! add a new marker instead. The full enumeration and the rules for when each
//! may fire live in `.docs/easyrag-data-endpoint-mode-matrix.md`.
//!
//! Signal hygiene, enforced at every emission site:
//!
//! * A marker is emitted **only** when the request actually asked for the
//!   capability it reports on. Not asking for structure retrieval is not
//!   degradation, and neither is a naive request running without a keyword
//!   index.
//! * Structure degradation and recall degradation are independent and may both
//!   appear in one response — which is why the field is a list and not a single
//!   string.

/// The deployment-level rollout gate (`QUERY_DATA_STRUCTURE_ENABLED`) is off, so
/// structure retrieval was not attempted even though the caller asked for it.
pub const STRUCTURE_DISABLED: &str = "structure_disabled_by_config";

/// No structure strategy was wired into the engine (a deployment shape without
/// the KV/vector handles structure retrieval needs).
pub const STRUCTURE_UNAVAILABLE: &str = "structure_strategy_unavailable";

/// The `bizId` + `structureScope` filter matched no processed document carrying
/// a structure index, so there was nothing to search.
pub const NO_DOCUMENT_IN_SCOPE: &str = "structure_no_document_in_scope";

/// A document was located but its structure-index row is missing or unreadable
/// (never built, or failed to deserialize).
pub const NO_INDEX: &str = "structure_no_index";

/// `useAgentMode` was requested, but every agent tool returns a prose string and
/// none of them can be projected into `sections[]`. Tree search ran instead.
pub const AGENT_MODE_UNSUPPORTED: &str = "structure_agent_mode_unsupported";

/// `sections[].content` came from the approximate vector heuristic instead of
/// the exact `chunks_list` join, because the document's registry row had no
/// chunk ids. The heuristic may return chunks outside the requested section, so
/// callers should treat `content` as best-effort when this marker is present.
pub const SECTION_CONTENT_APPROXIMATE: &str = "structure_section_content_approximate";

/// More structure documents were in scope than the fan-out limit allows; only
/// the most relevant `STRUCTURE_DATA_MAX_DOCS` were searched.
pub const SCOPE_TRUNCATED: &str = "structure_scope_truncated";

/// Hybrid recall was requested but no keyword index is configured, so recall
/// silently degraded to vector-only. Emitted **only** when this request's plan
/// asked for fusion — tagging every naive call would devalue the marker.
pub const KEYWORD_SEARCH_UNAVAILABLE: &str = "keyword_search_unavailable";

#[cfg(test)]
mod tests {
    use super::*;

    /// Every marker is a stable wire value: `snake_case`, no whitespace, and
    /// unique. The uniqueness check is what stops two call sites from emitting
    /// the same reason for different causes.
    #[test]
    fn degrade_markers_are_unique_snake_case() {
        let all = [
            STRUCTURE_DISABLED,
            STRUCTURE_UNAVAILABLE,
            NO_DOCUMENT_IN_SCOPE,
            NO_INDEX,
            AGENT_MODE_UNSUPPORTED,
            SECTION_CONTENT_APPROXIMATE,
            SCOPE_TRUNCATED,
            KEYWORD_SEARCH_UNAVAILABLE,
        ];
        for marker in all {
            assert!(
                marker
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit()),
                "degrade marker {marker:?} must be snake_case"
            );
        }
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(all.len(), unique.len(), "degrade markers must be distinct");
    }
}
