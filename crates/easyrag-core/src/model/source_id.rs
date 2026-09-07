//! `source_id`: the chunk-provenance list carried by every graph node and edge.
//!
//! A node's `source_id` records which document chunks the entity was extracted
//! from; an edge's records which chunks asserted the relationship. Deleting a
//! document therefore has to find every node/edge referencing its chunks, drop
//! those references, and delete whatever is left orphaned.
//!
//! **This module is the only place that decides how `source_id` is split,
//! joined, deduplicated and capped.** That rule used to be re-derived at five
//! call sites and had already drifted into three incompatible versions — one
//! split on `<SEP>` only, two split on `<SEP>` *and* `,`, one of those two
//! returned a `HashSet` so the rewritten value came back in a random order.
//! The same legacy data trimmed differently depending on which delete path
//! handled it. Re-deciding the rule at a call site is how that happened; don't.
//!
//! Two storage shapes are accepted on read and one is produced on write:
//!
//! - **Array** (canonical): `["chunk-1","chunk-2"]`
//! - **String** (legacy): `"chunk-1<SEP>chunk-2"`, occasionally comma-joined
//!
//! Legacy strings are still read because rows written before the array
//! migration remain in the wild (backfill pending, migration failed non-fatally,
//! or an old pod still writing during a rolling deploy). Read tolerance is
//! permanent and cheap; it is what makes the write-side switch and the backfill
//! independently revertible without ever risking a mis-read that deletes data.

use crate::constants::GRAPH_FIELD_SEP;
use crate::{JsonMap, JsonValue};
use std::collections::HashSet;

/// Parse a stored `source_id` into its chunk ids.
///
/// Accepts both the canonical array and the legacy `<SEP>`/`,`-joined string.
/// Entries are trimmed, empties dropped, and duplicates removed **preserving
/// first-occurrence order** — the order is observable, because the trimmed list
/// gets written back and FIFO capping drops from the front.
///
/// Anything else (`null`, absent, a number, an object) yields an empty list
/// rather than an error: a node with no readable provenance is treated as
/// referencing no chunks, which leaves it alive instead of orphaning it.
pub fn parse_source_ids(value: Option<&JsonValue>) -> Vec<String> {
    let Some(value) = value else {
        return Vec::new();
    };
    let raw: Vec<&str> = match value {
        JsonValue::Array(items) => items.iter().filter_map(|item| item.as_str()).collect(),
        // Legacy joined form. Comma is split as well as `<SEP>`: two of the
        // three historical readers split on it, so stored values may contain
        // either delimiter and dropping one would silently under-trim.
        JsonValue::String(joined) => joined
            .split(GRAPH_FIELD_SEP)
            .flat_map(|part| part.split(','))
            .collect(),
        _ => return Vec::new(),
    };

    let mut seen: HashSet<&str> = HashSet::with_capacity(raw.len());
    let mut ids = Vec::with_capacity(raw.len());
    for id in raw {
        let id = id.trim();
        if id.is_empty() || !seen.insert(id) {
            continue;
        }
        ids.push(id.to_string());
    }
    ids
}

/// Render chunk ids for storage — always a JSON array.
///
/// The single write-side rule. Element order is the caller's: [`merge_source_ids`]
/// produces recency order and cleanup preserves the stored order.
pub fn source_ids_value(ids: &[String]) -> JsonValue {
    JsonValue::Array(
        ids.iter()
            .map(|id| JsonValue::String(id.clone()))
            .collect(),
    )
}

/// Append `new_id` to an existing `source_id`, deduplicating and FIFO-capping.
///
/// `existing` may be either storage shape (or absent); the result is always the
/// canonical `Vec<String>` ready for [`source_ids_value`]. The new id goes last,
/// so when the list exceeds `max` the **oldest** entries are dropped.
///
/// `max == 0` means unlimited rather than "keep nothing": a zero cap would
/// silently erase provenance instead of expressing a policy, and the pre-array
/// implementation already bypassed its cap on the empty-existing fast path.
pub fn merge_source_ids(existing: Option<&JsonValue>, new_id: &str, max: usize) -> Vec<String> {
    let mut ids = parse_source_ids(existing);
    if !ids.iter().any(|id| id == new_id) {
        ids.push(new_id.to_string());
    }
    if max > 0 && ids.len() > max {
        // `split_off` keeps the head and returns the tail; the tail is the
        // most recent `max` entries, which is what FIFO retention wants.
        ids = ids.split_off(ids.len() - max);
    }
    ids
}

/// Does this property map's `source_id` reference any of `wanted`?
///
/// Membership is **whole-element equality**, never substring matching: that is
/// the entire point of the array model. A `CONTAINS`-style check on the legacy
/// joined string would match `chunk_1` inside `chunk_10` and delete unrelated
/// graph data.
///
/// Short-circuits without allocating, because the storage-layer default
/// implementation calls it once per node/edge while filtering a whole graph.
pub fn source_ids_intersect(props: &JsonMap, wanted: &HashSet<String>) -> bool {
    source_id_intersects(props.get("source_id"), wanted)
}

/// [`source_ids_intersect`] for a bare `source_id` value.
pub fn source_id_intersects(value: Option<&JsonValue>, wanted: &HashSet<String>) -> bool {
    if wanted.is_empty() {
        return false;
    }
    match value {
        Some(JsonValue::Array(items)) => items
            .iter()
            .filter_map(|item| item.as_str())
            .map(str::trim)
            .any(|id| !id.is_empty() && wanted.contains(id)),
        Some(JsonValue::String(joined)) => joined
            .split(GRAPH_FIELD_SEP)
            .flat_map(|part| part.split(','))
            .map(str::trim)
            .any(|id| !id.is_empty() && wanted.contains(id)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn wanted(ids: &[&str]) -> HashSet<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    #[test]
    fn array_is_read_element_wise() {
        let value = json!(["chunk-1", "chunk-2"]);
        assert_eq!(
            parse_source_ids(Some(&value)),
            vec!["chunk-1".to_string(), "chunk-2".to_string()]
        );
    }

    #[test]
    fn legacy_sep_string_is_split() {
        let value = json!("chunk-1<SEP>chunk-2");
        assert_eq!(
            parse_source_ids(Some(&value)),
            vec!["chunk-1".to_string(), "chunk-2".to_string()]
        );
    }

    #[test]
    fn legacy_comma_string_is_split_too() {
        // Both delimiters, plus the padding the old readers trimmed away.
        let value = json!("c1<SEP>c2,c3 , ");
        assert_eq!(
            parse_source_ids(Some(&value)),
            vec!["c1".to_string(), "c2".to_string(), "c3".to_string()]
        );
    }

    #[test]
    fn a_bare_string_is_a_single_id() {
        // `<SEP>`-splitting a delimiter-free string yields the string itself.
        let value = json!("chunk-7");
        assert_eq!(parse_source_ids(Some(&value)), vec!["chunk-7".to_string()]);
    }

    #[test]
    fn duplicates_collapse_preserving_first_occurrence() {
        let value = json!(["b", "a", "b", "a", "c"]);
        assert_eq!(
            parse_source_ids(Some(&value)),
            vec!["b".to_string(), "a".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn unreadable_shapes_yield_empty() {
        for value in [
            None,
            Some(&json!(null)),
            Some(&json!(42)),
            Some(&json!({"a": 1})),
            Some(&json!("")),
            Some(&json!("   ")),
            Some(&json!([])),
        ] {
            assert_eq!(parse_source_ids(value), Vec::<String>::new());
        }
    }

    #[test]
    fn non_string_array_elements_are_skipped() {
        let value = json!(["chunk-1", 7, null, ["nested"], "chunk-2"]);
        assert_eq!(
            parse_source_ids(Some(&value)),
            vec!["chunk-1".to_string(), "chunk-2".to_string()]
        );
    }

    #[test]
    fn encode_always_produces_an_array() {
        let ids: Vec<String> = vec!["chunk-1".into(), "chunk-2".into()];
        assert_eq!(source_ids_value(&ids), json!(["chunk-1", "chunk-2"]));
        // Even when empty: an absent provenance list is `[]`, never `""`.
        assert_eq!(source_ids_value(&[]), json!([]));
    }

    #[test]
    fn encode_then_parse_round_trips() {
        let ids: Vec<String> = vec!["chunk-1".into(), "chunk-2".into()];
        assert_eq!(parse_source_ids(Some(&source_ids_value(&ids))), ids);
    }

    #[test]
    fn merge_appends_a_new_id_last() {
        let existing = json!(["chunk-1", "chunk-2"]);
        assert_eq!(
            merge_source_ids(Some(&existing), "chunk-3", 30),
            vec!["chunk-1".to_string(), "chunk-2".to_string(), "chunk-3".to_string()]
        );
    }

    #[test]
    fn merge_does_not_duplicate_an_existing_id() {
        let existing = json!("chunk-7");
        assert_eq!(
            merge_source_ids(Some(&existing), "chunk-7", 30),
            vec!["chunk-7".to_string()]
        );
    }

    #[test]
    fn merge_drops_the_oldest_when_over_the_cap() {
        // FIFO: five existing + one new, cap 3 → the three most recent.
        let existing = json!("chunk-1<SEP>chunk-2<SEP>chunk-3<SEP>chunk-4<SEP>chunk-5");
        assert_eq!(
            merge_source_ids(Some(&existing), "chunk-7", 3),
            vec![
                "chunk-4".to_string(),
                "chunk-5".to_string(),
                "chunk-7".to_string()
            ]
        );
    }

    #[test]
    fn merge_accepts_both_storage_shapes_identically() {
        let as_string = json!("chunk-1<SEP>chunk-2");
        let as_array = json!(["chunk-1", "chunk-2"]);
        assert_eq!(
            merge_source_ids(Some(&as_string), "chunk-3", 30),
            merge_source_ids(Some(&as_array), "chunk-3", 30)
        );
    }

    #[test]
    fn merge_without_existing_starts_a_fresh_list() {
        assert_eq!(
            merge_source_ids(None, "chunk-1", 30),
            vec!["chunk-1".to_string()]
        );
    }

    #[test]
    fn merge_treats_a_zero_cap_as_unlimited() {
        let existing = json!(["chunk-1", "chunk-2"]);
        assert_eq!(
            merge_source_ids(Some(&existing), "chunk-3", 0),
            vec!["chunk-1".to_string(), "chunk-2".to_string(), "chunk-3".to_string()]
        );
    }

    #[test]
    fn intersect_matches_whole_elements_only() {
        // The correctness goal of the whole refactor: `chunk_1` must not match
        // `chunk_10`, which a substring check on the joined string would.
        let props: JsonMap = json!({"source_id": ["chunk_10", "chunk_100"]})
            .as_object()
            .cloned()
            .unwrap();
        assert!(!source_ids_intersect(&props, &wanted(&["chunk_1"])));
        assert!(source_ids_intersect(&props, &wanted(&["chunk_10"])));
    }

    #[test]
    fn intersect_matches_legacy_strings_too() {
        let props: JsonMap = json!({"source_id": "chunk_10<SEP>chunk_7"})
            .as_object()
            .cloned()
            .unwrap();
        assert!(source_ids_intersect(&props, &wanted(&["chunk_7"])));
        assert!(!source_ids_intersect(&props, &wanted(&["chunk_1"])));
    }

    #[test]
    fn intersect_is_false_for_an_empty_wanted_set() {
        // Guards the storage-layer default implementation: an empty chunk list
        // must select nothing, never everything.
        let props: JsonMap = json!({"source_id": ["chunk-1"]})
            .as_object()
            .cloned()
            .unwrap();
        assert!(!source_ids_intersect(&props, &HashSet::new()));
    }

    #[test]
    fn intersect_is_false_without_a_source_id() {
        let props: JsonMap = json!({"entity_type": "Person"}).as_object().cloned().unwrap();
        assert!(!source_ids_intersect(&props, &wanted(&["chunk-1"])));
    }
}
