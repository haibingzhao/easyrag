//! The one implementation of graph cleanup after a document is deleted.
//!
//! Both entry points — `easyrag-server`'s cascade delete and the
//! [`crate::easy_rag`] facade delete — used to carry their own copy of this
//! orchestration, and the two had already drifted apart: one split `source_id`
//! on `<SEP>` *and* `,` while the other split only on `<SEP>`, and one rewrote
//! the trimmed list through a `HashSet` (so the stored element order depended
//! on the hash seed) while the other preserved it. The same stored row
//! therefore produced two different trims depending on which entry point
//! happened to delete the document.
//!
//! It lives in `pipeline` because that is the lowest crate both callers can
//! reach (`server` → `pipeline`), and because the sequence is a cross-storage
//! business decision rather than a storage detail: deleting a graph node must
//! also delete its entity description from the vector store. Pushing that into
//! the three `GraphStorage` backends would mean three copies of the decision —
//! which is precisely how the drift above happened.
//!
//! # Cost
//!
//! Every step is scoped to the chunks being deleted, so the round-trip count
//! is independent of how large the graph is: two filtered reads
//! ([`GraphStorage::get_nodes_by_source_ids`] /
//! [`GraphStorage::get_edges_by_source_ids`]), one incident-edge lookup per
//! deleted node, then at most four writes. Before this, both callers pulled
//! the entire graph into `JsonMap`s — every node's full description text — to
//! find the handful that referenced the deleted chunks.

use std::collections::HashSet;

use easyrag_core::model::source_id::{parse_source_ids, source_ids_value};
use easyrag_core::storage::traits::{GraphStorage, VectorStorage};
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

/// What one cleanup run did, so each caller can keep its own log wording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphCleanupReport {
    /// Nodes whose every chunk reference was deleted, so nothing proves them.
    pub deleted_nodes: usize,
    /// Edges removed — either because the edge itself referenced a deleted
    /// chunk, or because one of its endpoints was deleted.
    pub deleted_edges: usize,
    /// Nodes that survived on their remaining chunks with a shortened
    /// `source_id`.
    pub trimmed_nodes: usize,
}

/// Drop every graph reference to `chunk_ids`: delete what is left unproven,
/// trim what still has other provenance.
///
/// A node is deleted only when *all* of its chunk references are in
/// `chunk_ids`; a node that also references live chunks keeps existing with
/// those remaining ids, written back in the canonical array form and in the
/// stored order. An edge goes when its own `source_id` references a deleted
/// chunk, or when either endpoint is being deleted.
///
/// An empty `chunk_ids` is a no-op: nothing was deleted, so nothing is
/// orphaned.
pub async fn cleanup_graph_by_chunks(
    graph: &dyn GraphStorage,
    vector: &dyn VectorStorage,
    chunk_ids: &HashSet<String>,
    biz_id: &str,
) -> easyrag_core::Result<GraphCleanupReport> {
    if chunk_ids.is_empty() {
        return Ok(GraphCleanupReport::default());
    }
    // The filtered reads take a slice; a `HashSet` keeps the callers' lookup
    // below O(1) and the storage layer does not care about order.
    let chunks: Vec<String> = chunk_ids.iter().cloned().collect();

    // --- nodes ---------------------------------------------------------
    let mut nodes_to_delete: Vec<String> = Vec::new();
    let mut nodes_to_trim: Vec<(String, JsonMap)> = Vec::new();

    for node in graph.get_nodes_by_source_ids(&chunks, biz_id).await? {
        let Some(node_id) = node.get("id").and_then(JsonValue::as_str) else {
            continue;
        };
        // `parse_source_ids` is the one split rule: array or legacy
        // `<SEP>`/`,`-joined string, trimmed, order-preserving dedup.
        let mut sources = parse_source_ids(node.get("source_id"));

        let before = sources.len();
        sources.retain(|id| !chunk_ids.contains(id));
        // Defensive: the server-side filter is at least as strict as this
        // parse, so a candidate normally always loses something here. Keeping
        // the guard means a backend whose predicate is looser than the Rust
        // rule (the SQL and Cypher legacy arms do not trim) can never make an
        // untouched node look trimmable.
        if sources.len() == before {
            continue;
        }

        if sources.is_empty() {
            nodes_to_delete.push(node_id.to_string());
        } else {
            let mut updated = node.clone();
            // Canonical array form, in the stored order — one caller used to
            // round-trip this through a `HashSet`, so the rewritten order
            // depended on the hash seed.
            updated.insert("source_id".to_string(), source_ids_value(&sources));
            nodes_to_trim.push((node_id.to_string(), updated));
        }
    }

    // --- edges ---------------------------------------------------------
    let mut edges_to_delete: Vec<(String, String)> = Vec::new();
    let mut seen_edges: HashSet<(String, String)> = HashSet::new();

    // (a) The edge itself is provenanced to a deleted chunk while both
    // endpoints survive — exactly what the filtered read answers.
    for edge in graph.get_edges_by_source_ids(&chunks, biz_id).await? {
        let Some(source) = edge.get("source").and_then(JsonValue::as_str) else {
            continue;
        };
        let Some(target) = edge.get("target").and_then(JsonValue::as_str) else {
            continue;
        };
        push_edge(&mut edges_to_delete, &mut seen_edges, source, target);
    }

    // (b) An endpoint is going away, which orphans the edge whatever its own
    // `source_id` says. Asking for the incident edges of just the deleted nodes
    // replaced scanning every edge in the graph and testing both endpoints
    // against the delete list — O(edges x deleted) before, which at 150k edges
    // and a 200-node delete was tens of millions of string comparisons.
    if !nodes_to_delete.is_empty() {
        for incident in graph
            .get_nodes_edges_batch(&nodes_to_delete, biz_id)
            .await?
            .into_values()
        {
            for (source, target) in incident {
                push_edge(&mut edges_to_delete, &mut seen_edges, &source, &target);
            }
        }
    }

    // --- writes --------------------------------------------------------
    if !edges_to_delete.is_empty() {
        graph.remove_edges(&edges_to_delete, biz_id).await?;
    }
    if !nodes_to_delete.is_empty() {
        graph.remove_nodes(&nodes_to_delete, biz_id).await?;
        // Entity descriptions are embedded into the vector store, so an entity
        // that stops existing has to stop being retrievable there too.
        vector.delete(&nodes_to_delete, biz_id).await?;
    }
    if !nodes_to_trim.is_empty() {
        // One round trip; both callers used to loop `upsert_node` per node.
        graph.upsert_nodes_batch(&nodes_to_trim, biz_id).await?;
    }

    Ok(GraphCleanupReport {
        deleted_nodes: nodes_to_delete.len(),
        deleted_edges: edges_to_delete.len(),
        trimmed_nodes: nodes_to_trim.len(),
    })
}

/// Append one edge to the delete list unless it is already there.
///
/// Identity is undirected: all three backends delete edges symmetrically
/// (memory normalises its map key, Postgres ORs both column orders, Cypher
/// matches either direction), so `(A,B)` from the endpoint lookup and `(B,A)`
/// from the filtered read are one edge. Counting them twice would inflate
/// `deleted_edges` and send a redundant DELETE.
fn push_edge(
    out: &mut Vec<(String, String)>,
    seen: &mut HashSet<(String, String)>,
    source: &str,
    target: &str,
) {
    let key = if source <= target {
        (source.to_string(), target.to_string())
    } else {
        (target.to_string(), source.to_string())
    };
    if seen.insert(key) {
        out.push((source.to_string(), target.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same edge reached from both directions is one deletion.
    #[test]
    fn push_edge_deduplicates_across_direction() {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        push_edge(&mut out, &mut seen, "A", "B");
        push_edge(&mut out, &mut seen, "B", "A");
        push_edge(&mut out, &mut seen, "A", "B");
        assert_eq!(out, vec![("A".to_string(), "B".to_string())]);
    }

    /// Distinct pairs stay distinct, and the first-seen direction is kept.
    #[test]
    fn push_edge_keeps_distinct_pairs_in_order() {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        push_edge(&mut out, &mut seen, "B", "A");
        push_edge(&mut out, &mut seen, "C", "A");
        assert_eq!(
            out,
            vec![
                ("B".to_string(), "A".to_string()),
                ("C".to_string(), "A".to_string())
            ]
        );
    }
}
