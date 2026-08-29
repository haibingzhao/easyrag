//! Reciprocal Rank Fusion (RRF) for hybrid search
//!
//! Combines vector search and keyword search results using rank-based
//! scoring, which is score-magnitude agnostic (works regardless of cosine
//! vs BM25 scale differences).
//!
//! Formula: score(d) = vectorWeight / (k + rank_vector(d))
//!                    + keywordWeight / (k + rank_keyword(d))

use crate::JsonMap;
use std::collections::HashMap;

/// Fuse vector search results and keyword search results using weighted RRF.
///
/// - `vector_results`: results from vector similarity search (ordered)
/// - `keyword_results`: results from keyword/BM25 search (ordered)
/// - `rrf_k`: RRF constant (default 60, higher = less emphasis on top ranks)
/// - `vector_weight`: weight for vector search contribution (default 0.7)
/// - `keyword_weight`: weight for keyword search contribution (default 0.3)
///
/// Returns fused results sorted by RRF score descending, deduplicated by
/// "id", each entry carrying additional `score` (f32) and `rrf_score` (f64)
/// fields.
pub fn rrf_fusion(
    vector_results: Vec<JsonMap>,
    keyword_results: Vec<JsonMap>,
    rrf_k: usize,
    vector_weight: f32,
    keyword_weight: f32,
) -> Vec<JsonMap> {
    if keyword_results.is_empty() && vector_results.is_empty() {
        return Vec::new();
    }
    // NOTE: a single-sided input intentionally flows through the scoring
    // pass below (the empty side contributes nothing) so every returned
    // entry still carries the documented `score` and `rrf_score` fields.

    // Accumulate RRF scores per document ID.
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut doc_data: HashMap<String, JsonMap> = HashMap::new();
    let mut insertion_order: Vec<String> = Vec::new();

    // Score vector results by rank.
    accumulate_scores(
        &vector_results,
        vector_weight,
        rrf_k,
        &mut scores,
        &mut doc_data,
        &mut insertion_order,
    );

    // Score keyword results by rank.
    accumulate_scores(
        &keyword_results,
        keyword_weight,
        rrf_k,
        &mut scores,
        &mut doc_data,
        &mut insertion_order,
    );

    // Sort by fused RRF score descending.
    let mut entries: Vec<(String, f64)> = scores.into_iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    entries
        .into_iter()
        .map(|(id, rrf_score)| {
            let mut result = doc_data
                .remove(&id)
                .unwrap_or_else(|| {
                    let mut m = JsonMap::new();
                    m.insert("id".to_string(), serde_json::Value::String(id.clone()));
                    m
                });
            result.insert(
                "score".to_string(),
                serde_json::Value::from(rrf_score as f32),
            );
            result.insert("rrf_score".to_string(), serde_json::Value::from(rrf_score));
            result
        })
        .collect()
}

/// RRF fusion with default parameters (k=60, weights 0.7/0.3).
pub fn rrf_fusion_default(vector_results: Vec<JsonMap>, keyword_results: Vec<JsonMap>) -> Vec<JsonMap> {
    rrf_fusion(vector_results, keyword_results, 60, 0.7, 0.3)
}

/// Accumulate weighted RRF scores per document ID.
/// Keeps the first-seen (richer) document data as base (putIfAbsent).
fn accumulate_scores(
    results: &[JsonMap],
    weight: f32,
    rrf_k: usize,
    scores: &mut HashMap<String, f64>,
    doc_data: &mut HashMap<String, JsonMap>,
    insertion_order: &mut Vec<String>,
) {
    for (rank, doc) in results.iter().enumerate() {
        let Some(id) = doc.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let id = id.to_string();
        let rrf_score = weight as f64 / (rrf_k + rank + 1) as f64;
        *scores.entry(id.clone()).or_insert(0.0) += rrf_score;
        // Keep the richer document data (prefer first-seen result as base).
        if !doc_data.contains_key(&id) {
            insertion_order.push(id.clone());
            doc_data.insert(id, doc.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    // Ported from HybridSearchFusionTest.kt (8 cases).
    use super::*;

    fn doc(id: &str) -> JsonMap {
        doc_with_score(id, 0.5)
    }

    fn doc_with_score(id: &str, score: f32) -> JsonMap {
        let mut m = JsonMap::new();
        m.insert("id".to_string(), serde_json::Value::String(id.to_string()));
        m.insert(
            "content".to_string(),
            serde_json::Value::String(format!("text-{id}")),
        );
        m.insert("score".to_string(), serde_json::Value::from(score));
        m
    }

    fn ids(results: &[JsonMap]) -> Vec<String> {
        results
            .iter()
            .map(|d| d["id"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn rrf_fusion_both_empty_returns_empty() {
        let result = rrf_fusion_default(Vec::new(), Vec::new());
        assert!(result.is_empty());
    }

    #[test]
    fn rrf_fusion_keyword_empty_returns_vector_results() {
        let vec = vec![doc("a"), doc("b")];
        let result = rrf_fusion_default(vec, Vec::new());
        assert_eq!(vec!["a", "b"], ids(&result));
        // Single-sided path still carries the documented score fields.
        assert!(result.iter().all(|d| d.contains_key("rrf_score")));
        assert!(result.iter().all(|d| d.contains_key("score")));
    }

    #[test]
    fn rrf_fusion_vector_empty_returns_keyword_results() {
        let kw = vec![doc("x"), doc("y")];
        let result = rrf_fusion_default(Vec::new(), kw);
        assert_eq!(vec!["x", "y"], ids(&result));
        assert!(result.iter().all(|d| d.contains_key("rrf_score")));
        assert!(result.iter().all(|d| d.contains_key("score")));
    }

    #[test]
    fn rrf_fusion_overlapping_docs_fused_score_higher() {
        let vec = vec![doc("a"), doc("b"), doc("c")];
        let kw = vec![doc("b"), doc("c"), doc("d")];

        let result = rrf_fusion(vec, kw, 60, 0.7, 0.3);

        // "b" appears in both lists (rank 1 in vec, rank 0 in kw) → highest
        assert!(!result.is_empty());
        let ids = ids(&result);
        assert_eq!("b", ids.first().unwrap());
        // All 4 unique docs should be present
        assert_eq!(4, result.len());
        for expected in ["a", "b", "c", "d"] {
            assert!(ids.contains(&expected.to_string()));
        }
    }

    #[test]
    fn rrf_fusion_deduplicates_by_id() {
        let vec = vec![doc("a"), doc("a"), doc("b")];
        let kw = vec![doc("a"), doc("c")];

        let result = rrf_fusion_default(vec, kw);
        let ids = ids(&result);
        let mut distinct = ids.clone();
        distinct.sort();
        distinct.dedup();
        // No duplicates
        assert_eq!(ids.len(), distinct.len());
    }

    #[test]
    fn rrf_fusion_scores_are_descending() {
        let vec = vec![doc("a"), doc("b"), doc("c"), doc("d")];
        let kw = vec![doc("e"), doc("a"), doc("f"), doc("b")];

        let result = rrf_fusion_default(vec, kw);
        let scores: Vec<f64> = result
            .iter()
            .map(|d| d["rrf_score"].as_f64().unwrap())
            .collect();
        for i in 0..scores.len() - 1 {
            assert!(scores[i] >= scores[i + 1], "Scores should be descending");
        }
    }

    #[test]
    fn rrf_fusion_custom_weights_affects_ranking() {
        let vec = vec![doc("a"), doc("b")];
        let kw = vec![doc("b"), doc("a")];

        // With high keyword weight, "b" (rank 0 in kw) beats "a"
        let result_kw_heavy = rrf_fusion(vec.clone(), kw.clone(), 60, 0.1, 0.9);
        assert_eq!("b", result_kw_heavy.first().unwrap()["id"].as_str().unwrap());

        // With high vector weight, "a" (rank 0 in vec) beats "b"
        let result_vec_heavy = rrf_fusion(vec, kw, 60, 0.9, 0.1);
        assert_eq!("a", result_vec_heavy.first().unwrap()["id"].as_str().unwrap());
    }

    #[test]
    fn rrf_fusion_result_contains_rrf_score_field() {
        let result = rrf_fusion_default(vec![doc("a")], vec![doc("b")]);
        assert!(result.iter().all(|d| d.contains_key("rrf_score")));
        assert!(result.iter().all(|d| d.contains_key("score")));
    }
}
