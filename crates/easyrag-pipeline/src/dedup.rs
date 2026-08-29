//! MinHash LSH near-duplicate detection (Feature 5).
//!
//! Pure Rust implementation of MinHash + Locality Sensitive Hashing for
//! detecting near-duplicate documents and chunks. Uses word-level shingles
//! and no external dependencies beyond the standard hash infrastructure.

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Result of a deduplication scan.
#[derive(Clone, Debug)]
pub struct DedupResult {
    /// Indices of documents/chunks that are unique (not duplicates).
    pub unique_indices: Vec<usize>,
    /// Groups of duplicate indices. Each group shares a near-duplicate
    /// cluster; the first index in each group is the "representative".
    pub duplicate_groups: Vec<Vec<usize>>,
}

/// MinHash LSH deduplicator for near-duplicate text detection.
///
/// # Parameters
/// - `num_hashes`: Number of hash functions in the MinHash signature (default: 128)
/// - `num_bands`: Number of LSH bands (default: 16)
/// - `band_rows`: Rows per band (default: 8); `num_bands * band_rows == num_hashes`
/// - `ngram_size`: Word-level shingle size (default: 5)
/// - `similarity_threshold`: Jaccard similarity threshold for considering duplicates
pub struct MinHashDeduplicator {
    num_hashes: usize,
    num_bands: usize,
    band_rows: usize,
    ngram_size: usize,
    similarity_threshold: f64,
}

impl Default for MinHashDeduplicator {
    fn default() -> Self {
        Self::new()
    }
}

impl MinHashDeduplicator {
    pub fn new() -> Self {
        Self {
            num_hashes: 128,
            num_bands: 16,
            band_rows: 8,
            ngram_size: 5,
            similarity_threshold: 0.8,
        }
    }

    pub fn with_params(
        mut self,
        num_hashes: usize,
        num_bands: usize,
        band_rows: usize,
        ngram_size: usize,
    ) -> Self {
        // Invalid combinations (e.g. from tenant/user config) must not take
        // the process down: warn and keep the current parameters instead.
        if num_hashes != num_bands * band_rows || num_hashes == 0 || ngram_size == 0 {
            tracing::warn!(
                "Invalid dedup params (num_hashes={num_hashes}, num_bands={num_bands}, band_rows={band_rows}, ngram_size={ngram_size}); keeping defaults"
            );
            return self;
        }
        self.num_hashes = num_hashes;
        self.num_bands = num_bands;
        self.band_rows = band_rows;
        self.ngram_size = ngram_size;
        self
    }

    pub fn with_similarity_threshold(mut self, threshold: f64) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Compute the MinHash signature for a text.
    ///
    /// The signature is a vector of `num_hashes` values, each being the
    /// minimum hash value across all shingles for a particular hash function.
    pub fn compute_signature(&self, text: &str) -> Vec<u64> {
        let shingles = self.extract_shingles(text);
        if shingles.is_empty() {
            return vec![u64::MAX; self.num_hashes];
        }

        (0..self.num_hashes)
            .map(|seed| {
                shingles
                    .iter()
                    .map(|shingle| hash_with_seed(shingle, seed))
                    .min()
                    .unwrap_or(u64::MAX)
            })
            .collect()
    }

    /// Estimate the Jaccard similarity between two MinHash signatures.
    pub fn estimate_similarity(sig_a: &[u64], sig_b: &[u64]) -> f64 {
        assert_eq!(sig_a.len(), sig_b.len());
        let matching = sig_a
            .iter()
            .zip(sig_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        matching as f64 / sig_a.len() as f64
    }

    /// Find near-duplicate texts and return a deduplication result.
    ///
    /// Uses LSH banding: signatures are split into `num_bands` bands of
    /// `band_rows` rows each. Two texts are candidate duplicates if any band
    /// matches exactly. The actual similarity is then verified using the
    /// full signature comparison.
    pub fn find_duplicates(&self, texts: &[&str]) -> DedupResult {
        if texts.is_empty() {
            return DedupResult {
                unique_indices: Vec::new(),
                duplicate_groups: Vec::new(),
            };
        }
        if texts.len() == 1 {
            return DedupResult {
                unique_indices: vec![0],
                duplicate_groups: Vec::new(),
            };
        }

        // Step 1: Compute signatures
        let signatures: Vec<Vec<u64>> = texts
            .iter()
            .map(|text| self.compute_signature(text))
            .collect();

        // Step 2: LSH banding — find candidate pairs
        let mut candidate_pairs: HashSet<(usize, usize)> = HashSet::new();
        for band_idx in 0..self.num_bands {
            let start = band_idx * self.band_rows;
            let end = start + self.band_rows;
            let mut buckets: std::collections::HashMap<Vec<u64>, Vec<usize>> =
                std::collections::HashMap::new();

            for (idx, sig) in signatures.iter().enumerate() {
                let band = sig[start..end].to_vec();
                buckets.entry(band).or_default().push(idx);
            }

            for (_, indices) in &buckets {
                if indices.len() > 1 {
                    for i in 0..indices.len() {
                        for j in (i + 1)..indices.len() {
                            let pair = (indices[i].min(indices[j]), indices[i].max(indices[j]));
                            candidate_pairs.insert(pair);
                        }
                    }
                }
            }
        }

        // Step 3: Verify candidates with full signature similarity
        let mut duplicate_map: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new(); // idx -> representative idx
        let mut confirmed_dup_groups: Vec<Vec<usize>> = Vec::new();

        for (i, j) in &candidate_pairs {
            let sim = Self::estimate_similarity(&signatures[*i], &signatures[*j]);
            if sim >= self.similarity_threshold {
                let rep_i = duplicate_map.get(i).copied().unwrap_or(*i);
                let rep_j = duplicate_map.get(j).copied().unwrap_or(*j);

                if rep_i == rep_j {
                    continue; // Already in same group
                }

                // Merge: use the smaller index as representative
                let (rep, dup) = if rep_i <= rep_j {
                    (rep_i, rep_j)
                } else {
                    (rep_j, rep_i)
                };

                // Update all entries pointing to `dup` to point to `rep`
                for val in duplicate_map.values_mut() {
                    if *val == dup {
                        *val = rep;
                    }
                }
                duplicate_map.insert(dup, rep);
            }
        }

        // Build groups
        let mut groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (dup, rep) in &duplicate_map {
            groups.entry(*rep).or_default().push(*dup);
        }
        for (rep, mut dups) in groups {
            dups.push(rep);
            dups.sort();
            confirmed_dup_groups.push(dups);
        }
        confirmed_dup_groups.sort_by_key(|g| g[0]);

        // Unique = all indices not in any duplicate group (or the representative)
        let dup_set: HashSet<usize> = duplicate_map.keys().copied().collect();
        let unique_indices: Vec<usize> = (0..texts.len())
            .filter(|i| !dup_set.contains(i))
            .collect();

        DedupResult {
            unique_indices,
            duplicate_groups: confirmed_dup_groups,
        }
    }

    /// Check if a single text is a near-duplicate of any text in the reference set.
    pub fn is_duplicate_of(&self, text: &str, reference_texts: &[&str]) -> Option<usize> {
        let sig = self.compute_signature(text);
        for (idx, ref_text) in reference_texts.iter().enumerate() {
            let ref_sig = self.compute_signature(ref_text);
            let sim = Self::estimate_similarity(&sig, &ref_sig);
            if sim >= self.similarity_threshold {
                return Some(idx);
            }
        }
        None
    }

    /// Extract word-level shingles (n-grams) from text.
    fn extract_shingles(&self, text: &str) -> HashSet<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < self.ngram_size {
            // If text is shorter than ngram_size, use the whole text as one shingle
            let normalized = text.to_lowercase().trim().to_string();
            if normalized.is_empty() {
                return HashSet::new();
            }
            let mut set = HashSet::new();
            set.insert(normalized);
            return set;
        }
        words
            .windows(self.ngram_size)
            .map(|window| {
                window
                    .iter()
                    .map(|w| w.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }
}

/// Hash a shingle with a given seed using a simple linear combination.
/// This simulates multiple independent hash functions.
fn hash_with_seed(shingle: &str, seed: usize) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    shingle.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_texts_are_duplicates() {
        let dedup = MinHashDeduplicator::new();
        let texts = vec![
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog",
        ];
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let result = dedup.find_duplicates(&text_refs);
        assert!(!result.duplicate_groups.is_empty());
    }

    #[test]
    fn different_texts_are_unique() {
        let dedup = MinHashDeduplicator::new();
        let texts = vec![
            "the quick brown fox jumps over the lazy dog",
            "quantum computing uses qubits instead of classical bits",
            "the stock market crashed due to global economic factors",
        ];
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let result = dedup.find_duplicates(&text_refs);
        assert_eq!(result.unique_indices.len(), 3);
        assert!(result.duplicate_groups.is_empty());
    }

    #[test]
    fn near_duplicates_detected() {
        let dedup = MinHashDeduplicator::new().with_similarity_threshold(0.5);
        let texts = vec![
            "the quick brown fox jumps over the lazy dog in the park",
            "the quick brown fox jumps over the lazy dog in the garden",
            "completely unrelated text about quantum physics experiments",
        ];
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let result = dedup.find_duplicates(&text_refs);
        // The first two should be near-duplicates
        assert!(result.duplicate_groups.iter().any(|g| g.contains(&0) && g.contains(&1)));
    }

    #[test]
    fn signature_computation_deterministic() {
        let dedup = MinHashDeduplicator::new();
        let text = "the quick brown fox jumps over the lazy dog";
        let sig1 = dedup.compute_signature(text);
        let sig2 = dedup.compute_signature(text);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn similarity_of_identical_signatures() {
        let dedup = MinHashDeduplicator::new();
        let sig = dedup.compute_signature("some text content here");
        let sim = MinHashDeduplicator::estimate_similarity(&sig, &sig);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_text_handling() {
        let dedup = MinHashDeduplicator::new();
        let sig = dedup.compute_signature("");
        assert!(sig.iter().all(|&v| v == u64::MAX));
    }

    #[test]
    fn single_text_no_duplicates() {
        let dedup = MinHashDeduplicator::new();
        let texts = vec!["only one text here"];
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_ref()).collect();
        let result = dedup.find_duplicates(&text_refs);
        assert_eq!(result.unique_indices, vec![0]);
        assert!(result.duplicate_groups.is_empty());
    }
}
