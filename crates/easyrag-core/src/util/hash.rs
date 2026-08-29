//! Hash utility functions.

use md5::{Digest, Md5};
use std::fmt::Display;

/// Compute MD5 hash for the given arguments. Arguments are joined with a
/// NUL separator so argument-boundary shifts cannot collide
/// (e.g. ["ab","c"] vs ["a","bc"]).
pub fn compute_args_hash<I, T>(args: I) -> String
where
    I: IntoIterator<Item = T>,
    T: Display,
{
    let mut joined = String::new();
    for arg in args {
        if !joined.is_empty() {
            joined.push('\0');
        }
        joined.push_str(&arg.to_string());
    }
    md5_hex(joined.as_bytes())
}

/// Compute MD5 hash with nullable arguments (`None` contributes an empty
/// string).
pub fn compute_args_hash_nullable(args: &[Option<&str>]) -> String {
    let mut joined = String::new();
    let mut first = true;
    for arg in args {
        if !first {
            joined.push('\0');
        }
        first = false;
        if let Some(a) = arg {
            joined.push_str(a);
        }
    }
    md5_hex(joined.as_bytes())
}

/// Compute a unique ID for a given content string: `prefix + md5(content)`.
pub fn compute_mdhash_id(content: &str, prefix: &str) -> String {
    format!("{}{}", prefix, compute_args_hash([content]))
}

/// Return candidate relation VDB IDs for an undirected edge.
/// The normalized (lexicographically sorted endpoints) ID comes first for all
/// new writes; the reverse concatenation ID is appended when different.
pub fn make_relation_vdb_ids(src_entity: &str, tgt_entity: &str) -> Vec<String> {
    let mut endpoints = [src_entity.to_string(), tgt_entity.to_string()];
    endpoints.sort();
    let (normalized_src, normalized_tgt) = (&endpoints[0], &endpoints[1]);

    let mut relation_ids = vec![compute_mdhash_id(
        &format!("{}{}", normalized_src, normalized_tgt),
        "rel-",
    )];
    let reverse_relation_id = compute_mdhash_id(
        &format!("{}{}", normalized_tgt, normalized_src),
        "rel-",
    );
    if !relation_ids.contains(&reverse_relation_id) {
        relation_ids.push(reverse_relation_id);
    }
    relation_ids
}

/// Generate a flattened cache key in the format `{mode}:{cache_type}:{hash}`.
pub fn generate_cache_key(mode: &str, cache_type: &str, hash_value: &str) -> String {
    format!("{}:{}:{}", mode, cache_type, hash_value)
}

/// Parse a flattened cache key back into its components.
/// Returns `(mode, cache_type, hash)` or `None` if invalid format.
pub fn parse_cache_key(cache_key: &str) -> Option<(String, String, String)> {
    let mut parts = cache_key.splitn(3, ':');
    let mode = parts.next()?;
    let cache_type = parts.next()?;
    let hash = parts.next()?;
    Some((mode.to_string(), cache_type.to_string(), hash.to_string()))
}

fn md5_hex(bytes: &[u8]) -> String {
    let digest = Md5::digest(bytes);
    hex::encode(digest)
}

#[cfg(test)]
mod tests {
    // Ported from HashUtilsTest.kt (13 cases).
    use super::*;

    #[test]
    fn compute_args_hash_same_input_returns_same_hash() {
        let hash1 = compute_args_hash(["hello", "world"]);
        let hash2 = compute_args_hash(["hello", "world"]);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn compute_args_hash_different_input_returns_different_hash() {
        let hash1 = compute_args_hash(["hello"]);
        let hash2 = compute_args_hash(["world"]);
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn compute_args_hash_returns_md5_format() {
        let hash = compute_args_hash(["test"]);
        // MD5 hex is 32 characters
        assert_eq!(32, hash.len());
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn compute_args_hash_null_args_handled_gracefully() {
        let hash = compute_args_hash_nullable(&[None, Some("test"), None]);
        assert_eq!(32, hash.len());
    }

    #[test]
    fn compute_mdhash_id_with_prefix_returns_prefixed_hash() {
        let id = compute_mdhash_id("content", "chunk-");
        assert!(id.starts_with("chunk-"));
        assert_eq!(32 + 6, id.len()); // "chunk-" + 32 hex chars
    }

    #[test]
    fn compute_mdhash_id_without_prefix_returns_plain_hash() {
        let id = compute_mdhash_id("content", "");
        assert_eq!(32, id.len());
    }

    #[test]
    fn compute_mdhash_id_same_content_returns_same_id() {
        let id1 = compute_mdhash_id("test content", "doc-");
        let id2 = compute_mdhash_id("test content", "doc-");
        assert_eq!(id1, id2);
    }

    #[test]
    fn make_relation_vdb_ids_returns_at_least_one_id() {
        let ids = make_relation_vdb_ids("entityA", "entityB");
        assert!(!ids.is_empty());
        assert!(ids[0].starts_with("rel-"));
    }

    #[test]
    fn make_relation_vdb_ids_undirected_same_result_regardless_of_order() {
        let ids1 = make_relation_vdb_ids("A", "B");
        let ids2 = make_relation_vdb_ids("B", "A");
        // First (normalized) ID should be the same
        assert_eq!(ids1[0], ids2[0]);
    }

    #[test]
    fn generate_cache_key_correct_format() {
        let key = generate_cache_key("mix", "query", "abc123");
        assert_eq!("mix:query:abc123", key);
    }

    #[test]
    fn parse_cache_key_valid_key_returns_components() {
        let result = parse_cache_key("mix:query:abc123");
        assert!(result.is_some());
        let (mode, cache_type, hash) = result.unwrap();
        assert_eq!("mix", mode);
        assert_eq!("query", cache_type);
        assert_eq!("abc123", hash);
    }

    #[test]
    fn parse_cache_key_invalid_key_returns_none() {
        let result = parse_cache_key("invalid");
        assert!(result.is_none());
    }

    #[test]
    fn generate_and_parse_cache_key_round_trip() {
        let key = generate_cache_key("local", "keywords", "hash42");
        let parsed = parse_cache_key(&key);
        assert!(parsed.is_some());
        let (mode, cache_type, hash) = parsed.unwrap();
        assert_eq!("local", mode);
        assert_eq!("keywords", cache_type);
        assert_eq!("hash42", hash);
    }
}
