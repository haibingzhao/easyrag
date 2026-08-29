//! Text utility functions.

use crate::util::tokenizer::Tokenizer;
use crate::CoreError;

/// Sentence-ending separators used by the hard-fallback splitter
/// (lookbehind `(?<=[。！？；.!?])` — implemented manually because the Rust
/// `regex` crate does not support lookaround).
const SENTENCE_TERMINATORS: &[char] = &['。', '！', '？', '；', '.', '!', '?'];

/// Sanitize text to ensure safe UTF-8 encoding by removing or replacing
/// problematic characters.
/// Handles: surrogate/non-characters, control characters, HTML unescape,
/// whitespace trimming.
pub fn sanitize_text_for_encoding(text: &str) -> String {
    sanitize_text_for_encoding_with(text, "")
}

pub fn sanitize_text_for_encoding_with(text: &str, replacement_char: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }

    let mut result = text.trim().to_string();
    if result.is_empty() {
        return result;
    }

    // 1. HTML unescape (basic: &amp; &lt; &gt; &quot; &#39; &apos; &nbsp;)
    result = unescape_html(&result);

    // 2. Remove surrogates and non-characters.
    // Rust strings are valid UTF-8 so lone surrogates cannot occur;
    // only the non-characters U+FFFE / U+FFFF need filtering.
    // 3. Remove control characters but preserve common whitespace (\t, \n, \r).
    result = result
        .chars()
        .filter_map(|c| {
            if c == '\u{FFFE}'
                || c == '\u{FFFF}'
                || is_removed_control_char(c)
            {
                if replacement_char.is_empty() {
                    None
                } else {
                    // Map to the first char of the replacement (callers pass ""
                    // or a single char in practice).
                    Some(replacement_char.chars().next().unwrap_or('\u{FFFD}'))
                }
            } else {
                Some(c)
            }
        })
        .collect();

    result.trim().to_string()
}

/// Control chars removed by the control char pattern:
/// [\u0000-\u0008 \u000B \u000C \u000E-\u001F \u007F]
fn is_removed_control_char(c: char) -> bool {
    matches!(c as u32,
        0x0000..=0x0008 | 0x000B | 0x000C | 0x000E..=0x001F | 0x007F)
}

/// Remove control/separator chars and surrogates while preserving text.
/// Does NOT html-unescape or trim the result.
pub fn strip_control_characters(text: &str) -> String {
    strip_control_characters_with(text, "")
}

pub fn strip_control_characters_with(text: &str, replacement_char: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    text.chars()
        .filter_map(|c| {
            if c == '\u{FFFE}' || c == '\u{FFFF}' || is_removed_control_char(c) {
                if replacement_char.is_empty() {
                    None
                } else {
                    Some(replacement_char.chars().next().unwrap_or('\u{FFFD}'))
                }
            } else {
                Some(c)
            }
        })
        .collect()
}

/// Split a string by multiple markers (each marker escaped as a literal).
pub fn split_string_by_multi_markers(content: &str, markers: &[&str]) -> Vec<String> {
    if markers.is_empty() {
        return vec![content.to_string()];
    }
    let pattern = markers
        .iter()
        .map(|m| regex::escape(m))
        .collect::<Vec<_>>()
        .join("|");
    let re = regex::Regex::new(&pattern).expect("escaped markers must compile");
    re.split(content)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Truncate a list of data by token size.
/// Returns the items that fit within `max_token_size`.
pub fn truncate_list_by_token_size<T, F>(
    list_data: &[T],
    key: F,
    max_token_size: usize,
    tokenizer: &Tokenizer,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> String,
{
    if max_token_size == 0 {
        return Vec::new();
    }
    let mut tokens = 0usize;
    for (i, item) in list_data.iter().enumerate() {
        tokens += tokenizer.encode(&key(item)).len();
        if tokens > max_token_size {
            return list_data[..i].to_vec();
        }
    }
    list_data.to_vec()
}

/// Split text into sentence/paragraph-like units for fallback chunking.
/// Equivalent to `split("\n\n")` then `split("(?<=[。！？；.!?])")`.
pub fn split_text_units_for_hard_fallback(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut units = Vec::new();
    for para in text.split("\n\n") {
        let p = para.trim();
        if p.is_empty() {
            continue;
        }
        // Split after sentence terminators (manual lookbehind).
        let mut current = String::new();
        for c in p.chars() {
            current.push(c);
            if SENTENCE_TERMINATORS.contains(&c) {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    units.push(s);
                }
                current.clear();
            }
        }
        let tail = current.trim().to_string();
        if !tail.is_empty() {
            units.push(tail);
        }
    }
    if units.is_empty() {
        vec![text.to_string()]
    } else {
        units
    }
}

/// Split text by token limit with sentence-first, token-window fallback.
pub fn split_text_by_token_limit(
    text: &str,
    tokenizer: &Tokenizer,
    max_tokens: usize,
) -> Vec<String> {
    if text.is_empty() || max_tokens == 0 {
        return Vec::new();
    }

    let units = split_text_units_for_hard_fallback(text);
    let mut chunks: Vec<String> = Vec::new();
    let mut current_chunk = String::new();
    let mut current_tokens = 0usize;

    for unit in units {
        let unit_tokens = tokenizer.encode(&unit).len();
        if current_tokens + unit_tokens > max_tokens && !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
            current_tokens = 0;
        }
        if unit_tokens > max_tokens {
            // Single unit exceeds limit: force-split by token window.
            let tokens = tokenizer.encode(&unit);
            let mut start = 0usize;
            while start < tokens.len() {
                let end = (start + max_tokens).min(tokens.len());
                chunks.push(
                    tokenizer
                        .decode(&tokens[start..end])
                        .unwrap_or_default(),
                );
                start = end;
            }
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push(' ');
            }
            current_chunk.push_str(&unit);
            current_tokens += unit_tokens;
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }
    chunks
}

/// Check if a string matches a float regex pattern.
pub fn is_float_regex(value: &str) -> bool {
    static RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"^[-+]?[0-9]*\.?[0-9]+$").unwrap());
    RE.is_match(value)
}

/// Pack alternating user/assistant messages for OpenAI-style chat format.
pub fn pack_user_ass_to_openai_messages(args: &[&str]) -> Vec<crate::JsonMap> {
    let roles = ["user", "assistant"];
    args.iter()
        .enumerate()
        .map(|(i, content)| {
            let mut map = crate::JsonMap::new();
            map.insert("role".to_string(), serde_json::Value::String(roles[i % 2].to_string()));
            map.insert(
                "content".to_string(),
                serde_json::Value::String((*content).to_string()),
            );
            map
        })
        .collect()
}

/// Cosine similarity between two float arrays.
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> Result<f32, CoreError> {
    if v1.len() != v2.len() {
        return Err(CoreError::InvalidArgument(
            "Vectors must have the same dimension".to_string(),
        ));
    }
    let mut dot_product = 0f32;
    let mut norm1 = 0f32;
    let mut norm2 = 0f32;
    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    let denom = norm1.sqrt() * norm2.sqrt();
    if denom == 0f32 {
        return Ok(0f32);
    }
    Ok(dot_product / denom)
}

/// Compute cosine distances (1 - similarity) between adjacent embedding vectors.
pub fn compute_adjacent_cosine_distances(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, CoreError> {
    let mut distances = Vec::new();
    for i in 0..embeddings.len().saturating_sub(1) {
        distances.push(1.0 - cosine_similarity(&embeddings[i], &embeddings[i + 1])?);
    }
    Ok(distances)
}

/// Compute a percentile-based effective distance threshold from adjacent
/// distances. Falls back to 0.5 when no distances are available.
pub fn compute_effective_threshold(distances: &[f32], similarity_threshold: f32) -> f32 {
    if distances.is_empty() {
        return 0.5;
    }
    let mut sorted = distances.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let percentile_idx =
        ((sorted.len() as f32) * (1.0 - similarity_threshold)) as usize;
    let percentile_idx = percentile_idx.min(sorted.len() - 1);
    sorted[percentile_idx].max(0.1)
}

/// Basic HTML entity unescape (replaces common HTML entities).
fn unescape_html(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
}

/// Clean and validate text before sending to LLM for entity extraction.
/// Removes control characters, normalizes whitespace, validates minimum
/// length. Returns `None` if text is too short or empty after cleaning.
pub fn clean_text_for_extraction(text: &str, min_length: Option<usize>) -> Option<String> {
    let min_length = min_length.unwrap_or(10);
    // 1. Sanitize encoding
    let mut cleaned = sanitize_text_for_encoding(text);

    // 2. Normalize excessive whitespace (3+ newlines -> 2, 3+ spaces -> 1)
    static NEWLINE_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(r"\n{3,}").unwrap());
    static SPACE_RE: once_cell::sync::Lazy<regex::Regex> =
        once_cell::sync::Lazy::new(|| regex::Regex::new(" {3,}").unwrap());
    cleaned = NEWLINE_RE.replace_all(&cleaned, "\n\n").to_string();
    cleaned = SPACE_RE.replace_all(&cleaned, " ").to_string();

    // 3. Remove zero-width characters
    cleaned = cleaned
        .chars()
        .filter(|c| !matches!(*c as u32, 0x200B..=0x200D | 0xFEFF))
        .collect();

    // 4. Trim and validate
    let cleaned = cleaned.trim().to_string();
    if cleaned.len() < min_length {
        return None;
    }
    Some(cleaned)
}

#[cfg(test)]
mod tests {
    // Ported from TextUtilsTest.kt (21 cases).
    use super::*;
    use crate::util::tokenizer::TokenizerProvider;

    #[test]
    fn sanitize_text_for_encoding_normal_text_unchanged() {
        let input = "Hello, world!";
        assert_eq!("Hello, world!", sanitize_text_for_encoding(input));
    }

    #[test]
    fn sanitize_text_for_encoding_empty_text_returns_empty() {
        assert_eq!("", sanitize_text_for_encoding(""));
    }

    #[test]
    fn sanitize_text_for_encoding_html_entities_unescaped() {
        let input = "a &amp; b &lt; c &gt; d";
        let result = sanitize_text_for_encoding(input);
        assert_eq!("a & b < c > d", result);
    }

    #[test]
    fn sanitize_text_for_encoding_control_chars_removed() {
        let input = "hello\u{0000}world\u{0001}test";
        let result = sanitize_text_for_encoding(input);
        assert_eq!("helloworldtest", result);
    }

    #[test]
    fn sanitize_text_for_encoding_preserves_newlines_and_tabs() {
        let input = "line1\nline2\ttab";
        let result = sanitize_text_for_encoding(input);
        assert_eq!("line1\nline2\ttab", result);
    }

    #[test]
    fn split_string_by_multi_markers_single_marker() {
        let result = split_string_by_multi_markers("a.b.c", &["."]);
        assert_eq!(vec!["a", "b", "c"], result);
    }

    #[test]
    fn split_string_by_multi_markers_multiple_markers() {
        let result = split_string_by_multi_markers("a.b;c", &[".", ";"]);
        assert_eq!(vec!["a", "b", "c"], result);
    }

    #[test]
    fn split_string_by_multi_markers_empty_markers_returns_whole() {
        let result = split_string_by_multi_markers("hello", &[]);
        assert_eq!(vec!["hello"], result);
    }

    #[test]
    fn split_string_by_multi_markers_empty_content_returns_empty() {
        let result = split_string_by_multi_markers("", &["."]);
        assert!(result.is_empty());
    }

    #[test]
    fn cosine_similarity_identical_vectors_returns_one() {
        let v = [1f32, 2f32, 3f32];
        let sim = cosine_similarity(&v, &v).unwrap();
        assert!(sim > 0.999);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors_returns_zero() {
        let v1 = [1f32, 0f32];
        let v2 = [0f32, 1f32];
        let sim = cosine_similarity(&v1, &v2).unwrap();
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn cosine_similarity_opposite_vectors_returns_negative_one() {
        let v1 = [1f32, 0f32];
        let v2 = [-1f32, 0f32];
        let sim = cosine_similarity(&v1, &v2).unwrap();
        assert!(sim < -0.999);
    }

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let v1 = [0f32, 0f32];
        let v2 = [1f32, 1f32];
        let sim = cosine_similarity(&v1, &v2).unwrap();
        assert_eq!(0f32, sim);
    }

    #[test]
    fn truncate_list_by_token_size_within_limit_returns_all() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let data = vec!["short".to_string(), "text".to_string()];
        let result = truncate_list_by_token_size(&data, |s| s.clone(), 1000, &tokenizer);
        assert_eq!(data, result);
    }

    #[test]
    fn truncate_list_by_token_size_exceeds_limit_truncates() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let data: Vec<String> = (0..3).map(|_| "word ".repeat(100)).collect();
        let result = truncate_list_by_token_size(&data, |s| s.clone(), 50, &tokenizer);
        assert!(result.len() < data.len());
    }

    #[test]
    fn truncate_list_by_token_size_zero_limit_returns_empty() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let data = vec!["hello".to_string()];
        let result = truncate_list_by_token_size(&data, |s| s.clone(), 0, &tokenizer);
        assert!(result.is_empty());
    }

    #[test]
    fn clean_text_for_extraction_valid_text_returns_cleaned() {
        let input = "This is a valid text for extraction testing.";
        let result = clean_text_for_extraction(input, None);
        assert_eq!(Some(input.to_string()), result);
    }

    #[test]
    fn clean_text_for_extraction_too_short_returns_null() {
        let result = clean_text_for_extraction("short", None);
        assert!(result.is_none());
    }

    #[test]
    fn clean_text_for_extraction_excessive_whitespace_normalized() {
        let input = "Hello   world\n\n\n\nNew paragraph here for testing.";
        let result = clean_text_for_extraction(input, None);
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.contains("   "));
        assert!(!result.contains("\n\n\n"));
    }

    #[test]
    fn is_float_regex_valid_floats_returns_true() {
        assert!(is_float_regex("3.14"));
        assert!(is_float_regex("-0.5"));
        assert!(is_float_regex("42"));
    }

    #[test]
    fn is_float_regex_invalid_floats_returns_false() {
        assert!(!is_float_regex("abc"));
        assert!(!is_float_regex(""));
    }
}
