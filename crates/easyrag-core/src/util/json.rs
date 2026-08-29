//! JSON utility functions including lenient parsing and repair
//!
//! ALLOW_SINGLE_QUOTES / ALLOW_TRAILING_COMMA. serde_json has no equivalent
//! flags, so [`normalize_lenient`] rewrites those constructs before parsing.

use crate::JsonValue;
use once_cell::sync::Lazy;
use regex::Regex;

// LaTeX escape damage patterns
static FORMFEED_LATEX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new("\u{000C}(?=[A-Za-z])").unwrap());
static BACKSPACE_LATEX_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new("\u{0008}(?=[A-Za-z])").unwrap());
static THINK_TAG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<think>[\s\S]*?</think>").unwrap());
static TRAILING_COMMA_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r",\s*([}\]])").unwrap());
static MD_JSON_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)```json\s*\n?([\s\S]*?)\n?```").unwrap());
static MD_ANY_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```\s*\n?([\s\S]*?)\n?```").unwrap());

/// Parse JSON string with multi-strategy repair capabilities.
/// Strategy order:
/// 1. Standard parsing (lenient: single quotes / unquoted keys / trailing comma)
/// 2. Strip thinking tags + markdown extraction
/// 3. Quote/control char repair
/// 4. Regex extraction of JSON object/array
pub fn parse_json(text: &str) -> Option<JsonValue> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Strategy 1: Standard parsing
    if let Some(v) = parse_lenient(trimmed) {
        return Some(v);
    }

    // Strategy 2: Strip thinking tags and extract from markdown
    let stripped = strip_thinking_tags(trimmed);
    if let Some(extracted) = extract_json_from_markdown(&stripped) {
        if let Some(v) = parse_lenient(&extracted) {
            return Some(v);
        }
    }

    // Strategy 3: Apply repairs and retry
    let repaired = repair_json(&stripped);
    if let Some(v) = parse_lenient(&repaired) {
        return Some(v);
    }

    // Strategy 4: Regex extraction of JSON object or array
    if let Some(regex_extracted) = extract_json_by_regex(&stripped) {
        if let Some(v) = parse_lenient(&regex_extracted) {
            return Some(v);
        }
        // Try repair on extracted content
        let repaired_extracted = repair_json(&regex_extracted);
        if let Some(v) = parse_lenient(&repaired_extracted) {
            return Some(v);
        }
    }

    tracing::warn!("Failed to parse JSON after all 4 strategies");
    None
}

/// Parse with serde_json first, then retry after lenient normalization
/// (single-quoted strings, unquoted property names).
fn parse_lenient(text: &str) -> Option<JsonValue> {
    if let Ok(v) = serde_json::from_str(text) {
        return Some(v);
    }
    let normalized = normalize_lenient(text);
    serde_json::from_str(&normalized).ok()
}

/// Rewrite Jackson-lenient constructs into strict JSON:
/// - single-quoted strings become double-quoted
/// - unquoted property names become quoted
fn normalize_lenient(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(text.len() + 16);
    let mut i = 0usize;

    while i < n {
        let ch = chars[i];
        match ch {
            '"' => {
                // Copy double-quoted string verbatim.
                out.push(ch);
                i += 1;
                while i < n {
                    let c = chars[i];
                    out.push(c);
                    if c == '\\' && i + 1 < n {
                        out.push(chars[i + 1]);
                        i += 2;
                        continue;
                    }
                    i += 1;
                    if c == '"' {
                        break;
                    }
                }
            }
            '\'' => {
                // Convert single-quoted string to double-quoted.
                out.push('"');
                i += 1;
                while i < n {
                    let c = chars[i];
                    if c == '\\' && i + 1 < n {
                        let nxt = chars[i + 1];
                        if nxt == '\'' {
                            out.push('\'');
                            i += 2;
                            continue;
                        }
                        out.push(c);
                        out.push(nxt);
                        i += 2;
                        continue;
                    }
                    if c == '\'' {
                        i += 1;
                        break;
                    }
                    if c == '"' {
                        out.push('\\');
                        out.push('"');
                        i += 1;
                        continue;
                    }
                    out.push(c);
                    i += 1;
                }
                out.push('"');
            }
            _ => {
                // Possible unquoted property name after '{' or ','.
                if ch.is_ascii_alphabetic() || ch == '_' || ch == '$' {
                    let last = out.chars().rev().find(|c| !c.is_whitespace());
                    if matches!(last, Some('{') | Some(',')) {
                        let start = i;
                        while i < n
                            && (chars[i].is_ascii_alphanumeric()
                                || chars[i] == '_'
                                || chars[i] == '$')
                        {
                            i += 1;
                        }
                        let ident: String = chars[start..i].iter().collect();
                        let mut j = i;
                        while j < n && chars[j].is_whitespace() {
                            j += 1;
                        }
                        if j < n && chars[j] == ':' {
                            out.push('"');
                            out.push_str(&ident);
                            out.push('"');
                        } else {
                            out.push_str(&ident);
                        }
                        continue;
                    }
                }
                out.push(ch);
                i += 1;
            }
        }
    }
    out
}

/// Strip thinking/reasoning tags from LLM output.
/// Handles DeepSeek-R1 `<think>...</think>` and Qwen-think patterns.
pub fn strip_thinking_tags(text: &str) -> String {
    // Remove <think>...</think> blocks (applied twice, mirroring the source).
    let once = THINK_TAG_RE.replace_all(text, "");
    let twice = THINK_TAG_RE.replace_all(&once, "");
    twice.trim().to_string()
}

/// Parse JSON string into a specific typed value.
pub fn parse_json_value<T: serde::de::DeserializeOwned>(text: &str) -> Option<T> {
    let node = parse_json(text)?;
    match serde_json::from_value(node) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(
                "Failed to convert JSON node to {}: {}",
                std::any::type_name::<T>(),
                e
            );
            None
        }
    }
}

/// Extract JSON content from markdown code fences.
pub fn extract_json_from_markdown(text: &str) -> Option<String> {
    for re in [&*MD_JSON_RE, &*MD_ANY_RE] {
        if let Some(caps) = re.captures(text) {
            let content = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            if content.starts_with('{') || content.starts_with('[') {
                return Some(content.to_string());
            }
        }
    }
    None
}

/// Extract JSON object or array using brace/bracket matching.
/// Finds the first { or [ and matches to its closing counterpart.
pub fn extract_json_by_regex(text: &str) -> Option<String> {
    let obj_start = text.find('{');
    let arr_start = text.find('[');

    let start = match (obj_start, arr_start) {
        (Some(o), Some(a)) => o.min(a),
        (Some(o), None) => o,
        (None, Some(a)) => a,
        (None, None) => return None,
    };

    let chars: Vec<char> = text.chars().collect();
    // Convert byte index to char index.
    let start_char = text[..start].chars().count();
    let open_char = chars[start_char];
    let close_char = if open_char == '{' { '}' } else { ']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, ch) in chars[start_char..].iter().enumerate() {
        let i = start_char + offset;
        if escaped {
            escaped = false;
            continue;
        }
        if *ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if *ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string && *ch == open_char {
            depth += 1;
        }
        if !in_string && *ch == close_char {
            depth -= 1;
            if depth == 0 {
                return Some(chars[start_char..=i].iter().collect());
            }
        }
    }

    // If unbalanced, return from start to end (repair will handle it).
    if depth > 0 {
        Some(chars[start_char..].iter().collect())
    } else {
        None
    }
}

/// Repair common JSON issues from LLM output.
pub fn repair_json(text: &str) -> String {
    let mut result = text.trim().to_string();

    // Remove markdown code fences if present.
    if let Some(extracted) = extract_json_from_markdown(&result) {
        result = extracted;
    }

    // Remove trailing commas before } or ]
    result = TRAILING_COMMA_RE.replace_all(&result, "$1").to_string();

    // Fix unescaped control characters in strings (basic repair).
    result = repair_control_chars_in_strings(&result);

    // Try to balance braces/brackets.
    result = balance_braces(&result);

    result
}

/// Repair LaTeX backslashes destroyed by JSON escape decoding.
/// Restores formfeed + letter -> `\f` (for \frac, \forall, etc.)
/// Restores backspace + letter -> `\b` (for \beta, \bar, etc.)
pub fn repair_vlm_json_escape_damage(text: &str) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let repaired = FORMFEED_LATEX_RE.replace_all(text, "\\f");
    BACKSPACE_LATEX_RE
        .replace_all(&repaired, "\\b")
        .to_string()
}

/// Apply LaTeX escape repair to every string in a parsed JSON structure.
pub fn repair_vlm_json_escape_damage_nested(node: JsonValue) -> JsonValue {
    match node {
        JsonValue::String(s) => JsonValue::String(repair_vlm_json_escape_damage(&s)),
        JsonValue::Object(map) => JsonValue::Object(
            map.into_iter()
                .map(|(k, v)| (k, repair_vlm_json_escape_damage_nested(v)))
                .collect(),
        ),
        JsonValue::Array(arr) => {
            JsonValue::Array(arr.into_iter().map(repair_vlm_json_escape_damage_nested).collect())
        }
        other => other,
    }
}

/// Serialize a value to JSON string.
pub fn to_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

/// Serialize a value to pretty-printed JSON string.
pub fn to_pretty_json<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string())
}

fn repair_control_chars_in_strings(text: &str) -> String {
    // Replace literal newlines/tabs inside JSON string values.
    let mut sb = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    for ch in text.chars() {
        if escaped {
            sb.push(ch);
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            sb.push(ch);
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            sb.push(ch);
            continue;
        }
        if in_string && ch == '\n' {
            sb.push_str("\\n");
            continue;
        }
        if in_string && ch == '\r' {
            sb.push_str("\\r");
            continue;
        }
        if in_string && ch == '\t' {
            sb.push_str("\\t");
            continue;
        }
        sb.push(ch);
    }
    sb
}

fn balance_braces(text: &str) -> String {
    let mut braces = 0i32;
    let mut brackets = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    for ch in text.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && in_string {
            escaped = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if !in_string {
            match ch {
                '{' => braces += 1,
                '}' => braces -= 1,
                '[' => brackets += 1,
                ']' => brackets -= 1,
                _ => {}
            }
        }
    }

    let mut sb = String::from(text);
    while braces > 0 {
        sb.push('}');
        braces -= 1;
    }
    while brackets > 0 {
        sb.push(']');
        brackets -= 1;
    }
    sb
}

#[cfg(test)]
mod tests {
    // Ported from JsonUtilsTest.kt (19 cases).
    use super::*;

    #[test]
    fn parse_json_valid_object_returns_node() {
        let result = parse_json(r#"{"key": "value"}"#);
        assert!(result.is_some());
        assert_eq!("value", result.unwrap()["key"].as_str().unwrap());
    }

    #[test]
    fn parse_json_valid_array_returns_node() {
        let result = parse_json("[1, 2, 3]");
        assert!(result.is_some());
        let v = result.unwrap();
        assert!(v.is_array());
        assert_eq!(3, v.as_array().unwrap().len());
    }

    #[test]
    fn parse_json_empty_string_returns_null() {
        assert!(parse_json("").is_none());
        assert!(parse_json("   ").is_none());
    }

    #[test]
    fn parse_json_invalid_json_returns_null() {
        assert!(parse_json("not json at all").is_none());
    }

    #[test]
    fn parse_json_json_in_markdown_extracted() {
        let input = "Here is the result:\n```json\n{\"name\": \"test\"}\n```\nDone.";
        let result = parse_json(input);
        assert!(result.is_some());
        assert_eq!("test", result.unwrap()["name"].as_str().unwrap());
    }

    #[test]
    fn parse_json_trailing_comma_repaired() {
        let input = r#"{"a": 1, "b": 2,}"#;
        let result = parse_json(input);
        assert!(result.is_some());
        assert_eq!(1, result.unwrap()["a"].as_i64().unwrap());
    }

    #[test]
    fn parse_json_unbalanced_braces_repaired() {
        let input = r#"{"key": "value""#;
        let result = parse_json(input);
        assert!(result.is_some());
        assert_eq!("value", result.unwrap()["key"].as_str().unwrap());
    }

    #[test]
    fn strip_thinking_tags_with_think_tag_stripped() {
        let input = "<think>reasoning here</think>actual answer";
        let result = strip_thinking_tags(input);
        assert_eq!("actual answer", result);
    }

    #[test]
    fn strip_thinking_tags_no_think_tag_unchanged() {
        let input = "just normal text";
        let result = strip_thinking_tags(input);
        assert_eq!("just normal text", result);
    }

    #[test]
    fn strip_thinking_tags_multiple_think_tags_all_stripped() {
        let input = "<think>first</think>middle<think>second</think>end";
        let result = strip_thinking_tags(input);
        assert_eq!("middleend", result.trim());
    }

    #[test]
    fn extract_json_from_markdown_json_block_extracted() {
        let input = "```json\n{\"hello\": \"world\"}\n```";
        let result = extract_json_from_markdown(input);
        assert!(result.is_some());
        assert!(result.unwrap().contains("hello"));
    }

    #[test]
    fn extract_json_from_markdown_no_block_returns_null() {
        assert!(extract_json_from_markdown("no json here").is_none());
    }

    #[test]
    fn extract_json_by_regex_embedded_object_extracted() {
        let input = r#"prefix {"key": "val"} suffix"#;
        let result = extract_json_by_regex(input);
        assert!(result.is_some());
        assert_eq!(r#"{"key": "val"}"#, result.unwrap());
    }

    #[test]
    fn extract_json_by_regex_embedded_array_extracted() {
        let input = "text [1, 2, 3] more";
        let result = extract_json_by_regex(input);
        assert!(result.is_some());
        assert_eq!("[1, 2, 3]", result.unwrap());
    }

    #[test]
    fn extract_json_by_regex_no_json_returns_null() {
        assert!(extract_json_by_regex("no json").is_none());
    }

    #[test]
    fn repair_json_trailing_commas_removed() {
        let input = r#"{"a": 1,}"#;
        let result = repair_json(input);
        assert!(!result.contains(",}"));
    }

    #[test]
    fn repair_json_unbalanced_braces_balanced() {
        let input = r#"{"key": "value""#;
        let result = repair_json(input);
        // Should have closing brace appended
        assert!(result.ends_with('}'));
    }

    #[test]
    fn to_json_map_serializes() {
        let map: crate::JsonMap = serde_json::from_str(r#"{"a": 1}"#).unwrap();
        let result = to_json(&map);
        assert!(result.contains("\"a\""));
        assert!(result.contains('1'));
    }

    #[test]
    fn parse_json_with_think_tag_and_json_parsed() {
        let input = "<think>analyzing...</think>{\"result\": 42}";
        let result = parse_json(input);
        assert!(result.is_some());
        assert_eq!(42, result.unwrap()["result"].as_i64().unwrap());
    }
}
