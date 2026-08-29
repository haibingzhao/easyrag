//! Shared Markdown heading parser utility

use once_cell::sync::Lazy;
use regex::Regex;

/// A parsed Markdown heading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadingInfo {
    pub title: String,
    pub level: usize,
    /// Byte offset of the heading line start in the source text
    /// ( uses char offsets; Rust uses byte offsets — both are
    /// only consumed by same-runtime code).
    pub offset: usize,
}

static HEADING_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^(#{1,6})\s+(.+)$").expect("heading regex"));

/// Parse all Markdown headings from text, in document order.
pub fn parse_headings(text: &str) -> Vec<HeadingInfo> {
    HEADING_REGEX
        .captures_iter(text)
        .map(|caps| HeadingInfo {
            title: caps[2].trim().to_string(),
            level: caps[1].len(),
            offset: caps.get(0).unwrap().start(),
        })
        .collect()
}

/// Quick check whether text contains any Markdown headings.
pub fn has_headings(text: &str) -> bool {
    HEADING_REGEX.is_match(text)
}
