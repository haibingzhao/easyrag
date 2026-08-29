//! Query result models.

use crate::JsonMap;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Text chunk schema representing a chunk of text from a document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextChunk {
    pub tokens: usize,
    pub content: String,
    pub full_doc_id: String,
    pub chunk_order_index: usize,
}

/// Unified query result data structure for all query modes.
pub struct QueryResult {
    pub content: Option<String>,
    pub response_iterator: Option<BoxStream<'static, String>>,
    pub raw_data: Option<JsonMap>,
    pub is_streaming: bool,
}

impl QueryResult {
    pub fn with_content(content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            response_iterator: None,
            raw_data: None,
            is_streaming: false,
        }
    }

    pub fn with_stream(stream: BoxStream<'static, String>, raw_data: Option<JsonMap>) -> Self {
        Self {
            content: None,
            response_iterator: Some(stream),
            raw_data,
            is_streaming: true,
        }
    }

    /// Convenient accessor to extract reference list from raw_data.
    pub fn reference_list(&self) -> Vec<crate::JsonValue> {
        extract_references(self.raw_data.as_ref())
    }

    /// Convenient accessor to extract metadata from raw_data.
    pub fn metadata(&self) -> JsonMap {
        self.raw_data
            .as_ref()
            .and_then(|d| d.get("metadata"))
            .and_then(|m| m.as_object())
            .cloned()
            .unwrap_or_default()
    }
}

/// Unified query context result data structure.
#[derive(Clone, Debug)]
pub struct QueryContextResult {
    pub context: String,
    pub raw_data: JsonMap,
}

impl QueryContextResult {
    /// Convenient accessor to extract reference list from raw_data.
    pub fn reference_list(&self) -> Vec<crate::JsonValue> {
        extract_references(Some(&self.raw_data))
    }
}

fn extract_references(raw_data: Option<&JsonMap>) -> Vec<crate::JsonValue> {
    raw_data
        .and_then(|d| d.get("data"))
        .and_then(|d| d.get("references"))
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default()
}
