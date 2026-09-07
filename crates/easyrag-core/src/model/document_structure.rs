//! Hierarchical document structure tree model
//! Represents the table-of-contents-like structure of a document, used for
//! reasoning-based retrieval (PageIndex approach).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionNode {
    pub title: String,
    pub level: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// character offset in original document
    pub content_start: usize,
    pub content_end: usize,
    #[serde(default)]
    pub children: Vec<SectionNode>,
}

/// Separator joining section titles into a human-readable path
/// (e.g. "Chapter 1 > 1.1 Introduction > Background").
pub const SECTION_PATH_SEP: &str = " > ";

impl SectionNode {
    /// Returns the full path from root to this node
    /// (e.g. "Chapter 1 > 1.1 Introduction > Background").
    pub fn path_with(&self, ancestors: &[String]) -> String {
        let mut parts: Vec<&str> = ancestors.iter().map(|s| s.as_str()).collect();
        parts.push(&self.title);
        parts.join(SECTION_PATH_SEP)
    }
}

/// Extract the `metadata.section_path` array from a chunk map and join it into a
/// `" > "` path string. Returns `None` when the chunk carries no section path.
///
/// Shared by the query/ingest paths that previously inlined the identical
/// five-line extraction; uses the same separator as [`SectionNode::path_with`].
pub fn section_path_of(chunk: &crate::JsonMap) -> Option<String> {
    chunk
        .get("metadata")
        .and_then(|m| m.get("section_path"))
        .and_then(|p| p.as_array())
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(SECTION_PATH_SEP)
        })
}

/// Document-level structure index stored in KV storage.
/// Built during ingestion and used for tree-search reasoning at query time.
///
/// Serialized as camelCase.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentStructureIndex {
    pub doc_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_description: Option<String>,
    pub sections: Vec<SectionNode>,
    pub total_chars: usize,
    #[serde(default = "default_version")]
    pub version: u32,
}

fn default_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_index_serializes_camel_case_like_jackson() {
        let index = DocumentStructureIndex {
            doc_id: "doc-1".to_string(),
            doc_title: Some("Title".to_string()),
            doc_description: None,
            sections: vec![SectionNode {
                title: "Intro".to_string(),
                level: 1,
                summary: None,
                content_start: 0,
                content_end: 100,
                children: vec![],
            }],
            total_chars: 100,
            version: 1,
        };
        let json = serde_json::to_string(&index).unwrap();
        // Property names must be preserved for storage compatibility.
        assert!(json.contains("\"docId\":\"doc-1\""));
        assert!(json.contains("\"docTitle\":\"Title\""));
        assert!(json.contains("\"totalChars\":100"));
        assert!(json.contains("\"contentStart\":0"));
        assert!(json.contains("\"contentEnd\":100"));
        assert!(!json.contains("doc_id"));
    }

    #[test]
    fn structure_index_round_trips_jackson_payload() {
        // Sample JSON input.
        let sample_json = r#"{"docId":"doc-1","docTitle":"Title","sections":[{"title":"S1","level":1,"contentStart":0,"contentEnd":5}],"totalChars":5,"version":1}"#;
        let index: DocumentStructureIndex = serde_json::from_str(sample_json).unwrap();
        assert_eq!("doc-1", index.doc_id);
        assert_eq!(Some("Title".to_string()), index.doc_title);
        assert_eq!(1, index.sections.len());
        assert_eq!(5, index.sections[0].content_end);
        assert_eq!(5, index.total_chars);
    }
}
