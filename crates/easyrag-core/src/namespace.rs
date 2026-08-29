//! Storage namespace constants.
//! All namespace values should not be changed after data is inserted.

use once_cell::sync::OnceCell;

// KV Store namespaces
pub const KV_STORE_FULL_DOCS: &str = "full_docs";
pub const KV_STORE_TEXT_CHUNKS: &str = "text_chunks";
pub const KV_STORE_LLM_RESPONSE_CACHE: &str = "llm_response_cache";
pub const KV_STORE_FULL_ENTITIES: &str = "full_entities";
pub const KV_STORE_FULL_RELATIONS: &str = "full_relations";
pub const KV_STORE_ENTITY_CHUNKS: &str = "entity_chunks";
pub const KV_STORE_RELATION_CHUNKS: &str = "relation_chunks";

// Vector Store namespaces
pub const VECTOR_STORE_ENTITIES: &str = "entities";
pub const VECTOR_STORE_RELATIONSHIPS: &str = "relationships";
pub const VECTOR_STORE_CHUNKS: &str = "chunks";

// Graph Store namespaces
pub const GRAPH_STORE_CHUNK_ENTITY_RELATION: &str = "chunk_entity_relation";

// Doc Status namespace
pub const DOC_STATUS: &str = "doc_status";

/// Check if a namespace matches a base namespace (suffix match.
pub fn is_namespace(namespace: &str, base_namespace: &str) -> bool {
    namespace.ends_with(base_namespace)
}

/// Check if a namespace matches any of multiple base namespaces.
pub fn is_namespace_any(namespace: &str, base_namespaces: &[&str]) -> bool {
    base_namespaces.iter().any(|b| is_namespace(namespace, b))
}

/// Charset shared by workspace and biz_id: both are spliced into SQL
/// identifiers/filters, Cypher labels and filesystem paths, so the charset
/// is restricted up front to rule out injection and path traversal.
fn scope_name_pattern() -> &'static regex::Regex {
    static RE: OnceCell<regex::Regex> = OnceCell::new();
    RE.get_or_init(|| regex::Regex::new(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-\.]{0,63}$").expect("valid regex"))
}

/// Validate a workspace or biz_id name against the shared scope charset.
pub fn validate_scope_name(kind: &str, value: &str) -> crate::Result<()> {
    if scope_name_pattern().is_match(value) {
        Ok(())
    } else {
        Err(crate::CoreError::InvalidArgument(format!(
            "Invalid {kind} '{value}': must match ^[a-zA-Z0-9_-][a-zA-Z0-9_-.]{{0,63}}$"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_suffix_matching() {
        assert!(is_namespace("ws_full_docs", "full_docs"));
        assert!(is_namespace("full_docs", "full_docs"));
        assert!(!is_namespace("full_doc", "full_docs"));
    }

    #[test]
    fn namespace_any_matching() {
        assert!(is_namespace_any("ws_text_chunks", &[KV_STORE_FULL_DOCS, KV_STORE_TEXT_CHUNKS]));
        assert!(!is_namespace_any("ws_other", &[KV_STORE_FULL_DOCS, KV_STORE_TEXT_CHUNKS]));
    }

    #[test]
    fn scope_name_validation() {
        assert!(validate_scope_name("workspace", "ws-1").is_ok());
        assert!(validate_scope_name("biz_id", "biz_a").is_ok());
        assert!(validate_scope_name("workspace", "ws.1-x_").is_ok());
        // Injection / traversal attempts must be rejected
        assert!(validate_scope_name("workspace", "ws'; DROP TABLE").is_err());
        assert!(validate_scope_name("biz_id", "../etc").is_err());
        assert!(validate_scope_name("biz_id", "").is_err());
        assert!(validate_scope_name("workspace", "`label`").is_err());
    }
}
