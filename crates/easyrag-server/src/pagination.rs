//! Pagination envelope — single source of truth for paginated list responses.
//!
//! The webui `PaginatedResponse` interface (`easyrag-webui/src/api/easyrag.ts`)
//! expects camelCase `pageSize` and the item key `documents`. The field order
//! below matches the wire bytes previously produced by the inline `json!` in
//! `list_documents` (serde_json `preserve_order` is enabled workspace-wide), so
//! routing through this struct is byte-identical on the wire.
//!
//! This replaces the former `util::paginated_response`, which emitted snake_case
//! `items`/`page_size`/`total_pages` and was never wired to a handler (dead
//! code). Enabling that version would have broken the frontend contract (A-10):
//! the webui reads `documents`/`pageSize`, not `items`/`page_size`.

use serde::Serialize;

/// Camel-case paginated list envelope aligned with webui `PaginatedResponse`.
///
/// Declaration order is fixed (`documents`, `total`, `page`, `pageSize`) to
/// match the preserved-insertion-order bytes of the previous `json!` literal.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Paginated<T> {
    pub documents: Vec<T>,
    pub total: usize,
    pub page: usize,
    pub page_size: usize,
}

impl<T> Paginated<T> {
    /// Build a pagination envelope.
    pub fn new(documents: Vec<T>, total: usize, page: usize, page_size: usize) -> Self {
        Self {
            documents,
            total,
            page,
            page_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The envelope must serialize byte-identically to the inline `json!` it
    /// replaced: camelCase `pageSize`, item key `documents`, and the exact
    /// field order (preserve_order is enabled workspace-wide).
    #[test]
    fn paginated_matches_inline_json_wire_bytes() {
        let documents = vec![json!({ "id": "d1" })];
        let envelope = Paginated::new(documents.clone(), 1usize, 1usize, 10usize);
        let via_struct = serde_json::to_string(&envelope).expect("serialize envelope");
        let via_literal = serde_json::to_string(&json!({
            "documents": documents,
            "total": 1,
            "page": 1,
            "pageSize": 10,
        }))
        .expect("serialize literal");
        assert_eq!(via_struct, via_literal);
        assert_eq!(
            via_struct,
            r#"{"documents":[{"id":"d1"}],"total":1,"page":1,"pageSize":10}"#
        );
    }

    /// Guard against the snake_case landmine reappearing: the serialized keys
    /// are exactly the four camelCase contract keys, never `items`/`page_size`/
    /// `total_pages`.
    #[test]
    fn paginated_uses_camel_case_contract_keys() {
        let value = serde_json::to_value(Paginated::new(Vec::<serde_json::Value>::new(), 0, 1, 50))
            .expect("serialize envelope");
        let obj = value.as_object().expect("object");
        let keys: Vec<&str> = obj.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["documents", "total", "page", "pageSize"]);
    }
}
