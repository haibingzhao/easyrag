//! In-memory storage implementations for development and testing.

pub mod doc_status;
pub mod graph;
pub mod kv;
mod kv_table;
pub mod tenant_config;
pub mod user;
pub mod vector;

pub use doc_status::InMemoryDocStatusStorage;
pub use graph::InMemoryGraphStorage;
pub use kv::InMemoryKVStorage;
pub use tenant_config::InMemoryTenantConfigStorage;
pub use user::InMemoryUserStorage;
pub use vector::InMemoryVectorStorage;

/// Convert a serde_json value to its display form
/// (strings unwrapped, null becomes empty string).
pub(crate) fn value_to_display_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Stringify all property values.
pub(crate) fn stringify_props(props: &easyrag_core::JsonMap) -> easyrag_core::JsonMap {
    props
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(value_to_display_string(v))))
        .collect()
}
