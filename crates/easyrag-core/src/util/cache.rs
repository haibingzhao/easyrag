//! Cache utility functions for LLM response caching

use crate::JsonMap;
use async_trait::async_trait;
use std::collections::HashMap;

/// Cache data payload.
#[derive(Clone, Debug)]
pub struct CacheData {
    pub args_hash: String,
    pub content: String,
    pub prompt: String,
    pub mode: String,
    pub cache_type: String,
    pub chunk_id: Option<String>,
    pub query_param: Option<JsonMap>,
}

impl CacheData {
    pub fn new(args_hash: impl Into<String>, content: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            args_hash: args_hash.into(),
            content: content.into(),
            prompt: prompt.into(),
            mode: "default".to_string(),
            cache_type: "query".to_string(),
            chunk_id: None,
            query_param: None,
        }
    }
}

/// Cache operations interface — decoupled from specific KV storage
/// implementation.
#[async_trait]
pub trait CacheStorage: Send + Sync {
    async fn get_by_id(&self, key: &str) -> Option<JsonMap>;
    async fn upsert(&self, data: HashMap<String, JsonMap>) -> crate::Result<()>;
    fn is_cache_enabled(&self, mode: &str) -> bool;
}

/// Generic cache handling function with flattened cache keys.
/// Returns `(content, create_time)` on cache hit, `None` on miss.
pub async fn handle_cache(
    hashing_kv: Option<&dyn CacheStorage>,
    args_hash: &str,
    mode: &str,
    cache_type: &str,
) -> Option<(String, i64)> {
    let hashing_kv = hashing_kv?;

    if !hashing_kv.is_cache_enabled(mode) {
        return None;
    }

    let flattened_key = crate::util::hash::generate_cache_key(mode, cache_type, args_hash);
    if let Some(cache_entry) = hashing_kv.get_by_id(&flattened_key).await {
        tracing::debug!("Flattened cache hit(key:{})", flattened_key);
        let content = cache_entry
            .get("return")
            .and_then(|v| v.as_str())?
            .to_string();
        let timestamp = cache_entry
            .get("create_time")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        return Some((content, timestamp));
    }

    tracing::debug!("Cache missed(mode:{} type:{})", mode, cache_type);
    None
}

/// Save data to cache using flattened key structure.
pub async fn save_to_cache(hashing_kv: Option<&dyn CacheStorage>, cache_data: &CacheData) {
    let Some(hashing_kv) = hashing_kv else {
        return;
    };
    if cache_data.content.is_empty() {
        return;
    }

    let flattened_key = crate::util::hash::generate_cache_key(
        &cache_data.mode,
        &cache_data.cache_type,
        &cache_data.args_hash,
    );

    // Check if we already have identical content cached.
    if let Some(existing_cache) = hashing_kv.get_by_id(&flattened_key).await {
        let existing_content = existing_cache
            .get("return")
            .and_then(|v| v.as_str());
        if existing_content == Some(cache_data.content.as_str()) {
            tracing::warn!(
                "Cache duplication detected for {}, skipping update",
                flattened_key
            );
            return;
        }
    }

    // Create cache entry with flattened structure.
    let mut cache_entry = JsonMap::new();
    cache_entry.insert(
        "return".to_string(),
        serde_json::Value::String(cache_data.content.clone()),
    );
    cache_entry.insert(
        "cache_type".to_string(),
        serde_json::Value::String(cache_data.cache_type.clone()),
    );
    cache_entry.insert(
        "original_prompt".to_string(),
        serde_json::Value::String(cache_data.prompt.clone()),
    );
    if let Some(chunk_id) = &cache_data.chunk_id {
        cache_entry.insert(
            "chunk_id".to_string(),
            serde_json::Value::String(chunk_id.clone()),
        );
    }
    if let Some(query_param) = &cache_data.query_param {
        cache_entry.insert(
            "queryparam".to_string(),
            serde_json::Value::Object(query_param.clone()),
        );
    }

    tracing::info!(" == LLM cache == saving: {}", flattened_key);
    let mut data = HashMap::new();
    data.insert(flattened_key, cache_entry);
    if let Err(e) = hashing_kv.upsert(data).await {
        tracing::warn!("Failed to save cache entry: {e}");
    }
}
