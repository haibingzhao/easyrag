//! Parse result cache.
//! Cache key = MD5(file_path + mtime + parser + parse_method + kwargs).

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use easyrag_core::storage::KVStorage;
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::{file_mtime_millis, kwargs_value_to_string};
use super::mineru::path_to_absolute;

/// Kwargs that influence parsing output and therefore the cache key.
const RELEVANT_KWARGS: &[&str] = &[
    "lang", "device", "start_page", "end_page", "formula", "table", "backend", "source",
];

/// Parse result cache using KVStorage.
pub struct ParseCache {
    kv_storage: Option<Arc<dyn KVStorage>>,
    /// Business line bound at construction; passed as the biz filter on
    /// every KV call.
    biz_id: String,
}

impl ParseCache {
    pub fn new(kv_storage: Option<Arc<dyn KVStorage>>, biz_id: String) -> Self {
        Self { kv_storage, biz_id }
    }

    /// Generate a cache key based on file path, modification time, and
    /// parsing configuration.
    pub fn generate_cache_key(
        &self,
        file: &Path,
        parser: &str,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> String {
        let mtime = file_mtime_millis(file);

        let mut config_map: BTreeMap<String, String> = BTreeMap::new();
        config_map.insert("file_path".to_string(), path_to_absolute(file));
        config_map.insert("mtime".to_string(), mtime.to_string());
        config_map.insert("parser".to_string(), parser.to_string());
        config_map.insert(
            "parse_method".to_string(),
            parse_method.unwrap_or("auto").to_string(),
        );

        // Add relevant kwargs.
        let relevant: HashSet<&str> = RELEVANT_KWARGS.iter().copied().collect();
        for (key, value) in kwargs.iter() {
            if relevant.contains(key.as_str()) {
                config_map.insert(key.clone(), kwargs_value_to_string(value));
            }
        }

        let config_str = config_map
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",");
        compute_mdhash_id(&config_str, "")
    }

    /// Get cached parse result if available and valid.
    /// Returns None on cache miss or invalid cache.
    pub async fn get_cached_result(
        &self,
        cache_key: &str,
        file: &Path,
        parser: &str,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> Option<ParseResult> {
        let kv_storage = self.kv_storage.as_ref()?;

        let cached_data = match kv_storage.get_by_id(cache_key, &self.biz_id).await {
            Ok(Some(data)) => data,
            Ok(None) => return None,
            Err(e) => {
                tracing::warn!("Error accessing parse cache: {e}");
                return None;
            }
        };

        // Validate file modification time.
        let current_mtime = file_mtime_millis(file);
        let cached_mtime = cached_data.get("mtime").and_then(|v| v.as_i64()).unwrap_or(0);
        if current_mtime != cached_mtime {
            tracing::debug!("Cache invalid - file modified: {}", cache_key);
            return None;
        }

        // Validate parsing configuration.
        let cached_config = cached_data
            .get("parse_config")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        let current_config = build_current_config(parser, parse_method, kwargs);
        if JsonValue::Object(cached_config) != JsonValue::Object(current_config) {
            tracing::debug!("Cache invalid - config changed: {}", cache_key);
            return None;
        }

        // Reconstruct ParseResult.
        let Some(content_list_raw) = cached_data.get("content_list").and_then(|v| v.as_array())
        else {
            return None;
        };
        let Some(doc_id) = cached_data.get("doc_id").and_then(|v| v.as_str()) else {
            return None;
        };

        let content_list: Vec<ContentItem> = content_list_raw
            .iter()
            .filter_map(|v| v.as_object().cloned())
            .map(|map| ContentItem::from_raw_map(map, -1))
            .collect();

        tracing::debug!("Found valid cached parsing result for key: {}", cache_key);
        let mut metadata = JsonMap::new();
        metadata.insert("cached".to_string(), JsonValue::from(true));
        metadata.insert("doc_id".to_string(), JsonValue::from(doc_id.to_string()));
        Some(ParseResult {
            content_list,
            doc_id: String::new(),
            file_path: path_to_absolute(file),
            metadata,
        })
    }

    /// Store parsing result in cache.
    #[allow(clippy::too_many_arguments)]
    pub async fn store_cached_result(
        &self,
        cache_key: &str,
        parse_result: &ParseResult,
        doc_id: &str,
        file: &Path,
        parser: &str,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) {
        let Some(kv_storage) = self.kv_storage.as_ref() else {
            return;
        };

        let file_mtime = file_mtime_millis(file);
        let parse_config = build_current_config(parser, parse_method, kwargs);

        let content_list_serialized: Vec<JsonValue> = parse_result
            .content_list
            .iter()
            .map(|item| JsonValue::Object(item.metadata().clone()))
            .collect();

        let mut cache_value = JsonMap::new();
        cache_value.insert(
            "content_list".to_string(),
            JsonValue::Array(content_list_serialized),
        );
        cache_value.insert("doc_id".to_string(), JsonValue::from(doc_id.to_string()));
        cache_value.insert("mtime".to_string(), JsonValue::from(file_mtime));
        cache_value.insert(
            "parse_config".to_string(),
            JsonValue::Object(parse_config),
        );
        cache_value.insert(
            "cached_at".to_string(),
            JsonValue::from(chrono::Utc::now().timestamp_millis()),
        );
        cache_value.insert("cache_version".to_string(), JsonValue::from("1.0"));

        let mut batch = std::collections::HashMap::new();
        batch.insert(cache_key.to_string(), cache_value);

        if let Err(e) = kv_storage.upsert(batch, &self.biz_id).await {
            tracing::warn!("Error storing to parse cache: {e}");
            return;
        }
        if let Err(e) = kv_storage.index_done_callback().await {
            tracing::warn!("Error committing parse cache: {e}");
            return;
        }
        tracing::info!("Stored parsing result in cache: {}", cache_key);
    }

    /// Invalidate cache entry for a specific file.
    pub async fn invalidate(&self, cache_key: &str) {
        let Some(kv_storage) = self.kv_storage.as_ref() else {
            return;
        };
        match kv_storage.delete(&[cache_key.to_string()], &self.biz_id).await {
            Ok(()) => tracing::debug!("Invalidated cache entry: {}", cache_key),
            Err(e) => tracing::warn!("Error invalidating cache: {e}"),
        }
    }
}

fn build_current_config(parser: &str, parse_method: Option<&str>, kwargs: &JsonMap) -> JsonMap {
    let mut config = JsonMap::new();
    config.insert("parser".to_string(), JsonValue::from(parser.to_string()));
    config.insert(
        "parse_method".to_string(),
        JsonValue::from(parse_method.unwrap_or("auto").to_string()),
    );
    let relevant: HashSet<&str> = RELEVANT_KWARGS.iter().copied().collect();
    for (key, value) in kwargs.iter() {
        if relevant.contains(key.as_str()) {
            config.insert(key.clone(), value.clone());
        }
    }
    config
}
