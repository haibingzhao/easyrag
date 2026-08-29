//! Multimodal query engine combining text and multimodal content
//!
//! Supports: pure text query, multimodal-enhanced query, VLM-enhanced query.
//!
//! The blueprint depends on the concrete `EasyRAG` facade; the Rust
//! port depends on the minimal [`TextQueryBackend`] trait instead, which
//! the `EasyRAG` facade implements (P4.7). Behavior is identical.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use easyrag_core::llm::{LlmRequest, LlmService};
use easyrag_core::model::{QueryMode, QueryParam};
use easyrag_core::storage::KVStorage;
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

use super::config::RagAnythingConfig;
use super::prompt_manager;
use super::util::encode_image_to_base64;

/// Minimal text-query backend the multimodal engine delegates to
/// (implemented by the `EasyRAG` facade).
#[async_trait]
pub trait TextQueryBackend: Send + Sync {
    async fn query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String>;
}

/// Multimodal query engine combining text and multimodal content for querying.
pub struct MultimodalQueryEngine {
    backend: Arc<dyn TextQueryBackend>,
    llm_service: Arc<dyn LlmService>,
    /// Held for API compatibility; unused there as well.
    #[allow(dead_code)]
    config: RagAnythingConfig,
    kv_storage: Option<Arc<dyn KVStorage>>,
    /// Business line bound at construction; used as the biz filter for
    /// cache KV calls and filled into delegated query params.
    biz_id: String,
}

impl MultimodalQueryEngine {
    pub fn new(
        backend: Arc<dyn TextQueryBackend>,
        llm_service: Arc<dyn LlmService>,
        config: RagAnythingConfig,
        kv_storage: Option<Arc<dyn KVStorage>>,
        biz_id: String,
    ) -> Self {
        Self {
            backend,
            llm_service,
            config,
            kv_storage,
            biz_id,
        }
    }

    /// Pure text query - delegates to the backend query functionality.
    /// `mode`: "local", "global", "hybrid", "naive", "mix".
    pub async fn query(
        &self,
        query: &str,
        mode: &str,
        _kwargs: &JsonMap,
    ) -> easyrag_core::Result<String> {
        tracing::info!("Executing text query: {}...", take_chars(query, 100));
        tracing::info!("Query mode: {}", mode);

        let param = QueryParam {
            mode: QueryMode::from_value(mode)?,
            biz_id: self.biz_id.clone(),
            ..Default::default()
        };
        self.backend.query(query, &param).await
    }

    /// Multimodal query - combines text and multimodal content for querying.
    pub async fn query_with_multimodal(
        &self,
        query: &str,
        multimodal_content: Option<&[JsonMap]>,
        mode: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<String> {
        tracing::info!("Executing multimodal query: {}...", take_chars(query, 100));

        // If no multimodal content, fallback to pure text query
        let Some(content) = multimodal_content.filter(|c| !c.is_empty()) else {
            tracing::info!("No multimodal content provided, executing text query");
            return self.query(query, mode, kwargs).await;
        };

        // Check cache
        let cache_key = generate_multimodal_cache_key(query, content, mode, kwargs);
        if let Some(cached) = self.get_cached_query_result(&cache_key).await {
            tracing::info!("Multimodal query cache hit: {}...", take_chars(&cache_key, 16));
            return Ok(cached);
        }

        // Process multimodal content to generate enhanced query
        let enhanced_query = process_multimodal_query_content(query, content).await;
        tracing::info!(
            "Generated enhanced query length: {} characters",
            enhanced_query.chars().count()
        );

        // Execute enhanced query via the backend
        let param = QueryParam {
            mode: QueryMode::from_value(mode)?,
            biz_id: self.biz_id.clone(),
            ..Default::default()
        };
        let result = self.backend.query(&enhanced_query, &param).await?;

        // Store result in cache
        self.store_cached_query_results(&cache_key, &result).await;

        Ok(result)
    }

    /// VLM-enhanced query - uses vision model to analyze images in context.
    pub async fn query_vlm_enhanced(
        &self,
        query: &str,
        image_paths: &[String],
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<String> {
        tracing::info!("Executing VLM-enhanced query: {}...", take_chars(query, 100));

        // First get text-based context
        let mode = kwargs
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("mix");
        let param = QueryParam {
            mode: QueryMode::from_value(mode)?,
            biz_id: self.biz_id.clone(),
            ..Default::default()
        };
        let base_result = self.backend.query(query, &param).await?;

        // If no images, return base result
        if image_paths.is_empty() {
            return Ok(base_result);
        }

        // Enhance with vision analysis of images
        let mut image_descriptions = Vec::new();
        for image_path in image_paths {
            let base64 = encode_image_to_base64(image_path);
            if !base64.is_empty() {
                // uses plain `complete` with a vision system
                // prompt (the base64 payload is only checked for non-empty).
                let vision_prompt = format!("Analyze this image in the context of the query: {query}");
                let request = match prompt_manager::get("IMAGE_ANALYSIS_SYSTEM") {
                    Ok(system_prompt) => LlmRequest::new(vision_prompt).with_system_prompt(system_prompt),
                    Err(_) => LlmRequest::new(vision_prompt),
                };
                match self.llm_service.complete(&request).await {
                    Ok(description) => image_descriptions.push(description),
                    Err(e) => {
                        tracing::warn!("Vision analysis failed for {}: {}", image_path, e);
                    }
                }
            }
        }

        // If no image descriptions generated, return base result
        if image_descriptions.is_empty() {
            return Ok(base_result);
        }

        // Combine base result with image analysis ( appendLine: each
        // line carries a trailing newline).
        let mut combined = String::new();
        combined.push_str("Based on the following information, provide a comprehensive answer to the query.\n");
        combined.push('\n');
        combined.push_str(&format!("Query: {query}\n"));
        combined.push('\n');
        combined.push_str("Text-based context:\n");
        combined.push_str(&format!("{base_result}\n"));
        combined.push('\n');
        combined.push_str("Image analysis:\n");
        for (idx, desc) in image_descriptions.iter().enumerate() {
            combined.push_str(&format!("Image {}: {desc}\n", idx + 1));
        }
        combined.push('\n');
        combined.push_str("Please synthesize all information into a coherent response.\n");

        self.llm_service.complete(&LlmRequest::new(combined)).await
    }

    // --- Private helpers ---

    async fn get_cached_query_result(&self, cache_key: &str) -> Option<String> {
        let kv_storage = self.kv_storage.as_ref()?;
        match kv_storage.get_by_id(cache_key, &self.biz_id).await {
            Ok(Some(cached)) => cached
                .get("return")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            Ok(None) => None,
            Err(e) => {
                tracing::debug!("Error accessing multimodal query cache: {}", e);
                None
            }
        }
    }

    async fn store_cached_query_results(&self, cache_key: &str, result: &str) {
        let Some(kv_storage) = self.kv_storage.as_ref() else {
            return;
        };
        let mut entry = JsonMap::new();
        entry.insert("return".to_string(), JsonValue::from(result.to_string()));
        entry.insert(
            "cached_at".to_string(),
            JsonValue::from(chrono::Utc::now().timestamp_millis()),
        );
        let mut data = HashMap::new();
        data.insert(cache_key.to_string(), entry);
        if let Err(e) = kv_storage.upsert(data, &self.biz_id).await {
            tracing::debug!("Error storing multimodal query cache: {}", e);
        }
    }
}

/// Process multimodal content to generate an enhanced query string.
async fn process_multimodal_query_content(query: &str, multimodal_content: &[JsonMap]) -> String {
    let mut content_descriptions = Vec::new();

    for item in multimodal_content {
        let type_str = item
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let description = match type_str {
            "image" => {
                let img_path = item
                    .get("img_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let captions = item
                    .get("image_caption")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                if captions.is_empty() {
                    format!("Image: {img_path}")
                } else {
                    format!("Image: {img_path} (captions: {captions})")
                }
            }
            "table" => {
                let table_data = item
                    .get("table_data")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("table_body").and_then(|v| v.as_str()))
                    .unwrap_or("");
                format!("Table data:\n{}", take_chars(table_data, 500))
            }
            "equation" => {
                let latex = item
                    .get("latex")
                    .and_then(|v| v.as_str())
                    .or_else(|| item.get("equation").and_then(|v| v.as_str()))
                    .unwrap_or("");
                format!("Equation: {latex}")
            }
            _ => {
                let entries = item
                    .iter()
                    .map(|(k, v)| format!("{k}={}", value_display(v)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("Content ({type_str}): {}", take_chars(&entries, 200))
            }
        };
        content_descriptions.push(description);
    }

    let mut out = String::new();
    out.push_str(&format!("{query}\n"));
    out.push('\n');
    out.push_str("Additional multimodal context:\n");
    for desc in &content_descriptions {
        out.push_str(&format!("- {desc}\n"));
    }
    out
}

/// Generate cache key for multimodal query
///
fn generate_multimodal_cache_key(
    query: &str,
    multimodal_content: &[JsonMap],
    mode: &str,
    kwargs: &JsonMap,
) -> String {
    // BTreeMap orders entries by key.
    let mut cache_data: BTreeMap<String, JsonValue> = BTreeMap::new();
    cache_data.insert("query".to_string(), JsonValue::from(query.trim().to_string()));
    cache_data.insert("mode".to_string(), JsonValue::from(mode.to_string()));

    // Normalize multimodal content
    let normalized: Vec<JsonMap> = multimodal_content
        .iter()
        .map(|item| {
            let mut normalized_item = JsonMap::new();
            for (key, value) in item {
                let normalized_value = match (key.as_str(), value) {
                    ("img_path" | "image_path" | "file_path", JsonValue::String(s)) => {
                        //
                        let name = Path::new(s)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default();
                        JsonValue::from(name)
                    }
                    ("table_data" | "table_body", JsonValue::String(s)) if s.chars().count() > 200 => {
                        JsonValue::from(md5_hex(s))
                    }
                    _ => value.clone(),
                };
                normalized_item.insert(key.clone(), normalized_value);
            }
            normalized_item
        })
        .collect();
    cache_data.insert(
        "multimodal_content".to_string(),
        JsonValue::Array(normalized.into_iter().map(JsonValue::Object).collect()),
    );

    // Add relevant kwargs
    let relevant_keys = [
        "stream",
        "response_type",
        "top_k",
        "max_tokens",
        "temperature",
        "system_prompt",
    ];
    for key in relevant_keys {
        if let Some(value) = kwargs.get(key) {
            cache_data.insert(key.to_string(), value.clone());
        }
    }

    let cache_str = cache_data
        .iter()
        .map(|(k, v)| format!("{k}={}", value_display(v)))
        .collect::<Vec<_>>()
        .join(",");
    format!("multimodal_query:{}", md5_hex(&cache_str))
}

/// Render a JSON value the way string interpolation would
/// (`Any?.toString`): strings unquoted, maps as `{k=v}`, lists as
/// `[a, b]`, integral floats with a trailing `.0`.
fn value_display(value: &JsonValue) -> String {
    match value {
        JsonValue::String(s) => s.clone(),
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => b.to_string(),
        JsonValue::Number(n) => {
            if let Some(f) = n.as_f64() {
                if n.is_f64() && f == f.trunc() && f.is_finite() {
                    return format!("{f:.1}");
                }
            }
            n.to_string()
        }
        JsonValue::Array(arr) => {
            let inner = arr
                .iter()
                .map(value_display)
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{inner}]")
        }
        JsonValue::Object(map) => {
            let inner = map
                .iter()
                .map(|(k, v)| format!("{k}={}", value_display(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{inner}}}")
        }
    }
}

/// Raw lowercase MD5 hex.
fn md5_hex(input: &str) -> String {
    compute_mdhash_id(input, "")
}

/// Take the first `n` chars.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
