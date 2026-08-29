//! Modal processors.
//!
//! Each processor handles a specific modal type (image, table, equation,
//! generic), generating an LLM description and persisting the resulting
//! entity + chunk into KV / vector / graph storage.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use easyrag_core::llm::{LlmRequest, LlmService};
use easyrag_core::storage::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::{CoreError, JsonMap};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value as JsonValue;

use super::context::ContextExtractor;
use super::model::{ContentItem, ModalProcessResult, ModalType};
use super::{prompt_manager, util};

/// Modal processor interface.
/// Each processor handles a specific modal type (image, table, equation, generic).
#[async_trait]
pub trait ModalProcessor: Send + Sync {
    /// Process a content item and produce a modal process result.
    /// Errors only on modal-type mismatch ( throws
    /// IllegalArgumentException); LLM/storage failures are absorbed into a
    /// fallback result.
    async fn process(
        &self,
        item: &ContentItem,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult>;

    /// Generate description only (without entity/chunk creation).
    /// Used for batch processing stage 1.
    /// Returns `(description, entity_info map)`; never fails
    /// (falls back on any error).
    async fn generate_description(&self, item: &ContentItem, context: &str) -> (String, JsonMap);

    /// The modal types this processor supports.
    fn supported_types(&self) -> HashSet<ModalType>;
}

/// Shared dependencies for all modal processors.
#[derive(Clone)]
pub struct BaseModalProcessor {
    pub llm_service: Arc<dyn LlmService>,
    pub kv_storage: Arc<dyn KVStorage>,
    pub vector_storage: Arc<dyn VectorStorage>,
    pub graph_storage: Arc<dyn GraphStorage>,
    /// Held for API compatibility; context extraction is
    /// performed upstream by the document processor.
    pub context_extractor: Arc<ContextExtractor>,
    /// Business line bound at construction; passed as the biz filter on
    /// every storage call.
    pub biz_id: String,
}

impl BaseModalProcessor {
    pub fn new(
        llm_service: Arc<dyn LlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        context_extractor: Arc<ContextExtractor>,
        biz_id: String,
    ) -> Self {
        Self {
            llm_service,
            kv_storage,
            vector_storage,
            graph_storage,
            context_extractor,
            biz_id,
        }
    }

    /// Create entity node and text chunk in storage.
    pub async fn create_entity_and_chunk(
        &self,
        modal_chunk: &str,
        entity_info: &JsonMap,
        file_path: &str,
        doc_id: Option<&str>,
        chunk_order_index: i64,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let entity_name = entity_info
            .get("entity_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entity_type = entity_info
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let summary = entity_info
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Generate chunk ID.
        let chunk_id = compute_mdhash_id(modal_chunk, "chunk-");
        // Approximate token count (: String.length / 4; char count is
        // the closest Rust analogue of UTF-16 length).
        let tokens = modal_chunk.chars().count() as i64 / 4;

        let actual_doc_id = doc_id.unwrap_or(&chunk_id).to_string();

        // Store chunk in KV storage.
        let mut chunk_data = JsonMap::new();
        chunk_data.insert("tokens".to_string(), JsonValue::from(tokens));
        chunk_data.insert("content".to_string(), JsonValue::from(modal_chunk));
        chunk_data.insert(
            "chunk_order_index".to_string(),
            JsonValue::from(chunk_order_index),
        );
        chunk_data.insert("full_doc_id".to_string(), JsonValue::from(actual_doc_id.clone()));
        chunk_data.insert("file_path".to_string(), JsonValue::from(file_path));
        let mut kv_batch = HashMap::new();
        kv_batch.insert(chunk_id.clone(), chunk_data);
        self.kv_storage.upsert(kv_batch, &self.biz_id).await?;

        // Store chunk in vector database.
        let mut chunk_vdb = JsonMap::new();
        chunk_vdb.insert("content".to_string(), JsonValue::from(modal_chunk));
        chunk_vdb.insert("full_doc_id".to_string(), JsonValue::from(actual_doc_id.clone()));
        chunk_vdb.insert("tokens".to_string(), JsonValue::from(tokens));
        chunk_vdb.insert(
            "chunk_order_index".to_string(),
            JsonValue::from(chunk_order_index),
        );
        chunk_vdb.insert("file_path".to_string(), JsonValue::from(file_path));
        let mut vdb_batch = HashMap::new();
        vdb_batch.insert(chunk_id.clone(), chunk_vdb);
        self.vector_storage.upsert(vdb_batch, &self.biz_id).await?;

        // Create entity node in knowledge graph.
        let mut node_data = JsonMap::new();
        node_data.insert("entity_id".to_string(), JsonValue::from(entity_name.clone()));
        node_data.insert("entity_type".to_string(), JsonValue::from(entity_type.clone()));
        node_data.insert("description".to_string(), JsonValue::from(summary.clone()));
        node_data.insert("source_id".to_string(), JsonValue::from(chunk_id.clone()));
        node_data.insert("file_path".to_string(), JsonValue::from(file_path));
        node_data.insert(
            "created_at".to_string(),
            JsonValue::from(chrono::Utc::now().timestamp()),
        );
        self.graph_storage
            .upsert_node(&entity_name, node_data, &self.biz_id)
            .await?;

        // Insert entity into vector database.
        let entity_id = compute_mdhash_id(&entity_name, "ent-");
        let mut entity_vdb = JsonMap::new();
        entity_vdb.insert("entity_name".to_string(), JsonValue::from(entity_name.clone()));
        entity_vdb.insert("entity_type".to_string(), JsonValue::from(entity_type.clone()));
        entity_vdb.insert(
            "content".to_string(),
            JsonValue::from(format!("{entity_name}\n{summary}")),
        );
        entity_vdb.insert("source_id".to_string(), JsonValue::from(chunk_id.clone()));
        entity_vdb.insert("file_path".to_string(), JsonValue::from(file_path));
        let mut entity_batch = HashMap::new();
        entity_batch.insert(entity_id, entity_vdb);
        self.vector_storage.upsert(entity_batch, &self.biz_id).await?;

        let mut metadata = JsonMap::new();
        metadata.insert("chunk_id".to_string(), JsonValue::from(chunk_id.clone()));
        metadata.insert("doc_id".to_string(), JsonValue::from(actual_doc_id.clone()));
        metadata.insert("file_path".to_string(), JsonValue::from(file_path));

        Ok(ModalProcessResult {
            entity_name,
            entity_type,
            description: summary,
            chunk_content: modal_chunk.to_string(),
            metadata,
            success: true,
            error_message: None,
        })
    }

    /// Robust JSON parsing with multiple fallback strategies.
    pub fn robust_json_parse(&self, response: &str) -> JsonMap {
        // Strategy 1: Try direct parsing.
        let candidates = extract_json_candidates(response);
        for candidate in &candidates {
            if let Some(result) = try_parse_json(candidate) {
                return result;
            }
        }

        // Strategy 2: Try with basic cleanup.
        for candidate in &candidates {
            let cleaned = basic_json_cleanup(candidate);
            if let Some(result) = try_parse_json(&cleaned) {
                return result;
            }
        }

        // Strategy 3: Fallback to regex field extraction.
        extract_fields_with_regex(response)
    }

    /// Parse LLM response into `(description, entity_info map)`.
    /// Falls back to the cleaned raw response when required fields are
    /// missing or unparsable.
    pub fn parse_response(
        &self,
        response: &str,
        entity_name: Option<&str>,
        default_entity_type: &str,
    ) -> (String, JsonMap) {
        let parsed = self.robust_json_parse(response);
        let description = parsed
            .get("detailed_description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let entity_data = parsed
            .get("entity_info")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        if description.is_empty() || entity_data.is_empty() {
            tracing::error!("Error parsing analysis response: Missing required fields in response");
            let cleaned = strip_thinking_tags(response);
            let fallback_name = entity_name.map(|n| n.to_string()).unwrap_or_else(|| {
                format!("{default_entity_type}_{}", compute_mdhash_id(&cleaned, ""))
            });
            // counts characters, not bytes.
            let summary = if take_chars(&cleaned, 101).chars().count() > 100 {
                format!("{}...", take_chars(&cleaned, 100))
            } else {
                cleaned.clone()
            };
            let mut fallback = JsonMap::new();
            fallback.insert("entity_name".to_string(), JsonValue::from(fallback_name));
            fallback.insert(
                "entity_type".to_string(),
                JsonValue::from(default_entity_type),
            );
            fallback.insert("summary".to_string(), JsonValue::from(summary));
            return (cleaned, fallback);
        }

        let entity_type_str = entity_data
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or(default_entity_type)
            .to_string();
        let resolved_name = entity_name.map(|n| n.to_string()).unwrap_or_else(|| {
            let name = entity_data
                .get("entity_name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("{name} ({entity_type_str})")
        });
        let summary = entity_data
            .get("summary")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| take_chars(&description, 100));

        let mut resolved = JsonMap::new();
        resolved.insert("entity_name".to_string(), JsonValue::from(resolved_name));
        resolved.insert("entity_type".to_string(), JsonValue::from(entity_type_str));
        resolved.insert("summary".to_string(), JsonValue::from(summary));

        (description, resolved)
    }

    /// Call LLM with prompt and system prompt.
    pub async fn call_llm(&self, prompt: &str, system_prompt: &str) -> easyrag_core::Result<String> {
        let request = LlmRequest::new(prompt).with_system_prompt(system_prompt);
        self.llm_service.complete(&request).await
    }

    /// Get prompt template by key from the PromptManager.
    pub fn get_prompt(key: &str) -> easyrag_core::Result<String> {
        prompt_manager::get(key)
    }
}

/// Strip `<think>`/`<thinking>` tags from reasoning models
/// companion `stripThinkingTags`).
pub fn strip_thinking_tags(text: &str) -> String {
    static THINK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?si)<think>.*?</think>").unwrap());
    static THINKING: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?si)<thinking>.*?</thinking>").unwrap());
    let cleaned = THINK.replace_all(text, "");
    let cleaned = THINKING.replace_all(&cleaned, "");
    cleaned.trim().to_string()
}

/// First `n` chars of `s`.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Display a metadata value the way string interpolation does,
/// defaulting to "None" for missing/null values.
fn metadata_or_none(value: Option<&JsonValue>) -> String {
    match value {
        None | Some(JsonValue::Null) => "None".to_string(),
        Some(JsonValue::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Join a metadata array value with ", ",
/// returning None when the value is not a list.
fn json_list_join(value: Option<&JsonValue>) -> Option<String> {
    let JsonValue::Array(items) = value? else {
        return None;
    };
    Some(
        items
            .iter()
            .map(|v| match v {
                JsonValue::String(s) => s.clone(),
                JsonValue::Null => "null".to_string(),
                other => other.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", "),
    )
}

/// Extract common file info fields from the processor file-info map.
fn read_file_info(file_info: &JsonMap) -> (String, Option<String>, i64) {
    let file_path = file_info
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("manual_creation")
        .to_string();
    let doc_id = file_info
        .get("doc_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let chunk_order_index = file_info
        .get("chunk_order_index")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    (file_path, doc_id, chunk_order_index)
}

// --- JSON candidate extraction helpers ---

static CODE_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)```(?:json)?\s*(\{.*?\})\s*```").unwrap());
static SIMPLE_JSON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)\{.*\}").unwrap());

fn extract_json_candidates(response: &str) -> Vec<String> {
    let mut candidates: Vec<String> = Vec::new();

    // Remove thinking tags.
    let cleaned = strip_thinking_tags(response);

    // Method 1: JSON in code blocks.
    for m in CODE_BLOCK_RE.captures_iter(&cleaned) {
        candidates.push(m[1].to_string());
    }

    // Method 2: Balanced braces.
    let mut brace_count: i32 = 0;
    let mut start_pos: Option<usize> = None;
    for (i, ch) in cleaned.char_indices() {
        match ch {
            '{' => {
                if brace_count == 0 {
                    start_pos = Some(i);
                }
                brace_count += 1;
            }
            '}' => {
                brace_count -= 1;
                if brace_count == 0 {
                    if let Some(start) = start_pos {
                        candidates.push(cleaned[start..i + 1].to_string());
                    }
                }
            }
            _ => {}
        }
    }

    // Method 3: Simple regex fallback.
    if let Some(m) = SIMPLE_JSON_RE.find(&cleaned) {
        candidates.push(m.as_str().to_string());
    }

    candidates
}

fn try_parse_json(json_str: &str) -> Option<JsonMap> {
    if json_str.trim().is_empty() {
        return None;
    }
    serde_json::from_str::<JsonMap>(json_str).ok()
}

static TRAILING_COMMA_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r",(\s*[}\]])").unwrap());

fn basic_json_cleanup(json_str: &str) -> String {
    let mut result = json_str.trim().to_string();
    // Fix smart quotes.
    result = result.replace('\u{201c}', "\"").replace('\u{201d}', "\"");
    result = result.replace('\u{2018}', "'").replace('\u{2019}', "'");
    // Fix trailing commas.
    TRAILING_COMMA_RE.replace_all(&result, "$1").to_string()
}

static FIELD_RE_DESC: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?s)"detailed_description":\s*"([^"]*(?:\\.[^"]*)*)""#).unwrap()
});
static FIELD_RE_NAME: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""entity_name":\s*"([^"]*(?:\\.[^"]*)*)""#).unwrap());
static FIELD_RE_TYPE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#""entity_type":\s*"([^"]*(?:\\.[^"]*)*)""#).unwrap());
static FIELD_RE_SUMMARY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)"summary":\s*"([^"]*(?:\\.[^"]*)*)""#).unwrap());

fn extract_fields_with_regex(response: &str) -> JsonMap {
    tracing::warn!("Using regex fallback for JSON parsing");

    let description = FIELD_RE_DESC
        .captures(response)
        .map(|m| m[1].to_string())
        .unwrap_or_default();
    let entity_name = FIELD_RE_NAME
        .captures(response)
        .map(|m| m[1].to_string())
        .unwrap_or_else(|| "unknown_entity".to_string());
    let entity_type = FIELD_RE_TYPE
        .captures(response)
        .map(|m| m[1].to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let summary = FIELD_RE_SUMMARY
        .captures(response)
        .map(|m| m[1].to_string())
        .unwrap_or_else(|| take_chars(&description, 100));

    let mut entity_info = JsonMap::new();
    entity_info.insert("entity_name".to_string(), JsonValue::from(entity_name));
    entity_info.insert("entity_type".to_string(), JsonValue::from(entity_type));
    entity_info.insert("summary".to_string(), JsonValue::from(summary));

    let mut out = JsonMap::new();
    out.insert(
        "detailed_description".to_string(),
        JsonValue::from(description),
    );
    out.insert("entity_info".to_string(), JsonValue::Object(entity_info));
    out
}

/// Processor specialized for image content.
/// Calls `LlmService.complete_with_image` for vision analysis with base64
/// encoded images. Falls back to filename/context-based description when
/// vision is unavailable.
pub struct ImageModalProcessor {
    base: BaseModalProcessor,
}

impl ImageModalProcessor {
    pub fn new(base: BaseModalProcessor) -> Self {
        Self { base }
    }

    async fn process_inner(
        &self,
        img_path: &str,
        captions: &[String],
        metadata: &JsonMap,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let (file_path, doc_id, chunk_order_index) = read_file_info(file_info);

        let (description, entity_info) = self.generate_description_inner(
            img_path, captions, metadata, context,
        ).await?;

        // Build complete image content chunk: every line
        // carries a trailing newline).
        let captions_str = captions.join(", ");
        let captions_str = if captions_str.is_empty() { "None".to_string() } else { captions_str };
        let modal_chunk = format!(
            "## Image Content\nSection: {}\nImage path: {}\nCaptions: {}\nFootnotes: {}\n\n### Analysis\n{}\n",
            metadata_or_none(metadata.get("_section_path")),
            img_path,
            captions_str,
            json_list_join(metadata.get("image_footnote")).unwrap_or_else(|| "None".to_string()),
            description
        );

        self.base
            .create_entity_and_chunk(&modal_chunk, &entity_info, &file_path, doc_id.as_deref(), chunk_order_index)
            .await
    }

    async fn generate_description_inner(
        &self,
        img_path: &str,
        captions: &[String],
        metadata: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<(String, JsonMap)> {
        let section_path = metadata
            .get("_section_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Build vision prompt.
        let captions_joined = captions.join(", ");
        let captions_value = if captions_joined.is_empty() { "None".to_string() } else { captions_joined };
        let section_value = if section_path.is_empty() { "None".to_string() } else { section_path.clone() };
        let vision_prompt = if !context.is_empty() {
            prompt_manager::get("vision_prompt_with_context")?
                .replace("{context}", context)
                .replace("{section_path}", &section_value)
                .replace("{entity_name}", "unique descriptive name for this image")
                .replace("{image_path}", img_path)
                .replace("{captions}", &captions_value)
                .replace("{footnotes}", "None")
        } else {
            prompt_manager::get("vision_prompt")?
                .replace("{section_path}", &section_value)
                .replace("{entity_name}", "unique descriptive name for this image")
                .replace("{image_path}", img_path)
                .replace("{captions}", &captions_value)
                .replace("{footnotes}", "None")
        };

        // Try vision-based analysis with actual image content.
        let image_base64 = util::encode_image_to_base64(img_path);
        let response = if !image_base64.is_empty() {
            let lower = img_path.to_lowercase();
            let mime_type = if lower.ends_with(".png") {
                "image/png"
            } else if lower.ends_with(".gif") {
                "image/gif"
            } else if lower.ends_with(".webp") {
                "image/webp"
            } else {
                "image/jpeg"
            };
            let request = LlmRequest::new(vision_prompt)
                .with_system_prompt(prompt_manager::get("IMAGE_ANALYSIS_SYSTEM")?);
            match self
                .base
                .llm_service
                .complete_with_image(&request, &image_base64, mime_type)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Vision completion failed, using text-only fallback: {e}");
                    self.generate_fallback_description(img_path, captions, context).await?
                }
            }
        } else {
            self.generate_fallback_description(img_path, captions, context).await?
        };

        Ok(self.base.parse_response(&response, None, "image"))
    }

    /// Generate fallback description when vision is unavailable.
    /// Uses filename, captions, and context to produce a text-based description.
    async fn generate_fallback_description(
        &self,
        img_path: &str,
        captions: &[String],
        context: &str,
    ) -> easyrag_core::Result<String> {
        let file_name = Path::new(img_path)
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let caption_text = captions.join(", ");
        let caption_value = if caption_text.is_empty() { "None".to_string() } else { caption_text };

        let mut prompt = String::new();
        prompt.push_str("Based on the following information, generate a JSON description for an image:\n");
        prompt.push_str(&format!("- File name: {file_name}\n"));
        prompt.push_str(&format!("- Captions: {caption_value}\n"));
        if !context.is_empty() {
            prompt.push_str(&format!("- Surrounding context: {}\n", take_chars(context, 500)));
        }
        prompt.push('\n');
        prompt.push_str("Return JSON with fields: detailed_description, entity_info (entity_name, entity_type, summary)\n");

        self.base
            .call_llm(&prompt, &prompt_manager::get("IMAGE_ANALYSIS_SYSTEM")?)
            .await
    }
}

#[async_trait]
impl ModalProcessor for ImageModalProcessor {
    async fn process(
        &self,
        item: &ContentItem,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let ContentItem::Image {
            img_path,
            captions,
            metadata,
            ..
        } = item
        else {
            return Err(CoreError::InvalidArgument(format!(
                "Expected ImageContent, got {}",
                item.modal_type().value()
            )));
        };

        match self.process_inner(img_path, captions, metadata, file_info, context).await {
            Ok(result) => Ok(result),
            Err(e) => {
                tracing::error!("Error processing image content: {e}");
                // Fallback processing.
                let mut fallback_meta = JsonMap::new();
                fallback_meta.insert("fallback".to_string(), JsonValue::from(true));
                Ok(ModalProcessResult {
                    entity_name: format!("image_{}", compute_mdhash_id(img_path, "")),
                    entity_type: "image".to_string(),
                    description: format!("Image content: {img_path}"),
                    chunk_content: img_path.clone(),
                    metadata: fallback_meta,
                    success: true,
                    error_message: None,
                })
            }
        }
    }

    async fn generate_description(&self, item: &ContentItem, context: &str) -> (String, JsonMap) {
        let ContentItem::Image {
            img_path,
            captions,
            metadata,
            ..
        } = item
        else {
            let mut entity = JsonMap::new();
            entity.insert("entity_name".to_string(), JsonValue::from("unknown_image"));
            entity.insert("entity_type".to_string(), JsonValue::from("image"));
            entity.insert("summary".to_string(), JsonValue::from(""));
            return (String::new(), entity);
        };

        match self
            .generate_description_inner(img_path, captions, metadata, context)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Error generating image description: {e}");
                let mut entity = JsonMap::new();
                entity.insert(
                    "entity_name".to_string(),
                    JsonValue::from(format!("image_{}", compute_mdhash_id(img_path, ""))),
                );
                entity.insert("entity_type".to_string(), JsonValue::from("image"));
                entity.insert(
                    "summary".to_string(),
                    JsonValue::from(format!("Image content: {}", take_chars(img_path, 100))),
                );
                (img_path.clone(), entity)
            }
        }
    }

    fn supported_types(&self) -> HashSet<ModalType> {
        HashSet::from([ModalType::Image])
    }
}

/// Processor specialized for table content.
/// Performs structured table analysis: row/column identification, data
/// summary, key insights.
pub struct TableModalProcessor {
    base: BaseModalProcessor,
}

impl TableModalProcessor {
    pub fn new(base: BaseModalProcessor) -> Self {
        Self { base }
    }

    async fn process_inner(
        &self,
        table_body: &str,
        captions: &[String],
        metadata: &JsonMap,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let (file_path, doc_id, chunk_order_index) = read_file_info(file_info);

        let (description, entity_info) = self
            .generate_description_inner(table_body, captions, metadata, context)
            .await?;

        // Build complete table content chunk.
        let captions_str = captions.join(", ");
        let captions_str = if captions_str.is_empty() { "None".to_string() } else { captions_str };
        let modal_chunk = format!(
            "## Table Content\nImage path: {}\nCaptions: {}\n\n### Table Data\n{}\n\n### Footnotes\n{}\n\n### Analysis\n{}\n",
            metadata_or_none(metadata.get("img_path")),
            captions_str,
            table_body,
            json_list_join(metadata.get("table_footnote")).unwrap_or_else(|| "None".to_string()),
            description
        );

        self.base
            .create_entity_and_chunk(&modal_chunk, &entity_info, &file_path, doc_id.as_deref(), chunk_order_index)
            .await
    }

    async fn generate_description_inner(
        &self,
        table_body: &str,
        captions: &[String],
        metadata: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<(String, JsonMap)> {
        let table_caption = captions.join(", ");
        let table_img_path = metadata
            .get("img_path")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let table_footnote = json_list_join(metadata.get("table_footnote")).unwrap_or_default();

        // Build table analysis prompt.
        let caption_value = if table_caption.is_empty() { "None".to_string() } else { table_caption };
        let footnote_value = if table_footnote.is_empty() { "None".to_string() } else { table_footnote };
        let table_prompt = if !context.is_empty() {
            prompt_manager::get("table_prompt_with_context")?
                .replace("{context}", context)
                .replace("{entity_name}", "descriptive name for this table")
                .replace("{table_img_path}", &table_img_path)
                .replace("{table_caption}", &caption_value)
                .replace("{table_body}", table_body)
                .replace("{table_footnote}", &footnote_value)
        } else {
            prompt_manager::get("table_prompt")?
                .replace("{entity_name}", "descriptive name for this table")
                .replace("{table_img_path}", &table_img_path)
                .replace("{table_caption}", &caption_value)
                .replace("{table_body}", table_body)
                .replace("{table_footnote}", &footnote_value)
        };

        let response = self
            .base
            .call_llm(&table_prompt, &prompt_manager::get("TABLE_ANALYSIS_SYSTEM")?)
            .await?;
        Ok(self.base.parse_response(&response, None, "table"))
    }
}

#[async_trait]
impl ModalProcessor for TableModalProcessor {
    async fn process(
        &self,
        item: &ContentItem,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let ContentItem::Table {
            table_body,
            captions,
            metadata,
            ..
        } = item
        else {
            return Err(CoreError::InvalidArgument(format!(
                "Expected TableContent, got {}",
                item.modal_type().value()
            )));
        };

        match self
            .process_inner(table_body, captions, metadata, file_info, context)
            .await
        {
            Ok(result) => Ok(result),
            Err(e) => {
                tracing::error!("Error processing table content: {e}");
                let mut fallback_meta = JsonMap::new();
                fallback_meta.insert("fallback".to_string(), JsonValue::from(true));
                Ok(ModalProcessResult {
                    entity_name: format!("table_{}", compute_mdhash_id(table_body, "")),
                    entity_type: "table".to_string(),
                    description: format!("Table content: {}", take_chars(table_body, 100)),
                    chunk_content: table_body.clone(),
                    metadata: fallback_meta,
                    success: true,
                    error_message: None,
                })
            }
        }
    }

    async fn generate_description(&self, item: &ContentItem, context: &str) -> (String, JsonMap) {
        let ContentItem::Table {
            table_body,
            captions,
            metadata,
            ..
        } = item
        else {
            let mut entity = JsonMap::new();
            entity.insert("entity_name".to_string(), JsonValue::from("unknown_table"));
            entity.insert("entity_type".to_string(), JsonValue::from("table"));
            entity.insert("summary".to_string(), JsonValue::from(""));
            return (String::new(), entity);
        };

        match self
            .generate_description_inner(table_body, captions, metadata, context)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Error generating table description: {e}");
                let mut entity = JsonMap::new();
                entity.insert(
                    "entity_name".to_string(),
                    JsonValue::from(format!("table_{}", compute_mdhash_id(table_body, ""))),
                );
                entity.insert("entity_type".to_string(), JsonValue::from("table"));
                entity.insert(
                    "summary".to_string(),
                    JsonValue::from(format!("Table content: {}", take_chars(table_body, 100))),
                );
                (table_body.clone(), entity)
            }
        }
    }

    fn supported_types(&self) -> HashSet<ModalType> {
        HashSet::from([ModalType::Table])
    }
}

/// Processor specialized for equation content.
/// Performs LaTeX equation parsing, variable identification, and physical
/// meaning description.
pub struct EquationModalProcessor {
    base: BaseModalProcessor,
}

impl EquationModalProcessor {
    pub fn new(base: BaseModalProcessor) -> Self {
        Self { base }
    }

    async fn process_inner(
        &self,
        equation_text: &str,
        equation_format: &str,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let (file_path, doc_id, chunk_order_index) = read_file_info(file_info);

        let (description, entity_info) = self
            .generate_description_inner(equation_text, equation_format, context)
            .await?;

        // Build complete equation content chunk.
        let modal_chunk = format!(
            "## Equation Content\nFormat: {}\n\n### Equation\n{}\n\n### Analysis\n{}\n",
            equation_format, equation_text, description
        );

        self.base
            .create_entity_and_chunk(&modal_chunk, &entity_info, &file_path, doc_id.as_deref(), chunk_order_index)
            .await
    }

    async fn generate_description_inner(
        &self,
        equation_text: &str,
        equation_format: &str,
        context: &str,
    ) -> easyrag_core::Result<(String, JsonMap)> {
        // Build equation analysis prompt.
        let equation_prompt = if !context.is_empty() {
            prompt_manager::get("equation_prompt_with_context")?
                .replace("{context}", context)
                .replace("{equation_text}", equation_text)
                .replace("{equation_format}", equation_format)
                .replace("{entity_name}", "descriptive name for this equation")
        } else {
            prompt_manager::get("equation_prompt")?
                .replace("{equation_text}", equation_text)
                .replace("{equation_format}", equation_format)
                .replace("{entity_name}", "descriptive name for this equation")
        };

        let response = self
            .base
            .call_llm(
                &equation_prompt,
                &prompt_manager::get("EQUATION_ANALYSIS_SYSTEM")?,
            )
            .await?;
        Ok(self.base.parse_response(&response, None, "equation"))
    }
}

#[async_trait]
impl ModalProcessor for EquationModalProcessor {
    async fn process(
        &self,
        item: &ContentItem,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let ContentItem::Equation {
            equation_text,
            equation_format,
            ..
        } = item
        else {
            return Err(CoreError::InvalidArgument(format!(
                "Expected EquationContent, got {}",
                item.modal_type().value()
            )));
        };

        match self
            .process_inner(equation_text, equation_format, file_info, context)
            .await
        {
            Ok(result) => Ok(result),
            Err(e) => {
                tracing::error!("Error processing equation content: {e}");
                let mut fallback_meta = JsonMap::new();
                fallback_meta.insert("fallback".to_string(), JsonValue::from(true));
                Ok(ModalProcessResult {
                    entity_name: format!("equation_{}", compute_mdhash_id(equation_text, "")),
                    entity_type: "equation".to_string(),
                    description: format!(
                        "Equation content: {}",
                        take_chars(equation_text, 100)
                    ),
                    chunk_content: equation_text.clone(),
                    metadata: fallback_meta,
                    success: true,
                    error_message: None,
                })
            }
        }
    }

    async fn generate_description(&self, item: &ContentItem, context: &str) -> (String, JsonMap) {
        let ContentItem::Equation {
            equation_text,
            equation_format,
            ..
        } = item
        else {
            let mut entity = JsonMap::new();
            entity.insert("entity_name".to_string(), JsonValue::from("unknown_equation"));
            entity.insert("entity_type".to_string(), JsonValue::from("equation"));
            entity.insert("summary".to_string(), JsonValue::from(""));
            return (String::new(), entity);
        };

        match self
            .generate_description_inner(equation_text, equation_format, context)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Error generating equation description: {e}");
                let mut entity = JsonMap::new();
                entity.insert(
                    "entity_name".to_string(),
                    JsonValue::from(format!("equation_{}", compute_mdhash_id(equation_text, ""))),
                );
                entity.insert("entity_type".to_string(), JsonValue::from("equation"));
                entity.insert(
                    "summary".to_string(),
                    JsonValue::from(format!(
                        "Equation content: {}",
                        take_chars(equation_text, 100)
                    )),
                );
                (equation_text.clone(), entity)
            }
        }
    }

    fn supported_types(&self) -> HashSet<ModalType> {
        HashSet::from([ModalType::Equation])
    }
}

/// Generic fallback processor for unknown modal types.
pub struct GenericModalProcessor {
    base: BaseModalProcessor,
}

impl GenericModalProcessor {
    pub fn new(base: BaseModalProcessor) -> Self {
        Self { base }
    }

    async fn process_inner(
        &self,
        content: &str,
        content_type: &str,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let (file_path, doc_id, chunk_order_index) = read_file_info(file_info);

        let (description, entity_info) = self
            .generate_description_inner(content, content_type, context)
            .await?;

        let modal_chunk = format!(
            "## {} Content\n\n### Raw Content\n{}\n\n### Analysis\n{}\n",
            capitalize_first(content_type),
            content,
            description
        );

        self.base
            .create_entity_and_chunk(&modal_chunk, &entity_info, &file_path, doc_id.as_deref(), chunk_order_index)
            .await
    }

    async fn generate_description_inner(
        &self,
        content: &str,
        content_type: &str,
        context: &str,
    ) -> easyrag_core::Result<(String, JsonMap)> {
        let entity_placeholder = format!("descriptive name for this {content_type}");
        let generic_prompt = if !context.is_empty() {
            prompt_manager::get("generic_prompt_with_context")?
                .replace("{context}", context)
                .replace("{content_type}", content_type)
                .replace("{entity_name}", &entity_placeholder)
                .replace("{content}", content)
        } else {
            prompt_manager::get("generic_prompt")?
                .replace("{content_type}", content_type)
                .replace("{entity_name}", &entity_placeholder)
                .replace("{content}", content)
        };

        let system_prompt = prompt_manager::get("GENERIC_ANALYSIS_SYSTEM")?
            .replace("{content_type}", content_type);
        let response = self.base.call_llm(&generic_prompt, &system_prompt).await?;
        Ok(self.base.parse_response(&response, None, content_type))
    }
}

/// Uppercase the first character.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

#[async_trait]
impl ModalProcessor for GenericModalProcessor {
    async fn process(
        &self,
        item: &ContentItem,
        file_info: &JsonMap,
        context: &str,
    ) -> easyrag_core::Result<ModalProcessResult> {
        let ContentItem::Generic {
            content,
            content_type,
            ..
        } = item
        else {
            return Err(CoreError::InvalidArgument(format!(
                "Expected GenericContent, got {}",
                item.modal_type().value()
            )));
        };

        match self
            .process_inner(content, content_type, file_info, context)
            .await
        {
            Ok(result) => Ok(result),
            Err(e) => {
                tracing::error!("Error processing {} content: {}", content_type, e);
                let mut fallback_meta = JsonMap::new();
                fallback_meta.insert("fallback".to_string(), JsonValue::from(true));
                Ok(ModalProcessResult {
                    entity_name: format!("{content_type}_{}", compute_mdhash_id(content, "")),
                    entity_type: content_type.clone(),
                    description: format!(
                        "{} content: {}",
                        content_type,
                        take_chars(content, 100)
                    ),
                    chunk_content: content.clone(),
                    metadata: fallback_meta,
                    success: true,
                    error_message: None,
                })
            }
        }
    }

    async fn generate_description(&self, item: &ContentItem, context: &str) -> (String, JsonMap) {
        let ContentItem::Generic {
            content,
            content_type,
            ..
        } = item
        else {
            let mut entity = JsonMap::new();
            entity.insert("entity_name".to_string(), JsonValue::from("unknown_content"));
            entity.insert("entity_type".to_string(), JsonValue::from("generic"));
            entity.insert("summary".to_string(), JsonValue::from(""));
            return (String::new(), entity);
        };

        match self
            .generate_description_inner(content, content_type, context)
            .await
        {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Error generating {} description: {}", content_type, e);
                let mut entity = JsonMap::new();
                entity.insert(
                    "entity_name".to_string(),
                    JsonValue::from(format!("{content_type}_{}", compute_mdhash_id(content, ""))),
                );
                entity.insert("entity_type".to_string(), JsonValue::from(content_type.as_str()));
                entity.insert(
                    "summary".to_string(),
                    JsonValue::from(format!(
                        "{} content: {}",
                        content_type,
                        take_chars(content, 100)
                    )),
                );
                (content.clone(), entity)
            }
        }
    }

    fn supported_types(&self) -> HashSet<ModalType> {
        HashSet::from([ModalType::Generic])
    }
}

/// Registry that routes ModalType to the corresponding ModalProcessor.
pub struct ModalProcessorRegistry;

impl ModalProcessorRegistry {
    fn inner() -> &'static Mutex<HashMap<ModalType, Arc<dyn ModalProcessor>>> {
        static INNER: OnceLock<Mutex<HashMap<ModalType, Arc<dyn ModalProcessor>>>> =
            OnceLock::new();
        INNER.get_or_init(|| Mutex::new(HashMap::new()))
    }

    /// Register a processor for a specific modal type.
    pub fn register(modal_type: ModalType, processor: Arc<dyn ModalProcessor>) {
        Self::inner()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(modal_type, processor);
    }

    /// Get the processor for a given modal type.
    /// Falls back to GENERIC processor if no specific processor is registered.
    pub fn get_processor(modal_type: ModalType) -> Option<Arc<dyn ModalProcessor>> {
        let guard = Self::inner().lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get(&modal_type)
            .or_else(|| guard.get(&ModalType::Generic))
            .cloned()
    }

    /// Check if a processor is registered for the given type.
    pub fn has_processor(modal_type: ModalType) -> bool {
        Self::inner()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&modal_type)
    }

    /// Get all registered modal types.
    pub fn get_registered_types() -> HashSet<ModalType> {
        Self::inner()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .copied()
            .collect()
    }

    /// Clear all registered processors.
    pub fn clear() {
        Self::inner()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}
