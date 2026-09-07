//! Core document processor orchestrating the full multimodal pipeline
//!
//! Flow: parse → separate_content → insert_text → process_modals.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use easyrag_core::llm::LlmService;
use easyrag_core::storage::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;
use tokio::sync::Semaphore;

use super::callback::CallbackManager;
use super::config::RagAnythingConfig;
use super::context::{ContextConfig, ContextExtractor};
use super::model::{ContentItem, ModalProcessResult, ParseResult};
use super::parser::{ParseCache, ExternalParserRegistry};
use super::processor::ModalProcessorRegistry;
use super::util::{get_file_reference, take_chars};

/// Core document processor orchestrating the full multimodal pipeline.
/// Flow: parse → separate_content → insert_text → process_modals → update_status
pub struct DocumentProcessor {
    config: RagAnythingConfig,
    /// Held for API compatibility; unused there as well.
    #[allow(dead_code)]
    llm_service: Arc<dyn LlmService>,
    kv_storage: Arc<dyn KVStorage>,
    vector_storage: Arc<dyn VectorStorage>,
    /// Held for API compatibility; unused there as well.
    #[allow(dead_code)]
    graph_storage: Arc<dyn GraphStorage>,
    callback_manager: Option<Arc<CallbackManager>>,
    parse_cache: Option<Arc<ParseCache>>,
    /// Business line bound at construction; passed as the biz filter on
    /// every storage call.
    biz_id: String,
    /// Lazy parser.
    parser: OnceLock<Arc<dyn super::parser::MultimodalParser>>,
    /// Lazy context extractor.
    context_extractor: OnceLock<ContextExtractor>,
}

impl DocumentProcessor {
    pub fn new(
        config: RagAnythingConfig,
        llm_service: Arc<dyn LlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        callback_manager: Option<Arc<CallbackManager>>,
        parse_cache: Option<Arc<ParseCache>>,
        biz_id: String,
    ) -> Self {
        Self {
            config,
            llm_service,
            kv_storage,
            vector_storage,
            graph_storage,
            callback_manager,
            parse_cache,
            biz_id,
            parser: OnceLock::new(),
            context_extractor: OnceLock::new(),
        }
    }

    fn parser(&self) -> easyrag_core::Result<&Arc<dyn super::parser::MultimodalParser>> {
        //: resolution errors surface on first access.
        if self.parser.get().is_none() {
            let parser = ExternalParserRegistry::get_parser(&self.config.parser)?;
            let _ = self.parser.set(parser);
        }
        Ok(self.parser.get().expect("parser initialized above"))
    }

    fn context_extractor(&self) -> &ContextExtractor {
        self.context_extractor.get_or_init(|| {
            ContextExtractor::new(ContextConfig {
                context_window: self.config.context_window as i64,
                context_mode: self.config.context_mode.clone(),
                max_context_tokens: self.config.max_context_tokens,
                include_headers: self.config.include_headers,
                include_captions: self.config.include_captions,
                ..Default::default()
            })
        })
    }

    /// Process a document completely: parse, insert text, process
    /// multimodal content. Returns processing result metadata.
    pub async fn process_document_complete(
        &self,
        file_path: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<JsonMap> {
        let file = Path::new(file_path);
        if !file.exists() {
            return Err(CoreError::InvalidArgument(format!(
                "File not found: {file_path}"
            )));
        }

        let start_time = Instant::now();
        let file_ref = get_file_reference(file_path, self.config.use_full_path);

        tracing::info!("Starting complete document processing: {}", file_ref);

        // Step 1: Parse document
        let parse_result = self.parse_document(file_path, kwargs).await?;
        let content_list = &parse_result.content_list;
        let doc_id = generate_content_based_doc_id(content_list);

        tracing::info!("Parsed {} content blocks, docId={}", content_list.len(), doc_id);

        // Step 2: Separate content into text and multimodal
        let text_items: Vec<&ContentItem> = content_list
            .iter()
            .filter(|i| matches!(i, ContentItem::Text { .. }))
            .collect();
        let multimodal_items: Vec<&ContentItem> = content_list
            .iter()
            .filter(|i| !matches!(i, ContentItem::Text { .. }))
            .collect();

        tracing::info!(
            "Content separated: {} text items, {} multimodal items",
            text_items.len(),
            multimodal_items.len()
        );

        // Step 3: Insert text content
        self.insert_text_content(&text_items, &doc_id, file_path)
            .await?;

        // Step 4: Process multimodal content
        let multimodal_results = self
            .process_multimodal_content(&multimodal_items, content_list, &doc_id, file_path)
            .await;

        let duration = start_time.elapsed().as_secs_f64();

        let mut result = JsonMap::new();
        result.insert("doc_id".to_string(), JsonValue::from(doc_id.clone()));
        result.insert("file_path".to_string(), JsonValue::from(file_path.to_string()));
        result.insert(
            "total_blocks".to_string(),
            JsonValue::from(content_list.len() as u64),
        );
        result.insert(
            "text_blocks".to_string(),
            JsonValue::from(text_items.len() as u64),
        );
        result.insert(
            "multimodal_blocks".to_string(),
            JsonValue::from(multimodal_items.len() as u64),
        );
        result.insert(
            "multimodal_processed".to_string(),
            JsonValue::from(multimodal_results.len() as u64),
        );
        result.insert(
            "processing_time_seconds".to_string(),
            JsonValue::from(duration),
        );
        result.insert("status".to_string(), JsonValue::from("completed"));

        tracing::info!("Document processing complete: {} ({}s)", file_ref, duration);
        Ok(result)
    }

    /// Parse document with caching support.
    pub async fn parse_document(
        &self,
        file_path: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
        let file = Path::new(file_path);
        if !file.exists() {
            return Err(CoreError::InvalidArgument(format!(
                "File not found: {file_path}"
            )));
        }

        if let Some(cm) = &self.callback_manager {
            cm.emit_parse_start(file_path, &self.config.parser);
        }
        let parse_start = Instant::now();

        // Check cache
        if let Some(parse_cache) = &self.parse_cache {
            let cache_key = parse_cache.generate_cache_key(
                file,
                &self.config.parser,
                Some(self.config.parse_method.as_str()),
                kwargs,
            );
            let cached = parse_cache
                .get_cached_result(
                    &cache_key,
                    file,
                    &self.config.parser,
                    Some(self.config.parse_method.as_str()),
                    kwargs,
                )
                .await;
            if let Some(cached) = cached {
                tracing::info!("Using cached parsing result for: {}", file_path);
                let duration_ms = parse_start.elapsed().as_millis() as i64;
                if let Some(cm) = &self.callback_manager {
                    cm.emit_parse_complete(file_path, cached.content_list.len(), "", duration_ms);
                }
                return Ok(cached);
            }
        }

        // Parse with appropriate method
        let parser = self.parser()?;
        let parse_result = parser
            .parse(file, Some(self.config.parse_method.as_str()), kwargs)
            .await?;

        if parse_result.content_list.is_empty() {
            return Err(CoreError::Internal(format!(
                "Parsing failed: No content was extracted from {file_path}"
            )));
        }

        // Store in cache
        if let Some(parse_cache) = &self.parse_cache {
            let doc_id = generate_content_based_doc_id(&parse_result.content_list);
            let cache_key = parse_cache.generate_cache_key(
                file,
                &self.config.parser,
                Some(self.config.parse_method.as_str()),
                kwargs,
            );
            parse_cache
                .store_cached_result(
                    &cache_key,
                    &parse_result,
                    &doc_id,
                    file,
                    &self.config.parser,
                    Some(self.config.parse_method.as_str()),
                    kwargs,
                )
                .await;
        }

        let duration_ms = parse_start.elapsed().as_millis() as i64;
        if let Some(cm) = &self.callback_manager {
            cm.emit_parse_complete(file_path, parse_result.content_list.len(), "", duration_ms);
        }

        Ok(parse_result)
    }

    /// Insert text content into the knowledge graph via storage.
    /// Returns the created chunk id (empty when there is no text).
    pub async fn insert_text_content(
        &self,
        text_items: &[&ContentItem],
        doc_id: &str,
        file_path: &str,
    ) -> easyrag_core::Result<String> {
        if text_items.is_empty() {
            tracing::debug!("No text content to insert");
            return Ok(String::new());
        }

        if let Some(cm) = &self.callback_manager {
            cm.emit_text_insert_start(doc_id, text_items.len());
        }
        let insert_start = Instant::now();

        // Combine text items into chunks for insertion
        let texts: Vec<&str> = text_items
            .iter()
            .map(|item| match item {
                ContentItem::Text { text, .. } => text.as_str(),
                _ => "",
            })
            .collect();
        let text_content = texts.join("\n\n");

        // Store as a chunk in KV storage
        let chunk_id = compute_mdhash_id(&text_content, "chunk-");
        // `textContent.length / 4` (char count based, project convention)
        let tokens = text_content.chars().count() / 4;

        let mut chunk_data = JsonMap::new();
        chunk_data.insert("tokens".to_string(), JsonValue::from(tokens as u64));
        chunk_data.insert("content".to_string(), JsonValue::from(text_content.clone()));
        chunk_data.insert("chunk_order_index".to_string(), JsonValue::from(0));
        chunk_data.insert("full_doc_id".to_string(), JsonValue::from(doc_id.to_string()));
        chunk_data.insert("file_path".to_string(), JsonValue::from(file_path.to_string()));
        let mut kv_data = std::collections::HashMap::new();
        kv_data.insert(chunk_id.clone(), chunk_data);
        self.kv_storage.upsert(kv_data, &self.biz_id).await?;

        // Store in vector database for retrieval
        let mut vdb_data = JsonMap::new();
        vdb_data.insert("content".to_string(), JsonValue::from(text_content));
        vdb_data.insert("full_doc_id".to_string(), JsonValue::from(doc_id.to_string()));
        vdb_data.insert("tokens".to_string(), JsonValue::from(tokens as u64));
        vdb_data.insert("chunk_order_index".to_string(), JsonValue::from(0));
        vdb_data.insert("file_path".to_string(), JsonValue::from(file_path.to_string()));
        let mut vector_data = std::collections::HashMap::new();
        vector_data.insert(chunk_id.clone(), vdb_data);
        self.vector_storage.upsert(vector_data, &self.biz_id).await?;

        if let Some(cm) = &self.callback_manager {
            let duration_ms = insert_start.elapsed().as_millis() as i64;
            cm.emit_text_insert_complete(doc_id, text_items.len(), duration_ms);
        }
        tracing::info!("Inserted {} text items as chunk {}", text_items.len(), chunk_id);

        Ok(chunk_id)
    }

    /// Process multimodal content items using specialized processors.
    pub async fn process_multimodal_content(
        &self,
        multimodal_items: &[&ContentItem],
        full_content_list: &[ContentItem],
        doc_id: &str,
        file_path: &str,
    ) -> Vec<ModalProcessResult> {
        if multimodal_items.is_empty() {
            tracing::debug!("No multimodal content to process");
            return Vec::new();
        }

        if let Some(cm) = &self.callback_manager {
            cm.emit_modal_process_start(file_path, "", multimodal_items.len());
        }
        let process_start = Instant::now();
        let mut results = Vec::new();
        let semaphore = Semaphore::new(self.config.max_concurrent_files.max(1));

        for (index, item) in multimodal_items.iter().enumerate() {
            // Check if processing is enabled for this type
            if !self.is_processing_enabled(item) {
                tracing::debug!("Processing disabled for type: {:?}", item.modal_type());
                continue;
            }

            let Some(processor) = ModalProcessorRegistry::get_processor(item.modal_type()) else {
                tracing::warn!("No processor registered for type: {:?}", item.modal_type());
                continue;
            };

            // semaphore.withPermit { ... } inside a sequential loop
            let Ok(permit) = semaphore.acquire().await else {
                continue;
            };
            let _permit = permit;

            // Extract context for this item
            let mut item_info = JsonMap::new();
            item_info.insert("page_idx".to_string(), JsonValue::from(item.page_idx()));
            item_info.insert(
                "index".to_string(),
                JsonValue::from(item.content_list_index()),
            );
            let context = self
                .context_extractor()
                .extract_context(full_content_list, &item_info);

            let mut file_info = JsonMap::new();
            file_info.insert("file_path".to_string(), JsonValue::from(file_path.to_string()));
            file_info.insert("doc_id".to_string(), JsonValue::from(doc_id.to_string()));
            file_info.insert("chunk_order_index".to_string(), JsonValue::from(index as u64));

            match processor.process(item, &file_info, &context).await {
                Ok(result) => {
                    tracing::debug!("Processed {:?} item: {}", item.modal_type(), result.entity_name);
                    results.push(result);
                }
                Err(e) => {
                    tracing::error!(
                        "Error processing {:?} item at index {}: {}",
                        item.modal_type(),
                        index,
                        e
                    );
                }
            }
        }

        if let Some(cm) = &self.callback_manager {
            let duration_ms = process_start.elapsed().as_millis() as i64;
            let failed = multimodal_items.len().saturating_sub(results.len());
            cm.emit_modal_process_complete(file_path, results.len(), failed, duration_ms);
        }
        tracing::info!(
            "Processed {}/{} multimodal items",
            results.len(),
            multimodal_items.len()
        );

        results
    }

    // --- Private helpers ---

    fn is_processing_enabled(&self, item: &ContentItem) -> bool {
        match item.modal_type() {
            super::model::ModalType::Image => self.config.enable_image_processing,
            super::model::ModalType::Table => self.config.enable_table_processing,
            super::model::ModalType::Equation => self.config.enable_equation_processing,
            _ => true,
        }
    }
}

/// Content-based document id: hash of a signature built from all content
/// items.
pub fn generate_content_based_doc_id(content_list: &[ContentItem]) -> String {
    let parts: Vec<String> = content_list
        .iter()
        .map(|item| match item {
            ContentItem::Text { text, .. } => text.trim().to_string(),
            ContentItem::Image { img_path, .. } => format!("image:{img_path}"),
            ContentItem::Table { table_body, .. } => {
                format!("table:{}", take_chars(table_body, 200))
            }
            ContentItem::Equation { equation_text, .. } => {
                format!("equation:{equation_text}")
            }
            ContentItem::Generic { content, .. } => {
                format!("generic:{}", take_chars(content, 200))
            }
        })
        .collect();
    let content_signature = parts.join("\n");
    compute_mdhash_id(&content_signature, "doc-")
}

/// Resolve a path the way does (no
/// normalization): absolute paths pass through, relative paths are
/// resolved against the current working directory.
pub(crate) fn absolute_path_string(path: &str) -> String {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p.to_string_lossy().to_string()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&p).to_string_lossy().to_string())
            .unwrap_or_else(|_| p.to_string_lossy().to_string())
    }
}
