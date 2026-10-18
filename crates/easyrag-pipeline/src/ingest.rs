//! Document ingestion pipeline:
//! clean → chunk → (structure) → embed/upsert → extract, with per-document
//! options, bounded concurrency and cancellation support.

use crate::chunker::{Chunker, ChunkerConfig, ChunkerRegistry};
use crate::extraction::{EntityExtractor, EntityExtractorConfig};
use crate::parser::DocumentProcessingOptions;
use crate::repository::{ArchiveMeta, DocStatusRepository, StructureIndexRepository};
use crate::structure::StructureTreeBuilder;
use crate::summarizer::DocumentSummarizer;
use easyrag_core::llm::embedding_func::EmbeddingFunc;
use easyrag_core::llm::service::LlmService;
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::section_path_of;
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, VectorStorage,
};
use easyrag_core::util::hash::compute_mdhash_id;
use easyrag_core::util::json::to_json;
use easyrag_core::util::text::clean_text_for_extraction;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
}

/// Extension parameters for document insertion (R1/R3/R4), all optional;
/// legacy callers pass `None`.
#[derive(Clone, Debug, Default)]
pub struct InsertDocumentExtras {
    /// R3: client business key; stored on doc and status records.
    pub external_id: Option<String>,
    /// R3: stable docId (upsert derives it from external_id).
    pub doc_id_override: Option<String>,
    /// R1: business creation time, epoch seconds.
    pub create_time: Option<i64>,
    /// R4: client metadata merged into every chunk.
    pub user_metadata: Option<HashMap<String, String>>,
    /// R1: archive flag (memory consolidation). Default false.
    pub archived: Option<bool>,
    /// R1: hierarchy level. Default 0.
    pub level: Option<i32>,
    /// R1: docIds absorbed by a consolidated document. Default empty.
    pub ref_ids: Option<Vec<String>>,
    /// R1: docId of the consolidated document that absorbed this source.
    pub merged_into: Option<String>,
}

impl InsertDocumentExtras {
    /// Extract the archive-related fields (R1) for the status row write.
    pub fn archive_meta(&self) -> ArchiveMeta {
        ArchiveMeta {
            archived: self.archived.unwrap_or(false),
            level: self.level.unwrap_or(0),
            ref_ids: self.ref_ids.clone().unwrap_or_default(),
            merged_into: self.merged_into.clone(),
        }
    }
}

/// RAII guard that clears an `AtomicBool` flag on drop, mirroring the
/// blueprint's `try/finally` state resets.
struct FlagGuard<'a>(&'a AtomicBool);

impl Drop for FlagGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Document ingestion pipeline — staged pipeline
/// (clean → chunk → embed/upsert → extract).
pub struct IngestionPipeline {
    llm_service: Arc<dyn LlmService>,
    kv_storage: Arc<dyn KVStorage>,
    vector_storage: Arc<dyn VectorStorage>,
    graph_storage: Arc<dyn GraphStorage>,
    doc_status_repo: DocStatusRepository,
    embedding_func: Arc<EmbeddingFunc>,
    chunker: Arc<dyn Chunker>,
    #[allow(dead_code)]
    language: String,
    max_concurrency: usize,
    /// Bounded-channel capacity for the streaming `PipelineService` (read via
    /// [`IngestionPipeline::queue_size`]); the batch path does not use it.
    queue_size: usize,
    keyword_search: Option<Arc<dyn KeywordSearchable>>,
    structure_tree_builder: StructureTreeBuilder,
    extractor: EntityExtractor,

    #[allow(dead_code)]
    is_running: AtomicBool,
    cancel_requested: AtomicBool,
    pipeline_active: AtomicBool,
    /// Documents currently in flight in the streaming `TaskProcessor`
    /// (always 0 when streaming is disabled). Folded into
    /// [`IngestionPipeline::pipeline_status`]'s `busy` so a persistent consumer
    /// still reports busy while it has work, without touching
    /// `check_pipeline_busy`.
    active_count: AtomicUsize,
    last_processed_count: AtomicUsize,
    last_run_at: Mutex<Option<String>>,
}

impl IngestionPipeline {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm_service: Arc<dyn LlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        doc_status_storage: Arc<dyn DocStatusStorage>,
        embedding_func: Arc<EmbeddingFunc>,
    ) -> Self {
        let extractor = EntityExtractor::new(
            Arc::clone(&llm_service),
            Arc::clone(&graph_storage),
            Arc::clone(&vector_storage),
            Arc::clone(&kv_storage),
            EntityExtractorConfig::default(),
        );
        Self {
            llm_service,
            kv_storage,
            vector_storage,
            graph_storage,
            doc_status_repo: DocStatusRepository::new(doc_status_storage),
            embedding_func,
            chunker: ChunkerRegistry::create("default", ChunkerConfig::default())
                .expect("default chunker"),
            language: "English".to_string(),
            max_concurrency: 4,
            queue_size: 100,
            keyword_search: None,
            structure_tree_builder: StructureTreeBuilder::default(),
            extractor,
            is_running: AtomicBool::new(false),
            cancel_requested: AtomicBool::new(false),
            pipeline_active: AtomicBool::new(false),
            active_count: AtomicUsize::new(0),
            last_processed_count: AtomicUsize::new(0),
            last_run_at: Mutex::new(None),
        }
    }

    pub fn with_chunker(mut self, chunker: Arc<dyn Chunker>) -> Self {
        self.chunker = chunker;
        self
    }

    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    pub fn with_queue_size(mut self, queue_size: usize) -> Self {
        self.queue_size = queue_size;
        self
    }

    pub fn with_keyword_search(mut self, keyword_search: Arc<dyn KeywordSearchable>) -> Self {
        self.keyword_search = Some(keyword_search);
        self
    }

    pub fn with_structure_tree_builder(mut self, builder: StructureTreeBuilder) -> Self {
        self.structure_tree_builder = builder;
        self
    }

    pub fn with_extractor(mut self, extractor: EntityExtractor) -> Self {
        self.extractor = extractor;
        self
    }

    /// Pipeline state for external monitoring.
    pub fn pipeline_status(&self) -> JsonMap {
        let mut map = JsonMap::new();
        // `busy` keeps its "only while actually processing" contract: the batch
        // flag OR any in-flight streaming document. With streaming disabled
        // `active_count` is always 0, so this is identical to the previous
        // `pipeline_active` read.
        let active = self.active_count.load(Ordering::SeqCst);
        map.insert(
            "busy".into(),
            JsonValue::Bool(self.pipeline_active.load(Ordering::SeqCst) || active > 0),
        );
        map.insert("active_count".into(), JsonValue::from(active as u64));
        map.insert(
            "cancel_requested".into(),
            JsonValue::Bool(self.cancel_requested.load(Ordering::SeqCst)),
        );
        map.insert(
            "docs_processed".into(),
            JsonValue::from(self.last_processed_count.load(Ordering::SeqCst) as u64),
        );
        let last_run = self.last_run_at.lock().unwrap().clone();
        map.insert(
            "last_run_at".into(),
            match last_run {
                Some(t) => JsonValue::String(t),
                None => JsonValue::Null,
            },
        );
        map
    }

    /// Documents currently in flight in the streaming processor (0 when the
    /// streaming path is disabled).
    pub(crate) fn active_count(&self) -> usize {
        self.active_count.load(Ordering::SeqCst)
    }

    /// Record one more in-flight streaming document.
    pub(crate) fn incr_active(&self) {
        self.active_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Record one fewer in-flight streaming document.
    pub(crate) fn decr_active(&self) {
        self.active_count.fetch_sub(1, Ordering::SeqCst);
    }

    /// Global concurrency ceiling; the streaming `Semaphore` is built with the
    /// same permit count the batch path uses (invariant: concurrency unchanged).
    pub(crate) fn max_concurrency(&self) -> usize {
        self.max_concurrency
    }

    /// Bounded-channel capacity default for the streaming `PipelineService`.
    pub(crate) fn queue_size(&self) -> usize {
        self.queue_size
    }

    /// Commit every storage index. This is the tail of
    /// [`IngestionPipeline::process_document_inner`], factored out so graceful
    /// shutdown can force one final commit after in-flight documents drain.
    pub(crate) async fn commit_indexes(&self) -> easyrag_core::Result<()> {
        self.kv_storage.index_done_callback().await?;
        self.vector_storage.index_done_callback().await?;
        self.graph_storage.index_done_callback().await?;
        if let Some(keyword_search) = &self.keyword_search {
            keyword_search.commit_index().await?;
        }
        Ok(())
    }

    /// Insert a document into the pipeline. Supports per-document
    /// processing options (chunk_method, skip_kg, etc.).
    /// Returns the document ID.
    pub async fn insert_document(
        &self,
        content: &str,
        file_path: &str,
        options: Option<&DocumentProcessingOptions>,
        extras: Option<&InsertDocumentExtras>,
        biz_id: &str,
    ) -> easyrag_core::Result<String> {
        let doc_id = match extras.and_then(|e| e.doc_id_override.as_deref()) {
            Some(id) => id.to_string(),
            None => compute_mdhash_id(content, "doc-"),
        };
        // R1: client-supplied business time takes precedence over ingestion time.
        let create_time = extras.and_then(|e| e.create_time);
        let create_time_iso = match create_time.and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
        {
            Some(dt) => dt.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
            None => now_iso(),
        };
        let external_id = extras.and_then(|e| e.external_id.as_deref());
        let user_metadata = extras.and_then(|e| e.user_metadata.as_ref());

        // Store full document with optional processing options.
        let mut doc_data = JsonMap::new();
        doc_data.insert("content".into(), JsonValue::String(content.to_string()));
        doc_data.insert("file_path".into(), JsonValue::String(file_path.to_string()));
        doc_data.insert("create_time".into(), JsonValue::String(create_time_iso.clone()));
        if let Some(ts) = create_time {
            doc_data.insert("create_time_epoch".into(), JsonValue::from(ts));
        }
        if let Some(ext) = external_id {
            doc_data.insert("external_id".into(), JsonValue::String(ext.to_string()));
        }
        if let Some(meta) = user_metadata.filter(|m| !m.is_empty()) {
            doc_data.insert(
                "user_metadata".into(),
                JsonValue::Object(
                    meta.iter()
                        .map(|(k, v)| (k.clone(), JsonValue::String(v.clone())))
                        .collect(),
                ),
            );
        }
        // R1: archive-related fields travel with the document payload too, so
        // chunk metadata and doc KV stay consistent with the status row.
        if let Some(extras) = extras {
            if extras.archived.unwrap_or(false) {
                doc_data.insert("archived".into(), JsonValue::Bool(true));
            }
            if let Some(level) = extras.level.filter(|l| *l != 0) {
                doc_data.insert("level".into(), JsonValue::from(level));
            }
            if let Some(ref_ids) = extras.ref_ids.as_ref().filter(|r| !r.is_empty()) {
                doc_data.insert(
                    "ref_ids".into(),
                    JsonValue::Array(
                        ref_ids.iter().map(|id| JsonValue::String(id.clone())).collect(),
                    ),
                );
            }
            if let Some(merged_into) = &extras.merged_into {
                doc_data.insert("merged_into".into(), JsonValue::String(merged_into.clone()));
            }
        }
        if let Some(opts) = options {
            if let Some(m) = &opts.chunk_method {
                doc_data.insert("chunk_method".into(), JsonValue::String(m.clone()));
            }
            if let Some(s) = opts.chunk_token_size {
                doc_data.insert("chunk_token_size".into(), JsonValue::from(s as u64));
            }
            if let Some(o) = opts.chunk_overlap {
                doc_data.insert("chunk_overlap".into(), JsonValue::from(o as u64));
            }
            if opts.skip_kg {
                doc_data.insert("skip_kg".into(), JsonValue::Bool(true));
            }
            if opts.build_structure {
                doc_data.insert("build_structure".into(), JsonValue::Bool(true));
            }
        }
        let mut kv_batch = HashMap::new();
        kv_batch.insert(doc_id.clone(), doc_data);
        self.kv_storage.upsert(kv_batch, biz_id).await?;

        // Update doc status.
        let archive_meta = extras.map(|e| e.archive_meta());
        self.doc_status_repo
            .mark_pending(
                &doc_id,
                content,
                file_path,
                &create_time_iso,
                external_id,
                archive_meta.as_ref(),
                biz_id,
            )
            .await?;

        Ok(doc_id)
    }

    /// Rebuild per-document processing options from stored doc KV data.
    pub fn rebuild_options_from_doc_data(doc_data: &JsonMap) -> DocumentProcessingOptions {
        DocumentProcessingOptions {
            chunk_method: doc_data
                .get("chunk_method")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            chunk_token_size: doc_data
                .get("chunk_token_size")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            chunk_overlap: doc_data
                .get("chunk_overlap")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
            skip_kg: doc_data.get("skip_kg").and_then(|v| v.as_bool()) == Some(true),
            build_structure: doc_data.get("build_structure").and_then(|v| v.as_bool())
                == Some(true),
            parser_engine: None,
        }
    }

    /// Extract client-supplied metadata stored on the doc record (R4).
    pub fn extract_user_metadata(doc_data: &JsonMap) -> Option<HashMap<String, String>> {
        let map = doc_data.get("user_metadata").and_then(|v| v.as_object())?;
        let extracted: HashMap<String, String> = map
            .iter()
            .map(|(k, v)| {
                let s = match v {
                    JsonValue::String(s) => s.clone(),
                    JsonValue::Null => String::new(),
                    other => other.to_string(),
                };
                (k.clone(), s)
            })
            .collect();
        if extracted.is_empty() {
            None
        } else {
            Some(extracted)
        }
    }

    /// Process a single document through the full pipeline. Respects
    /// per-document options: chunk_method, chunk_token_size, skip_kg.
    ///
    /// `create_time_override`: business create_time (epoch seconds) applied to
    /// all chunks (R1). `user_metadata`: client metadata merged into each
    /// chunk; chunker system keys win on conflict (R4).
    pub async fn process_document(
        &self,
        doc_id: &str,
        content: &str,
        file_path: &str,
        options: Option<&DocumentProcessingOptions>,
        create_time_override: Option<i64>,
        user_metadata: Option<&HashMap<String, String>>,
        biz_id: &str,
    ) {
        if let Err(e) = self
            .process_document_inner(
                doc_id,
                content,
                file_path,
                options,
                create_time_override,
                user_metadata,
                biz_id,
            )
            .await
        {
            tracing::error!("Failed to process document {doc_id}: {e}");
            self.update_doc_status(
                doc_id,
                DocStatus::Failed,
                None,
                None,
                Some(&e.to_string()),
                biz_id,
            )
            .await;
        }
    }

    async fn process_document_inner(
        &self,
        doc_id: &str,
        content: &str,
        file_path: &str,
        options: Option<&DocumentProcessingOptions>,
        create_time_override: Option<i64>,
        user_metadata: Option<&HashMap<String, String>>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        // Update status to PROCESSING.
        self.update_doc_status(doc_id, DocStatus::Processing, None, None, None, biz_id)
            .await;

        // Stage 0: Text cleaning — remove control chars, normalize
        // whitespace, validate length.
        let cleaned_content = clean_text_for_extraction(content, None);
        let cleaned_content = match cleaned_content {
            Some(c) if !c.trim().is_empty() => c,
            _ => {
                self.update_doc_status(
                    doc_id,
                    DocStatus::Failed,
                    None,
                    None,
                    Some("Document empty after text cleaning"),
                    biz_id,
                )
                .await;
                return Ok(());
            }
        };

        // R2/R1: chunk rows mirror the document's archive fields so vector
        // recall can filter archived memories without joining doc_status at
        // query time, and `query/data` chunks can echo the four fields. The doc
        // KV row is written before processing starts and is the source of truth
        // (ingest and reprocess both go through it).
        let doc_row = self.kv_storage.get_by_id(doc_id, biz_id).await.ok().flatten();
        let archived = doc_row
            .as_ref()
            .and_then(|d| d.get("archived"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let level = doc_row
            .as_ref()
            .and_then(|d| d.get("level"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let ref_ids = doc_row
            .as_ref()
            .and_then(|d| d.get("ref_ids"))
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let merged_into = doc_row
            .as_ref()
            .and_then(|d| d.get("merged_into"))
            .and_then(|v| v.as_str())
            .map(str::to_string);

        // Stage 1: Chunk the document (per-document chunker if specified).
        let active_chunker: Arc<dyn Chunker> = if let Some(method) =
            options.and_then(|o| o.chunk_method.as_deref())
        {
            ChunkerRegistry::create(
                method,
                ChunkerConfig {
                    chunk_token_size: options.and_then(|o| o.chunk_token_size),
                    chunk_overlap_token_size: options.and_then(|o| o.chunk_overlap),
                    embedding_func: Some(Arc::clone(&self.embedding_func)),
                    ..Default::default()
                },
            )?
        } else {
            Arc::clone(&self.chunker)
        };
        let chunks = active_chunker.chunk(&cleaned_content, doc_id).await?;
        tracing::info!("Document {doc_id} split into {} chunks", chunks.len());

        // Store chunks in KV storage. The insertion order is recorded
        // separately because `HashMap` iteration is unordered while the
        // blueprint iterates a `LinkedHashMap` (chunk order is
        // observable in `chunks_list` and KG merge order).
        let mut chunk_data: HashMap<String, JsonMap> = HashMap::new();
        let mut chunk_order: Vec<String> = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let chunk_id = compute_mdhash_id(&chunk.content, "chunk-");
            let mut data = JsonMap::new();
            data.insert("content".into(), JsonValue::String(chunk.content.clone()));
            data.insert("full_doc_id".into(), JsonValue::String(doc_id.to_string()));
            data.insert(
                "chunk_order_index".into(),
                JsonValue::from(chunk.chunk_order_index as u64),
            );
            data.insert("tokens".into(), JsonValue::from(chunk.tokens as u64));
            data.insert("file_path".into(), JsonValue::String(file_path.to_string()));
            data.insert(
                "create_time".into(),
                // R1: business time override, else ingestion time (epoch seconds).
                JsonValue::from(
                    create_time_override.unwrap_or_else(|| chrono::Utc::now().timestamp()),
                ),
            );
            // R4: user metadata goes underneath so chunker system keys
            // (section_path, ...) always take precedence on conflict.
            let mut merged_metadata = JsonMap::new();
            if let Some(um) = user_metadata {
                for (k, v) in um {
                    merged_metadata.insert(k.clone(), JsonValue::String(v.clone()));
                }
            }
            merged_metadata.extend(chunk.metadata.clone());
            data.insert("metadata".into(), JsonValue::Object(merged_metadata));
            // R2/R1: archive fields ride on every chunk so the vector backend
            // can filter archived memories without joining doc_status, and so
            // query/data hits can echo them back. Absent values keep the wire
            // defaults (active, level 0, no refs, not merged).
            if archived {
                data.insert("archived".into(), JsonValue::Bool(true));
            }
            if level != 0 {
                data.insert("level".into(), JsonValue::from(level));
            }
            if !ref_ids.is_empty() {
                data.insert("ref_ids".into(), JsonValue::Array(ref_ids.clone()));
            }
            if let Some(merged_into) = &merged_into {
                data.insert(
                    "merged_into".into(),
                    JsonValue::String(merged_into.clone()),
                );
            }
            chunk_data.insert(chunk_id.clone(), data);
            chunk_order.push(chunk_id);
        }
        self.kv_storage.upsert(chunk_data.clone(), biz_id).await?;

        // Stage 1.5: Build document structure index (optional).
        if options.map(|o| o.build_structure).unwrap_or(false) {
            let mut structure_index = self
                .structure_tree_builder
                .build(&cleaned_content, doc_id, None);
            if let Some(index) = structure_index.take() {
                // Enrich with LLM-generated summaries for better
                // tree-search reasoning.
                let index = {
                    let summarizer = DocumentSummarizer::new(Arc::clone(&self.llm_service));
                    let doc_description = summarizer.generate_doc_description(&cleaned_content).await;
                    let enriched_sections = summarizer
                        .generate_section_summaries(index.sections.clone(), &cleaned_content)
                        .await;
                    easyrag_core::model::document_structure::DocumentStructureIndex {
                        doc_description,
                        sections: enriched_sections,
                        ..index
                    }
                };
                StructureIndexRepository::new(&self.kv_storage)
                    .save(doc_id, &to_json(&index), biz_id)
                    .await?;
                // M5: mark document in the registry as structure-indexed.
                self.doc_status_repo.mark_has_structure(doc_id, biz_id).await?;
                tracing::info!(
                    "Document {doc_id} structure index built ({} top-level sections)",
                    index.sections.len()
                );
            }
        }

        // Stage 2: Embed chunks and store in vector storage.
        self.vector_storage.upsert(chunk_data.clone(), biz_id).await?;

        // Stage 2b: Index chunks for keyword search (if enabled).
        if let Some(keyword_search) = &self.keyword_search {
            keyword_search.index_documents(chunk_data.clone(), biz_id).await?;
        }

        // Stage 3: Extract entities and relationships (skip if skip_kg).
        if options.map(|o| o.skip_kg).unwrap_or(false) {
            tracing::info!("Document {doc_id} skipping KG extraction (skip_kg=true)");
        } else {
            for chunk_id in &chunk_order {
                let data = &chunk_data[chunk_id];
                let chunk_content = match data.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => continue,
                };
                let section_path = section_path_of(data).unwrap_or_default();
                let result = self
                    .extractor
                    .extract(chunk_content, chunk_id, &section_path)
                    .await?;
                self.extractor.merge_to_graph(&result, chunk_id, biz_id).await?;
            }
        }

        // Update status to PROCESSED.
        let chunks_list = chunk_order;
        self.update_doc_status(
            doc_id,
            DocStatus::Processed,
            Some(chunks.len()),
            Some(chunks_list),
            None,
            biz_id,
        )
        .await;

        // Commit all storage operations.
        self.commit_indexes().await?;

        tracing::info!(
            "Document {doc_id} processed successfully ({} chunks)",
            chunks.len()
        );
        Ok(())
    }

    /// Start the pipeline to process pending documents of the given biz line.
    /// Thread-safe: only one pipeline execution at a time.
    pub async fn start(self: &Arc<Self>, biz_id: &str) {
        if self
            .pipeline_active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::info!("Pipeline already running, skipping start request");
            return;
        }

        let result = self.run_pipeline(biz_id).await;
        if let Err(e) = result {
            tracing::error!("Pipeline run failed: {e}");
        }
    }

    async fn run_pipeline(self: &Arc<Self>, biz_id: &str) -> easyrag_core::Result<()> {
        // RAII guards mirror the blocks so the flags
        // are reset even on early `?` returns or panics (otherwise the
        // pipeline would stay "busy" forever).
        let _active_guard = FlagGuard(&self.pipeline_active);
        let _running_guard = FlagGuard(&self.is_running);
        self.is_running.store(true, Ordering::SeqCst);
        self.cancel_requested.store(false, Ordering::SeqCst);
        *self.last_run_at.lock().unwrap() = Some(now_iso());

        let pending_docs = self
            .doc_status_repo
            .storage()
            .get_docs_by_status(DocStatus::Pending, biz_id)
            .await?;
        tracing::info!(
            "Starting pipeline with {} pending documents",
            pending_docs.len()
        );

        let semaphore = Arc::new(Semaphore::new(self.max_concurrency.max(1)));
        let mut tasks = Vec::with_capacity(pending_docs.len());
        for (doc_id, doc_status) in pending_docs.iter() {
            let pipeline = Arc::clone(self);
            let semaphore = Arc::clone(&semaphore);
            let doc_id = doc_id.clone();
            // Per-document biz line: a wildcard run (empty biz_id) matches every
            // line, so each document must be read and written under its own line.
            // Never fall through to an empty biz_id — writing with one would
            // create a ghost status row keyed (id, workspace, '').
            let biz_id = if !doc_status.biz_id.is_empty() {
                doc_status.biz_id.clone()
            } else if !biz_id.is_empty() {
                biz_id.to_string()
            } else {
                tracing::warn!("Skipping pending doc {doc_id}: no biz line recorded");
                continue;
            };
            tasks.push(async move {
                if pipeline.cancel_requested.load(Ordering::SeqCst) {
                    return;
                }
                let _permit = match semaphore.acquire().await {
                    Ok(permit) => permit,
                    Err(_) => return,
                };
                if pipeline.cancel_requested.load(Ordering::SeqCst) {
                    return;
                }
                let doc_data = match pipeline.kv_storage.get_by_id(&doc_id, &biz_id).await {
                    Ok(Some(data)) => data,
                    _ => return,
                };
                let content = match doc_data.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c.to_string(),
                    None => return,
                };
                let file_path = doc_data
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                // Rebuild per-document options from stored data.
                let doc_options = Self::rebuild_options_from_doc_data(&doc_data);
                let create_time_override = doc_data
                    .get("create_time_epoch")
                    .and_then(|v| v.as_i64());
                let user_metadata = Self::extract_user_metadata(&doc_data);
                pipeline
                    .process_document(
                        &doc_id,
                        &content,
                        &file_path,
                        Some(&doc_options),
                        create_time_override,
                        user_metadata.as_ref(),
                        &biz_id,
                    )
                    .await;
            });
        }
        let processed_count = tasks.len();
        futures::future::join_all(tasks).await;

        self.last_processed_count
            .store(processed_count, Ordering::SeqCst);
        tracing::info!("Pipeline completed ({processed_count} documents processed)");
        Ok(())
    }

    /// Request cancellation of the running pipeline. Pending documents
    /// will be skipped; currently processing documents will finish.
    pub fn cancel(&self) {
        self.cancel_requested.store(true, Ordering::SeqCst);
        tracing::info!("Pipeline cancellation requested");
    }

    async fn update_doc_status(
        &self,
        doc_id: &str,
        status: DocStatus,
        chunks_count: Option<usize>,
        chunks_list: Option<Vec<String>>,
        error_msg: Option<&str>,
        biz_id: &str,
    ) {
        // A failed status write must not abort an otherwise-successful document
        // run, so the transition error is logged and swallowed here — the
        // repository returns it rather than baking in the policy. `error_msg`
        // is always written (None → null) so a stale error from a previous
        // FAILED run is cleared.
        if let Err(e) = self
            .doc_status_repo
            .mark_status(doc_id, status, chunks_count, chunks_list, error_msg, biz_id)
            .await
        {
            tracing::warn!("Failed to update status for document {doc_id}: {e}");
        }
    }
}
