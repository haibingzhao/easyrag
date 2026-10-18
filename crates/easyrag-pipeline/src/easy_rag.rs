//! Main EasyRAG class - unified entry point for RAG operations
//!
//! Provides:
//! - Document insertion and processing
//! - Query execution (local, global, hybrid, naive, mix modes)
//! - Knowledge graph management
//! - Storage orchestration

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::llm::embedding_func::EmbeddingFunc;
use easyrag_core::llm::{LlmService, RerankService};
use easyrag_core::model::doc_status::{
    DeletionResult, DeletionStatus, DocProcessingStatus, DocStatus,
};
use easyrag_core::model::types::KnowledgeGraph;
use easyrag_core::model::{QueryParam, QueryMode};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageNameSpace, VectorStorage,
};
use easyrag_core::util::hash::compute_mdhash_id;
use futures::stream::BoxStream;

use crate::chunker::{Chunker, TokenSizeChunker};
use crate::contextual_chunker::ContextualChunker;
use crate::easy_rag_config::EasyRAGConfig;
use crate::extraction::{EntityExtractor, EntityExtractorConfig};
use crate::graph_cleanup::cleanup_graph_by_chunks;
use crate::ingest::IngestionPipeline;
use crate::multimodal::TextQueryBackend;
use crate::query::{ClaimVerifier, EvidenceScorer, QueryEngine, QueryRouter};

/// Main EasyRAG class - unified entry point for RAG operations.
pub struct EasyRAG {
    pub config: EasyRAGConfig,
    vector_storage: Arc<dyn VectorStorage>,
    graph_storage: Arc<dyn GraphStorage>,
    kv_storage: Arc<dyn KVStorage>,
    doc_status_storage: Arc<dyn DocStatusStorage>,

    query_engine: Arc<QueryEngine>,
    ingestion_pipeline: Arc<IngestionPipeline>,

    initialized: AtomicBool,
}

impl EasyRAG {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: EasyRAGConfig,
        llm_service: Arc<dyn LlmService>,
        embedding_func: Arc<EmbeddingFunc>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        kv_storage: Arc<dyn KVStorage>,
        doc_status_storage: Arc<dyn DocStatusStorage>,
        rerank_service: Option<Arc<dyn RerankService>>,
        keyword_search: Option<Arc<dyn KeywordSearchable>>,
    ) -> Self {
        let mut chunker = Arc::new(TokenSizeChunker::new(
            config.chunk_size,
            config.chunk_overlap_size,
        )) as Arc<dyn Chunker>;

        // Wrap with ContextualChunker if enabled (Feature 4)
        if config.enable_contextual_chunking {
            chunker = Arc::new(ContextualChunker::new(Arc::clone(&llm_service), chunker));
        }

        let mut query_engine = QueryEngine::new(
            Arc::clone(&llm_service),
            Arc::clone(&vector_storage),
            Arc::clone(&graph_storage),
            Arc::clone(&kv_storage),
        )
        .with_language(config.language.clone());
        if let Some(rerank) = &rerank_service {
            query_engine = query_engine.with_rerank(Arc::clone(rerank));
        }
        if let Some(keyword) = &keyword_search {
            query_engine = query_engine.with_keyword_search(Arc::clone(keyword));
        }
        // Anti-hallucination guardrails (opt-in from config)
        if config.enable_verification_gate {
            let verifier = ClaimVerifier::new(Arc::clone(&llm_service))
                .with_threshold(config.verification_abstain_threshold);
            query_engine = query_engine.with_claim_verifier(Arc::new(verifier));
        }
        if config.enable_evidence_quality_scoring {
            let scorer = EvidenceScorer::new(Arc::clone(&llm_service))
                .with_thresholds(
                    config.evidence_quality_threshold_good,
                    config.evidence_quality_threshold_bad,
                );
            query_engine = query_engine.with_evidence_scorer(Arc::new(scorer));
        }
        if config.enable_query_routing {
            let router = QueryRouter::new(Arc::clone(&llm_service));
            query_engine = query_engine.with_query_router(Arc::new(router));
        }
        if config.enable_citation_validation {
            query_engine = query_engine.with_citation_validation(true);
        }
        let query_engine = Arc::new(query_engine);

        let mut pipeline_extractor_config = EntityExtractorConfig::default();
        pipeline_extractor_config.apply_ingest_settings(
            config.language.clone(),
            config.entity_extract_max_gleaning,
            &config.entity_types_guidance,
        );
        let mut ingestion_pipeline = IngestionPipeline::new(
            Arc::clone(&llm_service),
            Arc::clone(&kv_storage),
            Arc::clone(&vector_storage),
            Arc::clone(&graph_storage),
            Arc::clone(&doc_status_storage),
            Arc::clone(&embedding_func),
        )
        .with_chunker(Arc::clone(&chunker))
        .with_extractor(EntityExtractor::new(
            Arc::clone(&llm_service),
            Arc::clone(&graph_storage),
            Arc::clone(&vector_storage),
            Arc::clone(&kv_storage),
            pipeline_extractor_config,
        ))
        .with_language(config.language.clone())
        .with_max_concurrency(config.pipeline_max_parallel);
        if let Some(keyword) = &keyword_search {
            ingestion_pipeline = ingestion_pipeline.with_keyword_search(Arc::clone(keyword));
        }
        let ingestion_pipeline = Arc::new(ingestion_pipeline);

        Self {
            config,
            vector_storage,
            graph_storage,
            kv_storage,
            doc_status_storage,
            query_engine,
            ingestion_pipeline,
            initialized: AtomicBool::new(false),
        }
    }

    /// Access to the underlying query engine (for streaming etc.).
    pub fn query_engine(&self) -> &Arc<QueryEngine> {
        &self.query_engine
    }

    /// Access to the ingestion pipeline.
    pub fn ingestion_pipeline(&self) -> &Arc<IngestionPipeline> {
        &self.ingestion_pipeline
    }

    /// Initialize all storage backends.
    pub async fn initialize(&self) -> easyrag_core::Result<()> {
        // CAS claims the initialization slot; concurrent callers observe
        // `true` and return. On failure the flag is rolled back so a retry
        // can re-run.
        if self
            .initialized
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(());
        }

        tracing::info!(
            "Initializing EasyRAG: workspace={}, storage={}",
            self.config.workspace,
            self.config.storage_type
        );

        let result = async {
            self.vector_storage.initialize().await?;
            self.graph_storage.initialize().await?;
            self.kv_storage.initialize().await?;
            self.doc_status_storage.initialize().await?;
            Ok::<(), easyrag_core::CoreError>(())
        }
        .await;
        if result.is_err() {
            self.initialized.store(false, Ordering::SeqCst);
            return result;
        }

        tracing::info!("EasyRAG initialized successfully");
        Ok(())
    }

    /// Finalize and close all storage backends.
    pub async fn finalize(&self) -> easyrag_core::Result<()> {
        // CAS claims the finalization slot; on failure the flag is
        // restored
        if self
            .initialized
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(());
        }

        let result = async {
            self.vector_storage.finalize().await?;
            self.graph_storage.finalize().await?;
            self.kv_storage.finalize().await?;
            self.doc_status_storage.finalize().await?;
            Ok::<(), easyrag_core::CoreError>(())
        }
        .await;
        if result.is_err() {
            self.initialized.store(true, Ordering::SeqCst);
            return result;
        }

        tracing::info!("EasyRAG finalized");
        Ok(())
    }

    /// Insert a document into the RAG system. Returns the document id.
    pub async fn insert(
        &self,
        content: &str,
        doc_id: Option<&str>,
        metadata: &easyrag_core::JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<String> {
        self.ensure_initialized()?;

        // `docId ?: computeMdhashId(content)` (empty prefix)
        let effective_doc_id = doc_id
            .map(|s| s.to_string())
            .unwrap_or_else(|| compute_mdhash_id(content, ""));
        tracing::info!(
            "Inserting document: {} ({} chars)",
            effective_doc_id,
            content.chars().count()
        );

        let file_path = metadata
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        self.ingestion_pipeline
            .process_document(&effective_doc_id, content, file_path, None, None, None, biz_id)
            .await;

        Ok(effective_doc_id)
    }

    /// Insert multiple documents.
    pub async fn insert_batch(
        &self,
        documents: &[(String, easyrag_core::JsonMap)],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        self.ensure_initialized()?;
        let mut ids = Vec::with_capacity(documents.len());
        for (content, metadata) in documents {
            ids.push(self.insert(content, None, metadata, biz_id).await?);
        }
        Ok(ids)
    }

    /// Query the RAG system.
    pub async fn query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        self.ensure_initialized()?;
        tracing::debug!("Query: mode={:?}, topK={}", param.mode, param.top_k);
        self.query_engine.query(query, param).await
    }

    /// Query with streaming response.
    pub fn query_stream(&self, query: &str, param: &QueryParam) -> BoxStream<'static, String> {
        self.query_engine.query_stream(query, param)
    }

    /// Delete a document by ID, including all derived data:
    /// - Chunks from KV and vector storage
    /// - Entities/relationships that only reference this document
    /// - Document status record
    ///
    /// Semantics are aligned with DocumentLifecycleService.cascade_delete_document
    /// (server-side); this library API intentionally keeps its own implementation
    /// to avoid spreading server dependencies into the pipeline module.
    pub async fn delete_document(&self, doc_id: &str, biz_id: &str) -> easyrag_core::Result<DeletionResult> {
        self.ensure_initialized()?;

        let Some(doc) = self.doc_status_storage.get_by_id(doc_id, biz_id).await? else {
            return Ok(DeletionResult {
                status: DeletionStatus::NotFound,
                doc_id: doc_id.to_string(),
                message: format!("Document not found: {doc_id}"),
                status_code: 404,
                file_path: None,
            });
        };

        // 1. Get chunks associated with this document
        let chunks_list: Vec<String> = doc
            .get("chunks_list")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        // 2. Delete chunks from KV and vector storage
        if !chunks_list.is_empty() {
            self.kv_storage.delete(&chunks_list, biz_id).await?;
            self.vector_storage.delete(&chunks_list, biz_id).await?;
        }

        // 3. Clean up knowledge graph entities/relations referencing this doc
        let chunk_ids: HashSet<String> = chunks_list.iter().cloned().collect();
        self.cleanup_knowledge_graph(doc_id, &chunk_ids, biz_id).await?;

        // 4. Delete document from KV and doc status storage
        self.kv_storage
            .delete(
                &[
                    doc_id.to_string(),
                    format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"),
                ],
                biz_id,
            )
            .await?;
        self.doc_status_storage.delete(&[doc_id.to_string()], biz_id).await?;

        tracing::info!(
            "Document {} deleted with {} chunks cleaned up",
            doc_id,
            chunks_list.len()
        );

        Ok(DeletionResult {
            status: DeletionStatus::Success,
            doc_id: doc_id.to_string(),
            message: format!("Document deleted: {doc_id} ({} chunks removed)", chunks_list.len()),
            status_code: 200,
            file_path: None,
        })
    }

    /// Clean up knowledge graph after document deletion.
    /// Removes source_id references and deletes orphaned entities/relations.
    ///
    /// The sequence is shared with the server's cascade delete — see
    /// [`crate::graph_cleanup`] for why there is exactly one copy.
    async fn cleanup_knowledge_graph(
        &self,
        _doc_id: &str,
        chunk_ids: &HashSet<String>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let report = cleanup_graph_by_chunks(
            self.graph_storage.as_ref(),
            self.vector_storage.as_ref(),
            chunk_ids,
            biz_id,
        )
        .await?;

        if report.deleted_edges > 0 {
            tracing::debug!("Removed {} orphaned edges", report.deleted_edges);
        }
        if report.deleted_nodes > 0 {
            tracing::debug!("Removed {} orphaned nodes", report.deleted_nodes);
        }

        Ok(())
    }

    /// Get knowledge graph starting from a node.
    pub async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        self.ensure_initialized()?;
        self.graph_storage
            .get_knowledge_graph(node_label, max_depth, max_nodes, biz_id)
            .await
    }

    /// Get knowledge graph with the default parameters
    /// (`maxDepth = 3`, `maxNodes = 1000`).
    pub async fn get_knowledge_graph_default(
        &self,
        node_label: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        self.get_knowledge_graph(node_label, DEFAULT_KG_MAX_DEPTH, DEFAULT_KG_MAX_NODES, biz_id)
            .await
    }

    /// Get all labels in the knowledge graph.
    pub async fn get_graph_labels(&self, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        self.ensure_initialized()?;
        self.graph_storage.get_all_labels(biz_id).await
    }

    /// Get document processing status.
    pub async fn get_doc_status(
        &self,
        doc_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<DocProcessingStatus>> {
        self.ensure_initialized()?;
        let Some(doc) = self.doc_status_storage.get_by_id(doc_id, biz_id).await? else {
            return Ok(None);
        };
        let status_str = doc
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("pending");
        Ok(Some(DocProcessingStatus {
            content_summary: doc
                .get("content_summary")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            content_length: doc
                .get("content_length")
                .and_then(|v| v.as_i64())
                .unwrap_or(0),
            file_path: doc
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            status: DocStatus::from_value(status_str)?,
            created_at: doc
                .get("created_at")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            updated_at: doc
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            track_id: doc
                .get("track_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            chunks_count: doc.get("chunks_count").and_then(|v| v.as_i64()),
            chunks_list: doc
                .get("chunks_list")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            error_msg: doc
                .get("error_msg")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            metadata: doc
                .get("metadata")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default(),
            multimodal_processed: doc.get("multimodal_processed").and_then(|v| v.as_bool()),
            content_hash: doc
                .get("content_hash")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            has_structure: doc.get("has_structure").and_then(|v| v.as_bool()) == Some(true),
            external_id: doc
                .get("external_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            biz_id: biz_id.to_string(),
            archived: doc.get("archived").and_then(|v| v.as_bool()).unwrap_or(false),
            level: doc.get("level").and_then(|v| v.as_i64()).unwrap_or(0) as i32,
            ref_ids: doc
                .get("ref_ids")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_default(),
            merged_into: doc
                .get("merged_into")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
        }))
    }

    /// Get status counts for all documents.
    pub async fn get_status_counts(
        &self,
        biz_id: &str,
    ) -> easyrag_core::Result<std::collections::HashMap<String, usize>> {
        self.ensure_initialized()?;
        self.doc_status_storage.get_status_counts(biz_id).await
    }

    /// Drop all data from storage.
    ///
    /// each backend's `drop` catches failures internally
    /// and reports them through the status map, so the aggregation always
    /// covers all four backends and never short-circuits.
    pub async fn drop(
        &self,
        biz_id: &str,
    ) -> easyrag_core::Result<std::collections::HashMap<String, String>> {
        let mut results = std::collections::HashMap::new();
        // `drop` must be called via UFCS: method-call syntax `.drop`
        // resolves to the destructor `Drop::drop` (E0040).
        let vector_status =
            StorageNameSpace::drop(&*self.vector_storage as &dyn StorageNameSpace, biz_id).await;
        results.insert("vector".to_string(), drop_status_value(vector_status));
        let graph_status =
            StorageNameSpace::drop(&*self.graph_storage as &dyn StorageNameSpace, biz_id).await;
        results.insert("graph".to_string(), drop_status_value(graph_status));
        let kv_status =
            StorageNameSpace::drop(&*self.kv_storage as &dyn StorageNameSpace, biz_id).await;
        results.insert("kv".to_string(), drop_status_value(kv_status));
        let doc_status =
            StorageNameSpace::drop(&*self.doc_status_storage as &dyn StorageNameSpace, biz_id)
                .await;
        results.insert("docStatus".to_string(), drop_status_value(doc_status));
        Ok(results)
    }

    fn ensure_initialized(&self) -> easyrag_core::Result<()> {
        if !self.initialized.load(Ordering::SeqCst) {
            return Err(easyrag_core::CoreError::Internal(
                "EasyRAG not initialized. Call initialize() first.".to_string(),
            ));
        }
        Ok(())
    }
}

/// Default parameters for knowledge graph retrieval.
pub const DEFAULT_KG_MAX_DEPTH: usize = 3;
/// Default parameters for knowledge graph retrieval.
pub const DEFAULT_KG_MAX_NODES: usize = 1000;

/// Extract the `status` entry from a storage `drop` result, mapping
/// both missing keys and hard errors to `"error"`.
fn drop_status_value(
    result: easyrag_core::Result<std::collections::HashMap<String, String>>,
) -> String {
    match result {
        Ok(map) => map
            .get("status")
            .cloned()
            .unwrap_or_else(|| "error".to_string()),
        Err(_) => "error".to_string(),
    }
}

/// The EasyRAG facade is the concrete [`TextQueryBackend`] used by the
/// multimodal query engine.
#[async_trait]
impl TextQueryBackend for EasyRAG {
    async fn query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        EasyRAG::query(self, query, param).await
    }
}

/// Convenience: parse a query mode string with the same default ("mix")
/// used by the multimodal layer.
pub fn parse_query_mode(mode: &str) -> easyrag_core::Result<QueryMode> {
    QueryMode::from_value(mode)
}
