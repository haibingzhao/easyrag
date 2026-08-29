//! RAGAnything main entry point (facade)
//!
//! Composes: DocumentProcessor + MultimodalQueryEngine + BatchProcessor
//! + CallbackManager.

use std::sync::{Arc, Mutex};

use easyrag_core::llm::LlmService;
use easyrag_core::storage::{GraphStorage, KVStorage, VectorStorage};
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

use super::batch_processor::BatchProcessor;
use super::callback::{CallbackManager, MetricsCallback, ProcessingCallback};
use super::config::RagAnythingConfig;
use super::context::{ContextConfig, ContextExtractor};
use super::document_processor::DocumentProcessor;
use super::model::{BatchProcessingResult, ModalType};
use super::multimodal_query_engine::{MultimodalQueryEngine, TextQueryBackend};
use super::parser::ParseCache;
use super::processor::{
    BaseModalProcessor, EquationModalProcessor, GenericModalProcessor, ImageModalProcessor,
    ModalProcessorRegistry, TableModalProcessor,
};

struct Components {
    document_processor: Arc<DocumentProcessor>,
    query_engine: Arc<MultimodalQueryEngine>,
    batch_processor: Arc<BatchProcessor>,
}

/// RAGAnything main entry point (Facade).
pub struct RAGAnything {
    config: RagAnythingConfig,
    backend: Arc<dyn TextQueryBackend>,
    llm_service: Arc<dyn LlmService>,
    kv_storage: Arc<dyn KVStorage>,
    vector_storage: Arc<dyn VectorStorage>,
    graph_storage: Arc<dyn GraphStorage>,
    vision_llm_service: Option<Arc<dyn LlmService>>,
    /// Business line bound at construction; propagated to all internal
    /// components as the biz filter value.
    biz_id: String,

    pub callback_manager: Arc<CallbackManager>,
    metrics_callback: Arc<MetricsCallback>,
    components: Mutex<Option<Components>>,
}

impl RAGAnything {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: RagAnythingConfig,
        backend: Arc<dyn TextQueryBackend>,
        llm_service: Arc<dyn LlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        graph_storage: Arc<dyn GraphStorage>,
        vision_llm_service: Option<Arc<dyn LlmService>>,
        biz_id: String,
    ) -> Self {
        Self {
            config,
            backend,
            llm_service,
            kv_storage,
            vector_storage,
            graph_storage,
            vision_llm_service,
            biz_id,
            callback_manager: Arc::new(CallbackManager::new()),
            metrics_callback: Arc::new(MetricsCallback::new()),
            components: Mutex::new(None),
        }
    }

    /// Initialize all components and register modal processors.
    pub fn initialize(&self) {
        let mut guard = self.components.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_some() {
            return;
        }

        tracing::info!(
            "Initializing RAGAnything: parser={}, workspace={}",
            self.config.parser,
            self.kv_storage.workspace()
        );

        // Register metrics callback
        self.callback_manager.register(self.metrics_callback.clone());

        // Create parse cache
        let parse_cache = Arc::new(ParseCache::new(
            Some(self.kv_storage.clone()),
            self.biz_id.clone(),
        ));

        // Create context extractor
        let context_extractor = Arc::new(ContextExtractor::new(ContextConfig {
            context_window: self.config.context_window as i64,
            context_mode: self.config.context_mode.clone(),
            max_context_tokens: self.config.max_context_tokens,
            include_headers: self.config.include_headers,
            include_captions: self.config.include_captions,
            ..Default::default()
        }));

        // Register modal processors
        let effective_vision_service = self
            .vision_llm_service
            .clone()
            .unwrap_or_else(|| self.llm_service.clone());
        ModalProcessorRegistry::register(
            ModalType::Image,
            Arc::new(ImageModalProcessor::new(BaseModalProcessor::new(
                effective_vision_service.clone(),
                self.kv_storage.clone(),
                self.vector_storage.clone(),
                self.graph_storage.clone(),
                context_extractor.clone(),
                self.biz_id.clone(),
            ))),
        );
        ModalProcessorRegistry::register(
            ModalType::Table,
            Arc::new(TableModalProcessor::new(BaseModalProcessor::new(
                self.llm_service.clone(),
                self.kv_storage.clone(),
                self.vector_storage.clone(),
                self.graph_storage.clone(),
                context_extractor.clone(),
                self.biz_id.clone(),
            ))),
        );
        ModalProcessorRegistry::register(
            ModalType::Equation,
            Arc::new(EquationModalProcessor::new(BaseModalProcessor::new(
                self.llm_service.clone(),
                self.kv_storage.clone(),
                self.vector_storage.clone(),
                self.graph_storage.clone(),
                context_extractor.clone(),
                self.biz_id.clone(),
            ))),
        );
        ModalProcessorRegistry::register(
            ModalType::Generic,
            Arc::new(GenericModalProcessor::new(BaseModalProcessor::new(
                self.llm_service.clone(),
                self.kv_storage.clone(),
                self.vector_storage.clone(),
                self.graph_storage.clone(),
                context_extractor.clone(),
                self.biz_id.clone(),
            ))),
        );

        // Create document processor
        let document_processor = Arc::new(DocumentProcessor::new(
            self.config.clone(),
            self.llm_service.clone(),
            self.kv_storage.clone(),
            self.vector_storage.clone(),
            self.graph_storage.clone(),
            Some(self.callback_manager.clone()),
            Some(parse_cache),
            self.biz_id.clone(),
        ));

        // Create query engine
        let query_engine = Arc::new(MultimodalQueryEngine::new(
            self.backend.clone(),
            effective_vision_service,
            self.config.clone(),
            Some(self.kv_storage.clone()),
            self.biz_id.clone(),
        ));

        // Create batch processor
        let batch_processor = Arc::new(BatchProcessor::new(
            self.config.clone(),
            document_processor.clone(),
        ));

        *guard = Some(Components {
            document_processor,
            query_engine,
            batch_processor,
        });
        tracing::info!("RAGAnything initialized successfully");
    }

    /// Process a single file. Returns processing result metadata.
    pub async fn process_file(
        &self,
        file_path: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<JsonMap> {
        let processor = self.document_processor()?;
        processor.process_document_complete(file_path, kwargs).await
    }

    /// Process all supported files in a folder.
    pub async fn process_folder(
        &self,
        folder_path: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<BatchProcessingResult> {
        let batch_processor = self.batch_processor()?;
        let output_dir = kwargs
            .get("output_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let parse_method = kwargs
            .get("parse_method")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let recursive = kwargs
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(self.config.recursive_folder_processing);
        let max_workers = kwargs
            .get("max_workers")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(self.config.max_concurrent_files);

        batch_processor
            .process_folder(
                folder_path,
                output_dir.as_deref(),
                parse_method.as_deref(),
                None,
                recursive,
                max_workers,
            )
            .await
    }

    /// Pure text query. `mode`: "local", "global", "hybrid", "naive", "mix".
    pub async fn query(
        &self,
        query: &str,
        mode: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<String> {
        let engine = self.query_engine()?;
        engine.query(query, mode, kwargs).await
    }

    /// Multimodal query with content.
    pub async fn query_with_multimodal(
        &self,
        query: &str,
        multimodal_content: Option<&[JsonMap]>,
        mode: &str,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<String> {
        let engine = self.query_engine()?;
        engine
            .query_with_multimodal(query, multimodal_content, mode, kwargs)
            .await
    }

    /// Finalize and release resources.
    pub fn finalize(&self) {
        let mut guard = self.components.lock().unwrap_or_else(|e| e.into_inner());
        if guard.is_none() {
            return;
        }
        tracing::info!("Finalizing RAGAnything...");
        let metrics: Arc<dyn ProcessingCallback> = self.metrics_callback.clone();
        self.callback_manager.unregister(&metrics);
        *guard = None;
        tracing::info!("RAGAnything finalized");
    }

    /// Get processor information and status.
    pub fn get_processor_info(&self) -> JsonMap {
        let initialized = self
            .components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some();
        let mut registered: Vec<String> = ModalProcessorRegistry::get_registered_types()
            .into_iter()
            .map(|t| t.value().to_uppercase())
            .collect();
        registered.sort();

        let m = &self.metrics_callback;
        let mut metrics = JsonMap::new();
        metrics.insert(
            "total_processed".to_string(),
            JsonValue::from(m.documents_parsed.load(std::sync::atomic::Ordering::Relaxed) as i64),
        );
        metrics.insert(
            "total_errors".to_string(),
            JsonValue::from(m.documents_failed.load(std::sync::atomic::Ordering::Relaxed) as i64),
        );
        metrics.insert(
            "total_processing_time_ms".to_string(),
            JsonValue::from(
                m.total_parse_time_ms.load(std::sync::atomic::Ordering::Relaxed)
                    + m.total_insert_time_ms.load(std::sync::atomic::Ordering::Relaxed)
                    + m.total_modal_time_ms.load(std::sync::atomic::Ordering::Relaxed),
            ),
        );

        let mut info = JsonMap::new();
        info.insert("initialized".to_string(), JsonValue::from(initialized));
        info.insert("parser".to_string(), JsonValue::from(self.config.parser.clone()));
        info.insert(
            "parse_method".to_string(),
            JsonValue::from(self.config.parse_method.clone()),
        );
        info.insert(
            "image_processing_enabled".to_string(),
            JsonValue::from(self.config.enable_image_processing),
        );
        info.insert(
            "table_processing_enabled".to_string(),
            JsonValue::from(self.config.enable_table_processing),
        );
        info.insert(
            "equation_processing_enabled".to_string(),
            JsonValue::from(self.config.enable_equation_processing),
        );
        info.insert(
            "context_mode".to_string(),
            JsonValue::from(self.config.context_mode.clone()),
        );
        info.insert(
            "context_window".to_string(),
            JsonValue::from(self.config.context_window as u64),
        );
        info.insert(
            "max_concurrent_files".to_string(),
            JsonValue::from(self.config.max_concurrent_files as u64),
        );
        info.insert(
            "registered_processors".to_string(),
            JsonValue::Array(registered.into_iter().map(JsonValue::from).collect()),
        );
        info.insert("metrics".to_string(), JsonValue::Object(metrics));
        info
    }

    /// Register a custom processing callback.
    pub fn register_callback(&self, callback: Arc<dyn ProcessingCallback>) {
        self.callback_manager.register(callback);
    }

    /// Unregister a processing callback.
    pub fn unregister_callback(&self, callback: &Arc<dyn ProcessingCallback>) {
        self.callback_manager.unregister(callback);
    }

    // --- Private helpers ---

    fn document_processor(&self) -> easyrag_core::Result<Arc<DocumentProcessor>> {
        self.components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|c| c.document_processor.clone())
            .ok_or_else(|| {
                easyrag_core::CoreError::Internal(
                    "RAGAnything not initialized. Call initialize() first.".to_string(),
                )
            })
    }

    fn query_engine(&self) -> easyrag_core::Result<Arc<MultimodalQueryEngine>> {
        self.components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|c| c.query_engine.clone())
            .ok_or_else(|| {
                easyrag_core::CoreError::Internal(
                    "RAGAnything not initialized. Call initialize() first.".to_string(),
                )
            })
    }

    fn batch_processor(&self) -> easyrag_core::Result<Arc<BatchProcessor>> {
        self.components
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|c| c.batch_processor.clone())
            .ok_or_else(|| {
                easyrag_core::CoreError::Internal(
                    "RAGAnything not initialized. Call initialize() first.".to_string(),
                )
            })
    }
}
