//! Processing callbacks:
//! events, event types, the callback trait, a built-in metrics collector,
//! and a thread-safe callback manager.

use easyrag_core::JsonMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Processing event — immutable record of a pipeline event.
#[derive(Clone, Debug)]
pub struct ProcessingEvent {
    pub event_type: String,
    /// ISO-8601 timestamp.
    pub timestamp: String,
    pub file_path: Option<String>,
    pub doc_id: Option<String>,
    pub stage: Option<String>,
    pub details: JsonMap,
    pub duration_ms: Option<u64>,
    pub error: Option<String>,
}

impl ProcessingEvent {
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            // `Z` suffix matches.
            timestamp: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
            file_path: None,
            doc_id: None,
            stage: None,
            details: JsonMap::new(),
            duration_ms: None,
            error: None,
        }
    }

    pub fn with_file_path(mut self, file_path: impl Into<String>) -> Self {
        self.file_path = Some(file_path.into());
        self
    }

    pub fn with_doc_id(mut self, doc_id: impl Into<String>) -> Self {
        self.doc_id = Some(doc_id.into());
        self
    }

    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    pub fn with_details(mut self, details: JsonMap) -> Self {
        self.details = details;
        self
    }

    pub fn with_duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }
}

/// Event types for the processing pipeline.
pub mod event_types {
    pub const PARSE_START: &str = "parse_start";
    pub const PARSE_END: &str = "parse_end";
    pub const PARSE_ERROR: &str = "parse_error";
    pub const CHUNK_START: &str = "chunk_start";
    pub const CHUNK_END: &str = "chunk_end";
    pub const EXTRACTION_START: &str = "extraction_start";
    pub const EXTRACTION_END: &str = "extraction_end";
    pub const EMBEDDING_START: &str = "embedding_start";
    pub const EMBEDDING_END: &str = "embedding_end";
    pub const GRAPH_MERGE_START: &str = "graph_merge_start";
    pub const GRAPH_MERGE_END: &str = "graph_merge_end";
    pub const DOCUMENT_COMPLETE: &str = "document_complete";
    pub const DOCUMENT_FAILED: &str = "document_failed";
    pub const BATCH_START: &str = "batch_start";
    pub const BATCH_END: &str = "batch_end";
    pub const QUERY_START: &str = "query_start";
    pub const QUERY_END: &str = "query_end";
    pub const MULTIMODAL_START: &str = "multimodal_start";
    pub const MULTIMODAL_END: &str = "multimodal_end";
}

/// Processing callback interface. All methods have no-op defaults,
/// mirroring the interface default bodies.
pub trait ProcessingCallback: Send + Sync {
    fn on_event(&self, _event: &ProcessingEvent) {}
    fn on_parse_start(&self, _file_path: &str, _doc_id: &str) {}
    fn on_parse_end(&self, _file_path: &str, _doc_id: &str, _duration_ms: u64) {}
    fn on_parse_error(&self, _file_path: &str, _doc_id: &str, _error: &str) {}
    fn on_chunk_end(&self, _doc_id: &str, _chunks_count: usize) {}
    fn on_extraction_end(&self, _doc_id: &str, _entities_count: usize, _relations_count: usize) {}
    fn on_document_complete(&self, _doc_id: &str, _file_path: &str, _duration_ms: u64) {}
    fn on_document_failed(&self, _doc_id: &str, _file_path: &str, _error: &str) {}
    fn on_batch_start(&self, _total_documents: usize) {}
    fn on_batch_end(&self, _processed: usize, _failed: usize, _duration_ms: u64) {}
    fn on_query_start(&self, _query: &str, _mode: &str) {}
    fn on_query_end(&self, _query: &str, _mode: &str, _duration_ms: u64) {}
}

/// Metrics callback — built-in metrics collector (atomic counters).
#[derive(Default)]
pub struct MetricsCallback {
    documents_processed: AtomicUsize,
    documents_failed: AtomicUsize,
    total_chunks: AtomicUsize,
    total_entities: AtomicUsize,
    total_relations: AtomicUsize,
    total_processing_time_ms: AtomicU64,
    queries_executed: AtomicUsize,
    total_query_time_ms: AtomicU64,
}

impl MetricsCallback {
    pub fn documents_processed(&self) -> usize {
        self.documents_processed.load(Ordering::SeqCst)
    }

    pub fn documents_failed(&self) -> usize {
        self.documents_failed.load(Ordering::SeqCst)
    }

    pub fn total_chunks(&self) -> usize {
        self.total_chunks.load(Ordering::SeqCst)
    }

    pub fn total_entities(&self) -> usize {
        self.total_entities.load(Ordering::SeqCst)
    }

    pub fn total_relations(&self) -> usize {
        self.total_relations.load(Ordering::SeqCst)
    }

    pub fn queries_executed(&self) -> usize {
        self.queries_executed.load(Ordering::SeqCst)
    }

    pub fn average_processing_time_ms(&self) -> u64 {
        let docs = self.documents_processed();
        if docs > 0 {
            self.total_processing_time_ms.load(Ordering::SeqCst) / docs as u64
        } else {
            0
        }
    }

    pub fn average_query_time_ms(&self) -> u64 {
        let queries = self.queries_executed();
        if queries > 0 {
            self.total_query_time_ms.load(Ordering::SeqCst) / queries as u64
        } else {
            0
        }
    }

    pub fn success_rate(&self) -> f32 {
        let total = self.documents_processed() + self.documents_failed();
        if total > 0 {
            self.documents_processed() as f32 / total as f32
        } else {
            0.0
        }
    }

    pub fn get_metrics(&self) -> JsonMap {
        let mut map = JsonMap::new();
        map.insert(
            "documents_processed".into(),
            self.documents_processed().into(),
        );
        map.insert("documents_failed".into(), self.documents_failed().into());
        map.insert("success_rate".into(), serde_json::json!(self.success_rate()));
        map.insert("total_chunks".into(), self.total_chunks().into());
        map.insert("total_entities".into(), self.total_entities().into());
        map.insert("total_relations".into(), self.total_relations().into());
        map.insert(
            "avg_processing_time_ms".into(),
            self.average_processing_time_ms().into(),
        );
        map.insert("queries_executed".into(), self.queries_executed().into());
        map.insert(
            "avg_query_time_ms".into(),
            self.average_query_time_ms().into(),
        );
        map
    }

    pub fn reset(&self) {
        self.documents_processed.store(0, Ordering::SeqCst);
        self.documents_failed.store(0, Ordering::SeqCst);
        self.total_chunks.store(0, Ordering::SeqCst);
        self.total_entities.store(0, Ordering::SeqCst);
        self.total_relations.store(0, Ordering::SeqCst);
        self.total_processing_time_ms.store(0, Ordering::SeqCst);
        self.queries_executed.store(0, Ordering::SeqCst);
        self.total_query_time_ms.store(0, Ordering::SeqCst);
    }
}

impl ProcessingCallback for MetricsCallback {
    fn on_chunk_end(&self, _doc_id: &str, chunks_count: usize) {
        self.total_chunks.fetch_add(chunks_count, Ordering::SeqCst);
    }

    fn on_extraction_end(&self, _doc_id: &str, entities_count: usize, relations_count: usize) {
        self.total_entities.fetch_add(entities_count, Ordering::SeqCst);
        self.total_relations
            .fetch_add(relations_count, Ordering::SeqCst);
    }

    fn on_document_complete(&self, _doc_id: &str, _file_path: &str, duration_ms: u64) {
        self.documents_processed.fetch_add(1, Ordering::SeqCst);
        self.total_processing_time_ms
            .fetch_add(duration_ms, Ordering::SeqCst);
    }

    fn on_document_failed(&self, _doc_id: &str, _file_path: &str, _error: &str) {
        self.documents_failed.fetch_add(1, Ordering::SeqCst);
    }

    fn on_query_end(&self, _query: &str, _mode: &str, duration_ms: u64) {
        self.queries_executed.fetch_add(1, Ordering::SeqCst);
        self.total_query_time_ms
            .fetch_add(duration_ms, Ordering::SeqCst);
    }
}

/// Callback manager — thread-safe event dispatcher with a bounded event log.
#[derive(Default)]
pub struct CallbackManager {
    callbacks: Mutex<Vec<std::sync::Arc<dyn ProcessingCallback>>>,
    event_log: Mutex<std::collections::VecDeque<ProcessingEvent>>,
}

const MAX_EVENT_LOG_SIZE: usize = 1000;

impl CallbackManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_callback(&self, callback: std::sync::Arc<dyn ProcessingCallback>) {
        self.callbacks.lock().unwrap().push(callback);
    }

    pub fn unregister_callback(&self, callback: &std::sync::Arc<dyn ProcessingCallback>) {
        let mut callbacks = self.callbacks.lock().unwrap();
        callbacks.retain(|c| !std::sync::Arc::ptr_eq(c, callback));
    }

    /// Log the event and dispatch it to all registered callbacks.
    /// Callback panics are caught per-callback.
    pub fn dispatch(&self, event: ProcessingEvent) {
        {
            let mut log = self.event_log.lock().unwrap();
            log.push_back(event.clone());
            if log.len() > MAX_EVENT_LOG_SIZE {
                log.pop_front();
            }
        }

        let callbacks = self.callbacks.lock().unwrap().clone();
        for callback in callbacks {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                callback.on_event(&event);
                dispatch_typed(callback.as_ref(), &event);
            }));
            if let Err(err) = result {
                tracing::warn!(
                    "Callback error for event {}: {:?}",
                    event.event_type,
                    err.downcast_ref::<&str>().unwrap_or(&"?")
                );
            }
        }
    }

    pub fn get_recent_events(&self, limit: usize) -> Vec<ProcessingEvent> {
        let log = self.event_log.lock().unwrap();
        log.iter().rev().take(limit).cloned().rev().collect()
    }

    pub fn clear_event_log(&self) {
        self.event_log.lock().unwrap().clear();
    }
}

/// Dispatch to the type-specific callback method based on event type
///
fn dispatch_typed(callback: &dyn ProcessingCallback, event: &ProcessingEvent) {
    let file_path = event.file_path.as_deref().unwrap_or("");
    let doc_id = event.doc_id.as_deref().unwrap_or("");
    let duration = event.duration_ms.unwrap_or(0);
    let error = event.error.as_deref().unwrap_or("");
    let detail_u64 = |key: &str| -> u64 {
        event
            .details
            .get(key)
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
    };
    let detail_str = |key: &str| -> String {
        event
            .details
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };

    match event.event_type.as_str() {
        event_types::PARSE_START => callback.on_parse_start(file_path, doc_id),
        event_types::PARSE_END => callback.on_parse_end(file_path, doc_id, duration),
        event_types::PARSE_ERROR => callback.on_parse_error(file_path, doc_id, error),
        event_types::CHUNK_END => {
            callback.on_chunk_end(doc_id, detail_u64("chunks_count") as usize)
        }
        event_types::EXTRACTION_END => callback.on_extraction_end(
            doc_id,
            detail_u64("entities_count") as usize,
            detail_u64("relations_count") as usize,
        ),
        event_types::DOCUMENT_COMPLETE => {
            callback.on_document_complete(doc_id, file_path, duration)
        }
        event_types::DOCUMENT_FAILED => callback.on_document_failed(doc_id, file_path, error),
        event_types::BATCH_START => callback.on_batch_start(detail_u64("total") as usize),
        event_types::BATCH_END => callback.on_batch_end(
            detail_u64("processed") as usize,
            detail_u64("failed") as usize,
            duration,
        ),
        event_types::QUERY_START => {
            callback.on_query_start(&detail_str("query"), &detail_str("mode"))
        }
        event_types::QUERY_END => callback.on_query_end(
            &detail_str("query"),
            &detail_str("mode"),
            duration,
        ),
        _ => {}
    }
}
