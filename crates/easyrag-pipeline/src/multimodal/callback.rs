//! Processing callbacks and event system for RAGAnything
//! Provides a lightweight publish-subscribe mechanism for pipeline
//! observability.

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::panic::{catch_unwind, AssertUnwindSafe};

use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

/// Immutable record of a processing pipeline event.
#[derive(Debug, Clone)]
pub struct ProcessingEvent {
    pub event_type: String,
    pub timestamp: i64,
    pub file_path: Option<String>,
    pub doc_id: Option<String>,
    pub stage: Option<String>,
    pub details: JsonMap,
    pub duration_ms: Option<i64>,
    pub error: Option<String>,
}

impl ProcessingEvent {
    pub fn new(event_type: impl Into<String>) -> Self {
        Self {
            event_type: event_type.into(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            file_path: None,
            doc_id: None,
            stage: None,
            details: JsonMap::new(),
            duration_ms: None,
            error: None,
        }
    }
}

/// Base interface for processing pipeline callbacks.
/// Override any methods to hook into the corresponding stage.
/// All methods have default empty implementations.
pub trait ProcessingCallback: Send + Sync {
    // ── Parsing stage ─────────────────────────────────────────────
    fn on_parse_start(&self, _file_path: &str, _parser: &str) {}
    fn on_parse_complete(
        &self,
        _file_path: &str,
        _content_blocks: usize,
        _doc_id: &str,
        _duration_ms: i64,
    ) {
    }

    // ── Text insertion stage ──────────────────────────────────────
    fn on_text_insert_start(&self, _doc_id: &str, _content_length: usize) {}
    fn on_text_insert_complete(&self, _doc_id: &str, _chunks_created: usize, _duration_ms: i64) {}

    // ── Multimodal processing stage ───────────────────────────────
    fn on_modal_process_start(&self, _file_path: &str, _modal_type: &str, _item_count: usize) {}
    fn on_modal_process_complete(
        &self,
        _file_path: &str,
        _processed_count: usize,
        _failed_count: usize,
        _duration_ms: i64,
    ) {
    }

    // ── Query stage ───────────────────────────────────────────────
    fn on_query_start(&self, _query: &str, _mode: &str) {}
    fn on_query_complete(&self, _query: &str, _response_length: usize, _duration_ms: i64) {}

    // ── Error handling ────────────────────────────────────────────
    fn on_error(
        &self,
        _stage: &str,
        _error: &str,
        _file_path: Option<&str>,
        _doc_id: Option<&str>,
    ) {
    }

    // ── Batch processing ──────────────────────────────────────────
    fn on_batch_start(&self, _folder_path: &str, _total_files: usize) {}
    fn on_batch_complete(
        &self,
        _successful: usize,
        _failed: usize,
        _skipped: usize,
        _duration_ms: i64,
    ) {
    }
}

/// Thread-safe callback manager that dispatches events to registered
/// callbacks. The list is snapshotted under a mutex before dispatch
/// (Rust analogue of CopyOnWriteArrayList iteration).
#[derive(Default)]
pub struct CallbackManager {
    callbacks: Mutex<Vec<Arc<dyn ProcessingCallback>>>,
}

impl CallbackManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a callback to receive processing events.
    pub fn register(&self, callback: Arc<dyn ProcessingCallback>) {
        self.callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(callback);
    }

    /// Unregister a callback (by pointer identity).
    pub fn unregister(&self, callback: &Arc<dyn ProcessingCallback>) {
        let mut guard = self.callbacks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pos) = guard.iter().position(|c| Arc::ptr_eq(c, callback)) {
            guard.remove(pos);
        }
    }

    /// Get the number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Whether any callbacks are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // ── Dispatch methods ──────────────────────────────────────────

    pub fn emit_parse_start(&self, file_path: &str, parser: &str) {
        self.for_each_safe(|cb| cb.on_parse_start(file_path, parser));
    }

    pub fn emit_parse_complete(
        &self,
        file_path: &str,
        content_blocks: usize,
        doc_id: &str,
        duration_ms: i64,
    ) {
        self.for_each_safe(|cb| cb.on_parse_complete(file_path, content_blocks, doc_id, duration_ms));
    }

    pub fn emit_text_insert_start(&self, doc_id: &str, content_length: usize) {
        self.for_each_safe(|cb| cb.on_text_insert_start(doc_id, content_length));
    }

    pub fn emit_text_insert_complete(&self, doc_id: &str, chunks_created: usize, duration_ms: i64) {
        self.for_each_safe(|cb| cb.on_text_insert_complete(doc_id, chunks_created, duration_ms));
    }

    pub fn emit_modal_process_start(&self, file_path: &str, modal_type: &str, item_count: usize) {
        self.for_each_safe(|cb| cb.on_modal_process_start(file_path, modal_type, item_count));
    }

    pub fn emit_modal_process_complete(
        &self,
        file_path: &str,
        processed_count: usize,
        failed_count: usize,
        duration_ms: i64,
    ) {
        self.for_each_safe(|cb| {
            cb.on_modal_process_complete(file_path, processed_count, failed_count, duration_ms)
        });
    }

    pub fn emit_query_start(&self, query: &str, mode: &str) {
        self.for_each_safe(|cb| cb.on_query_start(query, mode));
    }

    pub fn emit_query_complete(&self, query: &str, response_length: usize, duration_ms: i64) {
        self.for_each_safe(|cb| cb.on_query_complete(query, response_length, duration_ms));
    }

    pub fn emit_error(
        &self,
        stage: &str,
        error: &str,
        file_path: Option<&str>,
        doc_id: Option<&str>,
    ) {
        self.for_each_safe(|cb| cb.on_error(stage, error, file_path, doc_id));
    }

    pub fn emit_batch_start(&self, folder_path: &str, total_files: usize) {
        self.for_each_safe(|cb| cb.on_batch_start(folder_path, total_files));
    }

    pub fn emit_batch_complete(
        &self,
        successful: usize,
        failed: usize,
        skipped: usize,
        duration_ms: i64,
    ) {
        self.for_each_safe(|cb| cb.on_batch_complete(successful, failed, skipped, duration_ms));
    }

    /// Snapshot the callback list and invoke each one, swallowing panics
    ///
    fn for_each_safe(&self, action: impl Fn(&dyn ProcessingCallback)) {
        let snapshot: Vec<Arc<dyn ProcessingCallback>> = self
            .callbacks
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        for callback in snapshot {
            if let Err(e) = catch_unwind(AssertUnwindSafe(|| action(callback.as_ref()))) {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_else(|| "unknown panic".to_string());
                tracing::warn!("Callback threw exception: {msg}");
            }
        }
    }
}

/// Metrics callback that collects processing statistics.
/// Useful for monitoring and observability.
#[derive(Default)]
pub struct MetricsCallback {
    pub documents_parsed: AtomicUsize,
    pub documents_failed: AtomicUsize,
    pub total_content_blocks: AtomicUsize,
    pub total_chunks_created: AtomicUsize,
    pub modal_items_processed: AtomicUsize,
    pub modal_items_failed: AtomicUsize,
    pub queries_executed: AtomicUsize,
    pub total_parse_time_ms: AtomicI64,
    pub total_insert_time_ms: AtomicI64,
    pub total_modal_time_ms: AtomicI64,
    pub total_query_time_ms: AtomicI64,
}

impl MetricsCallback {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a summary of collected metrics.
    pub fn summary(&self) -> JsonMap {
        let mut map = JsonMap::new();
        map.insert(
            "documents_parsed".to_string(),
            JsonValue::from(self.documents_parsed.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "documents_failed".to_string(),
            JsonValue::from(self.documents_failed.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "total_content_blocks".to_string(),
            JsonValue::from(self.total_content_blocks.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "total_chunks_created".to_string(),
            JsonValue::from(self.total_chunks_created.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "modal_items_processed".to_string(),
            JsonValue::from(self.modal_items_processed.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "modal_items_failed".to_string(),
            JsonValue::from(self.modal_items_failed.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "queries_executed".to_string(),
            JsonValue::from(self.queries_executed.load(Ordering::Relaxed) as i64),
        );
        map.insert(
            "total_parse_time_ms".to_string(),
            JsonValue::from(self.total_parse_time_ms.load(Ordering::Relaxed)),
        );
        map.insert(
            "total_insert_time_ms".to_string(),
            JsonValue::from(self.total_insert_time_ms.load(Ordering::Relaxed)),
        );
        map.insert(
            "total_modal_time_ms".to_string(),
            JsonValue::from(self.total_modal_time_ms.load(Ordering::Relaxed)),
        );
        map.insert(
            "total_query_time_ms".to_string(),
            JsonValue::from(self.total_query_time_ms.load(Ordering::Relaxed)),
        );
        map
    }

    /// Reset all metrics to zero.
    pub fn reset(&self) {
        self.documents_parsed.store(0, Ordering::Relaxed);
        self.documents_failed.store(0, Ordering::Relaxed);
        self.total_content_blocks.store(0, Ordering::Relaxed);
        self.total_chunks_created.store(0, Ordering::Relaxed);
        self.modal_items_processed.store(0, Ordering::Relaxed);
        self.modal_items_failed.store(0, Ordering::Relaxed);
        self.queries_executed.store(0, Ordering::Relaxed);
        self.total_parse_time_ms.store(0, Ordering::Relaxed);
        self.total_insert_time_ms.store(0, Ordering::Relaxed);
        self.total_modal_time_ms.store(0, Ordering::Relaxed);
        self.total_query_time_ms.store(0, Ordering::Relaxed);
    }
}

impl ProcessingCallback for MetricsCallback {
    fn on_parse_complete(
        &self,
        _file_path: &str,
        content_blocks: usize,
        _doc_id: &str,
        duration_ms: i64,
    ) {
        self.documents_parsed.fetch_add(1, Ordering::Relaxed);
        self.total_content_blocks.fetch_add(content_blocks, Ordering::Relaxed);
        self.total_parse_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn on_text_insert_complete(&self, _doc_id: &str, chunks_created: usize, duration_ms: i64) {
        self.total_chunks_created.fetch_add(chunks_created, Ordering::Relaxed);
        self.total_insert_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn on_modal_process_complete(
        &self,
        _file_path: &str,
        processed_count: usize,
        failed_count: usize,
        duration_ms: i64,
    ) {
        self.modal_items_processed.fetch_add(processed_count, Ordering::Relaxed);
        self.modal_items_failed.fetch_add(failed_count, Ordering::Relaxed);
        self.total_modal_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn on_query_complete(&self, _query: &str, _response_length: usize, duration_ms: i64) {
        self.queries_executed.fetch_add(1, Ordering::Relaxed);
        self.total_query_time_ms.fetch_add(duration_ms, Ordering::Relaxed);
    }

    fn on_error(&self, _stage: &str, _error: &str, _file_path: Option<&str>, _doc_id: Option<&str>) {
        self.documents_failed.fetch_add(1, Ordering::Relaxed);
    }
}
