//! Callback + summarizer behavior tests.
//! The project ships no tests for these classes, so these cover the
//! behavioral contract: metrics accumulation, dispatch matrix, bounded event
//! log, panic isolation, and summarizer truncation/failure handling.

use async_trait::async_trait;
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::document_structure::SectionNode;
use easyrag_pipeline::callback::{
    event_types, CallbackManager, MetricsCallback, ProcessingCallback, ProcessingEvent,
};
use easyrag_pipeline::summarizer::DocumentSummarizer;
use futures::stream::BoxStream;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------

struct ScriptedLlm {
    responses: Mutex<VecDeque<String>>,
    default: String,
    calls: AtomicUsize,
}

impl ScriptedLlm {
    fn new(default: impl Into<String>) -> Self {
        Self {
            responses: Mutex::new(VecDeque::new()),
            default: default.into(),
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmService for ScriptedLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| self.default.clone()))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

struct FailingLlm;

#[async_trait]
impl LlmService for FailingLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Err(easyrag_core::CoreError::Internal(
            "llm unavailable".to_string(),
        ))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

fn details(pairs: &[(&str, serde_json::Value)]) -> easyrag_core::JsonMap {
    let mut map = easyrag_core::JsonMap::new();
    for (k, v) in pairs {
        map.insert((*k).to_string(), v.clone());
    }
    map
}

// ---------------------------------------------------------------------
// MetricsCallback
// ---------------------------------------------------------------------

#[test]
fn metrics_callback_accumulates_and_computes_rates() {
    let metrics = MetricsCallback::default();
    metrics.on_chunk_end("doc1", 5);
    metrics.on_extraction_end("doc1", 3, 2);
    metrics.on_document_complete("doc1", "a.txt", 100);
    metrics.on_document_complete("doc2", "b.txt", 300);
    metrics.on_document_failed("doc3", "c.txt", "boom");
    metrics.on_query_end("q?", "hybrid", 80);

    assert_eq!(2, metrics.documents_processed());
    assert_eq!(1, metrics.documents_failed());
    assert_eq!(5, metrics.total_chunks());
    assert_eq!(3, metrics.total_entities());
    assert_eq!(2, metrics.total_relations());
    assert_eq!(1, metrics.queries_executed());
    assert_eq!(200, metrics.average_processing_time_ms());
    assert_eq!(80, metrics.average_query_time_ms());
    assert!((metrics.success_rate() - 2.0 / 3.0).abs() < 1e-6);

    let map = metrics.get_metrics();
    assert_eq!(
        Some(2),
        map.get("documents_processed").and_then(|v| v.as_u64())
    );
    assert!(map.contains_key("success_rate"));
}

#[test]
fn metrics_callback_zero_division_guard_and_reset() {
    let metrics = MetricsCallback::default();
    assert_eq!(0, metrics.average_processing_time_ms());
    assert_eq!(0, metrics.average_query_time_ms());
    assert_eq!(0.0, metrics.success_rate());

    metrics.on_document_complete("doc1", "a.txt", 50);
    metrics.reset();
    assert_eq!(0, metrics.documents_processed());
    assert_eq!(0, metrics.average_processing_time_ms());
}

// ---------------------------------------------------------------------
// CallbackManager
// ---------------------------------------------------------------------

#[test]
fn manager_dispatch_routes_typed_events_to_metrics() {
    let manager = CallbackManager::new();
    let metrics = Arc::new(MetricsCallback::default());
    manager.register_callback(metrics.clone());

    manager.dispatch(
        ProcessingEvent::new(event_types::CHUNK_END)
            .with_doc_id("doc1")
            .with_details(details(&[("chunks_count", json!(4))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::DOCUMENT_COMPLETE)
            .with_doc_id("doc1")
            .with_file_path("a.txt")
            .with_duration_ms(120),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::QUERY_END)
            .with_duration_ms(30)
            .with_details(details(&[
                ("query", json!("what?")),
                ("mode", json!("tree")),
            ])),
    );

    assert_eq!(4, metrics.total_chunks());
    assert_eq!(1, metrics.documents_processed());
    assert_eq!(1, metrics.queries_executed());
}

#[test]
fn manager_event_log_is_bounded_and_ordered() {
    let manager = CallbackManager::new();
    for i in 0..1005 {
        manager.dispatch(ProcessingEvent::new(format!("evt-{i}")));
    }
    let events = manager.get_recent_events(2000);
    assert_eq!(1000, events.len());
    // Oldest 5 evicted; remaining keep chronological order.
    assert_eq!("evt-5", events.first().unwrap().event_type);
    assert_eq!("evt-1004", events.last().unwrap().event_type);

    manager.clear_event_log();
    assert!(manager.get_recent_events(10).is_empty());
}

#[test]
fn manager_unregister_stops_delivery() {
    let manager = CallbackManager::new();
    let metrics = Arc::new(MetricsCallback::default());
    let callback: Arc<dyn ProcessingCallback> = metrics.clone();
    manager.register_callback(callback.clone());
    manager.dispatch(ProcessingEvent::new(event_types::DOCUMENT_COMPLETE).with_doc_id("d1"));
    assert_eq!(1, metrics.documents_processed());

    manager.unregister_callback(&callback);
    manager.dispatch(ProcessingEvent::new(event_types::DOCUMENT_COMPLETE).with_doc_id("d2"));
    assert_eq!(1, metrics.documents_processed());
}

/// Callback that panics on every event — must not break other callbacks.
struct PanickingCallback;

impl ProcessingCallback for PanickingCallback {
    fn on_event(&self, _event: &ProcessingEvent) {
        panic!("callback exploded");
    }
}

#[test]
fn manager_isolates_panicking_callback() {
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(PanickingCallback));
    let metrics = Arc::new(MetricsCallback::default());
    manager.register_callback(metrics.clone());

    manager.dispatch(ProcessingEvent::new(event_types::DOCUMENT_COMPLETE).with_doc_id("d1"));
    // Event is still logged and the healthy callback still fired.
    assert_eq!(1, metrics.documents_processed());
    assert_eq!(1, manager.get_recent_events(10).len());
}

// ---------------------------------------------------------------------
// DocumentSummarizer
// ---------------------------------------------------------------------

#[tokio::test]
async fn summarizer_doc_description_trimmed_and_blank_skipped() {
    let llm = Arc::new(ScriptedLlm::new("  A doc about cats.  "));
    let summarizer = DocumentSummarizer::new(llm.clone());

    let desc = summarizer.generate_doc_description("Some content").await;
    assert_eq!(Some("A doc about cats.".to_string()), desc);
    assert_eq!(1, llm.call_count());

    // Blank content short-circuits without an LLM call.
    assert!(summarizer.generate_doc_description("   ").await.is_none());
    assert_eq!(1, llm.call_count());
}

#[tokio::test]
async fn summarizer_doc_description_swallows_llm_failure() {
    let summarizer = DocumentSummarizer::new(Arc::new(FailingLlm));
    assert!(summarizer.generate_doc_description("Some content").await.is_none());
}

#[tokio::test]
async fn summarizer_sections_recursive_with_byte_offsets() {
    let content = "# Intro\nHello intro text.\n## Deep\nDeep body text.";
    let llm = Arc::new(ScriptedLlm::new("summary"));
    let summarizer = DocumentSummarizer::new(llm.clone());

    let child = SectionNode {
        title: "Deep".to_string(),
        level: 2,
        summary: None,
        content_start: content.find("Deep body").unwrap(),
        content_end: content.len(),
        children: Vec::new(),
    };
    let parent = SectionNode {
        title: "Intro".to_string(),
        level: 1,
        summary: None,
        content_start: content.find("Hello").unwrap(),
        content_end: content.find("## Deep").unwrap(),
        children: vec![child],
    };

    let summarized = summarizer
        .generate_section_summaries(vec![parent], content)
        .await;
    assert_eq!(1, summarized.len());
    assert_eq!(Some("summary".to_string()), summarized[0].summary);
    assert_eq!(1, summarized[0].children.len());
    assert_eq!(Some("summary".to_string()), summarized[0].children[0].summary);
    // Parent + child, one LLM call each.
    assert_eq!(2, llm.call_count());
}

#[tokio::test]
async fn summarizer_section_blank_and_out_of_range_yield_none() {
    let llm = Arc::new(ScriptedLlm::new("summary"));
    let summarizer = DocumentSummarizer::new(llm.clone());

    let blank = SectionNode {
        title: "Empty".to_string(),
        level: 1,
        summary: None,
        content_start: 0,
        content_end: 0,
        children: Vec::new(),
    };
    let out_of_range = SectionNode {
        title: "Bad".to_string(),
        level: 1,
        summary: None,
        content_start: 0,
        content_end: 9999,
        children: Vec::new(),
    };

    let summarized = summarizer
        .generate_section_summaries(vec![blank, out_of_range], "short content")
        .await;
    assert!(summarized.iter().all(|s| s.summary.is_none()));
    // No LLM call for blank or out-of-range sections.
    assert_eq!(0, llm.call_count());
}
