//! Byte-level characterization guardrail for the two callback modules.
//!
//! Captured on the PRE-5h code (`src/callback.rs` + `src/multimodal/callback.rs`)
//! and required to stay byte-identical after they are folded into
//! `src/callback/{event,manager,metrics}.rs`.
//!
//! Why this file exists separately from `callback_summarizer.rs`:
//!
//! * `src/multimodal/callback.rs` is **production** code (constructed by
//!   `RAGAnything`, driven by 7 `emit_*` sites in `DocumentProcessor`) and had
//!   **zero** test coverage before this file.
//! * `src/callback.rs` is covered only through `callback_summarizer.rs`, which
//!   exercises 3 of the 11 `dispatch_typed` arms and 2 of the 9 `get_metrics`
//!   keys. The other 8 arms, the 8 declared-but-unhandled `event_types`
//!   constants, the `on_event` hook, `unregister_callback`'s `retain`
//!   semantics and the full metrics key order were unpinned.
//!
//! Deliberate non-goals:
//!
//! * **No timestamp assertions.** `ProcessingEvent.timestamp` is a wall clock,
//!   so snapshotting it would be flaky. It also has **zero readers** anywhere
//!   in the workspace (every `.timestamp` hit in the tree is a
//!   `chrono::Utc::now().timestamp*()` *producer*), which is what makes the
//!   plan-mandated `String -> i64` change unobservable.
//! * **No panic-message assertions.** The two modules log panics with
//!   different wording and different `downcast` strategies
//!   (`"Callback error for event {}: {:?}"` downcasting only `&str` vs
//!   `"Callback threw exception: {msg}"` downcasting `String` then `&str`).
//!   Neither `tracing-subscriber` nor any other log-capture crate is a
//!   dev-dependency of `easyrag-pipeline`, so these lines cannot be captured
//!   here. Both spellings are preserved verbatim by the refactor and the
//!   *behaviour* around them (a panicking callback does not stop its
//!   successors) is pinned by the two `*_isolates_panicking_callback` tests.
//!
//! Both import paths are used in this file on purpose: `easyrag_pipeline::
//! callback::*` (the text module, also re-exported at the crate root by
//! `lib.rs:27-29`) and `easyrag_pipeline::multimodal::callback::*` (re-exported
//! by `multimodal/mod.rs:26`). That both keep resolving *and* keep denoting
//! two different trait/manager/metrics types is part of the contract.

use easyrag_pipeline::callback::{
    event_types, CallbackManager, MetricsCallback, ProcessingCallback, ProcessingEvent,
};
use easyrag_pipeline::multimodal::callback::{
    CallbackManager as MmCallbackManager, MetricsCallback as MmMetricsCallback,
    ProcessingCallback as MmProcessingCallback, ProcessingEvent as MmProcessingEvent,
};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------
// Snapshot plumbing
// ---------------------------------------------------------------------

fn golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-pipeline");
    root.join("testdata").join("golden").join("callback")
}

fn snap(name: &str, text: &str) {
    insta::with_settings! {
        {
            snapshot_path => golden_dir(),
            prepend_module_to_snapshot => false,
            omit_expression => true,
        },
        {
            insta::assert_snapshot!(name, text);
        }
    }
}

/// One recorded line per callback invocation, in call order.
#[derive(Default)]
struct Recorder(Mutex<Vec<String>>);

impl Recorder {
    fn log(&self, line: impl Into<String>) {
        self.0.lock().unwrap().push(line.into());
    }

    fn lines(&self) -> String {
        let lines = self.0.lock().unwrap();
        if lines.is_empty() {
            "<no callback fired>\n".to_string()
        } else {
            lines.iter().map(|l| format!("{l}\n")).collect()
        }
    }
}

fn opt(v: Option<&str>) -> String {
    match v {
        Some(s) => format!("Some({s})"),
        None => "None".to_string(),
    }
}

// ---------------------------------------------------------------------
// Text side: ProcessingCallback recorder
// ---------------------------------------------------------------------

struct TextRecorder(Arc<Recorder>);

impl ProcessingCallback for TextRecorder {
    fn on_event(&self, event: &ProcessingEvent) {
        self.0.log(format!("on_event({})", event.event_type));
    }
    fn on_parse_start(&self, file_path: &str, doc_id: &str) {
        self.0
            .log(format!("on_parse_start(file_path={file_path}, doc_id={doc_id})"));
    }
    fn on_parse_end(&self, file_path: &str, doc_id: &str, duration_ms: u64) {
        self.0.log(format!(
            "on_parse_end(file_path={file_path}, doc_id={doc_id}, duration_ms={duration_ms})"
        ));
    }
    fn on_parse_error(&self, file_path: &str, doc_id: &str, error: &str) {
        self.0.log(format!(
            "on_parse_error(file_path={file_path}, doc_id={doc_id}, error={error})"
        ));
    }
    fn on_chunk_end(&self, doc_id: &str, chunks_count: usize) {
        self.0
            .log(format!("on_chunk_end(doc_id={doc_id}, chunks_count={chunks_count})"));
    }
    fn on_extraction_end(&self, doc_id: &str, entities_count: usize, relations_count: usize) {
        self.0.log(format!(
            "on_extraction_end(doc_id={doc_id}, entities_count={entities_count}, \
             relations_count={relations_count})"
        ));
    }
    fn on_document_complete(&self, doc_id: &str, file_path: &str, duration_ms: u64) {
        self.0.log(format!(
            "on_document_complete(doc_id={doc_id}, file_path={file_path}, \
             duration_ms={duration_ms})"
        ));
    }
    fn on_document_failed(&self, doc_id: &str, file_path: &str, error: &str) {
        self.0.log(format!(
            "on_document_failed(doc_id={doc_id}, file_path={file_path}, error={error})"
        ));
    }
    fn on_batch_start(&self, total_documents: usize) {
        self.0
            .log(format!("on_batch_start(total_documents={total_documents})"));
    }
    fn on_batch_end(&self, processed: usize, failed: usize, duration_ms: u64) {
        self.0.log(format!(
            "on_batch_end(processed={processed}, failed={failed}, duration_ms={duration_ms})"
        ));
    }
    fn on_query_start(&self, query: &str, mode: &str) {
        self.0
            .log(format!("on_query_start(query={query}, mode={mode})"));
    }
    fn on_query_end(&self, query: &str, mode: &str, duration_ms: u64) {
        self.0.log(format!(
            "on_query_end(query={query}, mode={mode}, duration_ms={duration_ms})"
        ));
    }
}

// ---------------------------------------------------------------------
// Multimodal side: ProcessingCallback recorder
// ---------------------------------------------------------------------

struct MmRecorder(Arc<Recorder>);

impl MmProcessingCallback for MmRecorder {
    fn on_parse_start(&self, file_path: &str, parser: &str) {
        self.0
            .log(format!("on_parse_start(file_path={file_path}, parser={parser})"));
    }
    fn on_parse_complete(
        &self,
        file_path: &str,
        content_blocks: usize,
        doc_id: &str,
        duration_ms: i64,
    ) {
        self.0.log(format!(
            "on_parse_complete(file_path={file_path}, content_blocks={content_blocks}, \
             doc_id={doc_id}, duration_ms={duration_ms})"
        ));
    }
    fn on_text_insert_start(&self, doc_id: &str, content_length: usize) {
        self.0.log(format!(
            "on_text_insert_start(doc_id={doc_id}, content_length={content_length})"
        ));
    }
    fn on_text_insert_complete(&self, doc_id: &str, chunks_created: usize, duration_ms: i64) {
        self.0.log(format!(
            "on_text_insert_complete(doc_id={doc_id}, chunks_created={chunks_created}, \
             duration_ms={duration_ms})"
        ));
    }
    fn on_modal_process_start(&self, file_path: &str, modal_type: &str, item_count: usize) {
        self.0.log(format!(
            "on_modal_process_start(file_path={file_path}, modal_type={modal_type}, \
             item_count={item_count})"
        ));
    }
    fn on_modal_process_complete(
        &self,
        file_path: &str,
        processed_count: usize,
        failed_count: usize,
        duration_ms: i64,
    ) {
        self.0.log(format!(
            "on_modal_process_complete(file_path={file_path}, \
             processed_count={processed_count}, failed_count={failed_count}, \
             duration_ms={duration_ms})"
        ));
    }
    fn on_query_start(&self, query: &str, mode: &str) {
        self.0
            .log(format!("on_query_start(query={query}, mode={mode})"));
    }
    fn on_query_complete(&self, query: &str, response_length: usize, duration_ms: i64) {
        self.0.log(format!(
            "on_query_complete(query={query}, response_length={response_length}, \
             duration_ms={duration_ms})"
        ));
    }
    fn on_error(&self, stage: &str, error: &str, file_path: Option<&str>, doc_id: Option<&str>) {
        self.0.log(format!(
            "on_error(stage={stage}, error={error}, file_path={}, doc_id={})",
            opt(file_path),
            opt(doc_id)
        ));
    }
    fn on_batch_start(&self, folder_path: &str, total_files: usize) {
        self.0.log(format!(
            "on_batch_start(folder_path={folder_path}, total_files={total_files})"
        ));
    }
    fn on_batch_complete(
        &self,
        successful: usize,
        failed: usize,
        skipped: usize,
        duration_ms: i64,
    ) {
        self.0.log(format!(
            "on_batch_complete(successful={successful}, failed={failed}, skipped={skipped}, \
             duration_ms={duration_ms})"
        ));
    }
}

/// Prefixes every line it records, so registration order is visible.
struct Tagged(Arc<Recorder>, &'static str);

impl MmProcessingCallback for Tagged {
    fn on_query_start(&self, query: &str, mode: &str) {
        self.0
            .log(format!("{}: on_query_start(query={query}, mode={mode})", self.1));
    }
}

/// Panics on the first multimodal hook that `for_each_safe` reaches.
struct MmPanicker;

impl MmProcessingCallback for MmPanicker {
    fn on_parse_start(&self, _file_path: &str, _parser: &str) {
        panic!("multimodal callback exploded");
    }
}

/// Panics from the text-side typed hook only, so `on_event` still records and
/// proves the two hooks are separate calls inside one `catch_unwind`.
struct TextTypedPanicker(Arc<Recorder>);

impl ProcessingCallback for TextTypedPanicker {
    fn on_event(&self, event: &ProcessingEvent) {
        self.0.log(format!("on_event({})", event.event_type));
    }
    fn on_document_complete(&self, _doc_id: &str, _file_path: &str, _duration_ms: u64) {
        panic!("text typed callback exploded");
    }
}

fn details(pairs: &[(&str, serde_json::Value)]) -> easyrag_core::JsonMap {
    let mut map = easyrag_core::JsonMap::new();
    for (k, v) in pairs {
        map.insert((*k).to_string(), v.clone());
    }
    map
}

// =====================================================================
// Text side
// =====================================================================

/// All 11 dispatched event kinds, each with sentinel arguments chosen so a
/// swapped or dropped argument shows up as a different line.
#[test]
fn text_dispatch_matrix_routes_every_kind_with_exact_arguments() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    use serde_json::json;

    manager.dispatch(
        ProcessingEvent::new(event_types::PARSE_START)
            .with_file_path("fp-1")
            .with_doc_id("doc-1"),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::PARSE_END)
            .with_file_path("fp-2")
            .with_doc_id("doc-2")
            .with_duration_ms(22),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::PARSE_ERROR)
            .with_file_path("fp-3")
            .with_doc_id("doc-3")
            .with_error("err-3"),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::CHUNK_END)
            .with_doc_id("doc-4")
            .with_details(details(&[("chunks_count", json!(41))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::EXTRACTION_END)
            .with_doc_id("doc-5")
            .with_details(details(&[
                ("entities_count", json!(51)),
                ("relations_count", json!(52)),
            ])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::DOCUMENT_COMPLETE)
            .with_doc_id("doc-6")
            .with_file_path("fp-6")
            .with_duration_ms(66),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::DOCUMENT_FAILED)
            .with_doc_id("doc-7")
            .with_file_path("fp-7")
            .with_error("err-7"),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::BATCH_START).with_details(details(&[("total", json!(81))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::BATCH_END)
            .with_duration_ms(93)
            .with_details(details(&[("processed", json!(91)), ("failed", json!(92))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::QUERY_START).with_details(details(&[
            ("query", json!("q-10")),
            ("mode", json!("hybrid")),
        ])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::QUERY_END)
            .with_duration_ms(111)
            .with_details(details(&[("query", json!("q-11")), ("mode", json!("local"))])),
    );

    snap("text_dispatch_matrix", &rec.lines());
}

/// `event_types` declares 19 constants but `dispatch_typed` only branches on
/// 11. The other 8 fall into `_ => {}`: they are logged, and no typed hook
/// fires. Pinning this is what stops the unhandled set from growing silently.
#[test]
fn text_declared_kinds_without_a_dispatch_arm_log_but_dispatch_nothing() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    let unhandled = [
        event_types::CHUNK_START,
        event_types::EXTRACTION_START,
        event_types::EMBEDDING_START,
        event_types::EMBEDDING_END,
        event_types::GRAPH_MERGE_START,
        event_types::GRAPH_MERGE_END,
        event_types::MULTIMODAL_START,
        event_types::MULTIMODAL_END,
    ];
    for name in unhandled {
        manager.dispatch(
            ProcessingEvent::new(name)
                .with_file_path("fp")
                .with_doc_id("doc")
                .with_duration_ms(1)
                .with_error("err")
                .with_details(details(&[
                    ("chunks_count", serde_json::json!(1)),
                    ("entities_count", serde_json::json!(1)),
                    ("relations_count", serde_json::json!(1)),
                    ("total", serde_json::json!(1)),
                    ("processed", serde_json::json!(1)),
                    ("failed", serde_json::json!(1)),
                    ("query", serde_json::json!("q")),
                    ("mode", serde_json::json!("m")),
                ])),
        );
    }

    let logged: Vec<String> = manager
        .get_recent_events(100)
        .iter()
        .map(|e| e.event_type.clone())
        .collect();
    snap(
        "text_declared_kinds_without_dispatch_arm",
        &format!(
            "typed_and_on_event_calls:\n{}logged_event_count={}\nlogged_event_types={logged:?}\n",
            rec.lines(),
            logged.len()
        ),
    );
}

/// A name that is not one of the 19 constants (the "typo" case the plan calls
/// out) behaves exactly like the declared-unhandled ones: logged, no dispatch.
#[test]
fn text_unrecognized_event_name_is_logged_but_dispatches_nothing() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    manager.dispatch(ProcessingEvent::new("documnet_complete").with_doc_id("d1"));
    manager.dispatch(ProcessingEvent::new("").with_doc_id("d2"));
    manager.dispatch(ProcessingEvent::new("DOCUMENT_COMPLETE").with_doc_id("d3"));

    let logged: Vec<String> = manager
        .get_recent_events(100)
        .iter()
        .map(|e| e.event_type.clone())
        .collect();
    snap(
        "text_unrecognized_event_name",
        &format!(
            "calls:\n{}logged_event_types={logged:?}\n",
            rec.lines()
        ),
    );
}

/// `dispatch` invokes `on_event` **and then** `dispatch_typed` inside a single
/// `catch_unwind`, so a panic in the typed hook also discards nothing that was
/// already recorded but does prevent later callbacks... except that it does
/// not: the loop continues with the next callback. Both halves are pinned.
#[test]
fn text_on_event_and_typed_hook_are_separate_calls_and_panics_are_isolated() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(TextTypedPanicker(Arc::clone(&rec))));
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    manager.dispatch(
        ProcessingEvent::new(event_types::DOCUMENT_COMPLETE)
            .with_doc_id("d1")
            .with_file_path("fp-1")
            .with_duration_ms(5),
    );

    snap(
        "text_on_event_then_typed_hook_with_panic_isolation",
        &format!(
            "calls:\n{}logged_event_count={}\n",
            rec.lines(),
            manager.get_recent_events(10).len()
        ),
    );
}

/// `unregister_callback` uses `retain(!ptr_eq)`, so registering the same `Arc`
/// twice and unregistering once removes **both** entries. The multimodal
/// manager uses `position` + `remove`, which removes only the first. That
/// asymmetry is real and is pinned on both sides.
#[test]
fn text_unregister_removes_every_pointer_equal_entry() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    let callback: Arc<dyn ProcessingCallback> = Arc::new(TextRecorder(Arc::clone(&rec)));
    manager.register_callback(Arc::clone(&callback));
    manager.register_callback(Arc::clone(&callback));

    manager.dispatch(ProcessingEvent::new(event_types::PARSE_START).with_doc_id("d1"));
    manager.unregister_callback(&callback);
    manager.dispatch(ProcessingEvent::new(event_types::PARSE_START).with_doc_id("d2"));

    // Unregistering a callback that is not registered is a no-op, not a panic.
    let stranger: Arc<dyn ProcessingCallback> = Arc::new(TextRecorder(Arc::new(Recorder::default())));
    manager.unregister_callback(&stranger);
    manager.dispatch(ProcessingEvent::new(event_types::PARSE_START).with_doc_id("d3"));

    snap("text_unregister_retain_semantics", &rec.lines());
}

/// `dispatch_typed` reads four optional projections with fixed fallbacks:
/// `file_path`/`doc_id`/`error` -> `""`, `duration_ms` -> `0`, and the two
/// detail readers coerce by type (`as_u64` / `as_str`) so a wrong-typed value
/// degrades to the same fallback instead of erroring.
#[test]
fn text_optional_field_and_detail_coercion_fallbacks() {
    let rec = Arc::new(Recorder::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    use serde_json::json;

    // Everything absent.
    manager.dispatch(ProcessingEvent::new(event_types::PARSE_END));
    manager.dispatch(ProcessingEvent::new(event_types::PARSE_ERROR));
    manager.dispatch(ProcessingEvent::new(event_types::CHUNK_END));
    manager.dispatch(ProcessingEvent::new(event_types::EXTRACTION_END));
    manager.dispatch(ProcessingEvent::new(event_types::BATCH_START));
    manager.dispatch(ProcessingEvent::new(event_types::BATCH_END));
    manager.dispatch(ProcessingEvent::new(event_types::QUERY_START));
    manager.dispatch(ProcessingEvent::new(event_types::QUERY_END));

    // Present but wrong-typed: strings where a u64 is read, numbers where a
    // str is read.
    manager.dispatch(
        ProcessingEvent::new(event_types::CHUNK_END)
            .with_details(details(&[("chunks_count", json!("not-a-number"))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::QUERY_END)
            .with_details(details(&[("query", json!(7)), ("mode", json!(8))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::PARSE_END)
            .with_details(details(&[("duration_ms", json!(9))])),
    );

    snap("text_fallback_coercion", &rec.lines());
}

/// All six builders set exactly their own field and nothing else. The
/// projection excludes `timestamp` (wall clock, see module docs).
#[test]
fn text_event_builders_set_exactly_their_own_field() {
    fn project(e: &ProcessingEvent) -> String {
        format!(
            "event_type={}\nfile_path={}\ndoc_id={}\nstage={}\ndetails={}\nduration_ms={:?}\nerror={}\n",
            e.event_type,
            opt(e.file_path.as_deref()),
            opt(e.doc_id.as_deref()),
            opt(e.stage.as_deref()),
            serde_json::to_string(&e.details).unwrap(),
            e.duration_ms,
            opt(e.error.as_deref()),
        )
    }

    let bare = ProcessingEvent::new(event_types::QUERY_START);
    let full = ProcessingEvent::new(event_types::QUERY_END)
        .with_file_path("fp")
        .with_doc_id("doc")
        .with_stage("retrieve")
        .with_details(details(&[("k", serde_json::json!("v"))]))
        .with_duration_ms(9)
        .with_error("err");

    snap(
        "text_event_builder_projection",
        &format!("--- bare ---\n{}--- full ---\n{}", project(&bare), project(&full)),
    );
}

/// `get_metrics` builds a 9-key map; `serde_json` is compiled with
/// `preserve_order`, so the serialized string pins the insertion order too.
#[test]
fn text_metrics_map_has_nine_keys_in_insertion_order() {
    let metrics = MetricsCallback::default();
    metrics.on_chunk_end("d1", 5);
    metrics.on_extraction_end("d1", 3, 2);
    metrics.on_document_complete("d1", "a.txt", 100);
    metrics.on_document_complete("d2", "b.txt", 300);
    metrics.on_document_failed("d3", "c.txt", "boom");
    metrics.on_query_end("q?", "hybrid", 80);
    metrics.on_query_end("q?", "local", 120);

    let map = metrics.get_metrics();
    snap(
        "text_metrics_map_key_order",
        &format!(
            "keys={:?}\njson={}\n",
            map.keys().collect::<Vec<_>>(),
            serde_json::to_string(&map).unwrap()
        ),
    );

    metrics.reset();
    snap(
        "text_metrics_map_after_reset",
        &serde_json::to_string(&metrics.get_metrics()).unwrap(),
    );
}

// =====================================================================
// Multimodal side
// =====================================================================

/// All 11 `emit_*` methods, each with sentinel arguments chosen so a swapped
/// or dropped argument shows up as a different line. `emit_error` is exercised
/// with both `Some` and `None` optionals.
#[test]
fn multimodal_emit_matrix_routes_every_hook_with_exact_arguments() {
    let rec = Arc::new(Recorder::default());
    let manager = MmCallbackManager::new();
    manager.register(Arc::new(MmRecorder(Arc::clone(&rec))));

    manager.emit_parse_start("fp-1", "parser-1");
    manager.emit_parse_complete("fp-2", 22, "doc-2", 222);
    manager.emit_text_insert_start("doc-3", 33);
    manager.emit_text_insert_complete("doc-4", 44, 444);
    manager.emit_modal_process_start("fp-5", "image", 55);
    manager.emit_modal_process_complete("fp-6", 66, 67, 666);
    manager.emit_query_start("q-7", "hybrid");
    manager.emit_query_complete("q-8", 88, 888);
    manager.emit_error("stage-9", "err-9", Some("fp-9"), Some("doc-9"));
    manager.emit_error("stage-9b", "err-9b", None, None);
    manager.emit_batch_start("folder-10", 100);
    manager.emit_batch_complete(111, 112, 113, 114);

    snap("multimodal_emit_matrix", &rec.lines());
}

/// Callbacks fire in registration order, and `len`/`is_empty` track the list.
#[test]
fn multimodal_registration_order_and_len() {
    let rec = Arc::new(Recorder::default());
    let manager = MmCallbackManager::new();
    assert!(manager.is_empty());
    assert_eq!(0, manager.len());

    for tag in ["first", "second", "third"] {
        manager.register(Arc::new(Tagged(Arc::clone(&rec), tag)));
    }
    assert_eq!(3, manager.len());
    assert!(!manager.is_empty());

    manager.emit_query_start("q", "naive");
    snap("multimodal_registration_order", &rec.lines());
}

/// `unregister` uses `position` + `remove`, so with the same `Arc` registered
/// twice it removes only the first entry. Contrast with
/// `text_unregister_removes_every_pointer_equal_entry`.
#[test]
fn multimodal_unregister_removes_only_the_first_pointer_equal_entry() {
    let rec = Arc::new(Recorder::default());
    let manager = MmCallbackManager::new();
    let callback: Arc<dyn MmProcessingCallback> = Arc::new(MmRecorder(Arc::clone(&rec)));
    manager.register(Arc::clone(&callback));
    manager.register(Arc::clone(&callback));
    assert_eq!(2, manager.len());

    manager.emit_query_start("q1", "naive");
    manager.unregister(&callback);
    let len_after_first_unregister = manager.len();
    manager.emit_query_start("q2", "naive");
    manager.unregister(&callback);
    let len_after_second_unregister = manager.len();
    manager.emit_query_start("q3", "naive");

    // A stranger is not registered: unregistering it must not remove anything.
    let stranger: Arc<dyn MmProcessingCallback> = Arc::new(MmRecorder(Arc::new(Recorder::default())));
    manager.register(Arc::clone(&callback));
    let len_before_stranger = manager.len();
    manager.unregister(&stranger);
    let len_after_stranger = manager.len();

    snap(
        "multimodal_unregister_position_semantics",
        &format!(
            "len_after_first_unregister={len_after_first_unregister}\n\
             len_after_second_unregister={len_after_second_unregister}\n\
             len_before_stranger={len_before_stranger}\n\
             len_after_stranger={len_after_stranger}\n\
             calls:\n{}",
            rec.lines()
        ),
    );
}

/// `for_each_safe` snapshots the list under the lock and swallows panics, so a
/// panicking callback neither aborts the emit nor corrupts the registry.
#[test]
fn multimodal_for_each_safe_isolates_panicking_callback() {
    let rec = Arc::new(Recorder::default());
    let manager = MmCallbackManager::new();
    manager.register(Arc::new(MmPanicker));
    manager.register(Arc::new(MmRecorder(Arc::clone(&rec))));

    manager.emit_parse_start("fp", "docling");
    manager.emit_query_start("q", "naive");

    snap(
        "multimodal_panic_isolation",
        &format!("len={}\ncalls:\n{}", manager.len(), rec.lines()),
    );
}

/// `summary` builds an 11-key map; `preserve_order` makes the serialized
/// string pin the insertion order as well as the values.
#[test]
fn multimodal_metrics_summary_has_eleven_keys_in_insertion_order() {
    let metrics = MmMetricsCallback::new();
    metrics.on_parse_complete("a.pdf", 7, "doc-1", 100);
    metrics.on_parse_complete("b.pdf", 3, "doc-2", 200);
    metrics.on_text_insert_complete("doc-1", 12, 50);
    metrics.on_modal_process_complete("a.pdf", 4, 1, 300);
    metrics.on_query_complete("q", 256, 40);
    metrics.on_error("parse", "boom", Some("c.pdf"), Some("doc-3"));

    let summary = metrics.summary();
    snap(
        "multimodal_metrics_summary_key_order",
        &format!(
            "keys={:?}\njson={}\n",
            summary.keys().collect::<Vec<_>>(),
            serde_json::to_string(&summary).unwrap()
        ),
    );

    metrics.reset();
    snap(
        "multimodal_metrics_summary_after_reset",
        &serde_json::to_string(&metrics.summary()).unwrap(),
    );
}

/// Only 5 of the 11 hooks accumulate. Calling the other 6 must leave every
/// counter untouched — otherwise a merged `MetricsCallback` would silently
/// start counting multimodal events twice.
#[test]
fn multimodal_metrics_ignores_the_six_unoverridden_hooks() {
    let metrics = MmMetricsCallback::new();
    metrics.on_parse_start("a.pdf", "docling");
    metrics.on_text_insert_start("doc-1", 999);
    metrics.on_modal_process_start("a.pdf", "image", 999);
    metrics.on_query_start("q", "naive");
    metrics.on_batch_start("folder", 999);
    metrics.on_batch_complete(999, 999, 999, 999);

    snap(
        "multimodal_metrics_unoverridden_hooks_are_free",
        &serde_json::to_string(&metrics.summary()).unwrap(),
    );
}

/// The text-side metrics collector must stay blind to the multimodal hooks and
/// vice versa: `MetricsCallback` only overrides 5 text hooks, so driving the
/// full text matrix through it accumulates exactly what the 5 overrides count.
#[test]
fn text_metrics_only_counts_the_five_overridden_hooks() {
    let rec = Arc::new(Recorder::default());
    let metrics = Arc::new(MetricsCallback::default());
    let manager = CallbackManager::new();
    manager.register_callback(Arc::clone(&metrics) as Arc<dyn ProcessingCallback>);
    manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));

    use serde_json::json;
    manager.dispatch(ProcessingEvent::new(event_types::PARSE_START).with_doc_id("d"));
    manager.dispatch(
        ProcessingEvent::new(event_types::CHUNK_END)
            .with_doc_id("d")
            .with_details(details(&[("chunks_count", json!(4))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::EXTRACTION_END)
            .with_doc_id("d")
            .with_details(details(&[("entities_count", json!(5)), ("relations_count", json!(6))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::DOCUMENT_COMPLETE)
            .with_doc_id("d")
            .with_duration_ms(70),
    );
    manager.dispatch(ProcessingEvent::new(event_types::DOCUMENT_FAILED).with_doc_id("d"));
    manager.dispatch(
        ProcessingEvent::new(event_types::QUERY_END)
            .with_duration_ms(80)
            .with_details(details(&[("query", json!("q")), ("mode", json!("m"))])),
    );
    manager.dispatch(
        ProcessingEvent::new(event_types::BATCH_END)
            .with_duration_ms(90)
            .with_details(details(&[("processed", json!(1)), ("failed", json!(2))])),
    );

    snap(
        "text_metrics_five_overridden_hooks",
        &format!(
            "json={}\ncalls:\n{}",
            serde_json::to_string(&metrics.get_metrics()).unwrap(),
            rec.lines()
        ),
    );
}

/// `multimodal::ProcessingEvent` is public and re-exported by
/// `multimodal/mod.rs:26`, but nothing in the workspace constructs it. Its
/// field defaults are pinned anyway, because unifying the two structs means
/// this constructor becomes the shared one. `timestamp` is excluded (wall
/// clock); only its presence as a positive millisecond value is asserted.
#[test]
fn multimodal_event_defaults_are_the_documented_empty_projection() {
    let e = MmProcessingEvent::new("multimodal_start");
    assert!(
        e.timestamp > 1_600_000_000_000,
        "timestamp is epoch milliseconds, got {}",
        e.timestamp
    );
    snap(
        "multimodal_event_defaults",
        &format!(
            "event_type={}\nfile_path={}\ndoc_id={}\nstage={}\ndetails={}\nduration_ms={:?}\nerror={}\n",
            e.event_type,
            opt(e.file_path.as_deref()),
            opt(e.doc_id.as_deref()),
            opt(e.stage.as_deref()),
            serde_json::to_string(&e.details).unwrap(),
            e.duration_ms,
            opt(e.error.as_deref()),
        ),
    );
}

/// The two managers are independent registries: registering into one must not
/// deliver to the other. This is what a single merged `callback/` module has
/// to keep true, since both types remain reachable under their old paths.
#[test]
fn the_two_callback_systems_do_not_cross_deliver() {
    let rec = Arc::new(Recorder::default());
    let text_manager = CallbackManager::new();
    let mm_manager = MmCallbackManager::new();
    text_manager.register_callback(Arc::new(TextRecorder(Arc::clone(&rec))));
    mm_manager.register(Arc::new(MmRecorder(Arc::clone(&rec))));

    text_manager.dispatch(ProcessingEvent::new(event_types::QUERY_START).with_details(details(&[
        ("query", serde_json::json!("q")),
        ("mode", serde_json::json!("naive")),
    ])));
    mm_manager.emit_query_start("q", "naive");

    snap(
        "callback_systems_are_independent",
        &format!(
            "text_events_logged={}\nmm_len={}\ncalls:\n{}",
            text_manager.get_recent_events(10).len(),
            mm_manager.len(),
            rec.lines()
        ),
    );
}
