//! Streaming pipeline (`PipelineService`) integration tests.
//!
//! These exercise the discovery→channel→processor model added behind
//! `PIPELINE_STREAMING_ENABLED`: happy-path processing, the concurrency ceiling,
//! the claim/cancel invariant (an un-dequeued document stays `Pending`), and the
//! `busy`/`active_count` redefinition. The mock LLMs mirror `ingest_pipeline.rs`
//! (integration test files are separate crates, so the helpers are re-declared).

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::storage::traits::DocStatusStorage;
use easyrag_pipeline::extraction::{EntityExtractor, EntityExtractorConfig};
use easyrag_pipeline::{IngestionPipeline, PipelineService};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Default biz line used for the biz_id filter argument.
const BIZ: &str = "default";

const EXTRACTION_JSON: &str = r#"{"entities":[
    {"name":"Alice","type":"Person","description":"A researcher"}
],"relationships":[]}"#;

// ---------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------

struct ZeroFunc;

#[async_trait]
impl EmbeddingFunction for ZeroFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0f32; 4]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ZeroFunc))
}

/// LLM returning a fixed extraction payload (no gating, no counting).
struct MockLlm {
    default: String,
}

impl MockLlm {
    fn new(default: impl Into<String>) -> Self {
        Self {
            default: default.into(),
        }
    }
}

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok(self.default.clone())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

/// LLM whose completions block until a watch gate opens, letting tests observe
/// the in-flight steady state and cancel mid-run.
struct GateLlm {
    default: String,
    rx: tokio::sync::watch::Receiver<bool>,
    calls: AtomicUsize,
}

impl GateLlm {
    fn new(default: impl Into<String>, rx: tokio::sync::watch::Receiver<bool>) -> Self {
        Self {
            default: default.into(),
            rx,
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl LlmService for GateLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut rx = self.rx.clone();
        let _ = rx.wait_for(|open| *open).await;
        Ok(self.default.clone())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

struct TestEnv {
    pipeline: Arc<IngestionPipeline>,
    doc_status: Arc<InMemoryDocStatusStorage>,
}

/// Build a pipeline + a streaming service sharing the same in-memory storage.
/// A short poll interval keeps tests fast; `ensure_running` also nudges
/// discovery immediately so the first pass does not wait for the interval.
async fn setup_with_concurrency<L: LlmService + 'static>(
    llm: Arc<L>,
    max_concurrency: usize,
) -> (TestEnv, Arc<PipelineService>) {
    let kv = Arc::new(InMemoryKVStorage::new("kv", "test"));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "test"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "test"));
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vector",
        "test",
        test_embedding_func(),
    ));
    let pipeline = Arc::new(
        IngestionPipeline::new(
            llm.clone(),
            kv.clone(),
            vector,
            graph.clone(),
            doc_status.clone(),
            Arc::new(test_embedding_func()),
        )
        .with_max_concurrency(max_concurrency)
        .with_extractor(EntityExtractor::new(
            llm.clone(),
            graph.clone(),
            Arc::new(InMemoryVectorStorage::new(
                "vector2",
                "test",
                test_embedding_func(),
            )),
            kv.clone(),
            EntityExtractorConfig::default(),
        )),
    );
    let service = Arc::new(PipelineService::new(
        pipeline.clone(),
        doc_status.clone(),
        kv.clone(),
        BIZ,
        20,      // poll_interval_ms
        Some(64), // channel_capacity
    ));
    let env = TestEnv {
        pipeline,
        doc_status,
    };
    (env, service)
}

async fn setup<L: LlmService + 'static>(llm: Arc<L>) -> (TestEnv, Arc<PipelineService>) {
    setup_with_concurrency(llm, 4).await
}

/// Poll `cond` until it holds or the timeout elapses; returns success.
async fn wait_until(mut cond: impl FnMut() -> bool, timeout_ms: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if cond() {
            return true;
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn count_status(
    doc_status: &InMemoryDocStatusStorage,
    status: DocStatus,
) -> std::collections::HashMap<String, easyrag_core::model::doc_status::DocProcessingStatus> {
    doc_status.get_docs_by_status(status, BIZ).await.unwrap()
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

/// Happy path: discovery finds pending docs, the processor claims and processes
/// them, and `total_processed` reflects exactly one pass per document.
#[tokio::test]
async fn streaming_processes_pending_documents() {
    let (env, svc) = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let a = env
        .pipeline
        .insert_document("Alice works at The Lab streaming happy path", "s1.txt", None, None, BIZ)
        .await
        .unwrap();
    let b = env
        .pipeline
        .insert_document("Bob joins the streaming happy path lab too", "s2.txt", None, None, BIZ)
        .await
        .unwrap();

    svc.ensure_running();
    assert!(
        wait_until(|| svc.total_processed() >= 2, 5000).await,
        "both documents should be processed"
    );

    let processed = count_status(&env.doc_status, DocStatus::Processed).await;
    assert!(processed.contains_key(&a));
    assert!(processed.contains_key(&b));

    // Leave the service running for several poll intervals: processed docs are
    // no longer Pending, so discovery must not re-dispatch them (no double work).
    assert_eq!(svc.total_processed(), 2);
    tokio::time::sleep(Duration::from_millis(120)).await;
    assert_eq!(svc.total_processed(), 2, "documents must not be reprocessed");

    svc.stop().await;
    assert!(!svc.is_active());
    assert_eq!(svc.active_count(), 0);
}

/// Concurrency ceiling: with a gated LLM and `max_concurrency = 2`, at most two
/// documents are in flight at once; the rest wait in the channel.
#[tokio::test]
async fn streaming_respects_concurrency_ceiling() {
    let (tx, rx) = tokio::sync::watch::channel(false);
    let llm = Arc::new(GateLlm::new(EXTRACTION_JSON, rx));
    let (env, svc) = setup_with_concurrency(llm.clone(), 2).await;
    for i in 0..5 {
        env.pipeline
            .insert_document(
                &format!("streaming concurrency ceiling document number {i} content"),
                &format!("c{i}.txt"),
                None,
                None,
                BIZ,
            )
            .await
            .unwrap();
    }

    svc.ensure_running();
    // Wait until two documents are inside the gated extraction call.
    assert!(
        wait_until(|| llm.call_count() >= 2, 5000).await,
        "two documents should reach extraction"
    );
    // The semaphore ceiling holds: never more than max_concurrency in flight,
    // and nothing has finished while the gate is closed.
    assert!(svc.active_count() <= 2, "active_count must not exceed permits");
    assert!(svc.active_count() >= 1);
    assert_eq!(svc.total_processed(), 0);
    assert!(
        env.pipeline.pipeline_status().get("busy").and_then(|v| v.as_bool()) == Some(true),
        "busy must be true while documents are in flight"
    );

    // Release the gate: all five drain through two permits.
    let _ = tx.send(true);
    assert!(
        wait_until(|| svc.total_processed() >= 5, 5000).await,
        "all five documents should eventually process"
    );

    svc.stop().await;
    let processed = count_status(&env.doc_status, DocStatus::Processed).await;
    assert_eq!(processed.len(), 5);
    assert_eq!(svc.active_count(), 0);
}

/// Cancel invariant: a document that was dispatched but never claimed (blocked
/// behind the single permit) stays `Pending`, while the in-flight one finishes.
/// Mirrors `cancel_during_run_skips_remaining_documents` for the streaming path.
#[tokio::test]
async fn streaming_cancel_leaves_unclaimed_document_pending() {
    let (tx, rx) = tokio::sync::watch::channel(false);
    let llm = Arc::new(GateLlm::new(EXTRACTION_JSON, rx));
    let (env, svc) = setup_with_concurrency(llm.clone(), 1).await;
    let a = env
        .pipeline
        .insert_document("Alice streaming cancel run test one content", "k1.txt", None, None, BIZ)
        .await
        .unwrap();
    let b = env
        .pipeline
        .insert_document("Bob streaming cancel run test two content here", "k2.txt", None, None, BIZ)
        .await
        .unwrap();

    svc.ensure_running();
    // Wait until the first document is inside the gated extraction call; with a
    // single permit the second is dispatched but stuck before it can be claimed.
    assert!(
        wait_until(|| llm.call_count() >= 1, 5000).await,
        "first document should reach extraction"
    );

    svc.cancel();
    let _ = tx.send(true); // let the in-flight document finish
    svc.stop().await; // awaits the processor drain + in-flight JoinSet

    let processed = count_status(&env.doc_status, DocStatus::Processed).await;
    let pending = count_status(&env.doc_status, DocStatus::Pending).await;
    assert_eq!(processed.len(), 1, "exactly the in-flight document completes");
    assert_eq!(pending.len(), 1, "the un-claimed document stays Pending");
    for doc_id in [&a, &b] {
        assert!(processed.contains_key(doc_id) ^ pending.contains_key(doc_id));
    }
    assert_eq!(svc.active_count(), 0);
}

/// `busy` tracks `active_count`: true while a document is in flight, false once
/// the service drains — with `pipeline_active` (the batch flag) never set here.
#[tokio::test]
async fn streaming_busy_reflects_active_count() {
    let (tx, rx) = tokio::sync::watch::channel(false);
    let llm = Arc::new(GateLlm::new(EXTRACTION_JSON, rx));
    let (env, svc) = setup_with_concurrency(llm.clone(), 1).await;
    env.pipeline
        .insert_document("Alice streaming busy indicator test content", "b1.txt", None, None, BIZ)
        .await
        .unwrap();

    // Before the service runs, nothing is in flight.
    assert_eq!(
        env.pipeline.pipeline_status().get("busy").and_then(|v| v.as_bool()),
        Some(false)
    );

    svc.ensure_running();
    assert!(wait_until(|| llm.call_count() >= 1, 5000).await);
    let status = env.pipeline.pipeline_status();
    assert_eq!(status.get("busy").and_then(|v| v.as_bool()), Some(true));
    assert!(status.get("active_count").and_then(|v| v.as_u64()) >= Some(1));

    let _ = tx.send(true);
    assert!(wait_until(|| svc.total_processed() >= 1, 5000).await);
    svc.stop().await;

    let status = env.pipeline.pipeline_status();
    assert_eq!(status.get("busy").and_then(|v| v.as_bool()), Some(false));
    assert_eq!(status.get("active_count").and_then(|v| v.as_u64()), Some(0));
}

/// Restartability: after `cancel`, a later `ensure_running` starts a fresh run
/// that picks up still-Pending documents (the singleton is re-armed by
/// `spawn_pipeline` on each new insert).
#[tokio::test]
async fn streaming_restarts_after_cancel() {
    let (env, svc) = setup(Arc::new(MockLlm::new(EXTRACTION_JSON))).await;
    let a = env
        .pipeline
        .insert_document("Alice streaming restart after cancel content", "r1.txt", None, None, BIZ)
        .await
        .unwrap();

    svc.ensure_running();
    assert!(wait_until(|| svc.total_processed() >= 1, 5000).await);
    assert!(count_status(&env.doc_status, DocStatus::Processed).await.contains_key(&a));

    // Cancel, then wait for both loops to wind down (service_active clears).
    svc.cancel();
    assert!(wait_until(|| !svc.is_active(), 5000).await);

    // Insert a new document and re-arm the service; it must be picked up by a
    // fresh discovery→processor run.
    let b = env
        .pipeline
        .insert_document("Bob streaming restart second document content", "r2.txt", None, None, BIZ)
        .await
        .unwrap();
    svc.ensure_running();
    let mut found = false;
    for _ in 0..200 {
        if count_status(&env.doc_status, DocStatus::Processed).await.contains_key(&b) {
            found = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    assert!(found, "restarted service should process the new document");

    svc.stop().await;
    assert!(!svc.is_active());
}
