//! Streaming ingestion pipeline (BC-11 gradual rollout).
//!
//! Replaces the one-shot batch [`IngestionPipeline::start`] → `run_pipeline`
//! snapshot + `join_all` model with a persistent producer/consumer loop when
//! `PIPELINE_STREAMING_ENABLED` is set:
//!
//!   * [`DocDiscovery`] (producer) periodically polls
//!     `get_docs_by_status(Pending, biz)`, de-dupes in memory, reads each
//!     document's stored content, and sends a [`DocTask`] into a bounded mpsc
//!     channel. It **never claims** — an unclaimed document stays `Pending` so
//!     a cancel or crash can rediscover it.
//!   * [`TaskProcessor`] (consumer) receives tasks, acquires a permit from a
//!     global [`Semaphore`], then claims the document
//!     (`try_claim_for_processing`) and spawns `process_document` on a
//!     [`JoinSet`].
//!   * [`PipelineService`] owns both loops plus the cancel/stop lifecycle and
//!     exposes streaming counters for the status endpoint.
//!
//! **Claim ordering is a hard invariant**: acquire permit → check cancel →
//! claim → spawn. A document that is cancelled or never dequeued is therefore
//! never claimed and remains `Pending`, exactly like the batch path's
//! post-cancel residual (see `cancel_during_run_skips_remaining_documents`).
//!
//! Everything here is additive: with the flag off, [`PipelineService`] is never
//! constructed and the batch path in `ingest.rs` is byte-for-byte unchanged.

use crate::ingest::IngestionPipeline;
use crate::parser::DocumentProcessingOptions;
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, watch, Notify, Semaphore};
use tokio::task::JoinSet;

/// Upper bound for the discovery adaptive backoff. A productive poll (≥1 doc
/// dispatched) resets the interval to `poll_interval`; consecutive empty polls
/// double it up to this ceiling so an idle service costs almost nothing.
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// A unit of work handed from discovery to the processor. Carries everything
/// [`IngestionPipeline::process_document`] needs so the consumer never re-reads
/// the document record (mirrors the per-doc fields the batch path plucks inline
/// in `run_pipeline`).
#[derive(Clone, Debug)]
pub struct DocTask {
    pub doc_id: String,
    /// The document's own business line (per-doc resolved; never empty).
    pub biz_id: String,
    pub content: String,
    pub file_path: String,
    pub doc_options: DocumentProcessingOptions,
    pub create_time_override: Option<i64>,
    pub user_metadata: Option<HashMap<String, String>>,
}

/// Join handles for the two background loops, kept so [`PipelineService::stop`]
/// can await a graceful drain.
struct Handles {
    discovery: tokio::task::JoinHandle<()>,
    processor: tokio::task::JoinHandle<()>,
}

/// Persistent discovery→channel→processor service for one workspace.
///
/// A single instance is created for the default workspace when streaming is
/// enabled (see `easyrag-app` wiring). Per-workspace instances bound to the LRU
/// lifecycle are a Phase 5 follow-up.
pub struct PipelineService {
    pipeline: Arc<IngestionPipeline>,
    doc_status: Arc<dyn DocStatusStorage>,
    kv: Arc<dyn KVStorage>,
    /// Business-line filter for discovery. Empty = wildcard (all lines); each
    /// document is still read/written under its own recorded biz line.
    biz_id: String,
    poll_interval: Duration,
    channel_capacity: usize,
    /// Global concurrency ceiling — the same permit count the batch path uses,
    /// so streaming never changes the concurrency invariant.
    semaphore: Arc<Semaphore>,
    /// Wakes discovery out of its backoff sleep (nudged by `ensure_running`).
    notify: Arc<Notify>,
    /// Documents dispatched but not yet finished; shared by discovery (insert)
    /// and processor (remove) for in-memory de-duplication.
    in_flight: Arc<Mutex<HashSet<String>>>,
    total_processed: Arc<AtomicUsize>,
    /// Buffered-but-not-yet-dequeued task count. Tracked manually because
    /// `mpsc::Sender` exposes no `len` (only the `Receiver` does, and it lives
    /// inside the processor task): incremented on send, decremented on
    /// recv/drain.
    channel_len: Arc<AtomicUsize>,
    /// `compare_exchange` guard so concurrent `ensure_running` calls spawn the
    /// loops at most once.
    service_active: AtomicBool,
    /// Current channel sender (replaced per run) — read by `channel_len`.
    sender: Mutex<Option<mpsc::Sender<DocTask>>>,
    /// Current cancel signal (replaced per run) — `cancel` flips it to `true`.
    cancel_tx: Mutex<Option<watch::Sender<bool>>>,
    handles: Mutex<Option<Handles>>,
}

impl PipelineService {
    /// Build a service. `channel_capacity` defaults to the pipeline's
    /// `queue_size` (activating that otherwise-unused field) when `None`.
    pub fn new(
        pipeline: Arc<IngestionPipeline>,
        doc_status: Arc<dyn DocStatusStorage>,
        kv: Arc<dyn KVStorage>,
        biz_id: impl Into<String>,
        poll_interval_ms: u64,
        channel_capacity: Option<usize>,
    ) -> Self {
        let channel_capacity = channel_capacity
            .unwrap_or_else(|| pipeline.queue_size())
            .max(1);
        let semaphore = Arc::new(Semaphore::new(pipeline.max_concurrency().max(1)));
        Self {
            pipeline,
            doc_status,
            kv,
            biz_id: biz_id.into(),
            poll_interval: Duration::from_millis(poll_interval_ms.max(1)),
            channel_capacity,
            semaphore,
            notify: Arc::new(Notify::new()),
            in_flight: Arc::new(Mutex::new(HashSet::new())),
            total_processed: Arc::new(AtomicUsize::new(0)),
            channel_len: Arc::new(AtomicUsize::new(0)),
            service_active: AtomicBool::new(false),
            sender: Mutex::new(None),
            cancel_tx: Mutex::new(None),
            handles: Mutex::new(None),
        }
    }

    /// Ensure the discovery + processor loops are running. Idempotent: only the
    /// winner of the `compare_exchange` spawns; every call nudges discovery so a
    /// freshly inserted document is picked up promptly instead of waiting out
    /// the adaptive backoff. Mirrors the batch `start()` entry point.
    pub fn ensure_running(self: &Arc<Self>) {
        if self
            .service_active
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            self.spawn_loops();
        } else {
            tracing::debug!("Streaming pipeline already running; nudging discovery");
        }
        // Wake discovery immediately (Phase 5 wires this at the insert site).
        self.notify.notify_one();
    }

    fn spawn_loops(self: &Arc<Self>) {
        // Fresh channel + cancel signal per run; clear stale in-flight state.
        self.in_flight.lock().unwrap().clear();
        self.channel_len.store(0, Ordering::SeqCst);
        let (sender, receiver) = mpsc::channel(self.channel_capacity);
        let (cancel_tx, cancel_rx) = watch::channel(false);
        *self.sender.lock().unwrap() = Some(sender.clone());
        *self.cancel_tx.lock().unwrap() = Some(cancel_tx);

        let discovery = {
            let svc = Arc::clone(self);
            let cancel_rx = cancel_rx.clone();
            tokio::spawn(async move {
                Self::run_discovery(svc, sender, cancel_rx).await;
            })
        };
        let processor = {
            let svc = Arc::clone(self);
            tokio::spawn(async move {
                Self::run_processor(svc, receiver, cancel_rx).await;
            })
        };
        *self.handles.lock().unwrap() = Some(Handles {
            discovery,
            processor,
        });
        tracing::info!(
            "Streaming pipeline started (channel_capacity={}, concurrency={})",
            self.channel_capacity,
            self.pipeline.max_concurrency()
        );
    }

    /// Request cancellation: discovery stops polling and the processor drains
    /// the queue (un-claimed documents stay `Pending`) while letting in-flight
    /// documents finish. Non-blocking, mirroring the batch `cancel()`.
    pub fn cancel(&self) {
        if let Some(tx) = self.cancel_tx.lock().unwrap().as_ref() {
            // A send error just means the loops already exited.
            let _ = tx.send(true);
        }
        tracing::info!("Streaming pipeline cancellation requested");
    }

    /// Graceful shutdown: cancel, await the processor's in-flight drain, then
    /// force a final index commit (invariant #7). Called during app shutdown
    /// after HTTP traffic is drained and before workspace finalize.
    pub async fn stop(&self) {
        self.cancel();
        // Take the handles (dropping the guard before awaiting).
        let handles = self.handles.lock().unwrap().take();
        if let Some(h) = handles {
            let _ = h.discovery.await;
            // The processor drains queued tasks and awaits its JoinSet before
            // returning, so this join covers all in-flight documents.
            let _ = h.processor.await;
        }
        if let Err(e) = self.pipeline.commit_indexes().await {
            tracing::warn!("Streaming pipeline stop: final commit failed: {e}");
        }
        *self.sender.lock().unwrap() = None;
        *self.cancel_tx.lock().unwrap() = None;
        self.service_active.store(false, Ordering::SeqCst);
        tracing::info!("Streaming pipeline stopped");
    }

    /// True while the discovery/processor loops are alive.
    pub fn is_active(&self) -> bool {
        self.service_active.load(Ordering::SeqCst)
    }

    /// Documents dispatched but not yet finished (delegated to the pipeline so
    /// `busy` and this counter stay consistent).
    pub fn active_count(&self) -> usize {
        self.pipeline.active_count()
    }

    /// Cumulative documents processed since the service was built.
    pub fn total_processed(&self) -> usize {
        self.total_processed.load(Ordering::SeqCst)
    }

    /// Tasks currently buffered in the channel (atomic read, no DB cost).
    pub fn channel_len(&self) -> usize {
        self.channel_len.load(Ordering::SeqCst)
    }

    // -----------------------------------------------------------------
    // Discovery (producer)
    // -----------------------------------------------------------------

    async fn run_discovery(
        self: Arc<Self>,
        sender: mpsc::Sender<DocTask>,
        mut cancel_rx: watch::Receiver<bool>,
    ) {
        let mut backoff = self.poll_interval;
        loop {
            tokio::select! {
                // Wrapped so this arm's output is `()`: the `watch::Ref` that
                // `wait_for` yields is `!Send` and would otherwise poison the
                // spawned task across the other arm's await points.
                _ = async { cancel_rx.wait_for(|c| *c).await.ok(); } => break,
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(backoff) => {}
            }
            // The select! statement has ended, so borrowing cancel_rx is safe.
            if *cancel_rx.borrow() {
                break;
            }
            let dispatched = self.discover_once(&sender, &mut cancel_rx).await;
            backoff = if dispatched > 0 {
                self.poll_interval
            } else {
                (backoff * 2).min(MAX_BACKOFF)
            };
            // Processor gone (cancelled/stopped): nothing will consume the queue.
            if sender.is_closed() {
                break;
            }
        }
        tracing::info!("Streaming discovery exited");
    }

    /// One discovery pass. Returns the number of tasks dispatched so the caller
    /// can drive the adaptive backoff.
    async fn discover_once(
        &self,
        sender: &mpsc::Sender<DocTask>,
        cancel_rx: &mut watch::Receiver<bool>,
    ) -> usize {
        let pending = match self
            .doc_status
            .get_docs_by_status(DocStatus::Pending, &self.biz_id)
            .await
        {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Streaming discovery: get_docs_by_status failed: {e}");
                return 0;
            }
        };
        if pending.is_empty() {
            return 0;
        }
        let mut dispatched = 0usize;
        for (doc_id, doc_status) in pending.iter() {
            if self.is_cancelled(cancel_rx) {
                break;
            }
            // In-memory de-dup: skip anything already dispatched and unfinished.
            if self.in_flight.lock().unwrap().contains(doc_id) {
                continue;
            }
            // Per-doc biz line (mirrors run_pipeline): a wildcard discovery run
            // matches every line, so each doc is read/written under its own.
            // Never fall through to an empty biz_id.
            let biz_id = if !doc_status.biz_id.is_empty() {
                doc_status.biz_id.clone()
            } else if !self.biz_id.is_empty() {
                self.biz_id.clone()
            } else {
                tracing::warn!("Streaming discovery: skipping pending doc {doc_id}: no biz line");
                continue;
            };
            let doc_data = match self.kv.get_by_id(doc_id, &biz_id).await {
                Ok(Some(data)) => data,
                _ => continue,
            };
            let content = match doc_data.get("content").and_then(|v| v.as_str()) {
                Some(c) => c.to_string(),
                None => continue,
            };
            let file_path = doc_data
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let task = DocTask {
                doc_id: doc_id.clone(),
                biz_id,
                content,
                file_path,
                doc_options: IngestionPipeline::rebuild_options_from_doc_data(&doc_data),
                create_time_override: doc_data.get("create_time_epoch").and_then(|v| v.as_i64()),
                user_metadata: IngestionPipeline::extract_user_metadata(&doc_data),
            };
            // Mark in-flight BEFORE send so the next pass won't re-dispatch it.
            self.in_flight.lock().unwrap().insert(doc_id.clone());
            tokio::select! {
                _ = async { cancel_rx.wait_for(|c| *c).await.ok(); } => {
                    self.in_flight.lock().unwrap().remove(doc_id);
                    break;
                }
                res = sender.send(task) => {
                    if res.is_err() {
                        // Receiver dropped: undo the in-flight mark and stop.
                        self.in_flight.lock().unwrap().remove(doc_id);
                        break;
                    }
                }
            }
            self.channel_len.fetch_add(1, Ordering::SeqCst);
            dispatched += 1;
        }
        dispatched
    }

    fn is_cancelled(&self, cancel_rx: &watch::Receiver<bool>) -> bool {
        *cancel_rx.borrow()
    }

    // -----------------------------------------------------------------
    // Processor (consumer)
    // -----------------------------------------------------------------

    async fn run_processor(
        self: Arc<Self>,
        mut receiver: mpsc::Receiver<DocTask>,
        mut cancel_rx: watch::Receiver<bool>,
    ) {
        // A separate receiver for level checks inside the recv handler: the
        // select! arm below borrows `cancel_rx` mutably for the whole statement,
        // so the handler must not touch it directly.
        let cancel_flag = cancel_rx.clone();
        let mut join_set: JoinSet<()> = JoinSet::new();
        loop {
            tokio::select! {
                _ = async { cancel_rx.wait_for(|c| *c).await.ok(); } => {
                    tracing::info!("Streaming processor: cancel signalled, draining queue");
                    break;
                }
                maybe_task = receiver.recv() => {
                    let task = match maybe_task {
                        Some(t) => t,
                        None => {
                            tracing::info!("Streaming processor: channel closed");
                            break;
                        }
                    };
                    // Task left the buffer.
                    self.channel_len.fetch_sub(1, Ordering::SeqCst);
                    // Claim ordering invariant: permit → cancel check → claim → spawn.
                    let permit = match self.semaphore.clone().acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    if *cancel_flag.borrow() {
                        // Cancelled while waiting for a permit: leave it Pending.
                        drop(permit);
                        self.in_flight.lock().unwrap().remove(&task.doc_id);
                        break;
                    }
                    match self
                        .doc_status
                        .try_claim_for_processing(&task.doc_id, &task.biz_id)
                        .await
                    {
                        Ok(true) => {}
                        Ok(false) => {
                            // Already claimed/processed elsewhere: not our work.
                            drop(permit);
                            self.in_flight.lock().unwrap().remove(&task.doc_id);
                            continue;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Streaming processor: claim failed for {}: {e}",
                                task.doc_id
                            );
                            drop(permit);
                            self.in_flight.lock().unwrap().remove(&task.doc_id);
                            continue;
                        }
                    }
                    self.pipeline.incr_active();
                    let pipeline = Arc::clone(&self.pipeline);
                    let in_flight = Arc::clone(&self.in_flight);
                    let total = Arc::clone(&self.total_processed);
                    join_set.spawn(async move {
                        pipeline
                            .process_document(
                                &task.doc_id,
                                &task.content,
                                &task.file_path,
                                Some(&task.doc_options),
                                task.create_time_override,
                                task.user_metadata.as_ref(),
                                &task.biz_id,
                            )
                            .await;
                        in_flight.lock().unwrap().remove(&task.doc_id);
                        pipeline.decr_active();
                        total.fetch_add(1, Ordering::SeqCst);
                        // Release the permit last so concurrency stays bounded.
                        drop(permit);
                    });
                }
            }
        }
        // Drain queued-but-unprocessed tasks: they were never claimed, so
        // removing them from in-flight leaves them Pending for rediscovery.
        while let Ok(task) = receiver.try_recv() {
            self.channel_len.fetch_sub(1, Ordering::SeqCst);
            self.in_flight.lock().unwrap().remove(&task.doc_id);
        }
        // Let in-flight documents finish (invariant #7: graceful drain).
        while join_set.join_next().await.is_some() {}
        self.service_active.store(false, Ordering::SeqCst);
        tracing::info!("Streaming processor exited");
    }
}
