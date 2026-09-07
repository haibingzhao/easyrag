//! Background auto-scan service (inspired by the easy-rss scheduler design):
//! multiple independent scan tasks, each with its own enable switch, interval
//! and persisted scan position (KV-backed, one record per task — the analog of
//! easy-rss `rag_watermark` rows). Every cycle is a single-pass function with
//! full error suppression so the loop never exits (easy-rss `run_*_poll`
//! pattern); pipeline re-entrancy is guarded by `IngestionPipeline::start`'s
//! built-in compare_exchange (easy-rss `max_instances=1 + coalesce`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::{ScanCycleStats, ScanPosition, ScanTaskKind, ScanTaskSpec};
use easyrag_core::storage::traits::KVStorage;
use easyrag_server::service::WorkspaceResolver;
use easyrag_server::{AutoScanStats, ServerConfig};
use serde_json::Value as JsonValue;
use tokio::sync::oneshot;

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
}

/// Derive the effective task list from the server config: explicit
/// `auto_scan_tasks` win; an empty list yields one implicit pending-poll task
/// on the default workspace; a non-empty `auto_scan_input_dir` appends one
/// implicit directory-scan task.
pub fn build_scan_tasks(config: &ServerConfig) -> Vec<ScanTaskSpec> {
    let mut tasks = config.auto_scan_tasks.clone();
    if tasks.is_empty() {
        tasks.push(ScanTaskSpec {
            task_id: "pending-poll-default".to_string(),
            kind: ScanTaskKind::PendingPoll,
            workspace: config.workspace.clone(),
            biz_id: String::new(),
            interval_secs: config.auto_scan_interval_seconds,
            enabled: true,
            input_dir: None,
            extensions: Vec::new(),
            max_files_per_cycle: 0,
        });
    }
    let input_dir = config.auto_scan_input_dir.trim();
    if !input_dir.is_empty() {
        let extensions = config
            .auto_scan_extensions
            .split(',')
            .map(|e| e.trim().to_lowercase())
            .filter(|e| !e.is_empty())
            .collect();
        tasks.push(ScanTaskSpec {
            task_id: "dir-scan-default".to_string(),
            kind: ScanTaskKind::DirScan,
            workspace: config.workspace.clone(),
            biz_id: String::new(),
            interval_secs: config.auto_scan_interval_seconds,
            enabled: true,
            input_dir: Some(input_dir.to_string()),
            extensions,
            max_files_per_cycle: config.auto_scan_max_files_per_cycle,
        });
    }
    tasks
}

/// Merge a freshly computed position into the persisted one. The watermark
/// only moves forward (easy-rss monotonic UPSERT semantics); counters are
/// taken from the fresh value which already carries cumulative totals.
fn merge_position(existing: &ScanPosition, fresh: ScanPosition) -> ScanPosition {
    let mut merged = fresh;
    if merged.watermark_epoch_ms < existing.watermark_epoch_ms {
        merged.watermark_epoch_ms = existing.watermark_epoch_ms;
    }
    merged
}

/// Task due-check: a task with no recorded schedule yet runs immediately
/// (first cycle after boot); otherwise it runs at or past its next_run_at.
fn is_due(scheduled: Option<&Instant>, now: Instant) -> bool {
    scheduled.map_or(true, |scheduled| now >= *scheduled)
}

/// Outcome of one scan cycle: (docs seen, docs acted on).
type CycleResult = Result<(usize, usize), String>;

pub struct AutoScanner {
    tasks: Vec<ScanTaskSpec>,
    workspace_resolver: Arc<dyn WorkspaceResolver>,
    /// Default-workspace KV used as the position store (one record per task).
    kv: Arc<dyn KVStorage>,
    stats: Arc<AutoScanStats>,
    stop_rx: oneshot::Receiver<()>,
    /// When true, the persistent streaming `PipelineService` (started at app
    /// wiring) already discovers and processes pending documents with per-doc
    /// claim, so the scanner must NOT trigger the batch `start()` path — it
    /// does not claim per-document and would race the streaming processor. Set
    /// via `with_streaming`; `new()` leaves it false so existing callers/tests
    /// are byte-for-byte unchanged.
    streaming_enabled: bool,
}

impl AutoScanner {
    pub fn new(
        tasks: Vec<ScanTaskSpec>,
        workspace_resolver: Arc<dyn WorkspaceResolver>,
        kv: Arc<dyn KVStorage>,
        stats: Arc<AutoScanStats>,
        stop_rx: oneshot::Receiver<()>,
    ) -> Self {
        stats.register_tasks(tasks.clone());
        Self {
            tasks,
            workspace_resolver,
            kv,
            stats,
            stop_rx,
            streaming_enabled: false,
        }
    }

    /// Opt into the streaming pipeline path (BC-11 gradual rollout). Builder
    /// style so `new()`'s signature — and every existing caller/test — stays
    /// unchanged; defaults to false (batch behaviour).
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming_enabled = enabled;
        self
    }

    /// Stats handle for observability endpoints and tests.
    pub fn stats(&self) -> Arc<AutoScanStats> {
        Arc::clone(&self.stats)
    }

    /// Scheduler loop: 1s ticker; each due task runs one single-pass cycle.
    /// A long cycle occupies the tick (coalesce semantics — due tasks simply
    /// run on the next tick; pipeline re-entrancy is guarded by `start`).
    pub async fn run(mut self) {
        tracing::info!("Auto scanner started with {} task(s)", self.tasks.len());
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // In-memory per-task schedule; first cycle runs immediately after boot.
        let mut next_run_at: HashMap<String, Instant> = HashMap::new();

        loop {
            tokio::select! {
                _ = &mut self.stop_rx => {
                    tracing::info!("Auto scanner stop signal received");
                    break;
                }
                _ = interval.tick() => {}
            }
            let now = Instant::now();
            for task in &self.tasks {
                if !task.enabled {
                    continue;
                }
                let due = is_due(next_run_at.get(&task.task_id), now);
                if !due {
                    continue;
                }
                self.run_cycle(task).await;
                next_run_at.insert(
                    task.task_id.clone(),
                    Instant::now() + Duration::from_secs(task.interval_secs.max(1)),
                );
            }
        }
        tracing::info!("Auto scanner stopped");
    }

    /// One single-pass cycle for a task: execute, suppress all errors
    /// (warn-log only), then record stats and persist the position. Public
    /// so integration tests (and a future manual-trigger endpoint) can run
    /// a cycle without the scheduler loop.
    pub async fn run_cycle(&self, task: &ScanTaskSpec) {
        let started_at = now_iso();
        let started = Instant::now();
        let result = match task.kind {
            ScanTaskKind::PendingPoll => self.run_pending_poll(task).await,
            ScanTaskKind::DirScan => self.run_dir_scan(task).await,
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        let (seen, acted, error) = match &result {
            Ok((seen, acted)) => (*seen, *acted, None),
            Err(e) => {
                tracing::warn!("Auto scan task '{}' cycle failed: {e}", task.task_id);
                (0, 0, Some(e.as_str()))
            }
        };
        self.stats.record(ScanCycleStats {
            task_id: task.task_id.clone(),
            started_at: started_at.clone(),
            duration_ms,
            docs_seen: seen,
            docs_triggered_or_inserted: acted,
            error: error.map(str::to_string),
        });

        // Advance the persisted position (easy-rss watermark pattern).
        let pos = self.load_position(&task.task_id).await;
        let now_ms = chrono::Utc::now().timestamp_millis();
        let fresh = ScanPosition {
            task_id: task.task_id.clone(),
            watermark_epoch_ms: now_ms,
            last_run_at: Some(started_at),
            last_error: error.map(str::to_string),
            docs_processed_total: pos.docs_processed_total + acted as u64,
            docs_inserted_total: pos.docs_inserted_total
                + if task.kind == ScanTaskKind::DirScan {
                    acted as u64
                } else {
                    0
                },
            updated_at: now_iso(),
        };
        self.save_position(&merge_position(&pos, fresh)).await;
    }

    /// PendingPoll cycle: same path as `POST /api/documents/process` —
    /// workspace-specific storage via resolve, trigger the pipeline only
    /// when pending documents exist (`start` is idempotent/re-entrancy-safe).
    async fn run_pending_poll(&self, task: &ScanTaskSpec) -> CycleResult {
        // Streaming mode: the persistent PipelineService discovery loop
        // (started at app wiring via `ensure_running`, guaranteed alive before
        // the scanner is spawned) already polls Pending and processes documents
        // with per-doc claim. Triggering the batch `start()` here would race it
        // (run_pipeline does not claim per-document), so this cycle is a no-op —
        // discovery is the single source of pending-document work.
        if self.streaming_enabled {
            return Ok((0, 0));
        }
        let scope = self
            .workspace_resolver
            .resolve(Some(&task.workspace), Some(&task.biz_id))
            .await
            .map_err(|e| format!("resolve workspace '{}': {e:?}", task.workspace))?;
        // Blank task biz_id = wildcard: watch every biz line of the workspace.
        // resolve() would normalize it to the default line, so restore the
        // empty-string wildcard here; run_pipeline resolves each document's
        // own biz line before any write.
        let biz = if task.biz_id.trim().is_empty() {
            String::new()
        } else {
            scope.biz_id.clone()
        };
        let pending = scope
            .context
            .bundle
            .doc_status_storage
            .get_docs_by_status(DocStatus::Pending, &biz)
            .await
            .map_err(|e| format!("query pending documents: {e}"))?;
        if pending.is_empty() {
            return Ok((0, 0));
        }
        let count = pending.len();
        // Pipeline already running: start is a no-op (compare_exchange), so
        // nothing was actually triggered — report seen-but-not-acted to keep
        // the processed counter honest.
        let busy = scope
            .context
            .pipeline
            .pipeline_status()
            .get("busy")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if busy {
            tracing::info!(
                "Auto scan task '{}' found {count} pending document(s) but pipeline is busy; skipping trigger",
                task.task_id
            );
            return Ok((count, 0));
        }
        tracing::info!(
            "Auto scan task '{}' found {count} pending document(s); starting pipeline",
            task.task_id
        );
        scope.context.pipeline.start(&biz).await;
        Ok((count, count))
    }

    /// DirScan cycle: incrementally scan the input directory (shared
    /// `scan_directory` dedup logic) and trigger the pipeline for new docs.
    async fn run_dir_scan(&self, task: &ScanTaskSpec) -> CycleResult {
        let dir = task
            .input_dir
            .as_deref()
            .filter(|d| !d.trim().is_empty())
            .ok_or_else(|| "task has no input_dir configured".to_string())?;
        let path = std::path::Path::new(dir);
        if !path.exists() || !path.is_dir() {
            return Err(format!("directory not found: {dir}"));
        }
        let scope = self
            .workspace_resolver
            .resolve(Some(&task.workspace), Some(&task.biz_id))
            .await
            .map_err(|e| format!("resolve workspace '{}': {e:?}", task.workspace))?;
        // Write path: None -> "default" (same convention as every HTTP write
        // path, e.g. documents.rs scan_input_dir), so auto-scanned documents
        // land in the same partition as manual inserts and stay dedup-consistent.
        let biz = if scope.biz_id.is_empty() {
            easyrag_core::constants::DEFAULT_BIZ_ID.to_string()
        } else {
            scope.biz_id.clone()
        };
        let outcome = easyrag_pipeline::scan_directory(
            &scope.context.pipeline,
            &scope.context.bundle.doc_status_storage,
            path,
            &task.extensions,
            task.max_files_per_cycle,
            &biz,
        )
        .await
        .map_err(|e| format!("scan directory '{dir}': {e}"))?;
        if !outcome.inserted_doc_ids.is_empty() {
            if self.streaming_enabled {
                // Streaming mode: inserted documents are already Pending and the
                // persistent discovery loop picks them up (with per-doc claim).
                // Do NOT call the batch `start()` — it would race the streaming
                // processor. Directory scan/dedup logic is unchanged.
                tracing::info!(
                    "Auto scan task '{}' inserted {} document(s) from '{dir}'; streaming discovery will pick them up",
                    task.task_id,
                    outcome.inserted_doc_ids.len()
                );
            } else {
                tracing::info!(
                    "Auto scan task '{}' inserted {} document(s) from '{dir}'; starting pipeline",
                    task.task_id,
                    outcome.inserted_doc_ids.len()
                );
                scope.context.pipeline.start(&biz).await;
            }
        }
        Ok((outcome.files_seen, outcome.inserted_doc_ids.len()))
    }

    async fn load_position(&self, task_id: &str) -> ScanPosition {
        let key = ScanPosition::storage_key(task_id);
        match self.kv.get_by_id(&key, "").await {
            Ok(Some(map)) => serde_json::from_value(JsonValue::Object(map)).unwrap_or_else(|e| {
                tracing::warn!("Corrupt auto-scan position for '{task_id}': {e}; starting fresh");
                ScanPosition {
                    task_id: task_id.to_string(),
                    ..Default::default()
                }
            }),
            _ => ScanPosition {
                task_id: task_id.to_string(),
                ..Default::default()
            },
        }
    }

    async fn save_position(&self, pos: &ScanPosition) {
        let key = ScanPosition::storage_key(&pos.task_id);
        let Ok(value) = serde_json::to_value(pos) else {
            tracing::warn!("Failed to serialize auto-scan position for '{}'", pos.task_id);
            return;
        };
        let Some(map) = value.as_object().cloned() else {
            return;
        };
        let mut batch = HashMap::new();
        batch.insert(key, map);
        if let Err(e) = self.kv.upsert(batch, "").await {
            // Position persistence is an optimization; correctness relies on
            // dedup/claim logic, so a failure only warrants a warning.
            tracing::warn!("Failed to persist auto-scan position for '{}': {e}", pos.task_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with(auto_scan_input_dir: &str, tasks: Vec<ScanTaskSpec>) -> ServerConfig {
        let mut config = ServerConfig::default();
        config.auto_scan_enabled = true;
        config.workspace = "default".to_string();
        config.auto_scan_interval_seconds = 45;
        config.auto_scan_input_dir = auto_scan_input_dir.to_string();
        config.auto_scan_tasks = tasks;
        config
    }

    #[test]
    fn build_tasks_implicit_pending_poll_when_empty() {
        let tasks = build_scan_tasks(&config_with("", Vec::new()));
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "pending-poll-default");
        assert_eq!(tasks[0].kind, ScanTaskKind::PendingPoll);
        assert_eq!(tasks[0].workspace, "default");
        assert_eq!(tasks[0].interval_secs, 45);
    }

    #[test]
    fn build_tasks_appends_dir_scan_for_input_dir() {
        let tasks = build_scan_tasks(&config_with("./inbox", Vec::new()));
        assert_eq!(tasks.len(), 2);
        let dir_task = &tasks[1];
        assert_eq!(dir_task.kind, ScanTaskKind::DirScan);
        assert_eq!(dir_task.input_dir.as_deref(), Some("./inbox"));
        assert_eq!(dir_task.extensions, vec![".txt", ".md", ".pdf"]);
        assert_eq!(dir_task.max_files_per_cycle, 100);
    }

    #[test]
    fn build_tasks_explicit_list_wins_over_implicit_poll() {
        let explicit = ScanTaskSpec {
            task_id: "poll-ws1".to_string(),
            kind: ScanTaskKind::PendingPoll,
            workspace: "ws1".to_string(),
            biz_id: "biz-a".to_string(),
            interval_secs: 30,
            enabled: true,
            input_dir: None,
            extensions: Vec::new(),
            max_files_per_cycle: 0,
        };
        let tasks = build_scan_tasks(&config_with("", vec![explicit]));
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_id, "poll-ws1");
    }

    #[test]
    fn merge_position_watermark_never_moves_backwards() {
        let existing = ScanPosition {
            task_id: "t".to_string(),
            watermark_epoch_ms: 1_000,
            docs_processed_total: 5,
            ..Default::default()
        };
        // Fresh watermark behind the persisted one (clock skew / stale cycle).
        let fresh = ScanPosition {
            task_id: "t".to_string(),
            watermark_epoch_ms: 500,
            docs_processed_total: 7,
            ..Default::default()
        };
        let merged = merge_position(&existing, fresh);
        assert_eq!(merged.watermark_epoch_ms, 1_000);
        assert_eq!(merged.docs_processed_total, 7);
    }

    #[test]
    fn is_due_scheduling_rules() {
        let now = Instant::now();
        // No recorded schedule yet -> first cycle after boot runs immediately.
        assert!(is_due(None, now));
        // Scheduled in the future -> not due.
        assert!(!is_due(Some(&(now + Duration::from_secs(30))), now));
        // Exactly at the scheduled instant -> due.
        assert!(is_due(Some(&now), now));
        // Scheduled in the past -> overdue, due.
        assert!(is_due(Some(&(now - Duration::from_secs(1))), now));
    }
}
