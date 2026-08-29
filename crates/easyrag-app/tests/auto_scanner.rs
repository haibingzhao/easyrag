//! Auto scanner integration tests (spec test item 3): drive single scan
//! cycles end-to-end against in-memory backends — a seeded pending document
//! is consumed by one PendingPoll cycle, and a DirScan cycle persists its
//! position without re-inserting unchanged files on the second round.

use std::sync::Arc;

use async_trait::async_trait;
use easyrag_app::scanner::AutoScanner;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::{ScanPosition, ScanTaskKind, ScanTaskSpec};
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage, StorageBundle};
use easyrag_pipeline::{IngestionPipeline, QueryEngine};
use easyrag_server::service::{ResolvedScope, WorkspaceContext, WorkspaceResolver};
use easyrag_server::{AppResult, AutoScanStats};
use easyrag_storage::memory::{
 InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
use serde_json::Value as JsonValue;
use tokio::sync::oneshot;

struct ConstFunc;

#[async_trait]
impl EmbeddingFunction for ConstFunc {
 async fn embed(
 &self,
 texts: Vec<String>,
 _context: Option<EmbeddingContext>,
 ) -> easyrag_core::Result<Vec<Vec<f32>>> {
 Ok(texts.iter.map(|_| vec![1f32, 0f32, 0f32, 0f32]).collect)
 }
}

struct MockLlm;

#[async_trait]
impl LlmService for MockLlm {
 async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
 Ok("MOCK-ANSWER".to_string)
 }

 fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
 Box::pin(futures::stream::empty)
 }
}

/// Single-context resolver mirroring the production normalize semantics
/// (blank biz_id -> None = no biz filtering).
struct StubResolver {
 context: WorkspaceContext,
}

#[async_trait]
impl WorkspaceResolver for StubResolver {
 async fn resolve(
 &self,
 _workspace: Option<&str>,
 biz_id: Option<&str>,
 ) -> AppResult<ResolvedScope> {
 let normalized = biz_id
 .map(str::trim)
 .filter(|s| !s.is_empty)
 .map(str::to_string);
 Ok(ResolvedScope {
 context: self.context.clone,
 biz_id: normalized,
 })
 }
}

struct Harness {
 pipeline: Arc<IngestionPipeline>,
 doc_status: Arc<InMemoryDocStatusStorage>,
 position_kv: Arc<InMemoryKVStorage>,
 resolver: Arc<dyn WorkspaceResolver>,
 _temp_dir: tempfile::TempDir,
}

fn build_harness -> Harness {
 let temp_dir = tempfile::tempdir.unwrap;
 let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
 let embedding = Arc::new(EmbeddingFunc::new(4, Arc::new(ConstFunc)));
 let vector = Arc::new(InMemoryVectorStorage::new(
 "vdb",
 "scan-test",
 EmbeddingFunc::new(4, Arc::new(ConstFunc)),
 ));
 let graph = Arc::new(InMemoryGraphStorage::new("graph", "scan-test"));
 let kv = Arc::new(InMemoryKVStorage::new("kv", "scan-test"));
 let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "scan-test"));
 let pipeline = Arc::new(IngestionPipeline::new(
 llm.clone,
 kv.clone,
 vector.clone,
 graph.clone,
 doc_status.clone,
 embedding,
 ));
 let query_engine = Arc::new(QueryEngine::new(
 llm.clone,
 vector.clone,
 graph.clone,
 kv.clone,
 ));
 let resolver: Arc<dyn WorkspaceResolver> = Arc::new(StubResolver {
 context: WorkspaceContext {
 bundle: StorageBundle {
 workspace: "default".to_string,
 kv_storage: kv.clone,
 vector_storage: vector.clone,
 graph_storage: graph.clone,
 doc_status_storage: doc_status.clone,
 },
 pipeline: pipeline.clone,
 query_engine,
 structure_strategy: None,
 },
 });
 let position_kv = Arc::new(InMemoryKVStorage::new("positions", "scan-test"));
 Harness {
 pipeline,
 doc_status,
 position_kv,
 resolver,
 _temp_dir: temp_dir,
 }
}

fn make_scanner(harness: &Harness, tasks: Vec<ScanTaskSpec>) -> AutoScanner {
 let (_tx, rx) = oneshot::channel::<>;
 AutoScanner::new(
 tasks,
 Arc::clone(&harness.resolver),
 harness.position_kv.clone,
 Arc::new(AutoScanStats::new),
 rx,
 )
}

fn pending_poll_task -> ScanTaskSpec {
 ScanTaskSpec {
 task_id: "poll-test".to_string,
 kind: ScanTaskKind::PendingPoll,
 workspace: "default".to_string,
 biz_id: String::new,
 interval_secs: 60,
 enabled: true,
 input_dir: None,
 extensions: Vec::new,
 max_files_per_cycle: 0,
 }
}

#[tokio::test]
async fn pending_poll_cycle_consumes_pending_documents {
 let harness = build_harness;
 harness
 .pipeline
 .insert_document("hello auto scan", "manual.txt", None, None, "default")
 .await
 .unwrap;
 assert_eq!(
 harness
 .doc_status
 .get_docs_by_status(DocStatus::Pending, "default")
 .await
 .unwrap
 .len,
 1
 );

 let scanner = make_scanner(&harness, vec![pending_poll_task]);
 scanner.run_cycle(&pending_poll_task).await;

 // Pipeline consumed the pending document (processed or failed — the
 // mock LLM decides — but it must no longer be pending).
 assert!(harness
 .doc_status
 .get_docs_by_status(DocStatus::Pending, "default")
 .await
 .unwrap
 .is_empty);
 let cycle = &scanner.stats.snapshot[0];
 assert_eq!(cycle.docs_seen, 1);
 assert_eq!(cycle.docs_triggered_or_inserted, 1);
 assert!(cycle.error.is_none);

 // Position persisted with a forward watermark and processed counter.
 let pos_map = harness
 .position_kv
 .get_by_id(&ScanPosition::storage_key("poll-test"), "")
 .await
 .unwrap
 .expect("position row persisted");
 let pos: ScanPosition = serde_json::from_value(JsonValue::Object(pos_map)).unwrap;
 assert!(pos.watermark_epoch_ms > 0);
 assert_eq!(pos.docs_processed_total, 1);
 assert!(pos.last_error.is_none);
}

#[tokio::test]
async fn pending_poll_cycle_is_noop_without_pending {
 let harness = build_harness;
 let scanner = make_scanner(&harness, vec![pending_poll_task]);
 scanner.run_cycle(&pending_poll_task).await;
 let cycle = &scanner.stats.snapshot[0];
 assert_eq!(cycle.docs_seen, 0);
 assert_eq!(cycle.docs_triggered_or_inserted, 0);
}

#[tokio::test]
async fn dir_scan_cycle_inserts_once_and_persists_position {
 let harness = build_harness;
 let inbox = harness._temp_dir.path.join("inbox");
 std::fs::create_dir_all(&inbox).unwrap;
 std::fs::write(inbox.join("a.txt"), "first version").unwrap;

 let task = ScanTaskSpec {
 task_id: "dir-test".to_string,
 kind: ScanTaskKind::DirScan,
 workspace: "default".to_string,
 biz_id: String::new,
 interval_secs: 60,
 enabled: true,
 input_dir: Some(inbox.to_str.unwrap.to_string),
 extensions: vec![".txt".to_string],
 max_files_per_cycle: 100,
 };
 let scanner = make_scanner(&harness, vec![task.clone]);

 // First cycle: inserts the new file.
 scanner.run_cycle(&task).await;
 let cycle = &scanner.stats.snapshot[0];
 assert_eq!(cycle.docs_seen, 1);
 assert_eq!(cycle.docs_triggered_or_inserted, 1);

 // Second cycle: same file + hash -> dedup skip, counter unchanged.
 scanner.run_cycle(&task).await;
 let cycle = &scanner.stats.snapshot[0];
 assert_eq!(cycle.docs_seen, 1);
 assert_eq!(cycle.docs_triggered_or_inserted, 0);
 let pos_map = harness
 .position_kv
 .get_by_id(&ScanPosition::storage_key("dir-test"), "")
 .await
 .unwrap
 .expect("position row persisted");
 let pos: ScanPosition = serde_json::from_value(JsonValue::Object(pos_map)).unwrap;
 assert_eq!(pos.docs_inserted_total, 1);

 // Changed content -> re-inserted as a new document version.
 std::fs::write(inbox.join("a.txt"), "second version").unwrap;
 scanner.run_cycle(&task).await;
 let cycle = &scanner.stats.snapshot[0];
 assert_eq!(cycle.docs_triggered_or_inserted, 1);
}

#[tokio::test]
async fn dir_scan_cycle_missing_dir_records_error_without_crashing {
 let harness = build_harness;
 let task = ScanTaskSpec {
 task_id: "dir-missing".to_string,
 kind: ScanTaskKind::DirScan,
 workspace: "default".to_string,
 biz_id: String::new,
 interval_secs: 60,
 enabled: true,
 input_dir: Some("/nonexistent/auto-scan-dir".to_string),
 extensions: vec![".txt".to_string],
 max_files_per_cycle: 100,
 };
 let scanner = make_scanner(&harness, vec![task.clone]);
 scanner.run_cycle(&task).await;

 let cycle = &scanner.stats.snapshot[0];
 assert!(cycle.error.is_some);
 let pos_map = harness
 .position_kv
 .get_by_id(&ScanPosition::storage_key("dir-missing"), "")
 .await
 .unwrap
 .expect("position row persisted even on error");
 let pos: ScanPosition = serde_json::from_value(JsonValue::Object(pos_map)).unwrap;
 assert!(pos.last_error.is_some);
}
