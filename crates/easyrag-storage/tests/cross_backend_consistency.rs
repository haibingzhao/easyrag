//! Phase 0.5 — Cross-backend consistency guardrail (the most critical Phase 0 test).
//!
//! The same logical document fixture is fed through BOTH the in-memory mapper
//! (`memory::doc_status::map_to_doc_status`) and the PostgreSQL mapper
//! (`postgres::map_row_to_doc_status`), reached via the public `DocStatusStorage`
//! API. The two backends MUST produce identical `DocProcessingStatus` values; any
//! drift is a correctness bug because callers cannot tell which backend they hit.
//!
//! Two known defects were pinned here as RED tests (see `.docs` Appendix A); both
//! are now fixed and GREEN, and stay here as regression guardrails:
//!   * A-1 — `map_row_to_doc_status` hardcoded `chunks_list = []`, `error_msg = None`,
//!           `metadata = {}`, `multimodal_processed = None` (postgres.rs), while the
//!           memory mapper populated them. Fixed in Phase 3.1 (`from_json` uplift, BC-9).
//!   * A-6 — `KnowledgeGraph.is_truncated` used `>` on the `*` path (memory/graph.rs and
//!           postgres.rs) but `>=` on the BFS/CTE/neo4j paths, so the two disagreed at
//!           `nodes.len() == max_nodes`. Unified to `>=` in Phase 3.9 (BC-1).
//!
//! The RED→GREEN transition of each was the Phase 3 fix evidence. The remaining GREEN
//! tests lock the fields that already agreed so a regression there fails loudly.

#![cfg(feature = "postgres")]

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use easyrag_core::model::{DocProcessingStatus, DocStatus};
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage};
use easyrag_core::JsonMap;
use easyrag_storage::memory::{InMemoryDocStatusStorage, InMemoryGraphStorage};
use easyrag_storage::postgres::{PgDb, PgDocStatusStorage, PgGraphStorage};
use parking_lot::Mutex;
use serde_json::{json, Value};

const BIZ: &str = "default";
const WS: &str = "test_ws";

// ---------------------------------------------------------------------------
// Minimal mock PgDb: returns a canned row set from fetch_all.
// ---------------------------------------------------------------------------

struct MockPgDb {
    rows: Vec<JsonMap>,
    last_sql: Mutex<String>,
}

impl MockPgDb {
    fn with_rows(rows: Vec<JsonMap>) -> Self {
        Self {
            rows,
            last_sql: Mutex::new(String::new()),
        }
    }
}

#[async_trait]
impl PgDb for MockPgDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        *self.last_sql.lock() = sql.to_string();
        Ok(())
    }
    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        *self.last_sql.lock() = sql.to_string();
        Ok(self.rows.first().cloned())
    }
    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        *self.last_sql.lock() = sql.to_string();
        Ok(self.rows.clone())
    }
    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        *self.last_sql.lock() = sql.to_string();
        Ok(0)
    }
}

/// Shared-ownership wrapper so the mock can be inspected after moving into storage.
struct ArcDb(Arc<MockPgDb>);

#[async_trait]
impl PgDb for ArcDb {
    async fn execute(&self, sql: &str) -> easyrag_core::Result<()> {
        self.0.execute(sql).await
    }
    async fn fetch_one(&self, sql: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.0.fetch_one(sql).await
    }
    async fn fetch_all(&self, sql: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        self.0.fetch_all(sql).await
    }
    async fn execute_update(&self, sql: &str) -> easyrag_core::Result<u64> {
        self.0.execute_update(sql).await
    }
}

fn as_map(v: Value) -> JsonMap {
    match v {
        Value::Object(m) => m,
        other => panic!("expected object, got {other}"),
    }
}

/// The single logical document fixture, expressed for the MEMORY backend
/// (flat top-level fields — memory reads everything from the record root).
fn memory_doc() -> JsonMap {
    as_map(json!({
        "file_path": "/data/doc1.txt",
        "status": "processed",
        "content_summary": "A short summary",
        "content_length": 1234,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "chunks_count": 3,
        "content_hash": "hash-abc",
        "track_id": "track-1",
        "has_structure": true,
        "external_id": "ext-1",
        // --- A-1 fields (memory populates, postgres drops) ---
        "chunks_list": ["c1", "c2", "c3"],
        "error_msg": "boom",
        "metadata": {"source": "test"},
        "multimodal_processed": true
    }))
}

/// The same logical document expressed as a POSTGRES result row: identity/scope
/// columns live at the top level, everything else is nested inside the `content`
/// JSONB column (postgres.rs `read_content` + `map_row_to_doc_status`).
fn pg_row() -> JsonMap {
    as_map(json!({
        "id": "doc1",
        "biz_id": BIZ,
        "status": "processed",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-02T00:00:00Z",
        "content": {
            "file_path": "/data/doc1.txt",
            "content_summary": "A short summary",
            "content_length": 1234,
            "chunks_count": 3,
            "content_hash": "hash-abc",
            "track_id": "track-1",
            "has_structure": true,
            "external_id": "ext-1",
            // Present in the JSONB payload, but map_row_to_doc_status ignores them (A-1).
            "chunks_list": ["c1", "c2", "c3"],
            "error_msg": "boom",
            "metadata": {"source": "test"},
            "multimodal_processed": true
        }
    }))
}

async fn memory_status() -> DocProcessingStatus {
    let storage = InMemoryDocStatusStorage::new("doc_status", WS);
    storage
        .upsert(HashMap::from([("doc1".to_string(), memory_doc())]), BIZ)
        .await
        .unwrap();
    storage
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap()
        .remove("doc1")
        .expect("memory returned the doc")
}

async fn pg_status() -> DocProcessingStatus {
    let db = Arc::new(MockPgDb::with_rows(vec![pg_row()]));
    let storage =
        PgDocStatusStorage::with_options("doc_status", WS, JsonMap::new(), None, ArcDb(db))
            .expect("valid workspace");
    storage
        .get_docs_by_status(DocStatus::Processed, BIZ)
        .await
        .unwrap()
        .remove("doc1")
        .expect("pg returned the doc")
}

// ---------------------------------------------------------------------------
// GREEN: the fields both backends already agree on. Locks the baseline so a
// future regression in the shared mapping fails here (not silently in prod).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn doc_status_agrees_on_shared_fields() {
    let mem = serde_json::to_value(memory_status().await).unwrap();
    let pg = serde_json::to_value(pg_status().await).unwrap();

    // Fields mapped identically by both backends today.
    for field in [
        "file_path",
        "status",
        "content_summary",
        "content_length",
        "created_at",
        "updated_at",
        "chunks_count",
        "content_hash",
        "track_id",
        "has_structure",
        "external_id",
        "biz_id",
    ] {
        assert_eq!(
            mem.get(field),
            pg.get(field),
            "field `{field}` drifted between memory and postgres"
        );
    }
}

// ---------------------------------------------------------------------------
// RED (A-1): memory populates these, postgres hardcodes them empty. Written to
// assert the CORRECT (post-fix) behaviour, so it fails today and turns green
// when Phase 3.1 uplifts `DocProcessingStatus::from_json` into core.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn doc_status_agrees_on_a1_fields() {
    let mem = serde_json::to_value(memory_status().await).unwrap();
    let pg = serde_json::to_value(pg_status().await).unwrap();

    for field in ["chunks_list", "error_msg", "metadata", "multimodal_processed"] {
        assert_eq!(
            mem.get(field),
            pg.get(field),
            "A-1: field `{field}` drifts (memory populated, postgres empty)"
        );
    }
}

// ---------------------------------------------------------------------------
// GREEN (was RED, A-6 → fixed by Phase 3.9 / BC-1): the `*` path and the BFS
// path of get_knowledge_graph used to disagree on is_truncated at the
// nodes.len() == max_nodes boundary (`>` vs `>=`). Two code paths in the SAME
// backend contradicting each other is a bug regardless of which semantics
// product picks.
//
// Product decision (BC-1): unify on `>=` — a graph that exactly fills the
// budget IS reported truncated. That was already the semantics of 4 of the 6
// sites (postgres CTE, memory BFS, neo4j `get_full_graph`/`get_sub_graph`), so
// only the two `*` paths changed.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn knowledge_graph_truncation_boundary_is_self_consistent() {
    const N: usize = 5;
    let graph = InMemoryGraphStorage::new("graph", WS);
    for i in 0..N {
        graph
            .upsert_node(&format!("node{i}"), JsonMap::new(), BIZ)
            .await
            .unwrap();
    }

    // Both paths see exactly N visible nodes against a budget of N.
    let star = graph.get_knowledge_graph(Some("*"), 0, N, BIZ).await.unwrap();
    let bfs = graph
        .get_knowledge_graph(Some("node"), 0, N, BIZ)
        .await
        .unwrap();

    assert_eq!(star.nodes.len(), N, "star path should return all N nodes");
    assert_eq!(bfs.nodes.len(), N, "bfs path should return all N nodes");
    assert_eq!(
        star.is_truncated, bfs.is_truncated,
        "A-6: is_truncated must not disagree between two paths of the same backend"
    );
    // Pin the chosen semantics explicitly: self-consistency alone would also
    // pass if both paths were flipped back to `>`.
    assert!(
        star.is_truncated,
        "BC-1: nodes.len() == max_nodes must report truncated (`>=`, not `>`)"
    );
    assert!(
        bfs.is_truncated,
        "BC-1: nodes.len() == max_nodes must report truncated (`>=`, not `>`)"
    );

    // One node of headroom is NOT truncated — `>=` must not degrade into
    // "always true".
    let roomy = graph
        .get_knowledge_graph(Some("*"), 0, N + 1, BIZ)
        .await
        .unwrap();
    assert_eq!(roomy.nodes.len(), N);
    assert!(
        !roomy.is_truncated,
        "BC-1: nodes.len() < max_nodes must not report truncated"
    );
}

/// BC-1 across backends: the Postgres `*` path must land on the same boundary
/// verdict as the in-memory one, since callers cannot tell which backend they
/// hit. Fed N canned node rows against a budget of N.
#[tokio::test]
async fn knowledge_graph_truncation_boundary_agrees_across_backends() {
    const N: usize = 5;
    let rows: Vec<JsonMap> = (0..N)
        .map(|i| as_map(json!({ "id": format!("node{i}") })))
        .collect();
    let pg = PgGraphStorage::with_options(
        "graph",
        WS,
        JsonMap::new(),
        None,
        ArcDb(Arc::new(MockPgDb::with_rows(rows))),
    )
    .unwrap();

    let star = pg.get_knowledge_graph(Some("*"), 0, N, BIZ).await.unwrap();
    assert_eq!(star.nodes.len(), N, "postgres star path returns the N rows");
    assert!(
        star.is_truncated,
        "BC-1: postgres `*` path must use `>=` like the memory backend"
    );

    let roomy = pg
        .get_knowledge_graph(Some("*"), 0, N + 1, BIZ)
        .await
        .unwrap();
    assert!(
        !roomy.is_truncated,
        "BC-1: postgres `*` path must not report truncated under budget"
    );
}
