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

// ---------------------------------------------------------------------------
// BC-13 — `get_{nodes,edges}_by_source_ids`, the server-side half of document
// deletion.
//
// Every one of these invariants corrupts data *silently* when broken, which is
// why they are pinned rather than left to the backend implementations:
//
//   * membership is element equality, never substring — `chunk_1` matching
//     `chunk_10` deletes unrelated graph data;
//   * both stored shapes match — the canonical array and the legacy
//     `<SEP>`/`,`-joined string, which survives until the backfill has run (and
//     reappears whenever an older pod writes during a rolling deploy). Missing
//     one shape leaks orphaned nodes forever;
//   * an empty chunk list selects nothing, never everything — the opposite would
//     be worse than the full-graph pull these methods replace;
//   * a backend override must agree with the trait default, because every
//     backend that does not override (and every test double) falls through to it.
// ---------------------------------------------------------------------------

fn id_list(ids: &[&str]) -> Vec<String> {
    ids.iter().map(|s| (*s).to_string()).collect()
}

/// Node/edge ids a substring match would confuse with each other, plus one node
/// and one edge per stored shape and one unrelated pair.
async fn seed_source_id_graph(graph: &InMemoryGraphStorage) {
    for (id, source_id) in [
        // `chunk_1` is a prefix of `chunk_10` and `chunk_100`: only element
        // equality keeps them apart.
        ("prefix_node", json!(["chunk_10", "chunk_100"])),
        ("array_node", json!(["chunk_1", "chunk_2"])),
        // Legacy rows, both delimiters.
        ("sep_node", json!("chunk_1<SEP>chunk_9")),
        ("comma_node", json!("chunk_1,chunk_8")),
        ("unrelated_node", json!(["chunk_77"])),
    ] {
        let mut props = JsonMap::new();
        props.insert("entity_type".to_string(), json!("Person"));
        props.insert("source_id".to_string(), source_id);
        graph.upsert_node(id, props, BIZ).await.unwrap();
    }
    for (src, tgt, source_id) in [
        ("array_node", "prefix_node", json!(["chunk_1"])),
        ("sep_node", "comma_node", json!("chunk_1<SEP>chunk_3")),
        ("unrelated_node", "array_node", json!(["chunk_77"])),
    ] {
        let mut props = JsonMap::new();
        props.insert("keywords".to_string(), json!("knows"));
        props.insert("source_id".to_string(), source_id);
        graph.upsert_edge(src, tgt, props, BIZ).await.unwrap();
    }
}

fn node_ids(nodes: &[JsonMap]) -> Vec<String> {
    let mut ids: Vec<String> = nodes
        .iter()
        .map(|n| {
            n.get("id")
                .and_then(|v| v.as_str())
                .expect("injected id")
                .to_string()
        })
        .collect();
    ids.sort();
    ids
}

fn edge_keys(edges: &[JsonMap]) -> Vec<(String, String)> {
    let mut keys: Vec<(String, String)> = edges
        .iter()
        .map(|e| {
            (
                e.get("source").and_then(|v| v.as_str()).expect("source").to_string(),
                e.get("target").and_then(|v| v.as_str()).expect("target").to_string(),
            )
        })
        .collect();
    keys.sort();
    keys
}

/// The correctness goal of the whole array refactor.
#[tokio::test]
async fn source_id_filter_matches_whole_elements_only() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    seed_source_id_graph(&graph).await;

    let nodes = graph
        .get_nodes_by_source_ids(&id_list(&["chunk_1"]), BIZ)
        .await
        .unwrap();
    assert_eq!(
        node_ids(&nodes),
        vec![
            "array_node".to_string(),
            "comma_node".to_string(),
            "sep_node".to_string()
        ],
        "chunk_1 must not match chunk_10 / chunk_100"
    );

    // And the near-miss id does reach its own node.
    let tens = graph
        .get_nodes_by_source_ids(&id_list(&["chunk_10"]), BIZ)
        .await
        .unwrap();
    assert_eq!(node_ids(&tens), vec!["prefix_node".to_string()]);
}

/// Pre-backfill rows stay visible, otherwise cleanup leaks them as orphans.
#[tokio::test]
async fn source_id_filter_matches_legacy_joined_strings() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    seed_source_id_graph(&graph).await;

    // `<SEP>`-joined node and edge.
    assert_eq!(
        node_ids(
            &graph
                .get_nodes_by_source_ids(&id_list(&["chunk_9"]), BIZ)
                .await
                .unwrap()
        ),
        vec!["sep_node".to_string()]
    );
    // Comma-joined node.
    assert_eq!(
        node_ids(
            &graph
                .get_nodes_by_source_ids(&id_list(&["chunk_8"]), BIZ)
                .await
                .unwrap()
        ),
        vec!["comma_node".to_string()]
    );
    // Legacy edge, matched through its own `source_id`.
    assert_eq!(
        edge_keys(
            &graph
                .get_edges_by_source_ids(&id_list(&["chunk_3"]), BIZ)
                .await
                .unwrap()
        ),
        vec![("sep_node".to_string(), "comma_node".to_string())]
    );
}

/// Edges follow the same rules as nodes.
#[tokio::test]
async fn source_id_filter_on_edges_matches_whole_elements_only() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    seed_source_id_graph(&graph).await;

    // The array edge and the legacy `<SEP>` edge both reference `chunk_1`; the
    // `chunk_77` edge must not, and no edge here may be reached through the
    // `chunk_10`/`chunk_100` prefix trap.
    assert_eq!(
        edge_keys(
            &graph
                .get_edges_by_source_ids(&id_list(&["chunk_1"]), BIZ)
                .await
                .unwrap()
        ),
        vec![
            ("array_node".to_string(), "prefix_node".to_string()),
            ("sep_node".to_string(), "comma_node".to_string())
        ]
    );
}

#[tokio::test]
async fn source_id_filter_on_empty_chunk_list_selects_nothing() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    seed_source_id_graph(&graph).await;

    // Not "everything": an empty question has an empty answer.
    assert!(graph
        .get_nodes_by_source_ids(&[], BIZ)
        .await
        .unwrap()
        .is_empty());
    assert!(graph
        .get_edges_by_source_ids(&[], BIZ)
        .await
        .unwrap()
        .is_empty());
    // Sanity: the seed really is non-empty, so the assertions above are not
    // passing on an empty graph.
    assert_eq!(5, graph.get_all_nodes(BIZ).await.unwrap().len());
    assert_eq!(3, graph.get_all_edges(BIZ).await.unwrap().len());
}

/// An override and the trait default are two implementations of one contract;
/// a backend that has not overridden yet must give the same answers as one that
/// has, or callers see backend-dependent cleanup results.
#[tokio::test]
async fn source_id_filter_override_agrees_with_the_trait_default() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    seed_source_id_graph(&graph).await;

    for chunks in [
        id_list(&["chunk_1"]),
        id_list(&["chunk_9", "chunk_8"]),
        id_list(&["chunk_10"]),
        id_list(&["chunk_77"]),
        id_list(&["no-such-chunk"]),
        Vec::new(),
    ] {
        let wanted: std::collections::HashSet<String> = chunks.iter().cloned().collect();
        // The default, spelled out: full pull + `source_ids_intersect`.
        let expected_nodes: Vec<JsonMap> = graph
            .get_all_nodes(BIZ)
            .await
            .unwrap()
            .into_iter()
            .filter(|n| {
                !chunks.is_empty()
                    && easyrag_core::model::source_ids_intersect(n, &wanted)
            })
            .collect();
        let expected_edges: Vec<JsonMap> = graph
            .get_all_edges(BIZ)
            .await
            .unwrap()
            .into_iter()
            .filter(|e| {
                !chunks.is_empty()
                    && easyrag_core::model::source_ids_intersect(e, &wanted)
            })
            .collect();

        let mut got_nodes = graph
            .get_nodes_by_source_ids(&chunks, BIZ)
            .await
            .unwrap();
        let mut got_edges = graph
            .get_edges_by_source_ids(&chunks, BIZ)
            .await
            .unwrap();
        // Order is not part of the contract (a server-side filter cannot promise
        // the in-memory map's iteration order), the set and the shape are.
        got_nodes.sort_by_key(|n| n.get("id").and_then(|v| v.as_str()).map(str::to_string));
        let mut expected_nodes = expected_nodes;
        expected_nodes
            .sort_by_key(|n| n.get("id").and_then(|v| v.as_str()).map(str::to_string));
        got_edges.sort_by_key(|e| {
            (
                e.get("source").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                e.get("target").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            )
        });
        let mut expected_edges = expected_edges;
        expected_edges.sort_by_key(|e| {
            (
                e.get("source").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
                e.get("target").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            )
        });

        assert_eq!(expected_nodes, got_nodes, "nodes disagree for {chunks:?}");
        assert_eq!(expected_edges, got_edges, "edges disagree for {chunks:?}");
    }
}

/// Scope widening (BC-12/A-5) applies to the new methods exactly as it does to
/// `get_all_nodes`: an empty `biz_id` spans every biz line of the workspace.
#[tokio::test]
async fn source_id_filter_honours_the_biz_wildcard() {
    let graph = InMemoryGraphStorage::new("graph", WS);
    graph
        .upsert_node(
            "other_biz",
            as_map(json!({"source_id": ["chunk_1"]})),
            "line-b",
        )
        .await
        .unwrap();
    graph
        .upsert_node("this_biz", as_map(json!({"source_id": ["chunk_1"]})), BIZ)
        .await
        .unwrap();

    let chunks = id_list(&["chunk_1"]);
    assert_eq!(
        node_ids(&graph.get_nodes_by_source_ids(&chunks, BIZ).await.unwrap()),
        vec!["this_biz".to_string()],
        "an explicit biz_id stays exact"
    );
    assert_eq!(
        node_ids(&graph.get_nodes_by_source_ids(&chunks, "").await.unwrap()),
        vec!["other_biz".to_string(), "this_biz".to_string()],
        "an empty biz_id widens to the whole workspace"
    );
}

/// Postgres side: the predicate is hand-concatenated SQL, so both arms are
/// asserted textually (the mock returns canned rows regardless of the WHERE),
/// and the result mapping is asserted against `get_all_nodes`/`get_all_edges`
/// fed the same rows.
#[tokio::test]
async fn postgres_source_id_filter_query_and_shape() {
    // Each block needs its own mock so `last_sql` can be read in isolation.
    fn pg_with(rows: Vec<JsonMap>) -> (PgGraphStorage<ArcDb>, Arc<MockPgDb>) {
        let mock = Arc::new(MockPgDb::with_rows(rows));
        let pg = PgGraphStorage::with_options(
            "graph",
            WS,
            JsonMap::new(),
            None,
            ArcDb(mock.clone()),
        )
        .unwrap();
        (pg, mock)
    }

    fn node_rows() -> Vec<JsonMap> {
        vec![as_map(
            json!({"id": "A", "properties": {"source_id": ["chunk_1"], "entity_type": "Person"}}),
        )]
    }

    let (pg, mock) = pg_with(node_rows());
    let chunks = id_list(&["chunk_1"]);
    let by_source = pg.get_nodes_by_source_ids(&chunks, BIZ).await.unwrap();
    let sql = mock.last_sql.lock().clone();

    // Arm 1: the canonical array, element equality, GIN-servable.
    assert!(
        sql.contains("(properties -> 'source_id') ?| ARRAY['chunk_1']::text[]"),
        "array arm missing: {sql}"
    );
    // Arm 2: the legacy joined string. `?|` is false for a jsonb string scalar,
    // so without this arm pre-backfill rows would never be cleaned up.
    assert!(
        sql.contains("jsonb_typeof(properties -> 'source_id') = 'string'"),
        "legacy guard missing: {sql}"
    );
    assert!(
        sql.contains("string_to_array(replace(properties ->> 'source_id', '<SEP>', ','), ',') \
                      && ARRAY['chunk_1']::text[]"),
        "legacy arm must split on both delimiters and use element-wise overlap: {sql}"
    );
    // No substring matching anywhere.
    assert!(
        !sql.contains("LIKE") && !sql.contains("POSITION"),
        "substring matching would let chunk_1 hit chunk_10: {sql}"
    );

    // Same shape as the full pull the caller used to do.
    let all = pg.get_all_nodes(BIZ).await.unwrap();
    assert_eq!(all, by_source, "node shape must match get_all_nodes");
    assert_eq!(Some("A"), by_source[0].get("id").and_then(|v| v.as_str()));

    // Edges: injected `source`/`target`, same as `get_all_edges`.
    let edge_rows = vec![as_map(json!({
        "source": "A",
        "target": "B",
        "properties": {"source_id": "chunk_1<SEP>chunk_2"},
    }))];
    let (pg, mock) = pg_with(edge_rows);
    let by_source = pg.get_edges_by_source_ids(&chunks, BIZ).await.unwrap();
    let all = pg.get_all_edges(BIZ).await.unwrap();
    assert_eq!(all, by_source, "edge shape must match get_all_edges");
    assert_eq!(Some("A"), by_source[0].get("source").and_then(|v| v.as_str()));
    assert_eq!(Some("B"), by_source[0].get("target").and_then(|v| v.as_str()));
    let sql = mock.last_sql.lock().clone();
    assert!(
        sql.contains("SELECT source, target, properties FROM graph_edges_graph"),
        "edge query must read the endpoint columns: {sql}"
    );

    // Empty chunk list must not reach the database at all.
    let (pg, mock) = pg_with(node_rows());
    assert!(pg.get_nodes_by_source_ids(&[], BIZ).await.unwrap().is_empty());
    assert!(pg.get_edges_by_source_ids(&[], BIZ).await.unwrap().is_empty());
    assert!(
        mock.last_sql.lock().is_empty(),
        "an empty chunk list must not degrade into a full scan"
    );
}

