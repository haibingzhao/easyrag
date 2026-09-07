//! Byte-level golden guardrail for `extraction.rs` (Phase 0 discipline applied
//! to 5g).
//!
//! `extraction.rs` had **zero** test coverage before this file: no unit tests,
//! no integration tests, and `tests/ingest_pipeline.rs` only asserts that
//! ingestion as a whole succeeds. Nothing pinned the rendered extraction
//! prompts, the per-mode response format, either parser's field defaults, or
//! the exact property map handed to `upsert_node`/`upsert_edge`. A refactor
//! that swapped the JSON and tuple templates, dropped a prompt variable, or
//! reordered a graph property would have passed everything.
//!
//! Snapshots land in `<workspace>/testdata/golden/extraction/`.
//!
//! ## Why `serde_json` key order is snapshotted
//!
//! The workspace enables `serde_json`'s `preserve_order` feature, so `JsonMap`
//! is an `IndexMap` and the order properties are inserted in is observable —
//! both in the rendered JSON and in anything that round-trips it. Every graph
//! payload snapshot therefore prints the key list explicitly, ahead of the
//! pretty-printed object.
//!
//! Reading the payloads back out of `InMemoryGraphStorage` would *not* do: its
//! `upsert_node` does `entry.extend(node_data)`, so on an existing node the
//! stored order comes from the pre-existing entry and a reordering inside the
//! merge branch would be invisible. `RecordingGraph` captures the map at the
//! call boundary instead.
//!
//! ## The tuple-mode system prompt is not deterministic
//!
//! `format_prompt` iterates a `HashMap`, whose order is randomised per
//! process. In tuple mode the value substituted for `{examples}`
//! (`ENTITY_EXTRACTION_EXAMPLES`) itself contains `{tuple_delimiter}` and
//! `{completion_delimiter}`, so the rendered system prompt depends on whether
//! `{examples}` happens to be replaced before them:
//!
//! * `{examples}` first  -> the delimiters inside it get substituted (clean)
//! * `{examples}` last   -> both leak into the prompt as literal braces
//! * `{examples}` middle -> exactly one leaks
//!
//! That is four distinct renderings, three of which send unsubstituted
//! placeholder text to the model. It is a real pre-existing defect, but it is
//! a *behaviour* defect, so per the plan's first iron law it is characterised
//! here and left alone by the 5g refactor rather than fixed inside it.
//! `tuple_system_prompt_is_one_of_four_order_dependent_variants` pins the
//! exact set of four by content hash — deterministic, because all four
//! orderings are reachable from any insertion order and `DefaultHasher` has
//! fixed keys. That test renders the template with its own variables, so a
//! twin (`tuple_system_prompt_from_the_production_path_has_the_same_four_variants`)
//! drives the real extractor and must produce the same four hashes; without it
//! a variable dropped from `ExtractionMode::system_vars` would go unnoticed.
//! JSON mode is unaffected: `ENTITY_EXTRACTION_JSON_EXAMPLES` contains no
//! placeholders, so its system prompt renders one way.
//!
//! Tuple mode is also unreachable from every production assembly site
//! (`main.rs`, `workspace.rs`, `easy_rag.rs`, `ingest.rs` all take
//! `EntityExtractorConfig::default()`, which sets `use_json_extraction: true`,
//! and nothing exposes an override). It stays public API, so it stays pinned.

use async_trait::async_trait;
use easyrag_core::constants::GRAPH_FIELD_SEP;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService, MessageRole};
use easyrag_core::model::KnowledgeGraph;
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use easyrag_pipeline::extraction::{EntityExtractor, EntityExtractorConfig, ExtractionResult};
use easyrag_storage::memory::{InMemoryKVStorage, InMemoryVectorStorage};
use futures::stream::BoxStream;
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Passed as the biz_id filter argument by every merge test.
const BIZ: &str = "default";

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

/// Everything the extractor sends to the LLM, captured at the call boundary.
#[derive(Clone)]
struct Captured {
    prompt: String,
    system_prompt: Option<String>,
    response_format: String,
    temperature: f32,
    max_tokens: Option<u32>,
    history: Vec<(MessageRole, String)>,
}

/// Scripted LLM that records every request and replays queued responses.
struct CaptureLlm {
    captured: Mutex<Vec<Captured>>,
    responses: Mutex<VecDeque<String>>,
    /// Returned once the queue is drained.
    default: String,
}

impl CaptureLlm {
    fn new(responses: Vec<&str>) -> Arc<Self> {
        Arc::new(Self {
            captured: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into_iter().map(|s| s.to_string()).collect()),
            default: String::new(),
        })
    }

    /// Like [`CaptureLlm::new`] but with a fallback used once the queue drains,
    /// so a loop that runs more rounds than scripted still terminates.
    fn with_default(responses: Vec<&str>, default: &str) -> Arc<Self> {
        Arc::new(Self {
            captured: Mutex::new(Vec::new()),
            responses: Mutex::new(responses.into_iter().map(|s| s.to_string()).collect()),
            default: default.to_string(),
        })
    }

    fn requests(&self) -> Vec<Captured> {
        self.captured.lock().unwrap().clone()
    }
}

#[async_trait]
impl LlmService for CaptureLlm {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        self.captured.lock().unwrap().push(Captured {
            prompt: request.prompt.clone(),
            system_prompt: request.system_prompt.clone(),
            response_format: format!("{:?}", request.response_format),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            history: request
                .history
                .iter()
                .map(|m| (m.role, m.content.clone()))
                .collect(),
        });
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

/// Graph storage that records the exact property map handed to each upsert.
///
/// Only the four methods `merge_to_graph` calls are implemented; the rest of
/// the trait is unreachable from it, so they fail loudly rather than silently
/// returning empty data that a snapshot would happily record.
struct RecordingGraph {
    global: JsonMap,
    nodes: Mutex<HashMap<String, JsonMap>>,
    edges: Mutex<HashMap<(String, String), JsonMap>>,
    node_writes: Mutex<Vec<(String, JsonMap)>>,
    edge_writes: Mutex<Vec<(String, String, JsonMap)>>,
}

impl RecordingGraph {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            global: JsonMap::new(),
            nodes: Mutex::new(HashMap::new()),
            edges: Mutex::new(HashMap::new()),
            node_writes: Mutex::new(Vec::new()),
            edge_writes: Mutex::new(Vec::new()),
        })
    }

    /// Seed an existing node, so the merge branch runs instead of the create
    /// branch. `props` is inserted in the order given, which deliberately does
    /// not have to match the order the extractor writes.
    fn with_node(self: &Arc<Self>, name: &str, props: Vec<(&str, JsonValue)>) -> Arc<Self> {
        let mut map = JsonMap::new();
        for (k, v) in props {
            map.insert(k.to_string(), v);
        }
        self.nodes.lock().unwrap().insert(name.to_string(), map);
        Arc::clone(self)
    }

    fn with_edge(
        self: &Arc<Self>,
        src: &str,
        tgt: &str,
        props: Vec<(&str, JsonValue)>,
    ) -> Arc<Self> {
        let mut map = JsonMap::new();
        for (k, v) in props {
            map.insert(k.to_string(), v);
        }
        self.edges
            .lock()
            .unwrap()
            .insert((src.to_string(), tgt.to_string()), map);
        Arc::clone(self)
    }

    fn node_writes(&self) -> Vec<(String, JsonMap)> {
        self.node_writes.lock().unwrap().clone()
    }

    fn edge_writes(&self) -> Vec<(String, String, JsonMap)> {
        self.edge_writes.lock().unwrap().clone()
    }
}

#[async_trait]
impl StorageNameSpace for RecordingGraph {
    fn namespace(&self) -> &str {
        "extraction-golden"
    }

    fn workspace(&self) -> &str {
        "extraction-golden"
    }

    fn global_config(&self) -> &JsonMap {
        &self.global
    }

    async fn drop(&self, _biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        unimplemented!("merge_to_graph does not call drop")
    }
}

#[async_trait]
impl GraphStorage for RecordingGraph {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        None
    }

    async fn get_node(&self, node_id: &str, _biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self.nodes.lock().unwrap().get(node_id).cloned())
    }

    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        _biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .edges
            .lock()
            .unwrap()
            .get(&(source_node_id.to_string(), target_node_id.to_string()))
            .cloned())
    }

    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        _biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.node_writes
            .lock()
            .unwrap()
            .push((node_id.to_string(), node_data));
        Ok(())
    }

    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        _biz_id: &str,
    ) -> easyrag_core::Result<()> {
        self.edge_writes.lock().unwrap().push((
            source_node_id.to_string(),
            target_node_id.to_string(),
            edge_data,
        ));
        Ok(())
    }

    async fn has_node(&self, _node_id: &str, _biz_id: &str) -> easyrag_core::Result<bool> {
        unimplemented!("merge_to_graph does not call has_node")
    }

    async fn has_edge(
        &self,
        _s: &str,
        _t: &str,
        _biz_id: &str,
    ) -> easyrag_core::Result<bool> {
        unimplemented!("merge_to_graph does not call has_edge")
    }

    async fn node_degree(&self, _node_id: &str, _biz_id: &str) -> easyrag_core::Result<usize> {
        unimplemented!("merge_to_graph does not call node_degree")
    }

    async fn edge_degree(&self, _s: &str, _t: &str, _biz_id: &str) -> easyrag_core::Result<usize> {
        unimplemented!("merge_to_graph does not call edge_degree")
    }

    async fn get_node_edges(
        &self,
        _s: &str,
        _biz_id: &str,
    ) -> easyrag_core::Result<Option<Vec<(String, String)>>> {
        unimplemented!("merge_to_graph does not call get_node_edges")
    }

    async fn delete_node(&self, _node_id: &str, _biz_id: &str) -> easyrag_core::Result<()> {
        unimplemented!("merge_to_graph does not call delete_node")
    }

    async fn remove_nodes(&self, _ids: &[String], _biz_id: &str) -> easyrag_core::Result<()> {
        unimplemented!("merge_to_graph does not call remove_nodes")
    }

    async fn remove_edges(
        &self,
        _edges: &[(String, String)],
        _biz_id: &str,
    ) -> easyrag_core::Result<()> {
        unimplemented!("merge_to_graph does not call remove_edges")
    }

    async fn get_all_labels(&self, _biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        unimplemented!("merge_to_graph does not call get_all_labels")
    }

    async fn get_knowledge_graph(
        &self,
        _label: Option<&str>,
        _depth: usize,
        _max: usize,
        _biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        unimplemented!("merge_to_graph does not call get_knowledge_graph")
    }

    async fn get_all_nodes(&self, _biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        unimplemented!("merge_to_graph does not call get_all_nodes")
    }

    async fn get_all_edges(&self, _biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        unimplemented!("merge_to_graph does not call get_all_edges")
    }

    async fn get_popular_labels(
        &self,
        _limit: usize,
        _biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        unimplemented!("merge_to_graph does not call get_popular_labels")
    }

    async fn search_labels(
        &self,
        _q: &str,
        _limit: usize,
        _biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        unimplemented!("merge_to_graph does not call search_labels")
    }
}

// ---------------------------------------------------------------------
// Snapshot helpers
// ---------------------------------------------------------------------

fn golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-pipeline");
    root.join("testdata").join("golden").join("extraction")
}

fn snap(name: &str, text: &str) {
    insta::with_settings!({
        snapshot_path => golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, text);
    });
}

/// Key list first, then the object. The key list is what makes an insertion
/// reordering a one-line diff instead of something you have to spot inside a
/// JSON blob.
fn render_map(map: &JsonMap) -> String {
    let keys: Vec<&str> = map.keys().map(|k| k.as_str()).collect();
    let pretty =
        serde_json::to_string_pretty(&JsonValue::Object(map.clone())).expect("map serialises");
    format!("keys=[{}]\n{}\n", keys.join(", "), pretty)
}

fn render_writes(graph: &RecordingGraph) -> String {
    let mut out = String::new();
    let nodes = graph.node_writes();
    out.push_str(&format!("node_writes={}\n", nodes.len()));
    for (id, map) in &nodes {
        out.push_str(&format!("node id={:?}\n{}\n", id, render_map(map)));
    }
    let edges = graph.edge_writes();
    out.push_str(&format!("edge_writes={}\n", edges.len()));
    for (src, tgt, map) in &edges {
        out.push_str(&format!("edge {:?} -> {:?}\n{}\n", src, tgt, render_map(map)));
    }
    out
}

fn render_result(result: &ExtractionResult) -> String {
    let mut out = format!("entities={}\n", result.entities.len());
    for e in &result.entities {
        out.push_str(&format!(
            "  entity name={:?} type={:?} desc={:?}\n",
            e.entity_name, e.entity_type, e.entity_description
        ));
    }
    out.push_str(&format!("relationships={}\n", result.relationships.len()));
    for r in &result.relationships {
        out.push_str(&format!(
            "  rel src={:?} tgt={:?} kw={:?} desc={:?}\n",
            r.source_entity, r.target_entity, r.relationship_keywords, r.relationship_description
        ));
    }
    out
}

/// Request metadata without the prompt bodies, which are snapshotted on their
/// own. Lengths stand in for content so a shape snapshot stays readable.
///
/// The system prompt contributes only its *presence*, never its length or
/// hash: in tuple mode it renders one of four ways (see the module docs), so
/// any content-derived field here would make the shape snapshot flaky. Its
/// bytes are pinned instead by `json_system_prompt` (deterministic) and
/// `tuple_system_prompt_variants` (the four-variant characterisation).
fn render_requests(reqs: &[Captured]) -> String {
    let mut out = format!("requests={}\n", reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        let history: Vec<String> = r
            .history
            .iter()
            .map(|(role, content)| format!("{:?}:{}", role, content.len()))
            .collect();
        out.push_str(&format!(
            "[{i}] response_format={} temperature={} max_tokens={:?} system_prompt={} prompt_len={} history=[{}]\n",
            r.response_format,
            r.temperature,
            r.max_tokens,
            if r.system_prompt.is_some() { "present" } else { "absent" },
            r.prompt.len(),
            history.join(", "),
        ));
    }
    out
}

fn content_hash(text: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// Sentinel values, so every substitution is visible in the snapshot instead
/// of blending into prose that would look the same unsubstituted.
fn cfg(json: bool) -> EntityExtractorConfig {
    EntityExtractorConfig {
        language: "Klingon".to_string(),
        entity_types_guidance: "GUIDANCE-SENTINEL".to_string(),
        use_json_extraction: json,
        max_total_records: 41,
        max_entity_records: 23,
        gleaning_rounds: 0,
        force_llm_summary_on_merge: easyrag_core::constants::DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
        summary_max_tokens: easyrag_core::constants::DEFAULT_SUMMARY_MAX_TOKENS,
        max_source_ids_per_entity: 30,
        // Large enough that the gleaning token guard never trips by accident;
        // `gleaning_skipped_when_token_budget_exceeded` lowers it on purpose.
        max_extract_input_tokens: 100_000_000,
    }
}

const CHUNK: &str = "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine.";
const HEADING: &str = "Chapter 9 > Section 9.3";

/// One entity and one relationship, with a trailing delimiter row that the
/// tuple parser must drop and a short row it must ignore.
const TUPLE_RESPONSE: &str = "\
entity<|#|>Ada Lovelace<|#|>Person<|#|>Wrote notes on the Analytical Engine
relation<|#|>Ada Lovelace<|#|>Charles Babbage<|#|>collaborated with<|#|>Worked on the Analytical Engine
entity<|#|>OnlyThreeFields
<|COMPLETE|>
";

const JSON_RESPONSE: &str = r#"{
  "entities": [
    {"name": "Ada Lovelace", "type": "Person", "description": "Wrote notes on the Analytical Engine"},
    {"name": "  Charles Babbage  ", "type": null, "description": "  Designed the Analytical Engine  "},
    {"name": "   ", "type": "Person", "description": "skipped: blank name"},
    {"name": "NoType", "type": "", "description": "explicit empty type is kept"}
  ],
  "relationships": [
    {"source": "Ada Lovelace", "target": "Charles Babbage", "keywords": "collaborated with", "description": "Worked on the Analytical Engine"},
    {"source": "", "target": "Charles Babbage", "keywords": "x", "description": "skipped: blank source"}
  ]
}"#;

fn extractor(
    llm: Arc<dyn LlmService>,
    graph: Arc<dyn GraphStorage>,
    config: EntityExtractorConfig,
) -> EntityExtractor {
    EntityExtractor::new(
        llm,
        graph,
        Arc::new(InMemoryVectorStorage::new("g", "g", test_embedding_func())),
        Arc::new(InMemoryKVStorage::new("g", "g")),
        config,
    )
}

// ---------------------------------------------------------------------
// Group A: rendered prompts and request shape
// ---------------------------------------------------------------------

#[tokio::test]
async fn json_mode_first_round_prompt_and_request() {
    let llm = CaptureLlm::new(vec![JSON_RESPONSE]);
    let graph = RecordingGraph::new();
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        graph,
        cfg(true),
    );

    ex.extract(CHUNK, "chunk-1", "").await.expect("extract");

    let reqs = llm.requests();
    snap("json_system_prompt", reqs[0].system_prompt.as_deref().unwrap());
    snap("json_user_prompt_no_heading", &reqs[0].prompt);
    snap("json_request_shape", &render_requests(&reqs));
}

#[tokio::test]
async fn tuple_mode_first_round_prompt_and_request() {
    let llm = CaptureLlm::new(vec![TUPLE_RESPONSE]);
    let graph = RecordingGraph::new();
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        graph,
        cfg(false),
    );

    ex.extract(CHUNK, "chunk-1", "").await.expect("extract");

    let reqs = llm.requests();
    snap("tuple_user_prompt_no_heading", &reqs[0].prompt);
    snap("tuple_request_shape", &render_requests(&reqs));
}

/// The defect characterisation described in the module docs. Four renderings,
/// pinned by content hash so the assertion is deterministic while still being
/// exact about content.
#[test]
fn tuple_system_prompt_is_one_of_four_order_dependent_variants() {
    use easyrag_core::prompt::{format_prompt, ENTITY_EXTRACTION_SYSTEM_PROMPT};

    let mut variants: HashSet<String> = HashSet::new();
    for _ in 0..4000 {
        let mut vars = HashMap::new();
        vars.insert(
            "entity_types_guidance".to_string(),
            "GUIDANCE-SENTINEL".to_string(),
        );
        vars.insert(
            "examples".to_string(),
            easyrag_core::prompt::ENTITY_EXTRACTION_EXAMPLES.join("\n"),
        );
        vars.insert("max_total_records".to_string(), "41".to_string());
        vars.insert("max_entity_records".to_string(), "23".to_string());
        vars.insert("language".to_string(), "Klingon".to_string());
        vars.insert(
            "tuple_delimiter".to_string(),
            easyrag_core::prompt::DEFAULT_TUPLE_DELIMITER.to_string(),
        );
        vars.insert(
            "completion_delimiter".to_string(),
            easyrag_core::prompt::DEFAULT_COMPLETION_DELIMITER.to_string(),
        );
        variants.insert(format_prompt(&ENTITY_EXTRACTION_SYSTEM_PROMPT, &vars));
    }

    let mut hashes: Vec<String> = variants.iter().map(|v| content_hash(v)).collect();
    hashes.sort();
    // Which placeholders survived, per variant, sorted alongside the hashes so
    // a change in either shows up in the same snapshot.
    let mut leaks: Vec<String> = variants
        .iter()
        .map(|v| {
            format!(
                "tuple_delimiter_leaked={} completion_delimiter_leaked={}",
                v.contains("{tuple_delimiter}"),
                v.contains("{completion_delimiter}")
            )
        })
        .collect();
    leaks.sort();

    snap(
        "tuple_system_prompt_variants",
        &format!(
            "distinct_variants={}\nhashes={:?}\nleaks={:?}\n",
            hashes.len(),
            hashes,
            leaks
        ),
    );
}

/// The same four variants, reached through [`EntityExtractor`] instead of a
/// hand-built variable table.
///
/// The test above calls `format_prompt` directly with its own sentinels, so it
/// pins how the *template* reacts to substitution order and nothing else: it
/// would stay green if production stopped supplying a variable altogether,
/// because the variables it renders with never come from production. That gap
/// is what let a mutant drop `tuple_delimiter` from
/// `ExtractionMode::system_vars` unnoticed. This test closes it by driving the
/// real extractor and collecting the system prompt the model actually
/// receives, so a dropped key shrinks the variant set and changes every hash.
///
/// `cfg(false)` supplies exactly the sentinels the direct test hard-codes, so
/// the two snapshots must agree hash for hash — that agreement is itself the
/// evidence that the direct test still describes production.
///
/// Sampling is unavoidable (the rendering depends on `HashMap` iteration
/// order) but the projection is deterministic: all four orderings are
/// reachable from any insertion order, and `DefaultHasher` has fixed keys. The
/// explicit count assertion means under-sampling fails loudly instead of
/// quietly shrinking the snapshot.
#[tokio::test]
async fn tuple_system_prompt_from_the_production_path_has_the_same_four_variants() {
    let mut variants: HashSet<String> = HashSet::new();
    for _ in 0..4000 {
        // A fresh mock per round keeps the captured request list flat.
        let llm = CaptureLlm::new(vec![TUPLE_RESPONSE]);
        let ex = extractor(
            Arc::clone(&llm) as Arc<dyn LlmService>,
            RecordingGraph::new(),
            cfg(false),
        );
        ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
        let system = llm.requests()[0]
            .system_prompt
            .clone()
            .expect("tuple mode sends a system prompt");
        variants.insert(system);
    }

    assert_eq!(
        variants.len(),
        4,
        "all four order-dependent renderings must be reachable from the production path"
    );

    let mut hashes: Vec<String> = variants.iter().map(|v| content_hash(v)).collect();
    hashes.sort();
    let mut leaks: Vec<String> = variants
        .iter()
        .map(|v| {
            format!(
                "tuple_delimiter_leaked={} completion_delimiter_leaked={}",
                v.contains("{tuple_delimiter}"),
                v.contains("{completion_delimiter}")
            )
        })
        .collect();
    leaks.sort();

    snap(
        "tuple_system_prompt_production_variants",
        &format!(
            "distinct_variants={}\nhashes={:?}\nleaks={:?}\n",
            hashes.len(),
            hashes,
            leaks
        ),
    );
}

#[tokio::test]
async fn heading_path_block_is_injected_in_both_modes() {
    for json in [true, false] {
        let response = if json { JSON_RESPONSE } else { TUPLE_RESPONSE };
        let llm = CaptureLlm::new(vec![response]);
        let ex = extractor(
            Arc::clone(&llm) as Arc<dyn LlmService>,
            RecordingGraph::new(),
            cfg(json),
        );
        ex.extract(CHUNK, "chunk-1", HEADING)
            .await
            .expect("extract");
        let reqs = llm.requests();
        let mode = if json { "json" } else { "tuple" };
        snap(
            &format!("{mode}_user_prompt_with_heading"),
            &reqs[0].prompt,
        );
    }
}

#[tokio::test]
async fn gleaning_round_prompt_history_and_request_shape() {
    for json in [true, false] {
        let first = if json { JSON_RESPONSE } else { TUPLE_RESPONSE };
        // Second round returns nothing new, so the loop breaks after one
        // gleaning round and the request count stays at 2.
        let empty = if json {
            r#"{"entities": [], "relationships": []}"#
        } else {
            "<|COMPLETE|>"
        };
        let llm = CaptureLlm::new(vec![first, empty]);
        let mut config = cfg(json);
        config.gleaning_rounds = 2;
        let ex = extractor(
            Arc::clone(&llm) as Arc<dyn LlmService>,
            RecordingGraph::new(),
            config,
        );
        ex.extract(CHUNK, "chunk-1", "").await.expect("extract");

        let reqs = llm.requests();
        assert_eq!(reqs.len(), 2, "one first round + one gleaning round");
        let mode = if json { "json" } else { "tuple" };
        snap(&format!("{mode}_gleaning_prompt"), &reqs[1].prompt);
        snap(&format!("{mode}_gleaning_request_shape"), &render_requests(&reqs));
    }
}

#[tokio::test]
async fn gleaning_skipped_when_token_budget_exceeded() {
    let llm = CaptureLlm::new(vec![JSON_RESPONSE]);
    let mut config = cfg(true);
    config.gleaning_rounds = 3;
    // The system prompt alone is a few thousand chars, so chars/4 already
    // blows a budget of 10 and the guard breaks before the first round.
    config.max_extract_input_tokens = 10;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        config,
    );

    ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    assert_eq!(llm.requests().len(), 1, "every gleaning round must be skipped");
}

#[tokio::test]
async fn gleaning_accumulates_and_keeps_the_longer_description() {
    let second = r#"{
      "entities": [
        {"name": "ada lovelace", "type": "Person", "description": "A much longer description that must win the merge because it has more characters than the first round's."},
        {"name": "Analytical Engine", "type": "Machine", "description": "A new entity from round two."}
      ],
      "relationships": [
        {"source": "Analytical Engine", "target": "Charles Babbage", "keywords": "designed by", "description": "Round two relationship."}
      ]
    }"#;
    let third = r#"{"entities": [], "relationships": []}"#;
    let llm = CaptureLlm::new(vec![JSON_RESPONSE, second, third]);
    let mut config = cfg(true);
    config.gleaning_rounds = 3;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        config,
    );

    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    // Round three returns nothing, so the loop breaks there: 3 requests total.
    let reqs = llm.requests();
    assert_eq!(reqs.len(), 3);
    // History grows by exactly the assistant turn each round; the user turn is
    // seeded once.
    assert_eq!(reqs[1].history.len(), 2);
    assert_eq!(reqs[2].history.len(), 3);
    snap("gleaning_merged_result", &render_result(&result));
}

/// Pins the loop bound. The source comment records that Kuzu writes
/// `for (round in 1 downTo gleaningRounds)`, an empty descending range that
/// silently disables gleaning; this codebase implements the intended semantics
/// instead. A `gleaning_round()` extraction that off-by-ones the bound would
/// still pass every other test here, because they all break out of the loop
/// early on an empty round.
#[tokio::test]
async fn gleaning_runs_every_configured_round_when_none_is_empty() {
    // Always yields something new, so the "no new results" break never fires
    // and the loop is bounded only by `gleaning_rounds`.
    let always_new = r#"{"entities": [{"name": "Extra", "type": "Person", "description": "round"}], "relationships": []}"#;
    let llm = CaptureLlm::with_default(vec![JSON_RESPONSE], always_new);
    let mut config = cfg(true);
    config.gleaning_rounds = 3;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        config,
    );

    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    let reqs = llm.requests();
    assert_eq!(reqs.len(), 4, "one first round + three gleaning rounds");
    // History grows by exactly one assistant turn per round.
    assert_eq!(reqs[1].history.len(), 2);
    assert_eq!(reqs[2].history.len(), 3);
    assert_eq!(reqs[3].history.len(), 4);
    // All three gleaning rounds contributed, and `merge_gleaning_entities`
    // deduplicated the repeated "Extra" down to one entry (the first-round
    // JSON_RESPONSE yields three entities: the blank-name row is dropped).
    assert_eq!(result.entities.len(), 4, "3 first-round + 1 deduped gleaning");
}

// ---------------------------------------------------------------------
// Group B: parsers
// ---------------------------------------------------------------------

#[tokio::test]
async fn json_parser_field_defaults_and_row_filtering() {
    let llm = CaptureLlm::new(vec![JSON_RESPONSE]);
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        cfg(true),
    );
    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    snap("json_parser_result", &render_result(&result));
}

#[tokio::test]
async fn json_parser_accepts_fences_and_rejects_garbage() {
    let fenced = format!("```json\n{JSON_RESPONSE}\n```");
    let llm = CaptureLlm::new(vec![&fenced]);
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        cfg(true),
    );
    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    snap("json_parser_fenced_result", &render_result(&result));

    let llm = CaptureLlm::new(vec!["not json at all, just prose"]);
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        cfg(true),
    );
    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    snap("json_parser_garbage_result", &render_result(&result));
}

#[tokio::test]
async fn tuple_parser_row_filtering_and_case_insensitivity() {
    let response = "\
ENTITY<|#|>Upper Case Row<|#|>Person<|#|>Keyword match is case-insensitive
Relation<|#|>A<|#|>B<|#|>kw<|#|>Mixed case relation prefix
entity<|#|>Four<|#|>Person<|#|>Exactly four parts is an entity
relation<|#|>Four<|#|>Person<|#|>Only four parts is not a relation
entity<|#|>Three<|#|>Person
<|COMPLETE|>

entity<|#|>After Blank And Delimiter<|#|>Person<|#|>Blank lines are dropped
";
    let llm = CaptureLlm::new(vec![response]);
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        RecordingGraph::new(),
        cfg(false),
    );
    let result = ex.extract(CHUNK, "chunk-1", "").await.expect("extract");
    snap("tuple_parser_result", &render_result(&result));
}

// ---------------------------------------------------------------------
// Group C: graph merge payloads
// ---------------------------------------------------------------------

fn entity_result(desc: &str, entity_type: &str) -> ExtractionResult {
    ExtractionResult {
        entities: vec![easyrag_core::model::types::ExtractedEntity {
            entity_name: "Ada Lovelace".to_string(),
            entity_type: entity_type.to_string(),
            entity_description: desc.to_string(),
        }],
        relationships: Vec::new(),
    }
}

fn relation_result(desc: &str) -> ExtractionResult {
    ExtractionResult {
        entities: Vec::new(),
        relationships: vec![easyrag_core::model::types::ExtractedRelationship {
            source_entity: "Ada Lovelace".to_string(),
            target_entity: "Charles Babbage".to_string(),
            relationship_keywords: "collaborated with".to_string(),
            relationship_description: desc.to_string(),
        }],
    }
}

async fn merge(
    graph: Arc<RecordingGraph>,
    result: &ExtractionResult,
    config: EntityExtractorConfig,
) -> Arc<RecordingGraph> {
    let ex = extractor(
        CaptureLlm::new(vec![]) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(result, "chunk-7", BIZ).await.expect("merge");
    graph
}

#[tokio::test]
async fn new_entity_upsert_payload() {
    let graph = RecordingGraph::new();
    merge(
        Arc::clone(&graph),
        &entity_result("Brand new description", "Person"),
        cfg(true),
    )
    .await;
    snap("new_entity_payload", &render_writes(&graph));
}

#[tokio::test]
async fn existing_entity_upsert_payload() {
    // Seeded in a deliberately different key order from what the extractor
    // writes, so the snapshot proves the payload's order comes from the
    // extractor and not from the stored node.
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("source_id", JsonValue::String("chunk-1".to_string())),
            ("entity_type", JsonValue::String("Other".to_string())),
            ("description", JsonValue::String("An existing description".to_string())),
        ],
    );
    merge(
        Arc::clone(&graph),
        &entity_result("A new fragment", "Person"),
        cfg(true),
    )
    .await;
    snap("existing_entity_payload", &render_writes(&graph));
}

#[tokio::test]
async fn new_relation_upsert_payload() {
    let graph = RecordingGraph::new();
    merge(
        Arc::clone(&graph),
        &relation_result("Brand new relation description"),
        cfg(true),
    )
    .await;
    snap("new_relation_payload", &render_writes(&graph));
}

#[tokio::test]
async fn existing_relation_upsert_payload() {
    let graph = RecordingGraph::new().with_edge(
        "Ada Lovelace",
        "Charles Babbage",
        vec![
            ("weight", serde_json::json!(2.5_f64)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
            ("keywords", JsonValue::String("old keywords".to_string())),
            ("description", JsonValue::String("An existing relation description".to_string())),
        ],
    );
    merge(
        Arc::clone(&graph),
        &relation_result("A new relation fragment"),
        cfg(true),
    )
    .await;
    snap("existing_relation_payload", &render_writes(&graph));
}

/// Pins the *arithmetic width* of the weight bump. `existing_weight` is cast
/// to `f32`, so `existing_weight + 1.0` adds in `f32` and the widened result
/// carries the `f32` rounding error into the JSON number. A round seed value
/// like 2.5 would hide that: 2.5 and 3.5 are exact in both widths. 0.1 is not,
/// so this snapshot distinguishes `f32` addition (1.100000023841858) from
/// `f64` addition (1.1) — the exact drift a `GraphUpsert` that normalises the
/// weight field to one type would introduce.
#[tokio::test]
async fn existing_relation_weight_bump_is_f32_arithmetic() {
    let graph = RecordingGraph::new().with_edge(
        "Ada Lovelace",
        "Charles Babbage",
        vec![
            ("weight", serde_json::json!(0.1_f64)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
            ("keywords", JsonValue::String("kw".to_string())),
            ("description", JsonValue::String("d".to_string())),
        ],
    );
    merge(
        Arc::clone(&graph),
        // Neither contains the other and neither contains "d", so the
        // description really does merge and the weight is the only thing this
        // snapshot is about.
        &relation_result("fresh info"),
        cfg(true),
    )
    .await;
    snap("existing_relation_weight_precision_payload", &render_writes(&graph));
}

#[tokio::test]
async fn source_ids_are_capped_fifo() {
    let existing: Vec<String> = (1..=5).map(|i| format!("chunk-{i}")).collect();
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String("d".to_string())),
            (
                "source_id",
                JsonValue::String(existing.join(GRAPH_FIELD_SEP)),
            ),
        ],
    );
    let mut config = cfg(true);
    config.max_source_ids_per_entity = 3;
    merge(Arc::clone(&graph), &entity_result("f", "Person"), config).await;

    let writes = graph.node_writes();
    let source_id = writes[0]
        .1
        .get("source_id")
        .and_then(|v| v.as_str())
        .expect("source_id");
    // Oldest dropped, the new chunk id appended last.
    assert_eq!(
        source_id,
        format!("chunk-4{GRAPH_FIELD_SEP}chunk-5{GRAPH_FIELD_SEP}chunk-7")
    );
    snap("source_id_fifo_payload", &render_writes(&graph));
}

#[tokio::test]
async fn source_id_deduplicates_the_same_chunk() {
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String("d".to_string())),
            ("source_id", JsonValue::String("chunk-7".to_string())),
        ],
    );
    merge(Arc::clone(&graph), &entity_result("f", "Person"), cfg(true)).await;
    let writes = graph.node_writes();
    assert_eq!(
        writes[0].1.get("source_id").and_then(|v| v.as_str()),
        Some("chunk-7"),
        "an already-present chunk id must not be appended twice"
    );
}

#[tokio::test]
async fn description_merge_skips_contained_fragments_in_both_directions() {
    // Existing contains the new fragment -> unchanged.
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            (
                "description",
                JsonValue::String("A long existing description".to_string()),
            ),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    merge(
        Arc::clone(&graph),
        &entity_result("existing", "Person"),
        cfg(true),
    )
    .await;
    snap(
        "description_contained_in_existing",
        &render_writes(&graph),
    );

    // New fragment contains the existing one -> unchanged (existing kept).
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String("short".to_string())),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    merge(
        Arc::clone(&graph),
        &entity_result("a much longer description containing short", "Person"),
        cfg(true),
    )
    .await;
    snap("description_containing_existing", &render_writes(&graph));

    // Disjoint fragments -> joined with GRAPH_FIELD_SEP.
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String("first".to_string())),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    merge(Arc::clone(&graph), &entity_result("second", "Person"), cfg(true)).await;
    snap("description_disjoint_joined", &render_writes(&graph));

    // Blank existing description -> the fragment alone, no separator.
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String("   ".to_string())),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    merge(Arc::clone(&graph), &entity_result("only", "Person"), cfg(true)).await;
    snap("description_blank_existing", &render_writes(&graph));
}

#[tokio::test]
async fn entity_type_voting_matrix() {
    // (existing, new) -> voted. "Other"/"Unknown"/"entity"/"" are generic;
    // a generic existing type yields to a specific new one, everything else
    // keeps the existing type for stability.
    let cases = [
        ("Person", "Person", "equal, case-sensitively"),
        ("person", "Person", "equal ignoring case -> keeps existing spelling"),
        ("Other", "Person", "generic existing -> new wins"),
        ("", "Person", "empty existing is generic -> new wins"),
        ("Unknown", "Person", "generic existing -> new wins"),
        ("entity", "Person", "generic existing -> new wins"),
        ("Person", "Other", "generic new -> existing kept"),
        ("Person", "", "empty new is generic -> existing kept"),
        ("Person", "Organisation", "both specific -> existing kept for stability"),
    ];

    let mut out = String::new();
    for (existing, new, note) in cases {
        let graph = RecordingGraph::new().with_node(
            "Ada Lovelace",
            vec![
                ("entity_type", JsonValue::String(existing.to_string())),
                ("description", JsonValue::String("d".to_string())),
                ("source_id", JsonValue::String("chunk-1".to_string())),
            ],
        );
        merge(Arc::clone(&graph), &entity_result("f", new), cfg(true)).await;
        let voted = graph.node_writes()[0]
            .1
            .get("entity_type")
            .and_then(|v| v.as_str())
            .expect("entity_type")
            .to_string();
        out.push_str(&format!(
            "existing={existing:?} new={new:?} -> voted={voted:?}  ({note})\n"
        ));
    }
    snap("entity_type_voting_matrix", &out);
}

#[tokio::test]
async fn summary_single_pass_prompt() {
    // Two fragments and a threshold of 2 -> summarisation triggers, and the
    // estimated tokens fit the budget, so it is a single batch call.
    let existing = format!("fragment one{GRAPH_FIELD_SEP}fragment two");
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    let llm = CaptureLlm::new(vec!["  A cohesive summary.  "]);
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&entity_result("fragment three", "Person"), "chunk-7", BIZ)
        .await
        .expect("merge");

    let reqs = llm.requests();
    assert_eq!(reqs.len(), 1, "single-pass summary");
    snap("summary_single_pass_prompt", &reqs[0].prompt);
    snap("summary_single_pass_payload", &render_writes(&graph));
}

#[tokio::test]
async fn summary_map_reduce_prompts() {
    // A one-token budget forces the map-reduce path. Three fragments totalling
    // 38 chars estimate to 9 tokens, so `group_size = 3 / ((9 / 1) + 1) = 0`,
    // clamped to 1: one fragment per group, three map calls, then one reduce.
    let existing = format!("fragment one{GRAPH_FIELD_SEP}fragment two");
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    let llm = CaptureLlm::new(vec![
        "Summary of group one.",
        "Summary of group two.",
        "Summary of group three.",
        // Padded, to pin the `.trim()` the reduce branch applies.
        "  The reduced final summary.  ",
    ]);
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    config.summary_max_tokens = 1;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&entity_result("fragment three", "Person"), "chunk-7", BIZ)
        .await
        .expect("merge");

    let reqs = llm.requests();
    assert_eq!(reqs.len(), 4, "three map calls + one reduce call");
    let mut out = String::new();
    for (i, r) in reqs.iter().enumerate() {
        out.push_str(&format!("===== request {i} =====\n{}\n", r.prompt));
    }
    snap("summary_map_reduce_prompts", &out);
    snap("summary_map_reduce_payload", &render_writes(&graph));
}

/// The reduce branch has its own empty-response fallback, distinct from the
/// map branch's: a blank reduce answer joins the group summaries with a space
/// rather than returning the blank. This is the branch a 5g fold of the reduce
/// phase into `summarize_fragment_batch` would most easily lose, because the
/// batch helper has no equivalent.
#[tokio::test]
async fn summary_reduce_falls_back_to_joined_group_summaries_when_blank() {
    let existing = format!("fragment one{GRAPH_FIELD_SEP}fragment two");
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    // Three map answers, then a whitespace-only reduce answer.
    let llm = CaptureLlm::with_default(
        vec![
            "Summary of group one.",
            "Summary of group two.",
            "Summary of group three.",
        ],
        "   ",
    );
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    config.summary_max_tokens = 1;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&entity_result("fragment three", "Person"), "chunk-7", BIZ)
        .await
        .expect("merge");

    assert_eq!(llm.requests().len(), 4, "three map calls + one reduce call");
    let desc = graph.node_writes()[0]
        .1
        .get("description")
        .and_then(|v| v.as_str())
        .expect("description")
        .to_string();
    assert_eq!(
        desc,
        "Summary of group one. Summary of group two. Summary of group three."
    );
}

/// Third and last summariser fallback: the map phase drops blank group
/// summaries, so if every group answer is blank `group_summaries` ends up
/// empty and the `len() <= 1` arm returns the raw merged description (not an
/// empty string). Distinct from the reduce-branch fallback above.
#[tokio::test]
async fn summary_returns_raw_description_when_every_group_summary_is_blank() {
    let existing = format!("fragment one{GRAPH_FIELD_SEP}fragment two");
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    let llm = CaptureLlm::with_default(vec![], "   ");
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    config.summary_max_tokens = 1;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&entity_result("fragment three", "Person"), "chunk-7", BIZ)
        .await
        .expect("merge");

    assert_eq!(llm.requests().len(), 3, "three map calls, no reduce call");
    let desc = graph.node_writes()[0]
        .1
        .get("description")
        .and_then(|v| v.as_str())
        .expect("description")
        .to_string();
    assert_eq!(
        desc,
        format!("fragment one{GRAPH_FIELD_SEP}fragment two{GRAPH_FIELD_SEP}fragment three")
    );
}

#[tokio::test]
async fn summary_falls_back_to_raw_description_on_llm_error() {
    struct ErrLlm;

    #[async_trait]
    impl LlmService for ErrLlm {
        async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
            Err(easyrag_core::CoreError::ApiStatus {
                message: "boom".to_string(),
                status_code: 500,
                request_id: None,
            })
        }
        fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
            Box::pin(futures::stream::empty())
        }
    }

    let existing = format!("fragment one{GRAPH_FIELD_SEP}fragment two");
    let graph = RecordingGraph::new().with_node(
        "Ada Lovelace",
        vec![
            ("entity_type", JsonValue::String("Person".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
        ],
    );
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    let ex = extractor(
        Arc::new(ErrLlm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&entity_result("fragment three", "Person"), "chunk-7", BIZ)
        .await
        .expect("merge must not fail when summarisation fails");

    // The raw merged description is kept verbatim.
    let desc = graph.node_writes()[0]
        .1
        .get("description")
        .and_then(|v| v.as_str())
        .expect("description")
        .to_string();
    assert_eq!(
        desc,
        format!("fragment one{GRAPH_FIELD_SEP}fragment two{GRAPH_FIELD_SEP}fragment three")
    );
}

#[tokio::test]
async fn relation_summary_uses_the_pair_name_and_relation_type() {
    let existing = format!("r one{GRAPH_FIELD_SEP}r two");
    let graph = RecordingGraph::new().with_edge(
        "Ada Lovelace",
        "Charles Babbage",
        vec![
            ("keywords", JsonValue::String("kw".to_string())),
            ("description", JsonValue::String(existing)),
            ("source_id", JsonValue::String("chunk-1".to_string())),
            ("weight", serde_json::json!(1.0_f64)),
        ],
    );
    let llm = CaptureLlm::new(vec!["Relation summary."]);
    let mut config = cfg(true);
    config.force_llm_summary_on_merge = 2;
    let ex = extractor(
        Arc::clone(&llm) as Arc<dyn LlmService>,
        Arc::clone(&graph) as Arc<dyn GraphStorage>,
        config,
    );
    ex.merge_to_graph(&relation_result("r three"), "chunk-7", BIZ)
        .await
        .expect("merge");

    // The relation branch passes a different `name` and `description_type`
    // into the same summary prompt than the entity branch does.
    snap("relation_summary_prompt", &llm.requests()[0].prompt);
    snap("relation_summary_payload", &render_writes(&graph));
}
