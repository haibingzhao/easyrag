//! Phase 6.5 assembly factory.
//!
//! The ingestion pipeline, the query engine and the structure strategy are
//! built identically on two paths:
//! - the default-workspace path in `main`, fed by the global `ServerConfig`
//!   and the `AssembledStorage` handles;
//! - the per-workspace path in `WorkspaceResolverImpl::build_context`, fed by
//!   the tenant-merged `ServerConfig` and freshly assembled storage handles.
//!
//! Routing all three through this module keeps the wiring matrix
//! single-sourced, so a builder option added to one path can no longer be
//! silently omitted from the other — that drift is exactly how the rerank
//! asymmetry (A-2) appeared.
//!
//! # Scope — equivalence-preserving only
//!
//! Two things are deliberately NOT folded in here, because doing so would
//! change behavior rather than preserve it:
//!
//! - **The storage matrix.** `main` (via `assemble_storage`) owns pool
//!   creation and has a `json` branch; `build_context` reuses the shared
//!   pools and has no `json` branch. Unifying them would give per-workspace
//!   contexts the `json` backend they do not have today (BC-5), so the
//!   storage assembly stays on each path.
//! - **The rerank asymmetry.** `main` wires a reranker into the query engine;
//!   `build_context` does not. That difference is preserved here as an
//!   explicit `rerank: Option<_>` parameter — `main` passes its reranker,
//!   `build_context` passes `None`. Giving per-workspace queries a reranker
//!   is BC-10 and is intentionally not done.
//!
//! Every function is a plain parameterized re-assembly of the same builder
//! calls each path already made; no new indirection (`dyn`/`Box`) is
//! introduced and the resulting objects are field-for-field identical.
//!
//! # Stability
//!
//! Internal assembly surface for this workspace's binary and tests only — not
//! covered by semver. The signatures re-expose downstream crate types
//! (`easyrag_pipeline`, `easyrag_core`, `easyrag_server`) purely to re-wire
//! them; adding a builder option later is expected to change these
//! signatures, which is acceptable because no external crate consumes them.

use std::sync::Arc;

use easyrag_core::llm::embedding_func::EmbeddingFunc;
use easyrag_core::llm::service::{LlmService, RerankService, ToolCallingLlmService};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage, VectorStorage};
use easyrag_pipeline::extraction::{EntityExtractor, EntityExtractorConfig};
use easyrag_pipeline::{
    IngestionPipeline, QueryEngine, StructureAgentStrategy, StructureQueryStrategy,
    TreeSearchEngine,
};
use easyrag_server::ServerConfig;

/// Assemble the ingestion pipeline: extractor config derived from `config`,
/// then language / max-concurrency / optional keyword search.
///
/// `config` is the global config on the default path and the tenant-merged
/// config on the per-workspace path; both are plain `ServerConfig`, so the
/// extractor settings (`language` / `entity_extract_max_gleaning` /
/// `entity_types_guidance`) flow through the same override policy as before.
pub fn build_pipeline(
    llm: Arc<dyn LlmService>,
    config: &ServerConfig,
    kv: Arc<dyn KVStorage>,
    vector: Arc<dyn VectorStorage>,
    graph: Arc<dyn GraphStorage>,
    doc_status: Arc<dyn DocStatusStorage>,
    embedding_func: Arc<EmbeddingFunc>,
    keyword: Option<Arc<dyn KeywordSearchable>>,
) -> Arc<IngestionPipeline> {
    let mut extractor_config = EntityExtractorConfig::default();
    extractor_config.apply_ingest_settings(
        config.language.clone(),
        config.entity_extract_max_gleaning,
        &config.entity_types_guidance,
    );

    let mut pipeline = IngestionPipeline::new(
        llm.clone(),
        kv.clone(),
        vector.clone(),
        graph.clone(),
        doc_status,
        embedding_func,
    )
    .with_extractor(EntityExtractor::new(
        llm,
        graph,
        vector,
        kv,
        extractor_config,
    ))
    .with_language(config.language.clone())
    .with_max_concurrency(config.llm_max_async);
    if let Some(keyword) = keyword {
        pipeline = pipeline.with_keyword_search(keyword);
    }
    Arc::new(pipeline)
}

/// Assemble the structure strategy (tree search + optional agent strategy).
///
/// `tool_calling_llm` is `Some` on the default path and mirrors the resolver's
/// own `Option` on the per-workspace path: when it is `None` the agent
/// strategy is omitted exactly as `build_context` does today.
///
/// `config` supplies the two tree-search ceilings (R8 budget control). Their
/// defaults equal the values `TreeSearchEngine::new` hardcoded, so this is a
/// no-op for an operator who sets nothing.
pub fn build_structure_strategy(
    llm: Arc<dyn LlmService>,
    config: &ServerConfig,
    tool_calling_llm: Option<Arc<dyn ToolCallingLlmService>>,
    kv: Arc<dyn KVStorage>,
    vector: Arc<dyn VectorStorage>,
    doc_status: Arc<dyn DocStatusStorage>,
) -> Arc<StructureQueryStrategy> {
    let agent_strategy = tool_calling_llm.map(|tc_llm| {
        Arc::new(
            StructureAgentStrategy::new(tc_llm, kv.clone(), vector.clone())
                .with_doc_status_storage(doc_status.clone()),
        )
    });
    Arc::new(
        StructureQueryStrategy::new(
            TreeSearchEngine::with_limits(
                llm.clone(),
                config.structure_tree_max_depth,
                config.structure_tree_max_sections,
            ),
            agent_strategy,
            kv,
            vector,
            llm,
        )
        .with_doc_status_storage(doc_status),
    )
}

/// Assemble the query engine: language / rrf_k, then the optional keyword,
/// rerank and structure-strategy wiring.
///
/// The `rerank` parameter carries the intentional default-vs-workspace
/// asymmetry (see the module docs / BC-10): the default path passes its
/// reranker, the per-workspace path passes `None`.
pub fn build_query_engine(
    llm: Arc<dyn LlmService>,
    config: &ServerConfig,
    vector: Arc<dyn VectorStorage>,
    graph: Arc<dyn GraphStorage>,
    kv: Arc<dyn KVStorage>,
    keyword: Option<Arc<dyn KeywordSearchable>>,
    rerank: Option<Arc<dyn RerankService>>,
    structure: Option<Arc<StructureQueryStrategy>>,
) -> Arc<QueryEngine> {
    let mut engine = QueryEngine::new(llm, vector, graph, kv)
        .with_language(config.language.clone())
        .with_rrf_k(config.hybrid_rrf_k)
        .with_data_structure(config.query_data_structure_enabled);
    if let Some(keyword) = keyword {
        engine = engine.with_keyword_search(keyword);
    }
    if let Some(rerank) = rerank {
        engine = engine.with_rerank(rerank);
    }
    if let Some(strategy) = structure {
        engine = engine.with_structure_strategy(strategy);
    }
    Arc::new(engine)
}
