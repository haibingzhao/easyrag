//! Unified entry point for structure-based reasoning retrieval
//! Dispatches between multi-level tree search and agentic retrieval modes.

use super::agent::StructureAgentStrategy;
use super::degrade;
use super::tree_search::{ReasoningStep, TreeSearchEngine};
use crate::repository::{DocChunkRepository, StructureIndexRepository};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::{
    section_path_of, DocumentStructureIndex, SectionNode, SECTION_PATH_SEP,
};
use easyrag_core::model::query_param::QueryParam;
use easyrag_core::prompt::{format_prompt, STRUCTURE_QUERY_RESPONSE};
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage, VectorStorage};
use easyrag_core::JsonMap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Hard cap on registry candidates one structure retrieval will consider (R11).
///
/// Replaces a bare first-page `size = 100`, which silently made the 101st
/// document in a large scope unreachable. Enumeration is a pure registry query
/// with no LLM cost, so the cap is a memory/latency guard rather than a budget
/// one — the fan-out limit below is what bounds the LLM bill.
///
/// The caller-visible consequences of this and the other limits in this block
/// are documented in `.docs/easyrag-data-endpoint-mode-matrix.md` §6.
pub const STRUCTURE_DATA_MAX_CANDIDATES: usize = 500;

/// Hard cap on documents a single structure retrieval fans out to.
///
/// Equals the `take(5)` the cross-document answer path has always used, now
/// named so the LLM budget can be reasoned about and asserted:
/// `STRUCTURE_DATA_MAX_DOCS × (1 + per-level fan-out)` calls per request,
/// independent of corpus size.
pub const STRUCTURE_DATA_MAX_DOCS: usize = 5;

/// Per-section content budget for `sections[].content` (R8).
///
/// Same 4000 characters the agent's `get_section_content` tool and the answer
/// prompt have always used. Truncation is reported per hit via
/// `contentTruncated` rather than silently.
pub const MAX_SECTION_CONTENT_CHARS: usize = 4000;

/// Page size for walking the document registry.
const STRUCTURE_REGISTRY_PAGE_SIZE: usize = 100;

/// Result of a structure-based query.
#[derive(Clone, Debug)]
pub struct StructureQueryResult {
    pub answer: String,
    pub reasoning_chain: Vec<ReasoningStep>,
    pub referenced_sections: Vec<String>,
}

/// A processed document carrying a structure index, as enumerated from the M5
/// document registry (R11).
///
/// Carries `file_path` and `chunks_list` because `get_docs_paginated` already
/// returns them: dropping them at the enumeration site forced a second registry
/// round trip per document and left `sections[].filePath` with no source.
#[derive(Clone, Debug)]
pub struct StructureDocument {
    pub doc_id: String,
    pub file_path: String,
    pub chunks_list: Vec<String>,
}

/// One raw structure hit (R8). Projected onto the wire by the query engine.
#[derive(Clone, Debug)]
pub struct SectionHit {
    pub doc_id: String,
    pub file_path: String,
    /// Full ancestor path, `" > "`-joined — the same string the chunker wrote
    /// into `metadata.section_path`, so a caller can verify provenance.
    pub section_path: String,
    pub title: String,
    pub level: usize,
    /// Ingestion-time LLM section summary, when one was generated.
    pub summary: Option<String>,
    /// Joined body of the chunks belonging to this section. Empty is a legal
    /// result: the chunker skips headings that carry no body text while the
    /// structure index still records the heading.
    pub content: String,
    pub content_truncated: bool,
    /// Stored chunk ids behind `content`, in document order.
    pub chunk_ids: Vec<String>,
}

/// Raw structure retrieval result (R8).
///
/// Deliberately has no `answer` field: the data endpoint's caller is an Agent
/// tool, and an LLM retelling of the hits would cost an extra round trip while
/// destroying the provenance the Agent needs.
#[derive(Clone, Debug)]
pub struct StructureSectionsResult {
    pub sections: Vec<SectionHit>,
    pub reasoning_chain: Vec<ReasoningStep>,
    /// Stable markers from [`degrade`]; empty when everything asked for ran.
    pub degrade_reasons: Vec<String>,
}

pub struct StructureQueryStrategy {
    tree_search_engine: TreeSearchEngine,
    agent_strategy: Option<Arc<StructureAgentStrategy>>,
    kv_storage: Arc<dyn KVStorage>,
    vector_storage: Arc<dyn VectorStorage>,
    llm_service: Arc<dyn LlmService>,
    doc_status_storage: Option<Arc<dyn DocStatusStorage>>,
}

impl StructureQueryStrategy {
    pub fn new(
        tree_search_engine: TreeSearchEngine,
        agent_strategy: Option<Arc<StructureAgentStrategy>>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        llm_service: Arc<dyn LlmService>,
    ) -> Self {
        Self {
            tree_search_engine,
            agent_strategy,
            kv_storage,
            vector_storage,
            llm_service,
            doc_status_storage: None,
        }
    }

    /// M5 document registry: enables structure-document enumeration,
    /// scoped locating and cross-document queries.
    pub fn with_doc_status_storage(mut self, storage: Arc<dyn DocStatusStorage>) -> Self {
        self.doc_status_storage = Some(storage);
        self
    }

    /// Execute a structure-based reasoning query.
    pub async fn query(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<StructureQueryResult> {
        // M5: cross-document structure query scoped by file path prefix.
        if param.doc_id.is_none()
            && param
                .structure_scope
                .as_deref()
                .map_or(false, |s| !s.trim().is_empty())
        {
            if let Some(result) = self.cross_document_query(query, param).await? {
                return Ok(result);
            }
        }

        // 1. Determine target document.
        let doc_id = match &param.doc_id {
            Some(id) => Some(id.clone()),
            None => self
                .locate_document(query, param.structure_scope.as_deref(), &param.biz_id)
                .await?
                .map(|doc| doc.doc_id),
        };
        let Some(doc_id) = doc_id else {
            tracing::warn!("No document located for structure query, falling back");
            return Ok(StructureQueryResult {
                answer: "No document with structure index found for this query.".to_string(),
                reasoning_chain: Vec::new(),
                referenced_sections: Vec::new(),
            });
        };

        // 2. Load structure index.
        let Some(structure) = self.load_structure_index(&doc_id, &param.biz_id).await? else {
            tracing::warn!("No structure index found for document {}", doc_id);
            return Ok(StructureQueryResult {
                answer: format!(
                    "Document '{doc_id}' has no structure index. Use naive/hybrid mode instead."
                ),
                reasoning_chain: Vec::new(),
                referenced_sections: Vec::new(),
            });
        };

        // 3. Choose reasoning strategy.
        if param.use_agent_mode {
            if let Some(agent) = &self.agent_strategy {
                // Agentic mode: LLM autonomously explores via tool calls.
                tracing::info!("Using agentic retrieval for document {}", doc_id);
                let agent_result = agent.query(query, param).await?;
                return Ok(StructureQueryResult {
                    answer: agent_result.answer,
                    reasoning_chain: Vec::new(),
                    referenced_sections: agent_result.referenced_sections,
                });
            }
        }

        // Multi-level tree search: layer-by-layer LLM reasoning.
        tracing::info!("Using tree search for document {}", doc_id);
        let search_result = self.tree_search_engine.search(query, &structure).await?;

        // 4. Retrieve chunks for selected sections.
        let chunks = self
            .retrieve_chunks_for_sections(&doc_id, &search_result.selected_sections, &param.biz_id)
            .await?;

        // 5. Generate final answer with reasoning context.
        let answer = self
            .generate_answer(
                query,
                &chunks,
                &search_result.reasoning_chain,
                &param.response_type,
            )
            .await?;

        Ok(StructureQueryResult {
            answer,
            reasoning_chain: search_result.reasoning_chain,
            referenced_sections: search_result
                .selected_sections
                .iter()
                .map(|s| s.title.clone())
                .collect(),
        })
    }

    /// Locate the best document for a query.
    /// Enumerates documents with structure index via docStatusStorage (optionally
    /// scoped by file path prefix); falls back to the legacy vector enumeration.
    async fn locate_document(
        &self,
        query: &str,
        scope: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<StructureDocument>> {
        if let Some(storage) = &self.doc_status_storage {
            let candidates = self
                .list_structure_documents(storage, scope, biz_id)
                .await?;
            if candidates.is_empty() {
                return Ok(None);
            }
            // Multiple candidates: pick the most relevant by vector rank order.
            Ok(self
                .rank_candidates_by_relevance(query, candidates, biz_id)
                .await?
                .into_iter()
                .next())
        } else {
            let results = self
                .vector_storage
                .query(query, 3, None, None, None, biz_id)
                .await?;
            Ok(results.iter().find_map(|r| {
                r.get("full_doc_id")
                    .and_then(|v| v.as_str())
                    .map(|s| StructureDocument {
                        doc_id: s.to_string(),
                        // No registry row was consulted, so neither of these is
                        // known. The answer path never reads them; only the raw-hit
                        // path does, and it reports the gap as a degradation.
                        file_path: String::new(),
                        chunks_list: Vec::new(),
                    })
            }))
        }
    }

    /// Order registry candidates by relevance to `query`, most relevant first.
    ///
    /// Shared by single-document locating and the raw-hit path so both agree on
    /// "best": vector rank decides, and candidates no chunk points at keep their
    /// registry order (newest first) at the tail. Returns the input untouched
    /// for zero or one candidate, which is also what skips the vector query —
    /// the same short-circuit `locate_document` has always had.
    async fn rank_candidates_by_relevance(
        &self,
        query: &str,
        candidates: Vec<StructureDocument>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<StructureDocument>> {
        if candidates.len() <= 1 {
            return Ok(candidates);
        }
        let results = self
            .vector_storage
            .query(query, 10, None, None, None, biz_id)
            .await?;
        let mut rest = candidates;
        let mut ranked: Vec<StructureDocument> = Vec::with_capacity(rest.len());
        for r in &results {
            let Some(doc_id) = r.get("full_doc_id").and_then(|v| v.as_str()) else {
                continue;
            };
            if let Some(pos) = rest.iter().position(|c| c.doc_id == doc_id) {
                ranked.push(rest.remove(pos));
            }
        }
        ranked.extend(rest);
        Ok(ranked)
    }

    /// Enumerate processed documents carrying a structure index (M5 document
    /// registry), walking every page up to [`STRUCTURE_DATA_MAX_CANDIDATES`].
    ///
    /// Order is `updated_at desc` throughout, so the cap drops the least
    /// recently touched documents rather than an arbitrary page boundary.
    async fn list_structure_documents(
        &self,
        storage: &Arc<dyn DocStatusStorage>,
        scope: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<StructureDocument>> {
        let mut docs_out: Vec<StructureDocument> = Vec::new();
        let mut page = 1usize;
        loop {
            let (docs, total_count) = storage
                .get_docs_paginated(
                    Some(DocStatus::Processed),
                    None,
                    page,
                    STRUCTURE_REGISTRY_PAGE_SIZE,
                    "updated_at",
                    "desc",
                    scope,
                    biz_id,
                )
                .await?;
            let fetched = docs.len();
            for (doc_id, status) in docs {
                if status.has_structure {
                    docs_out.push(StructureDocument {
                        doc_id,
                        file_path: status.file_path,
                        chunks_list: status.chunks_list,
                    });
                }
            }
            let last_page = fetched < STRUCTURE_REGISTRY_PAGE_SIZE
                || page * STRUCTURE_REGISTRY_PAGE_SIZE >= total_count;
            if fetched == 0 || last_page || docs_out.len() >= STRUCTURE_DATA_MAX_CANDIDATES {
                break;
            }
            page += 1;
        }
        docs_out.truncate(STRUCTURE_DATA_MAX_CANDIDATES);
        Ok(docs_out)
    }

    /// M5: cross-document structure query - per-document tree search within the
    /// scope, merged into a single answer. Returns None when no structure
    /// document is available so the caller falls back to single-document flow.
    async fn cross_document_query(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<Option<StructureQueryResult>> {
        let Some(storage) = &self.doc_status_storage else {
            return Ok(None);
        };
        let candidates = self
            .list_structure_documents(storage, param.structure_scope.as_deref(), &param.biz_id)
            .await?;
        if candidates.is_empty() {
            tracing::warn!(
                "No structure documents in scope {:?}, falling back",
                param.structure_scope
            );
            return Ok(None);
        }

        let mut merged_chain: Vec<ReasoningStep> = Vec::new();
        let mut merged_sections: Vec<String> = Vec::new();
        let mut merged_chunks: Vec<JsonMap> = Vec::new();
        // Registry order, not relevance order: this is the answer path, whose
        // behaviour must not shift under R11. Only the raw-hit path re-ranks.
        for doc in candidates.iter().take(STRUCTURE_DATA_MAX_DOCS) {
            let doc_id = &doc.doc_id;
            let Some(structure) = self.load_structure_index(doc_id, &param.biz_id).await? else {
                continue;
            };
            let search_result = self.tree_search_engine.search(query, &structure).await?;
            merged_chain.extend(search_result.reasoning_chain.clone());
            merged_sections.extend(
                search_result
                    .selected_sections
                    .iter()
                    .map(|s| format!("{doc_id}: {}", s.title)),
            );
            merged_chunks.extend(
                self.retrieve_chunks_for_sections(
                    doc_id,
                    &search_result.selected_sections,
                    &param.biz_id,
                )
                .await?,
            );
        }
        if merged_chunks.is_empty() {
            return Ok(None);
        }

        let answer = self
            .generate_answer(query, &merged_chunks, &merged_chain, &param.response_type)
            .await?;
        Ok(Some(StructureQueryResult {
            answer,
            reasoning_chain: merged_chain,
            referenced_sections: merged_sections,
        }))
    }

    /// Load and deserialize a document's structure index from KV storage.
    async fn load_structure_index(
        &self,
        doc_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<DocumentStructureIndex>> {
        StructureIndexRepository::new(&self.kv_storage)
            .load_typed(doc_id, biz_id)
            .await
    }

    /// Raw structure retrieval for the pure-data endpoint (R8).
    ///
    /// Shares document locating, structure loading and tree search with
    /// [`query`](Self::query), with two deliberate differences:
    ///
    /// * **No answer synthesis.** [`generate_answer`] is skipped entirely. The
    ///   caller is an Agent tool that wants the hits themselves; an LLM retelling
    ///   would cost an extra call per request and replace verifiable provenance
    ///   with prose.
    /// * **No agent mode.** Every agent tool returns a prose string, so none of
    ///   them can be projected into `sections[]`. `useAgentMode` is reported as
    ///   [`degrade::AGENT_MODE_UNSUPPORTED`] and tree search runs instead.
    ///
    /// Unlike [`cross_document_query`], an empty outcome comes back as empty
    /// `sections` plus a degradation marker rather than `Ok(None)`: the data path
    /// has no single-document flow to hand off to, and returning `None` there
    /// would re-run the locate and the tree search for the same empty answer.
    pub async fn search_sections(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<StructureSectionsResult> {
        let mut degrade_reasons: Vec<String> = Vec::new();
        let mut reasoning_chain: Vec<ReasoningStep> = Vec::new();
        let mut sections: Vec<SectionHit> = Vec::new();

        if param.use_agent_mode {
            degrade_reasons.push(degrade::AGENT_MODE_UNSUPPORTED.to_string());
        }

        // 1. Candidate set: an explicit docId, or the scoped registry.
        let candidates = match &param.doc_id {
            Some(doc_id) => vec![self.describe_document(doc_id, param).await?],
            None => match &self.doc_status_storage {
                Some(storage) => {
                    self.list_structure_documents(
                        storage,
                        param.structure_scope.as_deref(),
                        &param.biz_id,
                    )
                    .await?
                }
                // No registry to enumerate, so fall back to the single best
                // vector hit — the same fallback `locate_document` uses.
                None => self
                    .locate_document(query, param.structure_scope.as_deref(), &param.biz_id)
                    .await?
                    .into_iter()
                    .collect(),
            },
        };
        if candidates.is_empty() {
            degrade_reasons.push(degrade::NO_DOCUMENT_IN_SCOPE.to_string());
            return Ok(StructureSectionsResult {
                sections,
                reasoning_chain,
                degrade_reasons: dedup_reasons(degrade_reasons),
            });
        }

        // 2. Relevance order, then the fan-out cap that bounds the LLM bill.
        let ranked = self
            .rank_candidates_by_relevance(query, candidates, &param.biz_id)
            .await?;
        if ranked.len() > STRUCTURE_DATA_MAX_DOCS {
            tracing::info!(
                "Structure scope holds {} documents, searching the {} most relevant",
                ranked.len(),
                STRUCTURE_DATA_MAX_DOCS
            );
            degrade_reasons.push(degrade::SCOPE_TRUNCATED.to_string());
        }

        // 3. Per-document tree search, then the exact section→chunk join.
        for doc in ranked.iter().take(STRUCTURE_DATA_MAX_DOCS) {
            let Some(structure) = self
                .load_structure_index(&doc.doc_id, &param.biz_id)
                .await?
            else {
                degrade_reasons.push(degrade::NO_INDEX.to_string());
                continue;
            };
            let search_result = self.tree_search_engine.search(query, &structure).await?;
            reasoning_chain.extend(search_result.reasoning_chain);
            sections.extend(
                self.build_section_hits(
                    doc,
                    &structure,
                    &search_result.selected_sections,
                    param,
                    &mut degrade_reasons,
                )
                .await?,
            );
        }

        Ok(StructureSectionsResult {
            sections,
            reasoning_chain,
            degrade_reasons: dedup_reasons(degrade_reasons),
        })
    }

    /// Registry view of one explicitly-named document.
    ///
    /// An absent row is not an error — the structure index may well exist for a
    /// document the registry never saw — but then `file_path` and `chunks_list`
    /// are unknown, which surfaces as an empty `filePath` and, via the content
    /// fallback, `structure_section_content_approximate`.
    async fn describe_document(
        &self,
        doc_id: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<StructureDocument> {
        if let Some(storage) = &self.doc_status_storage {
            if let Some(row) = storage.get_by_id(doc_id, &param.biz_id).await? {
                return Ok(StructureDocument {
                    doc_id: doc_id.to_string(),
                    file_path: row
                        .get("file_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    chunks_list: row
                        .get("chunks_list")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default(),
                });
            }
        }
        Ok(StructureDocument {
            doc_id: doc_id.to_string(),
            file_path: String::new(),
            chunks_list: Vec::new(),
        })
    }

    /// Turn tree-search selections into raw hits with their bodies attached.
    async fn build_section_hits(
        &self,
        doc: &StructureDocument,
        structure: &DocumentStructureIndex,
        selected: &[SectionNode],
        param: &QueryParam,
        degrade_reasons: &mut Vec<String>,
    ) -> easyrag_core::Result<Vec<SectionHit>> {
        if selected.is_empty() {
            return Ok(Vec::new());
        }
        let paths = section_paths(&structure.sections);

        // Section identity is (full path, body offset) — the same key tree
        // search dedupes its leaves by. Two branches may carry the same title;
        // they cannot carry the same title *and* the same offset.
        let mut targets: Vec<(String, &SectionNode)> = Vec::new();
        for node in selected {
            let path = paths
                .get(&node.title)
                .cloned()
                .unwrap_or_else(|| node.path_with(&[]));
            let duplicate = targets
                .iter()
                .any(|(p, n)| p == &path && n.content_start == node.content_start);
            if !duplicate {
                targets.push((path, node));
            }
        }

        let doc_chunks = self
            .document_chunks(doc, selected, param, degrade_reasons)
            .await?;

        // Assign every chunk to the **longest** target path it matches.
        //
        // Prefix matching is mandatory here, not an optimisation: the structure
        // tree builder drops headings deeper than its max depth while the
        // structure-aware chunker does not, so in any document with headings
        // below that depth an indexed node's path is a strict *prefix* of its
        // chunks' paths. Matching on equality alone would return empty bodies
        // for exactly those documents. Longest-wins is what keeps a chunk out of
        // both a parent and its selected child.
        let mut assigned: Vec<Vec<&JsonMap>> = vec![Vec::new(); targets.len()];
        for chunk in &doc_chunks {
            let Some(chunk_path) = section_path_of(chunk) else {
                continue;
            };
            // The preamble before the first heading carries an empty path and
            // belongs to no section.
            if chunk_path.is_empty() {
                continue;
            }
            let mut best: Option<usize> = None;
            for (index, (path, _)) in targets.iter().enumerate() {
                let matches = chunk_path == *path
                    || chunk_path.starts_with(&format!("{path}{SECTION_PATH_SEP}"));
                if !matches {
                    continue;
                }
                let longer = best.map_or(true, |prev| targets[prev].0.len() < path.len());
                if longer {
                    best = Some(index);
                }
            }
            if let Some(index) = best {
                assigned[index].push(chunk);
            }
        }

        let mut hits = Vec::with_capacity(targets.len());
        for ((path, node), chunks) in targets.into_iter().zip(assigned) {
            // Same join and budget as the agent's `get_section_content` tool.
            // An empty body is a legitimate result, not a degradation: the
            // chunker skips headings that carry no text while the index still
            // records the heading.
            let joined = chunks
                .iter()
                .filter_map(|c| c.get("content").and_then(|v| v.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n");
            let content: String = joined.chars().take(MAX_SECTION_CONTENT_CHARS).collect();
            let truncated = content.chars().count() < joined.chars().count();
            hits.push(SectionHit {
                doc_id: doc.doc_id.clone(),
                file_path: doc.file_path.clone(),
                section_path: path,
                title: node.title.clone(),
                level: node.level,
                summary: node.summary.clone(),
                content_truncated: truncated,
                content,
                chunk_ids: chunks
                    .iter()
                    .filter_map(|c| c.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
                    .collect(),
            });
        }
        Ok(hits)
    }

    /// Stored chunks of one document, for the section join.
    ///
    /// Prefers the exact registry join. Falls back to the answer path's vector
    /// heuristic only when the registry carries no chunk ids for the document,
    /// and always says so via
    /// [`degrade::SECTION_CONTENT_APPROXIMATE`] — that heuristic can return
    /// chunks from outside the requested section, which is acceptable when
    /// feeding an LLM and unacceptable when reporting provenance.
    async fn document_chunks(
        &self,
        doc: &StructureDocument,
        selected: &[SectionNode],
        param: &QueryParam,
        degrade_reasons: &mut Vec<String>,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if !doc.chunks_list.is_empty() {
            return DocChunkRepository::from_kv(&self.kv_storage)
                .chunks_by_ids(&doc.chunks_list, &param.biz_id)
                .await;
        }
        degrade_reasons.push(degrade::SECTION_CONTENT_APPROXIMATE.to_string());
        self.retrieve_chunks_for_sections(&doc.doc_id, selected, &param.biz_id)
            .await
    }

    /// Retrieve chunk content for selected sections by matching
    /// section_path metadata.
    async fn retrieve_chunks_for_sections(
        &self,
        doc_id: &str,
        sections: &[SectionNode],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if sections.is_empty() {
            return Ok(Vec::new());
        }

        // Use vector search scoped to the document, then filter by section path.
        let section_titles: HashSet<String> =
            sections.iter().map(|s| s.title.clone()).collect();
        let query_text = sections
            .iter()
            .map(|s| s.title.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let all_chunks: Vec<JsonMap> = self
            .vector_storage
            .query(&query_text, 20, None, None, None, biz_id)
            .await?
            .into_iter()
            .filter(|c| c.get("full_doc_id").and_then(|v| v.as_str()) == Some(doc_id))
            .collect();

        // Prefer chunks whose metadata.section_path matches selected sections.
        let matched: Vec<JsonMap> = all_chunks
            .iter()
            .filter(|chunk| {
                chunk
                    .get("metadata")
                    .and_then(|m| m.get("section_path"))
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts.iter().filter_map(|p| p.as_str()).any(|path| {
                            section_titles.iter().any(|t| t == path)
                        })
                    })
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if matched.is_empty() {
            Ok(all_chunks.into_iter().take(10).collect())
        } else {
            Ok(matched)
        }
    }

    /// Generate the final answer using retrieved chunks and reasoning context.
    async fn generate_answer(
        &self,
        query: &str,
        chunks: &[JsonMap],
        reasoning_chain: &[ReasoningStep],
        response_type: &str,
    ) -> easyrag_core::Result<String> {
        if chunks.is_empty() {
            return Ok("No relevant sections found for this query.".to_string());
        }

        let section_content = chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| {
                let path = section_path_of(chunk).unwrap_or_else(|| format!("Section {}", i + 1));
                let content = chunk.get("content").and_then(|v| v.as_str()).unwrap_or("");
                format!("--- [{}] ---\n{}", path, content)
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let reasoning_text = reasoning_chain
            .iter()
            .map(|step| {
                format!(
                    "Level {}: Considered [{}] → Selected [{}] ({})\nReasoning: {}",
                    step.level,
                    step.sections_considered.join(", "),
                    step.sections_selected.join(", "),
                    step.confidence,
                    step.thinking
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut vars = std::collections::HashMap::new();
        vars.insert("query".to_string(), query.to_string());
        vars.insert("section_content".to_string(), section_content);
        vars.insert(
            "retrieval_reasoning".to_string(),
            if reasoning_text.is_empty() {
                "Direct section selection".to_string()
            } else {
                reasoning_text
            },
        );
        vars.insert("response_type".to_string(), response_type.to_string());

        let prompt = format_prompt(STRUCTURE_QUERY_RESPONSE, &vars);
        let response = self
            .llm_service
            .complete(&LlmRequest::new(prompt).with_temperature(0.0))
            .await?;
        Ok(response)
    }
}

/// Map every section title in a structure tree to its full ancestor path.
///
/// `TreeSearchResult::selected_sections` carries the selected nodes but not
/// their ancestors, and a node cannot rebuild its own path without them — hence
/// this walk. Duplicate titles resolve **first-wins** in depth-first document
/// order. The alternative would be teaching `TreeSearchEngine` to return paths,
/// which disturbs its contract and its tests for little gain: the
/// early-termination path returns leaves, and the same leaf title appearing under
/// two different branches is rare. Every hit also carries `sectionPath`, so a
/// mis-resolution is visible to the caller rather than silent.
fn section_paths(sections: &[SectionNode]) -> HashMap<String, String> {
    fn walk(
        nodes: &[SectionNode],
        ancestors: &mut Vec<String>,
        paths: &mut HashMap<String, String>,
    ) {
        for node in nodes {
            paths
                .entry(node.title.clone())
                .or_insert_with(|| node.path_with(ancestors));
            ancestors.push(node.title.clone());
            walk(&node.children, ancestors, paths);
            ancestors.pop();
        }
    }
    let mut paths = HashMap::new();
    walk(sections, &mut Vec::new(), &mut paths);
    paths
}

/// Collapse repeated degradation markers, keeping first-occurrence order.
///
/// Several markers are emitted inside a per-document loop, so a scope of five
/// index-less documents would otherwise repeat one marker five times.
fn dedup_reasons(reasons: Vec<String>) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    reasons
        .into_iter()
        .filter(|reason| seen.insert(reason.clone()))
        .collect()
}
