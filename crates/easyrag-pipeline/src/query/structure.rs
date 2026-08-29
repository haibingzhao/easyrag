//! Unified entry point for structure-based reasoning retrieval
//! Dispatches between multi-level tree search and agentic retrieval modes.

use super::agent::StructureAgentStrategy;
use super::tree_search::{ReasoningStep, TreeSearchEngine};
use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};
use easyrag_core::model::query_param::QueryParam;
use easyrag_core::prompt::{format_prompt, STRUCTURE_QUERY_RESPONSE};
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage, VectorStorage};
use easyrag_core::util::json::parse_json_value;
use easyrag_core::JsonMap;
use std::collections::HashSet;
use std::sync::Arc;

/// Result of a structure-based query.
#[derive(Clone, Debug)]
pub struct StructureQueryResult {
    pub answer: String,
    pub reasoning_chain: Vec<ReasoningStep>,
    pub referenced_sections: Vec<String>,
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
            None => {
                self.locate_document(
                    query,
                    param.structure_scope.as_deref(),
                    &param.biz_id,
                )
                .await?
            }
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
            .retrieve_chunks_for_sections(
                &doc_id,
                &search_result.selected_sections,
                &param.biz_id,
            )
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
    ) -> easyrag_core::Result<Option<String>> {
        if let Some(storage) = &self.doc_status_storage {
            let candidates = self.list_structure_documents(storage, scope, biz_id).await?;
            if candidates.is_empty() {
                return Ok(None);
            }
            if candidates.len() == 1 {
                return Ok(Some(candidates.into_iter().next().unwrap()));
            }
            // Multiple candidates: pick the most relevant by vector rank order.
            let results = self.vector_storage.query(query, 10, None, None, None, biz_id).await?;
            for r in &results {
                if let Some(doc_id) = r.get("full_doc_id").and_then(|v| v.as_str()) {
                    if candidates.iter().any(|c| c == doc_id) {
                        return Ok(Some(doc_id.to_string()));
                    }
                }
            }
            return Ok(Some(candidates.into_iter().next().unwrap()));
        }
        let results = self.vector_storage.query(query, 3, None, None, None, biz_id).await?;
        Ok(results
            .iter()
            .find_map(|r| r.get("full_doc_id").and_then(|v| v.as_str()).map(|s| s.to_string())))
    }

    /// Enumerate processed documents carrying a structure index (M5 document registry).
    async fn list_structure_documents(
        &self,
        storage: &Arc<dyn DocStatusStorage>,
        scope: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        let (docs, _) = storage
            .get_docs_paginated(
                Some(DocStatus::Processed),
                None,
                1,
                100,
                "updated_at",
                "desc",
                scope,
                biz_id,
            )
            .await?;
        Ok(docs
            .into_iter()
            .filter(|(_, status)| status.has_structure)
            .map(|(id, _)| id)
            .collect())
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
        for doc_id in candidates.iter().take(5) {
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
        let data = self
            .kv_storage
            .get_by_id(&format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"), biz_id)
            .await?;
        let Some(data) = data else { return Ok(None) };
        let Some(structure_json) = data.get("structure").and_then(|v| v.as_str()) else {
            return Ok(None);
        };
        match parse_json_value::<DocumentStructureIndex>(structure_json) {
            Some(index) => Ok(Some(index)),
            None => {
                tracing::warn!("Failed to deserialize structure index for {}", doc_id);
                Ok(None)
            }
        }
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
                let path = chunk
                    .get("metadata")
                    .and_then(|m| m.get("section_path"))
                    .and_then(|p| p.as_array())
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|p| p.as_str())
                            .collect::<Vec<_>>()
                            .join(" > ")
                    })
                    .unwrap_or_else(|| format!("Section {}", i + 1));
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
