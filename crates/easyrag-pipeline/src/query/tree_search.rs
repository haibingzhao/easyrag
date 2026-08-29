//! Multi-level tree search engine implementing PageIndex-style reasoning
//! retrieval.
//!
//! Instead of one-shot full-tree reasoning, this engine navigates the
//! document structure layer by layer, using LLM reasoning at each level to
//! decide which branches to explore — mimicking how a human expert
//! navigates a table of contents.

use easyrag_core::llm::service::{LlmRequest, LlmService, ResponseFormat};
use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};
use easyrag_core::prompt::{format_prompt, STRUCTURE_DRILL_DOWN_PROMPT, STRUCTURE_NAVIGATE_PROMPT};
use easyrag_core::util::json::parse_json;
use std::collections::HashMap;
use std::sync::Arc;

/// A single step in the reasoning chain (one LLM reasoning call at one
/// tree level).
#[derive(Clone, Debug)]
pub struct ReasoningStep {
    pub level: usize,
    pub sections_considered: Vec<String>,
    pub sections_selected: Vec<String>,
    pub thinking: String,
    pub confidence: String,
}

/// Result of a multi-level tree search.
#[derive(Clone, Debug)]
pub struct TreeSearchResult {
    pub selected_sections: Vec<SectionNode>,
    pub reasoning_chain: Vec<ReasoningStep>,
    pub total_llm_calls: usize,
}

pub struct TreeSearchEngine {
    llm_service: Arc<dyn LlmService>,
    max_depth: usize,
    max_sections_per_level: usize,
}

impl TreeSearchEngine {
    pub fn new(llm_service: Arc<dyn LlmService>) -> Self {
        Self::with_limits(llm_service, 3, 3)
    }

    pub fn with_limits(
        llm_service: Arc<dyn LlmService>,
        max_depth: usize,
        max_sections_per_level: usize,
    ) -> Self {
        Self {
            llm_service,
            max_depth,
            max_sections_per_level,
        }
    }

    /// Perform multi-level tree search: navigate the structure tree layer
    /// by layer using LLM reasoning.
    pub async fn search(
        &self,
        query: &str,
        structure_index: &DocumentStructureIndex,
    ) -> easyrag_core::Result<TreeSearchResult> {
        let mut reasoning_chain: Vec<ReasoningStep> = Vec::new();
        let mut llm_calls = 0usize;

        // Level 0: Navigate top-level sections.
        let top_level = &structure_index.sections;
        if top_level.is_empty() {
            return Ok(TreeSearchResult {
                selected_sections: Vec::new(),
                reasoning_chain: Vec::new(),
                total_llm_calls: 0,
            });
        }

        // Small document optimization: ≤8 top-level leaf nodes → select
        // all without any LLM call.
        if top_level.len() <= 8 && top_level.iter().all(|s| s.children.is_empty()) {
            tracing::debug!(
                "Small document ({} leaf sections), selecting all without LLM",
                top_level.len()
            );
            return Ok(TreeSearchResult {
                selected_sections: top_level.clone(),
                reasoning_chain: Vec::new(),
                total_llm_calls: 0,
            });
        }

        let sections_text = format_sections_for_prompt(top_level);
        let mut vars = HashMap::new();
        vars.insert(
            "doc_title".to_string(),
            structure_index
                .doc_title
                .clone()
                .unwrap_or_else(|| "Unknown".to_string()),
        );
        vars.insert(
            "doc_description".to_string(),
            structure_index
                .doc_description
                .clone()
                .unwrap_or_else(|| "N/A".to_string()),
        );
        vars.insert("sections".to_string(), sections_text);
        vars.insert("query".to_string(), query.to_string());
        let navigate_prompt = format_prompt(STRUCTURE_NAVIGATE_PROMPT, &vars);

        let mut request = LlmRequest::new(navigate_prompt).with_temperature(0.0);
        request.response_format = Some(ResponseFormat::JsonObject);
        let navigate_response = self.llm_service.complete(&request).await?;
        llm_calls += 1;

        let navigate_result = parse_json(&navigate_response);
        let thinking = navigate_result
            .as_ref()
            .and_then(|j| j.get("thinking"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let confidence = navigate_result
            .as_ref()
            .and_then(|j| j.get("confidence"))
            .and_then(|v| v.as_str())
            .unwrap_or("medium")
            .to_string();
        let selected_ids: Vec<String> = navigate_result
            .as_ref()
            .and_then(|j| j.get("selected_sections"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_else(|| {
                top_level
                    .iter()
                    .take(self.max_sections_per_level)
                    .map(|s| s.title.clone())
                    .collect()
            });

        let mut selected_top_level: Vec<SectionNode> = top_level
            .iter()
            .filter(|node| selected_ids.iter().any(|id| *id == node.title))
            .cloned()
            .collect();
        if selected_top_level.is_empty() {
            selected_top_level = top_level
                .iter()
                .take(self.max_sections_per_level)
                .cloned()
                .collect();
        }

        reasoning_chain.push(ReasoningStep {
            level: 0,
            sections_considered: top_level.iter().map(|s| s.title.clone()).collect(),
            sections_selected: selected_top_level.iter().map(|s| s.title.clone()).collect(),
            thinking,
            confidence: confidence.clone(),
        });

        // Early termination: high confidence and few selections.
        if confidence == "high" && selected_top_level.len() <= 2 {
            let leaves = selected_top_level
                .iter()
                .flat_map(collect_leaves)
                .collect();
            return Ok(TreeSearchResult {
                selected_sections: leaves,
                reasoning_chain,
                total_llm_calls: llm_calls,
            });
        }

        // Drill down into selected sections.
        let mut current_level = selected_top_level;
        for depth in 1..self.max_depth {
            let sections_with_children: Vec<&SectionNode> =
                current_level.iter().filter(|s| !s.children.is_empty()).collect();
            if sections_with_children.is_empty() {
                break; // All are leaf nodes.
            }

            let mut next_level: Vec<SectionNode> = Vec::new();

            for parent in sections_with_children {
                let children_text = format_sections_for_prompt(&parent.children);
                let mut vars = HashMap::new();
                vars.insert("parent_title".to_string(), parent.title.clone());
                vars.insert(
                    "parent_summary".to_string(),
                    parent
                        .summary
                        .clone()
                        .unwrap_or_else(|| "N/A".to_string()),
                );
                vars.insert("sub_sections".to_string(), children_text);
                vars.insert("query".to_string(), query.to_string());
                let drill_prompt = format_prompt(STRUCTURE_DRILL_DOWN_PROMPT, &vars);

                let mut request = LlmRequest::new(drill_prompt).with_temperature(0.0);
                request.response_format = Some(ResponseFormat::JsonObject);
                let drill_response = self.llm_service.complete(&request).await?;
                llm_calls += 1;

                let drill_result = parse_json(&drill_response);
                let drill_thinking = drill_result
                    .as_ref()
                    .and_then(|j| j.get("thinking"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let need_deeper = drill_result
                    .as_ref()
                    .and_then(|j| j.get("need_deeper"))
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let drill_selected_ids: Vec<String> = drill_result
                    .as_ref()
                    .and_then(|j| j.get("selected_sections"))
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                reasoning_chain.push(ReasoningStep {
                    level: depth,
                    sections_considered: parent
                        .children
                        .iter()
                        .map(|s| s.title.clone())
                        .collect(),
                    sections_selected: drill_selected_ids.clone(),
                    thinking: drill_thinking,
                    confidence: if need_deeper { "medium" } else { "high" }.to_string(),
                });

                if drill_selected_ids.iter().any(|id| id == "__self__") {
                    // Parent section itself is sufficient.
                    next_level.push(parent.clone());
                } else {
                    let mut selected_children: Vec<SectionNode> = parent
                        .children
                        .iter()
                        .filter(|child| drill_selected_ids.iter().any(|id| *id == child.title))
                        .cloned()
                        .collect();
                    if selected_children.is_empty() {
                        selected_children = parent
                            .children
                            .iter()
                            .take(self.max_sections_per_level)
                            .cloned()
                            .collect();
                    }
                    next_level.extend(selected_children);
                }
            }

            // Also keep leaf nodes from current level (no children to
            // drill into).
            next_level.extend(
                current_level
                    .iter()
                    .filter(|s| s.children.is_empty())
                    .cloned(),
            );

            current_level = next_level;

            // Early termination if all are leaves.
            if current_level.iter().all(|s| s.children.is_empty()) {
                break;
            }
        }

        // Collect final leaf sections (deduplicated by position, mirroring
        // data-class `distinct`).
        let mut seen: Vec<(String, usize)> = Vec::new();
        let mut final_sections: Vec<SectionNode> = Vec::new();
        for node in current_level.iter().flat_map(collect_leaves) {
            let key = (node.title.clone(), node.content_start);
            if !seen.contains(&key) {
                seen.push(key);
                final_sections.push(node);
            }
        }

        tracing::debug!(
            "Tree search completed: {} LLM calls, {} reasoning steps, {} final sections",
            llm_calls,
            reasoning_chain.len(),
            final_sections.len()
        );

        Ok(TreeSearchResult {
            selected_sections: final_sections,
            reasoning_chain,
            total_llm_calls: llm_calls,
        })
    }
}

/// Format sections for LLM prompt input (compact: title + summary only).
fn format_sections_for_prompt(sections: &[SectionNode]) -> String {
    sections
        .iter()
        .enumerate()
        .map(|(index, s)| {
            let summary = s
                .summary
                .as_ref()
                .map(|sum| format!(" — {sum}"))
                .unwrap_or_default();
            let child_count = if !s.children.is_empty() {
                format!(" [{} sub-sections]", s.children.len())
            } else {
                String::new()
            };
            format!("{}. \"{}\"{}{}", index + 1, s.title, child_count, summary)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Collect all leaf nodes from a section (or the section itself if it is
/// a leaf).
fn collect_leaves(node: &SectionNode) -> Vec<SectionNode> {
    if node.children.is_empty() {
        vec![node.clone()]
    } else {
        node.children.iter().flat_map(collect_leaves).collect()
    }
}
