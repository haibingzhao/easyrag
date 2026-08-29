//! LLM-based document/section summarizer
//! Enriches the structure tree with concise descriptions for
//! tree-search reasoning.

use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::document_structure::SectionNode;
use easyrag_core::prompt::{format_prompt, DOC_DESCRIPTION_PROMPT, SECTION_SUMMARY_PROMPT};
use std::collections::HashMap;
use std::sync::Arc;

pub struct DocumentSummarizer {
    llm_service: Arc<dyn LlmService>,
}

impl DocumentSummarizer {
    pub fn new(llm_service: Arc<dyn LlmService>) -> Self {
        Self { llm_service }
    }

    /// Generate a one-sentence description of the full document.
    /// Returns `Ok(None)` on blank content or LLM failure.
    pub async fn generate_doc_description(&self, content: &str) -> Option<String> {
        if content.trim().is_empty() {
            return None;
        }
        let truncated: String = content.chars().take(8000).collect();
        let mut vars = HashMap::new();
        vars.insert("content".to_string(), truncated);
        let prompt = format_prompt(DOC_DESCRIPTION_PROMPT, &vars);
        match self
            .llm_service
            .complete(&LlmRequest::new(prompt).with_temperature(0.0))
            .await
        {
            Ok(result) => {
                let trimmed = result.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            Err(e) => {
                tracing::warn!("Failed to generate document description: {e}");
                None
            }
        }
    }

    /// Generate summaries for a list of sections (recursive over children).
    /// Returns new nodes with summaries populated.
    pub async fn generate_section_summaries(
        &self,
        sections: Vec<SectionNode>,
        content: &str,
    ) -> Vec<SectionNode> {
        let mut result = Vec::with_capacity(sections.len());
        for section in sections {
            result.push(Box::pin(self.summarize_node(section, content)).await);
        }
        result
    }

    async fn summarize_node(&self, node: SectionNode, content: &str) -> SectionNode {
        // Offsets are byte-based in the Rust port (heading parser contract).
        let section_text = if node.content_end > node.content_start
            && node.content_end <= content.len()
        {
            content
                .get(node.content_start..node.content_end)
                .unwrap_or("")
        } else {
            ""
        };

        let summary = if !section_text.trim().is_empty() {
            let truncated: String = section_text.chars().take(3000).collect();
            let mut vars = HashMap::new();
            vars.insert("title".to_string(), node.title.clone());
            vars.insert("content".to_string(), truncated);
            let prompt = format_prompt(SECTION_SUMMARY_PROMPT, &vars);
            match self
                .llm_service
                .complete(&LlmRequest::new(prompt).with_temperature(0.0))
                .await
            {
                Ok(result) => {
                    let trimmed = result.trim();
                    if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to summarize section '{}': {e}", node.title);
                    None
                }
            }
        } else {
            None
        };

        // Recursively summarize children.
        let mut summarized_children = Vec::with_capacity(node.children.len());
        for child in node.children {
            summarized_children.push(Box::pin(self.summarize_node(child, content)).await);
        }

        SectionNode {
            summary,
            children: summarized_children,
            ..node
        }
    }
}
