//! Agentic retrieval strategy implementing PageIndex-style autonomous
//! document exploration.
//!
//! The LLM agent decides which tools to call and which sections to
//! explore, mimicking how a human researcher navigates a document.
//! Requires a `ToolCallingLlmService`; the strategy is only constructed
//! when one is available (graceful fallback happens at the caller).

use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::llm::service::{
    ChatMessage, LlmRequest, MessageRole, ToolCall, ToolCallingLlmService, ToolDefinition,
};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::query_param::QueryParam;
use easyrag_core::prompt::STRUCTURE_AGENT_SYSTEM_PROMPT;
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage, VectorStorage};
use easyrag_core::util::json::{parse_json, to_pretty_json};
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;
use std::sync::Arc;

/// Result of an agentic query.
#[derive(Clone, Debug)]
pub struct AgentQueryResult {
    pub answer: String,
    pub tool_call_trace: Vec<ToolCallRecord>,
    pub referenced_sections: Vec<String>,
}

/// Record of a single tool call made by the agent.
#[derive(Clone, Debug)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub arguments: std::collections::HashMap<String, String>,
    pub result_preview: String,
    pub reasoning: String,
}

pub struct StructureAgentStrategy {
    llm_service: Arc<dyn ToolCallingLlmService>,
    kv_storage: Arc<dyn KVStorage>,
    vector_storage: Arc<dyn VectorStorage>,
    max_tool_calls: usize,
    tools: Vec<ToolDefinition>,
    doc_status_storage: Option<Arc<dyn DocStatusStorage>>,
}

fn prop(type_name: &str, description: &str) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("type".into(), JsonValue::String(type_name.to_string()));
    m.insert(
        "description".into(),
        JsonValue::String(description.to_string()),
    );
    m
}

fn object_params(properties: JsonMap) -> JsonMap {
    let mut m = JsonMap::new();
    m.insert("type".into(), JsonValue::String("object".into()));
    m.insert("properties".into(), JsonValue::Object(properties));
    m
}

impl StructureAgentStrategy {
    pub fn new(
        llm_service: Arc<dyn ToolCallingLlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
    ) -> Self {
        Self::with_max_tool_calls(llm_service, kv_storage, vector_storage, 6)
    }

    pub fn with_max_tool_calls(
        llm_service: Arc<dyn ToolCallingLlmService>,
        kv_storage: Arc<dyn KVStorage>,
        vector_storage: Arc<dyn VectorStorage>,
        max_tool_calls: usize,
    ) -> Self {
        let tools = vec![
            ToolDefinition {
                name: "list_documents".to_string(),
                description: "List all available documents with their IDs and descriptions"
                    .to_string(),
                parameters: object_params(JsonMap::new()),
            },
            ToolDefinition {
                name: "get_document_structure".to_string(),
                description: "Get the hierarchical tree structure (table of contents) of a document".to_string(),
                parameters: object_params({
                    let mut p = JsonMap::new();
                    p.insert(
                        "doc_id".into(),
                        JsonValue::Object(prop("string", "Document ID")),
                    );
                    p
                }),
            },
            ToolDefinition {
                name: "get_section_content".to_string(),
                description: "Get the text content of a specific section by its path".to_string(),
                parameters: object_params({
                    let mut p = JsonMap::new();
                    p.insert(
                        "doc_id".into(),
                        JsonValue::Object(prop("string", "Document ID")),
                    );
                    p.insert(
                        "section_path".into(),
                        JsonValue::Object(prop(
                            "string",
                            "Section path like 'Chapter 1 > Section 1.2'",
                        )),
                    );
                    p
                }),
            },
            ToolDefinition {
                name: "search_sections".to_string(),
                description: "Search section titles and summaries by keyword".to_string(),
                parameters: object_params({
                    let mut p = JsonMap::new();
                    p.insert(
                        "doc_id".into(),
                        JsonValue::Object(prop("string", "Document ID")),
                    );
                    p.insert(
                        "keyword".into(),
                        JsonValue::Object(prop("string", "Search keyword")),
                    );
                    p
                }),
            },
        ];
        Self {
            llm_service,
            kv_storage,
            vector_storage,
            max_tool_calls,
            tools,
            doc_status_storage: None,
        }
    }

    /// M5 document registry: list_documents enumerates processed docs
    /// via docStatusStorage instead of the fragile vector probe.
    pub fn with_doc_status_storage(mut self, storage: Arc<dyn DocStatusStorage>) -> Self {
        self.doc_status_storage = Some(storage);
        self
    }

    /// Execute agentic retrieval: LLM autonomously explores the document
    /// via tool calls.
    pub async fn query(
        &self,
        query: &str,
        param: &QueryParam,
    ) -> easyrag_core::Result<AgentQueryResult> {
        let mut messages = vec![
            ChatMessage {
                role: MessageRole::System,
                content: STRUCTURE_AGENT_SYSTEM_PROMPT.to_string(),
            },
            ChatMessage {
                role: MessageRole::User,
                content: query.to_string(),
            },
        ];

        let mut tool_call_trace: Vec<ToolCallRecord> = Vec::new();
        let mut tool_calls_used = 0usize;

        while tool_calls_used < self.max_tool_calls {
            let response = self
                .llm_service
                .complete_with_tools(messages.clone(), self.tools.clone())
                .await?;

            if !response.has_tool_calls() {
                // LLM produced final answer.
                return Ok(AgentQueryResult {
                    answer: response
                        .content
                        .unwrap_or_else(|| "No answer generated.".to_string()),
                    tool_call_trace: tool_call_trace.clone(),
                    referenced_sections: referenced_sections_from_trace(&tool_call_trace),
                });
            }

            // Process tool calls.
            let assistant_content = response.content.clone().unwrap_or_default();
            messages.push(ChatMessage {
                role: MessageRole::Assistant,
                content: assistant_content.clone(),
            });

            for tool_call in &response.tool_calls {
                if tool_calls_used >= self.max_tool_calls {
                    break;
                }

                let result = self
                    .execute_tool_call(tool_call, param.doc_id.as_deref(), &param.biz_id)
                    .await;
                let preview: String = result.chars().take(200).collect();
                tool_call_trace.push(ToolCallRecord {
                    tool_name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                    result_preview: preview,
                    reasoning: assistant_content.clone(),
                });

                // TOOL_RESULT_PREFIX contract: ToolCallingLlmService clients
                // (openai.rs / ollama.rs) detect this exact prefix to convert
                // User messages into API-native tool result messages. Do NOT
                // change this format without updating those implementations.
                messages.push(ChatMessage {
                    role: MessageRole::User,
                    content: format!("[Tool Result: {}]\n{}", tool_call.name, result),
                });
                tool_calls_used += 1;
            }
        }

        // Max tool calls reached, ask for final answer. Use the accumulated
        // messages as history; the final prompt itself is sent once (do not
        // push it into history first — that would duplicate the user turn).
        let final_prompt =
            "You have used all available tool calls. Please provide your final answer now.";
        let mut request = LlmRequest::new(final_prompt);
        request.history = messages;
        let final_response = self.llm_service.complete(&request).await?;

        Ok(AgentQueryResult {
            answer: final_response,
            tool_call_trace: tool_call_trace.clone(),
            referenced_sections: referenced_sections_from_trace(&tool_call_trace),
        })
    }

    /// Execute a tool call and return the result string.
    async fn execute_tool_call(
        &self,
        tool_call: &ToolCall,
        default_doc_id: Option<&str>,
        biz_id: &str,
    ) -> String {
        // `toolCall.arguments["doc_id"] ?: defaultDocId ?: ""`.
        let doc_id = tool_call
            .arguments
            .get("doc_id")
            .cloned()
            .or_else(|| default_doc_id.map(|d| d.to_string()))
            .unwrap_or_default();

        let result = match tool_call.name.as_str() {
            "list_documents" => self.execute_list_documents(biz_id).await,
            "get_document_structure" => self.execute_get_structure(&doc_id, biz_id).await,
            "get_section_content" => {
                let section_path = tool_call
                    .arguments
                    .get("section_path")
                    .cloned()
                    .unwrap_or_default();
                self.execute_get_section_content(&doc_id, &section_path, biz_id)
                    .await
            }
            "search_sections" => {
                let keyword = tool_call
                    .arguments
                    .get("keyword")
                    .cloned()
                    .unwrap_or_default();
                self.execute_search_sections(&doc_id, &keyword, biz_id).await
            }
            _ => Ok(format!("Unknown tool: {}", tool_call.name)),
        };

        match result {
            Ok(text) => text,
            Err(e) => {
                tracing::warn!("Tool call {} failed: {e}", tool_call.name);
                format!("Error executing {}: {e}", tool_call.name)
            }
        }
    }

    async fn execute_list_documents(&self, biz_id: &str) -> easyrag_core::Result<String> {
        // M5: enumerate via docStatusStorage instead of the fragile vector probe.
        if let Some(storage) = &self.doc_status_storage {
            let (docs, _) = storage
                .get_docs_paginated(
                    Some(DocStatus::Processed),
                    None,
                    1,
                    50,
                    "updated_at",
                    "desc",
                    None,
                    biz_id,
                )
                .await?;
            if docs.is_empty() {
                return Ok("No documents available.".to_string());
            }
            let lines: Vec<String> = docs
                .iter()
                .enumerate()
                .map(|(i, (doc_id, status))| {
                    let structure_mark = if status.has_structure {
                        ", has_structure=true"
                    } else {
                        ""
                    };
                    format!(
                        "{}. doc_id={}, file={}{}",
                        i + 1,
                        doc_id,
                        status.file_path,
                        structure_mark
                    )
                })
                .collect();
            return Ok(lines.join("\n"));
        }

        // Legacy fallback: query vector storage for available documents.
        let results = self
            .vector_storage
            .query("document", 10, None, None, None, biz_id)
            .await?;
        let mut doc_ids: Vec<String> = Vec::new();
        for r in &results {
            if let Some(id) = r.get("full_doc_id").and_then(|v| v.as_str()) {
                if !doc_ids.iter().any(|d| d == id) {
                    doc_ids.push(id.to_string());
                }
            }
        }
        if doc_ids.is_empty() {
            return Ok("No documents available.".to_string());
        }

        let mut lines = Vec::new();
        for (i, doc_id) in doc_ids.iter().enumerate() {
            let doc = self.kv_storage.get_by_id(doc_id, biz_id).await?;
            let file_path = doc
                .as_ref()
                .and_then(|d| d.get("file_path"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            lines.push(format!("{}. doc_id={}, file={}", i + 1, doc_id, file_path));
        }
        Ok(lines.join("\n"))
    }

    async fn execute_get_structure(&self, doc_id: &str, biz_id: &str) -> easyrag_core::Result<String> {
        let data = self
            .kv_storage
            .get_by_id(&format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"), biz_id)
            .await?;
        let Some(data) = data else {
            return Ok(format!("No structure index found for document {doc_id}"));
        };
        let Some(structure_json) = data.get("structure").and_then(|v| v.as_str()) else {
            return Ok("Structure data unavailable".to_string());
        };

        // Return a compact tree representation.
        match parse_json(structure_json) {
            Some(structure) => Ok(to_pretty_json(&structure).chars().take(3000).collect()),
            None => Ok(structure_json.chars().take(2000).collect()),
        }
    }

    async fn execute_get_section_content(
        &self,
        doc_id: &str,
        section_path: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<String> {
        // Search for chunks matching this section path.
        let chunks = self
            .vector_storage
            .query(section_path, 10, None, None, None, biz_id)
            .await?;
        let chunks: Vec<&JsonMap> = chunks
            .iter()
            .filter(|c| c.get("full_doc_id").and_then(|v| v.as_str()) == Some(doc_id))
            .collect();

        let matched: Vec<&JsonMap> = chunks
            .iter()
            .copied()
            .filter(|chunk| {
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
                    });
                match path {
                    Some(p) => p.to_lowercase().contains(&section_path.to_lowercase()),
                    None => false,
                }
            })
            .collect();

        let results: Vec<&JsonMap> = if matched.is_empty() {
            chunks.iter().copied().take(3).collect()
        } else {
            matched
        };
        if results.is_empty() {
            return Ok(format!("No content found for section: {section_path}"));
        }

        let joined = results
            .iter()
            .filter_map(|chunk| chunk.get("content").and_then(|v| v.as_str()))
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok(joined.chars().take(4000).collect())
    }

    async fn execute_search_sections(
        &self,
        doc_id: &str,
        keyword: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<String> {
        let data = self
            .kv_storage
            .get_by_id(&format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"), biz_id)
            .await?;
        let Some(data) = data else {
            return Ok(format!("No structure index found for document {doc_id}"));
        };
        let Some(structure_json) = data.get("structure").and_then(|v| v.as_str()) else {
            return Ok("Structure unavailable".to_string());
        };

        // Simple keyword search in structure JSON.
        let matches: Vec<&str> = structure_json
            .split("\"title\"")
            .filter(|part| part.to_lowercase().contains(&keyword.to_lowercase()))
            .take(5)
            .collect();

        if matches.is_empty() {
            Ok(format!("No sections matching keyword '{keyword}' found."))
        } else {
            Ok(format!("Found {} sections matching '{keyword}'", matches.len()))
        }
    }
}

fn referenced_sections_from_trace(trace: &[ToolCallRecord]) -> Vec<String> {
    trace
        .iter()
        .filter(|r| r.tool_name == "get_section_content")
        .filter_map(|r| r.arguments.get("section_path").cloned())
        .collect()
}
