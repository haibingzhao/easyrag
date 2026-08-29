//! Ollama LLM client.

use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, LlmToolResponse, MessageRole,
    ToolCall, ToolCallingLlmService, ToolDefinition,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

use crate::{arguments_to_map, parse_tool_result_message, serialize_tools};

/// Configuration for [`OllamaClient`].
#[derive(Clone, Debug)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Dimension of embeddings produced by the configured model.
    pub embedding_dimension: usize,
}

impl OllamaConfig {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(30),
            embedding_dimension: 4096,
        }
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Ollama client implementing chat completion (incl. streaming) and
/// embeddings via the native Ollama HTTP API.
pub struct OllamaClient {
    config: OllamaConfig,
    http: reqwest::Client,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> easyrag_core::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.timeout)
            .build()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("http client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Build the message list (system prompt + history + final user message).
    fn build_messages(request: &LlmRequest) -> Vec<JsonValue> {
        let mut messages = Vec::new();
        if let Some(sp) = &request.system_prompt {
            messages.push(json!({ "role": "system", "content": sp }));
        }
        for msg in &request.history {
            let role = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };
            messages.push(json!({ "role": role, "content": msg.content }));
        }
        messages.push(json!({ "role": "user", "content": request.prompt }));
        messages
    }

    ///: temperature (only when > 0) and
    /// `num_predict` for max tokens.
    fn build_options(request: &LlmRequest) -> JsonValue {
        let mut options = serde_json::Map::new();
        if request.temperature > 0.0 {
            options.insert("temperature".to_string(), json!(request.temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            options.insert("num_predict".to_string(), json!(max_tokens));
        }
        JsonValue::Object(options)
    }

    /// Build the JSON body for the /api/chat endpoint.
    fn build_chat_body(&self, request: &LlmRequest, stream: bool) -> JsonValue {
        json!({
            "model": self.config.model,
            "messages": Self::build_messages(request),
            "stream": stream,
            "options": Self::build_options(request),
        })
    }

    /// Send a POST request with a JSON body and return the raw response
    /// string ( — no status check, parse failures are
    /// tolerated downstream by defaulting to empty content).
    async fn post_json(&self, path: &str, body: &JsonValue) -> easyrag_core::Result<String> {
        let url = format!("{}{}", self.config.base_url, path);
        let response = self
            .http
            .post(&url)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    easyrag_core::CoreError::ApiTimeout
                } else {
                    easyrag_core::CoreError::ApiConnection(e.to_string())
                }
            })?;
        response
            .text()
            .await
            .map_err(|e| easyrag_core::CoreError::ApiConnection(e.to_string()))
    }

    // ===== tool-calling support =====

    /// Convert agent-loop messages into Ollama chat format. User messages
    /// matching the `[Tool Result: name]` content pattern (see the
    /// TOOL_RESULT_PREFIX contract in agent.rs) become `{role: "tool",
    /// tool_name, content}` messages; the preceding assistant message gets
    /// a synthesized `tool_calls` entry with an incrementing `function.index`
    /// (Ollama pairs tool results by order/name, not by id). Parallel tool
    /// results from one turn all append to the same assistant message.
    fn build_tool_call_messages(messages: &[ChatMessage]) -> Vec<JsonValue> {
        let mut out: Vec<JsonValue> = Vec::with_capacity(messages.len());
        let mut counter = 0usize;
        for msg in messages {
            if msg.role == MessageRole::User {
                if let Some((tool_name, result)) = parse_tool_result_message(&msg.content) {
                    if let Some(prev) = out
                        .iter_mut()
                        .rev()
                        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
                    {
                        let entry = json!({
                            "type": "function",
                            "function": { "index": counter, "name": tool_name, "arguments": {} }
                        });
                        counter += 1;
                        match prev.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
                            Some(arr) => arr.push(entry),
                            None => {
                                if let Some(obj) = prev.as_object_mut() {
                                    obj.insert(
                                        "tool_calls".to_string(),
                                        JsonValue::Array(vec![entry]),
                                    );
                                }
                            }
                        }
                        out.push(json!({
                            "role": "tool",
                            "tool_name": tool_name,
                            "content": result,
                        }));
                        continue;
                    }
                    // Defensive: no preceding assistant message — fall
                    // through and send it as a plain user message.
                }
            }
            let role = match msg.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
            };
            out.push(json!({ "role": role, "content": msg.content }));
        }
        out
    }

    /// Assemble the /api/chat request body with tool definitions.
    fn build_chat_body_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
    ) -> JsonValue {
        json!({
            "model": self.config.model,
            "messages": Self::build_tool_call_messages(messages),
            "tools": serialize_tools(tools),
            "stream": false,
        })
    }

    /// Parse an /api/chat response that may contain tool calls. Ollama
    /// returns `arguments` as a JSON object (unlike OpenAI's string form).
    fn parse_tool_response(response: &str) -> LlmToolResponse {
        let Ok(json) = serde_json::from_str::<JsonValue>(response) else {
            return LlmToolResponse::default();
        };
        let message = json.get("message");
        // Empty content is normalized to None (tool-call responses carry "").
        let content = message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string());
        let tool_calls = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|call| {
                        let name = call.pointer("/function/name")?.as_str()?.to_string();
                        let arguments = call
                            .pointer("/function/arguments")
                            .map(arguments_to_map)
                            .unwrap_or_default();
                        Some(ToolCall {
                            id: None,
                            name,
                            arguments,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        LlmToolResponse {
            content,
            tool_calls,
        }
    }
}

#[async_trait::async_trait]
impl LlmService for OllamaClient {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        let body = self.build_chat_body(request, false);
        let response = self.post_json("/api/chat", &body).await?;
        Ok(serde_json::from_str::<JsonValue>(&response)
            .ok()
            .and_then(|json| {
                json.pointer("/message/content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default())
    }

    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
        // Ollama returns newline-delimited JSON (not SSE).
        let http = self.http.clone();
        let url = format!("{}/api/chat", self.config.base_url);
        let body = self.build_chat_body(&request, true);
        Box::pin(async_stream::stream! {
            let response = match http
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("Ollama stream connection failed: {e}");
                    return;
                }
            };
            let mut chunks = response.bytes_stream();
            let mut buffer = String::new();
            while let Some(chunk) = chunks.next().await {
                let Ok(bytes) = chunk else { break };
                buffer.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(pos) = buffer.find('\n') {
                    let line: String = buffer.drain(..=pos).collect();
                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<JsonValue>(line) {
                        Ok(json) => {
                            let content = json
                                .pointer("/message/content")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if !content.is_empty() {
                                yield content.to_string();
                            }
                        }
                        Err(_) => {
                            tracing::debug!("Failed to parse Ollama chunk: {line}");
                        }
                    }
                }
            }
            // Tail without trailing newline.
            let tail = buffer.trim();
            if !tail.is_empty() {
                if let Ok(json) = serde_json::from_str::<JsonValue>(tail) {
                    let content = json
                        .pointer("/message/content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if !content.is_empty() {
                        yield content.to_string();
                    }
                }
            }
        })
    }
}

#[async_trait::async_trait]
impl ToolCallingLlmService for OllamaClient {
    async fn complete_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> easyrag_core::Result<LlmToolResponse> {
        // No circuit breaker/retry here, consistent with `complete`.
        let body = self.build_chat_body_with_tools(&messages, &tools);
        let response = self.post_json("/api/chat", &body).await?;
        Ok(Self::parse_tool_response(&response))
    }
}

#[async_trait::async_trait]
impl EmbeddingService for OllamaClient {
    async fn embed(&self, texts: Vec<String>) -> easyrag_core::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({ "model": self.config.model, "input": texts });
        let response = self.post_json("/api/embed", &body).await?;
        let json: JsonValue = serde_json::from_str(&response)?;
        let embeddings = json.get("embeddings").and_then(|e| e.as_array());
        Ok(embeddings
            .map(|items| {
                items
                    .iter()
                    .map(|emb| {
                        emb.as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_f64())
                                    .map(|f| f as f32)
                                    .collect::<Vec<f32>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }
}
