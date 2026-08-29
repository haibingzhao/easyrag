//! OpenAI-compatible HTTP client.
//! Supports OpenAI, Azure OpenAI, Aliyun Bailian, and any
//! OpenAI-compatible API.

use easyrag_core::concurrency::resilience::{BreakerError, CircuitBreaker};
use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, LlmToolResponse, MessageRole,
    ResponseFormat, ToolCall, ToolCallingLlmService, ToolDefinition,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

use crate::{parse_arguments_json, parse_tool_result_message, serialize_tools};

/// Configuration for [`OpenAiClient`].
#[derive(Clone, Debug)]
pub struct OpenAiConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_retries: u32,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Dimension of embeddings produced by the configured model.
    pub embedding_dimension: usize,
    /// DashScope-style thinking toggle for reasoning models (e.g. qwen3.8-max).
    /// None = send nothing (provider default); Some(false) disables the
    /// reasoning phase so long prompts don't blow past the request timeout.
    pub enable_thinking: Option<bool>,
    /// Base for the quadratic retry delay `base * attempt^2` ms
    ///
    pub retry_delay_base_ms: u64,
}

impl OpenAiConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4o".to_string(),
            max_retries: 3,
            timeout: Duration::from_secs(120),
            connect_timeout: Duration::from_secs(30),
            embedding_dimension: 1536,
            enable_thinking: None,
            retry_delay_base_ms: 1000,
        }
    }
}

/// OpenAI-compatible client implementing both chat completion
/// (incl. streaming and vision) and embeddings.
pub struct OpenAiClient {
    config: OpenAiConfig,
    http: reqwest::Client,
    /// Circuit breaker for LLM completion calls — opens after 5
    /// consecutive transient failures.
    llm_breaker: CircuitBreaker,
    /// Circuit breaker for embedding calls.
    embedding_breaker: CircuitBreaker,
}

impl OpenAiClient {
    pub fn new(config: OpenAiConfig) -> easyrag_core::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.timeout)
            .build()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("http client: {e}")))?;
        Ok(Self {
            llm_breaker: CircuitBreaker::new(5, Duration::from_secs(60), "llm-openai"),
            embedding_breaker: CircuitBreaker::new(5, Duration::from_secs(60), "embedding-openai"),
            config,
            http,
        })
    }

    // ===== request body builders (1:1 port) =====

    /// Build the message list (system prompt + history + final user message).
    fn build_messages(request: &LlmRequest, user_message: JsonValue) -> Vec<JsonValue> {
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
        messages.push(user_message);
        messages
    }

    /// Apply shared completion parameters (temperature, max tokens,
    /// response format)..
    fn apply_common_params(
        body: &mut serde_json::Map<String, JsonValue>,
        request: &LlmRequest,
        config: &OpenAiConfig,
    ) {
        if request.temperature > 0.0 {
            body.insert("temperature".to_string(), json!(request.temperature));
        }
        if let Some(max_tokens) = request.max_tokens {
            body.insert("max_tokens".to_string(), json!(max_tokens));
        }
        if request.response_format == Some(ResponseFormat::JsonObject) {
            body.insert(
                "response_format".to_string(),
                json!({ "type": "json_object" }),
            );
        }
        // Only send the field when explicitly configured — OpenAI's API
        // rejects unknown params in strict mode.
        if let Some(enable) = config.enable_thinking {
            body.insert("enable_thinking".to_string(), json!(enable));
        }
    }

    fn build_request_body(&self, request: &LlmRequest, stream: bool) -> JsonValue {
        let user_message = json!({ "role": "user", "content": request.prompt });
        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), json!(self.config.model));
        body.insert(
            "messages".to_string(),
            JsonValue::Array(Self::build_messages(request, user_message)),
        );
        body.insert("stream".to_string(), json!(stream));
        Self::apply_common_params(&mut body, request, &self.config);
        JsonValue::Object(body)
    }

    /// Build request body for vision (multimodal) completion with image
    /// content. Uses OpenAI's content array format with text + image_url
    /// parts.
    fn build_vision_body(
        &self,
        request: &LlmRequest,
        image_base64: &str,
        image_mime_type: &str,
    ) -> JsonValue {
        let content_parts = json!([
            { "type": "text", "text": request.prompt },
            {
                "type": "image_url",
                "image_url": { "url": format!("data:{image_mime_type};base64,{image_base64}") }
            }
        ]);
        let user_message = json!({ "role": "user", "content": content_parts });
        let mut body = serde_json::Map::new();
        body.insert("model".to_string(), json!(self.config.model));
        body.insert(
            "messages".to_string(),
            JsonValue::Array(Self::build_messages(request, user_message)),
        );
        body.insert("stream".to_string(), json!(false));
        Self::apply_common_params(&mut body, request, &self.config);
        JsonValue::Object(body)
    }

    // ===== transport helpers =====

    /// POST a JSON body; non-2xx responses become `ApiStatus` errors whose
    /// message embeds the status code (so the keyword-based retry matcher
    /// can see "502"/"503" etc.).
    async fn post_json(&self, path: &str, body: &JsonValue) -> easyrag_core::Result<String> {
        let url = format!("{}{}", self.config.base_url, path);
        let response = self
            .http
            .post(&url)
            .bearer_auth(&self.config.api_key)
            .json(body)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let status = response.status();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            return Err(easyrag_core::CoreError::api_status(
                status.as_u16(),
                format!(
                    "OpenAI API returned status {}: {}",
                    status.as_u16(),
                    truncate(&text, 500)
                ),
                None,
            ));
        }
        Ok(text)
    }

    /// Execute a request with retry-on-transient-failure semantics
    ///: delay `retry_delay_base_ms * attempt^2`.
    async fn execute_with_retry<T, F, Fut>(
        &self,
        request_name: &str,
        mut block: F,
    ) -> easyrag_core::Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = easyrag_core::Result<T>>,
    {
        let max = self.config.max_retries.max(1) as u64;
        let mut last_error: Option<easyrag_core::CoreError> = None;
        for attempt in 1..=max {
            match block().await {
                Ok(value) => return Ok(value),
                Err(e) => {
                    last_error = Some(e);
                    let err = last_error.as_ref().unwrap();
                    if is_retryable_message(&err.to_string()) && attempt < max {
                        let delay = self.config.retry_delay_base_ms * attempt * attempt;
                        tracing::warn!(
                            "{request_name} request failed (attempt {attempt}/{max}), \
                             retrying in {delay}ms: {err}"
                        );
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                    } else {
                        break;
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            easyrag_core::CoreError::Internal(format!("{request_name} request failed"))
        }))
    }

    /// Parse `choices[0].message.content` from a chat completion response.
    fn parse_completion(response: &str) -> String {
        serde_json::from_str::<JsonValue>(response)
            .ok()
            .and_then(|json| {
                json.pointer("/choices/0/message/content")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default()
    }

    // ===== tool-calling support =====

    /// Convert agent-loop messages into OpenAI chat format. User messages
    /// matching the `[Tool Result: name]` content pattern (see the
    /// TOOL_RESULT_PREFIX contract in agent.rs) become `role: "tool"`
    /// messages with synthesized `tool_call_id`s; the preceding assistant
    /// message gets a matching `tool_calls` entry. Parallel tool results
    /// from one turn all append to the same assistant message.
    fn build_tool_call_messages(messages: &[ChatMessage]) -> Vec<JsonValue> {
        let mut out: Vec<JsonValue> = Vec::with_capacity(messages.len());
        let mut counter = 0usize;
        for msg in messages {
            if msg.role == MessageRole::User {
                if let Some((tool_name, result)) = parse_tool_result_message(&msg.content) {
                    if let Some(prev) = out.iter_mut().rev().find(|m| {
                        m.get("role").and_then(|r| r.as_str()) == Some("assistant")
                    }) {
                        let tool_call_id = format!("call_{counter}");
                        counter += 1;
                        let entry = json!({
                            "id": tool_call_id,
                            "type": "function",
                            "function": { "name": tool_name, "arguments": "{}" }
                        });
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
                            "tool_call_id": tool_call_id,
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

    /// Assemble the full tool-calling request body.
    fn build_tool_call_body(&self, messages: &[ChatMessage], tools: &[ToolDefinition]) -> JsonValue {
        let mut body = json!({
            "model": self.config.model,
            "messages": Self::build_tool_call_messages(messages),
            "tools": serialize_tools(tools),
            "stream": false,
        });
        // Reasoning models must skip the thinking phase here too — entity
        // extraction prompts are long and would otherwise time out.
        if let Some(enable) = self.config.enable_thinking {
            body["enable_thinking"] = json!(enable);
        }
        body
    }

    /// Parse a chat completion response that may contain tool calls.
    fn parse_tool_response(response: &str) -> LlmToolResponse {
        let Ok(json) = serde_json::from_str::<JsonValue>(response) else {
            return LlmToolResponse::default();
        };
        let message = json.pointer("/choices/0/message");
        let content = message
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());
        let tool_calls = message
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|call| {
                        let name = call.pointer("/function/name")?.as_str()?.to_string();
                        let id = call
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        let arguments = call
                            .pointer("/function/arguments")
                            .and_then(|a| a.as_str())
                            .map(parse_arguments_json)
                            .unwrap_or_default();
                        Some(ToolCall {
                            id,
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

///: message contains rate limit / timeout /
/// connection / 503 / 502.
fn is_retryable_message(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("rate limit")
        || msg.contains("timeout")
        || msg.contains("connection")
        || msg.contains("503")
        || msg.contains("502")
}

fn map_reqwest_error(e: reqwest::Error) -> easyrag_core::CoreError {
    if e.is_timeout() {
        easyrag_core::CoreError::ApiTimeout
    } else {
        easyrag_core::CoreError::ApiConnection(e.to_string())
    }
}

/// Char-boundary-safe truncation for error messages.
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

/// SSE line handling (: only lines with the exact `"data: "` prefix;
/// `[DONE]` and empty payloads are skipped; parse failures are logged).
fn strip_data_prefix(line: &str) -> Option<&str> {
    line.strip_prefix("data: ").map(|d| d.trim())
}

fn parse_sse_chunk(data: &str) -> Option<String> {
    if data == "[DONE]" || data.is_empty() {
        return None;
    }
    match serde_json::from_str::<JsonValue>(data) {
        Ok(json) => {
            let content = json
                .pointer("/choices/0/delta/content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            }
        }
        Err(_) => {
            tracing::debug!("Failed to parse SSE chunk: {data}");
            None
        }
    }
}

fn breaker_result<T>(
    result: Result<T, BreakerError<easyrag_core::CoreError>>,
) -> easyrag_core::Result<T> {
    match result {
        Ok(v) => Ok(v),
        Err(BreakerError::Open(msg)) => Err(easyrag_core::CoreError::CircuitBreakerOpen(msg)),
        Err(BreakerError::Inner(e)) => Err(e),
    }
}

#[async_trait::async_trait]
impl LlmService for OpenAiClient {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        let body = self.build_request_body(request, false);
        let result = self
            .llm_breaker
            .execute(|| {
                let body = body.clone();
                async move {
                    self.execute_with_retry("OpenAI", || {
                        let body = body.clone();
                        async move {
                            let response = self.post_json("/chat/completions", &body).await?;
                            Ok(Self::parse_completion(&response))
                        }
                    })
                    .await
                }
            })
            .await;
        breaker_result(result)
    }

    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
        let http = self.http.clone();
        let url = format!("{}/chat/completions", self.config.base_url);
        let api_key = self.config.api_key.clone();
        let body = self.build_request_body(&request, true);
        Box::pin(async_stream::stream! {
            let response = match http
                .post(&url)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("SSE connection failed: {e}");
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
                    let line = line.trim_end_matches(['\r', '\n']);
                    if let Some(data) = strip_data_prefix(line) {
                        if let Some(content) = parse_sse_chunk(data) {
                            yield content;
                        }
                    }
                }
            }
            // Tail without trailing newline.
            let tail = buffer.trim_end_matches(['\r', '\n']);
            if let Some(data) = strip_data_prefix(tail) {
                if let Some(content) = parse_sse_chunk(data) {
                    yield content;
                }
            }
        })
    }

    async fn complete_with_image(
        &self,
        request: &LlmRequest,
        image_base64: &str,
        image_mime_type: &str,
    ) -> easyrag_core::Result<String> {
        // vision is retried but not wrapped by the circuit breaker.
        let body = self.build_vision_body(request, image_base64, image_mime_type);
        self.execute_with_retry("Vision", || {
            let body = body.clone();
            async move {
                let response = self.post_json("/chat/completions", &body).await?;
                Ok(Self::parse_completion(&response))
            }
        })
        .await
    }
}

#[async_trait::async_trait]
impl ToolCallingLlmService for OpenAiClient {
    async fn complete_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> easyrag_core::Result<LlmToolResponse> {
        let body = self.build_tool_call_body(&messages, &tools);
        let result = self
            .llm_breaker
            .execute(|| {
                let body = body.clone();
                async move {
                    self.execute_with_retry("OpenAI-tools", || {
                        let body = body.clone();
                        async move {
                            let response = self.post_json("/chat/completions", &body).await?;
                            Ok(Self::parse_tool_response(&response))
                        }
                    })
                    .await
                }
            })
            .await;
        breaker_result(result)
    }
}

#[async_trait::async_trait]
impl EmbeddingService for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> easyrag_core::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({ "model": self.config.model, "input": texts });
        let result = self
            .embedding_breaker
            .execute(|| {
                let body = body.clone();
                async move {
                    self.execute_with_retry("Embedding", || {
                        let body = body.clone();
                        async move {
                            let response = self.post_json("/embeddings", &body).await?;
                            let json: JsonValue = serde_json::from_str(&response)?;
                            let data = json.get("data").and_then(|d| d.as_array());
                            match data {
                                Some(items) if !items.is_empty() => items
                                    .iter()
                                    .map(|item| {
                                        item.get("embedding")
                                            .and_then(|e| e.as_array())
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|v| v.as_f64())
                                                    .map(|f| f as f32)
                                                    .collect::<Vec<f32>>()
                                            })
                                            .ok_or_else(|| {
                                                easyrag_core::CoreError::Internal(
                                                    "embedding item missing 'embedding' array"
                                                        .to_string(),
                                                )
                                            })
                                    })
                                    .collect(),
                                _ => Err(easyrag_core::CoreError::Internal(format!(
                                    "Embedding API returned empty data: {}",
                                    truncate(&response, 200)
                                ))),
                            }
                        }
                    })
                    .await
                }
            })
            .await;
        breaker_result(result)
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }
}
