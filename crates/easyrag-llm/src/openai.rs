//! OpenAI-compatible HTTP client.
//! Supports OpenAI, Azure OpenAI, Aliyun Bailian, and any
//! OpenAI-compatible API.

use easyrag_core::concurrency::resilience::CircuitBreaker;
use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, LlmToolResponse, MessageRole,
    ResponseFormat, ToolCall, ToolCallingLlmService, ToolDefinition,
};
use easyrag_core::util::text::truncate_with_ellipsis;
use futures::stream::BoxStream;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

use crate::transport::{
    build_chat_messages, execute_with_retry, f32_array, role_str, stream_chat, ChatWireFormat,
    HttpEndpointConfig, LlmTransport, PostFlavour, RetryPolicy, StreamAuth,
};
use crate::{parse_arguments_json, parse_tool_result_message, serialize_tools};

/// Embeddings endpoint (the chat endpoint is [`ChatWireFormat::CHAT_PATH`]).
/// OpenAI-compatible servers expose both under the same base URL.
const EMBEDDINGS_PATH: &str = "/embeddings";

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
    /// Base for the retry backoff, in milliseconds. The curve is exponential
    /// with equal jitter — see [`crate::transport::RetryPolicy`], and `[BC-3]`
    /// for the change away from the original quadratic `base * attempt^2`.
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

    /// The transport-relevant slice of this config (Phase 4.4). Feeding this to
    /// [`HttpEndpointConfig::build_transport`] is what replaced the hand-written
    /// `reqwest::Client::builder()` block that all three clients duplicated.
    pub fn endpoint(&self) -> HttpEndpointConfig {
        HttpEndpointConfig::new(self.base_url.clone(), self.timeout, self.connect_timeout)
    }
}

/// OpenAI-compatible client implementing both chat completion
/// (incl. streaming and vision) and embeddings.
pub struct OpenAiClient {
    config: OpenAiConfig,
    transport: LlmTransport,
    /// Circuit breaker for LLM completion calls — opens after 5
    /// consecutive transient failures.
    llm_breaker: CircuitBreaker,
    /// Circuit breaker for embedding calls.
    embedding_breaker: CircuitBreaker,
}

impl OpenAiClient {
    pub fn new(config: OpenAiConfig) -> easyrag_core::Result<Self> {
        let transport = config.endpoint().build_transport()?;
        Ok(Self {
            llm_breaker: CircuitBreaker::new(5, Duration::from_secs(60), "llm-openai"),
            embedding_breaker: CircuitBreaker::new(5, Duration::from_secs(60), "embedding-openai"),
            config,
            transport,
        })
    }

    // ===== request body builders (1:1 port) =====
    //
    // The system+history prefix comes from the shared `build_chat_messages`;
    // only the final user message differs per caller (plain string for chat,
    // content array for vision).

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
            JsonValue::Array(build_chat_messages(request, user_message)),
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
            JsonValue::Array(build_chat_messages(request, user_message)),
        );
        body.insert("stream".to_string(), json!(false));
        Self::apply_common_params(&mut body, request, &self.config);
        JsonValue::Object(body)
    }

    // ===== transport helpers =====

    /// POST a JSON body through the shared transport (bearer auth, status
    /// checked). Non-2xx responses become `ApiStatus` errors whose message
    /// embeds the status code, so the keyword-based retry matcher below can
    /// still see "502"/"503".
    async fn post_json(&self, path: &str, body: &JsonValue) -> easyrag_core::Result<String> {
        self.transport
            .post_bearer(path, &self.config.api_key, body, "OpenAI")
            .await
    }

    /// Retry budget for every OpenAI call, read from the two config fields
    /// `openai.rs` has always used. The curve itself is [`RetryPolicy`]:
    /// exponential (`base * 2^(attempt-1)`, capped at 60s) with equal jitter,
    /// which is what `[BC-3]` changed it to from the original quadratic
    /// `base * attempt^2`.
    fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy {
            max_retries: self.config.max_retries,
            delay_base_ms: self.config.retry_delay_base_ms,
        }
    }

    /// Auth flavour handed to the shared transport's resilient path.
    fn post_flavour(&self) -> PostFlavour {
        PostFlavour::Bearer {
            api_key: self.config.api_key.clone(),
            provider: "OpenAI",
        }
    }

    /// Circuit breaker + retry in one call (Phase 4.3).
    ///
    /// The nesting used to be spelled out three times here (`complete`,
    /// `complete_with_tools`, `embed`); it now lives once in
    /// [`LlmTransport::execute_resilient`] and this is a thin delegation, so a
    /// second provider can adopt the same resilience without re-deriving it.
    ///
    /// `parse` is a bare `fn` pointer rather than a closure: it captures
    /// nothing, so this stays monomorphized with no allocation per attempt.
    ///
    /// Vision (`complete_with_image`) deliberately does NOT route through here —
    /// it was retried but never wrapped by the breaker, and that asymmetry is
    /// preserved.
    async fn execute_resilient<T>(
        &self,
        breaker: &CircuitBreaker,
        request_name: &str,
        path: &str,
        body: JsonValue,
        parse: fn(&str) -> easyrag_core::Result<T>,
    ) -> easyrag_core::Result<T> {
        self.transport
            .execute_resilient(
                &self.post_flavour(),
                breaker,
                &self.retry_policy(),
                request_name,
                path,
                body,
                parse,
            )
            .await
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
            out.push(json!({ "role": role_str(msg.role), "content": msg.content }));
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

/// Parse an `/embeddings` response into vectors.
///
/// Unlike Ollama's `/api/embed`, this treats a missing `data` array, an empty
/// one, or an item with no `embedding` array as a hard error rather than
/// silently yielding nothing: an ingestion run that stores empty vectors
/// corrupts the index without ever reporting a failure, so failing loudly is
/// the point. The two providers' differing tolerance is therefore kept (Phase
/// 4.2 shares only the transport and the `f32` conversion, not the policy).
///
/// A free function so it can be handed to `execute_resilient` as a `fn` pointer.
fn parse_embedding_response(response: &str) -> easyrag_core::Result<Vec<Vec<f32>>> {
    let json: JsonValue = serde_json::from_str(response)?;
    let data = json.get("data").and_then(|d| d.as_array());
    match data {
        Some(items) if !items.is_empty() => items
            .iter()
            .map(|item| {
                item.get("embedding")
                    .and_then(|e| e.as_array())
                    .map(|arr| f32_array(arr))
                    .ok_or_else(|| {
                        easyrag_core::CoreError::Internal(
                            "embedding item missing 'embedding' array".to_string(),
                        )
                    })
            })
            .collect(),
        _ => Err(easyrag_core::CoreError::Internal(format!(
            "Embedding API returned empty data: {}",
            truncate_with_ellipsis(response, 200)
        ))),
    }
}

/// The OpenAI-compatible half of the shared transport (Phase 4.1).
///
/// Only framing and pointers live here; the request body, tool-call pairing and
/// vision shape stay on `OpenAiClient` because they are genuinely
/// provider-specific.
impl ChatWireFormat for OpenAiClient {
    const CHAT_PATH: &'static str = "/chat/completions";
    const CONTENT_POINTER: &'static str = "/choices/0/message/content";
    const STREAM_FAILURE_LOG: &'static str = "SSE connection failed";

    fn transport(&self) -> &LlmTransport {
        &self.transport
    }

    fn stream_auth(&self) -> StreamAuth {
        StreamAuth::Bearer(self.config.api_key.clone())
    }

    fn build_chat_body(&self, request: &LlmRequest, stream: bool) -> JsonValue {
        self.build_request_body(request, stream)
    }

    /// SSE framing. `is_tail` is ignored: the unterminated remainder goes
    /// through exactly the same handling as a newline-terminated line, which is
    /// what this client did before Phase 4.1.
    fn parse_stream_line(line: &str, _is_tail: bool) -> Option<String> {
        let line = line.trim_end_matches(['\r', '\n']);
        strip_data_prefix(line).and_then(parse_sse_chunk)
    }
}

#[async_trait::async_trait]
impl LlmService for OpenAiClient {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        let body = self.build_request_body(request, false);
        self.execute_resilient(
            &self.llm_breaker,
            "OpenAI",
            Self::CHAT_PATH,
            body,
            |response| Ok(Self::parse_completion(response)),
        )
        .await
    }

    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
        // The SSE skeleton (POST → warn-and-return on connect failure →
        // `bytes_stream()` split on `\n` → unterminated tail) is shared with
        // Ollama; only `parse_stream_line` above is OpenAI-specific.
        stream_chat(self, request)
    }

    async fn complete_with_image(
        &self,
        request: &LlmRequest,
        image_base64: &str,
        image_mime_type: &str,
    ) -> easyrag_core::Result<String> {
        // vision is retried but not wrapped by the circuit breaker.
        let body = self.build_vision_body(request, image_base64, image_mime_type);
        execute_with_retry(&self.retry_policy(), "Vision", || {
            let body = body.clone();
            async move {
                let response = self.post_json(Self::CHAT_PATH, &body).await?;
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
        // Shares the completion breaker: a tool call is the same upstream
        // dependency, and opening the breaker on one must protect the other.
        self.execute_resilient(
            &self.llm_breaker,
            "OpenAI-tools",
            Self::CHAT_PATH,
            body,
            |response| Ok(Self::parse_tool_response(response)),
        )
        .await
    }
}

#[async_trait::async_trait]
impl EmbeddingService for OpenAiClient {
    async fn embed(&self, texts: Vec<String>) -> easyrag_core::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({ "model": self.config.model, "input": texts });
        self.execute_resilient(
            &self.embedding_breaker,
            "Embedding",
            EMBEDDINGS_PATH,
            body,
            parse_embedding_response,
        )
        .await
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }
}
