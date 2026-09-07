//! Ollama LLM client.

use easyrag_core::concurrency::CircuitBreaker;
use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, LlmToolResponse, MessageRole,
    ToolCall, ToolCallingLlmService, ToolDefinition,
};
use futures::stream::BoxStream;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

use crate::transport::{
    build_chat_messages, f32_array, role_str, stream_chat, ChatWireFormat, HttpEndpointConfig,
    LlmTransport, PostFlavour, RetryPolicy, StreamAuth,
};
use crate::{arguments_to_map, parse_tool_result_message, serialize_tools};

/// Embeddings endpoint (the chat endpoint is [`ChatWireFormat::CHAT_PATH`]).
const EMBED_PATH: &str = "/api/embed";

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
    /// Attempt budget for a call, `[BC-4]`. Defaults to the same budget the
    /// OpenAI client has always used, so both providers behave alike under a
    /// transient upstream failure.
    pub max_retries: u32,
    /// Base of the retry backoff in milliseconds, `[BC-4]`.
    pub retry_delay_base_ms: u64,
}

impl OllamaConfig {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "llama3".to_string(),
            timeout: Duration::from_secs(300),
            connect_timeout: Duration::from_secs(30),
            embedding_dimension: 4096,
            max_retries: 3,
            retry_delay_base_ms: 1000,
        }
    }

    /// The transport-relevant slice of this config (Phase 4.4).
    pub fn endpoint(&self) -> HttpEndpointConfig {
        HttpEndpointConfig::new(self.base_url.clone(), self.timeout, self.connect_timeout)
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
    transport: LlmTransport,
    /// `[BC-4]` Chat/tool calls share one breaker, exactly as the OpenAI client
    /// does: they are the same upstream dependency, so opening the breaker on
    /// one must protect the other.
    llm_breaker: CircuitBreaker,
    /// `[BC-4]` Embeddings get their own breaker so a saturated embedding model
    /// does not take chat down with it (and vice versa).
    embedding_breaker: CircuitBreaker,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> easyrag_core::Result<Self> {
        let transport = config.endpoint().build_transport()?;
        Ok(Self {
            // Same thresholds as the OpenAI client: 5 failures open the breaker
            // for 60s. Only the name differs, so log lines say which provider
            // and which call class tripped.
            llm_breaker: CircuitBreaker::new(5, Duration::from_secs(60), "llm-ollama"),
            embedding_breaker: CircuitBreaker::new(
                5,
                Duration::from_secs(60),
                "embedding-ollama",
            ),
            config,
            transport,
        })
    }

    /// Build the message list (system prompt + history + final user message).
    /// The final message is always the plain prompt here — Ollama has no vision
    /// path, so unlike the OpenAI client there is nothing to substitute.
    fn build_messages(request: &LlmRequest) -> Vec<JsonValue> {
        build_chat_messages(
            request,
            json!({ "role": "user", "content": request.prompt }),
        )
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

    // The /api/chat request body lives in the `ChatWireFormat` impl below, so
    // the shared streaming skeleton can build it too.

    /// Retry budget, `[BC-4]`. The curve is [`RetryPolicy`], shared with the
    /// OpenAI client: exponential with equal jitter. `[BC-3]` changed that curve
    /// once, for both providers, which is the point of having one definition.
    fn retry_policy(&self) -> RetryPolicy {
        RetryPolicy {
            max_retries: self.config.max_retries,
            delay_base_ms: self.config.retry_delay_base_ms,
        }
    }

    /// Ollama takes no credentials. `provider` only feeds the non-2xx error
    /// message, so it reads "Ollama API returned status 503: …" the same way the
    /// OpenAI client reads "OpenAI API returned status 503: …".
    fn post_flavour(&self) -> PostFlavour {
        PostFlavour::Anonymous { provider: "Ollama" }
    }

    /// Circuit breaker + retry, `[BC-4]`.
    ///
    /// This is the same [`LlmTransport::execute_resilient`] the OpenAI client
    /// calls; nothing here is Ollama-specific except the breaker instances and
    /// the credential-free flavour. That is the point of Phase 4.1 — adopting
    /// resilience for a second provider is wiring, not re-implementation.
    ///
    /// The streaming path deliberately does *not* go through here: a stream
    /// failure warns and ends the stream (see [`stream_chat`]), it does not
    /// return `Err`, so there is nothing for a retry loop to catch. Changing
    /// that would be a second, separate behaviour change.
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
            out.push(json!({ "role": role_str(msg.role), "content": msg.content }));
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

/// The Ollama half of the shared transport (Phase 4.1).
///
/// Request body, tool-call pairing and the embedding response shape stay
/// Ollama-specific; only the endpoint, auth style, framing and completion
/// pointer are expressed through the shared trait.
impl ChatWireFormat for OllamaClient {
    const CHAT_PATH: &'static str = "/api/chat";
    const CONTENT_POINTER: &'static str = "/message/content";
    const STREAM_FAILURE_LOG: &'static str = "Ollama stream connection failed";

    fn transport(&self) -> &LlmTransport {
        &self.transport
    }

    /// Ollama is a local daemon and takes no credentials, so the stream sets an
    /// explicit `Content-Type` and nothing else.
    fn stream_auth(&self) -> StreamAuth {
        StreamAuth::ContentType
    }

    /// Build the JSON body for the `/api/chat` endpoint. Sampling parameters go
    /// under `options` (`num_predict` for max tokens) rather than at the top
    /// level — precisely the shape this client does not share with OpenAI.
    fn build_chat_body(&self, request: &LlmRequest, stream: bool) -> JsonValue {
        json!({
            "model": self.config.model,
            "messages": Self::build_messages(request),
            "stream": stream,
            "options": Self::build_options(request),
        })
    }

    /// NDJSON framing: every non-blank line is a whole JSON document.
    ///
    /// `is_tail` matters here and nowhere else. An unparseable newline-
    /// terminated line is logged at `debug`; an unparseable unterminated
    /// remainder is dropped silently. That asymmetry predates Phase 4.1 and is
    /// preserved on purpose — a truncated final frame is the normal shape of a
    /// stream cut off mid-response, so logging it would be noise.
    fn parse_stream_line(line: &str, is_tail: bool) -> Option<String> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        match serde_json::from_str::<JsonValue>(line) {
            Ok(json) => {
                let content = json
                    .pointer(Self::CONTENT_POINTER)
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if content.is_empty() {
                    None
                } else {
                    Some(content.to_string())
                }
            }
            Err(_) => {
                if !is_tail {
                    tracing::debug!("Failed to parse Ollama chunk: {line}");
                }
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl LlmService for OllamaClient {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        let body = self.build_chat_body(request, false);
        // `[BC-4]` Retried and breaker-wrapped. Before this, a 503 came back as
        // `Ok("")` because the response body was parsed without a status check.
        self.execute_resilient(
            &self.llm_breaker,
            "Ollama",
            Self::CHAT_PATH,
            body,
            |response| Ok(Self::parse_completion(response)),
        )
        .await
    }

    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
        // Ollama returns newline-delimited JSON (not SSE). The POST /
        // warn-and-return / split-on-`\n` / tail skeleton is shared with the
        // OpenAI client; the framing difference lives in `parse_stream_line`.
        stream_chat(self, request)
    }
}

#[async_trait::async_trait]
impl ToolCallingLlmService for OllamaClient {
    async fn complete_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> easyrag_core::Result<LlmToolResponse> {
        // `[BC-4]` Shares the chat breaker, as the OpenAI client's tool path
        // does. The previous comment here said "no circuit breaker/retry,
        // consistent with `complete`" — that consistency was the defect.
        let body = self.build_chat_body_with_tools(&messages, &tools);
        self.execute_resilient(
            &self.llm_breaker,
            "Ollama-tools",
            Self::CHAT_PATH,
            body,
            |response| Ok(Self::parse_tool_response(response)),
        )
        .await
    }
}

#[async_trait::async_trait]
impl EmbeddingService for OllamaClient {
    /// `/api/embed` returns `embeddings[]` (a bare array of arrays), where the
    /// OpenAI-compatible `/embeddings` returns `data[].embedding`. Phase 4.2
    /// puts both on the same transport and shares the `f32` conversion
    /// ([`f32_array`]), but not the failure policy: on a *200* whose body has no
    /// `embeddings` array this still degrades to an empty result where the
    /// OpenAI client raises an error, and that difference is preserved because
    /// changing it would be a second behaviour change.
    ///
    /// `[BC-4]` does change the non-2xx case: a 503 used to arrive here as a
    /// body with no `embeddings` key and so returned `Ok(vec![])`, letting an
    /// ingestion run store nothing while reporting success. It is now retried,
    /// breaker-wrapped, and surfaced as an error.
    async fn embed(&self, texts: Vec<String>) -> easyrag_core::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = json!({ "model": self.config.model, "input": texts });
        self.execute_resilient(
            &self.embedding_breaker,
            "Ollama-embed",
            EMBED_PATH,
            body,
            parse_embed_response,
        )
        .await
    }

    fn embedding_dimension(&self) -> usize {
        self.config.embedding_dimension
    }
}

/// Parse an `/api/embed` response: `embeddings[]`, each an array of numbers.
///
/// Tolerant by design and unchanged by `[BC-4]` — a 200 with a missing or
/// malformed `embeddings` field yields an empty list rather than an error (see
/// [`EmbeddingService::embed`] above for why that tolerance is kept here and not
/// in the OpenAI client).
///
/// A free function so it can be handed to `execute_resilient` as a `fn` pointer.
fn parse_embed_response(response: &str) -> easyrag_core::Result<Vec<Vec<f32>>> {
    let json: JsonValue = serde_json::from_str(response)?;
    let embeddings = json.get("embeddings").and_then(|e| e.as_array());
    Ok(embeddings
        .map(|items| {
            items
                .iter()
                .map(|emb| emb.as_array().map(|arr| f32_array(arr)).unwrap_or_default())
                .collect()
        })
        .unwrap_or_default())
}
