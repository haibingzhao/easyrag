//! Ollama LLM client.

use easyrag_core::llm::service::{
    EmbeddingService, LlmRequest, LlmService, MessageRole,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

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
