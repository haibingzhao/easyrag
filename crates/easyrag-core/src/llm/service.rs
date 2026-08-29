//! LLM service interfaces.

use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Message role in a conversation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// A chat message in a conversation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

/// Response format for structured output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
}

/// LLM completion request parameters.
#[derive(Clone, Debug)]
pub struct LlmRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub history: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub response_format: Option<ResponseFormat>,
}

impl LlmRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            history: Vec::new(),
            temperature: 0.0,
            max_tokens: None,
            response_format: None,
        }
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    pub fn with_history(mut self, history: Vec<ChatMessage>) -> Self {
        self.history = history;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
        self
    }
}

/// Core LLM service interface.
/// Supports both awaited completion and streaming.
#[async_trait]
pub trait LlmService: Send + Sync {
    /// Complete a prompt and return the full response.
    async fn complete(&self, request: &LlmRequest) -> crate::Result<String>;

    /// Complete a prompt with streaming response (stream of string chunks).
    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String>;

    /// Simple completion with just a prompt string.
    async fn complete_prompt(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
    ) -> crate::Result<String> {
        let mut request = LlmRequest::new(prompt.to_string());
        if let Some(sp) = system_prompt {
            request.system_prompt = Some(sp.to_string());
        }
        self.complete(&request).await
    }

    /// Complete a prompt with an image (vision multimodal completion).
    /// Default implementation returns Unsupported; vision-capable
    /// implementations should override this method.
    async fn complete_with_image(
        &self,
        _request: &LlmRequest,
        _image_base64: &str,
        _image_mime_type: &str,
    ) -> crate::Result<String> {
        Err(crate::CoreError::Unsupported(
            "Vision completion not supported by this LLM service".to_string(),
        ))
    }
}

/// Embedding service interface. Converts texts to vector embeddings.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Embed a list of texts and return their vector representations.
    async fn embed(&self, texts: Vec<String>) -> crate::Result<Vec<Vec<f32>>>;

    /// Embed a single text.
    async fn embed_one(&self, text: &str) -> crate::Result<Vec<f32>> {
        let mut results = self.embed(vec![text.to_string()]).await?;
        results.pop().ok_or_else(|| {
            crate::CoreError::Internal("embedding service returned no vector".to_string())
        })
    }

    /// The dimension of embeddings produced by this service.
    fn embedding_dimension(&self) -> usize;
}

/// Rerank result for a single document.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub score: f32,
    pub document: String,
}

/// Rerank service interface. Re-ranks documents by relevance to a query.
#[async_trait]
pub trait RerankService: Send + Sync {
    /// Rerank documents by relevance to the query.
    /// Returns reranked results sorted by relevance score (descending).
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_n: Option<usize>,
    ) -> crate::Result<Vec<RerankResult>>;
}

// ===== Tool-Calling Support (for Agentic retrieval) =====

/// Definition of a tool that the LLM can invoke.
#[derive(Clone, Debug)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: crate::JsonMap,
}

/// A tool call requested by the LLM.
#[derive(Clone, Debug)]
pub struct ToolCall {
    /// Provider-returned tool_call_id (e.g. OpenAI's `call_...`). `None`
    /// for providers that don't assign IDs (Ollama pairs by order/name).
    pub id: Option<String>,
    pub name: String,
    pub arguments: std::collections::HashMap<String, String>,
}

/// LLM response that may contain tool calls.
#[derive(Clone, Debug, Default)]
pub struct LlmToolResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

impl LlmToolResponse {
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Extended LLM service interface with tool-calling support.
/// Implementations that don't support tools should return empty tool_calls.
#[async_trait]
pub trait ToolCallingLlmService: LlmService {
    /// Complete with tool definitions. The LLM may respond with tool calls
    /// instead of content.
    async fn complete_with_tools(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<ToolDefinition>,
    ) -> crate::Result<LlmToolResponse>;
}
