//! LLM service traits and embedding function abstraction

pub mod embedding_func;
pub mod service;

pub use embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
pub use service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, LlmToolResponse, MessageRole,
    RerankResult, RerankService, ResponseFormat, ToolCall, ToolCallingLlmService, ToolDefinition,
};
