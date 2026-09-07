//! easyrag-llm: LLM clients for EasyRAG.
//!
//! - [`openai`]: OpenAI-compatible client (chat / SSE streaming / vision /
//!   tool calling, circuit breakers + exponential-backoff retries)
//! - [`ollama`]: Ollama native client (chat / NDJSON streaming / embeddings /
//!   tool calling, with the same circuit breakers and retries since `[BC-4]`)
//! - [`rerank`]: Jina / Cohere / custom rerank client with graceful fallback
//! - [`transport`]: the HTTP layer the three clients share (client builder,
//!   POST flavours, streaming skeleton, `ChatWireFormat`, retry + breaker
//!   plumbing)
//! - [`cache`]: LLM response cache (moka, flattened `{mode}:{type}:{hash}` keys)
//! - [`roles`]: role-based LLM manager with per-role concurrency control
//!
//! Note: the module's `springai/SpringAiServices.kt` is a Spring AI
//! framework adapter with no project-internal references; the
//! OpenAI-compatible client covers that role on the Rust side.

pub mod cache;
pub mod ollama;
pub mod openai;
pub mod rerank;
pub mod roles;
pub mod transport;

use easyrag_core::llm::service::ToolDefinition;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;

pub use cache::LlmCache;
pub use ollama::{OllamaClient, OllamaConfig};
pub use openai::{OpenAiClient, OpenAiConfig};
pub use rerank::{RerankClient, RerankConfig, RerankProvider};
pub use roles::{LlmRole, RoleConfig, RoleLlmManager};
pub use transport::{
    build_chat_messages, build_http_client, execute_with_retry, f32_array, is_retryable_message,
    map_reqwest_error, role_str, stream_chat, ChatWireFormat, HttpEndpointConfig, LlmTransport,
    PostFlavour, RetryPolicy, StreamAuth,
};

// ===== Tool-calling support shared by the openai/ollama clients =====

/// Content-pattern contract with the pipeline agent loop: tool results are
/// encoded as User messages starting with this prefix (see agent.rs — do
/// NOT change the format there without updating the client parsers here).
pub(crate) const TOOL_RESULT_PREFIX: &str = "[Tool Result: ";

/// Parse a tool-result message encoded as `[Tool Result: name]\ncontent`,
/// returning `(tool_name, content)`. Returns `None` for any other content.
pub(crate) fn parse_tool_result_message(content: &str) -> Option<(&str, &str)> {
    let rest = content.strip_prefix(TOOL_RESULT_PREFIX)?;
    let end = rest.find("]\n")?;
    Some((&rest[..end], &rest[end + 2..]))
}

/// Serialize tool definitions into the OpenAI-style `tools` array. Both
/// the OpenAI-compatible and Ollama `/api/chat` endpoints use this schema.
pub(crate) fn serialize_tools(tools: &[ToolDefinition]) -> JsonValue {
    JsonValue::Array(
        tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": JsonValue::Object(tool.parameters.clone()),
                    }
                })
            })
            .collect(),
    )
}

/// Convert a JSON object of tool-call arguments into a string map. String
/// values are kept as-is; other scalars are stringified (`42` -> `"42"`).
/// Non-object input yields an empty map.
pub(crate) fn arguments_to_map(arguments: &JsonValue) -> HashMap<String, String> {
    let Some(map) = arguments.as_object() else {
        return HashMap::new();
    };
    map.iter()
        .map(|(key, value)| {
            let text = match value {
                JsonValue::String(s) => s.clone(),
                other => other.to_string(),
            };
            (key.clone(), text)
        })
        .collect()
}

/// Parse a JSON-encoded arguments string (OpenAI returns `arguments` as a
/// JSON string). Tolerates non-string scalar values by stringifying them;
/// unparseable input degrades to an empty map.
pub(crate) fn parse_arguments_json(arguments_json: &str) -> HashMap<String, String> {
    if arguments_json.trim().is_empty() {
        return HashMap::new();
    }
    match serde_json::from_str::<JsonValue>(arguments_json) {
        Ok(value) => arguments_to_map(&value),
        Err(e) => {
            tracing::warn!("Failed to parse tool call arguments JSON: {e}");
            HashMap::new()
        }
    }
}
