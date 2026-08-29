//! easyrag-llm: LLM clients for EasyRAG.
//!
//! - [`openai`]: OpenAI-compatible client (chat / SSE streaming / vision /
//!   embeddings, circuit breakers + quadratic-backoff retries)
//! - [`ollama`]: Ollama native client (chat / NDJSON streaming / embeddings)
//! - [`rerank`]: Jina / Cohere / custom rerank client with graceful fallback
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

pub use cache::LlmCache;
pub use ollama::{OllamaClient, OllamaConfig};
pub use openai::{OpenAiClient, OpenAiConfig};
pub use rerank::{RerankClient, RerankConfig, RerankProvider};
pub use roles::{LlmRole, RoleConfig, RoleLlmManager};
