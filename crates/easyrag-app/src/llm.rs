//! `ServerConfig` → LLM client configuration (Phase 4.4).
//!
//! `main.rs` assembled the chat client field by field, `embedding.rs` assembled
//! the embedding client field by field, and `main.rs` additionally hand-mapped
//! the rerank provider string. Three hand-written assemblies reading the same
//! source struct is how a field ends up set on one path and forgotten on
//! another — the rerank/keyword asymmetry between `main.rs` and `workspace.rs`
//! (defect A-2, fixed in Phase 6e) is exactly that failure mode.
//!
//! The mapping cannot live in `easyrag-llm`: the dependency DAG is
//! `core ← llm ← … ← server ← app`, so the LLM crate must not know about
//! `ServerConfig`. It lives here in `app`, which depends on both, and is the
//! seed the Phase 6d assembly factory grows from.
//!
//! The chat and embedding mappings are deliberately *not* merged into one
//! function. The embedding client keeps the library defaults for `timeout`,
//! `connect_timeout` and `enable_thinking`; the chat client overrides the first
//! and last from config. Unifying them would silently change the embedding
//! client's request timeout, which an `[equivalent]` PR must not do.

use std::time::Duration;

use easyrag_llm::openai::OpenAiConfig;
use easyrag_llm::rerank::{RerankConfig, RerankProvider};
use easyrag_server::ServerConfig;

/// Stands in for a missing chat API key so client construction still succeeds
/// and the failure surfaces on the first call rather than at startup. The
/// accompanying warn stays in `main.rs`, where the rest of the startup
/// validation warnings are emitted.
const PLACEHOLDER_API_KEY: &str = "sk-placeholder";

/// Chat client config: base URL, model, request timeout and the reasoning-model
/// thinking toggle.
pub fn chat_config(config: &ServerConfig) -> OpenAiConfig {
    let api_key = if config.llm_api_key.is_empty() {
        PLACEHOLDER_API_KEY.to_string()
    } else {
        config.llm_api_key.clone()
    };
    let mut llm_config = OpenAiConfig::new(api_key);
    llm_config.base_url = config.llm_base_url.clone();
    llm_config.model = config.llm_model.clone();
    // Slow reasoning models (e.g. qwen3.8-max) can exceed the 120s default.
    llm_config.timeout = Duration::from_secs(config.llm_timeout_secs);
    // LLM_ENABLE_THINKING=false is what keeps reasoning models fast enough.
    llm_config.enable_thinking = config.llm_enable_thinking;
    llm_config
}

/// Embedding client config: a separate binding (key/URL/model) plus the vector
/// dimension the storage layer allocates for.
pub fn embedding_config(config: &ServerConfig) -> OpenAiConfig {
    // `config.embedding_api_key` already reflects EMBEDDING_BINDING_API_KEY
    // through the injected getter; no second env read here.
    let mut client_config = OpenAiConfig::new(config.embedding_api_key.clone());
    client_config.base_url = config.embedding_base_url.clone();
    client_config.model = config.embedding_model.clone();
    client_config.embedding_dimension = config.embedding_dim;
    client_config
}

/// Rerank client config, or `None` when reranking is disabled. The caller still
/// constructs the client, because `RerankClient::new` is what rejects a
/// `Custom` provider with no base URL.
pub fn rerank_config(config: &ServerConfig) -> Option<RerankConfig> {
    if !config.rerank_enabled {
        return None;
    }
    let mut rerank_config = RerankConfig::new(rerank_provider(&config.rerank_provider));
    rerank_config.api_key = config.rerank_api_key.clone();
    rerank_config.model = config.rerank_model.clone();
    Some(rerank_config)
}

/// Provider name → enum. Case-insensitive; anything unrecognized falls back to
/// Jina, which is both the `Default` and the previous behaviour.
pub fn rerank_provider(value: &str) -> RerankProvider {
    match value.to_lowercase().as_str() {
        "cohere" => RerankProvider::Cohere,
        "custom" => RerankProvider::Custom,
        _ => RerankProvider::Jina,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config with every LLM-relevant field set to a recognizable value, so a
    /// mapping that reads the wrong field (or forgets one) fails loudly. These
    /// are the assembly-drift guardrails the module exists to make possible.
    fn populated() -> ServerConfig {
        ServerConfig {
            llm_api_key: "chat-key".to_string(),
            llm_base_url: "https://chat.example/v1".to_string(),
            llm_model: "chat-model".to_string(),
            llm_timeout_secs: 42,
            llm_enable_thinking: Some(false),
            embedding_api_key: "embed-key".to_string(),
            embedding_base_url: "https://embed.example/v1".to_string(),
            embedding_model: "embed-model".to_string(),
            embedding_dim: 768,
            rerank_enabled: true,
            rerank_provider: "Cohere".to_string(),
            rerank_api_key: "rerank-key".to_string(),
            rerank_model: "rerank-model".to_string(),
            ..ServerConfig::default()
        }
    }

    #[test]
    fn chat_config_maps_every_chat_field() {
        let c = chat_config(&populated());
        assert_eq!("chat-key", c.api_key);
        assert_eq!("https://chat.example/v1", c.base_url);
        assert_eq!("chat-model", c.model);
        assert_eq!(Duration::from_secs(42), c.timeout);
        assert_eq!(Some(false), c.enable_thinking);
        // Not overridden from ServerConfig: the library defaults must survive.
        assert_eq!(Duration::from_secs(30), c.connect_timeout);
        assert_eq!(3, c.max_retries);
    }

    #[test]
    fn chat_config_substitutes_the_placeholder_for_a_missing_key() {
        let mut config = populated();
        config.llm_api_key = String::new();
        assert_eq!(PLACEHOLDER_API_KEY, chat_config(&config).api_key);
    }

    #[test]
    fn embedding_config_maps_the_embedding_binding_not_the_chat_one() {
        let c = embedding_config(&populated());
        assert_eq!("embed-key", c.api_key);
        assert_eq!("https://embed.example/v1", c.base_url);
        assert_eq!("embed-model", c.model);
        assert_eq!(768, c.embedding_dimension);
        // Deliberately untouched — merging the two mappings would start
        // applying `llm_timeout_secs` here and change embedding timeouts.
        assert_eq!(Duration::from_secs(120), c.timeout);
        assert_eq!(None, c.enable_thinking);
    }

    #[test]
    fn rerank_config_is_none_when_disabled_and_maps_key_and_model_when_on() {
        let mut config = populated();
        config.rerank_enabled = false;
        assert!(rerank_config(&config).is_none());

        let c = rerank_config(&populated()).expect("enabled must produce a config");
        assert_eq!(RerankProvider::Cohere, c.provider);
        assert_eq!("rerank-key", c.api_key);
        assert_eq!("rerank-model", c.model);
        // Base URL stays unset here: deriving it from the provider is
        // `RerankConfig::effective_base_url`'s job, not the mapping's.
        assert_eq!(None, c.base_url);
    }

    #[test]
    fn rerank_provider_is_case_insensitive_and_falls_back_to_jina() {
        assert_eq!(RerankProvider::Cohere, rerank_provider("cohere"));
        assert_eq!(RerankProvider::Cohere, rerank_provider("COHERE"));
        assert_eq!(RerankProvider::Custom, rerank_provider("custom"));
        assert_eq!(RerankProvider::Jina, rerank_provider("jina"));
        assert_eq!(RerankProvider::Jina, rerank_provider(""));
        assert_eq!(RerankProvider::Jina, rerank_provider("nonsense"));
    }
}
