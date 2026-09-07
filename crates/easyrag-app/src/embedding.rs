//! Embedding adapter: OpenAI-compatible client wrapped with the asymmetric
//! QUERY/DOCUMENT prefixes.

use std::sync::Arc;

use async_trait::async_trait;

use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_llm::openai::OpenAiClient;
use easyrag_server::ServerConfig;

/// Wraps an OpenAI-compatible embedding client, applying the asymmetric
/// QUERY/DOCUMENT prefixes configured in `ServerConfig`.
struct PrefixedEmbedding {
    client: Arc<OpenAiClient>,
    query_prefix: String,
    document_prefix: String,
}

#[async_trait]
impl EmbeddingFunction for PrefixedEmbedding {
    async fn embed(
        &self,
        texts: Vec<String>,
        context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        let prefixed: Vec<String> = match context {
            Some(EmbeddingContext::Query) if !self.query_prefix.is_empty() => texts
                .into_iter()
                .map(|t| format!("{}{}", self.query_prefix, t))
                .collect(),
            Some(EmbeddingContext::Document) if !self.document_prefix.is_empty() => texts
                .into_iter()
                .map(|t| format!("{}{}", self.document_prefix, t))
                .collect(),
            _ => texts,
        };
        use easyrag_core::llm::service::EmbeddingService;
        self.client.embed(prefixed).await
    }
}

/// Build the embedding function bean. Stateless,
/// so callers may construct multiple identical instances.
pub fn make_embedding_func(config: &ServerConfig) -> anyhow::Result<EmbeddingFunc> {
    // The `ServerConfig` → `OpenAiConfig` mapping is shared with the chat client
    // in `crate::llm`, so the two bindings cannot drift apart silently.
    let client_config = crate::llm::embedding_config(config);
    let mut func = EmbeddingFunc::new(
        config.embedding_dim,
        Arc::new(PrefixedEmbedding {
            client: Arc::new(OpenAiClient::new(client_config)?),
            query_prefix: config.embedding_query_prefix.clone(),
            document_prefix: config.embedding_document_prefix.clone(),
        }),
    );
    func.supports_asymmetric = !config.embedding_query_prefix.is_empty()
        || !config.embedding_document_prefix.is_empty();
    func.model_name = Some(config.embedding_model.clone());
    Ok(func)
}
