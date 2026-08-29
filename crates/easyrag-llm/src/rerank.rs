//! Rerank client supporting Jina, Cohere, and custom rerank APIs

use easyrag_core::llm::service::{RerankResult, RerankService};
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

/// Supported rerank providers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RerankProvider {
    #[default]
    Jina,
    Cohere,
    Custom,
}

/// Configuration for [`RerankClient`].
#[derive(Clone, Debug)]
pub struct RerankConfig {
    pub provider: RerankProvider,
    pub api_key: String,
    pub model: String,
    /// Override base URL; required for `Custom`.
    pub base_url: Option<String>,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

impl RerankConfig {
    pub fn new(provider: RerankProvider) -> Self {
        Self {
            provider,
            api_key: String::new(),
            model: "jina-reranker-v2-base-multilingual".to_string(),
            base_url: None,
            timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
        }
    }
}

/// HTTP rerank client with graceful degradation: on any failure the
/// original document order is returned with slightly decaying scores
///
pub struct RerankClient {
    config: RerankConfig,
    http: reqwest::Client,
    effective_base_url: String,
}

impl RerankClient {
    pub fn new(config: RerankConfig) -> easyrag_core::Result<Self> {
        let effective_base_url = match &config.base_url {
            Some(url) => url.clone(),
            None => match config.provider {
                RerankProvider::Jina => "https://api.jina.ai/v1".to_string(),
                RerankProvider::Cohere => "https://api.cohere.ai/v1".to_string(),
                RerankProvider::Custom => {
                    return Err(easyrag_core::CoreError::InvalidArgument(
                        "Custom provider requires baseUrl".to_string(),
                    ))
                }
            },
        };
        let http = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .timeout(config.timeout)
            .build()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("http client: {e}")))?;
        Ok(Self {
            config,
            http,
            effective_base_url,
        })
    }

    fn build_request_body(&self, query: &str, documents: &[String], top_n: Option<usize>) -> JsonValue {
        json!({
            "model": self.config.model,
            "query": query,
            "documents": documents,
            "top_n": top_n.unwrap_or(documents.len()),
        })
    }

    fn parse_response(response: &str, documents: &[String]) -> Vec<RerankResult> {
        let mut results = serde_json::from_str::<JsonValue>(response)
            .ok()
            .and_then(|json| json.get("results").cloned())
            .and_then(|results| results.as_array().cloned())
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        let index = item.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                        let score = item
                            .get("relevance_score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as f32;
                        RerankResult {
                            index,
                            score,
                            document: documents.get(index).cloned().unwrap_or_default(),
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        // Stable descending sort by score.
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Fallback preserving original order with decaying scores
    ///
    fn fallback_order(documents: &[String], top_n: Option<usize>) -> Vec<RerankResult> {
        let all = documents
            .iter()
            .enumerate()
            .map(|(index, doc)| RerankResult {
                index,
                score: 1.0 - (index as f32 * 0.01),
                document: doc.clone(),
            })
            .collect::<Vec<_>>();
        match top_n {
            Some(n) => all.into_iter().take(n).collect(),
            None => all,
        }
    }
}

#[async_trait::async_trait]
impl RerankService for RerankClient {
    async fn rerank(
        &self,
        query: &str,
        documents: Vec<String>,
        top_n: Option<usize>,
    ) -> easyrag_core::Result<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.build_request_body(query, &documents, top_n);
        let outcome = async {
            let mut request = self
                .http
                .post(format!("{}/rerank", self.effective_base_url))
                .header("Content-Type", "application/json")
                .json(&body);
            if !self.config.api_key.is_empty() {
                request = request.header("Authorization", format!("Bearer {}", self.config.api_key));
            }
            let response = request.send().await?;
            response.text().await
        }
        .await;
        match outcome {
            Ok(text) => Ok(Self::parse_response(&text, &documents)),
            Err(e) => {
                // Graceful degradation: return original order.
                tracing::warn!("Rerank failed, returning original order: {e}");
                Ok(Self::fallback_order(&documents, top_n))
            }
        }
    }
}
