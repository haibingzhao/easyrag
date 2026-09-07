//! Rerank client tests: golden request body, response parsing/sorting,
//! graceful degradation on failure, provider configuration.

use easyrag_core::llm::service::RerankService;
use easyrag_llm::{RerankClient, RerankConfig, RerankProvider};
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn start_client(server: &MockServer) -> RerankClient {
    let mut config = RerankConfig::new(RerankProvider::Custom);
    config.base_url = Some(server.uri());
    config.api_key = "rk-key".to_string();
    config.model = "test-reranker".to_string();
    RerankClient::new(config).unwrap_or_else(|e| panic!("client: {e}"))
}

async fn last_request(server: &MockServer) -> (Value, Option<String>) {
    let requests = server.received_requests().await.unwrap();
    let last = requests.last().expect("at least one request");
    let body: Value = serde_json::from_slice(&last.body).unwrap();
    let auth = last
        .headers
        .get("authorization")
        .map(|v| v.to_str().unwrap().to_string());
    (body, auth)
}

/// Phase 0.6 — raw request-body bytes exactly as sent (full-byte golden). See the
/// openai_client.rs helper of the same name for rationale.
async fn last_request_body_raw(server: &MockServer) -> String {
    let requests = server.received_requests().await.unwrap();
    let last = requests.last().expect("at least one request");
    String::from_utf8(last.body.clone()).expect("utf-8 request body")
}

fn llm_golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-llm");
    root.join("testdata").join("golden").join("llm")
}

fn snap_llm(name: &str, body: &str) {
    insta::with_settings!({
        snapshot_path => llm_golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, body);
    });
}

fn docs() -> Vec<String> {
    vec!["doc-a".to_string(), "doc-b".to_string(), "doc-c".to_string()]
}

#[tokio::test]
async fn rerank_sends_golden_body_with_default_top_n() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "results": [] })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    client.rerank("q", docs(), None).await.unwrap();
    let (body, auth) = last_request(&server).await;
    assert_eq!(
        json!({
            "model": "test-reranker",
            "query": "q",
            "documents": ["doc-a", "doc-b", "doc-c"],
            "top_n": 3
        }),
        body
    );
    assert_eq!(Some("Bearer rk-key".to_string()), auth);

    // Full-byte golden: rerank request body (default top_n = documents.len()).
    snap_llm(
        "rerank_request_body",
        &last_request_body_raw(&server).await,
    );
}

#[tokio::test]
async fn rerank_parses_and_sorts_results_descending() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "results": [
                { "index": 0, "relevance_score": 0.2 },
                { "index": 2, "relevance_score": 0.9 },
                { "index": 1, "relevance_score": 0.5 }
            ]
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let results = client.rerank("q", docs(), Some(2)).await.unwrap();
    assert_eq!(3, results.len(), "returns all parsed results");
    assert_eq!(2, results[0].index);
    assert_eq!("doc-c", results[0].document);
    assert!((results[0].score - 0.9).abs() < 1e-6);
    assert_eq!(1, results[1].index);
    assert_eq!(0, results[2].index);
}

#[tokio::test]
async fn rerank_connection_failure_falls_back_to_original_order() {
    // Point at a dead port: connection refused triggers degradation.
    let mut config = RerankConfig::new(RerankProvider::Custom);
    config.base_url = Some("http://127.0.0.1:9".to_string());
    config.connect_timeout = std::time::Duration::from_secs(2);
    config.timeout = std::time::Duration::from_secs(2);
    let client = RerankClient::new(config).unwrap_or_else(|e| panic!("client: {e}"));

    let results = client.rerank("q", docs(), None).await.unwrap();
    assert_eq!(3, results.len());
    for (i, result) in results.iter().enumerate() {
        assert_eq!(i, result.index);
        assert!((result.score - (1.0 - i as f32 * 0.01)).abs() < 1e-6);
    }
}

#[tokio::test]
async fn rerank_fallback_respects_top_n() {
    let mut config = RerankConfig::new(RerankProvider::Custom);
    config.base_url = Some("http://127.0.0.1:9".to_string());
    config.connect_timeout = std::time::Duration::from_secs(2);
    config.timeout = std::time::Duration::from_secs(2);
    let client = RerankClient::new(config).unwrap_or_else(|e| panic!("client: {e}"));

    let results = client.rerank("q", docs(), Some(1)).await.unwrap();
    assert_eq!(1, results.len());
    assert_eq!(0, results[0].index);
}

#[tokio::test]
async fn rerank_empty_documents_makes_no_request() {
    let server = MockServer::start().await;
    let client = start_client(&server).await;
    let results = client.rerank("q", Vec::new(), None).await.unwrap();
    assert!(results.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[test]
fn custom_provider_without_base_url_is_rejected() {
    let config = RerankConfig::new(RerankProvider::Custom);
    match RerankClient::new(config) {
        Err(easyrag_core::CoreError::InvalidArgument(_)) => {}
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("must require base_url"),
    }
}

#[test]
fn default_provider_is_jina() {
    assert_eq!(RerankProvider::Jina, RerankProvider::default());
}
