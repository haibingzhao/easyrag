//! Ollama client tests: golden chat body, NDJSON stream parsing,
//! embeddings, and tolerance of malformed responses.

use easyrag_core::llm::service::{EmbeddingService, LlmRequest, LlmService};
use easyrag_llm::{OllamaClient, OllamaConfig};
use futures::StreamExt;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn start_client(server: &MockServer) -> OllamaClient {
    let mut config = OllamaConfig::new();
    config.base_url = server.uri();
    config.model = "test-llama".to_string();
    OllamaClient::new(config).unwrap()
}

async fn last_request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    let last = requests.last().expect("at least one request");
    serde_json::from_slice(&last.body).unwrap()
}

#[tokio::test]
async fn complete_sends_golden_chat_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": { "role": "assistant", "content": "hi" }
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let request = LlmRequest::new("hello")
        .with_system_prompt("sys")
        .with_temperature(0.5)
        .with_max_tokens(42);
    let result = client.complete(&request).await.unwrap();
    assert_eq!("hi", result);

    assert_eq!(
        json!({
            "model": "test-llama",
            "messages": [
                { "role": "system", "content": "sys" },
                { "role": "user", "content": "hello" }
            ],
            "stream": false,
            "options": { "temperature": 0.5, "num_predict": 42 }
        }),
        last_request_body(&server).await
    );
}

#[tokio::test]
async fn complete_omits_zero_temperature_options() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": { "content": "ok" }
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    client.complete(&LlmRequest::new("x")).await.unwrap();
    let body = last_request_body(&server).await;
    assert_eq!(json!({}), body["options"], "zero temperature and no max_tokens");
}

#[tokio::test]
async fn complete_malformed_response_returns_empty_string() {
    // no status check, parse failures default to empty.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500).set_body_string("not json"))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let result = client.complete(&LlmRequest::new("x")).await.unwrap();
    assert_eq!("", result);
}

#[tokio::test]
async fn complete_stream_parses_ndjson_with_boundaries() {
    let server = MockServer::start().await;
    // Boundary cases: blank lines, malformed JSON line, empty content,
    // tail without trailing newline.
    let ndjson = "{\"message\":{\"content\":\"He\"}}\n\
                  \n\
                  {not valid json}\n\
                  {\"message\":{\"content\":\"\"}}\n\
                  {\"message\":{\"content\":\"llo\"}}";
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string(ndjson))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let chunks: Vec<String> = client
        .complete_stream(LlmRequest::new("hello"))
        .collect()
        .await;
    assert_eq!(vec!["He".to_string(), "llo".to_string()], chunks);

    // Streaming requests must set stream=true.
    assert_eq!(json!(true), last_request_body(&server).await["stream"]);
}

#[tokio::test]
async fn embed_sends_golden_body_and_parses_embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embeddings": [[0.1, 0.2], [0.3, 0.4]]
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let vectors = client
        .embed(vec!["a".to_string(), "b".to_string()])
        .await
        .unwrap();
    assert_eq!(vec![vec![0.1, 0.2], vec![0.3, 0.4]], vectors);
    assert_eq!(4096, client.embedding_dimension());

    assert_eq!(
        json!({ "model": "test-llama", "input": ["a", "b"] }),
        last_request_body(&server).await
    );
}

#[tokio::test]
async fn embed_empty_input_makes_no_request() {
    let server = MockServer::start().await;
    let client = start_client(&server).await;
    let vectors = client.embed(Vec::new()).await.unwrap();
    assert!(vectors.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}
