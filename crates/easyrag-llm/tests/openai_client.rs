//! OpenAI client tests: golden request-body assertions, SSE parsing
//! boundaries, retry and circuit breaker semantics.

use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, MessageRole, ResponseFormat,
};
use easyrag_llm::{OpenAiClient, OpenAiConfig};
use futures::StreamExt;
use serde_json::{json, Value};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

async fn start_client(server: &MockServer, mutate: impl FnOnce(&mut OpenAiConfig)) -> OpenAiClient {
    let mut config = OpenAiConfig::new("test-key");
    config.base_url = server.uri();
    config.model = "test-model".to_string();
    mutate(&mut config);
    OpenAiClient::new(config).unwrap()
}

async fn last_request_body(server: &MockServer) -> Value {
    let requests = server.received_requests().await.unwrap();
    let last = requests.last().expect("at least one request");
    serde_json::from_slice(&last.body).unwrap()
}

fn completion_response(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "choices": [{ "message": { "content": content } }]
    }))
}

#[tokio::test]
async fn complete_sends_golden_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion_response("ok"))
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    let request = LlmRequest::new("hello")
        .with_system_prompt("sys")
        .with_history(vec![
            ChatMessage {
                role: MessageRole::User,
                content: "u1".to_string(),
            },
            ChatMessage {
                role: MessageRole::Assistant,
                content: "a1".to_string(),
            },
        ])
        .with_temperature(0.5)
        .with_max_tokens(100)
        .with_response_format(ResponseFormat::JsonObject);

    let result = client.complete(&request).await.unwrap();
    assert_eq!("ok", result);

    assert_eq!(
        json!({
            "model": "test-model",
            "messages": [
                { "role": "system", "content": "sys" },
                { "role": "user", "content": "u1" },
                { "role": "assistant", "content": "a1" },
                { "role": "user", "content": "hello" }
            ],
            "stream": false,
            "temperature": 0.5,
            "max_tokens": 100,
            "response_format": { "type": "json_object" }
        }),
        last_request_body(&server).await
    );

    // Authorization header is a Bearer token.
    let requests = server.received_requests().await.unwrap();
    let auth = requests[0].headers.get("authorization").unwrap();
    assert_eq!("Bearer test-key", auth.to_str().unwrap());
}

#[tokio::test]
async fn complete_omits_zero_temperature_and_optional_params() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion_response("ok"))
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    client
        .complete(&LlmRequest::new("hello"))
        .await
        .unwrap();

    let body = last_request_body(&server).await;
    assert!(body.get("temperature").is_none(), "temperature=0 must be omitted");
    assert!(body.get("max_tokens").is_none());
    assert!(body.get("response_format").is_none());
    assert_eq!(json!(false), body["stream"]);
}

#[tokio::test]
async fn complete_retries_on_503_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion_response("recovered"))
        .mount(&server)
        .await;

    let client = start_client(&server, |c| {
        c.retry_delay_base_ms = 1; // fast tests
    })
    .await;

    let result = client
        .complete(&LlmRequest::new("hello"))
        .await
        .unwrap();
    assert_eq!("recovered", result);
    assert_eq!(3, server.received_requests().await.unwrap().len());
}

#[tokio::test]
async fn complete_non_retryable_error_fails_immediately() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
        .mount(&server)
        .await;
    let client = start_client(&server, |c| c.retry_delay_base_ms = 1).await;

    let result = client.complete(&LlmRequest::new("hello")).await;
    assert!(result.is_err());
    // 400 is not retryable → exactly one request.
    assert_eq!(1, server.received_requests().await.unwrap().len());
}

#[tokio::test]
async fn complete_circuit_breaker_opens_after_five_transient_failures() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;
    let client = start_client(&server, |c| {
        c.max_retries = 1;
        c.retry_delay_base_ms = 1;
    })
    .await;

    for _ in 0..5 {
        assert!(client.complete(&LlmRequest::new("x")).await.is_err());
    }
    // Sixth call must be rejected by the open breaker without hitting the wire.
    let err = client
        .complete(&LlmRequest::new("x"))
        .await
        .expect_err("breaker should be open");
    assert!(matches!(
        err,
        easyrag_core::CoreError::CircuitBreakerOpen(_)
    ));
    assert_eq!(5, server.received_requests().await.unwrap().len());
}

#[tokio::test]
async fn complete_stream_parses_sse_with_boundaries() {
    let server = MockServer::start().await;
    // Boundary cases: blank lines, `data:` without space (skipped per
    // ), empty delta content, malformed JSON, [DONE].
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\
               \n\
               data:{\"choices\":[{\"delta\":{\"content\":\"SKIPPED\"}}]}\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n\
               data: {bad json}\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\
               data: [DONE]\n";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(sse),
        )
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    let chunks: Vec<String> = client
        .complete_stream(LlmRequest::new("hello"))
        .collect()
        .await;
    assert_eq!(vec!["Hel".to_string(), "lo".to_string()], chunks);

    // Stream requests must set stream=true.
    assert_eq!(json!(true), last_request_body(&server).await["stream"]);
}

#[tokio::test]
async fn complete_stream_handles_tail_without_trailing_newline() {
    let server = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\
               data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}";
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse))
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    let chunks: Vec<String> = client
        .complete_stream(LlmRequest::new("hello"))
        .collect()
        .await;
    assert_eq!(vec!["A".to_string(), "B".to_string()], chunks);
}

#[tokio::test]
async fn complete_with_image_builds_vision_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(completion_response("described"))
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    let result = client
        .complete_with_image(&LlmRequest::new("what is this?"), "BASE64DATA", "image/png")
        .await
        .unwrap();
    assert_eq!("described", result);

    let body = last_request_body(&server).await;
    let content = &body["messages"][0]["content"];
    assert_eq!(
        json!([
            { "type": "text", "text": "what is this?" },
            {
                "type": "image_url",
                "image_url": { "url": "data:image/png;base64,BASE64DATA" }
            }
        ]),
        *content
    );
    assert_eq!(json!(false), body["stream"]);
}

#[tokio::test]
async fn embed_sends_golden_body_and_parses_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [
                { "embedding": [0.1, 0.2, 0.3] },
                { "embedding": [0.4, 0.5, 0.6] }
            ]
        })))
        .mount(&server)
        .await;
    let client = start_client(&server, |_| {}).await;

    let vectors = client
        .embed(vec!["a".to_string(), "b".to_string()])
        .await
        .unwrap();
    assert_eq!(2, vectors.len());
    assert_eq!(vec![0.1, 0.2, 0.3], vectors[0]);
    assert_eq!(1536, client.embedding_dimension());

    assert_eq!(
        json!({ "model": "test-model", "input": ["a", "b"] }),
        last_request_body(&server).await
    );
}

#[tokio::test]
async fn embed_empty_input_makes_no_request() {
    let server = MockServer::start().await;
    let client = start_client(&server, |_| {}).await;
    let vectors = client.embed(Vec::new()).await.unwrap();
    assert!(vectors.is_empty());
    assert!(server.received_requests().await.unwrap().is_empty());
}

#[tokio::test]
async fn embed_non_200_is_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    let client = start_client(&server, |c| {
        c.max_retries = 1;
        c.retry_delay_base_ms = 1;
    })
    .await;

    let result = client.embed(vec!["a".to_string()]).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn embed_empty_data_array_is_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "data": [] })))
        .mount(&server)
        .await;
    let client = start_client(&server, |c| c.max_retries = 1).await;

    let result = client.embed(vec!["a".to_string()]).await;
    let err = result.expect_err("empty data must be an error");
    assert!(err.to_string().contains("empty data"));
}
