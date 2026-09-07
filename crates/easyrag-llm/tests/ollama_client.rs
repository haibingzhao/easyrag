//! Ollama client tests: golden chat body, NDJSON stream parsing,
//! embeddings, and tolerance of malformed responses.

use easyrag_core::llm::service::{
    ChatMessage, EmbeddingService, LlmRequest, LlmService, MessageRole, ToolCallingLlmService,
    ToolDefinition,
};
use easyrag_core::JsonMap;
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

    // Full-byte golden: Ollama's provider-specific shape (`options.num_predict`,
    // not top-level `max_tokens`) is deliberately NOT abstracted in Phase 4.1.
    snap_llm(
        "ollama_complete_request_body",
        &last_request_body_raw(&server).await,
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
    // Parse tolerance on a *successful* response is unchanged by `[BC-4]`: a
    // 200 whose body is not JSON still degrades to empty content rather than
    // failing the call.
    //
    // This test previously asserted the same outcome for a **500**, with the
    // comment "no status check, parse failures default to empty" — i.e. it
    // pinned defect A-7 as the expected behaviour. `[BC-4]` splits the two
    // cases: tolerance for a malformed 200 stays here, and a non-2xx becomes an
    // error in `bc4_complete_non_2xx_is_an_error_not_empty_content` below.
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let result = client.complete(&LlmRequest::new("x")).await.unwrap();
    assert_eq!("", result);
    assert_eq!(1, server.received_requests().await.unwrap().len());
}

/// The other half of the split above. A 500 is not in the transient keyword set
/// (`rate limit`/`timeout`/`connection`/502/503), so it is reported once,
/// immediately, and with the upstream wording intact.
#[tokio::test]
async fn bc4_complete_non_2xx_is_an_error_not_empty_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(500).set_body_string("not json"))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let err = client
        .complete(&LlmRequest::new("x"))
        .await
        .expect_err("a 500 must not be reported as an empty success");
    assert!(err.to_string().contains("500"), "got {err}");
    assert!(err.to_string().contains("not json"), "got {err}");
    assert_eq!(1, server.received_requests().await.unwrap().len());
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

    // Full-byte golden: Ollama embeddings endpoint (`/api/embed`) body.
    snap_llm(
        "ollama_embed_request_body",
        &last_request_body_raw(&server).await,
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

// ===== tool calling =====

fn test_tool(name: &str) -> ToolDefinition {
    let mut params = JsonMap::new();
    params.insert("type".to_string(), json!("object"));
    ToolDefinition {
        name: name.to_string(),
        description: format!("{name} description"),
        parameters: params,
    }
}

#[tokio::test]
async fn complete_with_tools_sends_tools_in_request_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": { "role": "assistant", "content": "done" }
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    // Agent-loop shaped history: assistant turn + two parallel tool results.
    let messages = vec![
        ChatMessage {
            role: MessageRole::User,
            content: "find docs".to_string(),
        },
        ChatMessage {
            role: MessageRole::Assistant,
            content: "thinking".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "[Tool Result: list_documents]\n1. doc_id=d1".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "[Tool Result: get_document_structure]\n{\"tree\":[]}".to_string(),
        },
    ];

    let response = client
        .complete_with_tools(messages, vec![test_tool("list_documents")])
        .await
        .unwrap();
    assert_eq!(Some("done".to_string()), response.content);

    let body = last_request_body(&server).await;
    assert_eq!(
        json!([{
            "type": "function",
            "function": {
                "name": "list_documents",
                "description": "list_documents description",
                "parameters": { "type": "object" }
            }
        }]),
        body["tools"]
    );
    assert_eq!(json!(false), body["stream"]);

    let msgs = body["messages"].as_array().unwrap();
    // Synthesized assistant tool_calls use Ollama's `function.index` form;
    // both parallel results mount on the same assistant message.
    assert_eq!(
        json!([
            {
                "type": "function",
                "function": { "index": 0, "name": "list_documents", "arguments": {} }
            },
            {
                "type": "function",
                "function": { "index": 1, "name": "get_document_structure", "arguments": {} }
            }
        ]),
        msgs[1]["tool_calls"]
    );
    // Tool results use Ollama's `tool_name` pairing (no tool_call_id).
    assert_eq!(
        json!({ "role": "tool", "tool_name": "list_documents", "content": "1. doc_id=d1" }),
        msgs[2]
    );
    assert_eq!(
        json!({ "role": "tool", "tool_name": "get_document_structure", "content": "{\"tree\":[]}" }),
        msgs[3]
    );
}

#[tokio::test]
async fn complete_with_tools_parses_tool_calls_with_object_arguments() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "index": 0,
                        "name": "get_section_content",
                        "arguments": { "doc_id": "d1", "max_chars": 4000 }
                    }
                }]
            }
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let response = client
        .complete_with_tools(
            vec![ChatMessage {
                role: MessageRole::User,
                content: "q".to_string(),
            }],
            vec![test_tool("get_section_content")],
        )
        .await
        .unwrap();

    assert!(response.content.is_none(), "empty content maps to None");
    assert_eq!(1, response.tool_calls.len());
    let call = &response.tool_calls[0];
    assert!(call.id.is_none(), "Ollama assigns no tool_call ids");
    assert_eq!("get_section_content", call.name);
    assert_eq!("d1", call.arguments["doc_id"]);
    assert_eq!("4000", call.arguments["max_chars"]);
}

#[tokio::test]
async fn complete_with_tools_handles_no_tool_calls() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "message": { "role": "assistant", "content": "plain answer" }
        })))
        .mount(&server)
        .await;
    let client = start_client(&server).await;

    let response = client
        .complete_with_tools(Vec::new(), vec![test_tool("list_documents")])
        .await
        .unwrap();
    assert_eq!(Some("plain answer".to_string()), response.content);
    assert!(response.tool_calls.is_empty());
}

// ===== [BC-4] retry + circuit breaker (defect A-7) =====
//
// Before this change the Ollama client performed *no* status check and *no*
// retry: a 503 body was handed to the parser as ordinary text, which degraded
// to an empty completion, so `complete` returned `Ok("")` for a server that was
// visibly failing. Ingestion and query both treat `Ok` as success, so the
// failure was silent — the exact availability defect A-7 describes, and the
// reason the OpenAI client has had a breaker since before this refactor.
//
// The five `bc4_` tests below were written first and run against the
// un-retried client. Four failed, which is the evidence for the behaviour
// change: `complete` returned `Ok("")` where a recovered answer was expected
// (`left: "recovered", right: ""`), a 400 came back as an empty success
// (`a 400 must not be reported as an empty success: ""`), the breaker assertion
// failed because every call "succeeded", and `embed` returned `Ok(vec![])` for
// a 503 with one request on the wire where three were expected. The fifth,
// `bc4_embed_success_path_is_unchanged`, passed then and passes now — it is here
// to prove the change is confined to the failure path.
//
// A sixth test, `bc4_complete_non_2xx_is_an_error_not_empty_content`, was then
// split out of the pre-existing `complete_malformed_response_returns_empty_string`,
// which had been asserting the defect itself. See that test for the details.

/// `start_client` with a config tweak, mirroring the openai_client.rs helper so
/// the retry tests can shrink the backoff to 1ms.
async fn start_client_with(server: &MockServer, tweak: impl FnOnce(&mut OllamaConfig)) -> OllamaClient {
    let mut config = OllamaConfig::new();
    config.base_url = server.uri();
    config.model = "test-llama".to_string();
    tweak(&mut config);
    OllamaClient::new(config).unwrap()
}

fn chat_response(content: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "message": { "role": "assistant", "content": content }
    }))
}

#[tokio::test]
async fn bc4_complete_retries_on_503_then_succeeds() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_string("service unavailable"))
        .up_to_n_times(2)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(chat_response("recovered"))
        .mount(&server)
        .await;

    let client = start_client_with(&server, |c| {
        c.retry_delay_base_ms = 1; // fast tests
    })
    .await;

    let result = client.complete(&LlmRequest::new("hello")).await.unwrap();
    assert_eq!("recovered", result);
    assert_eq!(3, server.received_requests().await.unwrap().len());
}

#[tokio::test]
async fn bc4_complete_surfaces_a_fatal_status_instead_of_empty_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(400).set_body_string("model 'x' not found"))
        .mount(&server)
        .await;
    let client = start_client_with(&server, |c| {
        c.retry_delay_base_ms = 1;
    })
    .await;

    let err = client
        .complete(&LlmRequest::new("hello"))
        .await
        .expect_err("a 400 must not be reported as an empty success");
    // Non-transient: exactly one attempt, and the upstream wording survives so
    // operators can see which model was rejected.
    assert!(err.to_string().contains("400"), "got {err}");
    assert!(err.to_string().contains("not found"), "got {err}");
    assert_eq!(1, server.received_requests().await.unwrap().len());
}

#[tokio::test]
async fn bc4_complete_circuit_breaker_opens_after_five_transient_failures() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;
    let client = start_client_with(&server, |c| {
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

/// Embeddings get the same treatment. The pre-change behaviour was worse here
/// than for chat: `embeddings` is absent from an error body, so `embed`
/// returned `Ok(vec![])` and an ingestion run would have stored nothing while
/// reporting success.
#[tokio::test]
async fn bc4_embed_retries_and_then_surfaces_failure() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(503).set_body_string("down"))
        .mount(&server)
        .await;
    let client = start_client_with(&server, |c| {
        c.max_retries = 3;
        c.retry_delay_base_ms = 1;
    })
    .await;

    let err = client
        .embed(vec!["some text".to_string()])
        .await
        .expect_err("a 503 must not become an empty embedding list");
    assert!(err.to_string().contains("503"), "got {err}");
    assert_eq!(3, server.received_requests().await.unwrap().len());
}

/// The success path is untouched by `[BC-4]`: a 200 still yields the vectors,
/// and the retry budget is not consumed.
#[tokio::test]
async fn bc4_embed_success_path_is_unchanged() {
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
    assert_eq!(vec![vec![0.1f32, 0.2], vec![0.3, 0.4]], vectors);
    assert_eq!(1, server.received_requests().await.unwrap().len());
}
