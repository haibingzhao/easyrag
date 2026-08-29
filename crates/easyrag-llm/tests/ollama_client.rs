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
