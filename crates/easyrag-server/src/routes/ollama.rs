//! Ollama-compatible endpoints.
//!
//! Lets EasyRAG act as a drop-in replacement for Ollama: `/api/chat`,
//! `/api/chat/stream`, `/api/generate`, `/api/tags`, `/api/version`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::BoxStream;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::model::query_param::{QueryMode, QueryParam};

use crate::error::AppResult;
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/chat", post(chat))
        .route("/api/chat/stream", post(chat_stream))
        .route("/api/generate", post(generate))
        .route("/api/tags", get(list_models))
        .route("/api/version", get(version))
}

#[derive(Debug, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OllamaMessage>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
}

fn default_model() -> String {
    "easyrag".to_string()
}

/// default-argument rule: `doneReason = if (done) "stop" else null`;
/// the null is serialized (Jackson default inclusion).
fn chat_response(model: &str, content: &str, done: bool) -> Value {
    json!({
        "model": model,
        "message": { "role": "assistant", "content": content },
        "done": done,
        "done_reason": if done { json!("stop") } else { Value::Null },
        "total_duration": 0,
        "eval_count": 0,
    })
}

fn last_user_query(messages: &[OllamaMessage]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

async fn chat(state: AppState, Json(request): Json<OllamaChatRequest>) -> AppResult<Json<Value>> {
    let query = last_user_query(&request.messages);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Mix;
    param.stream = false;
    let response = state.query_engine.query(&query, &param).await?;
    Ok(Json(chat_response(&request.model, &response, true)))
}

/// Streaming chat. returns `Flow<OllamaChatResponse>` with
/// `text/event-stream` produces, so Spring writes each element as an SSE
/// event whose data payload is the JSON response.
async fn chat_stream(
    state: AppState,
    Json(request): Json<OllamaChatRequest>,
) -> Sse<BoxStream<'static, Result<Event, Infallible>>> {
    let query = last_user_query(&request.messages);
    let mut param = QueryParam::default();
    param.mode = QueryMode::Mix;
    param.stream = true;
    let model = request.model.clone();
    let chunks = state.query_engine.query_stream(&query, &param);
    let events = chunks.map(move |chunk| {
        let frame = chat_response(&model, &chunk, false);
        Ok::<_, Infallible>(Event::default().json_data(frame).unwrap_or_default())
    });
    Sse::new(events.boxed())
}

async fn generate(
    state: AppState,
    Json(request): Json<OllamaGenerateRequest>,
) -> AppResult<Json<Value>> {
    let mut param = QueryParam::default();
    param.mode = QueryMode::Mix;
    let response = state.query_engine.query(&request.prompt, &param).await?;
    Ok(Json(json!({
        "model": request.model,
        "response": response,
        "done": true,
        "done_reason": "stop",
    })))
}

async fn list_models() -> Json<Value> {
    Json(json!({
        "models": [{
            "name": "easyrag",
            "modified_at": "2024-01-01T00:00:00Z",
            "size": 0,
            "digest": "easyrag",
        }],
    }))
}

async fn version() -> Json<Value> {
    Json(json!({ "version": "1.0.0" }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_response_done_reason_rule() {
        let done = chat_response("easyrag", "hi", true);
        assert_eq!(done["done_reason"], "stop");
        assert_eq!(done["total_duration"], 0);
        assert_eq!(done["eval_count"], 0);
        assert_eq!(done["message"]["role"], "assistant");

        let chunk = chat_response("m", "partial", false);
        assert!(chunk["done_reason"].is_null()); // null serialized
        assert_eq!(chunk["done"], false);
    }

    #[test]
    fn last_user_message_wins() {
        let messages = vec![
            OllamaMessage { role: "user".into(), content: "first".into() },
            OllamaMessage { role: "assistant".into(), content: "reply".into() },
            OllamaMessage { role: "user".into(), content: "second".into() },
        ];
        assert_eq!(last_user_query(&messages), "second");
        assert_eq!(last_user_query(&[]), "");
    }

    #[test]
    fn request_defaults() {
        let req: OllamaChatRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(req.model, "easyrag");
        assert!(req.messages.is_empty());
        assert!(!req.stream);
        let req: OllamaGenerateRequest = serde_json::from_str("{}").unwrap();
        assert_eq!(req.model, "easyrag");
        assert_eq!(req.prompt, "");
    }
}
