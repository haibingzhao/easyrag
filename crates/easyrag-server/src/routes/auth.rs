//! Authentication endpoints.
//!
//! `POST /api/auth/login`: rate-limited by `{ip}:{username}` (sliding
//! window, 429 when exceeded), BCrypt credential check via `AuthService`,
//! `{token, username}` on success / 401 on failure.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::rejection::ExtensionRejection;
use axum::extract::ConnectInfo;
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{AppError, AppResult};
use crate::state::{AppServices, AppState};

#[derive(Deserialize)]
struct LoginRequest {
    username: Option<String>,
    password: Option<String>,
}

pub fn router() -> Router<Arc<AppServices>> {
    Router::new().route("/api/auth/login", post(login))
}

async fn login(
    state: AppState,
    // Err when the server runs without `into_make_service_with_connect_info`
    // (e.g. tower oneshot contract tests); falls back to "unknown".
    connect: Result<ConnectInfo<SocketAddr>, ExtensionRejection>,
    Json(body): Json<LoginRequest>,
) -> AppResult<Json<Value>> {
    let username = body
        .username
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let password = body.password.filter(|s| !s.is_empty());
    let (Some(username), Some(password)) = (username, password) else {
        return Err(AppError::bad_request("username and password are required"));
    };

    // rate limiter keyed by client IP + username.
    let ip = connect
        .ok()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let identifier = format!("{ip}:{username}");
    if !state.rate_limiter.is_allowed(&identifier) {
        return Err(AppError::status(
            429,
            "Too many login attempts; please try again later",
        ));
    }
    // Every attempt is accounted for, success and failure alike.
    state.rate_limiter.record_attempt(&identifier);

    match state.auth_service.authenticate(&username, &password).await? {
        Some(user) => {
            let token = state.auth_service.generate_token(&user.username)?;
            tracing::info!("login succeeded for user '{}' from {ip}", user.username);
            Ok(Json(json!({ "token": token, "username": user.username })))
        }
        None => Err(AppError::status(401, "Invalid username or password")),
    }
}
