//! Unified error handling.
//!
//! Error bodies are always `{"status":<code>,"message":"...","correlation_id":"..."}`;
//! raw exception details never reach the client (CWE-209).

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

/// Server-side error type. `Status` carries an explicit HTTP code and a
/// client-safe message; `Core` wraps
/// pipeline/storage errors and is sanitized on the way out.
#[derive(Debug)]
pub enum AppError {
    Status(u16, String),
    Core(easyrag_core::CoreError),
}

impl AppError {
    pub fn status(code: u16, message: impl Into<String>) -> Self {
        Self::Status(code, message.into())
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::Status(400, message.into())
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::Status(409, message.into())
    }

    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::Status(413, message.into())
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::Status(404, message.into())
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::Status(401, message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Status(500, message.into())
    }

    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self::Status(429, message.into())
    }
}

// Explicit conversions (a blanket `impl<E: Into<CoreError>>` would conflict
// with the reflexive `From<T> for T` if `CoreError: From<AppError>` is ever
// added, and would leave axum rejections / parse errors unconverted anyway).
impl From<easyrag_core::CoreError> for AppError {
    fn from(err: easyrag_core::CoreError) -> Self {
        Self::Core(err)
    }
}

impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        Self::Core(err.into())
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        Self::Core(err.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let correlation_id = short_uuid();

        // GlobalExceptionHandler mapping:
        // ResponseStatusException -> its status/reason
        // IllegalArgumentException -> 400 "Invalid request parameters"
        // UnsupportedOperationException -> 501 "Operation not supported"
        // anything else -> 500 "Internal server error" (details server-side).
        // NOTE: CoreError::ApiStatus intentionally falls into the catch-all:
        // ApiStatusException is a plain RuntimeException (not a
        // ResponseStatusException) and is likewise exposed as 500.
        let (code, message) = match self {
            Self::Status(code, message) => (code, message),
            Self::Core(easyrag_core::CoreError::InvalidArgument(_)) => {
                (400, "Invalid request parameters".to_string())
            }
            Self::Core(easyrag_core::CoreError::Unsupported(_)) => {
                (501, "Operation not supported".to_string())
            }
            Self::Core(err) => {
                // Debug form keeps the source chain.
                tracing::error!("[{correlation_id}] Unhandled exception: {err:?}");
                (500, "Internal server error".to_string())
            }
        };

        if (400..500).contains(&code) {
            tracing::warn!("[{correlation_id}] Client error {code}: {message}");
        }

        // Invalid codes fall back to 500 in BOTH the header and the body
        //
        // Note: http's StatusCode accepts custom codes like 999, while Spring
        // only resolves standard ones; canonical_reason filters those out.
        let status = StatusCode::from_u16(code)
            .ok()
            .filter(|s| s.canonical_reason().is_some())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let body = json!({
            "status": status.as_u16(),
            "message": message,
            "correlation_id": correlation_id,
        });
        (status, Json(body)).into_response()
    }
}

/// First 8 chars of a random UUID.
pub fn short_uuid() -> String {
    uuid::Uuid::new_v4().to_string()[..8].to_string()
}

/// Result alias used by all route handlers.
pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn body_json(err: AppError) -> (u16, serde_json::Value) {
        let resp = err.into_response();
        let code = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (code, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn error_body_has_fixed_shape() {
        let (code, body) = body_json(AppError::bad_request("boom")).await;
        assert_eq!(code, 400);
        assert_eq!(body["status"], 400);
        assert_eq!(body["message"], "boom");
        assert_eq!(body["correlation_id"].as_str().unwrap().len(), 8);
    }

    #[tokio::test]
    async fn invalid_argument_maps_to_400_with_sanitized_message() {
        let err = AppError::Core(easyrag_core::CoreError::InvalidArgument(
            "secret detail".into(),
        ));
        let (code, body) = body_json(err).await;
        assert_eq!(code, 400);
        assert_eq!(body["message"], "Invalid request parameters");
    }

    #[tokio::test]
    async fn unsupported_maps_to_501() {
        let err = AppError::Core(easyrag_core::CoreError::Unsupported("nope".into()));
        let (code, body) = body_json(err).await;
        assert_eq!(code, 501);
        assert_eq!(body["message"], "Operation not supported");
    }

    #[tokio::test]
    async fn other_core_errors_map_to_500_including_api_status() {
        // ApiStatusException is a plain RuntimeException -> 500.
        let err = AppError::Core(easyrag_core::CoreError::ApiStatus {
            message: "token expired".into(),
            status_code: 401,
            request_id: None,
        });
        let (code, body) = body_json(err).await;
        assert_eq!(code, 500);
        assert_eq!(body["message"], "Internal server error");
    }

    #[tokio::test]
    async fn invalid_status_code_falls_back_to_500_in_header_and_body() {
        let (code, body) = body_json(AppError::status(999, "weird")).await;
        assert_eq!(code, 500);
        assert_eq!(body["status"], 500);
        assert_eq!(body["message"], "weird");
    }
}
