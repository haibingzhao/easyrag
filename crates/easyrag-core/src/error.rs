//! Error hierarchy for EasyRAG.

use crate::concurrency::resilience::IsRetryable;

/// Unified error type for all EasyRAG crates.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// Raised when an API response has a status code of 4xx or 5xx.
    #[error("{message}")]
    ApiStatus {
        message: String,
        status_code: u16,
        request_id: Option<String>,
    },

    /// Raised when a connection error occurs during API calls.
    #[error("{0}")]
    ApiConnection(String),

    /// Raised when an API request times out.
    #[error("Request timed out.")]
    ApiTimeout,

    /// Raised when storage operations are attempted before initialization.
    #[error("{0} not initialized. Please ensure proper initialization: initialize_storages() must be called first.")]
    StorageNotInitialized(String),

    /// Raised when pipeline status is accessed before initialization.
    #[error("Pipeline namespace '{0}' not found. Pipeline status should be auto-initialized by initialize_storages().")]
    PipelineNotInitialized(String),

    /// Raised when pipeline processing is cancelled by user request.
    #[error("{0}")]
    PipelineCancelled(String),

    /// Raised when a storage backend fails to flush buffered index ops.
    #[error("{storage_name}[{namespace}] index flush failed: {message}")]
    IndexFlush {
        storage_name: String,
        namespace: String,
        message: String,
    },

    /// Raised when a chunk exceeds the configured token limit.
    #[error("{0}")]
    ChunkTokenLimitExceeded(String),

    /// Raised when a chunk cannot be located in the document's blocks.
    #[error("{0}")]
    ChunkBlockMatch(String),

    /// Raised when data migration from legacy collection/table fails.
    #[error("{0}")]
    DataMigration(String),

    /// Raised when multimodal analysis must fail the current document.
    #[error("{0}")]
    MultimodalAnalysis(String),

    /// Invalid argument / precondition violation.
    #[error("{0}")]
    InvalidArgument(String),

    /// Unsupported operation.
    #[error("{0}")]
    Unsupported(String),

    /// Circuit breaker rejected the call.
    #[error("{0}")]
    CircuitBreakerOpen(String),

    /// Catch-all internal error.
    #[error("{0}")]
    Internal(String),
}

impl CoreError {
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::api_status(400, message, None)
    }
    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::api_status(401, message, None)
    }
    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::api_status(403, message, None)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::api_status(404, message, None)
    }
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::api_status(409, message, None)
    }
    pub fn unprocessable(message: impl Into<String>) -> Self {
        Self::api_status(422, message, None)
    }
    pub fn rate_limited(message: impl Into<String>) -> Self {
        Self::api_status(429, message, None)
    }
    pub fn payload_too_large(message: impl Into<String>) -> Self {
        Self::api_status(413, message, None)
    }
    pub fn api_status(
        status_code: u16,
        message: impl Into<String>,
        request_id: Option<String>,
    ) -> Self {
        Self::ApiStatus {
            message: message.into(),
            status_code,
            request_id,
        }
    }

    /// HTTP status code associated with this error (500 for non-HTTP errors).
    pub fn status_code(&self) -> u16 {
        match self {
            Self::ApiStatus { status_code, .. } => *status_code,
            _ => 500,
        }
    }
}

impl IsRetryable for CoreError {
    /// Transient failures safe to retry: connection/timeout, rate limit, 5xx.
    fn is_retryable(&self) -> bool {
        match self {
            Self::ApiConnection(_) | Self::ApiTimeout => true,
            Self::ApiStatus { status_code, .. } => {
                *status_code == 429 || (500..600).contains(status_code)
            }
            _ => false,
        }
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        Self::Internal(e.to_string())
    }
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        Self::ApiConnection(e.to_string())
    }
}
