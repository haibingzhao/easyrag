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
    /// Transient failures safe to retry: connection/timeout, rate limit, 5xx,
    /// and the storage-driver transients that surface as `Internal(message)`.
    fn is_retryable(&self) -> bool {
        match self {
            Self::ApiConnection(_) | Self::ApiTimeout => true,
            Self::ApiStatus { status_code, .. } => {
                *status_code == 429 || (500..600).contains(status_code)
            }
            // SQL/graph drivers are stringified at the storage boundary, so a
            // dropped connection arrives as `Internal(msg)` rather than
            // `ApiConnection`. Classified here — not inside each backend — so
            // Postgres, Neo4j and any future driver agree on what "transient"
            // means (Phase 3.10).
            Self::Internal(msg) => is_transient_connection(msg) || is_transient_transaction(msg),
            _ => false,
        }
    }
}

/// Connection-level transients. The first six substrings are the classification
/// `postgres.rs::retry` already used inline; moving them here is [equivalent] —
/// the same messages retry, the same messages do not, and the backoff curve is
/// unchanged (see `PG_RETRY_CONFIG` / neo4j's `RETRY_CONFIG`).
fn is_transient_connection(msg: &str) -> bool {
    const PATTERNS: [&str; 7] = [
        "error occurred while waiting to get a connection",
        "error returned from database: connection refused",
        "error communicating with the server",
        "error communicating with database",
        "connection closed",
        "broken pipe",
        // [BC-3 widening] the peer resetting mid-statement is the same failure
        // class as a broken pipe, but was not in the original list.
        "connection reset",
    ];
    PATTERNS.iter().any(|p| msg.contains(p))
}

/// Transaction-level transients: the server aborted our statement because it
/// conflicted with a concurrent one, and asks the client to retry.
///
/// [BC-3 widening] These were NOT retried before Phase 3.10 — both backends
/// treated them as fatal. Retrying is safe here because every statement these
/// backends issue is idempotent: writes are `INSERT .. ON CONFLICT DO UPDATE` /
/// `MERGE`, deletes are keyed, reads have no side effects. A deadlock or a
/// serialization failure leaves nothing half-applied to replay.
fn is_transient_transaction(msg: &str) -> bool {
    const PATTERNS: [&str; 4] = [
        "deadlock detected",           // SQLSTATE 40P01
        "could not serialize access",  // SQLSTATE 40001
        "40P01",
        "40001",
    ];
    PATTERNS.iter().any(|p| msg.contains(p))
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

// ---------------------------------------------------------------------------
// Phase 3.10 guardrails: retry classification moved here from
// `postgres.rs::retry` so Postgres and Neo4j cannot drift on what counts as
// transient. The first test pins the six messages that were ALREADY retryable
// ([equivalent] move); the ones marked BC-3 pin the widening.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod retryable_tests {
    use super::*;

    fn internal(msg: &str) -> CoreError {
        CoreError::Internal(msg.to_string())
    }

    #[test]
    fn the_messages_postgres_already_retried_stay_retryable() {
        // Verbatim from the inline `matches!` the old `postgres.rs::retry` used.
        for msg in [
            "error occurred while waiting to get a connection",
            "error returned from database: connection refused",
            "error communicating with the server",
            "connection closed",
            "error communicating with database",
            "broken pipe",
        ] {
            assert!(
                internal(msg).is_retryable(),
                "[equivalent] regression: `{msg}` used to be retried"
            );
            // Real driver messages wrap the substring in context.
            assert!(
                internal(&format!("sqlx: {msg} (while executing SELECT 1)")).is_retryable(),
                "classification must work on a wrapped driver message: `{msg}`"
            );
        }
        // `ApiTimeout` was the one non-`Internal` case the old loop accepted.
        assert!(CoreError::ApiTimeout.is_retryable());
    }

    #[test]
    fn bc3_connection_reset_becomes_retryable() {
        // Widening: same failure class as `broken pipe`, previously fatal.
        assert!(internal("connection reset by peer").is_retryable());
    }

    #[test]
    fn bc3_deadlock_and_serialization_failures_become_retryable() {
        // Widening: the server aborted the statement and asks the client to
        // retry. Safe because every statement these backends issue is idempotent.
        assert!(internal("error returned from database: deadlock detected (SQLSTATE 40P01)").is_retryable());
        assert!(internal("could not serialize access due to concurrent update (SQLSTATE 40001)").is_retryable());
        // Some drivers surface only the SQLSTATE.
        assert!(internal("SQLSTATE 40P01").is_retryable());
        assert!(internal("SQLSTATE 40001").is_retryable());
    }

    #[test]
    fn data_and_programming_errors_are_never_retryable() {
        // Retrying these burns the backoff budget and returns the same error.
        for msg in [
            "duplicate key value violates unique constraint \"pk_docs\"",
            "relation \"graph_nodes\" does not exist",
            "syntax error at or near \"SELEC\"",
            "null value in column \"id\" violates not-null constraint",
            "",
        ] {
            assert!(
                !internal(msg).is_retryable(),
                "`{msg}` must not be retried"
            );
        }
    }

    #[test]
    fn api_errors_keep_their_previous_classification() {
        let status = |code: u16| CoreError::ApiStatus {
            message: "m".to_string(),
            status_code: code,
            request_id: None,
        };
        assert!(status(429).is_retryable());
        assert!(status(500).is_retryable());
        assert!(status(503).is_retryable());
        assert!(!status(400).is_retryable());
        assert!(!status(401).is_retryable());
        assert!(!status(404).is_retryable());
        assert!(CoreError::ApiConnection("boom".to_string()).is_retryable());
        // Storage-init errors are caller bugs, not transients.
        assert!(!CoreError::StorageNotInitialized("vector".to_string()).is_retryable());
        assert!(!CoreError::PipelineCancelled("user".to_string()).is_retryable());
    }
}
