//! Retry and resilience utilities.
//! Async implementation with non-blocking delay on top of tokio.

use rand::Rng;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;

/// Trait marking error types that can classify themselves as transient
/// (safe to retry)..
pub trait IsRetryable {
    fn is_retryable(&self) -> bool;
}

/// Retry configuration for resilience operations.
#[derive(Clone, Debug)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            exponential_base: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Construct a config with validation.
    pub fn try_new(
        max_attempts: u32,
        base_delay: Duration,
        max_delay: Duration,
        exponential_base: f64,
        jitter: bool,
    ) -> Result<Self, crate::CoreError> {
        let config = Self {
            max_attempts,
            base_delay,
            max_delay,
            exponential_base,
            jitter,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), crate::CoreError> {
        if self.max_attempts < 1 {
            return Err(crate::CoreError::InvalidArgument(
                "maxAttempts must be >= 1".to_string(),
            ));
        }
        if self.exponential_base <= 0.0 {
            return Err(crate::CoreError::InvalidArgument(
                "exponentialBase must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Builder-style helpers for tests and callers.
    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }
}

/// Execute an async function with retry on transient failures.
/// Uses exponential backoff with optional jitter to avoid thundering-herd
/// problems. Mirrors `suspend fun <T> withRetry(...)`.
///
/// `on_retry` is invoked before each retry sleep with (error, attempt, delay).
pub async fn with_retry<T, E, F, Fut, C>(
    config: &RetryConfig,
    on_retry: Option<C>,
    mut block: F,
) -> Result<T, E>
where
    E: IsRetryable + std::fmt::Debug,
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    C: Fn(&E, u32, Duration),
{
    for attempt in 1..=config.max_attempts {
        match block().await {
            Ok(value) => return Ok(value),
            Err(exc) => {
                if !exc.is_retryable() {
                    return Err(exc);
                }

                if attempt == config.max_attempts {
                    tracing::error!(
                        "Operation failed after {} attempts: {:?}",
                        config.max_attempts,
                        exc
                    );
                    return Err(exc);
                }

                let mut delay_ms = (config.base_delay.as_millis() as f64
                    * config.exponential_base.powi((attempt - 1) as i32))
                    as u128;
                delay_ms = delay_ms.min(config.max_delay.as_millis());

                if config.jitter {
                    let jitter_factor = 1.0 + rand::thread_rng().gen_range(0.0..0.5);
                    delay_ms = (delay_ms as f64 * jitter_factor) as u128;
                }

                if let Some(cb) = &on_retry {
                    cb(&exc, attempt, Duration::from_millis(delay_ms as u64));
                }

                tracing::warn!(
                    "Attempt {}/{} failed ({:?}), retrying in {}ms…",
                    attempt,
                    config.max_attempts,
                    exc,
                    delay_ms
                );

                tokio::time::sleep(Duration::from_millis(delay_ms as u64)).await;
            }
        }
    }

    unreachable!("Retry loop exited unexpectedly")
}

/// Circuit breaker state enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

const STATE_CLOSED: u8 = 0;
const STATE_OPEN: u8 = 1;
const STATE_HALF_OPEN: u8 = 2;

impl From<u8> for CircuitBreakerState {
    fn from(value: u8) -> Self {
        match value {
            STATE_OPEN => Self::Open,
            STATE_HALF_OPEN => Self::HalfOpen,
            _ => Self::Closed,
        }
    }
}

impl CircuitBreakerState {
    #[allow(dead_code)]
    fn as_u8(self) -> u8 {
        match self {
            Self::Closed => STATE_CLOSED,
            Self::Open => STATE_OPEN,
            Self::HalfOpen => STATE_HALF_OPEN,
        }
    }
}

/// Error returned by `CircuitBreaker::execute`: either the breaker rejected
/// the call (`Open`) or the wrapped block failed (`Inner`).
#[derive(Debug, thiserror::Error)]
pub enum BreakerError<E> {
    #[error("{0}")]
    Open(String),
    #[error(transparent)]
    Inner(E),
}

impl<E: IsRetryable> IsRetryable for BreakerError<E> {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Open(_) => false,
            Self::Inner(e) => e.is_retryable(),
        }
    }
}

/// Collapse a breaker outcome back into the crate's `Result`.
///
/// `CircuitBreaker::execute` reports "the breaker rejected this call" and "the
/// call itself failed" as two variants of [`BreakerError`], but every caller
/// wants a single `CoreError` to hand upstream. Keeping the mapping here (it
/// lived privately in `easyrag-llm/src/openai.rs` before Phase 4.3) means any
/// crate that adopts a breaker maps `Open` to the same
/// `CoreError::CircuitBreakerOpen` instead of inventing its own wording.
pub fn breaker_result<T>(
    result: Result<T, BreakerError<crate::CoreError>>,
) -> crate::Result<T> {
    match result {
        Ok(v) => Ok(v),
        Err(BreakerError::Open(msg)) => Err(crate::CoreError::CircuitBreakerOpen(msg)),
        Err(BreakerError::Inner(e)) => Err(e),
    }
}

/// Circuit breaker to prevent cascading failures.
///
/// When the failure count exceeds `failure_threshold` within the
/// `reset_timeout` window, the breaker opens and subsequent calls are
/// rejected immediately with [`BreakerError::Open`]. After `reset_timeout`
/// the breaker enters half-open state and allows one trial call.
///
/// Thread-safe implementation using atomic operations
///
pub struct CircuitBreaker {
    pub failure_threshold: usize,
    reset_timeout: Duration,
    pub name: String,
    failure_count: AtomicUsize,
    last_failure_time: AtomicI64,
    state: AtomicU8,
    trial_in_flight: AtomicBool,
    /// Start time (millis) of the in-flight half-open trial. Used to
    /// recover from a trial future that was dropped (cancelled) without
    /// ever reaching record_success/record_failure.
    trial_started_at: AtomicI64,
}

fn now_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, reset_timeout: Duration, name: impl Into<String>) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            name: name.into(),
            failure_count: AtomicUsize::new(0),
            last_failure_time: AtomicI64::new(0),
            state: AtomicU8::new(STATE_CLOSED),
            trial_in_flight: AtomicBool::new(false),
            trial_started_at: AtomicI64::new(0),
        }
    }

    /// Current circuit breaker state (with automatic open -> half-open
    /// transition once the reset timeout has elapsed).
    pub fn current_state(&self) -> CircuitBreakerState {
        if self.state.load(Ordering::SeqCst) == STATE_OPEN {
            let elapsed = now_millis() - self.last_failure_time.load(Ordering::SeqCst);
            if elapsed >= self.reset_timeout.as_millis() as i64 {
                let _ = self.state.compare_exchange(
                    STATE_OPEN,
                    STATE_HALF_OPEN,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            }
        }
        CircuitBreakerState::from(self.state.load(Ordering::SeqCst))
    }

    /// Record a successful call, resetting the breaker.
    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::SeqCst);
        self.state.store(STATE_CLOSED, Ordering::SeqCst);
        self.trial_in_flight.store(false, Ordering::SeqCst);
        self.trial_started_at.store(0, Ordering::SeqCst);
    }

    /// Record a failed call, potentially opening the breaker.
    pub fn record_failure(&self) {
        let now = now_millis();
        let current_state = self.state.load(Ordering::SeqCst);

        if current_state == STATE_HALF_OPEN {
            self.failure_count.store(self.failure_threshold, Ordering::SeqCst);
        } else {
            let last_time = self.last_failure_time.load(Ordering::SeqCst);
            if last_time > 0 && now - last_time >= self.reset_timeout.as_millis() as i64 {
                self.failure_count.store(0, Ordering::SeqCst);
            }
            self.failure_count.fetch_add(1, Ordering::SeqCst);
        }

        self.last_failure_time.store(now, Ordering::SeqCst);

        if self.failure_count.load(Ordering::SeqCst) >= self.failure_threshold {
            self.state.store(STATE_OPEN, Ordering::SeqCst);
            self.trial_in_flight.store(false, Ordering::SeqCst);
            self.trial_started_at.store(0, Ordering::SeqCst);
            tracing::warn!(
                "Circuit breaker '{}' opened after {} failures",
                self.name,
                self.failure_count.load(Ordering::SeqCst)
            );
        }
    }

    /// Execute an async function protected by the circuit breaker.
    pub async fn execute<T, E, F, Fut>(&self, block: F) -> Result<T, BreakerError<E>>
    where
        E: IsRetryable + std::fmt::Debug,
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        self.acquire_permission()?;
        match block().await {
            Ok(result) => {
                self.record_success();
                Ok(result)
            }
            Err(exc) => {
                if exc.is_retryable() {
                    self.record_failure();
                } else if self.state.load(Ordering::SeqCst) == STATE_HALF_OPEN {
                    // Non-transient error: clear half-open trial gate
                    self.trial_in_flight.store(false, Ordering::SeqCst);
                    self.trial_started_at.store(0, Ordering::SeqCst);
                }
                Err(BreakerError::Inner(exc))
            }
        }
    }

    fn acquire_permission<E>(&self) -> Result<(), BreakerError<E>> {
        let current_state = self.current_state(); // triggers open -> half-open transition

        match current_state {
            CircuitBreakerState::Open => Err(BreakerError::Open(format!(
                "Circuit breaker '{}' is open — call rejected",
                self.name
            ))),
            CircuitBreakerState::HalfOpen => {
                match self.trial_in_flight.compare_exchange(
                    false,
                    true,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        self.trial_started_at.store(now_millis(), Ordering::SeqCst);
                        Ok(())
                    }
                    Err(_) => {
                        // A trial is in progress. If it has exceeded the
                        // reset timeout it was most likely cancelled (its
                        // future dropped before recording an outcome); clear
                        // the stale gate so the breaker cannot wedge, then
                        // hand the gate to this call.
                        let started = self.trial_started_at.load(Ordering::SeqCst);
                        let stale = started > 0
                            && now_millis() - started >= self.reset_timeout.as_millis() as i64;
                        if stale
                            && self
                                .trial_in_flight
                                .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                                .is_ok()
                            && self
                                .trial_in_flight
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                                .is_ok()
                        {
                            self.trial_started_at.store(now_millis(), Ordering::SeqCst);
                            return Ok(());
                        }
                        Err(BreakerError::Open(format!(
                            "Circuit breaker '{}' is half-open — trial in progress",
                            self.name
                        )))
                    }
                }
            }
            CircuitBreakerState::Closed => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    // Ported from ResilienceTest.kt (13 cases).
    use super::*;
    use std::fmt;

    /// Transient (retryable) test error — equivalent to IOException.
    #[derive(Debug)]
    struct TransientError(String);

    impl fmt::Display for TransientError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl IsRetryable for TransientError {
        fn is_retryable(&self) -> bool {
            true
        }
    }

    /// Non-retryable test error — equivalent to IllegalArgumentException.
    #[derive(Debug)]
    struct FatalError(String);

    impl fmt::Display for FatalError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl IsRetryable for FatalError {
        fn is_retryable(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn with_retry_success_on_first_attempt_returns_result() {
        let config = RetryConfig::default()
            .with_max_attempts(3)
            .with_base_delay(Duration::from_millis(1));
        let result: Result<&str, TransientError> =
            with_retry(&config, None::<fn(&TransientError, u32, Duration)>, || async {
                Ok("success")
            })
            .await;
        assert_eq!("success", result.unwrap());
    }

    #[tokio::test]
    async fn with_retry_fails_then_succeeds_retries_and_returns() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let config = RetryConfig::default()
            .with_max_attempts(3)
            .with_base_delay(Duration::from_millis(1))
            .with_jitter(false);
        let a = attempts.clone();
        let result: Result<&str, TransientError> =
            with_retry(&config, None::<fn(&TransientError, u32, Duration)>, move || {
                let a = a.clone();
                async move {
                    let n = a.fetch_add(1, Ordering::SeqCst) + 1;
                    if n < 3 {
                        Err(TransientError("transient failure".to_string()))
                    } else {
                        Ok("recovered")
                    }
                }
            })
            .await;
        assert_eq!("recovered", result.unwrap());
        assert_eq!(3, attempts.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn with_retry_all_attempts_fail_throws_last_exception() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let config = RetryConfig::default()
            .with_max_attempts(3)
            .with_base_delay(Duration::from_millis(1))
            .with_jitter(false);
        let a = attempts.clone();
        let result: Result<&str, TransientError> =
            with_retry(&config, None::<fn(&TransientError, u32, Duration)>, move || {
                let a = a.clone();
                async move {
                    a.fetch_add(1, Ordering::SeqCst);
                    Err(TransientError("always fails".to_string()))
                }
            })
            .await;
        assert_eq!(3, attempts.load(Ordering::SeqCst));
        assert_eq!("always fails", result.unwrap_err().0);
    }

    #[tokio::test]
    async fn with_retry_non_retryable_exception_throws_immediately() {
        let attempts = std::sync::Arc::new(AtomicUsize::new(0));
        let config = RetryConfig::default()
            .with_max_attempts(5)
            .with_base_delay(Duration::from_millis(1));
        let a = attempts.clone();
        let result: Result<&str, FatalError> =
            with_retry(&config, None::<fn(&FatalError, u32, Duration)>, move || {
                let a = a.clone();
                async move {
                    a.fetch_add(1, Ordering::SeqCst);
                    Err(FatalError("not retryable".to_string()))
                }
            })
            .await;
        assert!(result.is_err());
        assert_eq!(1, attempts.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn with_retry_on_retry_callback_invoked() {
        let retry_log = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let config = RetryConfig::default()
            .with_max_attempts(3)
            .with_base_delay(Duration::from_millis(1))
            .with_jitter(false);
        let log = retry_log.clone();
        let result: Result<&str, TransientError> = with_retry(
            &config,
            Some(move |_e: &TransientError, attempt: u32, _d| {
                log.lock().unwrap().push(attempt);
            }),
            || async { Err(TransientError("connection refused".to_string())) },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(vec![1, 2], *retry_log.lock().unwrap());
    }

    #[test]
    fn retry_config_invalid_max_attempts_throws() {
        let result = RetryConfig::try_new(
            0,
            Duration::from_secs(1),
            Duration::from_secs(60),
            2.0,
            true,
        );
        assert!(result.is_err());
    }

    #[test]
    fn circuit_breaker_initial_state_is_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), "test");
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    #[test]
    fn circuit_breaker_below_threshold_stays_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), "test");
        cb.record_failure();
        cb.record_failure();
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    #[test]
    fn circuit_breaker_reaches_threshold_opens() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), "test");
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        assert_eq!(CircuitBreakerState::Open, cb.current_state());
    }

    #[tokio::test]
    async fn circuit_breaker_execute_success_resets_failure_count() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), "test");
        cb.record_failure();
        cb.record_failure();

        let result: Result<&str, BreakerError<TransientError>> =
            cb.execute(|| async { Ok("ok") }).await;
        assert_eq!("ok", result.unwrap());
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    #[tokio::test]
    async fn circuit_breaker_open_rejects_calls() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(60), "test");
        cb.record_failure();
        cb.record_failure();

        let result: Result<&str, BreakerError<TransientError>> =
            cb.execute(|| async { Ok("should not run") }).await;
        assert!(matches!(result, Err(BreakerError::Open(_))));
    }

    #[tokio::test]
    async fn circuit_breaker_execute_with_failure_records_failure() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(60), "test");
        let result: Result<&str, BreakerError<TransientError>> = cb
            .execute(|| async { Err(TransientError("fail".to_string())) })
            .await;
        assert!(matches!(result, Err(BreakerError::Inner(_))));
        // Should still be closed (1 failure < 5 threshold)
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    #[test]
    fn circuit_breaker_record_success_resets() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(60), "test");
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
        // After reset, need 3 more failures to open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    #[tokio::test]
    async fn circuit_breaker_cancelled_trial_recovers_after_timeout() {
        // A trial whose future is dropped (outer timeout / select! cancel)
        // never records an outcome; the gate must recover instead of wedging.
        let cb = CircuitBreaker::new(1, Duration::from_millis(50), "test");
        cb.record_failure(); // opens the breaker
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(CircuitBreakerState::HalfOpen, cb.current_state());

        // Start a trial and cancel it before it records an outcome.
        let cancelled = tokio::time::timeout(
            Duration::from_millis(10),
            cb.execute(|| async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok::<_, TransientError>("never")
            }),
        )
        .await;
        assert!(cancelled.is_err());

        // While the (fresh) trial is "in flight", other calls are rejected.
        let rejected: Result<&str, BreakerError<TransientError>> =
            cb.execute(|| async { Ok("should not run") }).await;
        assert!(matches!(rejected, Err(BreakerError::Open(_))));

        // Once the stale trial exceeds reset_timeout the gate is reclaimed
        // and the next call passes.
        tokio::time::sleep(Duration::from_millis(60)).await;
        let result: Result<&str, BreakerError<TransientError>> =
            cb.execute(|| async { Ok("recovered") }).await;
        assert_eq!("recovered", result.unwrap());
        assert_eq!(CircuitBreakerState::Closed, cb.current_state());
    }

    /// Phase 4.3: `breaker_result` moved here from `easyrag-llm/src/openai.rs`.
    /// These pin the mapping the OpenAI client already relied on — `Open`
    /// becomes `CoreError::CircuitBreakerOpen` carrying the breaker's message
    /// verbatim, `Inner` is passed through untouched, `Ok` is untouched.
    #[test]
    fn breaker_result_maps_open_to_circuit_breaker_open() {
        let outcome: Result<u8, BreakerError<crate::CoreError>> =
            Err(BreakerError::Open("llm-openai".to_string()));
        let err = breaker_result(outcome).unwrap_err();
        assert!(
            matches!(&err, crate::CoreError::CircuitBreakerOpen(msg) if msg == "llm-openai"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn breaker_result_passes_the_inner_error_through_unchanged() {
        let outcome: Result<u8, BreakerError<crate::CoreError>> =
            Err(BreakerError::Inner(crate::CoreError::ApiTimeout));
        assert!(matches!(
            breaker_result(outcome),
            Err(crate::CoreError::ApiTimeout)
        ));
    }

    #[test]
    fn breaker_result_keeps_success_values() {
        let outcome: Result<u8, BreakerError<crate::CoreError>> = Ok(7);
        assert_eq!(7u8, breaker_result(outcome).unwrap());
    }
}
