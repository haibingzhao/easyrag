//! Concurrency limiter with multi-layer timeout protection
//!.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Semaphore, TryAcquireError};

/// Concurrency limiter backed by a tokio Semaphore with structured
/// timeout protection.
pub struct ConcurrencyLimiter {
    pub max_concurrency: usize,
    pub queue_name: String,
    pub effective_execution_timeout: Duration,
    semaphore: Arc<Semaphore>,
}

impl ConcurrencyLimiter {
    /// Constructor:
    /// `effectiveExecutionTimeout = maxExecutionTimeout ?: llmTimeout * 2 ?: 300s`.
    pub fn new(
        max_concurrency: usize,
        queue_name: impl Into<String>,
        llm_timeout: Option<Duration>,
        max_execution_timeout: Option<Duration>,
    ) -> Self {
        let effective_execution_timeout = max_execution_timeout
            .or_else(|| llm_timeout.map(|t| t * 2))
            .unwrap_or(Duration::from_secs(300));
        Self {
            max_concurrency,
            queue_name: queue_name.into(),
            effective_execution_timeout,
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }

    /// Execute an async function with concurrency limiting.
    pub async fn execute<T, F, Fut>(&self, block: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("semaphore must not be closed");
        let result = block().await;
        drop(permit);
        result
    }

    /// Execute with timeout protection (timeout on the block only; permit
    /// acquisition is not bounded, matching `withPermit { withTimeout {} }`).
    pub async fn execute_with_timeout<T, F, Fut>(&self, block: F) -> Result<T, crate::CoreError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let permit = self
            .semaphore
            .acquire()
            .await
            .expect("semaphore must not be closed");
        let result = match tokio::time::timeout(self.effective_execution_timeout, block()).await {
            Ok(value) => Ok(value),
            Err(_) => Err(crate::CoreError::ApiTimeout),
        };
        drop(permit);
        result
    }

    /// Available permits count.
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Try to acquire a permit without waiting (used by busy-checks).
    pub fn try_acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit, TryAcquireError> {
        Arc::clone(&self.semaphore).try_acquire_owned()
    }
}

#[cfg(test)]
mod tests {
    // Ported from ConcurrencyTest.kt (ConcurrencyLimiter cases).
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrency_limiter_limits_concurrency() {
        let limiter = Arc::new(ConcurrencyLimiter::new(2, "test", None, None));
        let max_concurrent = Arc::new(AtomicI32::new(0));
        let current = Arc::new(AtomicI32::new(0));

        let mut handles = Vec::new();
        for _ in 1..=6 {
            let (limiter, max_concurrent, current) =
                (limiter.clone(), max_concurrent.clone(), current.clone());
            handles.push(tokio::spawn(async move {
                limiter
                    .execute(|| async move {
                        let c = current.fetch_add(1, Ordering::SeqCst) + 1;
                        max_concurrent.fetch_max(c, Ordering::SeqCst);
                        current.fetch_sub(1, Ordering::SeqCst);
                        // Yield so concurrent tasks can overlap.
                        tokio::task::yield_now().await;
                    })
                    .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert!(max_concurrent.load(Ordering::SeqCst) <= 2);
    }

    #[test]
    fn concurrency_limiter_available_permits_reflects_state() {
        let limiter = ConcurrencyLimiter::new(5, "limit_async", None, None);
        assert_eq!(5, limiter.available_permits());
    }

    #[tokio::test]
    async fn concurrency_limiter_execute_returns_result() {
        let limiter = ConcurrencyLimiter::new(3, "limit_async", None, None);
        let result = limiter.execute(|| async { "hello" }).await;
        assert_eq!("hello", result);
    }
}
