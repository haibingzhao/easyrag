//! Pipeline state lock.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;

/// Pipeline state lock for coordinating pipeline operations.
#[derive(Default)]
pub struct PipelineLock {
    mutex: Mutex<()>,
    locked: AtomicBool,
}

impl PipelineLock {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn with_lock<T, F, Fut>(&self, block: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let _mutex_guard = self.mutex.lock().await;
        self.locked.store(true, Ordering::SeqCst);
        // Clear the observable flag even if `block` is cancelled (future
        // dropped); the mutex guard itself still provides exclusion.
        struct LockedFlag<'a>(&'a AtomicBool);
        impl Drop for LockedFlag<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::SeqCst);
            }
        }
        let _flag_guard = LockedFlag(&self.locked);
        block().await
    }

    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    // Ported from ConcurrencyTest.kt (PipelineLock cases).
    use super::*;

    #[test]
    fn pipeline_lock_initially_unlocked() {
        let lock = PipelineLock::new();
        assert!(!lock.is_locked());
    }

    #[tokio::test]
    async fn pipeline_lock_with_lock_executes_block() {
        let lock = PipelineLock::new();
        let result = lock.with_lock(|| async { 42 }).await;
        assert_eq!(42, result);
        assert!(!lock.is_locked());
    }
}
