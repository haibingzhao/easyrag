//! Entity-level fine-grained lock.
//! DashMap with reference counting, equivalent to ConcurrentHashMap + Mutex.

use dashmap::DashMap;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{Mutex, OwnedMutexGuard};

struct MutexEntry {
    mutex: Arc<Mutex<()>>,
    ref_count: AtomicUsize,
}

impl MutexEntry {
    fn new() -> Self {
        Self {
            mutex: Arc::new(Mutex::new(())),
            ref_count: AtomicUsize::new(0),
        }
    }
}

/// Entity-level fine-grained lock using a concurrent map with reference
/// counting.
///
/// The map is shared through an `Arc`: `DashMap::clone` deep-copies the
/// map, so without the `Arc` cloned handles would not share lock state
///
#[derive(Clone, Default)]
pub struct KeyedMutex {
    locks: Arc<DashMap<String, Arc<MutexEntry>>>,
}

impl KeyedMutex {
    pub fn new() -> Self {
        Self {
            locks: Arc::new(DashMap::new()),
        }
    }

    /// Acquire a mutex for the given key. Returns a handle to release.
    ///
    /// The reference count increment happens while holding the DashMap shard
    /// lock so that `release` can never remove the entry out from under a
    /// concurrent acquirer (which would break mutual exclusion by letting
    /// two different mutex entries exist for the same key).
    pub async fn acquire(&self, key: &str) -> KeyedLockHandle {
        let entry = {
            let guard = self
                .locks
                .entry(key.to_string())
                .or_insert_with(|| Arc::new(MutexEntry::new()));
            guard.ref_count.fetch_add(1, Ordering::SeqCst);
            Arc::clone(&*guard)
        };
        let permit = Arc::clone(&entry.mutex).lock_owned().await;
        KeyedLockHandle {
            key: key.to_string(),
            _permit: permit,
            locks: Arc::clone(&self.locks),
        }
    }

    /// Execute a block while holding the lock for the given key.
    pub async fn with_lock<T, F, Fut>(&self, key: &str, block: F) -> T
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        let handle = self.acquire(key).await;
        let result = block().await;
        handle.release();
        result
    }
}

/// Handle for a keyed lock that can be released.
pub struct KeyedLockHandle {
    pub key: String,
    _permit: OwnedMutexGuard<()>,
    locks: Arc<DashMap<String, Arc<MutexEntry>>>,
}

impl KeyedLockHandle {
    pub fn release(self) {
        // Unlock first, then decrement the reference count.
        // The decrement and conditional removal happen atomically inside
        // the shard write lock (`remove_if` predicate), serialized with
        // `acquire`'s `entry` call, so no acquirer can observe a removed
        // entry or resurrect a stale one.
        let Self { key, _permit, locks } = self;
        drop(_permit);
        locks.remove_if(&key, |_, entry| {
            entry.ref_count.fetch_sub(1, Ordering::SeqCst) <= 1
        });
    }
}

#[cfg(test)]
mod tests {
    // Ported from ConcurrencyTest.kt (KeyedMutex cases).
    use super::*;
    use std::sync::atomic::AtomicI32;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn keyed_mutex_with_lock_mutual_exclusion() {
        // Run repeatedly: this guards the acquire/release reference-count
        // race where a removed-and-recreated entry used to break exclusion.
        for _ in 0..50 {
            let mutex = KeyedMutex::new();
            let counter = Arc::new(AtomicI32::new(0));
            let max_concurrent = Arc::new(AtomicI32::new(0));
            let current = Arc::new(AtomicI32::new(0));

            let mut handles = Vec::new();
            for _ in 1..=10 {
                let mutex = mutex.clone();
                let (counter, max_concurrent, current) =
                    (counter.clone(), max_concurrent.clone(), current.clone());
                handles.push(tokio::spawn(async move {
                    mutex
                        .with_lock("key1", || async move {
                            let c = current.fetch_add(1, Ordering::SeqCst) + 1;
                            max_concurrent.fetch_max(c, Ordering::SeqCst);
                            counter.fetch_add(1, Ordering::SeqCst);
                            current.fetch_sub(1, Ordering::SeqCst);
                        })
                        .await;
                }));
            }
            for h in handles {
                h.await.unwrap();
            }

            assert_eq!(10, counter.load(Ordering::SeqCst));
            assert_eq!(1, max_concurrent.load(Ordering::SeqCst));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn keyed_mutex_different_keys_can_run_concurrently() {
        let mutex = KeyedMutex::new();
        let results = Arc::new(DashMap::<i32, bool>::new());

        let mut handles = Vec::new();
        for i in 1..=5 {
            let mutex = mutex.clone();
            let results = results.clone();
            handles.push(tokio::spawn(async move {
                mutex
                    .with_lock(&format!("key{i}"), || async move {
                        results.insert(i, true);
                    })
                    .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(5, results.len());
    }

    #[tokio::test]
    async fn keyed_mutex_acquire_release_works() {
        let mutex = KeyedMutex::new();
        let handle = mutex.acquire("testKey").await;
        assert_eq!("testKey", handle.key);
        handle.release();
    }
}
