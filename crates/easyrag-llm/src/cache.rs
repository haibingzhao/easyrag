//! LLM response cache.
//! Caffeine local cache is replaced by moka (same semantics: max size +
//! expire-after-write). Flattened key format: `{mode}:{cache_type}:{hash}`.

use easyrag_core::util::hash::{compute_mdhash_id, generate_cache_key};
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Default max cache entries.
pub const DEFAULT_MAX_SIZE: u64 = 10_000;
/// Default entry TTL.
pub const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

/// LLM response cache with a moka local cache.
pub struct LlmCache {
    local_cache: moka::future::Cache<String, String>,
    enabled: AtomicBool,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
    eviction_count: Arc<AtomicU64>,
}

impl Default for LlmCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SIZE, DEFAULT_TTL, true)
    }
}

impl LlmCache {
    pub fn new(max_size: u64, ttl: Duration, enabled: bool) -> Self {
        let eviction_count = Arc::new(AtomicU64::new(0));
        let evictions = Arc::clone(&eviction_count);
        let local_cache = moka::future::Cache::builder()
            .max_capacity(max_size)
            .time_to_live(ttl)
            .eviction_listener(move |_key, _value, _cause| {
                evictions.fetch_add(1, Ordering::SeqCst);
            })
            .build();
        Self {
            local_cache,
            enabled: AtomicBool::new(enabled),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
            eviction_count,
        }
    }

    /// Get cached response.
    /// `mode`: query mode (local, global, hybrid, naive, mix);
    /// `cache_type`: cache type (query, extraction, keywords);
    /// `key`: input key (will be hashed).
    pub async fn get(&self, mode: &str, cache_type: &str, key: &str) -> Option<String> {
        if !self.enabled.load(Ordering::SeqCst) {
            return None;
        }
        let cache_key = Self::build_key(mode, cache_type, key);
        let result = self.local_cache.get(&cache_key).await;
        if result.is_some() {
            self.hit_count.fetch_add(1, Ordering::SeqCst);
            tracing::debug!("Cache hit: {cache_key}");
        } else {
            self.miss_count.fetch_add(1, Ordering::SeqCst);
        }
        result
    }

    /// Store response in cache.
    pub async fn put(&self, mode: &str, cache_type: &str, key: &str, value: String) {
        if !self.enabled.load(Ordering::SeqCst) {
            return;
        }
        let cache_key = Self::build_key(mode, cache_type, key);
        self.local_cache.insert(cache_key.clone(), value).await;
        tracing::debug!("Cache put: {cache_key}");
    }

    /// Invalidate a specific cache entry.
    pub async fn invalidate(&self, mode: &str, cache_type: &str, key: &str) {
        let cache_key = Self::build_key(mode, cache_type, key);
        self.local_cache.invalidate(&cache_key).await;
    }

    /// Clear all cache entries.
    pub async fn clear(&self) {
        self.local_cache.invalidate_all();
        tracing::info!("LLM cache cleared");
    }

    /// Run pending moka maintenance tasks (evictions, listeners).
    /// Useful for deterministic tests.
    pub async fn run_pending(&self) {
        self.local_cache.run_pending_tasks().await;
    }

    /// Cache statistics (Caffeine-style camelCase keys, matching the
    /// map).
    pub fn stats(&self) -> JsonMap {
        let hits = self.hit_count.load(Ordering::SeqCst);
        let misses = self.miss_count.load(Ordering::SeqCst);
        let requests = hits + misses;
        let hit_rate = if requests == 0 {
            0.0
        } else {
            hits as f64 / requests as f64
        };
        let mut stats = JsonMap::new();
        stats.insert("hitCount".to_string(), JsonValue::from(hits));
        stats.insert("missCount".to_string(), JsonValue::from(misses));
        stats.insert("hitRate".to_string(), JsonValue::from(hit_rate));
        stats.insert(
            "evictionCount".to_string(),
            JsonValue::from(self.eviction_count.load(Ordering::SeqCst)),
        );
        stats.insert(
            "size".to_string(),
            JsonValue::from(self.local_cache.entry_count()),
        );
        stats
    }

    /// Flattened key: `{mode}:{cache_type}:{md5(key)}`
    ///
    fn build_key(mode: &str, cache_type: &str, key: &str) -> String {
        let hash = compute_mdhash_id(key, "");
        generate_cache_key(mode, cache_type, &hash)
    }
}
