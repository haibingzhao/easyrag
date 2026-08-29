//! LlmCache and RoleLlmManager tests: cache semantics, stats, role
//! routing, temperature override rules, per-role concurrency limits.

use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_llm::{LlmCache, LlmRole, RoleConfig, RoleLlmManager};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

// ---------------------------------------------------------------------
// LlmCache
// ---------------------------------------------------------------------

#[tokio::test]
async fn cache_put_get_roundtrip_and_stats() {
    let cache = LlmCache::new(100, Duration::from_secs(60), true);
    assert!(cache.get("local", "query", "k").await.is_none());

    cache.put("local", "query", "k", "answer".to_string()).await;
    assert_eq!(
        Some("answer".to_string()),
        cache.get("local", "query", "k").await
    );

    cache.run_pending().await;
    let stats = cache.stats();
    assert_eq!(1, stats["hitCount"]);
    assert_eq!(1, stats["missCount"]);
    assert_eq!(0.5, stats["hitRate"]);
    assert_eq!(1, stats["size"]);
    assert_eq!(0, stats["evictionCount"]);
}

#[tokio::test]
async fn cache_keys_are_isolated_by_mode_and_type() {
    let cache = LlmCache::new(100, Duration::from_secs(60), true);
    cache.put("local", "query", "same", "v1".to_string()).await;
    cache.put("global", "query", "same", "v2".to_string()).await;
    cache
        .put("local", "extraction", "same", "v3".to_string())
        .await;

    assert_eq!(Some("v1".to_string()), cache.get("local", "query", "same").await);
    assert_eq!(Some("v2".to_string()), cache.get("global", "query", "same").await);
    assert_eq!(
        Some("v3".to_string()),
        cache.get("local", "extraction", "same").await
    );
    cache.run_pending().await;
    assert_eq!(3, cache.stats()["size"]);
}

#[tokio::test]
async fn cache_invalidate_and_clear() {
    let cache = LlmCache::new(100, Duration::from_secs(60), true);
    cache.put("local", "query", "a", "1".to_string()).await;
    cache.put("local", "query", "b", "2".to_string()).await;

    cache.invalidate("local", "query", "a").await;
    assert!(cache.get("local", "query", "a").await.is_none());
    assert_eq!(Some("2".to_string()), cache.get("local", "query", "b").await);

    cache.clear().await;
    cache.run_pending().await;
    assert_eq!(0, cache.stats()["size"]);
}

#[tokio::test]
async fn disabled_cache_stores_nothing() {
    let cache = LlmCache::new(100, Duration::from_secs(60), false);
    cache.put("local", "query", "k", "answer".to_string()).await;
    assert!(cache.get("local", "query", "k").await.is_none());
}

#[tokio::test]
async fn cache_evicts_beyond_max_size() {
    let cache = LlmCache::new(1, Duration::from_secs(60), true);
    cache.put("local", "query", "a", "1".to_string()).await;
    cache.put("local", "query", "b", "2".to_string()).await;
    cache.run_pending().await;

    let stats = cache.stats();
    assert_eq!(1, stats["size"]);
    assert!(stats["evictionCount"].as_u64().unwrap() >= 1);
}

// ---------------------------------------------------------------------
// LlmRole
// ---------------------------------------------------------------------

#[test]
fn role_value_roundtrip_and_unknown_fallback() {
    for role in [
        LlmRole::Extract,
        LlmRole::Query,
        LlmRole::Summary,
        LlmRole::Keywords,
        LlmRole::Default,
    ] {
        assert_eq!(role, LlmRole::from_value(role.value()));
    }
    assert_eq!(LlmRole::Default, LlmRole::from_value("bogus"));
    assert_eq!(LlmRole::Default, LlmRole::default());
}

// ---------------------------------------------------------------------
// RoleLlmManager
// ---------------------------------------------------------------------

/// Fake LLM capturing effective requests; optionally sleeps to test
/// concurrency limits.
struct FakeLlm {
    captured: Mutex<Vec<LlmRequest>>,
    sleep: Duration,
    inflight: AtomicUsize,
    max_inflight: AtomicUsize,
}

impl FakeLlm {
    fn new(sleep: Duration) -> Self {
        Self {
            captured: Mutex::new(Vec::new()),
            sleep,
            inflight: AtomicUsize::new(0),
            max_inflight: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl LlmService for FakeLlm {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        self.captured.lock().unwrap().push(request.clone());
        let now = self.inflight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_inflight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep(self.sleep).await;
        self.inflight.fetch_sub(1, Ordering::SeqCst);
        Ok("done".to_string())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

#[tokio::test]
async fn role_temperature_overrides_only_zero_requests() {
    let fake = Arc::new(FakeLlm::new(Duration::ZERO));
    let mut configs = HashMap::new();
    configs.insert(
        LlmRole::Extract,
        RoleConfig {
            temperature: 0.8,
            ..RoleConfig::default()
        },
    );
    let manager = RoleLlmManager::new(fake.clone(), configs);

    // Request temperature 0.0 -> overridden by role config.
    manager
        .complete_with_role(LlmRole::Extract, LlmRequest::new("a"))
        .await
        .unwrap();
    // Request temperature non-zero -> preserved.
    manager
        .complete_with_role(LlmRole::Extract, LlmRequest::new("b").with_temperature(0.3))
        .await
        .unwrap();
    // Role without config -> untouched.
    manager
        .complete_with_role(LlmRole::Query, LlmRequest::new("c"))
        .await
        .unwrap();

    let captured = fake.captured.lock().unwrap();
    assert!((captured[0].temperature - 0.8).abs() < 1e-6);
    assert!((captured[1].temperature - 0.3).abs() < 1e-6);
    assert_eq!(0.0, captured[2].temperature);
}

#[tokio::test]
async fn role_max_concurrency_is_enforced() {
    let fake = Arc::new(FakeLlm::new(Duration::from_millis(30)));
    let mut configs = HashMap::new();
    configs.insert(
        LlmRole::Extract,
        RoleConfig {
            max_concurrency: 2,
            ..RoleConfig::default()
        },
    );
    let manager = Arc::new(RoleLlmManager::new(fake.clone(), configs));

    let mut handles = Vec::new();
    for _ in 0..6 {
        let m = manager.clone();
        handles.push(tokio::spawn(async move {
            m.complete_with_role(LlmRole::Extract, LlmRequest::new("x"))
                .await
        }));
    }
    for h in handles {
        h.await.unwrap().unwrap();
    }
    let max_inflight = fake.max_inflight.load(Ordering::SeqCst);
    assert_eq!(2, max_inflight, "per-role semaphore must cap concurrency");
    assert_eq!(6, fake.captured.lock().unwrap().len());
}

#[tokio::test]
async fn manager_as_llm_service_routes_through_default_role() {
    let fake = Arc::new(FakeLlm::new(Duration::ZERO));
    let manager = RoleLlmManager::new(fake.clone(), HashMap::new());
    let service: Arc<dyn LlmService> = Arc::new(manager);

    let result = service.complete(&LlmRequest::new("hello")).await.unwrap();
    assert_eq!("done", result);
    let captured = fake.captured.lock().unwrap();
    assert_eq!(1, captured.len());
    assert_eq!("hello", captured[0].prompt);
}
