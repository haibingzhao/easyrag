//! Role-based LLM manager with per-role concurrency control

use easyrag_core::llm::service::{LlmRequest, LlmService};
use futures::stream::BoxStream;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// LLM role types for different pipeline stages.
/// Each role can have independent concurrency limits and timeout
/// configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LlmRole {
    Extract,
    Query,
    Summary,
    Keywords,
    #[default]
    Default,
}

impl LlmRole {
    pub fn value(&self) -> &'static str {
        match self {
            Self::Extract => "extract",
            Self::Query => "query",
            Self::Summary => "summary",
            Self::Keywords => "keywords",
            Self::Default => "default",
        }
    }

    /// Unknown values fall back to `Default`.
    pub fn from_value(value: &str) -> Self {
        match value {
            "extract" => Self::Extract,
            "query" => Self::Query,
            "summary" => Self::Summary,
            "keywords" => Self::Keywords,
            _ => Self::Default,
        }
    }
}

/// Configuration for a specific LLM role.
#[derive(Clone, Debug)]
pub struct RoleConfig {
    pub max_concurrency: usize,
    pub timeout: Duration,
    pub max_retries: u32,
    pub temperature: f32,
    pub model: Option<String>,
}

impl Default for RoleConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 4,
            timeout: Duration::from_secs(120),
            max_retries: 3,
            temperature: 0.0,
            model: None,
        }
    }
}

/// Role-based LLM manager with per-role concurrency control.
/// Wraps an [`LlmService`] and applies role-specific configurations.
pub struct RoleLlmManager {
    base_llm_service: Arc<dyn LlmService>,
    role_configs: HashMap<LlmRole, RoleConfig>,
    default_config: RoleConfig,
    semaphores: Mutex<HashMap<LlmRole, Arc<Semaphore>>>,
}

impl RoleLlmManager {
    pub fn new(
        base_llm_service: Arc<dyn LlmService>,
        role_configs: HashMap<LlmRole, RoleConfig>,
    ) -> Self {
        Self {
            base_llm_service,
            role_configs,
            default_config: RoleConfig::default(),
            semaphores: Mutex::new(HashMap::new()),
        }
    }

    fn get_semaphore(&self, role: LlmRole) -> Arc<Semaphore> {
        let mut semaphores = self.semaphores.lock().unwrap();
        semaphores
            .entry(role)
            .or_insert_with(|| {
                let config = self.role_configs.get(&role).unwrap_or(&self.default_config);
                Arc::new(Semaphore::new(config.max_concurrency.max(1)))
            })
            .clone()
    }

    fn get_config(&self, role: LlmRole) -> &RoleConfig {
        self.role_configs.get(&role).unwrap_or(&self.default_config)
    }

    /// Complete a request with a specific role.
    pub async fn complete_with_role(
        &self,
        role: LlmRole,
        request: LlmRequest,
    ) -> easyrag_core::Result<String> {
        let semaphore = self.get_semaphore(role);
        let config = self.get_config(role).clone();

        // copy request only when the role config overrides
        // temperature (model override is recorded but not applied to the
        // request body by the original).
        let effective_request = if config.model.is_some() || config.temperature > 0.0 {
            let mut r = request;
            if r.temperature == 0.0 {
                r.temperature = config.temperature;
            }
            r
        } else {
            request
        };

        let _permit: OwnedSemaphorePermit = Semaphore::acquire_owned(semaphore)
            .await
            .map_err(|e| easyrag_core::CoreError::Internal(format!("role semaphore: {e}")))?;
        tracing::debug!("Executing LLM request with role: {}", role.value());
        self.base_llm_service.complete(&effective_request).await
    }

    /// Stream a request with a specific role.
    pub fn complete_stream_with_role(&self, role: LlmRole, request: LlmRequest) -> BoxStream<'static, String> {
        let config = self.get_config(role);
        let effective_request = if config.temperature > 0.0 && request.temperature == 0.0 {
            let mut r = request;
            r.temperature = config.temperature;
            r
        } else {
            request
        };
        self.base_llm_service.complete_stream(effective_request)
    }
}

/// LlmService implementation using the DEFAULT role.
#[async_trait::async_trait]
impl LlmService for RoleLlmManager {
    async fn complete(&self, request: &LlmRequest) -> easyrag_core::Result<String> {
        self.complete_with_role(LlmRole::Default, request.clone()).await
    }

    fn complete_stream(&self, request: LlmRequest) -> BoxStream<'static, String> {
        self.complete_stream_with_role(LlmRole::Default, request)
    }
}
