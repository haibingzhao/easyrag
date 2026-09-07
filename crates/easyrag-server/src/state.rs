//! Shared application state for all route handlers.

use std::sync::Arc;

use easyrag_core::storage::tenant_config::TenantConfigStorage;
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage, VectorStorage};
use easyrag_pipeline::{
    EasyRAG, IngestionPipeline, PipelineService, QueryEngine, RAGAnything, StructureQueryStrategy,
    TextParserRegistry,
};

use crate::auth::{JwtAuthProvider, LoginRateLimiter};
use crate::config::ServerConfig;
use crate::scan_stats::AutoScanStats;
use crate::service::{AuthService, DocumentLifecycleService, WorkspaceResolver};

/// All services shared by route handlers. Constructed explicitly by the
/// application crate (no DI container).
/// controller constructor dependency.
pub struct AppServices {
    pub config: ServerConfig,
    /// Unified facade.
    pub easy_rag: Arc<EasyRAG>,
    pub query_engine: Arc<QueryEngine>,
    /// Optional: `/api/query/structure` returns 500 when absent
    ///
    pub structure_strategy: Option<Arc<StructureQueryStrategy>>,
    pub pipeline: Arc<IngestionPipeline>,
    /// Streaming ingestion service (BC-11 gradual rollout). `Some` only when
    /// `PIPELINE_STREAMING_ENABLED` is set; handlers then drive the persistent
    /// discovery/processor loop instead of the one-shot batch run. `None` keeps
    /// the exact pre-streaming behaviour.
    pub pipeline_service: Option<Arc<PipelineService>>,
    pub vector_storage: Arc<dyn VectorStorage>,
    pub graph_storage: Arc<dyn GraphStorage>,
    pub kv_storage: Arc<dyn KVStorage>,
    pub doc_status_storage: Arc<dyn DocStatusStorage>,
    pub parser_registry: Arc<TextParserRegistry>,
    /// Document lifecycle orchestration: R3 upsert / R5 full text / R6 index.
    pub lifecycle_service: Arc<DocumentLifecycleService>,
    /// R2 request-level workspace routing.
    pub workspace_resolver: Arc<dyn WorkspaceResolver>,
    /// Optional multimodal stack ( is
    /// conditionally registered).
    pub rag_anything: Option<Arc<RAGAnything>>,
    /// Present only when auth is enabled.
    pub jwt: Option<Arc<JwtAuthProvider>>,
    pub rate_limiter: Arc<LoginRateLimiter>,
    /// Credential verification + default admin seeding (always wired so the
    /// `/api/auth/login` endpoint is available regardless of `auth_enabled`).
    pub auth_service: Arc<AuthService>,
    /// Per-workspace tenant configuration profile storage.
    pub tenant_config_storage: Arc<dyn TenantConfigStorage>,
    /// Background auto-scanner observability (BC-11). Always present so the
    /// `/api/documents/auto_scan_status` endpoint works even when the scanner
    /// is disabled (it then reports `enabled: false` with empty tasks/cycles).
    pub auto_scan_stats: Arc<AutoScanStats>,
}

/// Axum state extractor used by every handler.
pub type AppState = axum::extract::State<Arc<AppServices>>;
