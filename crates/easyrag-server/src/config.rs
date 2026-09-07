//! Server configuration properties.
//!
//! Defaults that also exist in `easyrag_core::constants` reference those
//! constants directly to avoid drift between the two crates.

use easyrag_core::model::ScanTaskSpec;
use easyrag_core::storage::tenant_config::TenantConfig;

/// Server configuration properties.
/// Field names follow the `easyrag.*` configuration keys.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    // Server settings
    pub host: String,
    pub port: u16,
    pub workers: usize,

    // Authentication
    pub auth_enabled: bool,
    pub jwt_secret: String,
    pub jwt_expiration_hours: u64,
    pub login_rate_limit_per_minute: usize,

    // CORS
    pub cors_origins: Vec<String>,

    // Storage
    pub working_dir: String,
    pub workspace: String,
    /// Server-level default business line; request-level biz_id overrides it.
    pub biz_id: String,
    pub storage_type: String,

    // LLM
    pub llm_model: String,
    pub llm_api_key: String,
    pub llm_base_url: String,
    pub llm_temperature: f32,
    pub llm_max_tokens: Option<u32>,
    /// DashScope-style thinking toggle; None = provider default.
    pub llm_enable_thinking: Option<bool>,
    /// Per-request LLM timeout in seconds (larger local/slow models need more
    /// than the 120s client default).
    pub llm_timeout_secs: u64,

    // Embedding
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub embedding_api_key: String,
    pub embedding_base_url: String,
    pub embedding_query_prefix: String,
    pub embedding_document_prefix: String,

    // Chunking
    pub chunk_size: usize,
    pub chunk_overlap_size: usize,
    pub chunking_method: String,

    // Entity extraction
    pub entity_extract_max_gleaning: usize,
    /// empty = extractor's built-in entity-type guidance
    pub entity_types_guidance: String,

    // Query defaults
    pub default_top_k: usize,
    pub default_mode: String,

    // Concurrency
    pub llm_max_async: usize,
    pub embedding_max_async: usize,
    pub pipeline_max_parallel: usize,

    // PostgreSQL (if storage_type = "postgres")
    pub pg_host: String,
    pub pg_port: u16,
    pub pg_user: String,
    pub pg_password: String,
    pub pg_database: String,
    // P2.14: PG SSL/TLS
    pub pg_ssl_mode: Option<String>,
    pub pg_ssl_cert: Option<String>,
    pub pg_ssl_key: Option<String>,
    pub pg_ssl_root_cert: Option<String>,
    pub pg_ssl_crl: Option<String>,

    // Rerank
    pub rerank_enabled: bool,
    pub rerank_provider: String,
    pub rerank_api_key: String,
    pub rerank_model: String,

    // Neo4j (if graph_storage_type = "neo4j")
    /// empty = follow storage_type, "neo4j" = use Neo4j
    pub graph_storage_type: String,
    pub neo4j_uri: String,
    pub neo4j_user: String,
    pub neo4j_password: String,
    pub neo4j_database: String,
    pub neo4j_max_pool_size: usize,

    // Cache
    pub cache_enabled: bool,
    pub cache_max_size: u64,
    pub cache_ttl_hours: u64,

    // Full-text / Hybrid keyword search
    pub keyword_search_enabled: bool,
    /// PG tsvector config: simple, english, zhparser
    pub keyword_search_language: String,
    pub keyword_search_top_k: usize,
    pub hybrid_rrf_k: usize,
    /// empty = workingDir/lucene_index
    pub lucene_index_dir: String,

    // Misc
    pub language: String,
    pub verbose: bool,
    pub log_level: String,

    // Background auto-scan
    pub auto_scan_enabled: bool,
    pub auto_scan_tasks: Vec<ScanTaskSpec>,
    pub auto_scan_interval_seconds: u64,
    pub auto_scan_input_dir: String,
    pub auto_scan_extensions: String,
    pub auto_scan_max_files_per_cycle: usize,

    // Streaming ingestion pipeline (BC-11 gradual rollout). Default-off keeps
    // the batch `run_pipeline` path byte-for-byte equivalent to the status quo;
    // enabling it swaps in the discovery→channel→processor `PipelineService`.
    pub pipeline_streaming_enabled: bool,
    /// DocDiscovery idle poll interval in milliseconds (adaptive backoff floor).
    pub pipeline_poll_interval_ms: u64,
    /// Bounded mpsc channel capacity between discovery and processor.
    pub pipeline_channel_capacity: usize,
}

impl ServerConfig {
    /// Merge a tenant config profile into this global config, returning a
    /// new "resolved" `ServerConfig` for that workspace. Fields left as
    /// `None` in the tenant config fall back to the global `.env` defaults.
    pub fn merge_tenant_config(&self, tenant: &TenantConfig) -> Self {
        let mut resolved = self.clone();
        if let Some(ref v) = tenant.llm_model {
            resolved.llm_model = v.clone();
        }
        if let Some(ref v) = tenant.llm_api_key {
            resolved.llm_api_key = v.clone();
        }
        if let Some(ref v) = tenant.llm_base_url {
            resolved.llm_base_url = v.clone();
        }
        if let Some(v) = tenant.llm_temperature {
            resolved.llm_temperature = v;
        }
        if let Some(v) = tenant.llm_max_tokens {
            resolved.llm_max_tokens = Some(v);
        }
        if let Some(ref v) = tenant.embedding_model {
            resolved.embedding_model = v.clone();
        }
        if let Some(ref v) = tenant.embedding_api_key {
            resolved.embedding_api_key = v.clone();
        }
        if let Some(ref v) = tenant.embedding_base_url {
            resolved.embedding_base_url = v.clone();
        }
        if let Some(v) = tenant.embedding_dim {
            resolved.embedding_dim = v;
        }
        if let Some(v) = tenant.chunk_size {
            resolved.chunk_size = v;
        }
        if let Some(v) = tenant.chunk_overlap_size {
            resolved.chunk_overlap_size = v;
        }
        if let Some(ref v) = tenant.language {
            resolved.language = v.clone();
        }
        if let Some(v) = tenant.default_top_k {
            resolved.default_top_k = v;
        }
        if let Some(v) = tenant.rerank_enabled {
            resolved.rerank_enabled = v;
        }
        resolved
    }
}

/// The default JWT secret. Enabling auth while keeping it is an
/// error.
pub const DEFAULT_JWT_SECRET: &str = "easyrag-secret-key-change-in-production";

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8020,
            workers: 1,

            auth_enabled: true,
            jwt_secret: DEFAULT_JWT_SECRET.to_string(),
            jwt_expiration_hours: 24,
            login_rate_limit_per_minute: 5,

            cors_origins: vec!["*".to_string()],

            working_dir: "./rag_storage".to_string(),
            workspace: "default".to_string(),
            biz_id: easyrag_core::constants::DEFAULT_BIZ_ID.to_string(),
            // ServerConfig.kt data-class default; application.yml further
            // overrides it via ${STORAGE_BACKEND:memory} at app startup.
            storage_type: "json".to_string(),

            // application.yml overrides this with ${LLM_MODEL:gpt-4o-mini}
            // at app startup (see easyrag-app build_config).
            llm_model: "gpt-4o".to_string(),
            llm_api_key: String::new(),
            llm_base_url: "https://api.openai.com/v1".to_string(),
            llm_temperature: 0.0,
            llm_max_tokens: None,
            llm_enable_thinking: None,
            llm_timeout_secs: 120,

            embedding_model: "text-embedding-3-small".to_string(),
            embedding_dim: 1536,
            embedding_api_key: String::new(),
            embedding_base_url: "https://api.openai.com/v1".to_string(),
            embedding_query_prefix: String::new(),
            embedding_document_prefix: String::new(),

            chunk_size: 1200,
            chunk_overlap_size: 100,
            chunking_method: "token_size".to_string(),

            // Kotlin blueprint defaults for extraction settings; the
            // gleaning constant is u32 on the core side.
            entity_extract_max_gleaning:
                easyrag_core::constants::DEFAULT_MAX_GLEANING as usize,
            entity_types_guidance: String::new(),

            // Same value as the core constant.
            default_top_k: easyrag_core::constants::DEFAULT_TOP_K,
            default_mode: "mix".to_string(),

            llm_max_async: easyrag_core::constants::DEFAULT_MAX_ASYNC,
            embedding_max_async: easyrag_core::constants::DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
            pipeline_max_parallel: 2,

            pg_host: "localhost".to_string(),
            pg_port: 5432,
            pg_user: "postgres".to_string(),
            pg_password: "postgres".to_string(),
            pg_database: "easyrag".to_string(),
            pg_ssl_mode: None,
            pg_ssl_cert: None,
            pg_ssl_key: None,
            pg_ssl_root_cert: None,
            pg_ssl_crl: None,

            rerank_enabled: false,
            rerank_provider: "jina".to_string(),
            rerank_api_key: String::new(),
            rerank_model: "jina-reranker-v2-base-multilingual".to_string(),

            graph_storage_type: String::new(),
            neo4j_uri: "bolt://localhost:7687".to_string(),
            neo4j_user: "neo4j".to_string(),
            neo4j_password: String::new(),
            neo4j_database: "neo4j".to_string(),
            neo4j_max_pool_size: 100,

            cache_enabled: true,
            cache_max_size: 10_000,
            cache_ttl_hours: 24,

            keyword_search_enabled: true,
            keyword_search_language: "simple".to_string(),
            keyword_search_top_k: 20,
            hybrid_rrf_k: 60,
            lucene_index_dir: String::new(),

            language: "English".to_string(),
            verbose: false,
            log_level: "INFO".to_string(),

            // Background auto-scan defaults: disabled, no tasks, 60s interval.
            auto_scan_enabled: false,
            auto_scan_tasks: Vec::new(),
            auto_scan_interval_seconds: 60,
            auto_scan_input_dir: String::new(),
            auto_scan_extensions: ".txt,.md,.pdf".to_string(),
            auto_scan_max_files_per_cycle: 100,

            // Streaming pipeline defaults: disabled, 2s poll, 256 channel slots.
            pipeline_streaming_enabled: false,
            pipeline_poll_interval_ms: 2000,
            pipeline_channel_capacity: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    /// Contract test: defaults must match the expected values.
    #[test]
    fn defaults_match_blueprint() {
        let c = ServerConfig::default();
        assert_eq!(c.host, "0.0.0.0");
        assert_eq!(c.port, 8020);
        assert_eq!(c.workers, 1);
        assert!(c.auth_enabled);
        assert_eq!(c.jwt_secret, DEFAULT_JWT_SECRET);
        assert_eq!(c.jwt_expiration_hours, 24);
        assert_eq!(c.login_rate_limit_per_minute, 5);
        assert_eq!(c.cors_origins, vec!["*".to_string()]);
        assert_eq!(c.working_dir, "./rag_storage");
        assert_eq!(c.workspace, "default");
        assert_eq!(c.biz_id, "default");
        assert_eq!(c.storage_type, "json");
        assert_eq!(c.chunk_size, 1200);
        assert_eq!(c.chunk_overlap_size, 100);
        assert_eq!(c.chunking_method, "token_size");
        assert_eq!(
            c.entity_extract_max_gleaning,
            easyrag_core::constants::DEFAULT_MAX_GLEANING as usize
        );
        assert_eq!(c.entity_types_guidance, "");
        assert_eq!(c.default_top_k, 40);
        assert_eq!(c.default_mode, "mix");
        assert_eq!(c.llm_max_async, 4);
        assert_eq!(c.embedding_max_async, 8);
        assert_eq!(c.pipeline_max_parallel, 2);
        assert_eq!(c.rerank_provider, "jina");
        assert_eq!(c.neo4j_uri, "bolt://localhost:7687");
        assert_eq!(c.neo4j_max_pool_size, 100);
        assert!(c.cache_enabled);
        assert_eq!(c.cache_max_size, 10_000);
        assert_eq!(c.cache_ttl_hours, 24);
        assert!(c.keyword_search_enabled);
        assert_eq!(c.keyword_search_language, "simple");
        assert_eq!(c.keyword_search_top_k, 20);
        assert_eq!(c.hybrid_rrf_k, 60);
        assert_eq!(c.lucene_index_dir, "");
        assert_eq!(c.language, "English");
        assert_eq!(c.log_level, "INFO");
        // Streaming pipeline defaults (BC-11 gradual rollout: disabled by default).
        assert!(!c.pipeline_streaming_enabled);
        assert_eq!(c.pipeline_poll_interval_ms, 2000);
        assert_eq!(c.pipeline_channel_capacity, 256);
    }

    fn empty_tenant(workspace: &str) -> TenantConfig {
        let now = Utc::now();
        TenantConfig {
            workspace: workspace.to_string(),
            llm_model: None,
            llm_api_key: None,
            llm_base_url: None,
            llm_temperature: None,
            llm_max_tokens: None,
            embedding_model: None,
            embedding_api_key: None,
            embedding_base_url: None,
            embedding_dim: None,
            chunk_size: None,
            chunk_overlap_size: None,
            language: None,
            default_top_k: None,
            rerank_enabled: None,
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn merge_all_none_falls_back_to_global_defaults() {
        let global = ServerConfig::default();
        let tenant = empty_tenant("ws1");
        let resolved = global.merge_tenant_config(&tenant);
        assert_eq!(resolved.llm_model, global.llm_model);
        assert_eq!(resolved.llm_api_key, global.llm_api_key);
        assert_eq!(resolved.embedding_model, global.embedding_model);
        assert_eq!(resolved.embedding_dim, global.embedding_dim);
        assert_eq!(resolved.chunk_size, global.chunk_size);
        assert_eq!(resolved.language, global.language);
        assert_eq!(resolved.default_top_k, global.default_top_k);
        assert_eq!(resolved.rerank_enabled, global.rerank_enabled);
    }

    #[test]
    fn merge_partial_override_only_overrides_set_fields() {
        let global = ServerConfig::default();
        let mut tenant = empty_tenant("ws1");
        tenant.llm_model = Some("claude-3".to_string());
        tenant.chunk_size = Some(500);
        tenant.rerank_enabled = Some(true);

        let resolved = global.merge_tenant_config(&tenant);
        // Overridden
        assert_eq!(resolved.llm_model, "claude-3");
        assert_eq!(resolved.chunk_size, 500);
        assert!(resolved.rerank_enabled);
        // Unchanged
        assert_eq!(resolved.llm_api_key, global.llm_api_key);
        assert_eq!(resolved.embedding_model, global.embedding_model);
        assert_eq!(resolved.embedding_dim, global.embedding_dim);
        assert_eq!(resolved.chunk_overlap_size, global.chunk_overlap_size);
        assert_eq!(resolved.language, global.language);
    }

    #[test]
    fn merge_full_override_replaces_all_tenant_fields() {
        let global = ServerConfig::default();
        let now = Utc::now();
        let tenant = TenantConfig {
            workspace: "ws1".to_string(),
            llm_model: Some("gpt-4o-mini".to_string()),
            llm_api_key: Some("sk-tenant".to_string()),
            llm_base_url: Some("https://custom.api/v1".to_string()),
            llm_temperature: Some(0.7),
            llm_max_tokens: Some(2048),
            embedding_model: Some("text-embedding-3-large".to_string()),
            embedding_api_key: Some("sk-embed".to_string()),
            embedding_base_url: Some("https://embed.api/v1".to_string()),
            embedding_dim: Some(3072),
            chunk_size: Some(600),
            chunk_overlap_size: Some(50),
            language: Some("Chinese".to_string()),
            default_top_k: Some(10),
            rerank_enabled: Some(true),
            created_at: now,
            updated_at: now,
        };

        let resolved = global.merge_tenant_config(&tenant);
        assert_eq!(resolved.llm_model, "gpt-4o-mini");
        assert_eq!(resolved.llm_api_key, "sk-tenant");
        assert_eq!(resolved.llm_base_url, "https://custom.api/v1");
        assert_eq!(resolved.llm_temperature, 0.7);
        assert_eq!(resolved.llm_max_tokens, Some(2048));
        assert_eq!(resolved.embedding_model, "text-embedding-3-large");
        assert_eq!(resolved.embedding_api_key, "sk-embed");
        assert_eq!(resolved.embedding_base_url, "https://embed.api/v1");
        assert_eq!(resolved.embedding_dim, 3072);
        assert_eq!(resolved.chunk_size, 600);
        assert_eq!(resolved.chunk_overlap_size, 50);
        assert_eq!(resolved.language, "Chinese");
        assert_eq!(resolved.default_top_k, 10);
        assert!(resolved.rerank_enabled);
    }
}
