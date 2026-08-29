//! Configuration for the EasyRAG instance

/// Configuration for EasyRAG instance.
#[derive(Debug, Clone)]
pub struct EasyRAGConfig {
    // Working directory
    pub working_dir: String,
    pub workspace: String,

    // LLM settings
    pub llm_model: String,
    pub llm_temperature: f32,
    pub llm_max_tokens: Option<u32>,
    /// Seconds.
    pub llm_timeout_seconds: u64,

    // Embedding settings
    pub embedding_model: String,
    pub embedding_dim: usize,
    pub embedding_batch_size: usize,

    // Chunking settings
    pub chunking_method: String,
    pub chunk_size: usize,
    pub chunk_overlap_size: usize,

    // Extraction settings
    pub entity_extract_max_gleaning: usize,
    pub entity_types_guidance: String,
    pub language: String,

    // Query settings
    pub default_top_k: usize,
    pub default_chunk_top_k: usize,
    pub max_entity_tokens: usize,
    pub max_relation_tokens: usize,
    pub max_total_tokens: usize,

    // Concurrency settings
    pub llm_model_max_async: usize,
    pub embedding_max_async: usize,
    pub pipeline_max_parallel: usize,

    // Storage settings
    pub storage_type: String,
    pub vector_storage_type: String,
    pub graph_storage_type: String,

    // PostgreSQL settings (if storage_type = "postgres")
    pub pg_host: String,
    pub pg_port: u16,
    pub pg_user: String,
    pub pg_password: String,
    pub pg_database: String,

    // Cache settings
    pub enable_llm_cache: bool,
    pub cache_max_size: u64,
    pub cache_ttl_hours: u64,

    // Rerank settings
    pub enable_rerank: bool,
    pub rerank_provider: String,
    pub rerank_api_key: String,
    pub rerank_model: String,

    // Anti-hallucination guardrail settings
    pub enable_verification_gate: bool,
    pub verification_abstain_threshold: f32,
    pub enable_evidence_quality_scoring: bool,
    pub evidence_quality_threshold_good: f32,
    pub evidence_quality_threshold_bad: f32,
    pub enable_citation_validation: bool,
    pub enable_contextual_chunking: bool,
    pub enable_dedup: bool,
    pub dedup_similarity_threshold: f64,
    pub enable_query_routing: bool,

    // Misc
    pub verbose: bool,
    pub log_level: String,
}

impl Default for EasyRAGConfig {
    fn default() -> Self {
        Self {
            working_dir: "./rag_storage".to_string(),
            workspace: "default".to_string(),

            llm_model: "gpt-4o".to_string(),
            llm_temperature: 0.0,
            llm_max_tokens: None,
            llm_timeout_seconds: 120,

            embedding_model: "text-embedding-3-small".to_string(),
            embedding_dim: 1536,
            embedding_batch_size: 32,

            chunking_method: "token_size".to_string(),
            chunk_size: 1200,
            chunk_overlap_size: 100,

            entity_extract_max_gleaning: 0,
            entity_types_guidance: String::new(),
            language: "English".to_string(),

            default_top_k: 40,
            default_chunk_top_k: 20,
            max_entity_tokens: 6000,
            max_relation_tokens: 8000,
            max_total_tokens: 30000,

            llm_model_max_async: 4,
            embedding_max_async: 8,
            pipeline_max_parallel: 2,

            storage_type: "json".to_string(),
            vector_storage_type: "memory".to_string(),
            graph_storage_type: "memory".to_string(),

            pg_host: "localhost".to_string(),
            pg_port: 5432,
            pg_user: "postgres".to_string(),
            pg_password: "postgres".to_string(),
            pg_database: "easyrag".to_string(),

            enable_llm_cache: true,
            cache_max_size: 10_000,
            cache_ttl_hours: 24,

            enable_rerank: false,
            rerank_provider: "jina".to_string(),
            rerank_api_key: String::new(),
            rerank_model: "jina-reranker-v2-base-multilingual".to_string(),

            // Anti-hallucination guardrails (all opt-in, default off)
            enable_verification_gate: false,
            verification_abstain_threshold: 0.3,
            enable_evidence_quality_scoring: false,
            evidence_quality_threshold_good: 0.7,
            evidence_quality_threshold_bad: 0.4,
            enable_citation_validation: false,
            enable_contextual_chunking: false,
            enable_dedup: false,
            dedup_similarity_threshold: 0.8,
            enable_query_routing: false,

            verbose: false,
            log_level: "INFO".to_string(),
        }
    }
}

/// Builder for EasyRAGConfig with sensible defaults
///
#[derive(Clone, Debug, Default)]
pub struct EasyRAGConfigBuilder {
    config: EasyRAGConfig,
}

impl EasyRAGConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: EasyRAGConfig::default(),
        }
    }

    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.working_dir = dir.into();
        self
    }

    pub fn workspace(mut self, ws: impl Into<String>) -> Self {
        self.config.workspace = ws.into();
        self
    }

    pub fn llm_model(mut self, model: impl Into<String>) -> Self {
        self.config.llm_model = model.into();
        self
    }

    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.config.embedding_dim = dim;
        self
    }

    pub fn chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    pub fn chunk_overlap_size(mut self, size: usize) -> Self {
        self.config.chunk_overlap_size = size;
        self
    }

    pub fn storage_type(mut self, storage_type: impl Into<String>) -> Self {
        self.config.storage_type = storage_type.into();
        self
    }

    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.config.language = lang.into();
        self
    }

    pub fn entity_extract_max_gleaning(mut self, rounds: usize) -> Self {
        self.config.entity_extract_max_gleaning = rounds;
        self
    }

    pub fn entity_types_guidance(mut self, guidance: impl Into<String>) -> Self {
        self.config.entity_types_guidance = guidance.into();
        self
    }

    // Anti-hallucination guardrail builders

    pub fn enable_verification_gate(mut self, enable: bool) -> Self {
        self.config.enable_verification_gate = enable;
        self
    }

    pub fn verification_abstain_threshold(mut self, threshold: f32) -> Self {
        self.config.verification_abstain_threshold = threshold;
        self
    }

    pub fn enable_evidence_quality_scoring(mut self, enable: bool) -> Self {
        self.config.enable_evidence_quality_scoring = enable;
        self
    }

    pub fn evidence_quality_threshold_good(mut self, threshold: f32) -> Self {
        self.config.evidence_quality_threshold_good = threshold;
        self
    }

    pub fn evidence_quality_threshold_bad(mut self, threshold: f32) -> Self {
        self.config.evidence_quality_threshold_bad = threshold;
        self
    }

    pub fn enable_citation_validation(mut self, enable: bool) -> Self {
        self.config.enable_citation_validation = enable;
        self
    }

    pub fn enable_contextual_chunking(mut self, enable: bool) -> Self {
        self.config.enable_contextual_chunking = enable;
        self
    }

    pub fn enable_dedup(mut self, enable: bool) -> Self {
        self.config.enable_dedup = enable;
        self
    }

    pub fn dedup_similarity_threshold(mut self, threshold: f64) -> Self {
        self.config.dedup_similarity_threshold = threshold;
        self
    }

    pub fn enable_query_routing(mut self, enable: bool) -> Self {
        self.config.enable_query_routing = enable;
        self
    }

    pub fn build(self) -> EasyRAGConfig {
        self.config
    }
}
