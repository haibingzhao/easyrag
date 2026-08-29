//! Configuration loading: workspace name validation, environment variable
//! helpers, and `ServerConfig` assembly.
//!
//! Variable names and defaults mirror `easyrag-app/src/main/resources/application.yml`.

use anyhow::bail;
use easyrag_server::{ServerConfig, DEFAULT_JWT_SECRET};

/// Workspace name guard (P2.17, direct):
/// prevents SQL injection through workspace-derived table names.
pub(crate) fn validate_workspace(workspace: &str) -> anyhow::Result<()> {
    let pattern = regex::Regex::new(r"^[a-zA-Z0-9_\-][a-zA-Z0-9_\-.]{0,63}$").unwrap();
    if !pattern.is_match(workspace) {
        bail!(
            "Invalid workspace name '{workspace}': must be 1-64 chars, \
             alphanumeric/underscore/hyphen/dot, starting with alphanumeric/underscore/hyphen"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Environment loading (application.yml variable names + defaults)
// ---------------------------------------------------------------------

fn env_get(get: &dyn Fn(&str) -> Option<String>, key: &str, default: &str) -> String {
    get(key).filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn env_parse<T: std::str::FromStr>(
    get: &dyn Fn(&str) -> Option<String>,
    key: &str,
    default: T,
) -> anyhow::Result<T> {
    match get(key).filter(|v| !v.trim().is_empty()) {
        // Set-but-invalid values fail fast (Spring Boot conversion semantics)
        // instead of silently falling back to the default.
        Some(v) => v
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid value for {key}: '{v}'")),
        None => Ok(default),
    }
}

fn env_bool(
    get: &dyn Fn(&str) -> Option<String>,
    key: &str,
    default: bool,
) -> anyhow::Result<bool> {
    match get(key).filter(|v| !v.trim().is_empty()) {
        Some(v) => match v.trim().to_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" => Ok(false),
            other => bail!("invalid boolean for {key}: '{other}'"),
        },
        None => Ok(default),
    }
}

/// Build the server configuration from environment variables. Defaults
/// mirror `easyrag-app/src/main/resources/application.yml`.
pub(crate) fn build_config(get: &dyn Fn(&str) -> Option<String>) -> anyhow::Result<ServerConfig> {
    let mut config = ServerConfig::default();
    config.port = env_parse(get, "PORT", 8020)?;

    // application.yml default (${STORAGE_BACKEND:memory}); the ServerConfig
    // data-class default "json" only applies without application.yml.
    config.storage_type = env_get(get, "STORAGE_BACKEND", "memory");
    config.working_dir = env_get(get, "WORKING_DIR", "./rag_storage");
    config.workspace = env_get(get, "WORKSPACE", "default");
    config.biz_id = env_get(get, "BIZ_ID", easyrag_core::constants::DEFAULT_BIZ_ID);

    config.pg_host = env_get(get, "PG_HOST", "localhost");
    config.pg_port = env_parse(get, "PG_PORT", 5432)?;
    config.pg_user = env_get(get, "PG_USER", "postgres");
    config.pg_password = env_get(get, "PG_PASSWORD", "postgres");
    config.pg_database = env_get(get, "PG_DATABASE", "easyrag");
    config.pg_ssl_mode = get("PG_SSL_MODE").filter(|v| !v.is_empty());
    config.pg_ssl_cert = get("PG_SSL_CERT").filter(|v| !v.is_empty());
    config.pg_ssl_key = get("PG_SSL_KEY").filter(|v| !v.is_empty());
    config.pg_ssl_root_cert = get("PG_SSL_ROOT_CERT").filter(|v| !v.is_empty());
    config.pg_ssl_crl = get("PG_SSL_CRL").filter(|v| !v.is_empty());

    config.llm_model = env_get(get, "LLM_MODEL", "gpt-4o-mini");
    config.llm_api_key = env_get(get, "LLM_BINDING_API_KEY", "");
    config.llm_base_url = env_get(get, "LLM_BINDING_HOST", "https://api.openai.com/v1");
    config.llm_temperature = env_parse(get, "LLM_TEMPERATURE", 0.0f32)?;
    config.llm_max_tokens = Some(env_parse(get, "LLM_MAX_TOKENS", 4096)?);

    config.embedding_model = env_get(get, "EMBEDDING_MODEL", "text-embedding-3-small");
    config.embedding_api_key = env_get(get, "EMBEDDING_BINDING_API_KEY", "");
    config.embedding_base_url = env_get(get, "EMBEDDING_BINDING_HOST", "https://api.openai.com/v1");
    config.embedding_dim = env_parse(get, "EMBEDDING_DIM", 1536)?;
    config.embedding_query_prefix = env_get(get, "EMBEDDING_QUERY_PREFIX", "");
    config.embedding_document_prefix = env_get(get, "EMBEDDING_DOCUMENT_PREFIX", "");

    config.chunk_size = env_parse(get, "CHUNK_SIZE", 1200)?;
    config.chunk_overlap_size = env_parse(get, "CHUNK_OVERLAP", 100)?;
    config.llm_max_async = env_parse(get, "MAX_CONCURRENCY", 4)?;
    config.language = env_get(get, "LANGUAGE", "English");
    config.default_top_k = env_parse(get, "TOP_K", 40)?;
    config.default_mode = env_get(get, "QUERY_MODE", "mix");

    config.graph_storage_type = env_get(get, "GRAPH_STORAGE_TYPE", "");
    config.neo4j_uri = env_get(get, "NEO4J_URI", "bolt://localhost:7687");
    config.neo4j_user = env_get(get, "NEO4J_USER", "neo4j");
    config.neo4j_password = env_get(get, "NEO4J_PASSWORD", "");
    config.neo4j_database = env_get(get, "NEO4J_DATABASE", "neo4j");
    config.neo4j_max_pool_size = env_parse(get, "NEO4J_MAX_POOL_SIZE", 100)?;

    config.keyword_search_enabled = env_bool(get, "KEYWORD_SEARCH_ENABLED", true)?;
    config.keyword_search_language = env_get(get, "KEYWORD_SEARCH_LANGUAGE", "simple");
    config.keyword_search_top_k = env_parse(get, "KEYWORD_SEARCH_TOP_K", 20)?;
    config.hybrid_rrf_k = env_parse(get, "HYBRID_RRF_K", 60)?;
    config.lucene_index_dir = env_get(get, "LUCENE_INDEX_DIR", "");

    config.auth_enabled = env_bool(get, "AUTH_ENABLED", true)?;
    config.jwt_secret = env_get(get, "JWT_SECRET", DEFAULT_JWT_SECRET);
    config.jwt_expiration_hours = env_parse(get, "JWT_EXPIRATION_HOURS", 24)?;
    config.login_rate_limit_per_minute = env_parse(get, "LOGIN_RATE_LIMIT_PER_MINUTE", 5)?;

    config.rerank_enabled = env_bool(get, "RERANK_ENABLED", false)?;
    config.rerank_provider = env_get(get, "RERANK_PROVIDER", "jina");
    config.rerank_api_key = env_get(get, "RERANK_API_KEY", "");
    config.rerank_model = env_get(
        get,
        "RERANK_MODEL",
        "jina-reranker-v2-base-multilingual",
    );

    config.cache_enabled = env_bool(get, "CACHE_ENABLED", true)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn getter(vars: HashMap<&'static str, String>) -> Box<dyn Fn(&str) -> Option<String>> {
        Box::new(move |key: &str| vars.get(key).cloned())
    }

    #[test]
    fn workspace_validation_pattern() {
        assert!(validate_workspace("default").is_ok());
        assert!(validate_workspace("ws-1.x_2").is_ok());
        assert!(validate_workspace("").is_err());
        assert!(validate_workspace(".leading-dot").is_err());
        assert!(validate_workspace("has space").is_err());
        assert!(validate_workspace("semi;colon").is_err());
        assert!(validate_workspace(&"a".repeat(65)).is_err());
    }

    #[test]
    fn config_defaults_match_application_yml() {
        let empty: HashMap<&str, String> = HashMap::new();
        let config = build_config(&getter(empty)).unwrap();
        assert_eq!(config.port, 8020);
        assert_eq!(config.storage_type, "memory");
        assert_eq!(config.working_dir, "./rag_storage");
        assert_eq!(config.workspace, "default");
        assert_eq!(config.biz_id, "default");
        assert_eq!(config.llm_model, "gpt-4o-mini");
        assert_eq!(config.llm_base_url, "https://api.openai.com/v1");
        assert_eq!(config.embedding_model, "text-embedding-3-small");
        assert_eq!(config.embedding_dim, 1536);
        assert_eq!(config.chunk_size, 1200);
        assert_eq!(config.chunk_overlap_size, 100);
        assert_eq!(config.llm_max_async, 4);
        assert_eq!(config.language, "English");
        assert_eq!(config.default_top_k, 40);
        assert_eq!(config.default_mode, "mix");
        assert!(config.keyword_search_enabled);
        assert_eq!(config.keyword_search_language, "simple");
        assert_eq!(config.hybrid_rrf_k, 60);
        assert_eq!(config.neo4j_max_pool_size, 100);
    }

    #[test]
    fn config_reads_env_overrides() {
        let mut vars: HashMap<&str, String> = HashMap::new();
        vars.insert("STORAGE_BACKEND", "postgres".into());
        vars.insert("LLM_MODEL", "qwen3.6-flash".into());
        vars.insert("EMBEDDING_DIM", "1024".into());
        vars.insert("KEYWORD_SEARCH_ENABLED", "false".into());
        vars.insert("PG_PORT", "6432".into());
        let config = build_config(&getter(vars)).unwrap();
        assert_eq!(config.storage_type, "postgres");
        assert_eq!(config.llm_model, "qwen3.6-flash");
        assert_eq!(config.embedding_dim, 1024);
        assert!(!config.keyword_search_enabled);
        assert_eq!(config.pg_port, 6432);
    }

    #[test]
    fn config_rejects_invalid_values() {
        // Set-but-invalid values must fail fast instead of silently
        // falling back to defaults (Spring Boot conversion semantics).
        let mut vars: HashMap<&str, String> = HashMap::new();
        vars.insert("PORT", "70000".into());
        assert!(build_config(&getter(vars)).is_err());

        let mut vars: HashMap<&str, String> = HashMap::new();
        vars.insert("KEYWORD_SEARCH_ENABLED", "enabled".into());
        assert!(build_config(&getter(vars)).is_err());

        // Spring-style on/off booleans are accepted.
        let mut vars: HashMap<&str, String> = HashMap::new();
        vars.insert("KEYWORD_SEARCH_ENABLED", "off".into());
        assert!(!build_config(&getter(vars)).unwrap().keyword_search_enabled);

        // Empty values fall back to defaults (commented-out .env lines).
        let mut vars: HashMap<&str, String> = HashMap::new();
        vars.insert("PORT", "".into());
        assert_eq!(build_config(&getter(vars)).unwrap().port, 8020);
    }
}
