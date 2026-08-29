//! EasyRAG application entrypoint.
//!
//! Loads configuration from `.env` / environment variables with the same
//! variable names and defaults as `application.yml`, assembles the storage
//! backend matrix, and serves the axum
//! router plus the webui static bundle on port 8020.
//!
//! Assembly code lives in sibling modules:
//! - `config` : env loading + `ServerConfig` build
//! - `embedding` : OpenAI-compatible embedding adapter
//! - `storage` : backend selection matrix (`assemble_storage`)
//! - `workspace` : R2 per-workspace context resolver

mod config;
mod embedding;
mod storage;
mod workspace;

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};

use easyrag_core::llm::service::{LlmService, ToolCallingLlmService};
use easyrag_core::storage::tenant_config::TenantConfigStorage;
use easyrag_core::storage::traits::StorageBundle;
use easyrag_core::storage::user::UserStorage;
use easyrag_llm::openai::{OpenAiClient, OpenAiConfig};
use easyrag_llm::rerank::{RerankClient, RerankConfig, RerankProvider};
use easyrag_pipeline::multimodal::{RagAnythingConfig, RAGAnything, TextQueryBackend};
use easyrag_pipeline::{
    EasyRAGConfigBuilder, EasyRAG, IngestionPipeline, ParserRegistry, QueryEngine,
    StructureAgentStrategy, StructureQueryStrategy, TreeSearchEngine,
};
use easyrag_server::auth::{JwtAuthProvider, LoginRateLimiter};
use easyrag_server::service::{DocumentLifecycleService, WorkspaceContext};
use easyrag_server::state::AppServices;
use easyrag_server::{build_router, AppError, DEFAULT_JWT_SECRET};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let dotenv_result = dotenvy::dotenv();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    match dotenv_result {
        Ok(path) => tracing::debug!("loaded environment from {}", path.display()),
        Err(dotenvy::Error::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => tracing::warn!(".env present but could not be loaded: {e}"),
    }

    let get = |key: &str| std::env::var(key).ok();
    let config = config::build_config(&get)?;
    config::validate_workspace(&config.workspace)?;
    // biz_id shares the workspace charset (it lands in SQL/Cypher/paths).
    config::validate_workspace(&config.biz_id)?;

    if config.auth_enabled {
        if config.jwt_secret == DEFAULT_JWT_SECRET {
            bail!("Auth is enabled but JWT_SECRET is still the default; set a secret of at least 32 bytes");
        }
        if config.jwt_secret.as_bytes().len() < 32 {
            bail!("Auth is enabled but JWT_SECRET is shorter than 32 bytes; refusing to start with a weak signing key");
        }
    }

    let storage = storage::assemble_storage(&config).await?;

    // --- LLM client -----------------------
    if config.llm_api_key.is_empty() {
        tracing::warn!("LLM_BINDING_API_KEY is not set; LLM calls will fail until configured");
    }
    let llm_api_key = if config.llm_api_key.is_empty() {
        "sk-placeholder".to_string()
    } else {
        config.llm_api_key.clone()
    };
    let mut llm_config = OpenAiConfig::new(llm_api_key);
    llm_config.base_url = config.llm_base_url.clone();
    llm_config.model = config.llm_model.clone();
    // Slow reasoning models (e.g. qwen3.8-max) can exceed the 120s default.
    llm_config.timeout = Duration::from_secs(config.llm_timeout_secs);
    // LLM_ENABLE_THINKING=false is what keeps reasoning models fast enough.
    llm_config.enable_thinking = config.llm_enable_thinking;
    // One client, two trait views: plain completion + tool calling.
    let openai_client = Arc::new(OpenAiClient::new(llm_config)?);
    let llm: Arc<dyn LlmService> = openai_client.clone();
    let tool_calling_llm: Arc<dyn ToolCallingLlmService> = openai_client;

    // Stateless embedding bean: construct a fresh identical instance for
    // the facade (storage layers consumed their own in assemble_storage).
    let embedding_func = Arc::new(embedding::make_embedding_func(&config)?);

    // --- Optional reranker ----------------------------------------------
    let rerank: Option<Arc<dyn easyrag_core::llm::service::RerankService>> =
        if config.rerank_enabled {
            let provider = match config.rerank_provider.to_lowercase().as_str() {
                "cohere" => RerankProvider::Cohere,
                "custom" => RerankProvider::Custom,
                _ => RerankProvider::Jina,
            };
            let mut rerank_config = RerankConfig::new(provider);
            rerank_config.api_key = config.rerank_api_key.clone();
            rerank_config.model = config.rerank_model.clone();
            Some(Arc::new(RerankClient::new(rerank_config)?))
        } else {
            None
        };

    // --- EasyRAG facade --------------------------------------------------
    let rag_config = EasyRAGConfigBuilder::new()
        .working_dir(config.working_dir.clone())
        .workspace(config.workspace.clone())
        .llm_model(config.llm_model.clone())
        .embedding_dim(config.embedding_dim)
        .chunk_size(config.chunk_size)
        .chunk_overlap_size(config.chunk_overlap_size)
        .entity_extract_max_gleaning(config.entity_extract_max_gleaning)
        .entity_types_guidance(config.entity_types_guidance.clone())
        .storage_type(config.storage_type.clone())
        .language(config.language.clone())
        .build();
    let easy_rag = Arc::new(EasyRAG::new(
        rag_config,
        llm.clone(),
        embedding_func.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
        storage.kv.clone(),
        storage.doc_status.clone(),
        rerank.clone(),
        storage.keyword.clone(),
    ));
    easy_rag.initialize().await.context("EasyRAG initialize")?;

    // --- Structure strategy -------
    let agent_strategy = Some(Arc::new(
        StructureAgentStrategy::new(
            tool_calling_llm.clone(),
            storage.kv.clone(),
            storage.vector.clone(),
        )
        .with_doc_status_storage(storage.doc_status.clone()),
    ));
    let structure_strategy = Some(Arc::new(
        StructureQueryStrategy::new(
            TreeSearchEngine::new(llm.clone()),
            agent_strategy,
            storage.kv.clone(),
            storage.vector.clone(),
            llm.clone(),
        )
        .with_doc_status_storage(storage.doc_status.clone()),
    ));

    // --- Query engine -----------------------------------------------------
    let mut query_engine = QueryEngine::new(
        llm.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
        storage.kv.clone(),
    )
    .with_language(config.language.clone())
    .with_rrf_k(config.hybrid_rrf_k);
    if let Some(keyword) = &storage.keyword {
        query_engine = query_engine.with_keyword_search(keyword.clone());
    }
    if let Some(rerank) = &rerank {
        query_engine = query_engine.with_rerank(rerank.clone());
    }
    if let Some(strategy) = &structure_strategy {
        query_engine = query_engine.with_structure_strategy(strategy.clone());
    }
    let query_engine = Arc::new(query_engine);

    // --- Ingestion pipeline ------------------------------------------------
    let mut extractor_config =
        easyrag_pipeline::extraction::EntityExtractorConfig::default();
    extractor_config.apply_ingest_settings(
        config.language.clone(),
        config.entity_extract_max_gleaning,
        &config.entity_types_guidance,
    );
    let mut pipeline = IngestionPipeline::new(
        llm.clone(),
        storage.kv.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
        storage.doc_status.clone(),
        embedding_func.clone(),
    )
    .with_extractor(easyrag_pipeline::extraction::EntityExtractor::new(
        llm.clone(),
        storage.graph.clone(),
        storage.vector.clone(),
        storage.kv.clone(),
        extractor_config,
    ))
    .with_language(config.language.clone())
    .with_max_concurrency(config.llm_max_async);
    if let Some(keyword) = &storage.keyword {
        pipeline = pipeline.with_keyword_search(keyword.clone());
    }
    let pipeline = Arc::new(pipeline);

    // --- Document lifecycle service + R2 workspace resolver ---------------
    let lifecycle_service = Arc::new(DocumentLifecycleService::new(
        pipeline.clone(),
        storage.doc_status.clone(),
        storage.kv.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
    ));
    let default_context = WorkspaceContext {
        bundle: StorageBundle {
            workspace: config.workspace.clone(),
            kv_storage: storage.kv.clone(),
            vector_storage: storage.vector.clone(),
            graph_storage: storage.graph.clone(),
            doc_status_storage: storage.doc_status.clone(),
        },
        pipeline: pipeline.clone(),
        query_engine: query_engine.clone(),
        structure_strategy: structure_strategy.clone(),
    };
    // --- Tenant config storage (per-workspace config overrides) -----------
    let tenant_config_storage: Arc<dyn TenantConfigStorage> = if let Some(db) = &storage.pg_db {
        let tc = easyrag_storage::postgres::PgTenantConfigStorage::new(db.clone());
        tc.initialize().await.context("tenant config storage initialize")?;
        Arc::new(tc)
    } else {
        Arc::new(easyrag_storage::memory::InMemoryTenantConfigStorage::new())
    };

    let workspace_resolver = Arc::new(workspace::WorkspaceResolverImpl {
        default_workspace: config.workspace.clone(),
        default_biz_id: config.biz_id.clone(),
        default_context,
        llm: llm.clone(),
        tool_calling_llm: Some(tool_calling_llm.clone()),
        config: config.clone(),
        tenant_config_storage: tenant_config_storage.clone(),
        pg_db: storage.pg_db.clone(),
        neo4j_db: storage.neo4j_db.clone(),
        cache: tokio::sync::Mutex::new(Vec::new()),
    });

    // --- Multimodal (RAGAnything) -------------------------------------------
    let rag_anything_config = RagAnythingConfig {
        working_dir: config.working_dir.clone(),
        ..Default::default()
    };
    let rag_anything = Arc::new(RAGAnything::new(
        rag_anything_config,
        Arc::clone(&easy_rag) as Arc<dyn TextQueryBackend>,
        llm.clone(),
        storage.kv.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
        None,
        config.biz_id.clone(),
    ));

    // --- Auth ---------------------------------------------------------------
    // The provider is always constructed so the login endpoint can issue
    // tokens even when auth enforcement is disabled (: JwtAuthProvider
    // bean is unconditional; only the filter chain is conditional).
    let jwt_provider = Arc::new(JwtAuthProvider::new(
        &config.jwt_secret,
        config.jwt_expiration_hours,
    ));
    if config.auth_enabled {
        // Signing-key fingerprint (first 4 bytes only): restarts or
        // multi-instance deployments must yield the same fingerprint,
        // otherwise every token issued before the swap is rejected with
        // 401 (JWT is stateless; the key is the whole validation state).
        tracing::info!(
            "JWT signing key fingerprint: {} (expiration {}h)",
            jwt_provider.key_fingerprint(),
            config.jwt_expiration_hours
        );
    }
    let jwt = config.auth_enabled.then(|| Arc::clone(&jwt_provider));

    // --- User storage + AuthService ----
    let user_storage: Arc<dyn UserStorage> = if let Some(db) = &storage.pg_db {
        let users = easyrag_storage::postgres::PgUserStorage::new(db.clone());
        users.initialize().await.context("user storage initialize")?;
        Arc::new(users)
    } else {
        Arc::new(easyrag_storage::memory::InMemoryUserStorage::new())
    };
    let auth_service = Arc::new(easyrag_server::service::AuthService::new(
        user_storage,
        Arc::clone(&jwt_provider),
    ));
    if let Err(e) = auth_service.ensure_default_admin().await {
        bail!("seed default admin user: {e:?}");
    }

    let services = Arc::new(AppServices {
        config: config.clone(),
        easy_rag,
        query_engine,
        structure_strategy,
        pipeline,
        vector_storage: storage.vector.clone(),
        graph_storage: storage.graph.clone(),
        kv_storage: storage.kv.clone(),
        doc_status_storage: storage.doc_status.clone(),
        parser_registry: Arc::new(ParserRegistry::default()),
        lifecycle_service,
        workspace_resolver: workspace_resolver.clone(),
        rag_anything: Some(rag_anything),
        jwt,
        rate_limiter: Arc::new(LoginRateLimiter::new(
            config.login_rate_limit_per_minute,
            10_000,
        )),
        auth_service,
        tenant_config_storage,
    });

    let mut app = build_router(services);

    // --- webui static hosting (SPA fallback to index.html) -------------------
    let webui_dir = std::env::var("WEBUI_DIR").unwrap_or_else(|_| "easyrag-webui/dist".into());
    if Path::new(&webui_dir).join("index.html").exists() {
        app = app.fallback(
            axum::routing::any(spa_fallback).with_state(std::path::PathBuf::from(webui_dir.clone())),
        );
        tracing::info!("Serving webui from {webui_dir}");
    } else {
        tracing::warn!("webui bundle not found at {webui_dir}; API only");
    }

    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .context("invalid host/port")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("EasyRAG listening on http://{addr}");

    let pg_db = storage.pg_db.clone();
    let neo4j_db = storage.neo4j_db.clone();

    // Graceful shutdown with a drain deadline: long-lived SSE streams must
    // not keep the process (and the pg pool close) alive indefinitely.
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let serve_task = tokio::spawn(async move {
        // ConnectInfo powers the `{ip}:{username}` login rate-limit key.
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
    });

    shutdown_signal().await;
    tracing::info!("Shutdown signal received; draining connections...");
    let _ = shutdown_tx.send(());
    match tokio::time::timeout(Duration::from_secs(10), serve_task).await {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(err))) => tracing::warn!("HTTP server error during shutdown: {err}"),
        Ok(Err(err)) if err.is_cancelled() => {}
        Ok(Err(err)) => tracing::warn!("serve task join error: {err}"),
        Err(_) => tracing::warn!(
            "Shutdown drain timeout exceeded; forcing exit (open SSE streams dropped)"
        ),
    }

    // close pools and release resources.
    tracing::info!("EasyRAG shutting down gracefully...");
    workspace_resolver.finalize_all().await;
    if let Some(db) = &pg_db {
        db.close().await;
        tracing::info!("PostgreSQL connection pool closed");
    }
    // neo4rs::Graph has no explicit close API; the pool is released on drop
    //. Same for the tantivy writer, which
    // commits on every write so dropping it loses no data.
    drop(neo4j_db);
    tracing::info!("EasyRAG shutdown complete");
    Ok(())
}

/// Static file handler with SPA fallback: serves the requested file from
/// the webui bundle when it exists, otherwise `index.html` (so client-side
/// routes like `/documents` deep-link correctly).
async fn spa_fallback(
    axum::extract::State(webui): axum::extract::State<std::path::PathBuf>,
    uri: axum::http::Uri,
) -> axum::response::Response {
    use axum::http::header;
    use axum::response::IntoResponse;

    // Unknown API paths keep the JSON 404 contract: the fallback is mounted
    // after the CORS/auth layers, so it must not mask API route misses.
    if uri.path() == "/api" || uri.path().starts_with("/api/") {
        return AppError::status(404, "No API route").into_response();
    }

    let decoded = percent_encoding::percent_decode_str(uri.path()).decode_utf8_lossy();
    let rel = decoded.trim_start_matches('/');
    // Path traversal guard: only serve files resolving inside the bundle.
    let file = if !rel.is_empty() {
        match (webui.join(rel).canonicalize(), webui.canonicalize()) {
            (Ok(f), Ok(root)) if f.starts_with(&root) && f.is_file() => f,
            _ => webui.join("index.html"),
        }
    } else {
        webui.join("index.html")
    };
    let content_type = mime_guess::from_path(&file)
        .first_or_octet_stream()
        .to_string();
    match tokio::fs::read(&file).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, content_type)], bytes).into_response(),
        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
