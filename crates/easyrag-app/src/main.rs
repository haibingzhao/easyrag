//! EasyRAG application entrypoint.
//!
//! Loads configuration from `.env` / environment variables with the same
//! variable names and defaults as `application.yml`, assembles the storage
//! backend matrix, and serves the axum
//! router plus the webui static bundle on port 8020.
//!
//! Assembly code lives in the `easyrag_app` library modules (shared by this
//! binary and the integration tests, so assembly drift is compiler-checked):
//! - `config` : env loading + `ServerConfig` build
//! - `embedding` : OpenAI-compatible embedding adapter
//! - `storage` : backend selection matrix (`assemble_storage`)
//! - `workspace` : R2 per-workspace context resolver

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};

use easyrag_app::{config, embedding, factory, scanner, storage, workspace};
use easyrag_core::llm::service::{LlmService, ToolCallingLlmService};
use easyrag_core::storage::tenant_config::TenantConfigStorage;
use easyrag_core::storage::traits::StorageBundle;
use easyrag_core::storage::user::UserStorage;
use easyrag_llm::openai::OpenAiClient;
use easyrag_llm::rerank::RerankClient;
use easyrag_pipeline::multimodal::{RagAnythingConfig, RAGAnything, TextQueryBackend};
use easyrag_pipeline::{EasyRAGConfigBuilder, EasyRAG, TextParserRegistry};
use easyrag_server::auth::{JwtAuthProvider, LoginRateLimiter};
use easyrag_server::service::{DocumentLifecycleService, WorkspaceContext, WorkspaceResolver};
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
    // Field-by-field assembly lives in `app::llm` so this path and the embedding
    // path read the same mapping (Phase 4.4).
    let llm_config = easyrag_app::llm::chat_config(&config);
    // One client, two trait views: plain completion + tool calling.
    let openai_client = Arc::new(OpenAiClient::new(llm_config)?);
    let llm: Arc<dyn LlmService> = openai_client.clone();
    let tool_calling_llm: Arc<dyn ToolCallingLlmService> = openai_client;

    // Stateless embedding bean: construct a fresh identical instance for
    // the facade (storage layers consumed their own in assemble_storage).
    let embedding_func = Arc::new(embedding::make_embedding_func(&config)?);

    // --- Optional reranker ----------------------------------------------
    let rerank: Option<Arc<dyn easyrag_core::llm::service::RerankService>> =
        match easyrag_app::llm::rerank_config(&config) {
            // `RerankClient::new` still rejects a `Custom` provider with no
            // base URL, so a bad configuration fails startup as before.
            Some(rerank_config) => Some(Arc::new(RerankClient::new(rerank_config)?)),
            None => None,
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

    // --- Structure strategy + query engine (Phase 6.5 factory) -------------
    // Both are assembled through `app::factory` so this default path and the
    // per-workspace path in `WorkspaceResolverImpl::build_context` share one
    // wiring matrix. The reranker is threaded here (default path only); the
    // per-workspace path passes `None` — that asymmetry is BC-10, tracked
    // separately, and is now an explicit argument instead of a silent omission.
    let structure_strategy = Some(factory::build_structure_strategy(
        llm.clone(),
        Some(tool_calling_llm.clone()),
        storage.kv.clone(),
        storage.vector.clone(),
        storage.doc_status.clone(),
    ));
    let query_engine = factory::build_query_engine(
        llm.clone(),
        &config,
        storage.vector.clone(),
        storage.graph.clone(),
        storage.kv.clone(),
        storage.keyword.clone(),
        rerank.clone(),
        structure_strategy.clone(),
    );

    // --- Ingestion pipeline (Phase 6.5 factory) ---------------------------
    let pipeline = factory::build_pipeline(
        llm.clone(),
        &config,
        storage.kv.clone(),
        storage.vector.clone(),
        storage.graph.clone(),
        storage.doc_status.clone(),
        embedding_func.clone(),
        storage.keyword.clone(),
    );

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

    let workspace_resolver = Arc::new(workspace::WorkspaceResolverImpl::new(
        config.workspace.clone(),
        config.biz_id.clone(),
        default_context,
        llm.clone(),
        Some(tool_calling_llm.clone()),
        config.clone(),
        tenant_config_storage.clone(),
        storage.pg_db.clone(),
        storage.neo4j_db.clone(),
    ));

    // Auto-scan stats are built here because `AppServices` below consumes the
    // handle. The scanner task itself is spawned only AFTER the HTTP listener
    // binds (see the spawn site after `serve_task`), so a fallible startup
    // step — tenant/user storage init, the admin seed, or the port bind — can
    // never leave an immediate first scan cycle racing a failed boot or
    // writing to storage that is about to be finalized on the way out.
    let scan_stats = Arc::new(easyrag_server::AutoScanStats::new());

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
        parser_registry: Arc::new(TextParserRegistry::default()),
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
        auto_scan_stats: Arc::clone(&scan_stats),
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

    // --- Background auto scanner (A-9 wiring, BC-11) ------------------------
    // Default-disabled gradual rollout: when AUTO_SCAN_ENABLED=false (the
    // default) nothing is spawned and the binary behaves exactly as before.
    // When enabled, tasks are derived from config (`build_scan_tasks`) and run
    // on a 1s ticker with MissedTickBehavior::Delay; dir-scan reuses the shared
    // content_hash dedup (`scan_directory`) and both task kinds coordinate with
    // foreground work through the pipeline's re-entrancy guard + busy check.
    // Spawned only now — after every fallible startup step (tenant/user init,
    // admin seed, port bind) has succeeded — so the first immediate cycle can
    // never race a bind failure. The `JoinHandle` is kept so shutdown waits
    // for an in-flight cycle before finalizing storage.
    let scanner_task = if config.auto_scan_enabled {
        let scan_tasks = scanner::build_scan_tasks(&config);
        let task_count = scan_tasks.len();
        let (stop_tx, stop_rx) = tokio::sync::oneshot::channel::<()>();
        let auto_scanner = scanner::AutoScanner::new(
            scan_tasks,
            workspace_resolver.clone() as Arc<dyn WorkspaceResolver>,
            storage.kv.clone(),
            Arc::clone(&scan_stats),
            stop_rx,
        );
        let handle = tokio::spawn(auto_scanner.run());
        tracing::info!("Auto scanner spawned with {task_count} task(s)");
        Some((stop_tx, handle))
    } else {
        tracing::debug!("Auto scanner disabled (AUTO_SCAN_ENABLED not set/false)");
        None
    };

    shutdown_signal().await;
    tracing::info!("Shutdown signal received; draining connections...");
    // Stop the background scanner and WAIT for the in-flight cycle to finish
    // before draining HTTP and finalizing storage. `run()` only observes
    // `stop_rx` at the top of its loop, so a cycle already executing would
    // otherwise keep writing to the tantivy writer / pg pool after
    // `finalize_all()` and `pg_db.close()` below.
    if let Some((stop_tx, handle)) = scanner_task {
        let _ = stop_tx.send(());
        tracing::info!("Auto scanner stop signal sent");
        match tokio::time::timeout(Duration::from_secs(5), handle).await {
            Ok(_) => tracing::info!("Auto scanner stopped"),
            Err(_) => tracing::warn!(
                "Auto scanner did not stop within 5s; proceeding with shutdown"
            ),
        }
    }
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
