//! R2: request-level workspace routing (
//! bean + `WorkspaceResolverImpl`): LRU-cached per-workspace contexts with
//! dedicated storage instances.

use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;

use easyrag_core::llm::service::{LlmService, ToolCallingLlmService};
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::tenant_config::TenantConfigStorage;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageBundle, StorageNameSpace, VectorStorage,
};
use easyrag_server::service::{ResolvedScope, WorkspaceContext, WorkspaceResolver};
use easyrag_server::{AppError, AppResult, ServerConfig};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};

use crate::config::validate_workspace;
use crate::embedding::make_embedding_func;
use crate::factory;

/// R2: upper bound of cached per-workspace contexts (LRU eviction).
const MAX_CACHED_WORKSPACES: usize = 32;

/// LRU-cached per-workspace contexts; eviction finalizes released
/// storages. Null/blank workspace resolves to the server-level singleton
/// context; other workspaces get a cached context with dedicated storage
/// instances (Pg storages share the connection pool - the workspace is a
/// constructor param plus SQL filter column, so instantiation is cheap
/// and DDL is idempotent). biz_id is never part of the cache key: it is
/// a per-call filter value, normalized in resolve and returned to the
/// caller alongside the context.
pub struct WorkspaceResolverImpl {
    default_workspace: String,
    default_biz_id: String,
    default_context: WorkspaceContext,
    llm: Arc<dyn LlmService>,
    tool_calling_llm: Option<Arc<dyn ToolCallingLlmService>>,
    config: ServerConfig,
    tenant_config_storage: Arc<dyn TenantConfigStorage>,
    pg_db: Option<easyrag_storage::postgres::PgDatabase>,
    neo4j_db: Option<easyrag_storage::neo4j::Neo4jDatabase>,
    cache: tokio::sync::Mutex<
        Vec<(
            String,
            WorkspaceContext,
            Option<Arc<easyrag_storage::tantivy_search::TantivyKeywordSearch>>,
        )>,
    >,
}

impl WorkspaceResolverImpl {
    /// Construct the resolver. Fields are private on purpose: the LRU `cache`
    /// and the shared `pg_db`/`neo4j_db` pools carry a lifecycle invariant
    /// (eviction must run `finalize_entry` — close the tantivy directory lock,
    /// then finalize each storage), so callers go through `resolve` /
    /// `finalize_all` / `invalidate_cached_workspace` instead of touching the
    /// raw state. This constructor exists because the binary is a separate
    /// crate from this library, where `pub(crate)` fields are invisible to
    /// `main`; keeping the fields private still blocks *external* crates.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        default_workspace: String,
        default_biz_id: String,
        default_context: WorkspaceContext,
        llm: Arc<dyn LlmService>,
        tool_calling_llm: Option<Arc<dyn ToolCallingLlmService>>,
        config: ServerConfig,
        tenant_config_storage: Arc<dyn TenantConfigStorage>,
        pg_db: Option<easyrag_storage::postgres::PgDatabase>,
        neo4j_db: Option<easyrag_storage::neo4j::Neo4jDatabase>,
    ) -> Self {
        Self {
            default_workspace,
            default_biz_id,
            default_context,
            llm,
            tool_calling_llm,
            config,
            tenant_config_storage,
            pg_db,
            neo4j_db,
            cache: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    async fn build_context(
        &self,
        workspace: &str,
    ) -> anyhow::Result<(
        WorkspaceContext,
        Option<Arc<easyrag_storage::tantivy_search::TantivyKeywordSearch>>,
    )> {
        // Load per-workspace tenant config and merge with global defaults.
        // If no tenant config exists, all fields fall back to self.config.
        let resolved_config = match self.tenant_config_storage.get_by_workspace(workspace).await {
            Ok(Some(tenant)) => {
                tracing::info!("Loaded tenant config for workspace: {workspace}");
                self.config.merge_tenant_config(&tenant)
            }
            Ok(None) => {
                tracing::debug!("No tenant config for workspace {workspace}, using global defaults");
                self.config.clone()
            }
            Err(e) => {
                tracing::warn!("Failed to load tenant config for {workspace}: {e}, using global defaults");
                self.config.clone()
            }
        };

        // Validated in resolve; construct unchecked.
        let (kv, vector, graph, doc_status): (
            Arc<dyn KVStorage>,
            Arc<dyn VectorStorage>,
            Arc<dyn GraphStorage>,
            Arc<dyn DocStatusStorage>,
        );
        // PgVectorStorage doubles as the keyword search backend.
        let mut pg_keyword: Option<Arc<dyn KeywordSearchable>> = None;

        if let Some(db) = &self.pg_db {
            let pg_config = easyrag_storage::postgres::PgConfig {
                host: resolved_config.pg_host.clone(),
                port: resolved_config.pg_port,
                user: resolved_config.pg_user.clone(),
                password: resolved_config.pg_password.clone(),
                database: resolved_config.pg_database.clone(),
                workspace: workspace.to_string(),
                embedding_dim: resolved_config.embedding_dim,
                ssl_mode: resolved_config.pg_ssl_mode.clone(),
                ssl_cert: resolved_config.pg_ssl_cert.clone(),
                ssl_key: resolved_config.pg_ssl_key.clone(),
                ssl_root_cert: resolved_config.pg_ssl_root_cert.clone(),
                ssl_crl: resolved_config.pg_ssl_crl.clone(),
                fts_config: resolved_config.keyword_search_language.clone(),
                ..Default::default()
            };
            let kv_storage = easyrag_storage::postgres::PgKVStorage::new(
                "kv",
                workspace.to_string(),
                db.clone(),
            )?;
            kv_storage.initialize().await?;
            let vector_storage = Arc::new(easyrag_storage::postgres::PgVectorStorage::new(
                "vectors",
                workspace.to_string(),
                make_embedding_func(&resolved_config)?,
                db.clone(),
                pg_config.clone(),
            )?);
            vector_storage.initialize().await?;
            pg_keyword = Some(Arc::clone(&vector_storage) as Arc<dyn KeywordSearchable>);
            let doc_status_storage = easyrag_storage::postgres::PgDocStatusStorage::new(
                "doc_status",
                workspace.to_string(),
                db.clone(),
            )?;
            doc_status_storage.initialize().await?;
            graph = if let Some(neo4j) = &self.neo4j_db {
                let storage = easyrag_storage::neo4j::Neo4jGraphStorage::new(
                    "graph",
                    workspace.to_string(),
                    neo4j.clone(),
                )?;
                storage.initialize().await?;
                Arc::new(storage)
            } else {
                let storage = easyrag_storage::postgres::PgGraphStorage::new(
                    "graph",
                    workspace.to_string(),
                    db.clone(),
                )?;
                storage.initialize().await?;
                Arc::new(storage)
            };
            kv = Arc::new(kv_storage);
            vector = vector_storage;
            doc_status = Arc::new(doc_status_storage);
        } else {
            kv = Arc::new(InMemoryKVStorage::new("kv", workspace.to_string()));
            vector = Arc::new(InMemoryVectorStorage::new(
                "vectors",
                workspace.to_string(),
                make_embedding_func(&resolved_config)?,
            ));
            graph = if let Some(neo4j) = &self.neo4j_db {
                let storage = easyrag_storage::neo4j::Neo4jGraphStorage::new(
                    "graph",
                    workspace.to_string(),
                    neo4j.clone(),
                )?;
                storage.initialize().await?;
                Arc::new(storage)
            } else {
                Arc::new(InMemoryGraphStorage::new(
                    "graph",
                    workspace.to_string(),
                ))
            };
            doc_status = Arc::new(InMemoryDocStatusStorage::new(
                "doc_status",
                workspace.to_string(),
            ));
        }

        // Keyword search: PG tsvector or a per-workspace embedded tantivy index.
        let mut tantivy_handle: Option<
            Arc<easyrag_storage::tantivy_search::TantivyKeywordSearch>,
        > = None;
        let keyword: Option<Arc<dyn KeywordSearchable>> = if !resolved_config.keyword_search_enabled {
            None
        } else if let Some(pg_kw) = pg_keyword {
            Some(pg_kw)
        } else {
            let index_dir = if resolved_config.lucene_index_dir.is_empty() {
                resolved_config.working_dir.clone()
            } else {
                resolved_config.lucene_index_dir.clone()
            };
            let search = Arc::new(
                easyrag_storage::tantivy_search::TantivyKeywordSearch::new(
                    workspace.to_string(),
                    &index_dir,
                    64,
                )
                .context("tantivy keyword index")?,
            );
            tantivy_handle = Some(Arc::clone(&search));
            Some(search as Arc<dyn KeywordSearchable>)
        };

        // Pipeline / query engine / structure strategy are assembled through
        // `app::factory` — the same wiring matrix the default path in `main`
        // uses, so the two paths cannot drift apart. The per-workspace path
        // passes `rerank = None` (a per-workspace reranker is BC-10 and is
        // intentionally not wired) and threads its own `keyword` handle; the
        // tenant-resolved `resolved_config` feeds the extractor settings.
        let pipeline = factory::build_pipeline(
            self.llm.clone(),
            &resolved_config,
            kv.clone(),
            vector.clone(),
            graph.clone(),
            doc_status.clone(),
            Arc::new(make_embedding_func(&resolved_config)?),
            keyword.clone(),
        );

        let structure = factory::build_structure_strategy(
            self.llm.clone(),
            &resolved_config,
            self.tool_calling_llm.clone(),
            kv.clone(),
            vector.clone(),
            doc_status.clone(),
        );

        let engine = factory::build_query_engine(
            self.llm.clone(),
            &resolved_config,
            vector.clone(),
            graph.clone(),
            kv.clone(),
            keyword.clone(),
            None, // rerank: per-workspace reranker is BC-10, intentionally not wired.
            Some(structure.clone()),
        );

        tracing::info!("Created workspace context: {workspace}");
        Ok((
            WorkspaceContext {
                bundle: StorageBundle {
                    workspace: workspace.to_string(),
                    kv_storage: kv,
                    vector_storage: vector,
                    graph_storage: graph,
                    doc_status_storage: doc_status,
                },
                pipeline,
                query_engine: engine,
                structure_strategy: Some(structure),
            },
            tantivy_handle,
        ))
    }

    /// Release the keyword index first (the tantivy writer holds a directory
    /// lock and the workspace may be recreated after eviction), then finalize
    /// each storage. Failures are logged, never propagated.
    async fn finalize_entry(
        &self,
        scope_key: &str,
        context: &WorkspaceContext,
        keyword: Option<Arc<easyrag_storage::tantivy_search::TantivyKeywordSearch>>,
    ) {
        if let Some(kw) = keyword {
            if let Err(e) = kw.close() {
                tracing::warn!("Failed to close keyword index for scope {scope_key}: {e}");
            }
        }
        let bundle = &context.bundle;
        if let Err(e) = bundle.kv_storage.finalize().await {
            tracing::warn!("Failed to finalize KV storage for scope {scope_key}: {e}");
        }
        if let Err(e) = bundle.vector_storage.finalize().await {
            tracing::warn!("Failed to finalize vector storage for scope {scope_key}: {e}");
        }
        if let Err(e) = bundle.graph_storage.finalize().await {
            tracing::warn!("Failed to finalize graph storage for scope {scope_key}: {e}");
        }
        if let Err(e) = bundle.doc_status_storage.finalize().await {
            tracing::warn!("Failed to finalize doc status storage for scope {scope_key}: {e}");
        }
    }

    /// Shutdown hook: finalize every cached context
    /// `extraLucene.forEach { runCatching { it.close } }`).
    pub async fn finalize_all(&self) {
        let mut cache = self.cache.lock().await;
        for (workspace, context, keyword) in cache.drain(..) {
            self.finalize_entry(&workspace, &context, keyword).await;
        }
    }

    /// Invalidate the cached context for a specific workspace so the next
    /// request rebuilds it (potentially with a new tenant config).
    pub(crate) async fn invalidate_cached_workspace(&self, workspace: &str) {
        let mut cache = self.cache.lock().await;
        if let Some(pos) = cache.iter().position(|(w, _, _)| w == workspace) {
            let (key, ctx, kw) = cache.remove(pos);
            drop(cache);
            self.finalize_entry(&key, &ctx, kw).await;
            tracing::info!("Invalidated cached workspace context: {workspace}");
        }
    }
}

#[async_trait]
impl WorkspaceResolver for WorkspaceResolverImpl {
    async fn resolve(
        &self,
        workspace: Option<&str>,
        biz_id: Option<&str>,
    ) -> AppResult<ResolvedScope> {
        let workspace = workspace.map(str::trim).filter(|w| !w.is_empty());
        let biz_id = biz_id.map(str::trim).filter(|b| !b.is_empty());
        // Omitted/blank components resolve to the server defaults.
        let workspace = workspace.unwrap_or(self.default_workspace.as_str());
        let biz_id = biz_id.unwrap_or(self.default_biz_id.as_str());
        // require(WORKSPACE_PATTERN) -> IllegalArgumentException -> 400.
        // biz_id shares the charset (it lands in SQL/Cypher/file paths too).
        validate_workspace(workspace).map_err(|e| AppError::bad_request(e.to_string()))?;
        validate_workspace(biz_id).map_err(|e| AppError::bad_request(e.to_string()))?;

        if workspace == self.default_workspace {
            return Ok(ResolvedScope {
                context: self.default_context.clone(),
                biz_id: biz_id.to_string(),
            });
        }

        let mut cache = self.cache.lock().await;
        if let Some(pos) = cache.iter().position(|(w, _, _)| w == workspace) {
            // LRU access: move to the tail (most recently used).
            let entry = cache.remove(pos);
            let context = entry.1.clone();
            cache.push(entry);
            return Ok(ResolvedScope {
                context,
                biz_id: biz_id.to_string(),
            });
        }

        let (context, keyword) = self
            .build_context(workspace)
            .await
            .map_err(|e| AppError::status(500, format!("Failed to build workspace context: {e}")))?;

        if cache.len() >= MAX_CACHED_WORKSPACES {
            let (evicted_workspace, evicted_ctx, evicted_kw) = cache.remove(0);
            self.finalize_entry(&evicted_workspace, &evicted_ctx, evicted_kw)
                .await;
        }
        cache.push((workspace.to_string(), context.clone(), keyword));
        Ok(ResolvedScope {
            context,
            biz_id: biz_id.to_string(),
        })
    }

    async fn invalidate_workspace(&self, workspace: &str) {
        self.invalidate_cached_workspace(workspace).await;
    }
}
