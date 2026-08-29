//! R2: request-level workspace routing (
//! bean + `WorkspaceResolverImpl`): LRU-cached per-workspace contexts with
//! dedicated storage instances.

use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;

use easyrag_core::llm::service::LlmService;
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::tenant_config::TenantConfigStorage;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageBundle, StorageNameSpace, VectorStorage,
};
use easyrag_pipeline::{IngestionPipeline, QueryEngine, StructureQueryStrategy, TreeSearchEngine};
use easyrag_server::service::{ResolvedScope, WorkspaceContext, WorkspaceResolver};
use easyrag_server::{AppError, AppResult, ServerConfig};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};

use crate::config::validate_workspace;
use crate::embedding::make_embedding_func;

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
pub(crate) struct WorkspaceResolverImpl {
    pub(crate) default_workspace: String,
    pub(crate) default_biz_id: String,
    pub(crate) default_context: WorkspaceContext,
    pub(crate) llm: Arc<dyn LlmService>,
    pub(crate) config: ServerConfig,
    pub(crate) tenant_config_storage: Arc<dyn TenantConfigStorage>,
    pub(crate) pg_db: Option<easyrag_storage::postgres::PgDatabase>,
    pub(crate) neo4j_db: Option<easyrag_storage::neo4j::Neo4jDatabase>,
    pub(crate) cache: tokio::sync::Mutex<
        Vec<(
            String,
            WorkspaceContext,
            Option<Arc<easyrag_storage::tantivy_search::TantivyKeywordSearch>>,
        )>,
    >,
}

impl WorkspaceResolverImpl {
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

        let mut pipeline = IngestionPipeline::new(
            self.llm.clone(),
            kv.clone(),
            vector.clone(),
            graph.clone(),
            doc_status.clone(),
            Arc::new(make_embedding_func(&resolved_config)?),
        )
        .with_language(resolved_config.language.clone())
        .with_max_concurrency(resolved_config.llm_max_async);
        if let Some(keyword) = &keyword {
            pipeline = pipeline.with_keyword_search(keyword.clone());
        }
        let pipeline = Arc::new(pipeline);

        let structure = Arc::new(
            StructureQueryStrategy::new(
                TreeSearchEngine::new(self.llm.clone()),
                None, // no tool-calling LLM impl available
                kv.clone(),
                vector.clone(),
                self.llm.clone(),
            )
            .with_doc_status_storage(doc_status.clone()),
        );

        let mut engine = QueryEngine::new(
            self.llm.clone(),
            vector.clone(),
            graph.clone(),
            kv.clone(),
        )
        .with_language(resolved_config.language.clone())
        .with_rrf_k(resolved_config.hybrid_rrf_k);
        if let Some(keyword) = &keyword {
            engine = engine.with_keyword_search(keyword.clone());
        }
        engine = engine.with_structure_strategy(structure.clone());
        let engine = Arc::new(engine);

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
    pub(crate) async fn finalize_all(&self) {
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
