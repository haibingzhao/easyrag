//! Storage assembly matrix: picks the KV / Vector / Graph / DocStatus /
//! keyword-search backends from `ServerConfig`. The assembled set is handed to `main` and to `WorkspaceResolverImpl`.

use std::sync::Arc;

use anyhow::Context;

use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::storage::traits::{
    DocStatusStorage, GraphStorage, KVStorage, StorageNameSpace, VectorStorage,
};
use easyrag_server::ServerConfig;
use easyrag_storage::json::JsonKVStorage;
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};

use crate::embedding::make_embedding_func;

pub(crate) struct AssembledStorage {
    pub(crate) kv: Arc<dyn KVStorage>,
    pub(crate) vector: Arc<dyn VectorStorage>,
    pub(crate) graph: Arc<dyn GraphStorage>,
    pub(crate) doc_status: Arc<dyn DocStatusStorage>,
    pub(crate) keyword: Option<Arc<dyn KeywordSearchable>>,
    /// Held for graceful shutdown.
    pub(crate) pg_db: Option<easyrag_storage::postgres::PgDatabase>,
    pub(crate) neo4j_db: Option<easyrag_storage::neo4j::Neo4jDatabase>,
}

pub(crate) async fn assemble_storage(config: &ServerConfig) -> anyhow::Result<AssembledStorage> {
    let use_postgres = config.storage_type.eq_ignore_ascii_case("postgres");
    let use_neo4j_graph = config.graph_storage_type.eq_ignore_ascii_case("neo4j");
    // Workspace is the only structural isolation dimension; biz_id is a
    // per-call filter value and never participates in storage assembly.
    let workspace = config.workspace.clone();

    // --- PostgreSQL family -------------------------------------------
    let mut pg_db: Option<(
        easyrag_storage::postgres::PgDatabase,
        easyrag_storage::postgres::PgConfig,
    )> = None;
    if use_postgres {
        let pg_config = easyrag_storage::postgres::PgConfig {
            host: config.pg_host.clone(),
            port: config.pg_port,
            user: config.pg_user.clone(),
            password: config.pg_password.clone(),
            database: config.pg_database.clone(),
            workspace: config.workspace.clone(),
            embedding_dim: config.embedding_dim,
            ssl_mode: config.pg_ssl_mode.clone(),
            ssl_cert: config.pg_ssl_cert.clone(),
            ssl_key: config.pg_ssl_key.clone(),
            ssl_root_cert: config.pg_ssl_root_cert.clone(),
            ssl_crl: config.pg_ssl_crl.clone(),
            fts_config: config.keyword_search_language.clone(),
            ..Default::default()
        };
        let db = easyrag_storage::postgres::PgDatabase::new(pg_config.clone())
            .context("PostgreSQL connect")?;
        db.initialize().await.context("PostgreSQL initialize")?;
        tracing::info!(
            "PostgreSQL storage enabled: {}:{}/{}",
            config.pg_host,
            config.pg_port,
            config.pg_database
        );
        pg_db = Some((db, pg_config));
    }

    // --- Neo4j --------------------------------------------------------
    let mut neo4j_db: Option<easyrag_storage::neo4j::Neo4jDatabase> = None;
    if use_neo4j_graph {
        let db = easyrag_storage::neo4j::Neo4jDatabase::new(easyrag_storage::neo4j::Neo4jConfig {
            uri: config.neo4j_uri.clone(),
            user: config.neo4j_user.clone(),
            password: config.neo4j_password.clone(),
            database: config.neo4j_database.clone(),
            max_connections: config.neo4j_max_pool_size,
        })
        .await
        .context("Neo4j connect")?;
        neo4j_db = Some(db);
    }

    if use_neo4j_graph && !use_postgres {
        tracing::warn!(
            "storageType=neo4j only affects graph storage; KV/Vector/DocStatus fall back to \
             InMemory (non-persistent). Consider graph-storage-type=neo4j with \
             storage-type=postgres for full persistence."
        );
    }

    // --- KV / Vector / DocStatus --------------------------------------
    // PgVectorStorage doubles as the keyword search backend; keep the
    // concrete Arc so both trait views can be derived from it.
    let mut pg_keyword: Option<Arc<dyn KeywordSearchable>> = None;
    let (kv, vector, doc_status): (Arc<dyn KVStorage>, Arc<dyn VectorStorage>, Arc<dyn DocStatusStorage>) =
        if let Some((db, pg_config)) = &pg_db {
            let kv = easyrag_storage::postgres::PgKVStorage::new(
                "kv",
                workspace.clone(),
                db.clone(),
            )?;
            kv.initialize().await?;
            let vector = Arc::new(easyrag_storage::postgres::PgVectorStorage::new(
                "vectors",
                workspace.clone(),
                make_embedding_func(config)?,
                db.clone(),
                pg_config.clone(),
            )?);
            vector.initialize().await?;
            pg_keyword = Some(Arc::clone(&vector) as Arc<dyn KeywordSearchable>);
            let doc_status = easyrag_storage::postgres::PgDocStatusStorage::new(
                "doc_status",
                workspace.clone(),
                db.clone(),
            )?;
            doc_status.initialize().await?;
            (Arc::new(kv), vector, Arc::new(doc_status))
        } else if config.storage_type.eq_ignore_ascii_case("json") {
            // Rust port: JSON persistence covers the KV namespace; the
            // other namespaces stay in-memory for now.
            tracing::warn!(
                "storage-type=json persists only the KV namespace; \
                 vector/graph/doc-status remain in-memory"
            );
            let kv = JsonKVStorage::with_storage_dir(
                "kv",
                workspace.clone(),
                config.working_dir.clone(),
            );
            kv.initialize().await?;
            (
                Arc::new(kv),
                Arc::new(InMemoryVectorStorage::new(
                    "vectors",
                    workspace.clone(),
                    make_embedding_func(config)?,
                )),
                Arc::new(InMemoryDocStatusStorage::new(
                    "doc_status",
                    workspace.clone(),
                )),
            )
        } else {
            (
                Arc::new(InMemoryKVStorage::new("kv", workspace.clone())),
                Arc::new(InMemoryVectorStorage::new(
                    "vectors",
                    workspace.clone(),
                    make_embedding_func(config)?,
                )),
                Arc::new(InMemoryDocStatusStorage::new(
                    "doc_status",
                    workspace.clone(),
                )),
            )
        };

    // --- Graph: neo4j > postgres > in-memory ---------------------------
    let graph: Arc<dyn GraphStorage> = if let Some(db) = &neo4j_db {
        let storage = easyrag_storage::neo4j::Neo4jGraphStorage::new(
            "graph",
            workspace.clone(),
            db.clone(),
        )?;
        storage.initialize().await?;
        tracing::info!("Neo4j graph storage enabled: {}", config.neo4j_uri);
        Arc::new(storage)
    } else if let Some((db, _)) = &pg_db {
        let storage = easyrag_storage::postgres::PgGraphStorage::new(
            "graph",
            workspace.clone(),
            db.clone(),
        )?;
        storage.initialize().await?;
        Arc::new(storage)
    } else {
        Arc::new(InMemoryGraphStorage::new("graph", workspace.clone()))
    };

    // --- Keyword search: PG tsvector or embedded tantivy ----------------
    let keyword: Option<Arc<dyn KeywordSearchable>> = if !config.keyword_search_enabled {
        None
    } else if let Some(pg_kw) = pg_keyword {
        // PgVectorStorage already implements KeywordSearchable.
        Some(pg_kw)
    } else {
        let index_dir = if config.lucene_index_dir.is_empty() {
            config.working_dir.clone()
        } else {
            config.lucene_index_dir.clone()
        };
        Some(Arc::new({
            let search = easyrag_storage::tantivy_search::TantivyKeywordSearch::new(
                workspace.clone(),
                &index_dir,
                64,
            )
            .context("tantivy keyword index")?;
            tracing::info!(
                "tantivy keyword index under {index_dir} (workspace '{}'); the writer lock \
                 allows only one instance per workspace",
                config.workspace
            );
            search
        }))
    };

    let pg_db = pg_db.map(|(db, _)| db);
    Ok(AssembledStorage {
        kv,
        vector,
        graph,
        doc_status,
        keyword,
        pg_db,
        neo4j_db,
    })
}
