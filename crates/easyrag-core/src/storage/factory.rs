//! Storage factory.

use crate::llm::EmbeddingFunc;
use crate::storage::traits::{DocStatusStorage, GraphStorage, KVStorage, VectorStorage};
use crate::JsonMap;
use std::sync::Arc;

/// Storage backend type enumeration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageBackend {
    /// JSON file-based (zero dependency, dev/test)
    Json,
    /// In-memory (fastest, dev/test)
    Memory,
    /// PostgreSQL (production)
    Postgres,
    /// Neo4j (graph storage only, mixed backend)
    Neo4j,
}

impl StorageBackend {
    pub fn from_value(value: &str) -> Result<Self, crate::CoreError> {
        match value.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "memory" => Ok(Self::Memory),
            "postgres" | "postgresql" => Ok(Self::Postgres),
            "neo4j" => Ok(Self::Neo4j),
            other => Err(crate::CoreError::InvalidArgument(format!(
                "Unknown storage backend: {other}"
            ))),
        }
    }
}

/// Storage configuration for factory creation.
#[derive(Clone, Debug)]
pub struct StorageConfig {
    pub backend: StorageBackend,
    pub workspace: String,
    pub working_dir: String,
    pub global_config: JsonMap,
    pub embedding_func: Option<Arc<EmbeddingFunc>>,
    // PostgreSQL specific
    pub postgres_url: Option<String>,
    pub postgres_user: Option<String>,
    pub postgres_password: Option<String>,
    pub postgres_database: Option<String>,
    // Neo4j specific
    pub neo4j_uri: Option<String>,
    pub neo4j_user: Option<String>,
    pub neo4j_password: Option<String>,
    pub neo4j_database: Option<String>,
    pub neo4j_max_connection_pool_size: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Json,
            workspace: "default".to_string(),
            working_dir: "./rag_storage".to_string(),
            global_config: JsonMap::new(),
            embedding_func: None,
            postgres_url: None,
            postgres_user: None,
            postgres_password: None,
            postgres_database: None,
            neo4j_uri: None,
            neo4j_user: None,
            neo4j_password: None,
            neo4j_database: None,
            neo4j_max_connection_pool_size: 100,
        }
    }
}

/// Factory interface for creating storage instances.
/// Implementations are provided by the easyrag-storage crate.
pub trait StorageFactory: Send + Sync {
    fn create_kv_storage(
        &self,
        namespace: &str,
        config: &StorageConfig,
    ) -> crate::Result<Arc<dyn KVStorage>>;

    fn create_vector_storage(
        &self,
        namespace: &str,
        config: &StorageConfig,
    ) -> crate::Result<Arc<dyn VectorStorage>>;

    fn create_graph_storage(
        &self,
        namespace: &str,
        config: &StorageConfig,
    ) -> crate::Result<Arc<dyn GraphStorage>>;

    fn create_doc_status_storage(
        &self,
        namespace: &str,
        config: &StorageConfig,
    ) -> crate::Result<Arc<dyn DocStatusStorage>>;
}
