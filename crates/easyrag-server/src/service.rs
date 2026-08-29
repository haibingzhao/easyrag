//! Service layer:
//! `DocumentLifecycleService` (cascade deletion, R3 idempotent upsert,
//! R5 full-text retrieval, R6 single-document indexing),
//! `WorkspaceResolver` (R2 request-level workspace routing) and
//! `AuthService` (BCrypt credential verification + default admin seeding).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures::FutureExt;
use serde_json::{json, Value as JsonValue};
use tokio::sync::Mutex as AsyncMutex;

use easyrag_core::constants::{GRAPH_FIELD_SEP, STRUCTURE_INDEX_KEY_PREFIX};
use easyrag_core::model::DocStatus;
use easyrag_core::storage::traits::StorageBundle;
use easyrag_core::storage::user::{UserRecord, UserStorage};
use easyrag_core::util::hash::{compute_args_hash, compute_mdhash_id};
use easyrag_core::JsonMap;
use easyrag_pipeline::ingest::InsertDocumentExtras;
use easyrag_pipeline::parser::DocumentProcessingOptions;
use easyrag_pipeline::{IngestionPipeline, QueryEngine, StructureQueryStrategy};

use crate::auth::JwtAuthProvider;
use crate::error::{AppError, AppResult};

// ---------------------------------------------------------------------
// WorkspaceResolver (R2)
// ---------------------------------------------------------------------

/// Per-workspace execution context (R2 request-level workspace routing).
/// The default context wraps the server-level singletons; additional
/// contexts are cached per requested workspace by the resolver implementation.
#[derive(Clone)]
pub struct WorkspaceContext {
    pub bundle: StorageBundle,
    pub pipeline: Arc<IngestionPipeline>,
    pub query_engine: Arc<QueryEngine>,
    pub structure_strategy: Option<Arc<StructureQueryStrategy>>,
}

/// Resolves the execution context for a request-level workspace (R2).
/// A null/blank workspace resolves to the server default workspace; a
/// null/blank biz_id normalizes to the default business line. The
/// normalized biz_id is returned alongside the context so callers can
/// pass it down as the per-call storage filter.
#[async_trait]
pub trait WorkspaceResolver: Send + Sync {
    async fn resolve(
        &self,
        workspace: Option<&str>,
        biz_id: Option<&str>,
    ) -> AppResult<ResolvedScope>;

    /// Invalidate the cached context for a workspace so the next resolve
    /// rebuilds it (e.g. after a tenant config update). Default is a no-op.
    async fn invalidate_workspace(&self, _workspace: &str) {
        // Default: no-op. The app-level resolver overrides this.
    }
}

/// Result of [`WorkspaceResolver::resolve`]: the per-workspace context
/// plus the normalized biz_id to use as the storage filter value.
#[derive(Clone)]
pub struct ResolvedScope {
    pub context: WorkspaceContext,
    pub biz_id: String,
}

// ---------------------------------------------------------------------
// AuthService
// ---------------------------------------------------------------------

/// BCrypt cost matching Spring Security's `BCryptPasswordEncoder` default.
pub const BCRYPT_COST: u32 = 10;

/// Username/password of the default admin seeded into an empty user store.
pub const DEFAULT_ADMIN_USERNAME: &str = "admin";
pub const DEFAULT_ADMIN_PASSWORD: &str = "admin";

/// Credential verification and default-admin seeding: BCrypt hashes via the `bcrypt` crate.
pub struct AuthService {
    user_storage: Arc<dyn UserStorage>,
    jwt: Arc<JwtAuthProvider>,
}

impl AuthService {
    pub fn new(user_storage: Arc<dyn UserStorage>, jwt: Arc<JwtAuthProvider>) -> Self {
        Self { user_storage, jwt }
    }

    /// Verify credentials; missing user and wrong password both yield `None`
    ///
    pub async fn authenticate(
        &self,
        username: &str,
        password: &str,
    ) -> AppResult<Option<UserRecord>> {
        let Some(user) = self.user_storage.find_by_username(username).await? else {
            return Ok(None);
        };
        match bcrypt::verify(password, &user.password_hash) {
            Ok(true) => Ok(Some(user)),
            _ => Ok(None),
        }
    }

    /// Issue a signed JWT for an authenticated user.
    pub fn generate_token(&self, username: &str) -> AppResult<String> {
        self.jwt
            .generate_token(username)
            .map_err(|e| AppError::status(500, format!("JWT generation failed: {e}")))
    }

    /// Seed the default `admin/admin` user when the user store is empty
    ///; logs a WARN urging a password change.
    pub async fn ensure_default_admin(&self) -> AppResult<()> {
        if self.user_storage.count().await? > 0 {
            return Ok(());
        }
        let hash = bcrypt::hash(DEFAULT_ADMIN_PASSWORD, BCRYPT_COST)
            .map_err(|e| AppError::status(500, format!("BCrypt hashing failed: {e}")))?;
        let now = Utc::now();
        let admin = UserRecord {
            username: DEFAULT_ADMIN_USERNAME.to_string(),
            password_hash: hash,
            role: "admin".to_string(),
            created_at: now,
            updated_at: now,
        };
        self.user_storage.upsert(&admin).await?;
        tracing::warn!(
            "User store was empty; seeded default user '{DEFAULT_ADMIN_USERNAME}' with the \
             default password. Change it as soon as possible."
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------
// DocumentLifecycleService
// ---------------------------------------------------------------------

/// Storage bundle + pipeline pair that lifecycle operations execute against.
#[derive(Clone)]
pub struct LifecycleOps {
    pub bundle: StorageBundle,
    pub pipeline: Arc<IngestionPipeline>,
}

/// Result of an idempotent upsert (R3).
pub struct UpsertResult {
    pub doc_id: String,
    pub created: bool,
    pub idempotent_skip: bool,
}

/// Document lifecycle orchestration: cascade deletion, idempotent upsert (R3),
/// full-text retrieval (R5) and single-document synchronous indexing (R6).
///
/// All operations accept an optional [`LifecycleOps`] so request-level
/// workspace routing (R2) can swap in a per-workspace storage bundle and
/// pipeline while defaulting to the server-level singletons.
pub struct DocumentLifecycleService {
    pub default_ops: LifecycleOps,
    /// Striped mutexes serializing concurrent upserts on the same externalId.
    upsert_mutexes: std::sync::Mutex<HashMap<String, Arc<AsyncMutex<()>>>>,
}

impl DocumentLifecycleService {
    pub fn new(
        pipeline: Arc<IngestionPipeline>,
        doc_status_storage: Arc<dyn easyrag_core::storage::traits::DocStatusStorage>,
        kv_storage: Arc<dyn easyrag_core::storage::traits::KVStorage>,
        vector_storage: Arc<dyn easyrag_core::storage::traits::VectorStorage>,
        graph_storage: Arc<dyn easyrag_core::storage::traits::GraphStorage>,
    ) -> Self {
        let workspace = doc_status_storage.workspace().to_string();
        Self {
            default_ops: LifecycleOps {
                bundle: StorageBundle {
                    workspace,
                    kv_storage,
                    vector_storage,
                    graph_storage,
                    doc_status_storage,
                },
                pipeline,
            },
            upsert_mutexes: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// R3: deterministic docId derivation from the client business key.
    /// The "extid:" salt keeps this namespace disjoint from content-hash docIds.
    pub fn derive_external_doc_id(external_id: &str) -> String {
        compute_mdhash_id(&format!("extid:{external_id}"), "doc-")
    }

    // ------------------------------------------------------------------
    // Cascade deletion (M0)
    // ------------------------------------------------------------------

    /// Cascade delete a document: chunks (KV + vector) + graph entities/edges +
    /// structure index + doc KV + doc_status.
    ///
    /// Graph cleanup follows the EasyRAG facade semantics: nodes whose source_id
    /// partially references this document keep living with trimmed source_id;
    /// orphaned nodes/edges are removed. Documents ingested with skip_kg=true
    /// never touched the graph, so the expensive full-graph scan is skipped.
    pub async fn cascade_delete_document(
        &self,
        doc_id: &str,
        delete_file: bool,
        _delete_llm_cache: bool,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<()> {
        let bundle = &ops.bundle;
        let doc_status = bundle.doc_status_storage.get_by_id(doc_id, biz_id).await?;
        let doc_kv = bundle.kv_storage.get_by_id(doc_id, biz_id).await?;
        let chunks_list: Vec<String> = doc_status
            .as_ref()
            .and_then(|m| m.get("chunks_list"))
            .and_then(JsonValue::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(JsonValue::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();

        // 1. Delete chunks from KV and vector storage
        if !chunks_list.is_empty() {
            bundle.kv_storage.delete(&chunks_list, biz_id).await?;
            bundle.vector_storage.delete(&chunks_list, biz_id).await?;
        }

        // 2. Clean graph references (skip entirely for skip_kg documents: fast path)
        let skip_kg = doc_kv
            .as_ref()
            .and_then(|m| m.get("skip_kg"))
            .and_then(JsonValue::as_bool)
            == Some(true);
        if !chunks_list.is_empty() && !skip_kg {
            let chunk_ids: HashSet<String> = chunks_list.iter().cloned().collect();
            self.cleanup_knowledge_graph(
                bundle.graph_storage.as_ref(),
                bundle.vector_storage.as_ref(),
                &chunk_ids,
                biz_id,
            )
            .await?;
        }

        // 3. Delete the document itself, including its structure index (if any)
        bundle
            .kv_storage
            .delete(
                &[
                    doc_id.to_string(),
                    format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}"),
                ],
                biz_id,
            )
            .await?;
        bundle.doc_status_storage.delete(&[doc_id.to_string()], biz_id).await?;

        // 4. Optionally delete source file
        if delete_file {
            let file_path = doc_status
                .as_ref()
                .and_then(|m| m.get("file_path"))
                .and_then(JsonValue::as_str)
                .or_else(|| {
                    doc_kv
                        .as_ref()
                        .and_then(|m| m.get("file_path"))
                        .and_then(JsonValue::as_str)
                });
            if let Some(file_path) = file_path {
                if file_path != "api_input" && file_path != "unknown_source" {
                    if let Err(e) = std::fs::remove_file(file_path) {
                        // a missing/unremovable file never fails the delete.
                        if e.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!(
                                "Failed to delete source file for doc {}: {}",
                                doc_id,
                                e
                            );
                        }
                    }
                }
            }
        }
        tracing::info!(
            "Cascade delete doc {} ({} chunks, skipKg={})",
            doc_id,
            chunks_list.len(),
            skip_kg
        );
        Ok(())
    }

    /// Remove graph entities/edges that reference the deleted chunks.
    /// Nodes partially referencing other chunks keep living with a trimmed
    /// source_id; fully orphaned nodes/edges are deleted.
    async fn cleanup_knowledge_graph(
        &self,
        graph_storage: &dyn easyrag_core::storage::traits::GraphStorage,
        vector_storage: &dyn easyrag_core::storage::traits::VectorStorage,
        chunk_ids: &HashSet<String>,
        biz_id: &str,
    ) -> AppResult<()> {
        let split_sources = |source_id: &str| -> HashSet<String> {
            // split on either delimiter.
            source_id
                .split(GRAPH_FIELD_SEP)
                .flat_map(|part| part.split(','))
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        };

        let mut nodes_to_delete: Vec<String> = Vec::new();
        let mut nodes_to_update: Vec<(String, JsonMap)> = Vec::new();

        for node in graph_storage.get_all_nodes(biz_id).await? {
            let Some(node_id) = node.get("id").and_then(JsonValue::as_str) else {
                continue;
            };
            let source_id = node
                .get("source_id")
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            let mut sources = split_sources(source_id);

            let before = sources.len();
            sources.retain(|s| !chunk_ids.contains(s));
            if sources.len() == before {
                continue;
            }

            if sources.is_empty() {
                nodes_to_delete.push(node_id.to_string());
            } else {
                let mut updated = node.clone();
                updated.insert(
                    "source_id".into(),
                    JsonValue::String(sources.into_iter().collect::<Vec<_>>().join(GRAPH_FIELD_SEP)),
                );
                nodes_to_update.push((node_id.to_string(), updated));
            }
        }

        let mut edges_to_delete: Vec<(String, String)> = Vec::new();
        for edge in graph_storage.get_all_edges(biz_id).await? {
            let Some(source) = edge.get("source").and_then(JsonValue::as_str) else {
                continue;
            };
            let Some(target) = edge.get("target").and_then(JsonValue::as_str) else {
                continue;
            };
            let source_id = edge
                .get("source_id")
                .and_then(JsonValue::as_str)
                .unwrap_or("");
            let sources = split_sources(source_id);

            if sources.iter().any(|s| chunk_ids.contains(s))
                || nodes_to_delete.iter().any(|n| n == source)
                || nodes_to_delete.iter().any(|n| n == target)
            {
                edges_to_delete.push((source.to_string(), target.to_string()));
            }
        }

        if !edges_to_delete.is_empty() {
            graph_storage.remove_edges(&edges_to_delete, biz_id).await?;
        }
        if !nodes_to_delete.is_empty() {
            graph_storage.remove_nodes(&nodes_to_delete, biz_id).await?;
            // Entity description vectors live in vector storage
            vector_storage.delete(&nodes_to_delete, biz_id).await?;
        }
        for (node_id, data) in nodes_to_update.iter() {
            graph_storage.upsert_node(node_id, data.clone(), biz_id).await?;
        }
        tracing::debug!(
            "Graph cleanup: removed {} nodes, {} edges; trimmed {} nodes",
            nodes_to_delete.len(),
            edges_to_delete.len(),
            nodes_to_update.len()
        );
        Ok(())
    }

    // ------------------------------------------------------------------
    // R3: idempotent upsert by external business key
    // ------------------------------------------------------------------

    /// Insert or idempotently overwrite a document.
    ///
    /// Without externalId the behavior is identical to the legacy insert path.
    /// With externalId the docId is derived deterministically; an existing
    /// document with identical content short-circuits (zero embedding/LLM cost),
    /// otherwise the old version is cascade-deleted before re-insert.
    #[allow(clippy::too_many_arguments)]
    pub async fn upsert_text(
        &self,
        text: &str,
        file_path: &str,
        external_id: Option<&str>,
        create_time: Option<i64>,
        metadata: Option<&HashMap<String, String>>,
        options: Option<&DocumentProcessingOptions>,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<UpsertResult> {
        let external_id = external_id.filter(|id| !id.trim().is_empty());
        let Some(external_id) = external_id else {
            let doc_id = ops
                .pipeline
                .insert_document(
                    text,
                    file_path,
                    options,
                    Some(&InsertDocumentExtras {
                        create_time,
                        user_metadata: metadata.cloned(),
                        ..Default::default()
                    }),
                    biz_id,
                )
                .await?;
            return Ok(UpsertResult {
                doc_id,
                created: true,
                idempotent_skip: false,
            });
        };

        // Serialize concurrent writes for the same business key
        let mutex = {
            let mut map = self.upsert_mutexes.lock().unwrap();
            map.entry(external_id.to_string())
                .or_insert_with(|| Arc::new(AsyncMutex::new(())))
                .clone()
        };
        let _guard = mutex.lock().await;

        let doc_id = Self::derive_external_doc_id(external_id);
        let existing = ops.bundle.doc_status_storage.get_by_id(&doc_id, biz_id).await?;
        let content_hash = compute_args_hash([text]);

        if existing.as_ref().map_or(false, |e| {
            e.get("content_hash").and_then(JsonValue::as_str) == Some(content_hash.as_str())
        }) {
            // Idempotent short-circuit: identical content, nothing to do
            return Ok(UpsertResult {
                doc_id,
                created: false,
                idempotent_skip: true,
            });
        }
        if existing.is_some() {
            Self::check_pipeline_busy(&ops.pipeline)?;
            self.cascade_delete_document(&doc_id, false, false, ops, biz_id).await?;
        }
        ops.pipeline
            .insert_document(
                text,
                file_path,
                options,
                Some(&InsertDocumentExtras {
                    external_id: Some(external_id.to_string()),
                    doc_id_override: Some(doc_id.clone()),
                    create_time,
                    user_metadata: metadata.cloned(),
                }),
                biz_id,
            )
            .await?;
        Ok(UpsertResult {
            doc_id,
            created: existing.is_none(),
            idempotent_skip: false,
        })
    }

    /// R3: look up a document by external business key. Returns 404 when absent.
    pub async fn get_by_external_id(
        &self,
        external_id: &str,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<JsonValue> {
        let doc_id = Self::derive_external_doc_id(external_id);
        let status = ops.bundle.doc_status_storage.get_by_id(&doc_id, biz_id).await?;
        let doc = ops.bundle.kv_storage.get_by_id(&doc_id, biz_id).await?;
        // Verify the stored external_id to rule out any hash misuse
        let stored_external_id = doc
            .as_ref()
            .and_then(|m| m.get("external_id"))
            .and_then(JsonValue::as_str);
        if status.is_none() || stored_external_id != Some(external_id) {
            return Err(AppError::status(
                404,
                format!("No document with external_id: {external_id}"),
            ));
        }
        let status = status.unwrap();
        let doc = doc.unwrap();
        Ok(json!({
            "doc_id": doc_id,
            "external_id": external_id,
            "status": status.get("status").cloned().unwrap_or(JsonValue::Null),
            "file_path": doc.get("file_path").cloned()
                .unwrap_or_else(|| status.get("file_path").cloned().unwrap_or(JsonValue::Null)),
            "create_time": doc.get("create_time").cloned().unwrap_or(JsonValue::Null),
            "content": doc.get("content").cloned().unwrap_or(JsonValue::Null),
            "chunks_count": status.get("chunks_count").cloned().unwrap_or(JsonValue::Null),
        }))
    }

    // ------------------------------------------------------------------
    // R5: full document retrieval
    // ------------------------------------------------------------------

    /// R5: return full document text plus status meta information.
    /// Returns 404 when absent.
    pub async fn get_document_full(
        &self,
        doc_id: &str,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<JsonValue> {
        let doc = ops
            .bundle
            .kv_storage
            .get_by_id(doc_id, biz_id)
            .await?
            .ok_or_else(|| AppError::status(404, format!("Document not found: {doc_id}")))?;
        let status = ops.bundle.doc_status_storage.get_by_id(doc_id, biz_id).await?;
        let content_length = doc
            .get("content")
            .and_then(JsonValue::as_str)
            .map(|s| s.chars().count());
        Ok(json!({
            "doc_id": doc_id,
            "content": doc.get("content").cloned().unwrap_or(JsonValue::Null),
            "file_path": doc.get("file_path").cloned().unwrap_or_else(|| status
                .as_ref()
                .and_then(|s| s.get("file_path"))
                .cloned()
                .unwrap_or(JsonValue::Null)),
            "status": status.as_ref().and_then(|s| s.get("status")).cloned().unwrap_or(JsonValue::Null),
            "create_time": doc.get("create_time").cloned().unwrap_or(JsonValue::Null),
            "external_id": doc.get("external_id").cloned().unwrap_or(JsonValue::Null),
            "content_length": content_length,
            "chunks_count": status.as_ref().and_then(|s| s.get("chunks_count")).cloned().unwrap_or(JsonValue::Null),
        }))
    }

    // ------------------------------------------------------------------
    // R6: single-document async indexing (background spawn)
    // ------------------------------------------------------------------

    /// R6: index a single document asynchronously (background spawn), bypassing
    /// the global pipeline queue. The document is atomically claimed
    /// (pending/failed -> processing); a failed claim yields 409. The pipeline
    /// execution is spawned as a background task and returns immediately with
    /// status "processing". The global pipeline only picks up PENDING documents,
    /// so a claimed document cannot be double-processed.
    pub async fn index_single_document(
        &self,
        doc_id: &str,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<JsonValue> {
        let bundle = &ops.bundle;
        let doc_data = bundle
            .kv_storage
            .get_by_id(doc_id, biz_id)
            .await?
            .ok_or_else(|| AppError::status(404, format!("Document not found: {doc_id}")))?;

        // Validate before claiming: a claim flips status to processing, and a
        // post-claim failure would leave the document unclaimable forever.
        let content = doc_data
            .get("content")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                AppError::bad_request(format!("Document {doc_id} has no stored content"))
            })?
            .to_string();

        if !bundle.doc_status_storage.try_claim_for_processing(doc_id, biz_id).await? {
            return Err(AppError::conflict(format!(
                "Document {doc_id} is not claimable for indexing \
                 (missing status record, already processing or processed)"
            )));
        }

        let file_path = doc_data
            .get("file_path")
            .and_then(JsonValue::as_str)
            .unwrap_or("unknown")
            .to_string();
        let options = IngestionPipeline::rebuild_options_from_doc_data(&doc_data);
        let create_time_override = doc_data.get("create_time_epoch").and_then(JsonValue::as_i64);
        let user_metadata = IngestionPipeline::extract_user_metadata(&doc_data);

        // Spawn background task for pipeline execution (fire-and-forget).
        // process_document internally catches all failures and commits FAILED
        // status; the outer guard catches panics so a crashed task still marks
        // the document FAILED instead of leaving it stuck in "processing".
        let pipeline = Arc::clone(&ops.pipeline);
        let doc_id_owned = doc_id.to_string();
        let biz_id_owned = biz_id.to_string();
        let bundle_guard = ops.bundle.clone();
        tokio::spawn(async move {
            tracing::info!("Background indexing started for doc {doc_id_owned}");
            let result = std::panic::AssertUnwindSafe(
                pipeline.process_document(
                    &doc_id_owned,
                    &content,
                    &file_path,
                    Some(&options),
                    create_time_override,
                    user_metadata.as_ref(),
                    &biz_id_owned,
                ),
            )
            .catch_unwind()
            .await;
            match result {
                Ok(()) => {
                    // process_document already committed the final status
                    // (PROCESSED or FAILED); the log stays neutral because
                    // the outcome is only visible via the doc_status record.
                    tracing::info!(
                        "Background indexing task finished for doc {doc_id_owned} \
                         (see doc_status for result)"
                    );
                }
                Err(_) => {
                    tracing::error!(
                        "Background indexing task panicked for doc {doc_id_owned}"
                    );
                    // Fallback so a panicked task does not strand the document
                    // in "processing" (claim only accepts pending/failed).
                    // Mirrors the read-modify-upsert shape of try_claim_for_processing.
                    let fallback = async {
                        let mut updated = bundle_guard
                            .doc_status_storage
                            .get_by_id(&doc_id_owned, &biz_id_owned)
                            .await?
                            .unwrap_or_default();
                        updated.insert(
                            "status".to_string(),
                            JsonValue::String(DocStatus::Failed.value().to_string()),
                        );
                        updated.insert(
                            "error_msg".to_string(),
                            JsonValue::String(
                                "Internal indexing task panicked".to_string(),
                            ),
                        );
                        updated.insert(
                            "updated_at".to_string(),
                            JsonValue::String(
                                Utc::now()
                                    .to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true),
                            ),
                        );
                        let mut map = HashMap::new();
                        map.insert(doc_id_owned.clone(), updated);
                        bundle_guard.doc_status_storage.upsert(map, &biz_id_owned).await
                    };
                    if let Err(e) = fallback.await {
                        tracing::error!(
                            "Failed to mark panicked doc {doc_id_owned} as FAILED: {e}"
                        );
                    }
                }
            }
        });

        Ok(json!({
            "status": "success",
            "doc_id": doc_id,
            "final_status": "processing",
        }))
    }

    fn check_pipeline_busy(pipeline: &IngestionPipeline) -> AppResult<()> {
        let busy = pipeline
            .pipeline_status()
            .get("busy")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        if busy {
            return Err(AppError::conflict(
                "Pipeline is busy processing documents. Please wait or cancel first.",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;
    use easyrag_storage::memory::InMemoryUserStorage;

    fn service(user_storage: Arc<dyn UserStorage>) -> AuthService {
        AuthService::new(user_storage, Arc::new(JwtAuthProvider::new("test-secret", 24)))
    }

    #[tokio::test]
    async fn empty_store_seeds_default_admin() {
        let store = Arc::new(InMemoryUserStorage::new());
        let auth = service(store.clone());
        assert_eq!(store.count().await.unwrap(), 0);
        auth.ensure_default_admin().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        // Idempotent: a second run must not overwrite existing users.
        auth.ensure_default_admin().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn authenticate_verifies_bcrypt_password() {
        let store = Arc::new(InMemoryUserStorage::new());
        let auth = service(store);
        auth.ensure_default_admin().await.unwrap();

        let ok = auth.authenticate("admin", "admin").await.unwrap();
        assert!(ok.is_some());
        assert_eq!(ok.unwrap().role, "admin");

        assert!(auth.authenticate("admin", "wrong").await.unwrap().is_none());
        assert!(auth.authenticate("ghost", "admin").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn generate_token_round_trips_through_provider() {
        let store = Arc::new(InMemoryUserStorage::new());
        let jwt = Arc::new(JwtAuthProvider::new("test-secret", 24));
        let auth = AuthService::new(store, Arc::clone(&jwt));
        let token = auth.generate_token("admin").unwrap();
        assert_eq!(jwt.validate_token(&token).as_deref(), Some("admin"));
    }
}
