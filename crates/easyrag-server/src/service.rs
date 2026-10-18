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

use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::llm::embedding_func::EmbeddingContext;
use easyrag_core::model::DocStatus;
use easyrag_core::storage::traits::StorageBundle;
use easyrag_core::storage::user::{UserRecord, UserStorage};
use easyrag_core::util::hash::{compute_args_hash, compute_mdhash_id};
use easyrag_core::util::text::clean_text_for_extraction;
use easyrag_core::util::tokenizer::TokenizerProvider;
use easyrag_core::JsonMap;
use easyrag_pipeline::ingest::InsertDocumentExtras;
use easyrag_pipeline::parser::DocumentProcessingOptions;
use easyrag_pipeline::{
    cleanup_graph_by_chunks, IngestionPipeline, QueryEngine, StructureQueryStrategy,
};

use crate::auth::JwtAuthProvider;
use crate::error::{AppError, AppResult};
use crate::util::truncate_string;

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
            .map_err(|e| AppError::internal(format!("JWT generation failed: {e}")))
    }

    /// Seed the default `admin/admin` user when the user store is empty
    ///; logs a WARN urging a password change.
    pub async fn ensure_default_admin(&self) -> AppResult<()> {
        if self.user_storage.count().await? > 0 {
            return Ok(());
        }
        let hash = bcrypt::hash(DEFAULT_ADMIN_PASSWORD, BCRYPT_COST)
            .map_err(|e| AppError::internal(format!("BCrypt hashing failed: {e}")))?;
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

/// R1: archive-related fields accepted by ingest (`POST /api/documents/text`).
/// All optional; absent fields keep the existing/inherited value.
#[derive(Clone, Debug, Default)]
pub struct ArchiveFields {
    pub archived: Option<bool>,
    pub level: Option<i32>,
    pub ref_ids: Option<Vec<String>>,
    pub merged_into: Option<String>,
}

impl ArchiveFields {
    /// Whether the caller supplied any archive field at all.
    pub fn is_empty(&self) -> bool {
        self.archived.is_none()
            && self.level.is_none()
            && self.ref_ids.is_none()
            && self.merged_into.is_none()
    }

    /// Build the partial-update map for a doc_status row.
    pub fn to_status_patch(&self) -> JsonMap {
        let mut patch = JsonMap::new();
        if let Some(archived) = self.archived {
            patch.insert("archived".into(), JsonValue::Bool(archived));
        }
        if let Some(level) = self.level {
            patch.insert("level".into(), JsonValue::from(level));
        }
        if let Some(ref_ids) = &self.ref_ids {
            patch.insert(
                "ref_ids".into(),
                JsonValue::Array(
                    ref_ids
                        .iter()
                        .map(|id| JsonValue::String(id.clone()))
                        .collect(),
                ),
            );
        }
        if let Some(merged_into) = &self.merged_into {
            patch.insert(
                "merged_into".into(),
                JsonValue::String(merged_into.clone()),
            );
        }
        patch
    }

    /// Read archive fields from a status/KV JSON map (legacy rows return None).
    pub fn from_map(map: &JsonMap) -> Self {
        Self {
            archived: map.get("archived").and_then(JsonValue::as_bool),
            level: map
                .get("level")
                .and_then(|v| v.as_i64().map(|n| n as i32)),
            ref_ids: map.get("ref_ids").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
            }),
            merged_into: map
                .get("merged_into")
                .and_then(|v| match v {
                    JsonValue::String(s) => Some(s.clone()),
                    _ => None,
                }),
        }
    }

    /// Fill in any None fields from an existing status/KV row so that a
    /// content overwrite does not silently reset archive flags.
    pub fn inherit_from(self, base: &JsonMap) -> Self {
        let base = Self::from_map(base);
        Self {
            archived: self.archived.or(base.archived),
            level: self.level.or(base.level),
            ref_ids: self.ref_ids.or_else(|| base.ref_ids.clone()),
            merged_into: self.merged_into.or_else(|| base.merged_into.clone()),
        }
    }
}

/// Wire defaults of the ingest-time similarity probe (`returnSimilar` on
/// `POST /api/documents/text`). Constants rather than config: a caller that
/// wants other values passes them per request.
pub const DEFAULT_SIMILAR_TOP_K: usize = 5;
pub const DEFAULT_SIMILAR_THRESHOLD: f32 = 0.8;
/// `contentSummary` budget, counted in characters and never in bytes.
pub const SIMILAR_SUMMARY_CHARS: usize = 500;

/// Chunk-recall budget: returning K *documents* needs more than K chunks, since
/// one document can hold several of the top chunk hits.
const SIMILAR_CHUNK_BUDGET_FACTOR: usize = 4;
const SIMILAR_CHUNK_BUDGET_MIN: usize = 20;
const SIMILAR_CHUNK_BUDGET_MAX: usize = 200;
/// Token ceiling for the probe text handed to the embedding provider. An
/// ingested document can be arbitrarily long while the provider's window is
/// not; for a duplicate check the leading window carries the topic.
const SIMILAR_EMBED_TOKEN_CAP: usize = 1024;

/// One already-stored document similar to the content being ingested.
pub struct SimilarDocument {
    pub doc_id: String,
    pub external_id: Option<String>,
    pub file_path: String,
    /// Cosine similarity against the ingested text, higher = more similar.
    pub score: f32,
    pub content_summary: String,
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
    ///
    /// R5 stability contract: this is a pure function of `external_id` alone, so
    /// repeated upserts of the same key (identical or changed content, re-index,
    /// archive, unarchive) always resolve to the same docId. The docId is never
    /// re-derived from content on the external-id path.
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
    ///
    /// The sequence itself lives in [`easyrag_pipeline::graph_cleanup`], shared
    /// with the facade's delete path so the two cannot drift apart again.
    async fn cleanup_knowledge_graph(
        &self,
        graph_storage: &dyn easyrag_core::storage::traits::GraphStorage,
        vector_storage: &dyn easyrag_core::storage::traits::VectorStorage,
        chunk_ids: &HashSet<String>,
        biz_id: &str,
    ) -> AppResult<()> {
        let report = cleanup_graph_by_chunks(graph_storage, vector_storage, chunk_ids, biz_id)
            .await?;
        tracing::debug!(
            "Graph cleanup: removed {} nodes, {} edges; trimmed {} nodes",
            report.deleted_nodes,
            report.deleted_edges,
            report.trimmed_nodes
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
        archive: &ArchiveFields,
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
                        archived: archive.archived,
                        level: archive.level,
                        ref_ids: archive.ref_ids.clone(),
                        merged_into: archive.merged_into.clone(),
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

        if existing.as_ref().is_some_and(|e| {
            e.get("content_hash").and_then(JsonValue::as_str) == Some(content_hash.as_str())
        }) {
            // Idempotent short-circuit: identical content, nothing to re-index.
            // R1: archive fields can still change without touching chunks.
            if !archive.is_empty() {
                let patch = archive.to_status_patch();
                // R4: chunk rows carry the flags set at the last real ingest,
                // so a flag-only change must be mirrored here too.
                ops.bundle
                    .vector_storage
                    .update_doc_archive_fields(&doc_id, &patch, biz_id)
                    .await?;
                // The doc KV row is the source of truth for reprocessing, so
                // keep it in sync with the status/chunk changes.
                if let Some(mut doc) = ops.bundle.kv_storage.get_by_id(&doc_id, biz_id).await? {
                    doc.extend(patch.clone());
                    let mut kv_batch = HashMap::new();
                    kv_batch.insert(doc_id.clone(), doc);
                    ops.bundle.kv_storage.upsert(kv_batch, biz_id).await?;
                }
                let mut batch = HashMap::new();
                batch.insert(doc_id.clone(), patch);
                ops.bundle
                    .doc_status_storage
                    .upsert(batch, biz_id)
                    .await?;
            }
            return Ok(UpsertResult {
                doc_id,
                created: false,
                idempotent_skip: true,
            });
        }
        // Content is changing: preserve existing archive fields that the caller
        // did not explicitly supply, so a content rewrite does not unarchive or
        // reset consolidation metadata.
        let effective_archive = existing
            .as_ref()
            .map(|e| archive.clone().inherit_from(e))
            .unwrap_or_else(|| archive.clone());
        if existing.is_some() {
            check_pipeline_busy(&ops.pipeline)?;
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
                    archived: effective_archive.archived,
                    level: effective_archive.level,
                    ref_ids: effective_archive.ref_ids.clone(),
                    merged_into: effective_archive.merged_into.clone(),
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
            return Err(AppError::not_found(format!(
                "No document with external_id: {external_id}"
            )));
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
            // Surfaced so a polling caller can report *why* a document is
            // failed instead of only that it is — the retry loop otherwise
            // has no signal but the server log.
            "error_msg": status.get("error_msg").cloned().unwrap_or(JsonValue::Null),
            "metadata": doc.get("user_metadata").cloned().unwrap_or_else(|| json!({})),
            // R1 (archive layer): consolidation fields, defaults for legacy rows.
            "archived": status.get("archived").and_then(JsonValue::as_bool).unwrap_or(false),
            "level": status.get("level").and_then(JsonValue::as_i64).unwrap_or(0),
            "ref_ids": status.get("ref_ids").cloned().unwrap_or_else(|| json!([])),
            "merged_into": status.get("merged_into").cloned().unwrap_or(JsonValue::Null),
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
            .ok_or_else(|| AppError::not_found(format!("Document not found: {doc_id}")))?;
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
            "error_msg": status.as_ref().and_then(|s| s.get("error_msg")).cloned().unwrap_or(JsonValue::Null),
            "metadata": doc.get("user_metadata").cloned().unwrap_or_else(|| json!({})),
            // R1 (archive layer): consolidation fields, defaults for legacy rows.
            "archived": status.as_ref()
                .and_then(|s| s.get("archived"))
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            "level": status.as_ref()
                .and_then(|s| s.get("level"))
                .and_then(JsonValue::as_i64)
                .unwrap_or(0),
            "ref_ids": status.as_ref()
                .and_then(|s| s.get("ref_ids"))
                .cloned()
                .unwrap_or_else(|| json!([])),
            "merged_into": status.as_ref()
                .and_then(|s| s.get("merged_into"))
                .cloned()
                .unwrap_or(JsonValue::Null),
        }))
    }

    // ------------------------------------------------------------------
    // R3 (archive layer): batch read by docId
    // ------------------------------------------------------------------

    /// Batch-read full documents by docId, **including archived ones**.
    ///
    /// Missing/unauthorised ids are simply absent from the result instead of
    /// failing the whole request; the caller inspects the response to detect
    /// them. Callers bound the input size at the REST boundary.
    pub async fn get_documents_by_ids(
        &self,
        doc_ids: &[String],
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<JsonValue> {
        let mut documents = Vec::with_capacity(doc_ids.len());
        for doc_id in doc_ids {
            let doc = ops.bundle.kv_storage.get_by_id(doc_id, biz_id).await?;
            let status = ops
                .bundle
                .doc_status_storage
                .get_by_id(doc_id, biz_id)
                .await?;
            // A doc that exists in either store is reported; missing in both
            // means "not found here" and is omitted by contract.
            if doc.is_none() && status.is_none() {
                continue;
            }
            documents.push(json!({
                "doc_id": doc_id,
                "external_id": doc.as_ref()
                    .and_then(|d| d.get("external_id"))
                    .or_else(|| status.as_ref().and_then(|s| s.get("external_id")))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "file_path": doc.as_ref()
                    .and_then(|d| d.get("file_path"))
                    .or_else(|| status.as_ref().and_then(|s| s.get("file_path")))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "content": doc.as_ref()
                    .and_then(|d| d.get("content"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "status": status.as_ref()
                    .and_then(|s| s.get("status"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "create_time": doc.as_ref()
                    .and_then(|d| d.get("create_time"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "chunks_count": status.as_ref()
                    .and_then(|s| s.get("chunks_count"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "error_msg": status.as_ref()
                    .and_then(|s| s.get("error_msg"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
                "has_structure": status.as_ref()
                    .and_then(|s| s.get("has_structure"))
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                "metadata": doc.as_ref()
                    .and_then(|d| d.get("user_metadata"))
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                // R1: archive fields, always present (defaults when a legacy
                // row predates them).
                "archived": status.as_ref()
                    .and_then(|s| s.get("archived"))
                    .and_then(JsonValue::as_bool)
                    .unwrap_or(false),
                "level": status.as_ref()
                    .and_then(|s| s.get("level"))
                    .and_then(JsonValue::as_i64)
                    .unwrap_or(0),
                "ref_ids": status.as_ref()
                    .and_then(|s| s.get("ref_ids"))
                    .cloned()
                    .unwrap_or_else(|| json!([])),
                "merged_into": status.as_ref()
                    .and_then(|s| s.get("merged_into"))
                    .cloned()
                    .unwrap_or(JsonValue::Null),
            }));
        }
        Ok(JsonValue::Array(documents))
    }

    // ------------------------------------------------------------------
    // R4 (archive layer): flag-only update, no re-index
    // ------------------------------------------------------------------

    /// Apply archive-related flags to existing documents without touching
    /// chunks or embeddings.
    ///
    /// Returns one entry per requested id: `status` is `success`,
    /// `not_found`, or `error` (with a message). A bad id never fails the
    /// batch. Only fields actually supplied are written (partial update).
    pub async fn archive_documents(
        &self,
        doc_ids: &[String],
        fields: &ArchiveFields,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<JsonValue> {
        let patch = fields.to_status_patch();
        let mut results = Vec::with_capacity(doc_ids.len());
        for doc_id in doc_ids {
            // Scope check: the status row is the authoritative existence
            // signal (it carries biz_id and is what retrieval filters on).
            let existing = ops
                .bundle
                .doc_status_storage
                .get_by_id(doc_id, biz_id)
                .await?;
            if existing.is_none() {
                results.push(json!({
                    "docId": doc_id,
                    "status": "not_found",
                }));
                continue;
            }
            if patch.is_empty() {
                results.push(json!({
                    "docId": doc_id,
                    "status": "success",
                    "updated": false,
                }));
                continue;
            }
            // R4: propagate the archive fields to the chunk rows first. The
            // JSONB merge is idempotent, so a failure here leaves both stores
            // untouched and the caller can simply retry.
            if let Err(e) = ops
                .bundle
                .vector_storage
                .update_doc_archive_fields(doc_id, &patch, biz_id)
                .await
            {
                tracing::warn!("Archive vector sync failed for doc {doc_id}: {e:?}");
                results.push(json!({
                    "docId": doc_id,
                    "status": "error",
                    "message": format!("vector store update failed: {e}"),
                }));
                continue;
            }
            // The doc KV row is the source of truth for reprocessing; keep it
            // in sync so a later re-index does not silently revert the flags.
            if let Some(mut doc) = ops.bundle.kv_storage.get_by_id(doc_id, biz_id).await? {
                doc.extend(patch.clone());
                let mut kv_batch = HashMap::new();
                kv_batch.insert(doc_id.clone(), doc);
                if let Err(e) = ops.bundle.kv_storage.upsert(kv_batch, biz_id).await {
                    tracing::warn!("Archive KV sync failed for doc {doc_id}: {e:?}");
                    results.push(json!({
                        "docId": doc_id,
                        "status": "error",
                        "message": format!("kv store update failed: {e}"),
                    }));
                    continue;
                }
            }
            let mut batch = HashMap::new();
            batch.insert(doc_id.clone(), patch.clone());
            match ops.bundle.doc_status_storage.upsert(batch, biz_id).await {
                Ok(()) => results.push(json!({
                    "docId": doc_id,
                    "status": "success",
                    "updated": true,
                })),
                Err(e) => {
                    tracing::warn!("Archive update failed for doc {doc_id}: {e:?}");
                    results.push(json!({
                        "docId": doc_id,
                        "status": "error",
                        "message": e.to_string(),
                    }));
                }
            }
        }
        Ok(JsonValue::Array(results))
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
            .ok_or_else(|| AppError::not_found(format!("Document not found: {doc_id}")))?;

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

    // ------------------------------------------------------------------
    // PUT semantics: update document content by docId
    // ------------------------------------------------------------------

    /// Replace the content of an existing document identified by `doc_id`.
    ///
    /// Returns 404 when the document does not exist. When content is identical
    /// the operation short-circuits (zero embedding/LLM cost). Otherwise the old
    /// derived data (chunks, graph, structure index) is cascade-deleted and the
    /// document is re-inserted under the same `doc_id`.
    #[allow(clippy::too_many_arguments)]
    pub async fn update_document_by_id(
        &self,
        doc_id: &str,
        text: &str,
        file_path: Option<&str>,
        create_time: Option<i64>,
        metadata: Option<&HashMap<String, String>>,
        options: Option<&DocumentProcessingOptions>,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<UpsertResult> {
        // 1. Verify existence
        let existing_status = ops
            .bundle
            .doc_status_storage
            .get_by_id(doc_id, biz_id)
            .await?;
        if existing_status.is_none() {
            return Err(AppError::not_found(format!(
                "Document not found: {doc_id}"
            )));
        }

        // 2. Content-hash idempotency
        let content_hash = compute_args_hash([text]);
        let existing = existing_status.unwrap();
        if existing
            .get("content_hash")
            .and_then(JsonValue::as_str)
            == Some(content_hash.as_str())
        {
            return Ok(UpsertResult {
                doc_id: doc_id.to_string(),
                created: false,
                idempotent_skip: true,
            });
        }

        // Archive flags are not part of the PUT payload; preserve them across
        // a content rewrite so the document stays archived/merged.
        let archive = ArchiveFields::from_map(&existing);

        // 3. Pipeline must not be mid-batch
        check_pipeline_busy(&ops.pipeline)?;

        // 4. Preserve existing external_id and file_path from KV
        let doc_kv = ops.bundle.kv_storage.get_by_id(doc_id, biz_id).await?;
        let external_id = doc_kv
            .as_ref()
            .and_then(|m| m.get("external_id"))
            .and_then(JsonValue::as_str)
            .map(str::to_string);
        let resolved_file_path = file_path
            .or_else(|| {
                doc_kv
                    .as_ref()
                    .and_then(|m| m.get("file_path"))
                    .and_then(JsonValue::as_str)
            })
            .unwrap_or("api_input")
            .to_string();

        // 5. Cascade-delete old derived data (chunks, graph, structure, status)
        self.cascade_delete_document(doc_id, false, false, ops, biz_id)
            .await?;

        // 6. Re-insert with same docId
        ops.pipeline
            .insert_document(
                text,
                &resolved_file_path,
                options,
                Some(&InsertDocumentExtras {
                    external_id,
                    doc_id_override: Some(doc_id.to_string()),
                    create_time,
                    user_metadata: metadata.cloned(),
                    archived: archive.archived,
                    level: archive.level,
                    ref_ids: archive.ref_ids.clone(),
                    merged_into: archive.merged_into.clone(),
                }),
                biz_id,
            )
            .await?;

        Ok(UpsertResult {
            doc_id: doc_id.to_string(),
            created: false,
            idempotent_skip: false,
        })
    }

    // ------------------------------------------------------------------
    // Ingest-time similarity probe (returnSimilar)
    // ------------------------------------------------------------------

    /// Rank chunk-level vector hits into document-level matches.
    ///
    /// The vector store holds more than document chunks — extracted and
    /// hand-made entity vectors live in the same table — and only chunk records
    /// carry `full_doc_id`, so its absence is what marks a hit as not a
    /// document at all. Several chunks of one document collapse into a single
    /// entry scored by the best of them.
    fn rank_similar_hits(
        hits: &[JsonMap],
        exclude_doc_id: &str,
        top_k: usize,
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let mut best: HashMap<String, f32> = HashMap::new();
        for hit in hits {
            let Some(doc_id) = hit
                .get("full_doc_id")
                .and_then(JsonValue::as_str)
                .filter(|id| !id.is_empty() && *id != exclude_doc_id)
            else {
                continue;
            };
            let score = hit
                .get("score")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.0) as f32;
            best.entry(doc_id.to_string())
                .and_modify(|kept| *kept = kept.max(score))
                .or_insert(score);
        }
        let mut ranked: Vec<(String, f32)> = best
            .into_iter()
            .filter(|(_, score)| *score >= threshold)
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    /// Already-indexed documents of the same biz shard similar to `text`,
    /// excluding `exclude_doc_id` — the document this text is being written as.
    ///
    /// Pure vector recall on purpose: the hybrid path fuses with RRF, which
    /// overwrites `score` with a rank-derived value around 0.01 and would turn
    /// the caller's `threshold` into noise. The probe text is embedded with
    /// [`EmbeddingContext::Document`] because this compares a document against
    /// documents — indistinguishable from `Query` while no asymmetric embedding
    /// prefix is configured, and the correct side of the prefix once one is.
    ///
    /// Only documents whose chunks already reached the vector store can match:
    /// ingestion is fire-and-forget, so a pending neighbour is invisible here.
    pub async fn find_similar_documents(
        &self,
        text: &str,
        exclude_doc_id: &str,
        top_k: usize,
        threshold: f32,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<Vec<SimilarDocument>> {
        // The same cleaning the pipeline applies before chunking, so the probe
        // is compared against what actually got indexed. `None` means the text
        // is too short to have been indexed at all.
        let Some(cleaned) = clean_text_for_extraction(text, None) else {
            return Ok(Vec::new());
        };
        let tokenizer = TokenizerProvider::get_tokenizer();
        let tokens = tokenizer.encode(&cleaned);
        let probe = if tokens.len() > SIMILAR_EMBED_TOKEN_CAP {
            // `decode_window`, not `decode`: the cap lands mid-character on a
            // CJK body, where one character spans several tokens.
            tokenizer.decode_window(&tokens[..SIMILAR_EMBED_TOKEN_CAP])
        } else {
            cleaned
        };

        let vector_storage = &ops.bundle.vector_storage;
        let embeddings = vector_storage
            .embedding_func()
            .invoke(vec![probe.clone()], Some(EmbeddingContext::Document))
            .await?;
        let Some(embedding) = embeddings.into_iter().next() else {
            return Ok(Vec::new());
        };

        let budget = (top_k * SIMILAR_CHUNK_BUDGET_FACTOR)
            .clamp(SIMILAR_CHUNK_BUDGET_MIN, SIMILAR_CHUNK_BUDGET_MAX);
        let hits = vector_storage
            .query(&probe, budget, Some(&embedding), None, None, false, biz_id)
            .await?;
        let ranked = Self::rank_similar_hits(&hits, exclude_doc_id, top_k, threshold);

        let mut similar = Vec::with_capacity(ranked.len());
        for (doc_id, score) in ranked {
            match self
                .describe_similar_document(&doc_id, score, ops, biz_id)
                .await?
            {
                Some(document) => similar.push(document),
                None => tracing::warn!(
                    "Similar-document hit {doc_id} has neither a KV nor a registry record; \
                     dropped as a dangling chunk"
                ),
            }
        }
        Ok(similar)
    }

    /// Resolve one ranked hit into its payload. The document body's head is the
    /// summary; the registry's own 100-character `content_summary` is the
    /// fallback for a KV record that is missing or stored without a body.
    /// `None` when neither store knows the document, i.e. the hit was a chunk
    /// outliving its document.
    async fn describe_similar_document(
        &self,
        doc_id: &str,
        score: f32,
        ops: &LifecycleOps,
        biz_id: &str,
    ) -> AppResult<Option<SimilarDocument>> {
        let bundle = &ops.bundle;
        let doc = bundle.kv_storage.get_by_id(doc_id, biz_id).await?;
        let content = doc
            .as_ref()
            .and_then(|d| d.get("content"))
            .and_then(JsonValue::as_str);
        let status = if content.is_some() {
            None
        } else {
            bundle.doc_status_storage.get_by_id(doc_id, biz_id).await?
        };
        if doc.is_none() && status.is_none() {
            return Ok(None);
        }
        // The document record wins per field: it is written at insert time, the
        // registry row only carries what a later status transition knew.
        let field = |key: &str| -> Option<String> {
            doc.as_ref()
                .and_then(|d| d.get(key))
                .and_then(JsonValue::as_str)
                .or_else(|| {
                    status
                        .as_ref()
                        .and_then(|s| s.get(key))
                        .and_then(JsonValue::as_str)
                })
                .map(str::to_string)
        };
        let content_summary = match content {
            Some(body) => truncate_string(Some(body), SIMILAR_SUMMARY_CHARS),
            None => field("content_summary").unwrap_or_default(),
        };
        Ok(Some(SimilarDocument {
            doc_id: doc_id.to_string(),
            external_id: field("external_id"),
            file_path: field("file_path").unwrap_or_default(),
            score,
            content_summary,
        }))
    }
}

/// P2.16 pipeline concurrency guard: 409 while the pipeline is busy.
/// Checks the target workspace's pipeline so cross-workspace requests
/// are not blocked by the default workspace's busy state.
///
/// Single source of truth shared by `service` and `routes::documents`.
/// `AppError::conflict(msg)` and `AppError::status(409, msg)` are wire-identical
/// (both build `Status(409, msg)`), so merging the two former copies is
/// behavior-preserving.
pub(crate) fn check_pipeline_busy(pipeline: &IngestionPipeline) -> AppResult<()> {
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

#[cfg(test)]
mod similar_ranking_tests {
    use super::*;

    /// A chunk hit as the vector backends return it. `doc_id: None` reproduces
    /// an entity vector, which shares the table but is not a document.
    fn hit(doc_id: Option<&str>, score: Option<f64>) -> JsonMap {
        let mut record = JsonMap::new();
        if let Some(doc_id) = doc_id {
            record.insert(
                "full_doc_id".into(),
                JsonValue::String(doc_id.to_string()),
            );
        } else {
            record.insert("entity_name".into(), JsonValue::String("Alice".into()));
        }
        if let Some(score) = score {
            record.insert("score".into(), JsonValue::from(score));
        }
        record
    }

    fn rank(hits: &[JsonMap], exclude: &str, top_k: usize, threshold: f32) -> Vec<(String, f32)> {
        DocumentLifecycleService::rank_similar_hits(hits, exclude, top_k, threshold)
    }

    #[test]
    fn threshold_drops_weak_matches_and_keeps_the_boundary() {
        let hits = vec![
            hit(Some("doc-strong"), Some(0.95)),
            hit(Some("doc-exact"), Some(0.8)),
            hit(Some("doc-weak"), Some(0.75)),
        ];
        let ranked = rank(&hits, "doc-self", 5, 0.8);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, "doc-strong");
        assert_eq!(ranked[1].0, "doc-exact");
    }

    #[test]
    fn the_ingested_document_never_matches_itself() {
        // The idempotent-skip path writes content whose chunks are already
        // indexed, so without this exclusion a document tops its own list at 1.0.
        let hits = vec![
            hit(Some("doc-self"), Some(1.0)),
            hit(Some("doc-other"), Some(0.9)),
        ];
        let ranked = rank(&hits, "doc-self", 5, 0.8);
        assert_eq!(ranked, vec![("doc-other".to_string(), 0.9)]);
    }

    #[test]
    fn entity_vectors_and_blank_doc_ids_are_not_documents() {
        let hits = vec![
            hit(None, Some(0.99)),
            hit(Some(""), Some(0.99)),
            hit(Some("doc-real"), Some(0.85)),
        ];
        let ranked = rank(&hits, "doc-self", 5, 0.8);
        assert_eq!(ranked, vec![("doc-real".to_string(), 0.85)]);
    }

    #[test]
    fn chunks_of_one_document_collapse_to_its_best_score() {
        let hits = vec![
            hit(Some("doc-a"), Some(0.82)),
            hit(Some("doc-b"), Some(0.9)),
            hit(Some("doc-a"), Some(0.97)),
        ];
        let ranked = rank(&hits, "doc-self", 5, 0.8);
        assert_eq!(
            ranked,
            vec![("doc-a".to_string(), 0.97), ("doc-b".to_string(), 0.9)]
        );
    }

    #[test]
    fn results_are_score_descending_and_truncated_to_top_k() {
        let hits = vec![
            hit(Some("doc-1"), Some(0.81)),
            hit(Some("doc-2"), Some(0.99)),
            hit(Some("doc-3"), Some(0.9)),
        ];
        let ranked = rank(&hits, "doc-self", 2, 0.8);
        assert_eq!(
            ranked,
            vec![("doc-2".to_string(), 0.99), ("doc-3".to_string(), 0.9)]
        );
    }

    #[test]
    fn a_hit_without_a_score_ranks_as_zero() {
        // A backend that stops emitting `score` must yield no matches rather
        // than silently pass every document through the threshold.
        let hits = vec![hit(Some("doc-a"), None)];
        assert!(rank(&hits, "doc-self", 5, 0.8).is_empty());
        assert_eq!(rank(&hits, "doc-self", 5, 0.0), vec![("doc-a".to_string(), 0.0)]);
    }

    #[test]
    fn no_hits_means_no_similar_documents() {
        assert!(rank(&[], "doc-self", 5, 0.8).is_empty());
    }
}
