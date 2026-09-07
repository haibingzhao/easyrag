//! Domain repositories over the raw storage traits (Phase 5.9 / blueprint #9).
//!
//! Both types are thin, statically-dispatched wrappers around the same
//! `Arc<dyn …Storage>` the pipeline already holds — they add **no** new
//! indirection layer (no `dyn`/`Box` beyond the one the caller passed in).
//! Their job is to pull the hand-built JSON maps out of the call sites:
//!
//!   * three bare `doc_status_storage.upsert(map)` calls each built an
//!     overlapping-but-drifting status map inline (pending insert,
//!     has-structure mark, status transition), and
//!   * three copy-pasted structure-index reads (`get_by_id(PREFIX+doc_id)`
//!     then pluck the `"structure"` field) had already started to diverge.
//!
//! Centralising the map construction means a field can never be added to one
//! site and silently forgotten in another. Behaviour is byte-for-byte the
//! same as the inlined code it replaces — including each caller's distinct
//! "not found" wording, which is why [`StructureIndexRepository::load_raw`]
//! hands back the raw record instead of collapsing the two miss cases.

use std::collections::HashMap;
use std::sync::Arc;

use easyrag_core::constants::STRUCTURE_INDEX_KEY_PREFIX;
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::model::document_structure::DocumentStructureIndex;
use easyrag_core::storage::traits::{DocStatusStorage, KVStorage};
use easyrag_core::util::hash::compute_args_hash;
use easyrag_core::util::json::parse_json_value;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

/// RFC3339 (second precision, UTC `Z`) timestamp — identical to the private
/// `now_iso` the ingestion pipeline has always stamped status/structure rows
/// with, kept local so the repository owns every field it writes.
fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
}

/// Writes document processing status.
///
/// Replaces the three bare `doc_status_storage.upsert(map)` calls that each
/// hand-built a status map inline. Reads and deletes the repository does not
/// model (e.g. `get_docs_by_status`, cascade `delete`) go through
/// [`DocStatusRepository::storage`].
pub struct DocStatusRepository {
    storage: Arc<dyn DocStatusStorage>,
}

impl DocStatusRepository {
    pub fn new(storage: Arc<dyn DocStatusStorage>) -> Self {
        Self { storage }
    }

    /// Underlying storage handle for the read/delete paths the repository does
    /// not wrap.
    pub fn storage(&self) -> &Arc<dyn DocStatusStorage> {
        &self.storage
    }

    /// Create the initial PENDING status row for a freshly-inserted document.
    ///
    /// `created_at` is supplied by the caller (client business time takes
    /// precedence over ingestion time); `updated_at` is stamped now.
    pub async fn mark_pending(
        &self,
        doc_id: &str,
        content: &str,
        file_path: &str,
        created_at_iso: &str,
        external_id: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let mut status_data = JsonMap::new();
        status_data.insert(
            "content_summary".into(),
            JsonValue::String(content.chars().take(100).collect()),
        );
        status_data.insert(
            "content_length".into(),
            JsonValue::from(content.chars().count() as u64),
        );
        status_data.insert("file_path".into(), JsonValue::String(file_path.to_string()));
        status_data.insert(
            "status".into(),
            JsonValue::String(DocStatus::Pending.value().to_string()),
        );
        status_data.insert(
            "created_at".into(),
            JsonValue::String(created_at_iso.to_string()),
        );
        status_data.insert("updated_at".into(), JsonValue::String(now_iso()));
        status_data.insert(
            "content_hash".into(),
            JsonValue::String(compute_args_hash([content])),
        );
        if let Some(ext) = external_id {
            status_data.insert("external_id".into(), JsonValue::String(ext.to_string()));
        }
        let mut batch = HashMap::new();
        batch.insert(doc_id.to_string(), status_data);
        self.storage.upsert(batch, biz_id).await
    }

    /// Transition a document's processing status.
    ///
    /// `error_msg` is **always** written (`None` → JSON null) so a stale error
    /// from a previous FAILED run is cleared on the next transition. The caller
    /// decides the error policy: the ingestion pipeline logs-and-swallows a
    /// failed transition (a status write must not abort an otherwise-successful
    /// document run), so this returns the `Result` rather than hiding it.
    pub async fn mark_status(
        &self,
        doc_id: &str,
        status: DocStatus,
        chunks_count: Option<usize>,
        chunks_list: Option<Vec<String>>,
        error_msg: Option<&str>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let mut update = JsonMap::new();
        update.insert(
            "status".into(),
            JsonValue::String(status.value().to_string()),
        );
        update.insert("updated_at".into(), JsonValue::String(now_iso()));
        update.insert(
            "error_msg".into(),
            match error_msg {
                Some(msg) => JsonValue::String(msg.to_string()),
                None => JsonValue::Null,
            },
        );
        if let Some(count) = chunks_count {
            update.insert("chunks_count".into(), JsonValue::from(count as u64));
        }
        if let Some(list) = chunks_list {
            update.insert(
                "chunks_list".into(),
                JsonValue::Array(list.into_iter().map(JsonValue::String).collect()),
            );
        }
        let mut batch = HashMap::new();
        batch.insert(doc_id.to_string(), update);
        self.storage.upsert(batch, biz_id).await
    }

    /// Flag a document as having a structure index (M5 registry mark).
    pub async fn mark_has_structure(&self, doc_id: &str, biz_id: &str) -> easyrag_core::Result<()> {
        let mut status_mark = JsonMap::new();
        status_mark.insert("has_structure".into(), JsonValue::Bool(true));
        status_mark.insert("updated_at".into(), JsonValue::String(now_iso()));
        let mut batch = HashMap::new();
        batch.insert(doc_id.to_string(), status_mark);
        self.storage.upsert(batch, biz_id).await
    }
}

/// Reads and writes the per-document structure index stored under the
/// [`STRUCTURE_INDEX_KEY_PREFIX`] KV namespace.
///
/// Borrows the caller's `Arc<dyn KVStorage>` so the three structs that touch
/// the index (ingestion pipeline, structure strategy, agent executor) share one
/// implementation without any of them changing a field type.
pub struct StructureIndexRepository<'a> {
    kv: &'a Arc<dyn KVStorage>,
}

impl<'a> StructureIndexRepository<'a> {
    pub fn new(kv: &'a Arc<dyn KVStorage>) -> Self {
        Self { kv }
    }

    fn key(doc_id: &str) -> String {
        format!("{STRUCTURE_INDEX_KEY_PREFIX}{doc_id}")
    }

    /// Persist a serialized structure index for `doc_id`.
    pub async fn save(
        &self,
        doc_id: &str,
        structure_json: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let mut structure_data = JsonMap::new();
        structure_data.insert("doc_id".into(), JsonValue::String(doc_id.to_string()));
        structure_data.insert(
            "structure".into(),
            JsonValue::String(structure_json.to_string()),
        );
        structure_data.insert("created_at".into(), JsonValue::String(now_iso()));
        let mut batch = HashMap::new();
        batch.insert(Self::key(doc_id), structure_data);
        self.kv.upsert(batch, biz_id).await
    }

    /// Fetch the raw structure-index record, or `None` when the key is absent.
    ///
    /// Deliberately returns the whole record rather than the inner
    /// `"structure"` string: callers distinguish "no index row" from "row
    /// exists but has no structure field" with their own user-facing wording,
    /// and collapsing the two here would change those messages.
    pub async fn load_raw(&self, doc_id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        self.kv.get_by_id(&Self::key(doc_id), biz_id).await
    }

    /// Fetch and deserialize the structure index. Returns `None` when the row
    /// is absent, has no `"structure"` field, or fails to deserialize (the last
    /// case is logged, matching the structure strategy's prior behaviour).
    pub async fn load_typed(
        &self,
        doc_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<DocumentStructureIndex>> {
        let Some(data) = self.load_raw(doc_id, biz_id).await? else {
            return Ok(None);
        };
        let Some(structure_json) = data.get("structure").and_then(|v| v.as_str()) else {
            return Ok(None);
        };
        match parse_json_value::<DocumentStructureIndex>(structure_json) {
            Some(index) => Ok(Some(index)),
            None => {
                tracing::warn!("Failed to deserialize structure index for {}", doc_id);
                Ok(None)
            }
        }
    }
}
