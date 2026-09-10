//! Full-text keyword search interface.
//!
//! Implementations provide BM25-style lexical retrieval independent of
//! vector search:
//! - PgVectorStorage: PostgreSQL tsvector + GIN index + ts_rank_cd scoring
//! - TantivyKeywordSearch: embedded tantivy index (replaces Lucene)

use crate::JsonMap;
use async_trait::async_trait;
use std::collections::HashMap;

#[async_trait]
pub trait KeywordSearchable: Send + Sync {
    /// Keyword search returning scored results.
    /// Return format is consistent with VectorStorage.query: list of maps
    /// with "id", "content", "score" keys.
    ///
    /// - `query`: raw query text (tokenized/analyzed internally)
    /// - `time_range_start`: Unix timestamp (seconds), create_time >= this
    /// - `time_range_end`: Unix timestamp (seconds), create_time <= this
    /// - `biz_id`: business-line filter
    async fn keyword_query(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> crate::Result<Vec<JsonMap>>;

    /// Multi-biz variant for `/api/query/data` (bizIds): the biz lines are one
    /// corpus, single global top_k. Backends with a native set filter
    /// (tantivy terms query, PostgreSQL `biz_id = ANY(...)`) override for exact
    /// ordering; the default merges per-biz results by the shared
    /// score-descending rule.
    async fn keyword_query_multi_biz(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_ids: &[&str],
    ) -> crate::Result<Vec<JsonMap>> {
        if biz_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut batches = Vec::with_capacity(biz_ids.len());
        for biz in biz_ids {
            batches.push(crate::storage::traits::tag_multi_biz_batch(
                self.keyword_query(query, top_k, time_range_start, time_range_end, biz)
                    .await?,
                biz,
            ));
        }
        Ok(crate::storage::traits::merge_multi_biz_scored(
            batches, top_k,
        ))
    }

    /// Batch index documents (called during ingestion).
    /// `docs` maps docId -> document data (must contain "content" key).
    async fn index_documents(
        &self,
        docs: HashMap<String, JsonMap>,
        biz_id: &str,
    ) -> crate::Result<()>;

    /// Remove documents from the keyword index.
    async fn remove_documents(&self, ids: &[String], biz_id: &str) -> crate::Result<()>;

    /// Commit/flush the index.
    /// For tantivy: commits the IndexWriter and refreshes the reader.
    /// For PostgreSQL: no-op (trigger maintains tsvector automatically).
    async fn commit_index(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Close and release resources.
    fn close(&self) -> crate::Result<()> {
        Ok(())
    }
}
