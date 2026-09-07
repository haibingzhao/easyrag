//! Batched `INSERT ... ON CONFLICT` upsert shared by the Postgres backends.
//!
//! Two rules are defined exactly once here, because each one is a promise the
//! rest of the code relies on rather than a formatting preference:
//!
//! 1. **One batch = one statement.** The chunk size comes from
//!    `PgConfig::upsert_batch_size` clamped by `PgBatchLimits`; the resulting
//!    round-trip count is pinned by the SQL golden tests in
//!    `tests/postgres_storage.rs` (`3 records = 1 round-trip`).
//! 2. **A failed batch is replayed record by record, and the replay keeps `?`.**
//!    `postgres.rs` documents this as "preserves per-record failure
//!    granularity": one oversized or malformed row must not take the whole
//!    batch down with it, and a row that fails on its own must still abort the
//!    upsert instead of being swallowed.
//!
//! Static dispatch only: both helpers are generic and monomorphize per backend,
//! so there is no `dyn`, no `Box`, and no allocation beyond what the
//! hand-written loops they replace already did.

use std::future::Future;

use crate::postgres::PgDb;

/// Conflict target of every workspace-scoped table: the tenant primary key.
///
/// Interpolated verbatim into `ON CONFLICT (...)`, so it is a constant instead
/// of a literal repeated at five call sites that could drift from the DDL.
pub(crate) const TENANT_CONFLICT_TARGET: &str = "id, workspace, biz_id";

/// Renders the `INSERT ... ON CONFLICT ... DO UPDATE SET` statement shared by
/// the batched KV and vector upserts.
///
/// Only the two batched builders use it: the single-row graph/doc-status/user
/// upserts keep their own hand-laid multi-line SQL (pinned by golden
/// snapshots), so forcing them through this renderer would change statement
/// bytes for no reduction in duplicated decisions.
pub(crate) struct PgUpsertBuilder<'a> {
    table: &'a str,
    columns: &'a str,
    assignments: &'a str,
}

impl<'a> PgUpsertBuilder<'a> {
    /// `columns` is the INSERT column list, `assignments` the text after
    /// `DO UPDATE SET`. The conflict target is always [`TENANT_CONFLICT_TARGET`].
    pub(crate) fn new(table: &'a str, columns: &'a str, assignments: &'a str) -> Self {
        Self {
            table,
            columns,
            assignments,
        }
    }

    /// `INSERT INTO <table> (<columns>) VALUES <row>, <row> ON CONFLICT
    /// (id, workspace, biz_id) DO UPDATE SET <assignments>` — value tuples
    /// joined by `", "`, byte-identical to the concatenated statements this
    /// replaces.
    pub(crate) fn sql<I>(&self, rows: I) -> String
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut sql = format!("INSERT INTO {} ({}) VALUES ", self.table, self.columns);
        let mut first = true;
        for row in rows {
            if !first {
                sql.push_str(", ");
            }
            first = false;
            sql.push_str(row.as_ref());
        }
        sql.push_str(&format!(
            " ON CONFLICT ({}) DO UPDATE SET {}",
            TENANT_CONFLICT_TARGET, self.assignments
        ));
        sql
    }
}

/// Split `entries` into batches of at most `batch_size`, `prepare` each batch
/// into the rows to write, then execute `build_sql(&rows)` as ONE statement.
///
/// `prepare` runs **once per batch and never during the replay**, so an
/// expensive or externally visible side effect — the vector path's single
/// `embed(texts)` call per batch — is not repeated (and re-paid for) per record.
/// `build_sql` is deliberately synchronous: the helper owns every `db.execute`,
/// which is what keeps the round-trip count a property of this one function.
///
/// On a batch failure the batch is replayed one record at a time; the `?` on
/// the replay means the first record that still fails aborts the whole call,
/// which is the per-record failure granularity `postgres.rs` promises.
pub(crate) async fn batched_with_fallback<'e, D, T, R, P, B, PF>(
    db: &D,
    entries: &'e [T],
    batch_size: usize,
    mut prepare: P,
    mut build_sql: B,
) -> easyrag_core::Result<()>
where
    D: PgDb,
    P: FnMut(&'e [T]) -> PF,
    PF: Future<Output = easyrag_core::Result<Vec<R>>>,
    B: FnMut(&[R]) -> String,
{
    // `max(1)` only guards `chunks(0)` (which panics); both callers already
    // clamp through `PgBatchLimits`, so the batch sizes are unchanged.
    for batch in entries.chunks(batch_size.max(1)) {
        let rows = prepare(batch).await?;
        if db.execute(&build_sql(&rows)).await.is_err() {
            // Fall back to row-by-row (preserves per-record failure granularity).
            for row in &rows {
                db.execute(&build_sql(std::slice::from_ref(row))).await?;
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "postgres"))]
mod tests {
    use super::*;

    /// Pinned byte-for-byte by `testdata/golden/sql/kv_upsert_single.snap` and
    /// `vector_upsert_single.snap`; changing this renderer must change them.
    #[test]
    fn renders_the_batched_kv_statement() {
        let upsert = PgUpsertBuilder::new(
            "kv_cache",
            "id, workspace, biz_id, content, updated_at",
            "content = EXCLUDED.content, updated_at = NOW()",
        );
        assert_eq!(
            "INSERT INTO kv_cache (id, workspace, biz_id, content, updated_at) VALUES \
             ('k1', 'ws', 'default', '{}'::jsonb, NOW()) \
             ON CONFLICT (id, workspace, biz_id) DO UPDATE SET \
             content = EXCLUDED.content, updated_at = NOW()",
            upsert.sql(["('k1', 'ws', 'default', '{}'::jsonb, NOW())"])
        );
    }

    #[test]
    fn joins_multiple_value_tuples_with_comma_space() {
        let upsert = PgUpsertBuilder::new("t", "id", "id = EXCLUDED.id");
        assert_eq!(
            "INSERT INTO t (id) VALUES ('a'), ('b'), ('c') \
             ON CONFLICT (id, workspace, biz_id) DO UPDATE SET id = EXCLUDED.id",
            upsert.sql(["('a')", "('b')", "('c')"])
        );
    }

    #[test]
    fn an_empty_batch_still_renders_a_complete_statement() {
        let upsert = PgUpsertBuilder::new("t", "id", "id = EXCLUDED.id");
        assert_eq!(
            "INSERT INTO t (id) VALUES  ON CONFLICT (id, workspace, biz_id) DO UPDATE SET id = EXCLUDED.id",
            upsert.sql(std::iter::empty::<&str>())
        );
    }

    #[test]
    fn the_tenant_conflict_target_is_the_scoped_primary_key() {
        assert_eq!("id, workspace, biz_id", TENANT_CONFLICT_TARGET);
    }
}
