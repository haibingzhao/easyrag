//! Storage-side renderers for [`easyrag_core::model::TenantScope`].
//!
//! `core` owns the *data* (workspace + biz_id); rendering that pair into a
//! query language is an implementation detail of the backends, so the
//! renderers live here. Two rules are defined exactly once and must never be
//! re-decided at a call site, because getting either wrong is a cross-tenant
//! leak rather than a cosmetic bug:
//!
//! 1. **Every business-table predicate goes through [`TenantScopeSql`]** — no
//!    hand-written `workspace = '...' AND biz_id = '...'` strings.
//! 2. **An empty `biz_id` is the read-path wildcard** ([`TenantScopeCypher`] /
//!    [`TenantScopeSql::list_sql`]), never a literal match against `''`.
//!    Write paths and single-record reads keep the exact scope.
//!
//! All renderers are `impl`-by-generic (static dispatch): no `dyn`, no `Box`,
//! no extra allocation over the `format!` calls they replace.

use easyrag_core::model::AsTenantScope;

/// Escape a value for inclusion in a single-quoted SQL literal.
#[cfg(feature = "postgres")]
pub(crate) fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

/// Render values as a SQL `IN` list body: `'a','b','c'` (no surrounding
/// parentheses, no spaces — byte-identical to the six inline copies it
/// replaces).
#[cfg(feature = "postgres")]
pub(crate) fn sql_in_list<'a, I>(values: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    values
        .into_iter()
        .map(|v| format!("'{}'", sql_escape(v)))
        .collect::<Vec<_>>()
        .join(",")
}

/// SQL rendering of a tenant scope.
#[cfg(feature = "postgres")]
pub(crate) trait TenantScopeSql: AsTenantScope {
    /// Exact scope: `workspace = '<ws>' AND biz_id = '<biz>'`, qualified with
    /// `<alias>.` when the query joins more than one scoped table.
    fn sql(&self, alias: Option<&str>) -> String {
        let (w, b) = match alias {
            Some(a) => (format!("{a}.workspace"), format!("{a}.biz_id")),
            None => ("workspace".to_string(), "biz_id".to_string()),
        };
        format!(
            "{w} = '{}' AND {b} = '{}'",
            sql_escape(self.workspace()),
            sql_escape(self.biz_id())
        )
    }

    /// List/count scope: an empty `biz_id` widens to the whole workspace
    /// instead of matching the literal empty string. Alias-qualified the same
    /// way [`Self::sql`] is.
    fn list_sql(&self, alias: Option<&str>) -> String {
        if self.is_wildcard() {
            let w = match alias {
                Some(a) => format!("{a}.workspace"),
                None => "workspace".to_string(),
            };
            format!("{w} = '{}'", sql_escape(self.workspace()))
        } else {
            self.sql(alias)
        }
    }

    /// Multi-biz list scope for `/api/query/data` (bizIds): pins the workspace
    /// and filters `biz_id` against a set, so the recall query returns one
    /// globally-ordered top_k instead of per-biz partitions. The workspace
    /// half keeps the same rendering as [`Self::list_sql`]; every element is
    /// escaped by [`sql_in_list`]. Alias-qualified the same way.
    fn list_sql_in(&self, alias: Option<&str>, biz_ids: &[&str]) -> String {
        let w = match alias {
            Some(a) => format!("{a}.workspace"),
            None => "workspace".to_string(),
        };
        let b = match alias {
            Some(a) => format!("{a}.biz_id"),
            None => "biz_id".to_string(),
        };
        format!(
            "{w} = '{}' AND {b} IN ({})",
            sql_escape(self.workspace()),
            sql_in_list(biz_ids.iter().copied())
        )
    }
}

#[cfg(feature = "postgres")]
impl<T: AsTenantScope> TenantScopeSql for T {}

/// Back-tick a workspace name into a Cypher label. Shared by the scope
/// renderers and by the label-only call sites in `neo4j.rs` so the escaping
/// rule has one definition.
#[cfg(feature = "neo4j")]
pub(crate) fn cypher_ws_label(workspace: &str) -> String {
    format!("`{}`", workspace)
}

/// Cypher rendering of a tenant scope. The workspace is a node **label**
/// (back-ticked) and the biz line travels as the `$biz` query parameter, so
/// nothing here interpolates user data into the statement text.
#[cfg(feature = "neo4j")]
pub(crate) trait TenantScopeCypher: AsTenantScope {
    /// Back-ticked workspace label: `` `ws` ``.
    fn ws_label(&self) -> String {
        cypher_ws_label(self.workspace())
    }

    /// Property-map form used right after a label: `` `ws` `` for the wildcard,
    /// `` `ws` {biz_id: $biz} `` for an exact biz line.
    fn node_label(&self) -> String {
        match self.biz_prop_field() {
            Some(field) => format!("{} {{biz_id: {field}}}", self.ws_label()),
            None => self.ws_label(),
        }
    }

    /// The `$biz` binding as it appears inside a property map, or `None` when
    /// the scope is the read-path wildcard: `Some("$biz")`.
    fn biz_prop_field(&self) -> Option<&'static str> {
        if self.is_wildcard() {
            None
        } else {
            Some("$biz")
        }
    }

    /// Entity property map: `{entity_id: <key>}`, extended with `, biz_id: $biz`
    /// for an exact biz line. `key` is a Cypher expression such as `$label` or
    /// an `UNWIND` variable — never user text.
    fn entity_map(&self, key: &str) -> String {
        match self.biz_prop_field() {
            Some(field) => format!("{{entity_id: {key}, biz_id: {field}}}"),
            None => format!("{{entity_id: {key}}}"),
        }
    }

    /// Standalone predicate `<var>.biz_id = $biz`, or `None` for the wildcard.
    /// Callers add their own connector (`WHERE `, ` AND `) and spacing.
    fn biz_predicate(&self, var: &str) -> Option<String> {
        if self.is_wildcard() {
            None
        } else {
            Some(format!("{var}.biz_id = $biz"))
        }
    }
}

#[cfg(feature = "neo4j")]
impl<T: AsTenantScope> TenantScopeCypher for T {}

#[cfg(all(test, feature = "postgres"))]
mod sql_tests {
    use super::*;
    use easyrag_core::model::{TenantScope, TenantScopeRef};

    /// These literals are pinned by `tests/postgres_storage.rs` SQL golden
    /// snapshots; changing one here must change them there.
    #[test]
    fn exact_scope_matches_the_legacy_scope_filter() {
        let scope = TenantScope::new("ws", "default");
        assert_eq!("workspace = 'ws' AND biz_id = 'default'", scope.sql(None));
    }

    #[test]
    fn alias_qualifies_both_columns() {
        let scope = TenantScopeRef::new("ws", "b");
        assert_eq!("e.workspace = 'ws' AND e.biz_id = 'b'", scope.sql(Some("e")));
    }

    #[test]
    fn single_quotes_are_doubled() {
        assert_eq!("workspace = 'a''b' AND biz_id = ''", TenantScope::new("a'b", "").sql(None));
    }

    #[test]
    fn empty_biz_id_widens_only_on_the_list_path() {
        let scope = TenantScope::new("ws", "");
        assert_eq!("workspace = 'ws'", scope.list_sql(None));
        assert_eq!("e.workspace = 'ws'", scope.list_sql(Some("e")));
        // The exact form still pins biz_id — writes must never widen.
        assert_eq!("workspace = 'ws' AND biz_id = ''", scope.sql(None));
        assert_eq!(
            "workspace = 'ws' AND biz_id = 'b'",
            TenantScope::new("ws", "b").list_sql(None)
        );
        assert_eq!(
            "n.workspace = 'ws' AND n.biz_id = 'b'",
            TenantScope::new("ws", "b").list_sql(Some("n"))
        );
    }

    #[test]
    fn in_list_has_no_spaces_and_escapes() {
        assert_eq!("'a','b'", sql_in_list(["a", "b"]));
        assert_eq!("'o''ne'", sql_in_list(["o'ne"]));
        assert_eq!("", sql_in_list(std::iter::empty::<&str>()));
    }

    #[test]
    fn multi_biz_list_scope_pins_workspace_and_escapes_each_element() {
        let scope = TenantScope::new("ws", "unused");
        assert_eq!(
            "workspace = 'ws' AND biz_id IN ('a','b')",
            scope.list_sql_in(None, &["a", "b"])
        );
        assert_eq!(
            "n.workspace = 'ws' AND n.biz_id IN ('a','o''ne')",
            scope.list_sql_in(Some("n"), &["a", "o'ne"])
        );
        // Single-element set renders as a one-item IN, not the `=` form —
        // callers pick the multi-biz path only when they asked for it.
        assert_eq!(
            "workspace = 'ws' AND biz_id IN ('only')",
            scope.list_sql_in(None, &["only"])
        );
    }
}

#[cfg(all(test, feature = "neo4j"))]
mod cypher_tests {
    use super::*;
    use easyrag_core::model::TenantScope;

    #[test]
    fn exact_biz_line_renders_the_property_map() {
        let scope = TenantScope::new("ws", "b");
        assert_eq!("`ws`", scope.ws_label());
        assert_eq!("`ws` {biz_id: $biz}", scope.node_label());
        assert_eq!(Some("$biz"), scope.biz_prop_field());
        assert_eq!("{entity_id: $label, biz_id: $biz}", scope.entity_map("$label"));
        assert_eq!("{entity_id: srcId, biz_id: $biz}", scope.entity_map("srcId"));
        assert_eq!(Some("n.biz_id = $biz".to_string()), scope.biz_predicate("n"));
    }

    #[test]
    fn wildcard_drops_the_biz_predicate_entirely() {
        let scope = TenantScope::new("ws", "");
        assert_eq!("`ws`", scope.node_label());
        assert_eq!(None, scope.biz_prop_field());
        assert_eq!("{entity_id: $label}", scope.entity_map("$label"));
        assert_eq!(None, scope.biz_predicate("node"));
    }
}
