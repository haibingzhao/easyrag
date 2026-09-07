//! Tenant scope: the `(workspace, biz_id)` pair every stored record is tagged with.
//!
//! This is the **data body only**. Rendering a scope into SQL or Cypher is a
//! storage-implementation detail and deliberately lives in `easyrag-storage`
//! (`scope_sql`), so `core` stays free of query-language knowledge and the
//! server never has to depend on storage internals to talk about a tenant.
//!
//! Named `TenantScope` (not `Scope`) because the HTTP layer has its own
//! request-scoped `Scope` extractor with a different job.

/// Owned tenant scope.
///
/// Use this where a scope travels with a request or is stored on a struct
/// (e.g. a per-call CRUD helper); use [`TenantScopeRef`] where the workspace
/// and biz line are already borrowed from somewhere else and cloning them into
/// every query would be pure overhead.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TenantScope {
    /// Tenant workspace name (already validated before it reaches storage).
    pub workspace: String,
    /// Business line within the workspace. Empty means "every biz line" on
    /// read paths and "the default line" on write paths.
    pub biz_id: String,
}

impl TenantScope {
    pub fn new(workspace: impl Into<String>, biz_id: impl Into<String>) -> Self {
        Self {
            workspace: workspace.into(),
            biz_id: biz_id.into(),
        }
    }

    /// Same workspace, different business line.
    pub fn with_biz(&self, biz_id: impl Into<String>) -> Self {
        Self {
            workspace: self.workspace.clone(),
            biz_id: biz_id.into(),
        }
    }

    /// Borrowed view of this scope (no allocation).
    pub fn as_ref(&self) -> TenantScopeRef<'_> {
        TenantScopeRef {
            workspace: &self.workspace,
            biz_id: &self.biz_id,
        }
    }
}

/// Borrowed tenant scope: the `&str` counterpart of [`TenantScope`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TenantScopeRef<'a> {
    pub workspace: &'a str,
    pub biz_id: &'a str,
}

impl<'a> TenantScopeRef<'a> {
    pub fn new(workspace: &'a str, biz_id: &'a str) -> Self {
        Self { workspace, biz_id }
    }
}

impl From<TenantScopeRef<'_>> for TenantScope {
    fn from(scope: TenantScopeRef<'_>) -> Self {
        TenantScope {
            workspace: scope.workspace.to_string(),
            biz_id: scope.biz_id.to_string(),
        }
    }
}

/// Accessor shared by the owned and borrowed forms, so a single renderer
/// definition serves both without `dyn` or extra allocation.
pub trait AsTenantScope {
    fn workspace(&self) -> &str;
    fn biz_id(&self) -> &str;

    /// Empty `biz_id` = read-path wildcard over every business line.
    fn is_wildcard(&self) -> bool {
        self.biz_id().is_empty()
    }

    /// Owned copy of this scope.
    fn to_scope(&self) -> TenantScope {
        TenantScope {
            workspace: self.workspace().to_string(),
            biz_id: self.biz_id().to_string(),
        }
    }
}

impl AsTenantScope for TenantScope {
    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn biz_id(&self) -> &str {
        &self.biz_id
    }
}

impl AsTenantScope for TenantScopeRef<'_> {
    fn workspace(&self) -> &str {
        self.workspace
    }

    fn biz_id(&self) -> &str {
        self.biz_id
    }
}

impl AsTenantScope for &TenantScope {
    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn biz_id(&self) -> &str {
        &self.biz_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_biz_id_is_the_wildcard() {
        assert!(TenantScope::new("ws", "").is_wildcard());
        assert!(!TenantScope::new("ws", "default").is_wildcard());
        assert!(TenantScopeRef::new("ws", "").is_wildcard());
        assert!(!TenantScopeRef::new("ws", "default").is_wildcard());
    }

    #[test]
    fn with_biz_keeps_the_workspace() {
        let scope = TenantScope::new("ws", "a");
        assert_eq!(TenantScope::new("ws", "b"), scope.with_biz("b"));
    }

    #[test]
    fn borrowed_and_owned_forms_agree() {
        let owned = TenantScope::new("ws", "biz");
        let borrowed = owned.as_ref();
        assert_eq!(owned.workspace, borrowed.workspace());
        assert_eq!(owned.biz_id, borrowed.biz_id());
        assert_eq!(owned, TenantScope::from(borrowed));
        assert_eq!(owned, borrowed.to_scope());
    }

    #[test]
    fn accessor_trait_serves_both_forms() {
        fn render<S: AsTenantScope>(scope: &S) -> String {
            format!("{}/{}", scope.workspace(), scope.biz_id())
        }
        assert_eq!("ws/biz", render(&TenantScope::new("ws", "biz")));
        assert_eq!("ws/biz", render(&TenantScopeRef::new("ws", "biz")));
    }
}
