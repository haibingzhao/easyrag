//! User credential storage.
//!
//! Users are server-level entities without workspace semantics, so
//! `UserStorage` deliberately does not extend `StorageNameSpace`.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A stored user credential record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserRecord {
    pub username: String,
    /// BCrypt hash of the password.
    pub password_hash: String,
    pub role: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// User credential storage interface. Global scope (no workspace column):
/// every workspace shares the same user base.
#[async_trait]
pub trait UserStorage: Send + Sync {
    /// Find a user by exact username.
    async fn find_by_username(&self, username: &str) -> crate::Result<Option<UserRecord>>;

    /// Insert or update a user record (keyed by username).
    async fn upsert(&self, user: &UserRecord) -> crate::Result<()>;

    /// Number of stored users.
    async fn count(&self) -> crate::Result<u64>;

    /// Initialize the storage (e.g. create tables).
    async fn initialize(&self) -> crate::Result<()> {
        Ok(())
    }
}
