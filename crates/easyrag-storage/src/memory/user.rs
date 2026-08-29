//! In-memory user storage.
//! For development and testing without external dependencies.

use async_trait::async_trait;
use easyrag_core::storage::user::{UserRecord, UserStorage};
use parking_lot::Mutex;
use std::collections::HashMap;

pub struct InMemoryUserStorage {
    users: Mutex<HashMap<String, UserRecord>>,
}

impl InMemoryUserStorage {
    pub fn new() -> Self {
        Self {
            users: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryUserStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl UserStorage for InMemoryUserStorage {
    async fn find_by_username(&self, username: &str) -> easyrag_core::Result<Option<UserRecord>> {
        Ok(self.users.lock().get(username).cloned())
    }

    async fn upsert(&self, user: &UserRecord) -> easyrag_core::Result<()> {
        self.users.lock().insert(user.username.clone(), user.clone());
        Ok(())
    }

    async fn count(&self) -> easyrag_core::Result<u64> {
        Ok(self.users.lock().len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn record(username: &str, hash: &str) -> UserRecord {
        let now = Utc::now();
        UserRecord {
            username: username.to_string(),
            password_hash: hash.to_string(),
            role: "admin".to_string(),
            created_at: now,
            updated_at: now,
        }
    }

    #[tokio::test]
    async fn upsert_find_and_count() {
        let storage = InMemoryUserStorage::new();
        assert_eq!(storage.count().await.unwrap(), 0);
        assert!(storage.find_by_username("admin").await.unwrap().is_none());

        storage.upsert(&record("admin", "$2a$hash")).await.unwrap();
        assert_eq!(storage.count().await.unwrap(), 1);
        let found = storage.find_by_username("admin").await.unwrap().unwrap();
        assert_eq!(found.password_hash, "$2a$hash");

        // Upsert overwrites by username.
        storage.upsert(&record("admin", "$2a$new")).await.unwrap();
        assert_eq!(storage.count().await.unwrap(), 1);
        let found = storage.find_by_username("admin").await.unwrap().unwrap();
        assert_eq!(found.password_hash, "$2a$new");
    }
}
