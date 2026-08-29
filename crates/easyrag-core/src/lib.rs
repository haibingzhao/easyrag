//! EasyRAG core crate.
//!
//! LLM service traits, concurrency/resilience utilities, tokenizer, hashing,
//! JSON repair utilities and all prompt templates.

pub mod concurrency;
pub mod constants;
pub mod error;
pub mod llm;
pub mod model;
pub mod namespace;
pub mod prompt;
pub mod storage;
pub mod util;

pub use error::CoreError;
pub type Result<T> = std::result::Result<T, CoreError>;

/// JSON object carrier, used across
/// the storage interfaces. Values are passed through opaquely at the storage
/// layer and only strongly typed at REST/pipeline hot paths.
pub type JsonMap = serde_json::Map<String, serde_json::Value>;
pub type JsonValue = serde_json::Value;
