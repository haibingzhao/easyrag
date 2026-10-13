//! Routes module: one file per controller.

pub mod auth;
pub mod documents;
pub mod evaluation;
pub mod graph;
pub mod health;
pub mod multimodal;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod ollama;
pub mod query;
pub mod tenant_config;
