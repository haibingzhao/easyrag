//! Storage backends for EasyRAG.
//!
//! - `memory`: in-memory implementations (dev/test, default backend)
//! - `json`: JSON file-backed KV storage
//! - `postgres`: PostgreSQL + pgvector (feature `postgres`)
//! - `neo4j`: Neo4j graph storage (feature `neo4j`)
//! - `tantivy_search`: embedded tantivy keyword index (feature `tantivy-backend`)

mod drop_result;

#[cfg(feature = "postgres")]
mod pg_upsert;

#[cfg(any(feature = "postgres", feature = "neo4j"))]
mod scope_sql;

#[cfg(feature = "json")]
pub mod json;
pub mod memory;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "neo4j")]
pub mod neo4j;

#[cfg(feature = "tantivy-backend")]
pub mod tantivy_search;
