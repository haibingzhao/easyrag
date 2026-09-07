//! easyrag-app library surface: assembly modules (config/embedding/factory/
//! llm/storage/workspace) and the background auto scanner are exposed as
//! library modules so the binary and integration tests share one compilation
//! unit. Sharing the units makes assembly drift between the binary and tests
//! discoverable by the compiler; `factory` is the Phase 6.5 assembly matrix
//! that both `main` and the per-workspace resolver build through.

pub mod config;
pub mod embedding;
pub mod factory;
pub mod llm;
pub mod scanner;
pub mod storage;
pub mod workspace;
