//! easyrag-app library surface: the background auto scanner is exposed as a
//! library module so integration tests (and future tooling) can drive scan
//! cycles without booting the full binary.

pub mod scanner;
