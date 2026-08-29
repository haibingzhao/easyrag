//! Shared directory-scan tests: extension filtering, max_files cap and
//! incremental path+content_hash deduplication (in-memory backends).

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::model::doc_status::DocStatus;
use easyrag_core::storage::traits::{DocStatusStorage, GraphStorage, KVStorage, VectorStorage};
use easyrag_pipeline::{scan_directory, IngestionPipeline};
use easyrag_storage::memory::{
    InMemoryDocStatusStorage, InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage,
};
use futures::stream::BoxStream;
use std::sync::Arc;

const BIZ: &str = "default";

struct ZeroFunc;

#[async_trait]
impl EmbeddingFunction for ZeroFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0f32; 4]).collect())
    }
}

fn test_embedding_func() -> EmbeddingFunc {
    EmbeddingFunc::new(4, Arc::new(ZeroFunc))
}

struct MockLlm;

#[async_trait]
impl LlmService for MockLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Ok("MOCK".to_string())
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

struct Env {
    pipeline: Arc<IngestionPipeline>,
    doc_status: Arc<InMemoryDocStatusStorage>,
}

fn build_env() -> Env {
    let llm: Arc<dyn LlmService> = Arc::new(MockLlm);
    let embedding = Arc::new(test_embedding_func());
    let vector = Arc::new(InMemoryVectorStorage::new(
        "vdb",
        "dir-scan",
        test_embedding_func(),
    ));
    let graph = Arc::new(InMemoryGraphStorage::new("graph", "dir-scan"));
    let kv = Arc::new(InMemoryKVStorage::new("kv", "dir-scan"));
    let doc_status = Arc::new(InMemoryDocStatusStorage::new("status", "dir-scan"));
    let pipeline = Arc::new(IngestionPipeline::new(
        llm,
        kv as Arc<dyn KVStorage>,
        vector as Arc<dyn VectorStorage>,
        graph as Arc<dyn GraphStorage>,
        doc_status.clone() as Arc<dyn DocStatusStorage>,
        embedding,
    ));
    Env {
        pipeline,
        doc_status,
    }
}

fn extensions(list: &[&str]) -> Vec<String> {
    list.iter().map(|e| e.to_string()).collect()
}

#[tokio::test]
async fn scan_filters_by_extension_and_caps_files() {
    let env = build_env();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    std::fs::write(dir.path().join("b.md"), "beta").unwrap();
    std::fs::write(dir.path().join("c.bin"), "gamma").unwrap();

    // .bin excluded by the extension filter.
    let outcome = scan_directory(
        &env.pipeline,
        &(env.doc_status.clone() as Arc<dyn DocStatusStorage>),
        dir.path(),
        &extensions(&[".txt", ".md"]),
        100,
        BIZ,
    )
    .await
    .unwrap();
    assert_eq!(outcome.files_seen, 2);
    assert_eq!(outcome.inserted_doc_ids.len(), 2);

    // max_files caps the collected file list.
    let outcome = scan_directory(
        &env.pipeline,
        &(env.doc_status.clone() as Arc<dyn DocStatusStorage>),
        dir.path(),
        &extensions(&[".txt", ".md"]),
        1,
        BIZ,
    )
    .await
    .unwrap();
    assert_eq!(outcome.files_seen, 1);
}

#[tokio::test]
async fn scan_dedups_unchanged_files_and_reinserts_changed_content() {
    let env = build_env();
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "first version").unwrap();
    let doc_status: Arc<dyn DocStatusStorage> = env.doc_status.clone();

    // First pass: new file inserted as pending.
    let outcome = scan_directory(
        &env.pipeline,
        &doc_status,
        dir.path(),
        &extensions(&[".txt"]),
        100,
        BIZ,
    )
    .await
    .unwrap();
    assert_eq!(outcome.inserted_doc_ids.len(), 1);
    assert_eq!(outcome.skipped_duplicate, 0);
    assert_eq!(
        env.doc_status
            .get_docs_by_status(DocStatus::Pending, BIZ)
            .await
            .unwrap()
            .len(),
        1
    );

    // Second pass: identical path + content hash -> skipped.
    let outcome = scan_directory(
        &env.pipeline,
        &doc_status,
        dir.path(),
        &extensions(&[".txt"]),
        100,
        BIZ,
    )
    .await
    .unwrap();
    assert_eq!(outcome.inserted_doc_ids.len(), 0);
    assert_eq!(outcome.skipped_duplicate, 1);

    // Changed content: re-inserted as a new document version.
    std::fs::write(dir.path().join("a.txt"), "second version").unwrap();
    let outcome = scan_directory(
        &env.pipeline,
        &doc_status,
        dir.path(),
        &extensions(&[".txt"]),
        100,
        BIZ,
    )
    .await
    .unwrap();
    assert_eq!(outcome.inserted_doc_ids.len(), 1);
    assert_eq!(outcome.skipped_duplicate, 0);
}
