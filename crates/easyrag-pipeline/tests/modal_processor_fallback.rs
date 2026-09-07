//! Characterization guardrail for the four modal processors' fallback paths.
//!
//! The success path of `ModalProcessor::{process, generate_description}` is
//! covered by `multimodal_pipeline.rs`, but the *fallback* branches (LLM
//! failure -> synthesized `ModalProcessResult`, and modal-type mismatch ->
//! `InvalidArgument` / "unknown_*" entity) had no coverage. These branches
//! produce byte-level outputs that are persisted to storage (entity_name via
//! `compute_mdhash_id`, description/summary strings, chunk_content), so the
//! Phase 1.1 "quadruplets -> trait default methods" refactor must not change
//! them.
//!
//! Each test snapshots the exact current output. Regenerate deliberately with:
//! `INSTA_UPDATE=always cargo test -p easyrag-pipeline --test modal_processor_fallback`
//!
//! Snapshots land in `<workspace>/testdata/golden/modal/`.

use std::sync::Arc;

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingContext, EmbeddingFunc, EmbeddingFunction};
use easyrag_core::llm::service::{LlmRequest, LlmService};
use easyrag_core::{CoreError, JsonMap};
use easyrag_pipeline::multimodal::context::{ContextConfig, ContextExtractor};
use easyrag_pipeline::multimodal::model::ContentItem;
use easyrag_pipeline::multimodal::processor::{
    BaseModalProcessor, EquationModalProcessor, GenericModalProcessor, ImageModalProcessor,
    ModalProcessor, TableModalProcessor,
};
use easyrag_storage::memory::{InMemoryGraphStorage, InMemoryKVStorage, InMemoryVectorStorage};
use futures::stream::BoxStream;

const BIZ: &str = "default";

// ---------------------------------------------------------------------
// Mocks: an LLM that always fails, forcing every processor down its
// fallback branch.
// ---------------------------------------------------------------------

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

struct FailingLlm;

#[async_trait]
impl LlmService for FailingLlm {
    async fn complete(&self, _request: &LlmRequest) -> easyrag_core::Result<String> {
        Err(CoreError::Unsupported("llm down".to_string()))
    }

    fn complete_stream(&self, _request: LlmRequest) -> BoxStream<'static, String> {
        Box::pin(futures::stream::empty())
    }
}

fn base() -> BaseModalProcessor {
    BaseModalProcessor::new(
        Arc::new(FailingLlm),
        Arc::new(InMemoryKVStorage::new("kv", "fb")),
        Arc::new(InMemoryVectorStorage::new(
            "vdb",
            "fb",
            EmbeddingFunc::new(4, Arc::new(ZeroFunc)),
        )),
        Arc::new(InMemoryGraphStorage::new("graph", "fb")),
        Arc::new(ContextExtractor::new(ContextConfig::default())),
        BIZ.to_string(),
    )
}

// ---------------------------------------------------------------------
// ContentItem fixtures (one per modality). The fallback output only reads
// the "primary" field of each variant, but we fill the rest realistically.
// ---------------------------------------------------------------------

fn image_item() -> ContentItem {
    ContentItem::Image {
        img_path: "/figures/nonexistent_figure.png".to_string(),
        captions: vec!["Figure caption".to_string()],
        footnotes: vec![],
        section_path: "Intro > Background".to_string(),
        neighbor_text: "some surrounding text".to_string(),
        content_list_index: 1,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

fn table_item() -> ContentItem {
    ContentItem::Table {
        table_body: "<table><tr><td>a</td><td>b</td></tr></table>".to_string(),
        table_img_path: "/tables/t1.png".to_string(),
        captions: vec!["Table caption".to_string()],
        footnotes: vec![],
        content_list_index: 2,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

fn equation_item() -> ContentItem {
    ContentItem::Equation {
        equation_text: "E = mc^2".to_string(),
        equation_format: "latex".to_string(),
        content_list_index: 3,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

fn generic_item() -> ContentItem {
    ContentItem::Generic {
        content: "some generic content body".to_string(),
        content_type: "chart".to_string(),
        content_list_index: 4,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

// A long primary text (> 100 chars) to exercise the `take_chars(_, 100)`
// truncation in the Table/Equation/Generic fallbacks (Image's process()
// description deliberately does NOT truncate).
fn long_table_item() -> ContentItem {
    ContentItem::Table {
        table_body: "X".repeat(250),
        table_img_path: "/tables/long.png".to_string(),
        captions: vec![],
        footnotes: vec![],
        content_list_index: 5,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

// A long, nonexistent image path (> 100 chars). Locks the asymmetry that
// Image's `process()` description embeds the FULL path (no `take_chars`)
// while Image's `generate_description()` summary truncates to 100 chars.
fn long_image_item() -> ContentItem {
    ContentItem::Image {
        img_path: format!("/figures/{}.png", "p".repeat(250)),
        captions: vec![],
        footnotes: vec![],
        section_path: String::new(),
        neighbor_text: String::new(),
        content_list_index: 6,
        page_idx: 0,
        metadata: JsonMap::new(),
    }
}

// ---------------------------------------------------------------------
// Snapshot helpers
// ---------------------------------------------------------------------

fn modal_golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-pipeline");
    root.join("testdata").join("golden").join("modal")
}

fn snap(name: &str, text: &str) {
    insta::with_settings!({
        snapshot_path => modal_golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, text);
    });
}

fn render_result(r: &easyrag_pipeline::multimodal::model::ModalProcessResult) -> String {
    let metadata: Vec<String> = r
        .metadata
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    format!(
        "entity_name: {}\nentity_type: {}\ndescription: {}\nchunk_content: {}\nsuccess: {}\nerror_message: {:?}\nmetadata: [{}]",
        r.entity_name,
        r.entity_type,
        r.description,
        r.chunk_content,
        r.success,
        r.error_message,
        metadata.join(", "),
    )
}

fn render_pair(desc: &str, entity: &JsonMap) -> String {
    let mut keys: Vec<&String> = entity.keys().collect();
    keys.sort();
    let fields: Vec<String> = keys
        .iter()
        .map(|k| format!("{k}={}", entity.get(*k).unwrap()))
        .collect();
    format!("description: {}\nentity: {{{}}}", desc, fields.join(", "))
}

fn render_err(e: &CoreError) -> String {
    format!("{e}")
}

// ---------------------------------------------------------------------
// process(): correct variant + failing LLM -> fallback ModalProcessResult
// ---------------------------------------------------------------------

#[tokio::test]
async fn process_image_fallback() {
    let p = ImageModalProcessor::new(base());
    let r = p.process(&image_item(), &JsonMap::new(), "ctx").await.unwrap();
    snap("process_image_fallback", &render_result(&r));
}

#[tokio::test]
async fn process_image_fallback_long_path_not_truncated() {
    let p = ImageModalProcessor::new(base());
    let r = p
        .process(&long_image_item(), &JsonMap::new(), "ctx")
        .await
        .unwrap();
    snap(
        "process_image_fallback_long_path_not_truncated",
        &render_result(&r),
    );
}

#[tokio::test]
async fn process_table_fallback() {
    let p = TableModalProcessor::new(base());
    let r = p.process(&table_item(), &JsonMap::new(), "ctx").await.unwrap();
    snap("process_table_fallback", &render_result(&r));
}

#[tokio::test]
async fn process_table_fallback_truncates_long_body() {
    let p = TableModalProcessor::new(base());
    let r = p
        .process(&long_table_item(), &JsonMap::new(), "ctx")
        .await
        .unwrap();
    snap(
        "process_table_fallback_truncates_long_body",
        &render_result(&r),
    );
}

#[tokio::test]
async fn process_equation_fallback() {
    let p = EquationModalProcessor::new(base());
    let r = p
        .process(&equation_item(), &JsonMap::new(), "ctx")
        .await
        .unwrap();
    snap("process_equation_fallback", &render_result(&r));
}

#[tokio::test]
async fn process_generic_fallback() {
    let p = GenericModalProcessor::new(base());
    let r = p
        .process(&generic_item(), &JsonMap::new(), "ctx")
        .await
        .unwrap();
    snap("process_generic_fallback", &render_result(&r));
}

// ---------------------------------------------------------------------
// process(): wrong variant -> InvalidArgument (message must be preserved)
// ---------------------------------------------------------------------

#[tokio::test]
async fn process_mismatch_errors() {
    let mut out = String::new();

    let img = ImageModalProcessor::new(base());
    let e = img.process(&table_item(), &JsonMap::new(), "").await.unwrap_err();
    out.push_str(&format!("image_given_table: {}\n", render_err(&e)));

    let tbl = TableModalProcessor::new(base());
    let e = tbl
        .process(&equation_item(), &JsonMap::new(), "")
        .await
        .unwrap_err();
    out.push_str(&format!("table_given_equation: {}\n", render_err(&e)));

    let eq = EquationModalProcessor::new(base());
    let e = eq.process(&generic_item(), &JsonMap::new(), "").await.unwrap_err();
    out.push_str(&format!("equation_given_generic: {}\n", render_err(&e)));

    let gen = GenericModalProcessor::new(base());
    let e = gen.process(&image_item(), &JsonMap::new(), "").await.unwrap_err();
    out.push_str(&format!("generic_given_image: {}\n", render_err(&e)));

    snap("process_mismatch_errors", &out);
}

// ---------------------------------------------------------------------
// generate_description(): correct variant + failing LLM -> fallback pair
// ---------------------------------------------------------------------

#[tokio::test]
async fn generate_description_image_fallback() {
    let p = ImageModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&image_item(), "ctx").await;
    snap(
        "generate_description_image_fallback",
        &render_pair(&desc, &entity),
    );
}

#[tokio::test]
async fn generate_description_image_fallback_long_path_truncated() {
    let p = ImageModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&long_image_item(), "ctx").await;
    snap(
        "generate_description_image_fallback_long_path_truncated",
        &render_pair(&desc, &entity),
    );
}

#[tokio::test]
async fn generate_description_table_fallback() {
    let p = TableModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&table_item(), "ctx").await;
    snap(
        "generate_description_table_fallback",
        &render_pair(&desc, &entity),
    );
}

#[tokio::test]
async fn generate_description_table_fallback_truncates_long_body() {
    let p = TableModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&long_table_item(), "ctx").await;
    snap(
        "generate_description_table_fallback_truncates_long_body",
        &render_pair(&desc, &entity),
    );
}

#[tokio::test]
async fn generate_description_equation_fallback() {
    let p = EquationModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&equation_item(), "ctx").await;
    snap(
        "generate_description_equation_fallback",
        &render_pair(&desc, &entity),
    );
}

#[tokio::test]
async fn generate_description_generic_fallback() {
    let p = GenericModalProcessor::new(base());
    let (desc, entity) = p.generate_description(&generic_item(), "ctx").await;
    snap(
        "generate_description_generic_fallback",
        &render_pair(&desc, &entity),
    );
}

// ---------------------------------------------------------------------
// generate_description(): wrong variant -> "unknown_*" entity map
// ---------------------------------------------------------------------

#[tokio::test]
async fn generate_description_mismatch_unknown_entity() {
    let mut out = String::new();

    let img = ImageModalProcessor::new(base());
    let (desc, entity) = img.generate_description(&table_item(), "").await;
    out.push_str(&format!(
        "image_given_table: {}\n",
        render_pair(&desc, &entity)
    ));

    let tbl = TableModalProcessor::new(base());
    let (desc, entity) = tbl.generate_description(&equation_item(), "").await;
    out.push_str(&format!(
        "table_given_equation: {}\n",
        render_pair(&desc, &entity)
    ));

    let eq = EquationModalProcessor::new(base());
    let (desc, entity) = eq.generate_description(&generic_item(), "").await;
    out.push_str(&format!(
        "equation_given_generic: {}\n",
        render_pair(&desc, &entity)
    ));

    let gen = GenericModalProcessor::new(base());
    let (desc, entity) = gen.generate_description(&image_item(), "").await;
    out.push_str(&format!(
        "generic_given_image: {}\n",
        render_pair(&desc, &entity)
    ));

    snap("generate_description_mismatch_unknown_entity", &out);
}
