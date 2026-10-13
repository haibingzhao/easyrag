//! Byte-level golden guardrail for all five chunkers (Phase 0 discipline
//! applied to 5f).
//!
//! `tests/chunker.rs` and `tests/paragraph_semantic.rs` assert *properties*
//! (indices are sequential, tokens are positive, overlapping chunks share
//! content). None of them pins the exact chunk boundaries, token counts or
//! metadata a given input produces, so a refactor of the shared windowing /
//! assembly code could move every boundary and still pass. These snapshots
//! close that gap: each one is the complete chunk list, content escaped with
//! `{:?}` so trailing whitespace and newlines are visible, metadata rendered
//! in insertion order (`serde_json` preserves it).
//!
//! Captured against the pre-5f implementations; the 5f refactor must leave
//! every byte below unchanged.
//!
//! Three snapshots were added *after* the refactor landed, because mutation
//! testing showed the original set left three guarantees unpinned. They are
//! regression fixtures rather than equivalence evidence, and each is marked
//! with what it exists to kill:
//!
//! * `recursive_overlap_boundary` — `SlidingWindow::flush`'s overlap walk uses
//!   `tokens > overlap`, not `>=`.
//! * `semantic_flush_boundary` — the same window's flush test compares before
//!   adding, on the sentence-level caller.
//! * `paragraph_semantic_nested` — `sub_block` carries the parent block's
//!   `parent_headings` through an oversized *nested* section, and
//!   `build_chunks` counts tokens on the untrimmed `full_content`. The flat
//!   `markdown()` fixture only ever splits a top-level block, whose
//!   `parent_headings` is empty, and never holds a body-less heading, so it
//!   cannot see either.

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingFunc, EmbeddingFunction};
use easyrag_pipeline::{
    Chunker, ChunkerConfig, ChunkerRegistry, ParagraphSemanticChunker, RecursiveCharacterChunker,
    SemanticVectorChunker, StructureAwareChunker, TextChunk, TokenSizeChunker,
};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ---------------------------------------------------------------------
// Deterministic embedding mock (same construction as tests/chunker.rs)
// ---------------------------------------------------------------------

struct HashEmbeddingFunc {
    dim: usize,
}

fn text_seed(text: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[async_trait]
impl EmbeddingFunction for HashEmbeddingFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<easyrag_core::llm::embedding_func::EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let seed = text_seed(text);
                (0..self.dim)
                    .map(|i| (splitmix64(seed.wrapping_add(i as u64)) % 10_000) as f32 / 10_000.0)
                    .collect()
            })
            .collect())
    }
}

fn mock_embedding_func(dim: usize) -> Arc<EmbeddingFunc> {
    Arc::new(EmbeddingFunc::new(dim, Arc::new(HashEmbeddingFunc { dim })))
}

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// Latin prose, one sentence per index. Token counts land in the tens so the
/// small chunk sizes below produce several windows.
fn latin(n: usize) -> String {
    (0..n)
        .map(|i| format!("Sentence number {i} discusses topic {}.", i % 4))
        .collect::<Vec<_>>()
        .join(" ")
}

/// CJK prose with no spaces, so only the `。` separator applies.
fn cjk(n: usize) -> String {
    (0..n)
        .map(|i| format!("第{i}句话讲述主题{}的内容。", i % 3))
        .collect::<Vec<_>>()
        .join("")
}

/// Markdown with a preamble, one oversized section (many sentences) and two
/// tiny sibling sections — exercises oversized splitting *and* undersized
/// merging in the paragraph chunker, and sub-chunking in the structure one.
fn markdown() -> String {
    format!(
        "Introductory preamble text with no heading above it.\n\n\
         # Chapter One\n{}\n\n\
         ## Sub A\nTiny body.\n\n\
         ## Sub B\nAlso tiny.\n\n\
         # Chapter Two\n{}\n",
        latin(6),
        latin(3)
    )
}

/// Six identical newline-separated lines. Every piece has the same token count,
/// so a budget expressed as a multiple of that count lands exactly on the
/// boundaries where `>` and `>=` disagree — which prose fixtures hit only by
/// accident.
fn uniform_lines() -> (String, usize) {
    let line = "alpha beta gamma delta epsilon zeta eta theta";
    let text = (0..6).map(|_| line).collect::<Vec<_>>().join("\n");
    (text, probe_tokens(line))
}

/// A *nested* section whose body is oversized, so `split_oversized_blocks`
/// produces sub-blocks that carry a non-empty `parent_headings`, plus a
/// body-less heading — the one shape where `build_chunks`' stored content
/// (trimmed) and the string it counts tokens on (untrimmed) differ, by the
/// trailing newline.
///
/// The body-less heading has to sit at level 1. At level 2
/// `merge_undersized_blocks` absorbs it into the preceding block and drops it
/// (its content is empty, so the heading text is never appended), and the
/// difference disappears before `build_chunks` sees it.
fn nested_markdown() -> String {
    format!(
        "# Top Level\nShort intro under top.\n\n# Empty Top\n\n## Nested Section\n{}\n",
        latin(8)
    )
}

// ---------------------------------------------------------------------
// Snapshot helpers
// ---------------------------------------------------------------------

fn golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-pipeline");
    root.join("testdata").join("golden").join("chunker")
}

fn snap(name: &str, text: &str) {
    insta::with_settings!({
        snapshot_path => golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, text);
    });
}

/// The complete observable output of a `chunk()` call. Content goes through
/// `{:?}` so a single added or dropped space is a snapshot diff; metadata is
/// rendered in insertion order because `serde_json` preserves it.
fn render(chunks: &[TextChunk]) -> String {
    let mut out = format!("chunks={}\n", chunks.len());
    for c in chunks {
        out.push_str(&format!(
            "order={} tokens={} doc_id={:?} content={:?}\n",
            c.chunk_order_index, c.tokens, c.full_doc_id, c.content
        ));
        if !c.metadata.is_empty() {
            let md: Vec<String> = c
                .metadata
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect();
            out.push_str(&format!("  metadata: {}\n", md.join(" ")));
        }
    }
    out
}

async fn snap_chunks(name: &str, chunker: &dyn Chunker, text: &str) {
    let chunks = chunker
        .chunk(text, "doc-1")
        .await
        .expect("chunk must succeed");
    snap(name, &render(&chunks));
}

// ---------------------------------------------------------------------
// TokenSizeChunker
// ---------------------------------------------------------------------

#[tokio::test]
async fn token_size_sliding_window() {
    snap_chunks("token_size_sliding_window", &TokenSizeChunker::new(24, 8), &latin(6)).await;
}

/// `tokens.len() == chunk_token_size` exactly: the single-chunk shortcut is
/// `<=`, so this must produce one chunk, not a window walk.
#[tokio::test]
async fn token_size_exact_boundary_is_single_chunk() {
    let text = latin(6);
    let n = probe_tokens(&text);
    let chunker = TokenSizeChunker::new(n, 0);
    snap_chunks("token_size_exact_boundary", &chunker, &text).await;
}

/// `overlap >= chunk size` — the forward-progress guard
/// (`end.saturating_sub(overlap).max(start + 1)`) is what stops this looping
/// forever. Any window rewrite must keep terminating here.
#[tokio::test]
async fn token_size_overlap_larger_than_window_terminates() {
    snap_chunks(
        "token_size_overlap_gt_size",
        &TokenSizeChunker::new(6, 40),
        &latin(6),
    )
    .await;
}

/// Chunk size 0 with overlap 0: the other degenerate the guard covers.
///
/// The property under test is *termination* — a zero-width window must not spin
/// forever. Its output is no chunks at all: every window is empty, and an empty
/// window carries no text, so it is dropped rather than indexed as one of the
/// 27 empty-content chunks this used to produce.
#[tokio::test]
async fn token_size_zero_chunk_size_terminates() {
    snap_chunks(
        "token_size_zero_size",
        &TokenSizeChunker::new(0, 0),
        &latin(3),
    )
    .await;
}

#[tokio::test]
async fn token_size_empty_and_blank() {
    snap_chunks("token_size_empty", &TokenSizeChunker::new(24, 8), "").await;
    snap_chunks(
        "token_size_blank",
        &TokenSizeChunker::new(24, 8),
        "   \n\t ",
    )
    .await;
}

// ---------------------------------------------------------------------
// RecursiveCharacterChunker
// ---------------------------------------------------------------------

#[tokio::test]
async fn recursive_latin_paragraphs() {
    snap_chunks(
        "recursive_latin",
        &RecursiveCharacterChunker::new(24, 6),
        &format!("{}\n\n{}", latin(4), latin(4)),
    )
    .await;
}

/// CJK exercises the `。` separator and the char-level fallback.
#[tokio::test]
async fn recursive_cjk_separators() {
    snap_chunks(
        "recursive_cjk",
        &RecursiveCharacterChunker::new(16, 4),
        &cjk(6),
    )
    .await;
}

/// `merge_splits` overlap retention: the `while current_len > chunk_overlap`
/// drop-from-front loop. Overlap 0 and a large overlap give the two ends.
#[tokio::test]
async fn recursive_overlap_zero_and_large() {
    let text = format!("{}\n{}", latin(3), latin(3));
    snap_chunks(
        "recursive_overlap_zero",
        &RecursiveCharacterChunker::new(16, 0),
        &text,
    )
    .await;
    snap_chunks(
        "recursive_overlap_large",
        &RecursiveCharacterChunker::new(16, 14),
        &text,
    )
    .await;
}

/// Added post-refactor: kills `>` → `>=` in the shared window's overlap walk.
///
/// Budget `3t` with overlap `2t` over identical `t`-token pieces: the flush
/// drains from the front while the running total exceeds `2t`, so it must stop
/// holding exactly two pieces. Under `>=` it would stop holding one and every
/// later chunk would shift.
#[tokio::test]
async fn recursive_overlap_walk_boundary_is_strict() {
    let (text, t) = uniform_lines();
    snap_chunks(
        "recursive_overlap_boundary",
        &RecursiveCharacterChunker::new(3 * t, 2 * t).with_separators(vec!["\n".to_string()]),
        &text,
    )
    .await;
}

#[tokio::test]
async fn recursive_custom_separators() {
    snap_chunks(
        "recursive_custom_separators",
        &RecursiveCharacterChunker::new(20, 5).with_separators(vec!["\n".to_string()]),
        &format!("{}\n{}\n{}", latin(2), latin(2), latin(2)),
    )
    .await;
}

#[tokio::test]
async fn recursive_empty() {
    snap_chunks("recursive_empty", &RecursiveCharacterChunker::new(24, 6), "").await;
}

// ---------------------------------------------------------------------
// SemanticVectorChunker
// ---------------------------------------------------------------------

#[tokio::test]
async fn semantic_vector_default_regex() {
    let c = SemanticVectorChunker::new(mock_embedding_func(8), 30, 0.5, None).unwrap();
    snap_chunks("semantic_vector_default", &c, &latin(8)).await;
}

#[tokio::test]
async fn semantic_vector_custom_regex() {
    let c =
        SemanticVectorChunker::new(mock_embedding_func(8), 30, 0.5, Some("\n".to_string())).unwrap();
    let text = (0..5)
        .map(|i| format!("Line {i} of the document about topic {}.", i % 3))
        .collect::<Vec<_>>()
        .join("\n");
    snap_chunks("semantic_vector_custom_regex", &c, &text).await;
}

/// Added post-refactor: kills `>` → `>=` in the shared window's flush test on
/// the *sentence-level* caller, which the prose fixtures above only reach by
/// accident.
///
/// Splitting on newlines gives six byte-identical sentences, so their embeddings
/// are identical too and every adjacent distance is 0 — no semantic breakpoint
/// can fire, leaving the token budget as the only thing that can flush. With the
/// budget at exactly `3t` the third sentence fits (`3t` is not `> 3t`) and the
/// fourth trips it, yielding two chunks of three.
#[tokio::test]
async fn semantic_vector_flush_boundary_is_strict() {
    let (text, t) = uniform_lines();
    let c = SemanticVectorChunker::new(mock_embedding_func(8), 3 * t, 0.5, Some("\n".to_string()))
        .unwrap();
    snap_chunks("semantic_flush_boundary", &c, &text).await;
}

/// A single sentence short-circuits before any embedding call and returns the
/// raw text, not the joined-sentence form.
#[tokio::test]
async fn semantic_vector_single_sentence_shortcut() {
    let c = SemanticVectorChunker::new(mock_embedding_func(8), 30, 0.5, None).unwrap();
    snap_chunks("semantic_single_sentence", &c, "Just one sentence here.").await;
}

#[tokio::test]
async fn semantic_vector_empty() {
    let c = SemanticVectorChunker::new(mock_embedding_func(8), 30, 0.5, None).unwrap();
    snap_chunks("semantic_empty", &c, "").await;
}

// ---------------------------------------------------------------------
// ParagraphSemanticChunker
// ---------------------------------------------------------------------

/// Oversized `# Chapter One` gets sentence-split; the two tiny `## Sub`
/// blocks get merged. `build_chunks` prepends the heading and counts tokens on
/// the *untrimmed* string while storing the trimmed one — pinned here.
#[tokio::test]
async fn paragraph_semantic_headings() {
    let c = ParagraphSemanticChunker::new(mock_embedding_func(8), 40, 0.5);
    snap_chunks("paragraph_semantic_headings", &c, &markdown()).await;
}

/// Added post-refactor: kills two things the flat `markdown()` fixture cannot
/// see.
///
/// 1. A `sub_block` that drops or mis-copies the parent block's heading
///    context. `markdown()` splits only `# Chapter One`, a top-level block whose
///    `parent_headings` is empty, so copying `Vec::new()` instead of the
///    parent's list is invisible there. Here the oversized block is
///    `## Nested Section`, which sits under `# Empty Top`, and every piece it
///    splits into must show `parent_headings=["Empty Top"]` in its metadata.
/// 2. Counting tokens on the *untrimmed* `full_content` in `build_chunks`. For
///    every block with a body, `full_content` is already trimmed and the two
///    agree; `# Empty Top` is the exception, storing `"# Empty Top"` while
///    counting `"# Empty Top\n"`.
#[tokio::test]
async fn paragraph_semantic_nested_oversized_keeps_parent_headings() {
    let c = ParagraphSemanticChunker::new(mock_embedding_func(8), 40, 0.5);
    snap_chunks("paragraph_semantic_nested", &c, &nested_markdown()).await;
}

/// No headings at all falls back to `SemanticVectorChunker`, so the output
/// must match the semantic snapshot for the same text.
#[tokio::test]
async fn paragraph_semantic_falls_back_without_headings() {
    let c = ParagraphSemanticChunker::new(mock_embedding_func(8), 40, 0.5);
    snap_chunks("paragraph_semantic_fallback", &c, &latin(6)).await;
}

#[tokio::test]
async fn paragraph_semantic_empty() {
    let c = ParagraphSemanticChunker::new(mock_embedding_func(8), 40, 0.5);
    snap_chunks("paragraph_semantic_empty", &c, "").await;
}

// ---------------------------------------------------------------------
// StructureAwareChunker
// ---------------------------------------------------------------------

/// Sections at or under `max_chunk_tokens` become one chunk each with
/// `section_path`/`section_level`/`heading` metadata; the oversized chapter is
/// re-chunked by `TokenSizeChunker` and every piece inherits that metadata.
#[tokio::test]
async fn structure_aware_headings() {
    snap_chunks("structure_aware_headings", &StructureAwareChunker::new(25), &markdown()).await;
}

#[tokio::test]
async fn structure_aware_falls_back_without_headings() {
    snap_chunks(
        "structure_aware_fallback",
        &StructureAwareChunker::new(25),
        &latin(6),
    )
    .await;
}

#[tokio::test]
async fn structure_aware_empty() {
    snap_chunks("structure_aware_empty", &StructureAwareChunker::new(25), "").await;
}

// ---------------------------------------------------------------------
// ChunkerRegistry
// ---------------------------------------------------------------------

/// Every type name and alias maps to the same `chunker_type()`, and the two
/// embedding-requiring types keep their exact error text.
#[test]
fn registry_type_table_and_errors() {
    let mut out = String::new();
    for name in [
        "token_size",
        "default",
        "recursive_character",
        "semantic_vector",
        "paragraph_semantic",
        "p",
        "structure_aware",
        "s",
        "TOKEN_SIZE",
        "no_such_chunker",
        "",
    ] {
        let got = ChunkerRegistry::create(name, ChunkerConfig::default());
        match got {
            Ok(c) => out.push_str(&format!("{name:?} -> ok({})\n", c.chunker_type())),
            Err(e) => out.push_str(&format!("{name:?} -> err({e})\n")),
        }
    }

    // With an embedding_func present, the two semantic types succeed.
    let with_embedding = ChunkerConfig {
        embedding_func: Some(mock_embedding_func(8)),
        ..ChunkerConfig::default()
    };
    for name in ["semantic_vector", "paragraph_semantic", "p"] {
        let got = ChunkerRegistry::create(name, with_embedding.clone()).unwrap();
        out.push_str(&format!("{name:?} +embedding -> ok({})\n", got.chunker_type()));
    }
    snap("registry_type_table", &out);
}

/// Config values reach the chunker they name: `chunk_token_size` feeds
/// token_size/semantic/paragraph/structure, `chunk_size`+`chunk_overlap` feed
/// recursive_character only. Observed through the resulting boundaries.
#[tokio::test]
async fn registry_config_values_take_effect() {
    let config = ChunkerConfig {
        chunk_token_size: Some(12),
        chunk_overlap_token_size: Some(3),
        chunk_size: Some(14),
        chunk_overlap: Some(4),
        similarity_threshold: Some(0.9),
        embedding_func: Some(mock_embedding_func(8)),
        ..ChunkerConfig::default()
    };
    let text = latin(8);
    let mut out = String::new();
    for name in ["token_size", "recursive_character", "semantic_vector"] {
        let c = ChunkerRegistry::create(name, config.clone()).unwrap();
        let chunks = c.chunk(&text, "doc-1").await.unwrap();
        out.push_str(&format!("--- {name} ---\n{}", render(&chunks)));
    }
    snap("registry_config_values", &out);
}

// ---------------------------------------------------------------------
// Fixture helpers needing the tokenizer
// ---------------------------------------------------------------------

/// Token count of `text` under the same tokenizer the chunkers use, so the
/// exact-boundary fixture can be expressed without hardcoding a number that
/// would silently drift if the fixture text changed.
fn probe_tokens(text: &str) -> usize {
    easyrag_core::util::tokenizer::TokenizerProvider::get_tokenizer().count_tokens(text)
}
