//! Chunker tests:
//! TokenSizeChunker, RecursiveCharacterChunker, SemanticVectorChunker,
//! ChunkerRegistry, StructureAwareChunker.

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingFunc, EmbeddingFunction};
use easyrag_pipeline::{
    Chunker, ChunkerConfig, ChunkerRegistry, RecursiveCharacterChunker, SemanticVectorChunker,
    StructureAwareChunker, TokenSizeChunker,
};
use serde_json::json;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Deterministic pseudo-embedding based on a stable text hash
///)` mock).
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
// TokenSizeChunker (7 cases)
// ---------------------------------------------------------------------

#[tokio::test]
async fn token_size_empty_text_returns_empty_list() {
    let chunker = TokenSizeChunker::new(100, 10);
    assert!(chunker.chunk("", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn token_size_blank_text_returns_empty_list() {
    let chunker = TokenSizeChunker::new(100, 10);
    assert!(chunker.chunk("   \n\t  ", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn token_size_short_text_returns_single_chunk() {
    let chunker = TokenSizeChunker::new(1000, 100);
    let text = "Hello, world!";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert_eq!(1, result.len());
    assert_eq!(text, result[0].content);
    assert_eq!(0, result[0].chunk_order_index);
    assert_eq!("doc1", result[0].full_doc_id);
}

#[tokio::test]
async fn token_size_long_text_splits_into_multiple_chunks() {
    let chunker = TokenSizeChunker::new(10, 2);
    let text = "The quick brown fox jumps over the lazy dog. \
                Pack my box with five dozen liquor jugs. \
                How vexingly quick daft zebras jump!";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(result.len() > 1);
    for (index, chunk) in result.iter().enumerate() {
        assert_eq!(index as u32, chunk.chunk_order_index);
    }
}

#[tokio::test]
async fn token_size_overlap_chunks_share_content() {
    let chunker = TokenSizeChunker::new(20, 5);
    let text = "word ".repeat(100);
    let result = chunker.chunk(&text, "doc1").await.unwrap();
    assert!(result.len() > 1);
    for chunk in &result {
        assert!(chunk.tokens > 0);
    }
}

#[tokio::test]
async fn token_size_all_chunks_have_doc_id() {
    let chunker = TokenSizeChunker::new(10, 2);
    let text = "Some longer text that will definitely be split into multiple chunks for testing.";
    let result = chunker.chunk(text, "myDoc").await.unwrap();
    for chunk in &result {
        assert_eq!("myDoc", chunk.full_doc_id);
    }
}

#[tokio::test]
async fn token_size_token_count_positive() {
    let chunker = TokenSizeChunker::new(50, 10);
    let result = chunker
        .chunk("Hello world, this is a test.", "doc1")
        .await
        .unwrap();
    for chunk in &result {
        assert!(chunk.tokens > 0);
    }
}

/// Regression: a token window can end — and the next one begin — inside a
/// multi-byte character, because byte-level BPE splits one CJK character across
/// tokens. Decoding such a window raised
/// `tokenizer decode failed: ... incomplete utf-8 byte sequence`, which aborted
/// the whole document: `stage1-2026-09-28-gold-crash-verdict.md` (and every
/// other document reaching this chunker with a CJK body) never finished
/// indexing. Nothing is lost by trimming the partial character — the
/// overlapping window still carries it.
///
/// The window sizes are swept because *whether* a boundary lands inside a
/// character depends on where the boundary falls, not on the text alone: with
/// this fixture the split characters start at token indices 3, 5, 12, 23, 25,
/// 32, … so a single size could pass by luck.
#[tokio::test]
async fn token_size_chunking_cjk_does_not_fail_on_a_cut_character() {
    // 慌 / 挤 / 兑 / 崩 are the characters cl100k_base splits across two
    // tokens, i.e. the ones a window boundary can land inside.
    let text = "市场恐慌挤兑风险与金价崩盘的分母冲击。".repeat(20);

    for chunk_size in 2..=40 {
        let chunker = TokenSizeChunker::new(chunk_size, 1);
        let chunks = chunker.chunk(&text, "doc-cjk").await.unwrap_or_else(|e| {
            panic!("chunk_size={chunk_size} failed the whole document: {e}")
        });

        assert!(chunks.len() > 1, "chunk_size={chunk_size} must span windows");
        for chunk in &chunks {
            assert!(
                !chunk.content.is_empty(),
                "chunk_size={chunk_size}: no window may decode to nothing"
            );
            assert!(
                !chunk.content.contains('\u{FFFD}'),
                "chunk_size={chunk_size}: partial characters are dropped, not replaced: {:?}",
                chunk.content
            );
            // Everything kept still comes from the source text.
            assert!(
                text.contains(&chunk.content),
                "chunk_size={chunk_size}: chunk text must come from the source"
            );
        }
    }
}

// ---------------------------------------------------------------------
// RecursiveCharacterChunker (7 cases)
// ---------------------------------------------------------------------

#[tokio::test]
async fn recursive_empty_text_returns_empty_list() {
    let chunker = RecursiveCharacterChunker::new(100, 10);
    assert!(chunker.chunk("", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn recursive_short_text_returns_single_chunk() {
    let chunker = RecursiveCharacterChunker::new(1000, 100);
    let text = "Short text.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert_eq!(1, result.len());
    assert_eq!(text, result[0].content);
}

#[tokio::test]
async fn recursive_paragraph_splitting_splits_on_double_newline() {
    let chunker = RecursiveCharacterChunker::new(10, 0);
    let text = "First paragraph here with extra words.\n\n\
                Second paragraph here with extra words.\n\n\
                Third paragraph here with extra words.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(result.len() >= 2, "Expected >= 2 chunks but got {}", result.len());
}

#[tokio::test]
async fn recursive_sequential_order_indices() {
    let chunker = RecursiveCharacterChunker::new(15, 0);
    let text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    for (index, chunk) in result.iter().enumerate() {
        assert_eq!(index as u32, chunk.chunk_order_index);
    }
}

#[tokio::test]
async fn recursive_chinese_separators_work() {
    let chunker = RecursiveCharacterChunker::new(10, 0).with_separators(
        ["。", "，", " ", ""]
            .iter()
            .map(|s| s.to_string())
            .collect(),
    );
    let text = "第一句话。第二句话。第三句话。第四句话。第五句话。";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(!result.is_empty());
}

#[tokio::test]
async fn recursive_all_chunks_have_positive_tokens() {
    let chunker = RecursiveCharacterChunker::new(20, 5);
    let text = "word ".repeat(200);
    let result = chunker.chunk(&text, "doc1").await.unwrap();
    for chunk in &result {
        assert!(chunk.tokens > 0);
        assert_eq!("doc1", chunk.full_doc_id);
    }
}

#[tokio::test]
async fn recursive_no_blank_chunks() {
    let chunker = RecursiveCharacterChunker::new(30, 5);
    let text = "Hello world.\n\n\n\nAnother section.\n\n\n\nThird part.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    for chunk in &result {
        assert!(!chunk.content.trim().is_empty());
    }
}

// ---------------------------------------------------------------------
// SemanticVectorChunker (6 cases)
// ---------------------------------------------------------------------

#[tokio::test]
async fn semantic_empty_text_returns_empty_list() {
    let chunker =
        SemanticVectorChunker::new(mock_embedding_func(8), 1200, 0.5, None).unwrap();
    assert!(chunker.chunk("", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn semantic_single_sentence_returns_single_chunk() {
    let chunker =
        SemanticVectorChunker::new(mock_embedding_func(8), 1200, 0.5, None).unwrap();
    let result = chunker
        .chunk("Just one sentence without any splits", "doc1")
        .await
        .unwrap();
    assert_eq!(1, result.len());
}

#[tokio::test]
async fn semantic_multiple_sentences_produces_chunks() {
    let chunker = SemanticVectorChunker::new(mock_embedding_func(8), 50, 0.5, None).unwrap();
    let text = "First sentence about cats. \
                Second sentence about dogs. \
                Third sentence about birds. \
                Fourth sentence about fish. \
                Fifth sentence about trees.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(!result.is_empty());
    for chunk in &result {
        assert!(!chunk.content.trim().is_empty());
        assert_eq!("doc1", chunk.full_doc_id);
    }
}

#[tokio::test]
async fn semantic_sequential_order_indices() {
    let chunker = SemanticVectorChunker::new(mock_embedding_func(8), 20, 0.5, None).unwrap();
    let text = "Alpha. Beta. Gamma. Delta. Epsilon. Zeta. Eta. Theta.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    for (index, chunk) in result.iter().enumerate() {
        assert_eq!(index as u32, chunk.chunk_order_index);
    }
}

#[tokio::test]
async fn semantic_respects_max_token_size() {
    // High threshold = fewer breakpoints.
    let chunker = SemanticVectorChunker::new(mock_embedding_func(8), 30, 0.9, None).unwrap();
    let text = "Word ".repeat(200);
    let result = chunker.chunk(&text, "doc1").await.unwrap();
    assert!(!result.is_empty());
}

#[tokio::test]
async fn semantic_all_chunks_have_positive_tokens() {
    let chunker =
        SemanticVectorChunker::new(mock_embedding_func(8), 1200, 0.5, None).unwrap();
    let text = "Hello world. How are you? I am fine. Thank you very much.";
    let result = chunker.chunk(text, "doc1").await.unwrap();
    for chunk in &result {
        assert!(chunk.tokens > 0);
    }
}

// ---------------------------------------------------------------------
// ChunkerRegistry (11 cases)
// ---------------------------------------------------------------------

fn zero_embedding_func() -> Arc<EmbeddingFunc> {
    struct ZeroFunc;
    #[async_trait]
    impl EmbeddingFunction for ZeroFunc {
        async fn embed(
            &self,
            texts: Vec<String>,
            _context: Option<easyrag_core::llm::embedding_func::EmbeddingContext>,
        ) -> easyrag_core::Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0f32; 4]).collect())
        }
    }
    Arc::new(EmbeddingFunc::new(4, Arc::new(ZeroFunc)))
}

#[test]
fn registry_create_token_size() {
    let chunker = ChunkerRegistry::create("token_size", ChunkerConfig::default()).unwrap();
    assert_eq!("token_size", chunker.chunker_type());
}

#[test]
fn registry_create_default() {
    let chunker = ChunkerRegistry::create("default", ChunkerConfig::default()).unwrap();
    assert_eq!("token_size", chunker.chunker_type());
}

#[test]
fn registry_create_recursive_character() {
    let chunker = ChunkerRegistry::create("recursive_character", ChunkerConfig::default()).unwrap();
    assert_eq!("recursive_character", chunker.chunker_type());
}

#[test]
fn registry_create_semantic_vector() {
    let chunker = ChunkerRegistry::create(
        "semantic_vector",
        ChunkerConfig {
            embedding_func: Some(zero_embedding_func()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!("semantic_vector", chunker.chunker_type());
}

#[test]
fn registry_create_semantic_vector_without_embedding_func_throws() {
    let err = ChunkerRegistry::create("semantic_vector", ChunkerConfig::default())
        .expect_err("must require embedding_func");
    assert!(matches!(err, easyrag_core::CoreError::InvalidArgument(_)));
}

#[test]
fn registry_create_unknown_type_falls_back_to_token_size() {
    let chunker = ChunkerRegistry::create("unknown_type", ChunkerConfig::default()).unwrap();
    assert_eq!("token_size", chunker.chunker_type());
}

#[test]
fn registry_create_case_insensitive() {
    let chunker = ChunkerRegistry::create("TOKEN_SIZE", ChunkerConfig::default()).unwrap();
    assert_eq!("token_size", chunker.chunker_type());
}

#[tokio::test]
async fn registry_create_with_custom_config_applies_config() {
    let chunker = ChunkerRegistry::create(
        "token_size",
        ChunkerConfig {
            chunk_token_size: Some(500),
            chunk_overlap_token_size: Some(50),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!("token_size", chunker.chunker_type());
    let result = chunker.chunk("Short text", "doc1").await.unwrap();
    assert!(!result.is_empty());
}

#[test]
fn registry_create_paragraph_semantic() {
    let chunker = ChunkerRegistry::create(
        "paragraph_semantic",
        ChunkerConfig {
            embedding_func: Some(zero_embedding_func()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!("paragraph_semantic", chunker.chunker_type());
}

#[test]
fn registry_create_paragraph_semantic_alias_p() {
    let chunker = ChunkerRegistry::create(
        "p",
        ChunkerConfig {
            embedding_func: Some(zero_embedding_func()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!("paragraph_semantic", chunker.chunker_type());
}

#[test]
fn registry_create_paragraph_semantic_without_embedding_func_throws() {
    let err = ChunkerRegistry::create("paragraph_semantic", ChunkerConfig::default())
        .expect_err("must require embedding_func");
    assert!(matches!(err, easyrag_core::CoreError::InvalidArgument(_)));
}

// ---------------------------------------------------------------------
// StructureAwareChunker (9 cases)
// ---------------------------------------------------------------------

fn structure_chunker() -> StructureAwareChunker {
    StructureAwareChunker::new(500)
}

#[tokio::test]
async fn structure_empty_text_returns_empty_list() {
    assert!(structure_chunker().chunk("", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn structure_no_headings_falls_back_to_token_size() {
    let text = "Plain text without any headings. Just regular content.";
    let result = structure_chunker().chunk(text, "doc1").await.unwrap();
    assert_eq!(1, result.len());
    assert!(result[0].metadata.is_empty());
}

#[tokio::test]
async fn structure_with_headings_includes_metadata() {
    let text = "# Introduction\nThis is the introduction section.\n\n\
                # Methods\nThis describes the methods used.";
    let result = structure_chunker().chunk(text, "doc1").await.unwrap();
    assert_eq!(2, result.len());

    let intro = &result[0];
    assert_eq!(Some(&json!("Introduction")), intro.metadata.get("heading"));
    assert_eq!(Some(&json!(1)), intro.metadata.get("section_level"));
    assert_eq!(
        Some(&json!(["Introduction"])),
        intro.metadata.get("section_path")
    );

    let methods = &result[1];
    assert_eq!(Some(&json!("Methods")), methods.metadata.get("heading"));
}

#[tokio::test]
async fn structure_nested_headings_track_section_path() {
    let text = "# Chapter 1\n## Section 1.1\nContent of section 1.1\n\n\
                ## Section 1.2\nContent of section 1.2";
    let result = structure_chunker().chunk(text, "doc1").await.unwrap();
    assert_eq!(2, result.len());
    assert_eq!(
        Some(&json!(["Chapter 1", "Section 1.1"])),
        result[0].metadata.get("section_path")
    );
    assert_eq!(
        Some(&json!(["Chapter 1", "Section 1.2"])),
        result[1].metadata.get("section_path")
    );
}

#[tokio::test]
async fn structure_order_indices_are_sequential() {
    let text = "# A\nContent A\n# B\nContent B\n# C\nContent C";
    let result = structure_chunker().chunk(text, "doc1").await.unwrap();
    for (index, chunk) in result.iter().enumerate() {
        assert_eq!(index as u32, chunk.chunk_order_index);
    }
}

#[tokio::test]
async fn structure_preamble_included_as_chunk() {
    let text = "Some preamble text before any heading.\n\n# First Section\nContent here.";
    let result = structure_chunker().chunk(text, "doc1").await.unwrap();
    assert!(result.len() >= 2);
    // Preamble has an empty heading.
    assert_eq!(Some(&json!("")), result[0].metadata.get("heading"));
}

#[test]
fn structure_registry_creates_correct_type() {
    let chunker = ChunkerRegistry::create(
        "structure_aware",
        ChunkerConfig {
            chunk_token_size: Some(800),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!("structure_aware", chunker.chunker_type());
}

#[test]
fn structure_registry_shorthand_creates() {
    let chunker = ChunkerRegistry::create("s", ChunkerConfig::default()).unwrap();
    assert_eq!("structure_aware", chunker.chunker_type());
}
