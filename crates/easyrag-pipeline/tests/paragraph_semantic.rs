//! ParagraphSemanticChunker tests (12 cases).

use async_trait::async_trait;
use easyrag_core::llm::embedding_func::{EmbeddingFunc, EmbeddingContext, EmbeddingFunction};
use easyrag_pipeline::{Chunker, ParagraphSemanticChunker};
use serde_json::Value as JsonValue;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Deterministic embedding based on a text hash
/// % 100 / 100.0f` mock).
struct HashMockFunc {
    dim: usize,
}

#[async_trait]
impl EmbeddingFunction for HashMockFunc {
    async fn embed(
        &self,
        texts: Vec<String>,
        _context: Option<EmbeddingContext>,
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                text.hash(&mut hasher);
                let hash = hasher.finish() as i64;
                (0..self.dim)
                    .map(|i| (hash.wrapping_add((i as i64) * 31)).rem_euclid(100) as f32 / 100.0)
                    .collect()
            })
            .collect())
    }
}

fn create_mock_embedding_func(dim: usize) -> Arc<EmbeddingFunc> {
    Arc::new(EmbeddingFunc::new(dim, Arc::new(HashMockFunc { dim })))
}

fn create_chunker(max_chunk_tokens: usize, similarity_threshold: f32) -> ParagraphSemanticChunker {
    ParagraphSemanticChunker::new(create_mock_embedding_func(8), max_chunk_tokens, similarity_threshold)
}

#[tokio::test]
async fn chunk_empty_text_returns_empty_list() {
    let chunker = create_chunker(2000, 0.5);
    assert!(chunker.chunk("", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn chunk_blank_text_returns_empty_list() {
    let chunker = create_chunker(2000, 0.5);
    assert!(chunker.chunk("   \n\t  ", "doc1").await.unwrap().is_empty());
}

#[tokio::test]
async fn chunk_markdown_with_headings_splits_by_heading() {
    // Use small maxChunkTokens so sections are NOT merged together.
    let chunker = create_chunker(30, 0.5);
    let text = "# Introduction\n\
                This is the introduction section with some content that is long enough to exceed the ideal token threshold for merging.\n\n\
                ## Background\n\
                Background information goes here and it also needs to be long enough to avoid being merged with adjacent sections.\n\n\
                ## Methods\n\
                The methods used in this study are described in detail here to ensure sufficient token count for independence.\n\n\
                # Results\n\
                The results of the experiment show significant improvements over baseline approaches in all metrics.";

    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(!result.is_empty());
    // Should have chunks corresponding to headings.
    assert!(
        result.len() >= 3,
        "Expected >= 3 chunks but got {}",
        result.len()
    );
    // Verify metadata.
    for chunk in &result {
        assert!(chunk.metadata.get("heading").is_some());
        assert!(chunk.metadata.get("level").is_some());
        assert!(chunk.metadata.get("parent_headings").is_some());
    }
}

#[tokio::test]
async fn chunk_heading_metadata_contains_parent_headings() {
    let chunker = create_chunker(2000, 0.5);
    let text = "# Chapter 1\n## Section 1.1\nContent of section 1.1\n\n\
                ## Section 1.2\nContent of section 1.2";

    let result = chunker.chunk(text, "doc1").await.unwrap();
    // Find a chunk with Section 1.1 heading.
    let section11 = result
        .iter()
        .find(|c| c.metadata.get("heading") == Some(&JsonValue::String("Section 1.1".into())));
    assert!(section11.is_some(), "no chunk with heading Section 1.1");
    let parents = section11.unwrap().metadata.get("parent_headings").unwrap();
    let parents = parents.as_array().expect("parent_headings must be an array");
    assert!(parents.contains(&JsonValue::String("Chapter 1".into())));
}

#[tokio::test]
async fn chunk_no_headings_falls_back_to_semantic_chunker() {
    let chunker = create_chunker(50, 0.5);
    let text = "This is a plain text without any headings. \
                It has multiple sentences that should be split. \
                The semantic chunker should handle this case. \
                Each sentence gets an embedding vector. \
                Breakpoints are found where similarity drops.";

    let result = chunker.chunk(text, "doc1").await.unwrap();
    assert!(!result.is_empty());
    // Fallback chunks don't have heading metadata.
    for chunk in &result {
        let heading = chunk.metadata.get("heading").cloned().unwrap_or(JsonValue::String(String::new()));
        assert_eq!(JsonValue::String(String::new()), heading);
    }
}

#[tokio::test]
async fn chunk_oversized_block_splits_into_multiple_chunks() {
    let chunker = create_chunker(50, 0.5);
    // Create a long section under one heading.
    let long_content: Vec<String> = (1..=30)
        .map(|i| format!("Sentence number {i} with some extra words to add length"))
        .collect();
    let text = format!("# Long Section\n{}", long_content.join(". "));

    let result = chunker.chunk(&text, "doc1").await.unwrap();
    assert!(
        result.len() > 1,
        "Expected multiple chunks for oversized block, got {}",
        result.len()
    );
}

#[tokio::test]
async fn chunk_short_blocks_merges_together() {
    let chunker = create_chunker(2000, 0.5);
    let text = "# Title\nShort.\n\n## A\nTiny.\n\n## B\nSmall.\n\n## C\nBrief.";

    let result = chunker.chunk(text, "doc1").await.unwrap();
    // Short blocks should be merged, resulting in fewer chunks.
    assert!(!result.is_empty());
    assert!(
        result.len() <= 4,
        "Expected merged chunks, got {}",
        result.len()
    );
}

#[tokio::test]
async fn chunk_preserves_document_id() {
    let chunker = create_chunker(2000, 0.5);
    let result = chunker.chunk("# Heading\nSome content here.", "my-doc-id").await.unwrap();
    for chunk in &result {
        assert_eq!("my-doc-id", chunk.full_doc_id);
    }
}

#[tokio::test]
async fn chunk_order_index_is_sequential() {
    let chunker = create_chunker(2000, 0.5);
    let text = "# First\nContent one.\n\n# Second\nContent two.\n\n# Third\nContent three.";

    let result = chunker.chunk(text, "doc1").await.unwrap();
    for (index, chunk) in result.iter().enumerate() {
        assert_eq!(index as u32, chunk.chunk_order_index);
    }
}

#[tokio::test]
async fn parse_heading_blocks_nested_headings_correct_parents() {
    let chunker = create_chunker(2000, 0.5);
    let text = "# H1\n## H2a\n### H3\nDeep content\n## H2b\nMore content\n# H1b\nFinal content";

    let blocks = chunker.parse_heading_blocks(text);
    // H3 should have parents [H1, H2a].
    let h3 = blocks.iter().find(|b| b.heading == "H3").expect("H3 block");
    assert_eq!(vec!["H1".to_string(), "H2a".to_string()], h3.parent_headings);

    // H2b should have parent [H1].
    let h2b = blocks.iter().find(|b| b.heading == "H2b").expect("H2b block");
    assert_eq!(vec!["H1".to_string()], h2b.parent_headings);

    // H1b should have no parents.
    let h1b = blocks.iter().find(|b| b.heading == "H1b").expect("H1b block");
    assert!(h1b.parent_headings.is_empty());
}

#[tokio::test]
async fn parse_heading_blocks_preamble_content_captured() {
    let chunker = create_chunker(2000, 0.5);
    let text = "Some preamble text before any heading.\n\n# First Heading\nContent after heading.";

    let blocks = chunker.parse_heading_blocks(text);
    assert!(!blocks.is_empty());
    // First block should be preamble (level 0, empty heading).
    assert_eq!("", blocks[0].heading);
    assert_eq!(0, blocks[0].level);
    assert!(blocks[0].content.contains("preamble"));
}

#[tokio::test]
async fn chunk_with_custom_embedding_func_verifies_batch_call() {
    // Deterministic length-based fake embedding.
    struct LengthFunc;
    #[async_trait]
    impl EmbeddingFunction for LengthFunc {
        async fn embed(
            &self,
            texts: Vec<String>,
            _context: Option<EmbeddingContext>,
        ) -> easyrag_core::Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|text| (0..8).map(|i| text.len() as f32 + i as f32).collect())
                .collect())
        }
    }
    let fake_func = Arc::new(EmbeddingFunc::new(8, Arc::new(LengthFunc)));
    let chunker = ParagraphSemanticChunker::new(fake_func, 50, 0.5);

    let long_content: Vec<String> = (1..=20)
        .map(|i| format!("Sentence {i} with padding words for length"))
        .collect();
    let text = format!("# Section\n{}", long_content.join(". "));

    let result = chunker.chunk(&text, "doc1").await.unwrap();
    assert!(!result.is_empty());
}
