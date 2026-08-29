//! Text chunkers.
//!
//! synchronous `Chunker` interface becomes async here because the
//! semantic chunkers need embedding calls (the originals bridge to
//! coroutines with `runBlocking`).

use async_trait::async_trait;
use easyrag_core::constants::{DEFAULT_CHUNK_P_SIZE, DEFAULT_EMBEDDING_BATCH_NUM};
use easyrag_core::llm::embedding_func::EmbeddingFunc;
use easyrag_core::util::text::{compute_adjacent_cosine_distances, compute_effective_threshold};
use easyrag_core::util::tokenizer::{Tokenizer, TokenizerProvider};
use easyrag_core::JsonMap;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value as JsonValue;
use std::sync::Arc;

/// A text chunk produced by a chunker.
#[derive(Clone, Debug)]
pub struct TextChunk {
    pub content: String,
    pub chunk_order_index: u32,
    pub full_doc_id: String,
    pub tokens: usize,
    pub metadata: JsonMap,
}

impl TextChunk {
    fn new(content: String, chunk_order_index: u32, full_doc_id: &str, tokens: usize) -> Self {
        Self {
            content,
            chunk_order_index,
            full_doc_id: full_doc_id.to_string(),
            tokens,
            metadata: JsonMap::new(),
        }
    }
}

/// Base chunker interface.
#[async_trait]
pub trait Chunker: Send + Sync {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>>;

    /// Registry type name (replaces type assertions in tests
    /// and callers, since trait objects cannot be downcast here).
    fn chunker_type(&self) -> &'static str;
}

/// Lets `Result<Arc<dyn Chunker>>` be used with `expect`/`expect_err`
/// and debug-printed downstream (e.g. server/ingest layers).
impl std::fmt::Debug for dyn Chunker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Chunker({})", self.chunker_type())
    }
}

/// Token-size based chunker: approximately `chunk_token_size` tokens per
/// chunk with token overlap.
pub struct TokenSizeChunker {
    chunk_token_size: usize,
    chunk_overlap_token_size: usize,
    tokenizer: Arc<Tokenizer>,
}

impl Default for TokenSizeChunker {
    fn default() -> Self {
        Self::new(1200, 100)
    }
}

impl TokenSizeChunker {
    pub fn new(chunk_token_size: usize, chunk_overlap_token_size: usize) -> Self {
        Self {
            chunk_token_size,
            chunk_overlap_token_size,
            tokenizer: TokenizerProvider::get_tokenizer(),
        }
    }
}

#[async_trait]
impl Chunker for TokenSizeChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let tokens = self.tokenizer.encode(text);
        if tokens.len() <= self.chunk_token_size {
            return Ok(vec![TextChunk::new(text.to_string(), 0, doc_id, tokens.len())]);
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut order_index = 0u32;
        while start < tokens.len() {
            let end = (start + self.chunk_token_size).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let content = self.tokenizer.decode(chunk_tokens)?;
            chunks.push(TextChunk::new(
                content,
                order_index,
                doc_id,
                chunk_tokens.len(),
            ));
            order_index += 1;
            if end >= tokens.len() {
                break;
            }
            // Guarantee forward progress even for degenerate configs
            // (overlap >= chunk size or chunk size 0), which would loop
            // forever in the original.
            start = end
                .saturating_sub(self.chunk_overlap_token_size)
                .max(start + 1);
        }
        Ok(chunks)
    }

    fn chunker_type(&self) -> &'static str {
        "token_size"
    }
}

/// Recursive character text splitter: splits recursively by separators,
/// keeping semantically related pieces together.
pub struct RecursiveCharacterChunker {
    chunk_size: usize,
    chunk_overlap: usize,
    separators: Vec<String>,
    tokenizer: Arc<Tokenizer>,
}

impl Default for RecursiveCharacterChunker {
    fn default() -> Self {
        Self::new(1200, 100)
    }
}

impl RecursiveCharacterChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: ["\n\n", "\n", "。", ".", " ", ""]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            tokenizer: TokenizerProvider::get_tokenizer(),
        }
    }

    /// Override the separator list.
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_text(&self, text: &str, seps: &[String]) -> Vec<String> {
        if self.tokenizer.count_tokens(text) <= self.chunk_size {
            let trimmed = text.trim();
            return if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            };
        }

        let (current_sep, remaining) = match seps.split_first() {
            Some((first, rest)) => (first.as_str(), rest),
            None => ("", &[] as &[String]),
        };

        let splits: Vec<String> = if current_sep.is_empty() {
            text.chars().map(|c| c.to_string()).collect()
        } else {
            text.split(current_sep).map(|s| s.to_string()).collect()
        };

        let mut good_splits: Vec<String> = Vec::new();
        let mut results: Vec<String> = Vec::new();

        for split in splits {
            if self.tokenizer.count_tokens(&split) < self.chunk_size {
                good_splits.push(split);
            } else {
                if !good_splits.is_empty() {
                    results.extend(self.merge_splits(&good_splits, current_sep));
                    good_splits.clear();
                }
                if !remaining.is_empty() {
                    results.extend(self.split_text(&split, remaining));
                } else {
                    results.push(split);
                }
            }
        }

        if !good_splits.is_empty() {
            results.extend(self.merge_splits(&good_splits, current_sep));
        }

        results
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .collect()
    }

    fn merge_splits(&self, splits: &[String], separator: &str) -> Vec<String> {
        let mut results: Vec<String> = Vec::new();
        let mut current_batch: Vec<String> = Vec::new();
        let mut current_len = 0usize;

        for split in splits {
            let split_len = self.tokenizer.count_tokens(split);
            if current_len + split_len > self.chunk_size && !current_batch.is_empty() {
                results.push(current_batch.join(separator).trim().to_string());
                // Keep overlap.
                while !current_batch.is_empty() && current_len > self.chunk_overlap {
                    let first_tokens = self.tokenizer.count_tokens(&current_batch[0]);
                    current_len = current_len.saturating_sub(first_tokens);
                    current_batch.remove(0);
                }
            }
            current_batch.push(split.clone());
            current_len += split_len;
        }

        if !current_batch.is_empty() {
            results.push(current_batch.join(separator).trim().to_string());
        }
        results
    }
}

#[async_trait]
impl Chunker for RecursiveCharacterChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let chunks = self.split_text(text, &self.separators);
        Ok(chunks
            .into_iter()
            .enumerate()
            .map(|(index, content)| {
                let tokens = self.tokenizer.count_tokens(&content);
                TextChunk::new(content, index as u32, doc_id, tokens)
            })
            .collect())
    }

    fn chunker_type(&self) -> &'static str {
        "recursive_character"
    }
}

/// Default sentence splitter replicating the lookbehind regex
/// `(?<=[.?!])\s+|(?<=[。？！])` (the Rust regex crate has no lookbehind,
/// so this is implemented as an equivalent scanner).
fn split_sentences_default(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut segments: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        if matches!(c, '。' | '？' | '！') {
            // Zero-width split right after CJK sentence punctuation.
            segments.push(chars[start..=i].iter().collect());
            start = i + 1;
        } else if matches!(c, '.' | '?' | '!') {
            // Split on the whitespace that follows ASCII punctuation.
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j > i + 1 {
                segments.push(chars[start..=i].iter().collect());
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    if start < chars.len() {
        segments.push(chars[start..].iter().collect());
    }
    segments
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn split_into_sentences(text: &str, custom_regex: Option<&Regex>) -> Vec<String> {
    match custom_regex {
        Some(regex) => regex
            .split(text)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        None => split_sentences_default(text),
    }
}

/// Semantic vector chunker: breaks where the cosine distance between
/// adjacent sentence embeddings exceeds a (percentile-adjusted) threshold,
/// while respecting a max chunk token size.
pub struct SemanticVectorChunker {
    embedding_func: Arc<EmbeddingFunc>,
    max_chunk_tokens: usize,
    similarity_threshold: f32,
    sentence_split_regex: Option<Regex>,
    tokenizer: Arc<Tokenizer>,
}

impl SemanticVectorChunker {
    pub fn new(
        embedding_func: Arc<EmbeddingFunc>,
        max_chunk_tokens: usize,
        similarity_threshold: f32,
        sentence_split_regex: Option<String>,
    ) -> easyrag_core::Result<Self> {
        let regex = match sentence_split_regex {
            Some(pattern) => Some(
                Regex::new(&pattern)
                    .map_err(|e| easyrag_core::CoreError::InvalidArgument(format!("sentence_split_regex: {e}")))?,
            ),
            None => None,
        };
        Ok(Self {
            embedding_func,
            max_chunk_tokens,
            similarity_threshold,
            sentence_split_regex: regex,
            tokenizer: TokenizerProvider::get_tokenizer(),
        })
    }

    fn create_chunk(&self, sentences: &[String], order_index: u32, doc_id: &str) -> TextChunk {
        let content = sentences.join(" ");
        let content = content.trim().to_string();
        let tokens = self.tokenizer.count_tokens(&content);
        TextChunk::new(content, order_index, doc_id, tokens)
    }
}

#[async_trait]
impl Chunker for SemanticVectorChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let sentences = split_into_sentences(text, self.sentence_split_regex.as_ref());
        if sentences.len() <= 1 {
            return Ok(vec![TextChunk::new(
                text.to_string(),
                0,
                doc_id,
                self.tokenizer.count_tokens(text),
            )]);
        }

        let embeddings = self
            .embedding_func
            .invoke_simple(sentences.clone())
            .await?;
        let distances = compute_adjacent_cosine_distances(&embeddings)?;
        let effective_threshold = compute_effective_threshold(&distances, self.similarity_threshold);

        let mut breakpoints = Vec::new();
        for (i, distance) in distances.iter().enumerate() {
            if *distance > effective_threshold {
                // Value is the index of the first sentence of the next
                // segment; the loop flushes when the *current*
                // sentence index is in this list (see below), so the
                // sentence at the breakpoint index is included in the
                // flushed chunk.
                breakpoints.push(i + 1);
            }
        }

        let mut chunks = Vec::new();
        let mut current_sentences: Vec<String> = Vec::new();
        let mut current_tokens = 0usize;
        let mut order_index = 0u32;

        for i in 0..sentences.len() {
            let sentence_tokens = self.tokenizer.count_tokens(&sentences[i]);
            if current_tokens + sentence_tokens > self.max_chunk_tokens
                && !current_sentences.is_empty()
            {
                chunks.push(self.create_chunk(&current_sentences, order_index, doc_id));
                order_index += 1;
                current_sentences.clear();
                current_tokens = 0;
            }
            current_sentences.push(sentences[i].clone());
            current_tokens += sentence_tokens;
            // literal semantics: flush after pushing the sentence
            // whose index is a recorded breakpoint, so that sentence stays
            // in the flushed chunk (off-by-one vs. the intuitive "break
            // before sentence i", intentionally preserved).
            if breakpoints.contains(&i) && !current_sentences.is_empty() {
                chunks.push(self.create_chunk(&current_sentences, order_index, doc_id));
                order_index += 1;
                current_sentences.clear();
                current_tokens = 0;
            }
        }
        if !current_sentences.is_empty() {
            chunks.push(self.create_chunk(&current_sentences, order_index, doc_id));
        }

        tracing::debug!(
            "Semantic chunking: {} sentences -> {} chunks (threshold={})",
            sentences.len(),
            chunks.len(),
            effective_threshold
        );
        Ok(chunks)
    }

    fn chunker_type(&self) -> &'static str {
        "semantic_vector"
    }
}

/// A parsed heading block from Markdown text
///
#[derive(Clone, Debug)]
pub struct HeadingBlock {
    pub heading: String,
    pub level: usize,
    pub parent_headings: Vec<String>,
    pub content: String,
    pub tokens: usize,
}

/// Paragraph semantic chunker: heading-aware chunking with oversized-block
/// sentence splitting and undersized-block merging. Falls back to
/// [`SemanticVectorChunker`] for heading-less text.
pub struct ParagraphSemanticChunker {
    embedding_func: Arc<EmbeddingFunc>,
    max_chunk_tokens: usize,
    similarity_threshold: f32,
    tokenizer: Arc<Tokenizer>,
}

const IDEAL_RATIO: f64 = 0.75;

static HEADING_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^(#{1,6})\s+(.+)$").expect("heading regex"));

impl ParagraphSemanticChunker {
    pub fn new(
        embedding_func: Arc<EmbeddingFunc>,
        max_chunk_tokens: usize,
        similarity_threshold: f32,
    ) -> Self {
        Self {
            embedding_func,
            max_chunk_tokens,
            similarity_threshold,
            tokenizer: TokenizerProvider::get_tokenizer(),
        }
    }

    /// Parse text into heading blocks with parent tracking.
    pub fn parse_heading_blocks(&self, text: &str) -> Vec<HeadingBlock> {
        let matches: Vec<_> = HEADING_REGEX.captures_iter(text).collect();
        if matches.is_empty() {
            return Vec::new();
        }

        let mut blocks = Vec::new();
        let mut heading_stack: Vec<(usize, String)> = Vec::new();

        // Preamble before the first heading.
        let first_start = matches[0].get(0).unwrap().start();
        let preamble = text[..first_start].trim();
        if !preamble.trim().is_empty() {
            blocks.push(HeadingBlock {
                heading: String::new(),
                level: 0,
                parent_headings: Vec::new(),
                content: preamble.to_string(),
                tokens: self.tokenizer.count_tokens(preamble),
            });
        }

        for (i, caps) in matches.iter().enumerate() {
            let whole = caps.get(0).unwrap();
            let level = caps[1].len();
            let heading = caps[2].trim().to_string();

            while let Some((top_level, _)) = heading_stack.last() {
                if *top_level >= level {
                    heading_stack.pop();
                } else {
                    break;
                }
            }
            let parent_headings: Vec<String> =
                heading_stack.iter().map(|(_, h)| h.clone()).collect();
            heading_stack.push((level, heading.clone()));

            let content_start = whole.end();
            let content_end = if i + 1 < matches.len() {
                matches[i + 1].get(0).unwrap().start()
            } else {
                text.len()
            };
            let content = text[content_start..content_end].trim().to_string();
            let tokens = self.tokenizer.count_tokens(&content);
            blocks.push(HeadingBlock {
                heading,
                level,
                parent_headings,
                content,
                tokens,
            });
        }
        blocks
    }

    async fn compute_embeddings_batched(
        &self,
        texts: &[String],
    ) -> easyrag_core::Result<Vec<Vec<f32>>> {
        let mut all = Vec::new();
        for batch in texts.chunks(DEFAULT_EMBEDDING_BATCH_NUM) {
            let embeddings = self.embedding_func.invoke_simple(batch.to_vec()).await?;
            all.extend(embeddings);
        }
        Ok(all)
    }

    fn find_breakpoints(&self, sentences: &[String], embeddings: &[Vec<f32>]) -> easyrag_core::Result<Vec<usize>> {
        if sentences.len() <= 1 {
            return Ok(Vec::new());
        }
        let distances = compute_adjacent_cosine_distances(embeddings)?;
        let effective_threshold = compute_effective_threshold(&distances, self.similarity_threshold);

        let mut breakpoints = Vec::new();
        let mut current_tokens = 0usize;
        for i in 0..sentences.len() {
            current_tokens += self.tokenizer.count_tokens(&sentences[i]);
            // Force break if exceeding max tokens.
            if current_tokens >= self.max_chunk_tokens && i < sentences.len() - 1 {
                breakpoints.push(i + 1);
                current_tokens = 0;
                continue;
            }
            // Break at similarity boundary.
            if i < distances.len() && distances[i] > effective_threshold && i < sentences.len() - 1
            {
                breakpoints.push(i + 1);
                current_tokens = 0;
            }
        }
        Ok(breakpoints)
    }

    async fn split_oversized_blocks(
        &self,
        blocks: Vec<HeadingBlock>,
    ) -> easyrag_core::Result<Vec<HeadingBlock>> {
        let mut result = Vec::new();
        for block in blocks {
            if block.tokens <= self.max_chunk_tokens || block.content.trim().is_empty() {
                result.push(block);
                continue;
            }
            let sentences = split_into_sentences(&block.content, None);
            if sentences.len() <= 1 {
                result.push(block);
                continue;
            }
            let embeddings = self.compute_embeddings_batched(&sentences).await?;
            let breakpoints = self.find_breakpoints(&sentences, &embeddings)?;

            let mut start = 0usize;
            for bp in breakpoints {
                let sub = sentences[start..bp].join(" ");
                let sub = sub.trim();
                if !sub.is_empty() {
                    result.push(HeadingBlock {
                        heading: block.heading.clone(),
                        level: block.level,
                        parent_headings: block.parent_headings.clone(),
                        content: sub.to_string(),
                        tokens: self.tokenizer.count_tokens(sub),
                    });
                }
                start = bp;
            }
            if start < sentences.len() {
                let sub = sentences[start..].join(" ");
                let sub = sub.trim();
                if !sub.is_empty() {
                    result.push(HeadingBlock {
                        heading: block.heading.clone(),
                        level: block.level,
                        parent_headings: block.parent_headings.clone(),
                        content: sub.to_string(),
                        tokens: self.tokenizer.count_tokens(sub),
                    });
                }
            }
        }
        Ok(result)
    }

    fn merge_undersized_blocks(&self, blocks: Vec<HeadingBlock>) -> Vec<HeadingBlock> {
        let ideal_tokens = (self.max_chunk_tokens as f64 * IDEAL_RATIO) as usize;
        let mut result = Vec::new();
        let mut i = 0usize;

        while i < blocks.len() {
            let block = &blocks[i];
            if block.tokens >= ideal_tokens || block.content.trim().is_empty() {
                result.push(block.clone());
                i += 1;
                continue;
            }

            let mut merged_content = block.content.clone();
            let mut merged_tokens = block.tokens;
            let mut j = i + 1;
            while j < blocks.len() && merged_tokens < ideal_tokens {
                let next = &blocks[j];
                // Only merge with same-or-deeper level blocks (or preamble).
                if next.level > block.level || next.level == 0 || block.level == 0 {
                    if !next.content.trim().is_empty() {
                        if !merged_content.is_empty() {
                            merged_content.push_str("\n\n");
                        }
                        if !next.heading.trim().is_empty() {
                            merged_content
                                .push_str(&format!("{} {}\n", "#".repeat(next.level), next.heading));
                        }
                        merged_content.push_str(&next.content);
                        merged_tokens += next.tokens;
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            let merged = merged_content.trim().to_string();
            result.push(HeadingBlock {
                heading: block.heading.clone(),
                level: block.level,
                parent_headings: block.parent_headings.clone(),
                tokens: self.tokenizer.count_tokens(&merged),
                content: merged,
            });
            i = j;
        }
        result
    }

    fn build_chunks(&self, blocks: Vec<HeadingBlock>, doc_id: &str) -> Vec<TextChunk> {
        blocks
            .into_iter()
            .enumerate()
            .map(|(index, block)| {
                let full_content = if !block.heading.trim().is_empty() {
                    format!("{} {}\n{}", "#".repeat(block.level), block.heading, block.content)
                } else {
                    block.content.clone()
                };
                let mut chunk = TextChunk::new(
                    full_content.trim().to_string(),
                    index as u32,
                    doc_id,
                    self.tokenizer.count_tokens(&full_content),
                );
                chunk.metadata.insert(
                    "heading".to_string(),
                    JsonValue::String(block.heading.clone()),
                );
                chunk
                    .metadata
                    .insert("level".to_string(), JsonValue::from(block.level as u64));
                chunk.metadata.insert(
                    "parent_headings".to_string(),
                    JsonValue::Array(
                        block
                            .parent_headings
                            .iter()
                            .map(|h| JsonValue::String(h.clone()))
                            .collect(),
                    ),
                );
                chunk
            })
            .filter(|c| !c.content.trim().is_empty())
            .collect()
    }
}

#[async_trait]
impl Chunker for ParagraphSemanticChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if !HEADING_REGEX.is_match(text) {
            tracing::debug!("No headings found, falling back to SemanticVectorChunker");
            let fallback = SemanticVectorChunker::new(
                Arc::clone(&self.embedding_func),
                self.max_chunk_tokens,
                self.similarity_threshold,
                None,
            )?;
            return fallback.chunk(text, doc_id).await;
        }

        let blocks = self.parse_heading_blocks(text);
        if blocks.is_empty() {
            return Ok(Vec::new());
        }
        let split_blocks = self.split_oversized_blocks(blocks).await?;
        let merged_blocks = self.merge_undersized_blocks(split_blocks);
        Ok(self.build_chunks(merged_blocks, doc_id))
    }

    fn chunker_type(&self) -> &'static str {
        "paragraph_semantic"
    }
}

/// Structure-aware chunker: splits along heading boundaries, each section
/// becomes chunk(s) with hierarchy metadata. Falls back to
/// [`TokenSizeChunker`] for heading-less documents and oversized sections.
pub struct StructureAwareChunker {
    max_chunk_tokens: usize,
    tokenizer: Arc<Tokenizer>,
}

impl StructureAwareChunker {
    pub fn new(max_chunk_tokens: usize) -> Self {
        Self {
            max_chunk_tokens,
            tokenizer: TokenizerProvider::get_tokenizer(),
        }
    }

    fn metadata_for(heading: &str, level: usize, path: &[String]) -> JsonMap {
        let mut metadata = JsonMap::new();
        metadata.insert(
            "section_path".to_string(),
            JsonValue::Array(path.iter().map(|p| JsonValue::String(p.clone())).collect()),
        );
        metadata.insert("section_level".to_string(), JsonValue::from(level as u64));
        metadata.insert("heading".to_string(), JsonValue::String(heading.to_string()));
        metadata
    }
}

#[async_trait]
impl Chunker for StructureAwareChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if !crate::heading::has_headings(text) {
            tracing::debug!(
                "No headings found in document {doc_id}, falling back to TokenSizeChunker"
            );
            let fallback = TokenSizeChunker::new(self.max_chunk_tokens, 100);
            return fallback.chunk(text, doc_id).await;
        }

        let headings = crate::heading::parse_headings(text);

        // Build flat sections with ancestor-path tracking.
        #[derive(Clone)]
        struct Section {
            heading: String,
            level: usize,
            content: String,
            path: Vec<String>,
        }
        let mut sections: Vec<Section> = Vec::new();
        let mut heading_stack: Vec<(usize, String)> = Vec::new();

        let preamble = text[..headings[0].offset].trim();
        if !preamble.is_empty() {
            sections.push(Section {
                heading: String::new(),
                level: 0,
                content: preamble.to_string(),
                path: Vec::new(),
            });
        }

        for (i, h) in headings.iter().enumerate() {
            while let Some((top_level, _)) = heading_stack.last() {
                if *top_level >= h.level {
                    heading_stack.pop();
                } else {
                    break;
                }
            }
            let ancestors: Vec<String> = heading_stack.iter().map(|(_, t)| t.clone()).collect();
            heading_stack.push((h.level, h.title.clone()));

            let content_start = match text[h.offset..].find('\n') {
                Some(pos) => h.offset + pos + 1,
                None => text.len(),
            };
            let content_end = if i + 1 < headings.len() {
                headings[i + 1].offset
            } else {
                text.len()
            };
            let content = text[content_start..content_end].trim();
            if !content.is_empty() {
                let mut path = ancestors;
                path.push(h.title.clone());
                sections.push(Section {
                    heading: h.title.clone(),
                    level: h.level,
                    content: content.to_string(),
                    path,
                });
            }
        }

        let mut chunks = Vec::new();
        let mut order_index = 0u32;
        for section in &sections {
            let section_tokens = self.tokenizer.count_tokens(&section.content);
            if section_tokens <= self.max_chunk_tokens {
                let mut chunk = TextChunk::new(
                    section.content.clone(),
                    order_index,
                    doc_id,
                    section_tokens,
                );
                chunk.metadata = Self::metadata_for(&section.heading, section.level, &section.path);
                chunks.push(chunk);
                order_index += 1;
            } else {
                let sub_chunker = TokenSizeChunker::new(self.max_chunk_tokens, 100);
                let sub_chunks = sub_chunker.chunk(&section.content, doc_id).await?;
                for sub in sub_chunks {
                    let mut chunk = TextChunk::new(sub.content, order_index, doc_id, sub.tokens);
                    chunk.metadata =
                        Self::metadata_for(&section.heading, section.level, &section.path);
                    chunks.push(chunk);
                    order_index += 1;
                }
            }
        }

        tracing::debug!(
            "Structure-aware chunking: {} headings -> {} chunks for document {doc_id}",
            headings.len(),
            chunks.len()
        );
        Ok(chunks)
    }

    fn chunker_type(&self) -> &'static str {
        "structure_aware"
    }
}

/// Configuration for [`ChunkerRegistry::create`]
///
#[derive(Clone, Default)]
pub struct ChunkerConfig {
    pub chunk_token_size: Option<usize>,
    pub chunk_overlap_token_size: Option<usize>,
    pub chunk_size: Option<usize>,
    pub chunk_overlap: Option<usize>,
    pub similarity_threshold: Option<f32>,
    pub sentence_split_regex: Option<String>,
    pub embedding_func: Option<Arc<EmbeddingFunc>>,
}

/// Chunker registry — routes to the appropriate chunker by type name
///
pub struct ChunkerRegistry;

impl ChunkerRegistry {
    pub fn create(
        chunker_type: &str,
        config: ChunkerConfig,
    ) -> easyrag_core::Result<Arc<dyn Chunker>> {
        Ok(match chunker_type.to_lowercase().as_str() {
            "token_size" | "default" => Arc::new(TokenSizeChunker::new(
                config.chunk_token_size.unwrap_or(1200),
                config.chunk_overlap_token_size.unwrap_or(100),
            )),
            "recursive_character" => Arc::new(RecursiveCharacterChunker::new(
                config.chunk_size.unwrap_or(1200),
                config.chunk_overlap.unwrap_or(100),
            )),
            "semantic_vector" => {
                let embedding_func = config.embedding_func.ok_or_else(|| {
                    easyrag_core::CoreError::InvalidArgument(
                        "semantic_vector chunker requires 'embedding_func' in config".to_string(),
                    )
                })?;
                Arc::new(SemanticVectorChunker::new(
                    embedding_func,
                    config.chunk_token_size.unwrap_or(1200),
                    config.similarity_threshold.unwrap_or(0.5),
                    config.sentence_split_regex,
                )?)
            }
            "paragraph_semantic" | "p" => {
                let embedding_func = config.embedding_func.ok_or_else(|| {
                    easyrag_core::CoreError::InvalidArgument(
                        "paragraph_semantic chunker requires 'embedding_func' in config"
                            .to_string(),
                    )
                })?;
                Arc::new(ParagraphSemanticChunker::new(
                    embedding_func,
                    config.chunk_token_size.unwrap_or(DEFAULT_CHUNK_P_SIZE),
                    config.similarity_threshold.unwrap_or(0.5),
                ))
            }
            "structure_aware" | "s" => Arc::new(StructureAwareChunker::new(
                config.chunk_token_size.unwrap_or(1200),
            )),
            _ => Arc::new(TokenSizeChunker::default()),
        })
    }
}
