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

/// Shared state of the five chunkers: one tokenizer plus the two size knobs.
///
/// Each chunker used to hold the same triple under its own names
/// (`chunk_token_size` / `chunk_size` / `max_chunk_tokens` for the size,
/// `chunk_overlap_token_size` / `chunk_overlap` / nothing for the overlap) and
/// each fetched `TokenizerProvider::get_tokenizer()` itself. Both are here once
/// now. This is composition, not a base class to inherit from, so every chunker
/// keeps its own public constructor.
pub(super) struct ChunkerBase {
    tokenizer: Arc<Tokenizer>,
    /// The chunker's size knob, whatever its own config key calls it.
    chunk_size: usize,
    /// The chunker's overlap knob; 0 where it has none.
    overlap: usize,
}

impl ChunkerBase {
    fn new(chunk_size: usize, overlap: usize) -> Self {
        Self {
            tokenizer: TokenizerProvider::get_tokenizer(),
            chunk_size,
            overlap,
        }
    }

    /// For the one chunker that needs encode/decode rather than a count.
    fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn overlap(&self) -> usize {
        self.overlap
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.count_tokens(text)
    }

    /// Turn finished pieces of text into chunks numbered from 0, counting each
    /// piece's own tokens — the `TextChunk` output contract every chunker
    /// shares.
    fn assemble(&self, pieces: Vec<String>, doc_id: &str) -> Vec<TextChunk> {
        pieces
            .into_iter()
            .enumerate()
            .map(|(index, content)| {
                let tokens = self.tokenizer.count_tokens(&content);
                TextChunk::new(content, index as u32, doc_id, tokens)
            })
            .collect()
    }

    /// One chunk at an explicit order index, for the chunkers that number as
    /// they go rather than from a finished list.
    fn assemble_at(&self, content: String, order_index: u32, doc_id: &str) -> TextChunk {
        let tokens = self.tokenizer.count_tokens(&content);
        TextChunk::new(content, order_index, doc_id, tokens)
    }
}

/// The blank-input guard all five `chunk()` implementations open with.
fn is_blank(text: &str) -> bool {
    text.trim().is_empty()
}

/// What a [`SlidingWindow`] keeps behind after a flush.
enum Retention {
    /// Drop leading pieces while more than this many tokens stay buffered —
    /// `RecursiveCharacterChunker`'s overlap.
    Overlap(usize),
    /// Drop everything — the sentence-level window has no overlap.
    Clear,
}

/// Pieces accumulated until the next one would overflow a token budget.
///
/// Both callers that share this compared *before* adding the next piece
/// (`current + tokens > max`) and refused to flush an empty buffer, so that
/// comparison lives here once. They differed only in what stayed behind, which
/// is [`Retention`] — kept as data rather than folded into `Overlap(0)`, so the
/// equivalence is structural instead of resting on no piece ever counting as
/// zero tokens. `sliding_window_tests` constructs the case where the two really
/// do diverge, which no text fixture reaches.
///
/// Two windows are deliberately NOT built on this:
///
/// * `TokenSizeChunker` slides over raw token ids and `decode`s each window.
///   Routing it through pieces would mean re-encoding decoded text, and BPE
///   decode-then-encode is not the identity at arbitrary boundaries, so token
///   counts would move. That is a behaviour change, not a refactor.
/// * `ParagraphSemanticChunker::find_breakpoints` counts *after* adding and
///   breaks on `>=` — the opposite comparison — and yields indices rather than
///   batches. Sharing a window with it would mean parameterising the
///   comparison away, which leaves nothing shared.
struct SlidingWindow {
    max_tokens: usize,
    retention: Retention,
    /// Piece plus the token count it was added with, so the overlap walk does
    /// not have to re-tokenise what the caller already counted.
    batch: Vec<(String, usize)>,
    tokens: usize,
}

impl SlidingWindow {
    fn new(max_tokens: usize, retention: Retention) -> Self {
        Self {
            max_tokens,
            retention,
            batch: Vec::new(),
            tokens: 0,
        }
    }

    /// Whether `tokens` more would overflow. Never true on an empty buffer, so
    /// a single oversized piece still lands in a window of its own.
    fn should_flush(&self, tokens: usize) -> bool {
        self.tokens + tokens > self.max_tokens && !self.batch.is_empty()
    }

    /// Hand over the buffered pieces, keeping what [`Retention`] says.
    fn flush(&mut self) -> Vec<String> {
        let out = self.batch.iter().map(|(piece, _)| piece.clone()).collect();
        match self.retention {
            Retention::Clear => {
                self.batch.clear();
                self.tokens = 0;
            }
            Retention::Overlap(overlap) => {
                while !self.batch.is_empty() && self.tokens > overlap {
                    let first = self.batch[0].1;
                    self.tokens = self.tokens.saturating_sub(first);
                    self.batch.remove(0);
                }
            }
        }
        out
    }

    fn add(&mut self, piece: String, tokens: usize) {
        self.batch.push((piece, tokens));
        self.tokens += tokens;
    }

    fn is_empty(&self) -> bool {
        self.batch.is_empty()
    }

    fn into_pieces(self) -> Vec<String> {
        self.batch.into_iter().map(|(piece, _)| piece).collect()
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
    base: ChunkerBase,
}

impl Default for TokenSizeChunker {
    fn default() -> Self {
        Self::new(1200, 100)
    }
}

impl TokenSizeChunker {
    pub fn new(chunk_token_size: usize, chunk_overlap_token_size: usize) -> Self {
        Self {
            base: ChunkerBase::new(chunk_token_size, chunk_overlap_token_size),
        }
    }
}

#[async_trait]
impl Chunker for TokenSizeChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if is_blank(text) {
            return Ok(Vec::new());
        }
        let tokens = self.base.tokenizer().encode(text);
        if tokens.len() <= self.base.chunk_size() {
            return Ok(vec![TextChunk::new(text.to_string(), 0, doc_id, tokens.len())]);
        }

        let mut chunks = Vec::new();
        let mut start = 0usize;
        let mut order_index = 0u32;
        while start < tokens.len() {
            let end = (start + self.base.chunk_size()).min(tokens.len());
            let chunk_tokens = &tokens[start..end];
            let content = self.base.tokenizer().decode(chunk_tokens)?;
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
                .saturating_sub(self.base.overlap())
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
    base: ChunkerBase,
    separators: Vec<String>,
}

impl Default for RecursiveCharacterChunker {
    fn default() -> Self {
        Self::new(1200, 100)
    }
}

impl RecursiveCharacterChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            base: ChunkerBase::new(chunk_size, chunk_overlap),
            separators: ["\n\n", "\n", "。", ".", " ", ""]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }

    /// Override the separator list.
    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_text(&self, text: &str, seps: &[String]) -> Vec<String> {
        if self.base.count_tokens(text) <= self.base.chunk_size() {
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
            if self.base.count_tokens(&split) < self.base.chunk_size() {
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
        let mut window = SlidingWindow::new(
            self.base.chunk_size(),
            // Keep overlap.
            Retention::Overlap(self.base.overlap()),
        );

        for split in splits {
            let split_len = self.base.count_tokens(split);
            if window.should_flush(split_len) {
                results.push(window.flush().join(separator).trim().to_string());
            }
            window.add(split.clone(), split_len);
        }

        if !window.is_empty() {
            results.push(window.into_pieces().join(separator).trim().to_string());
        }
        results
    }
}

#[async_trait]
impl Chunker for RecursiveCharacterChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if is_blank(text) {
            return Ok(Vec::new());
        }
        let chunks = self.split_text(text, &self.separators);
        Ok(self.base.assemble(chunks, doc_id))
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
    base: ChunkerBase,
    similarity_threshold: f32,
    sentence_split_regex: Option<Regex>,
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
            // Overlap 0: this chunker's window empties completely on flush
            // (there is no overlap knob), so it takes `Retention::Clear` below
            // rather than expressing "nothing kept" as `Overlap(0)`.
            base: ChunkerBase::new(max_chunk_tokens, 0),
            similarity_threshold,
            sentence_split_regex: regex,
        })
    }

    fn create_chunk(&self, sentences: &[String], order_index: u32, doc_id: &str) -> TextChunk {
        let content = sentences.join(" ");
        let content = content.trim().to_string();
        self.base.assemble_at(content, order_index, doc_id)
    }
}

#[async_trait]
impl Chunker for SemanticVectorChunker {
    async fn chunk(&self, text: &str, doc_id: &str) -> easyrag_core::Result<Vec<TextChunk>> {
        if is_blank(text) {
            return Ok(Vec::new());
        }
        let sentences = split_into_sentences(text, self.sentence_split_regex.as_ref());
        if sentences.len() <= 1 {
            return Ok(vec![TextChunk::new(
                text.to_string(),
                0,
                doc_id,
                self.base.count_tokens(text),
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
        let mut window = SlidingWindow::new(self.base.chunk_size(), Retention::Clear);
        let mut order_index = 0u32;

        for i in 0..sentences.len() {
            let sentence_tokens = self.base.count_tokens(&sentences[i]);
            if window.should_flush(sentence_tokens) {
                chunks.push(self.create_chunk(&window.flush(), order_index, doc_id));
                order_index += 1;
            }
            window.add(sentences[i].clone(), sentence_tokens);
            // literal semantics: flush after pushing the sentence
            // whose index is a recorded breakpoint, so that sentence stays
            // in the flushed chunk (off-by-one vs. the intuitive "break
            // before sentence i", intentionally preserved).
            if breakpoints.contains(&i) && !window.is_empty() {
                chunks.push(self.create_chunk(&window.flush(), order_index, doc_id));
                order_index += 1;
            }
        }
        if !window.is_empty() {
            chunks.push(self.create_chunk(&window.into_pieces(), order_index, doc_id));
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
    base: ChunkerBase,
    similarity_threshold: f32,
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
            base: ChunkerBase::new(max_chunk_tokens, 0),
            similarity_threshold,
        }
    }

    /// A piece of an oversized block, carrying the parent block's heading
    /// context and its own freshly counted token total.
    fn sub_block(&self, block: &HeadingBlock, content: &str) -> HeadingBlock {
        HeadingBlock {
            heading: block.heading.clone(),
            level: block.level,
            parent_headings: block.parent_headings.clone(),
            content: content.to_string(),
            tokens: self.base.count_tokens(content),
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
                tokens: self.base.count_tokens(preamble),
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
            let tokens = self.base.count_tokens(&content);
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
            current_tokens += self.base.count_tokens(&sentences[i]);
            // Force break if exceeding max tokens.
            if current_tokens >= self.base.chunk_size() && i < sentences.len() - 1 {
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
            if block.tokens <= self.base.chunk_size() || block.content.trim().is_empty() {
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
                    result.push(self.sub_block(&block, sub));
                }
                start = bp;
            }
            if start < sentences.len() {
                let sub = sentences[start..].join(" ");
                let sub = sub.trim();
                if !sub.is_empty() {
                    result.push(self.sub_block(&block, sub));
                }
            }
        }
        Ok(result)
    }

    fn merge_undersized_blocks(&self, blocks: Vec<HeadingBlock>) -> Vec<HeadingBlock> {
        let ideal_tokens = (self.base.chunk_size() as f64 * IDEAL_RATIO) as usize;
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
                tokens: self.base.count_tokens(&merged),
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
                // Deliberately not `assemble_at`: the stored content is the
                // trimmed string while the token count is taken on the
                // *untrimmed* one. Blocks that have a body are already trimmed
                // at every producer, so the two agree there — but a heading with
                // an empty body stores `"# X"` while counting `"# X\n"`. That
                // is existing observable behaviour, pinned by the
                // `paragraph_semantic_nested` golden, and must not change here.
                let mut chunk = TextChunk::new(
                    full_content.trim().to_string(),
                    index as u32,
                    doc_id,
                    self.base.count_tokens(&full_content),
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
        if is_blank(text) {
            return Ok(Vec::new());
        }
        if !HEADING_REGEX.is_match(text) {
            tracing::debug!("No headings found, falling back to SemanticVectorChunker");
            let fallback = SemanticVectorChunker::new(
                Arc::clone(&self.embedding_func),
                self.base.chunk_size(),
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
    base: ChunkerBase,
}

/// Overlap handed to the [`TokenSizeChunker`] this chunker delegates to, for
/// both the heading-less fallback and the oversized-section split. It is the
/// sub-chunker's overlap, not this chunker's own — sections are emitted whole,
/// so `ChunkerBase::overlap` stays 0 here.
const SUB_CHUNK_OVERLAP: usize = 100;

impl StructureAwareChunker {
    pub fn new(max_chunk_tokens: usize) -> Self {
        Self {
            base: ChunkerBase::new(max_chunk_tokens, 0),
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
        if is_blank(text) {
            return Ok(Vec::new());
        }
        if !crate::heading::has_headings(text) {
            tracing::debug!(
                "No headings found in document {doc_id}, falling back to TokenSizeChunker"
            );
            let fallback = TokenSizeChunker::new(self.base.chunk_size(), SUB_CHUNK_OVERLAP);
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
            let section_tokens = self.base.count_tokens(&section.content);
            if section_tokens <= self.base.chunk_size() {
                // `assemble_at` would re-count tokens this branch already has,
                // and both branches overwrite `metadata` outright, so the
                // constructor stays explicit here.
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
                let sub_chunker = TokenSizeChunker::new(self.base.chunk_size(), SUB_CHUNK_OVERLAP);
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

/// Unit-level pins for [`SlidingWindow`].
///
/// The byte-level chunker goldens (`tests/chunker_golden.rs`) cover the window
/// through real text, but three of its guarantees are hard to reach from a
/// fixture: the strictness of each comparison, and the case where
/// [`Retention::Clear`] and `Retention::Overlap(0)` genuinely diverge. These
/// drive the window directly with explicit token counts, so each assertion
/// names one comparison rather than hoping a text fixture happens to land on
/// it.
#[cfg(test)]
mod sliding_window_tests {
    use super::{Retention, SlidingWindow};

    /// The overlap walk keeps pieces while `tokens > overlap`, strictly. With
    /// `>=` it would drop one more piece and the retained tail would shrink.
    #[test]
    fn overlap_walk_stops_at_strictly_greater() {
        let mut w = SlidingWindow::new(3, Retention::Overlap(2));
        for (piece, tokens) in [("a", 1), ("b", 1), ("c", 1)] {
            w.add(piece.to_string(), tokens);
        }
        assert!(w.should_flush(1), "3 buffered + 1 more must overflow a budget of 3");
        assert_eq!(w.flush(), vec!["a", "b", "c"], "flush yields the pre-trim batch");
        // Retained exactly 2 tokens' worth; a third piece refills to the budget.
        w.add("d".to_string(), 1);
        assert_eq!(w.flush(), vec!["b", "c", "d"], "overlap must retain 2 pieces, not 1");
    }

    /// `should_flush` compares *before* adding and never fires on an empty
    /// buffer, so a single oversized piece still gets a window of its own.
    #[test]
    fn should_flush_compares_before_adding_and_never_on_empty() {
        let mut w = SlidingWindow::new(3, Retention::Clear);
        assert!(!w.should_flush(99), "empty buffer must never flush");
        w.add("x".to_string(), 1);
        w.add("y".to_string(), 1);
        assert!(!w.should_flush(1), "2 + 1 == 3 is not > 3");
        assert!(w.should_flush(2), "2 + 2 == 4 is > 3");
        let mut big = SlidingWindow::new(3, Retention::Clear);
        big.add("oversized".to_string(), 99);
        assert!(big.should_flush(0), "a buffered oversized piece flushes on the next one");
    }

    /// Why `Retention` is an enum rather than `Overlap(0)` for "keep nothing".
    ///
    /// The two agree whenever every buffered piece has a nonzero token count,
    /// but a trailing zero-token piece (an empty split, which the recursive
    /// chunker's separator pass really can produce) is drained by `Clear` and
    /// left behind by `Overlap(0)`, whose walk stops as soon as the running
    /// total hits zero. Folding them together would therefore rest on that
    /// never happening; keeping them apart makes the equivalence structural.
    #[test]
    fn clear_and_overlap_zero_diverge_on_a_trailing_zero_token_piece() {
        let mut clear = SlidingWindow::new(10, Retention::Clear);
        let mut zero = SlidingWindow::new(10, Retention::Overlap(0));
        for w in [&mut clear, &mut zero] {
            w.add("a".to_string(), 1);
            w.add(String::new(), 0);
        }
        assert_eq!(clear.flush(), zero.flush(), "the flushed chunk itself is identical");
        assert!(clear.is_empty(), "Clear drains the zero-token tail");
        assert!(!zero.is_empty(), "Overlap(0) strands it");

        for w in [&mut clear, &mut zero] {
            w.add("b".to_string(), 1);
        }
        assert_eq!(clear.flush(), vec!["b"]);
        assert_eq!(zero.flush(), vec!["", "b"], "the stranded piece leaks into the next chunk");
    }

    /// `into_pieces` and `is_empty` see the retained tail, not the flushed one.
    #[test]
    fn into_pieces_returns_the_retained_tail() {
        let mut w = SlidingWindow::new(2, Retention::Overlap(1));
        w.add("a".to_string(), 1);
        w.add("b".to_string(), 1);
        assert_eq!(w.flush(), vec!["a", "b"]);
        assert!(!w.is_empty());
        assert_eq!(w.into_pieces(), vec!["b"]);
    }
}
