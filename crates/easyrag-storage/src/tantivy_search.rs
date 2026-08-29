//! Embedded tantivy full-text search (Rust replacement for
//! `lucene/LuceneKeywordSearch.kt`).
//!
//! Equivalence notes:
//! - BM25 scoring (tantivy default) matches Lucene's default BM25Similarity.
//! - NRT semantics: `index_documents`/`remove_documents` commit the writer and
//!   reload the reader so new docs are immediately searchable
//!   (equivalent of SearcherManager.maybeRefresh).
//! - Lucene's 3x over-fetch + post time-filter is replaced by a proper
//!   fast-field range query inside the boolean query.
//! - Tokenization: "simple" analyzer — whitespace/punctuation splitting,
//!   lowercasing; CJK ideographs are emitted as single-character tokens
//!   (mirrors Lucene StandardAnalyzer's CJK behavior). jieba can be enabled
//!   via the `jieba` feature in a later iteration.

use async_trait::async_trait;
use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, Occur, Query, RangeQuery, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, TextFieldIndexing, TextOptions, Value, FAST, STORED, STRING,
};
use tantivy::tokenizer::{Token, TokenStream, Tokenizer};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const SIMPLE_TOKENIZER_NAME: &str = "easyrag_simple";

/// Simple analyzer: split on non-alphanumeric boundaries, lowercase;
/// CJK ideographs become single-character tokens.
#[derive(Clone, Default)]
pub struct SimpleTokenizer;

#[derive(Default)]
pub struct SimpleTokenStream {
    tokens: Vec<Token>,
    index: usize,
}

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'
        | '\u{3400}'..='\u{4DBF}'
        | '\u{F900}'..='\u{FAFF}'
        | '\u{3040}'..='\u{30FF}'
        | '\u{AC00}'..='\u{D7A3}' // Hangul syllables (Lucene StandardAnalyzer CJK class)
    )
}

fn tokenize_text(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut start = 0usize;
    let mut in_word = false;

    let flush = |current: &mut String, start: usize, offset: usize, tokens: &mut Vec<Token>| {
        if !current.is_empty() {
            tokens.push(Token {
                offset_from: start,
                offset_to: offset,
                position: tokens.len(),
                text: current.to_lowercase(),
                position_length: 1,
            });
            current.clear();
        }
    };

    for (i, c) in text.char_indices() {
        if is_cjk(c) {
            // Flush any pending latin word, then emit the CJK char as a token.
            flush(&mut current, start, i, &mut tokens);
            in_word = false;
            tokens.push(Token {
                offset_from: i,
                offset_to: i + c.len_utf8(),
                position: tokens.len(),
                text: c.to_lowercase().to_string(),
                position_length: 1,
            });
        } else if c.is_alphanumeric() {
            if !in_word {
                start = i;
                in_word = true;
            }
            current.push(c);
        } else {
            flush(&mut current, start, i, &mut tokens);
            in_word = false;
        }
    }
    flush(&mut current, start, text.len(), &mut tokens);
    tokens
}

impl TokenStream for SimpleTokenStream {
    fn advance(&mut self) -> bool {
        if self.index < self.tokens.len() {
            self.index += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.tokens[self.index - 1]
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.tokens[self.index - 1]
    }
}

impl Tokenizer for SimpleTokenizer {
    type TokenStream<'a> = SimpleTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> SimpleTokenStream {
        let mut stream = SimpleTokenStream::default();
        stream.tokens = tokenize_text(text);
        stream
    }
}

/// Embedded tantivy keyword search (implements `KeywordSearchable`).
///
/// One index per workspace; `biz_id` is a stored/indexed schema field used
/// as a query filter rather than a directory partition.
pub struct TantivyKeywordSearch {
    workspace: String,
    schema: Schema,
    id_field: Field,
    biz_field: Field,
    content_field: Field,
    file_path_field: Field,
    create_time_field: Field,
    // Kept alive: reader/writer reference this index.
    #[allow(dead_code)]
    index: Index,
    reader: IndexReader,
    writer: Mutex<Option<IndexWriter>>,
}

impl TantivyKeywordSearch {
    /// Create/open a keyword index for the workspace.
    /// `index_dir` empty → `./rag_storage/lucene_index/{workspace}`
    /// All biz lines of a workspace share this index; queries and deletes
    /// filter on the `biz_id` field.
    pub fn new(
        workspace: impl Into<String>,
        index_dir: &str,
        ram_buffer_mb: usize,
    ) -> easyrag_core::Result<Self> {
        let workspace = workspace.into();
        let base = if !index_dir.trim().is_empty() {
            PathBuf::from(index_dir)
        } else {
            PathBuf::from("./rag_storage")
        };
        let index_path = base.join("lucene_index").join(&workspace);
        std::fs::create_dir_all(&index_path).map_err(|e| {
            easyrag_core::CoreError::Internal(format!(
                "failed to create tantivy index dir {}: {e}",
                index_path.display()
            ))
        })?;

        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let biz_field = schema_builder.add_text_field("biz_id", STRING | STORED);
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer(SIMPLE_TOKENIZER_NAME)
            .set_index_option(IndexRecordOption::WithFreqs);
        let text_options = TextOptions::default()
            .set_indexing_options(text_indexing)
            .set_stored();
        let content_field = schema_builder.add_text_field("content", text_options);
        let file_path_field = schema_builder.add_text_field("file_path", STRING | STORED);
        let create_time_field = schema_builder.add_i64_field("create_time", FAST | STORED);
        let schema = schema_builder.build();

        let index = if Path::new(&index_path.join("meta.json")).exists() {
            Index::open_in_dir(&index_path).map_err(|e| {
                easyrag_core::CoreError::Internal(format!("failed to open tantivy index: {e}"))
            })?
        } else {
            // Fail fast on a non-empty directory without tantivy metadata:
            // this is a legacy Lucene index dir, and silently creating
            // tantivy files inside it would mix two engines' files.
            let mut non_empty = false;
            if let Ok(mut entries) = std::fs::read_dir(&index_path) {
                non_empty = entries.next().is_some();
            }
            if non_empty {
                return Err(easyrag_core::CoreError::InvalidArgument(format!(
                    "index dir {} is not empty and has no tantivy meta.json \
                     (legacy Lucene index?); migrate explicitly or use a fresh dir",
                    index_path.display()
                )));
            }
            Index::create_in_dir(&index_path, schema.clone()).map_err(|e| {
                easyrag_core::CoreError::Internal(format!(
                    "failed to create tantivy index: {e}"
                ))
            })?
        };
        index.tokenizers().register(
            SIMPLE_TOKENIZER_NAME,
            SimpleTokenizer::default(),
        );

        let writer = index
            .writer(ram_buffer_mb * 1_000_000)
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy writer: {e}")))?;

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy reader: {e}")))?;

        tracing::info!(
            "TantivyKeywordSearch initialized: path={}, workspace={}",
            index_path.display(),
            workspace
        );

        Ok(Self {
            workspace,
            schema,
            id_field,
            biz_field,
            content_field,
            file_path_field,
            create_time_field,
            index,
            reader,
            writer: Mutex::new(Some(writer)),
        })
    }

    fn now_seconds() -> i64 {
        chrono::Utc::now().timestamp()
    }

    /// Tokenize the query with the same simple analyzer and build an
    /// OR boolean query (mirrors QueryParser with defaultOperator=OR).
    fn build_query(
        &self,
        query: &str,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
    ) -> Option<Box<dyn Query>> {
        let terms: Vec<String> = tokenize_text(query)
            .into_iter()
            .map(|t| t.text)
            .collect();
        if terms.is_empty() {
            return None;
        }
        let mut clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .into_iter()
            .map(|t| {
                let term = Term::from_field_text(self.content_field, &t);
                let q: Box<dyn Query> = Box::new(TermQuery::new(
                    term,
                    IndexRecordOption::WithFreqs,
                ));
                (Occur::Should, q)
            })
            .collect();

        if time_range_start.is_some() || time_range_end.is_some() {
            let lower = time_range_start
                .map(std::ops::Bound::Included)
                .unwrap_or(std::ops::Bound::Unbounded);
            let upper = time_range_end
                .map(std::ops::Bound::Included)
                .unwrap_or(std::ops::Bound::Unbounded);
            let range: Box<dyn Query> = Box::new(RangeQuery::new_i64_bounds(
                self.schema.get_field_name(self.create_time_field).to_string(),
                lower,
                upper,
            ));
            clauses = vec![
                (Occur::Must, Box::new(BooleanQuery::new(clauses))),
                (Occur::Must, range),
            ];
        }

        Some(Box::new(BooleanQuery::new(clauses)))
    }
}

#[async_trait]
impl KeywordSearchable for TantivyKeywordSearch {
    async fn keyword_query(
        &self,
        query: &str,
        top_k: usize,
        time_range_start: Option<i64>,
        time_range_end: Option<i64>,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let Some(content_query) = self.build_query(query, time_range_start, time_range_end) else {
            return Ok(Vec::new());
        };
        // Shared per-workspace index: restrict hits to the requested biz line.
        let biz_filter: Box<dyn Query> = Box::new(TermQuery::new(
            Term::from_field_text(self.biz_field, biz_id),
            IndexRecordOption::Basic,
        ));
        let query_obj = BooleanQuery::new(vec![
            (Occur::Must, content_query),
            (Occur::Must, biz_filter),
        ]);

        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(&query_obj, &TopDocs::with_limit(top_k))
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy search: {e}")))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved: tantivy::TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy doc: {e}")))?;
            let get_str = |field: Field| -> String {
                retrieved
                    .get_first(field)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string()
            };
            let mut map = JsonMap::new();
            map.insert("id".to_string(), serde_json::Value::String(get_str(self.id_field)));
            map.insert(
                "content".to_string(),
                serde_json::Value::String(get_str(self.content_field)),
            );
            map.insert(
                "file_path".to_string(),
                serde_json::Value::String(get_str(self.file_path_field)),
            );
            map.insert("score".to_string(), serde_json::Value::from(score));
            results.push(map);
        }
        Ok(results)
    }

    async fn index_documents(
        &self,
        docs: HashMap<String, JsonMap>,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut writer_guard = self.writer.lock();
        let writer = writer_guard.as_mut().ok_or_else(|| {
            easyrag_core::CoreError::Internal("tantivy writer already closed".to_string())
        })?;
        for (id, data) in &docs {
            let Some(content) = data.get("content").and_then(|v| v.as_str()) else {
                continue;
            };
            let file_path = data
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let create_time = data
                .get("create_time")
                .and_then(|v| v.as_i64())
                .unwrap_or_else(Self::now_seconds);

            // update = delete + add (idempotent), mirrors Lucene updateDocument.
            // Scope the delete to this biz line so a same-id document owned by
            // another biz line in the shared index is not destroyed.
            writer
                .delete_query(Box::new(BooleanQuery::new(vec![
                    (
                        Occur::Must,
                        Box::new(TermQuery::new(
                            Term::from_field_text(self.id_field, id),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    ),
                    (
                        Occur::Must,
                        Box::new(TermQuery::new(
                            Term::from_field_text(self.biz_field, biz_id),
                            IndexRecordOption::Basic,
                        )),
                    ),
                ])))
                .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy delete: {e}")))?;
            writer
                .add_document(doc!(
                    self.id_field => id.as_str(),
                    self.biz_field => biz_id,
                    self.content_field => content,
                    self.file_path_field => file_path,
                    self.create_time_field => create_time,
                ))
                .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy add: {e}")))?;
        }
        writer
            .commit()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy commit: {e}")))?;
        drop(writer_guard);
        self.reader
            .reload()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy reload: {e}")))?;
        tracing::debug!(
            "Tantivy indexed {} documents for workspace={}, biz_id={}",
            docs.len(),
            self.workspace,
            biz_id
        );
        Ok(())
    }

    async fn remove_documents(&self, ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut writer_guard = self.writer.lock();
        let writer = writer_guard.as_mut().ok_or_else(|| {
            easyrag_core::CoreError::Internal("tantivy writer already closed".to_string())
        })?;
        for id in ids {
            // BooleanQuery(id AND biz_id): shared index must not delete a
            // same-id document owned by another biz line.
            writer
                .delete_query(Box::new(BooleanQuery::new(vec![
                    (
                        Occur::Must,
                        Box::new(TermQuery::new(
                            Term::from_field_text(self.id_field, id),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    ),
                    (
                        Occur::Must,
                        Box::new(TermQuery::new(
                            Term::from_field_text(self.biz_field, biz_id),
                            IndexRecordOption::Basic,
                        )),
                    ),
                ])))
                .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy delete: {e}")))?;
        }
        writer
            .commit()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy commit: {e}")))?;
        drop(writer_guard);
        self.reader
            .reload()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy reload: {e}")))?;
        tracing::debug!(
            "Tantivy removed {} documents for workspace={}, biz_id={}",
            ids.len(),
            self.workspace,
            biz_id
        );
        Ok(())
    }

    async fn commit_index(&self) -> easyrag_core::Result<()> {
        let mut writer_guard = self.writer.lock();
        if let Some(writer) = writer_guard.as_mut() {
            writer.commit().map_err(|e| {
                easyrag_core::CoreError::Internal(format!("tantivy commit: {e}"))
            })?;
        }
        drop(writer_guard);
        self.reader
            .reload()
            .map_err(|e| easyrag_core::CoreError::Internal(format!("tantivy reload: {e}")))?;
        tracing::debug!(
            "Tantivy index committed for workspace={}",
            self.workspace
        );
        Ok(())
    }

    fn close(&self) -> easyrag_core::Result<()> {
        let writer = self.writer.lock().take();
        if let Some(writer) = writer {
            let _ = writer.wait_merging_threads();
        }
        tracing::info!(
            "TantivyKeywordSearch closed for workspace={}",
            self.workspace
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_tokenizer_splits_and_lowercases() {
        let tokens = tokenize_text("Hello, World! 123");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(vec!["hello", "world", "123"], words);
    }

    #[test]
    fn simple_tokenizer_cjk_single_chars() {
        let tokens = tokenize_text("知识图谱");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(vec!["知", "识", "图", "谱"], words);
    }

    #[test]
    fn simple_tokenizer_mixed_cjk_latin() {
        let tokens = tokenize_text("RAG技术test");
        let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(vec!["rag", "技", "术", "test"], words);
    }
}
