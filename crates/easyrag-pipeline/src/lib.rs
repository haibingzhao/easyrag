//! easyrag-pipeline: RAG pipeline for EasyRAG.
//!
//! - [`parser`]: document parsers + registry (txt/md/pdf/docx/pptx/xlsx/html/code)
//! - [`chunker`]: 5 chunkers + registry; [`structure`]: heading-based structure tree
//! - [`ingest`]: ingestion pipeline; [`extraction`]: entity/relation extraction
//! - [`graph_cleanup`]: graph trim/delete after a document is removed
//! - [`callback`]: processing events/metrics; [`summarizer`]: LLM structure summaries
//! - [`query`]: query engine + tree search / structure / agent strategies
//! - multimodal: implemented in P4.6

pub mod callback;
pub mod chunker;
pub mod contextual_chunker;
pub mod dedup;
pub mod dir_scan;
pub mod easy_rag;
pub mod easy_rag_config;
pub mod evaluation;
pub mod extraction;
pub mod graph_cleanup;
pub mod heading;
pub mod ingest;
pub mod multimodal;
pub mod parser;
pub mod query;
pub mod repository;
pub mod structure;
pub mod summarizer;

pub use callback::{
    event_types, CallbackManager, MetricsCallback, ProcessingCallback, ProcessingEvent,
};
pub use easy_rag::{
    parse_query_mode, EasyRAG, DEFAULT_KG_MAX_DEPTH, DEFAULT_KG_MAX_NODES,
};
pub use easy_rag_config::{EasyRAGConfig, EasyRAGConfigBuilder};
pub use chunker::{
    Chunker, ChunkerConfig, ChunkerRegistry, HeadingBlock, ParagraphSemanticChunker,
    RecursiveCharacterChunker, SemanticVectorChunker, StructureAwareChunker, TextChunk,
    TokenSizeChunker,
};
pub use contextual_chunker::ContextualChunker;
pub use dedup::MinHashDeduplicator;
pub use dir_scan::{scan_directory, ScanOutcome};
pub use evaluation::GoldenSetEvaluator;
// Re-export evaluation types from easyrag_core for convenience
pub use easyrag_core::model::evaluation::{EvalReport, GoldenItem};
pub use extraction::{EntityExtractor, EntityExtractorConfig, ExtractionResult};
pub use graph_cleanup::{cleanup_graph_by_chunks, GraphCleanupReport};
pub use heading::{has_headings, parse_headings, HeadingInfo};
pub use ingest::{InsertDocumentExtras, IngestionPipeline};
pub use multimodal::{
    BatchProcessingResult, BatchProcessor, ContentItem, DocumentProcessor,
    ModalProcessResult, ModalType, MultimodalQueryEngine, ParseResult, RAGAnything,
    RagAnythingConfig, TextQueryBackend,
};
pub use parser::{
    CodeFileParser, DocxParser, DocumentParser, DocumentProcessingOptions, HtmlParser,
    MarkdownParser, TextParserRegistry, PdfParser, PlainTextParser, PptxParser, XlsxParser,
};
pub use query::{
    AgentQueryResult, KeywordsResult, QueryDataResult, QueryEngine, QueryReference,
    QueryWithRefsResult, ReasoningStep, StructureAgentStrategy, StructureQueryResult,
    StructureQueryStrategy, ToolCallRecord, TreeSearchEngine, TreeSearchResult,
};
pub use structure::StructureTreeBuilder;
pub use summarizer::DocumentSummarizer;
