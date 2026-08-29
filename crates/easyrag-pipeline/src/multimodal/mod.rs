//! RAGAnything multimodal processing subsystem
//!.
//!
//! Modules:
//! - [`model`]: content items, parse results, processing results
//! - [`config`]: `RagAnythingConfig` with environment-variable defaults
//! - [`prompts_en`] / [`prompts_zh`]: bilingual prompt templates
//! - [`prompt_manager`]: global prompt manager with language switching

pub mod batch_processor;
pub mod callback;
pub mod config;
pub mod context;
pub mod document_processor;
pub mod model;
pub mod multimodal_query_engine;
pub mod parser;
pub mod processor;
pub mod prompt_manager;
pub mod prompts_en;
pub mod prompts_zh;
pub mod rag_anything;
pub mod util;

pub use batch_processor::BatchProcessor;
pub use callback::{CallbackManager, MetricsCallback, ProcessingCallback, ProcessingEvent};
pub use config::RagAnythingConfig;
pub use context::{ContextConfig, ContextExtractor};
pub use document_processor::DocumentProcessor;
pub use model::{
    BatchProcessingResult, ContentItem, ModalProcessResult, ModalType, ParseResult,
};
pub use multimodal_query_engine::{MultimodalQueryEngine, TextQueryBackend};
pub use parser::{
    BatchParser, DoclingParser, MinerUParser, MultimodalParser, PaddleOCRParser, ParseCache,
};
pub use processor::{
    BaseModalProcessor, EquationModalProcessor, GenericModalProcessor, ImageModalProcessor,
    ModalProcessor, ModalProcessorRegistry, TableModalProcessor,
};
pub use rag_anything::RAGAnything;
