//! Centralized configuration constants for EasyRAG.

// Default values for server settings
pub const DEFAULT_WORKERS: u32 = 2;
pub const DEFAULT_MAX_GRAPH_NODES: usize = 1000;

/// Default business line id used when a request does not specify biz_id.
pub const DEFAULT_BIZ_ID: &str = "default";

// Default values for extraction settings
pub const DEFAULT_SUMMARY_LANGUAGE: &str = "English";
pub const DEFAULT_MAX_GLEANING: u32 = 1;
pub const DEFAULT_ENTITY_NAME_MAX_LENGTH: usize = 256;
pub const DEFAULT_ENTITY_NAME_MAX_BYTES: usize = 512;

// Per-response output limits for entity extraction prompts
pub const DEFAULT_MAX_EXTRACTION_RECORDS: usize = 100;
pub const DEFAULT_MAX_EXTRACTION_ENTITIES: usize = 40;

// Number of description fragments to trigger LLM summary
pub const DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE: usize = 8;
pub const DEFAULT_SUMMARY_MAX_TOKENS: usize = 8000;
pub const DEFAULT_SUMMARY_OUTPUT_MAX_TOKENS: usize = 1200;
pub const DEFAULT_SUMMARY_LENGTH_RECOMMENDED: usize = 600;
pub const DEFAULT_SUMMARY_CONTEXT_SIZE: usize = 12000;
pub const DEFAULT_MAX_EXTRACT_INPUT_TOKENS: usize = 20480;
pub const DEFAULT_MAX_SECTION_CONTEXT_TOKENS: usize = 256;
pub const DEFAULT_HEADING_LEVEL_MAX_CHARS: usize = 80;

/// Separator used to join `description` fragments during entity/relation merge
/// (see `extraction.rs`), and to read the legacy joined `source_id` string form.
///
/// `source_id` is written as a JSON array now; this constant survives purely as
/// the delimiter for parsing pre-array rows. Relation keys use `"<->"`, not
/// this constant.
pub const GRAPH_FIELD_SEP: &str = "<SEP>";

// Query and retrieval configuration defaults
pub const DEFAULT_TOP_K: usize = 40;
pub const DEFAULT_CHUNK_TOP_K: usize = 20;
pub const DEFAULT_MAX_ENTITY_TOKENS: usize = 6000;
pub const DEFAULT_MAX_RELATION_TOKENS: usize = 8000;
pub const DEFAULT_MAX_TOTAL_TOKENS: usize = 30000;
pub const DEFAULT_COSINE_THRESHOLD: f32 = 0.2;
pub const DEFAULT_RELATED_CHUNK_NUMBER: usize = 5;
pub const DEFAULT_KG_CHUNK_PICK_METHOD: &str = "VECTOR";

// Rerank configuration defaults
pub const DEFAULT_MIN_RERANK_SCORE: f32 = 0.0;
pub const DEFAULT_RERANK_BINDING: &str = "null";

// Default source ids limit in metadata for entity and relation
//
// NOT WIRED UP — registered as a follow-up by the `source_id` array refactor,
// deliberately left untouched so that deciding them stays out of that diff:
//
// - These two caps have no readers. `EntityExtractorConfig::default` hardcodes
//   `max_source_ids_per_entity: 30` (see `extraction.rs`), so 200 and 30 are two
//   unrelated numbers for the same knob. Unify or delete.
// - The `*_LIMIT_METHOD_*` group has no readers either. `merge_source_ids` is
//   unconditionally FIFO (drop the oldest chunk id past the cap), so `KEEP` —
//   which `DEFAULT_SOURCE_IDS_LIMIT_METHOD` names as the default — describes a
//   behaviour that does not exist. Wiring it up is a behaviour change; deleting
//   it is not. Decide one way.
pub const DEFAULT_MAX_SOURCE_IDS_PER_ENTITY: usize = 200;
pub const DEFAULT_MAX_SOURCE_IDS_PER_RELATION: usize = 200;

pub const SOURCE_IDS_LIMIT_METHOD_KEEP: &str = "KEEP";
pub const SOURCE_IDS_LIMIT_METHOD_FIFO: &str = "FIFO";
pub const DEFAULT_SOURCE_IDS_LIMIT_METHOD: &str = SOURCE_IDS_LIMIT_METHOD_KEEP;
pub const VALID_SOURCE_IDS_LIMIT_METHODS: &[&str] =
    &[SOURCE_IDS_LIMIT_METHOD_KEEP, SOURCE_IDS_LIMIT_METHOD_FIFO];

pub const DEFAULT_MAX_FILE_PATHS: usize = 75;
pub const DEFAULT_MAX_FILE_PATH_LENGTH: usize = 32768;
pub const DEFAULT_FILE_PATH_MORE_PLACEHOLDER: &str = "truncated";

// Default temperature for LLM
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

// Async configuration defaults
pub const DEFAULT_MAX_ASYNC: usize = 4;
pub const DEFAULT_MAX_PARALLEL_INSERT: usize = 3;

// Chunker defaults
pub const DEFAULT_R_SEPARATORS: &[&str] = &["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""];

pub const DEFAULT_CHUNK_P_SIZE: usize = 2000;

// Paragraph-semantic "drop references" detection defaults
pub const DEFAULT_P_REFERENCES_TAIL_N: usize = 2;
pub const DEFAULT_P_REFERENCES_HEADINGS: &[&str] = &["References", "Bibliography", "参考文献"];

// Native docx smart_heading tunables
pub const DEFAULT_DOCX_SMART_HEADING: bool = false;
pub const DEFAULT_DOCX_SMART_ENNUM_BLACKLIST: &str =
    "年,月,日,时,分,秒,个,人,只,条,次,项,天,号,元,角,件,名,台,种,倍,\
     万,亿,岁,周,米,克,吨,页,寸,层,间,句,字,笔,轮,批,组,套,户,家,期";
pub const DEFAULT_DOCX_SMART_CAPTION_PREFIXES: &str = "图,表,公式,Figure,Table,Fig.,Eq.,Chart";
pub const DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES: &str = "抄送,主题词";
pub const DEFAULT_DOCX_SMART_IMPRINT_CLOSER_PREFIXES: &str = "印发,印发机关";
pub const DEFAULT_DOCX_SMART_IMPRINT_CLOSER_TRAILING: &str = "印发";
pub const DEFAULT_DOCX_SMART_IMPRINT_FORWARD_PARAS: usize = 3;
pub const DEFAULT_DOCX_SMART_TOC_MIN_LINES: usize = 3;
pub const DEFAULT_DOCX_SMART_TOC_KEEP_LINES: usize = 5;
pub const DEFAULT_DOCX_SMART_DENSITY_MAX: f64 = 0.40;
pub const DEFAULT_DOCX_SMART_DENSITY_BASELINE_MARGIN: f64 = 0.10;
pub const DEFAULT_DOCX_SMART_MIN_INTER_HEADING_CHARS: usize = 200;
pub const DEFAULT_DOCX_SMART_CB2_BODY_RATIO: f64 = 0.20;
pub const DEFAULT_DOCX_SMART_CB2_OUTLINE_RATIO: f64 = 0.50;
pub const DEFAULT_DOCX_SMART_CONFIDENCE_RATIO: f64 = 0.60;
pub const DEFAULT_DOCX_SMART_MIN_TOKENS: usize = 1800;
pub const DEFAULT_DOCX_SMART_SUBDOC_MIN_TOKENS: usize = 1000;
pub const DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS: usize = 1000;
pub const DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA: f64 = 2.0;
pub const DEFAULT_DOCX_SMART_TITLE_HEAD_ZONE_RECORDS: usize = 8;
pub const DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS: usize = 0;
pub const DEFAULT_DOCX_SMART_HEADING_MAX_CHARS: usize = 180;
pub const MAX_HEADING_LENGTH: usize = 200;

// Document pipeline
pub const FULL_DOCS_FORMAT_RAW: &str = "raw";
pub const FULL_DOCS_FORMAT_EASYRAG: &str = "easyrag";
pub const FULL_DOCS_FORMAT_PENDING_PARSE: &str = "pending_parse";
pub const EASYRAG_DOC_CONTENT_PREFIX: &str = "{{ERdoc}}";

// Parser engine identifiers
pub const PARSER_ENGINE_LEGACY: &str = "legacy";
pub const PARSER_ENGINE_NATIVE: &str = "native";
pub const PARSER_ENGINE_MINERU: &str = "mineru";
pub const PARSER_ENGINE_DOCLING: &str = "docling";
pub const PARSED_DIR_NAME: &str = "__parsed__";
pub const FILE_EXTRACTION_SUMMARY_PREFIX: &str = "[File Extraction]";

// Parsed artifact directory suffixes
pub const PARSED_DIR_SUFFIX: &str = ".parsed";
pub const MINERU_RAW_DIR_SUFFIX: &str = ".mineru_raw";
pub const DOCLING_RAW_DIR_SUFFIX: &str = ".docling_raw";
pub const NATIVE_RAW_DIR_SUFFIX: &str = ".native_raw";
pub const PARSED_ARTIFACT_DIR_SUFFIXES: &[&str] = &[
    PARSED_DIR_SUFFIX,
    MINERU_RAW_DIR_SUFFIX,
    DOCLING_RAW_DIR_SUFFIX,
    NATIVE_RAW_DIR_SUFFIX,
];

// Per-file processing options
pub const PROCESS_OPTION_IMAGES: &str = "i";
pub const PROCESS_OPTION_TABLES: &str = "t";
pub const PROCESS_OPTION_EQUATIONS: &str = "e";
pub const PROCESS_OPTION_SKIP_KG: &str = "!";
pub const PROCESS_OPTION_CHUNK_FIXED: &str = "F";
pub const PROCESS_OPTION_CHUNK_RECURSIVE: &str = "R";
pub const PROCESS_OPTION_CHUNK_VECTOR: &str = "V";
pub const PROCESS_OPTION_CHUNK_PARAGRAPH: &str = "P";

pub const PROCESS_OPTION_CHUNK_CHARS: &[&str] = &[
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
    PROCESS_OPTION_CHUNK_PARAGRAPH,
];
pub const SUPPORTED_PROCESS_OPTIONS: &[&str] = &[
    PROCESS_OPTION_IMAGES,
    PROCESS_OPTION_TABLES,
    PROCESS_OPTION_EQUATIONS,
    PROCESS_OPTION_SKIP_KG,
    PROCESS_OPTION_CHUNK_FIXED,
    PROCESS_OPTION_CHUNK_RECURSIVE,
    PROCESS_OPTION_CHUNK_VECTOR,
    PROCESS_OPTION_CHUNK_PARAGRAPH,
];

pub const DEFAULT_MAX_PARALLEL_ANALYZE: usize = 5;
pub const DEFAULT_MAX_PARALLEL_PARSE_NATIVE: usize = 5;
pub const DEFAULT_MAX_PARALLEL_PARSE_MINERU: usize = 2;
pub const DEFAULT_MAX_PARALLEL_PARSE_DOCLING: usize = 2;

// Staged pipeline queue size defaults
pub const DEFAULT_QUEUE_SIZE_PARSE: usize = 20;
pub const DEFAULT_QUEUE_SIZE_ANALYZE: usize = 100;
pub const DEFAULT_QUEUE_SIZE_INSERT: usize = 4;

// LLM / embedding call priority levels
pub const DEFAULT_QUERY_PRIORITY: u32 = 5;
pub const DEFAULT_SUMMARY_PRIORITY: u32 = 8;
pub const DEFAULT_PROCESSING_PRIORITY: u32 = 10;
pub const DEFAULT_MM_ANALYSIS_PRIORITY: u32 = DEFAULT_PROCESSING_PRIORITY;

// Multimodal analysis / chunk thresholds
pub const DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS: usize = 100;
pub const DEFAULT_MM_IMAGE_MIN_PIXEL: u32 = 64;

// Embedding configuration defaults
pub const DEFAULT_EMBEDDING_FUNC_MAX_ASYNC: usize = 8;
pub const DEFAULT_EMBEDDING_BATCH_NUM: usize = 10;

// Timeout defaults (seconds)
pub const DEFAULT_TIMEOUT: u64 = 300;
pub const DEFAULT_LLM_TIMEOUT: u64 = 240;
pub const DEFAULT_EMBEDDING_TIMEOUT: u64 = 30;
pub const DEFAULT_RERANK_MAX_ASYNC: usize = DEFAULT_MAX_ASYNC;
pub const DEFAULT_RERANK_TIMEOUT: u64 = 30;

// Logging configuration defaults
pub const DEFAULT_LOG_MAX_BYTES: u64 = 10_485_760;
pub const DEFAULT_LOG_BACKUP_COUNT: u32 = 5;
pub const DEFAULT_LOG_FILENAME: &str = "easyrag.log";

// Ollama server configuration defaults
pub const DEFAULT_OLLAMA_MODEL_NAME: &str = "easyrag";
pub const DEFAULT_OLLAMA_MODEL_TAG: &str = "latest";
pub const DEFAULT_OLLAMA_MODEL_SIZE: u64 = 7_365_960_935;
pub const DEFAULT_OLLAMA_CREATED_AT: &str = "2024-01-15T00:00:00Z";
pub const DEFAULT_OLLAMA_DIGEST: &str = "sha256:easyrag";

// Document structure index (PageIndex-style reasoning retrieval)
pub const STRUCTURE_INDEX_KEY_PREFIX: &str = "struct-idx-";
pub const DEFAULT_STRUCTURE_MAX_DEPTH: usize = 4;
pub const DEFAULT_STRUCTURE_SUMMARY_MAX_TOKENS: usize = 200;

// Sentence splitting (Constants.kt). NB: the Rust `regex` crate does not
// support lookbehind, so consumers (P4 chunker) must implement the split
// manually rather than compiling this pattern verbatim.
pub const DEFAULT_SENTENCE_SPLIT_REGEX: &str = r"(?<=[.?!])\s+|(?<=[。？！])";

// Global slot scheduler defaults (cross-worker coordination, seconds)
pub const DEFAULT_GLOBAL_SLOT_HEARTBEAT_TTL: f64 = 20.0;
pub const DEFAULT_GLOBAL_SLOT_SUSPECT_GRACE: f64 = 20.0;
pub const DEFAULT_GLOBAL_SLOT_POLL_MIN: f64 = 0.05;
pub const DEFAULT_GLOBAL_SLOT_POLL_DEFERRED_MAX: f64 = 0.4;
pub const DEFAULT_GLOBAL_SLOT_WAITER_STALE_TTL: f64 = 1.0;
pub const DEFAULT_GLOBAL_SLOT_DRAIN_LIMIT: usize = 16;
pub const DEFAULT_ZOMBIE_COMPACT_THRESHOLD: usize = 64;
pub const DEFAULT_COMPACT_BATCH_LIMIT: usize = 512;

// Queue stats defaults (seconds)
pub const DEFAULT_QUEUE_STATS_STALE_TTL: f64 = 15.0;
pub const DEFAULT_QUEUE_STATS_MIN_PUBLISH_INTERVAL: f64 = 0.1;
