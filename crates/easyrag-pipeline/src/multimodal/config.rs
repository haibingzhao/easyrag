//! Configuration for RAGAnything.
//! All fields default from environment variables, matching the
//! data class defaults.

/// Configuration for RAGAnything multimodal processing pipeline.
#[derive(Debug, Clone)]
pub struct RagAnythingConfig {
    // Directory Configuration
    pub working_dir: String,

    // Parser Configuration
    pub parse_method: String,
    pub parser_output_dir: String,
    pub parser: String,
    pub display_content_stats: bool,

    // Multimodal Processing Configuration
    pub enable_image_processing: bool,
    pub enable_table_processing: bool,
    pub enable_equation_processing: bool,

    // Batch Processing Configuration
    pub max_concurrent_files: usize,
    pub supported_file_extensions: Vec<String>,
    pub recursive_folder_processing: bool,

    // Context Extraction Configuration
    pub context_window: usize,
    pub context_mode: String,
    pub max_context_tokens: usize,
    pub include_headers: bool,
    pub include_captions: bool,
    pub context_filter_content_types: Vec<String>,
    pub content_format: String,

    // Path Handling Configuration
    pub use_full_path: bool,

    // Timeout Configuration
    pub parser_timeout_seconds: u64,
}

impl Default for RagAnythingConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RagAnythingConfig {
    /// Build the config from environment variables with the same defaults
    /// as the data class.
    pub fn from_env() -> Self {
        Self {
            working_dir: std::env::var("WORKING_DIR").unwrap_or_else(|_| "./rag_storage".into()),
            parse_method: std::env::var("PARSE_METHOD").unwrap_or_else(|_| "auto".into()),
            parser_output_dir: std::env::var("OUTPUT_DIR").unwrap_or_else(|_| "./output".into()),
            parser: std::env::var("PARSER").unwrap_or_else(|_| "mineru".into()),
            display_content_stats: parse_bool_env("DISPLAY_CONTENT_STATS", true),

            enable_image_processing: parse_bool_env("ENABLE_IMAGE_PROCESSING", true),
            enable_table_processing: parse_bool_env("ENABLE_TABLE_PROCESSING", true),
            enable_equation_processing: parse_bool_env("ENABLE_EQUATION_PROCESSING", true),

            max_concurrent_files: parse_int_env("MAX_CONCURRENT_FILES", 1),
            supported_file_extensions: parse_list_env(
                "SUPPORTED_FILE_EXTENSIONS",
                vec![
                    ".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
                    ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".txt", ".md",
                ],
            ),
            recursive_folder_processing: parse_bool_env("RECURSIVE_FOLDER_PROCESSING", true),

            context_window: parse_int_env("CONTEXT_WINDOW", 1),
            context_mode: std::env::var("CONTEXT_MODE").unwrap_or_else(|_| "page".into()),
            max_context_tokens: parse_int_env("MAX_CONTEXT_TOKENS", 2000),
            include_headers: parse_bool_env("INCLUDE_HEADERS", true),
            include_captions: parse_bool_env("INCLUDE_CAPTIONS", true),
            context_filter_content_types: parse_list_env(
                "CONTEXT_FILTER_CONTENT_TYPES",
                vec!["text"],
            ),
            content_format: std::env::var("CONTENT_FORMAT").unwrap_or_else(|_| "minerU".into()),

            use_full_path: parse_bool_env("USE_FULL_PATH", false),

            parser_timeout_seconds: parse_long_env("PARSER_TIMEOUT_SECONDS", 300),
        }
    }
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => value.eq_ignore_ascii_case("true") || value == "1",
        Err(_) => default,
    }
}

fn parse_int_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_long_env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn parse_list_env(name: &str, default: Vec<&str>) -> Vec<String> {
    match std::env::var(name) {
        Ok(value) => value
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Err(_) => default.into_iter().map(|s| s.to_string()).collect(),
    }
}
