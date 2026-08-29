//! Multimodal document parsers.
//!
//! Independent from the text `parser` module: these parsers produce
//! structured [`ParseResult`] content lists instead of plain strings.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use easyrag_core::{CoreError, JsonMap};

pub mod batch;
pub mod docling;
pub mod mineru;
pub mod paddleocr;
pub mod parse_cache;

pub use batch::BatchParser;
pub use docling::DoclingParser;
pub use mineru::MinerUParser;
pub use paddleocr::PaddleOCRParser;
pub use parse_cache::ParseCache;

use super::model::ParseResult;

/// Common file format sets.
pub const OFFICE_FORMATS: &[&str] = &[".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"];
pub const IMAGE_FORMATS: &[&str] = &[
    ".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
];
pub const TEXT_FORMATS: &[&str] = &[".txt", ".md"];
pub const PDF_FORMATS: &[&str] = &[".pdf"];

/// All formats supported across built-in parsers.
pub fn all_supported() -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    for list in [OFFICE_FORMATS, IMAGE_FORMATS, TEXT_FORMATS, PDF_FORMATS] {
        for f in list {
            all.push((*f).to_string());
        }
    }
    all
}

/// Multimodal document parser interface.
/// Independent from the existing text parser interface (which returns String).
#[async_trait]
pub trait MultimodalParser: Send + Sync {
    /// Parser name identifier.
    fn name(&self) -> &str;

    /// Supported file format extensions (e.g., ".pdf", ".png").
    fn supported_formats(&self) -> Vec<String>;

    /// Parse a document file into structured content items.
    /// `parse_method` hint: "auto", "ocr", "txt".
    async fn parse(
        &self,
        file: &Path,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult>;

    /// Parse a document from a URL. Default: unsupported
    ///
    async fn parse_url(
        &self,
        _url: &str,
        _parse_method: Option<&str>,
    ) -> easyrag_core::Result<ParseResult> {
        Err(CoreError::Unsupported(format!(
            "URL parsing not supported by {}",
            self.name()
        )))
    }

    /// Check if the parser's external dependencies are properly installed.
    fn check_installation(&self) -> bool;

    /// Check if a file format is supported by this parser
    /// (extension normalized to lowercase with leading dot).
    fn is_supported(&self, extension: &str) -> bool {
        let lower = extension.to_lowercase();
        let ext = if lower.starts_with('.') {
            lower
        } else {
            format!(".{lower}")
        };
        self.supported_formats().iter().any(|f| *f == ext)
    }
}

/// Error thrown when a parser's external tool execution fails
///
#[derive(Debug, thiserror::Error)]
#[error("Parser '{parser_name}' command failed with return code {return_code}: {error_msg}")]
pub struct ParserExecutionError {
    pub return_code: i32,
    pub error_msg: String,
    pub parser_name: String,
}

impl ParserExecutionError {
    pub fn new(return_code: i32, error_msg: impl Into<String>, parser_name: &str) -> CoreError {
        CoreError::Internal(
            Self {
                return_code,
                error_msg: error_msg.into(),
                parser_name: parser_name.to_string(),
            }
            .to_string(),
        )
    }
}

/// Parser registry and factory.
/// Supports built-in parsers and custom parser registration.
pub struct ParserRegistry;

type ParserFactory = Arc<dyn Fn() -> Arc<dyn MultimodalParser> + Send + Sync>;

impl ParserRegistry {
    fn parsers() -> &'static Mutex<HashMap<String, ParserFactory>> {
        static PARSERS: OnceLock<Mutex<HashMap<String, ParserFactory>>> = OnceLock::new();
        PARSERS.get_or_init(|| {
            let mut map: HashMap<String, ParserFactory> = HashMap::new();
            // Register built-in parsers (lazy instantiation).
            map.insert(
                "mineru".to_string(),
                Arc::new(|| Arc::new(MinerUParser::new()) as Arc<dyn MultimodalParser>),
            );
            map.insert(
                "docling".to_string(),
                Arc::new(|| Arc::new(DoclingParser::new()) as Arc<dyn MultimodalParser>),
            );
            map.insert(
                "paddleocr".to_string(),
                Arc::new(|| Arc::new(PaddleOCRParser::new()) as Arc<dyn MultimodalParser>),
            );
            Mutex::new(map)
        })
    }

    /// Get a parser instance by type name.
    /// Errors on unknown parser types.
    pub fn get_parser(parser_type: &str) -> easyrag_core::Result<Arc<dyn MultimodalParser>> {
        let guard = Self::parsers().lock().unwrap_or_else(|e| e.into_inner());
        let factory = guard.get(&parser_type.to_lowercase());
        match factory {
            Some(f) => Ok(f()),
            None => {
                let mut names: Vec<String> = guard.keys().cloned().collect();
                names.sort();
                Err(CoreError::InvalidArgument(format!(
                    "Unknown parser type: '{parser_type}'. Available: {}",
                    names.join(", ")
                )))
            }
        }
    }

    /// Register a custom parser.
    pub fn register_parser(name: &str, factory: ParserFactory) {
        Self::parsers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_lowercase(), factory);
        tracing::info!("Registered custom parser: {name}");
    }

    /// Unregister a parser by name.
    pub fn unregister_parser(name: &str) {
        Self::parsers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&name.to_lowercase());
        tracing::info!("Unregistered parser: {name}");
    }

    /// Get all supported parser names (sorted).
    pub fn get_supported_parsers() -> Vec<String> {
        let mut names: Vec<String> = Self::parsers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// Check if a parser type is registered.
    pub fn has_parser(parser_type: &str) -> bool {
        Self::parsers()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&parser_type.to_lowercase())
    }

    /// Check installation status of all registered parsers.
    pub fn check_all_installations() -> HashMap<String, bool> {
        let guard = Self::parsers().lock().unwrap_or_else(|e| e.into_inner());
        let mut out = HashMap::new();
        for (name, factory) in guard.iter() {
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                factory().check_installation()
            }))
            .unwrap_or(false);
            out.insert(name.clone(), ok);
        }
        out
    }
}

// --- Shared helpers used by the external-tool parsers ---

/// File modification time in milliseconds since epoch
///; 0 on failure.
pub(crate) fn file_mtime_millis(path: &Path) -> i64 {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Render a kwargs JsonValue the way interpolates it
/// (strings unquoted, null as "null").
pub(crate) fn kwargs_value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => "null".to_string(),
        // Match: integral floats render with a
        // trailing ".0" (e.g. 1.0), unlike serde_json's compact form.
        serde_json::Value::Number(n) => {
            if let Some(f) = n.as_f64() {
                if n.is_f64() && f == f.trunc() && f.is_finite() {
                    return format!("{f:.1}");
                }
            }
            n.to_string()
        }
        other => other.to_string(),
    }
}

/// Recursive pre-order directory walk,
/// entries sorted by name for determinism.
pub(crate) fn walk_top_down(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_top_down_into(root, &mut out);
    out
}

fn walk_top_down_into(dir: &Path, out: &mut Vec<PathBuf>) {
    out.push(dir.to_path_buf());
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            walk_top_down_into(&entry, out);
        } else {
            out.push(entry);
        }
    }
}

/// Synchronous installation check: run a command, wait up to
/// `timeout_secs`, return whether it exited with code 0.
pub(crate) fn run_installation_check(program_args: &[&str], timeout_secs: u64) -> bool {
    let Some(program) = program_args.first() else {
        return false;
    };
    let result = std::process::Command::new(program)
        .args(&program_args[1..])
        // Discard output: leaving pipes unread could block the child and
        // hang the check until the timeout.
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let Ok(mut child) = result else {
        return false;
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.code() == Some(0),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    // Reap the killed child to avoid a zombie process.
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => return false,
        }
    }
}

/// Take the last `n` chars of a string.
pub(crate) fn take_last_chars(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= n {
        s.to_string()
    } else {
        chars[chars.len() - n..].iter().collect()
    }
}

/// Read an async stream fully into a String (used for child process
/// stdout/stderr draining).
pub(crate) async fn read_stream<R: tokio::io::AsyncRead + Unpin + Send + 'static>(
    mut reader: Option<R>,
) -> String {
    use tokio::io::AsyncReadExt;
    let mut bytes = Vec::new();
    if let Some(r) = reader.as_mut() {
        // Lossy conversion: external tools may emit non-UTF-8 bytes on
        // stderr (e.g. Python tracebacks); keep the output rather than
        // dropping it on decode failure.
        let _ = r.read_to_end(&mut bytes).await;
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
