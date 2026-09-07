//! Multimodal document parsers.
//!
//! Independent from the text `parser` module: these parsers produce
//! structured [`ParseResult`] content lists instead of plain strings.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;

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

use super::model::{ContentItem, ParseResult};

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

/// Shared skeleton of the three external-tool parsers (docling, paddleocr,
/// mineru). Each spawns a CLI, waits under a timeout, drains both streams and
/// turns a non-zero exit into [`ParserExecutionError`]; each then resolves an
/// output directory, finds the JSON the tool left there and wraps the content
/// in the same [`ParseResult`]. Only wording and a few choices differ, so those
/// are data below and the skeleton lives here once.
///
/// Trait default methods rather than a base struct: these parsers are already
/// handed around as `Arc<dyn MultimodalParser>`, so a second trait object would
/// add an indirect layer without adding a dispatch site. The `#[async_trait]`
/// box is allocated once per subprocess spawn — a path already dominated by a
/// process that may run for minutes — never per parsed item.
#[async_trait]
pub(super) trait ExternalCliParser: MultimodalParser {
    /// Wording for log and error messages: "Docling", "PaddleOCR", "MinerU".
    /// Not derivable from [`MultimodalParser::name`] — capitalising
    /// "paddleocr" would read "Paddleocr".
    const DISPLAY_NAME: &'static str;

    /// Command whose exit code 0 means the tool is installed.
    const INSTALL_CHECK: &'static [&'static str];

    /// How long the installation check may run before it is killed.
    const INSTALL_CHECK_TIMEOUT: u64;

    /// Format lists this parser supports, concatenated in this order.
    const FORMAT_LISTS: &'static [&'static [&'static str]];

    /// How long the parse itself may run.
    fn timeout_seconds(&self) -> u64;

    /// What a non-zero exit quotes. Docling and PaddleOCR redirect their error
    /// stream into stdout, so they join both; MinerU quotes stderr alone and
    /// discards the stdout it drained.
    fn failure_output(&self, stdout: &str, stderr: &str) -> String {
        format!("{stdout}\n{stderr}")
    }

    /// Spawn `command` with `file`'s parent as cwd, wait up to
    /// [`Self::timeout_seconds`] and drain both streams. On timeout the child is
    /// killed and reaped so it cannot linger as a zombie.
    async fn execute_command(&self, command: &[String], file: &Path) -> easyrag_core::Result<()> {
        let display = Self::DISPLAY_NAME;
        let mut cmd = tokio::process::Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(file.parent().unwrap_or_else(|| Path::new(".")))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| CoreError::Internal(format!("Failed to start {display} process: {e}")))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_task = tokio::spawn(read_stream(stdout));
        let stderr_task = tokio::spawn(read_stream(stderr));

        let timeout_seconds = self.timeout_seconds();
        let wait_result =
            tokio::time::timeout(Duration::from_secs(timeout_seconds), child.wait()).await;

        let status = match wait_result {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => {
                return Err(CoreError::Internal(format!("{display} process error: {e}")))
            }
            Err(_) => {
                let _ = child.start_kill();
                // Reap the killed child to avoid a zombie process.
                let _ = child.wait().await;
                return Err(ParserExecutionError::new(
                    -1,
                    format!("{display} timed out after {}s", timeout_seconds),
                    self.name(),
                ));
            }
        };

        // Drain reader tasks briefly.
        let stdout = tokio::time::timeout(Duration::from_secs(5), stdout_task)
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default();
        let stderr = tokio::time::timeout(Duration::from_secs(5), stderr_task)
            .await
            .ok()
            .and_then(|r| r.ok())
            .unwrap_or_default();

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            let error_msg = take_last_chars(&self.failure_output(&stdout, &stderr), 2000);
            // Positional, not `{display}`: the tracing macro resolves a bare
            // `display` to `tracing::field::display` instead of this local.
            tracing::error!("{} failed (code={}): {}", Self::DISPLAY_NAME, code, error_msg);
            return Err(ParserExecutionError::new(code, error_msg, self.name()));
        }
        Ok(())
    }

    /// Shared head of `parse()`: reject a missing file, then resolve where the
    /// tool should write — `kwargs["output_dir"]` when given, else
    /// `<parent>/<stem>_<name>_output`, made absolute.
    fn resolve_output_dir(&self, file: &Path, kwargs: &JsonMap) -> easyrag_core::Result<String> {
        if !file.exists() {
            return Err(CoreError::InvalidArgument(format!(
                "File not found: {}",
                path_to_absolute(file)
            )));
        }

        let stem = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let default_output = file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{stem}_{}_output", self.name()));
        Ok(kwargs
            .get("output_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| path_to_absolute(&default_output)))
    }

    /// Whether the tool left an output directory behind. A missing one costs a
    /// single document its content — warned about, not an error — so one bad
    /// file cannot fail a batch.
    fn output_dir_exists(&self, output_dir: &str) -> bool {
        if Path::new(output_dir).exists() {
            return true;
        }
        tracing::warn!(
            "{} output directory not found: {}",
            Self::DISPLAY_NAME,
            output_dir
        );
        false
    }

    /// Read and parse one of the tool's JSON files, logging under the parser's
    /// display name when either step fails.
    fn read_output_json(&self, json_file: &Path) -> Option<JsonMap> {
        let Some(root) = std::fs::read_to_string(json_file)
            .ok()
            .and_then(|text| serde_json::from_str::<JsonMap>(&text).ok())
        else {
            tracing::error!(
                "Failed to parse {} output: {}",
                Self::DISPLAY_NAME,
                json_file.display()
            );
            return None;
        };
        Some(root)
    }

    /// Shared tail of `parse()`. `before_output_dir` is inserted *between*
    /// `parser` and `output_dir` because `serde_json` preserves insertion order
    /// and MinerU's `parse_method` has always sat there; appending it would
    /// reorder metadata that consumers already see.
    fn cli_parse_result(
        &self,
        content_list: Vec<ContentItem>,
        file: &Path,
        output_dir: &str,
        before_output_dir: Option<(&str, JsonValue)>,
    ) -> ParseResult {
        let mut metadata = JsonMap::new();
        metadata.insert("parser".to_string(), JsonValue::from(self.name()));
        if let Some((key, value)) = before_output_dir {
            metadata.insert(key.to_string(), value);
        }
        metadata.insert("output_dir".to_string(), JsonValue::from(output_dir));

        ParseResult {
            content_list,
            doc_id: String::new(),
            file_path: path_to_absolute(file),
            metadata,
        }
    }

    /// [`MultimodalParser::supported_formats`] from [`Self::FORMAT_LISTS`].
    fn supported_format_list(&self) -> Vec<String> {
        let mut formats = Vec::new();
        for list in Self::FORMAT_LISTS {
            formats.extend(list.iter().map(|f| (*f).to_string()));
        }
        formats
    }

    /// [`MultimodalParser::check_installation`] from [`Self::INSTALL_CHECK`].
    fn installation_ok(&self) -> bool {
        let ok = run_installation_check(Self::INSTALL_CHECK, Self::INSTALL_CHECK_TIMEOUT);
        if !ok {
            tracing::debug!("{} installation check failed", Self::DISPLAY_NAME);
        }
        ok
    }
}

/// Parser registry and factory for the external-tool multimodal parsers.
/// Supports built-in parsers and custom parser registration.
///
/// Process-global (a unit struct over a `OnceLock` map, addressed by parser
/// type name) and unrelated to [`crate::TextParserRegistry`], which is a
/// per-instance extension-keyed map for text parsers. The two only ever shared
/// a name, so they are named apart rather than abstracted together.
pub struct ExternalParserRegistry;

type ParserFactory = Arc<dyn Fn() -> Arc<dyn MultimodalParser> + Send + Sync>;

impl ExternalParserRegistry {
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

/// Make `path` absolute without normalising it: relative paths are joined onto
/// the process cwd, absolute ones are returned as they are.
pub(crate) fn path_to_absolute(path: &Path) -> String {
    if path.is_absolute() {
        path.to_string_lossy().to_string()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_string()
    }
}

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

/// The first regular `.json` file in a [`walk_top_down`] listing, in walk
/// order. Takes the listing rather than the directory because MinerU walks once
/// and consults the result twice; passing a directory would make it walk again.
pub(super) fn first_json_file(walked: &[PathBuf]) -> Option<PathBuf> {
    walked
        .iter()
        .filter(|p| p.is_file())
        .find(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .cloned()
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

// --- Characterization guardrail for the external CLI parsers ---

#[cfg(test)]
mod cli_characterization {
    //! The three external CLI parsers (docling, paddleocr, mineru) spawn real
    //! processes and had no coverage at all, so every message asserted below
    //! was read off the original implementations: they are the bytes the
    //! shared skeleton has to keep producing.
    //!
    //! Everything runs under `/bin/sh -c`, so the assertions hold without any
    //! of the three external tools being installed.

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use easyrag_core::JsonMap;
    use serde_json::Value as JsonValue;

    use super::docling::DoclingParser;
    use super::mineru::MinerUParser;
    use super::paddleocr::PaddleOCRParser;
    // The shared skeleton's methods are trait methods, so the trait has to be in
    // scope for a concrete parser to reach them.
    use super::ExternalCliParser;
    use super::MultimodalParser;

    /// A real file on disk. The executor runs the child with the file's parent
    /// as cwd, so a dangling parent would fail at `spawn` and mask whatever
    /// the test meant to observe.
    fn temp_input() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("doc.pdf");
        std::fs::write(&file, b"not really a pdf").unwrap();
        (dir, file)
    }

    fn sh(script: &str) -> Vec<String> {
        vec![
            "/bin/sh".to_string(),
            "-c".to_string(),
            script.to_string(),
        ]
    }

    fn argv(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| (*s).to_string()).collect()
    }

    /// Docling and PaddleOCR redirect their error stream into stdout, so a
    /// non-zero exit reports both joined by a newline…
    #[tokio::test]
    async fn failure_joins_both_streams_for_docling_and_paddleocr() {
        let (_dir, file) = temp_input();
        let cmd = sh("printf OUT; printf ERR >&2; exit 7");

        let got = DoclingParser::new()
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'docling' command failed with return code 7: OUT\nERR"
        );

        let got = PaddleOCRParser::new()
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'paddleocr' command failed with return code 7: OUT\nERR"
        );
    }

    /// …while MinerU reports stderr alone, discarding the stdout it drained.
    #[tokio::test]
    async fn failure_reports_stderr_only_for_mineru() {
        let (_dir, file) = temp_input();
        let cmd = sh("printf OUT; printf ERR >&2; exit 7");

        let got = MinerUParser::new()
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'mineru' command failed with return code 7: ERR"
        );
    }

    /// A zero exit needs no output inspection for the two stream-draining
    /// parsers.
    #[tokio::test]
    async fn zero_exit_succeeds() {
        let (_dir, file) = temp_input();
        let cmd = sh("true");
        DoclingParser::new()
            .execute_command(&cmd, &file)
            .await
            .unwrap();
        PaddleOCRParser::new()
            .execute_command(&cmd, &file)
            .await
            .unwrap();
    }

    /// A tool that is not installed surfaces the parser's display name —
    /// "Docling"/"PaddleOCR"/"MinerU", which is deliberately *not* `name()`
    /// capitalised ("paddleocr" would give "Paddleocr").
    #[tokio::test]
    async fn spawn_failure_names_the_display_name() {
        let (_dir, file) = temp_input();
        let cmd = vec!["easyrag-no-such-tool".to_string()];

        for (got, display) in [
            (
                DoclingParser::new()
                    .execute_command(&cmd, &file)
                    .await
                    .unwrap_err()
                    .to_string(),
                "Docling",
            ),
            (
                PaddleOCRParser::new()
                    .execute_command(&cmd, &file)
                    .await
                    .unwrap_err()
                    .to_string(),
                "PaddleOCR",
            ),
            (
                MinerUParser::new()
                    .execute_command(&cmd, &file)
                    .await
                    .unwrap_err()
                    .to_string(),
                "MinerU",
            ),
        ] {
            assert!(
                got.starts_with(&format!("Failed to start {display} process: ")),
                "expected the {display} prefix in: {got}"
            );
        }
    }

    /// `with_timeout` is honoured, and the kill is reported under the display
    /// name with a `-1` return code.
    #[tokio::test]
    async fn timeout_is_honoured_and_reported() {
        let (_dir, file) = temp_input();
        let cmd = sh("sleep 30");

        let got = DoclingParser::with_timeout(1)
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'docling' command failed with return code -1: Docling timed out after 1s"
        );

        let got = PaddleOCRParser::with_timeout(1)
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'paddleocr' command failed with return code -1: PaddleOCR timed out after 1s"
        );

        let got = MinerUParser::with_timeout(1)
            .execute_command(&cmd, &file)
            .await
            .unwrap_err();
        assert_eq!(
            got.to_string(),
            "Parser 'mineru' command failed with return code -1: MinerU timed out after 1s"
        );
    }

    /// On success MinerU resolves the JSON directory itself, descending
    /// `<stem>_mineru_output/<file>/<method>/` to the first directory holding
    /// a JSON file. This is the one place the `{stem}_{name}_output` naming
    /// convention is observable without a working `magic-pdf`.
    ///
    /// The assertion is unchanged from when this went through
    /// `execute_command`; only the call moved, because resolution is now its
    /// own step rather than the executor's return value.
    #[test]
    fn mineru_resolves_the_json_output_directory_on_success() {
        let (dir, file) = temp_input();
        let auto = dir
            .path()
            .join("doc_mineru_output")
            .join("doc")
            .join("auto");
        std::fs::create_dir_all(&auto).unwrap();
        std::fs::write(auto.join("content_list.json"), b"[]").unwrap();

        let got = MinerUParser::new().resolve_json_output_dir(&file);
        assert_eq!(got, auto.to_string_lossy().to_string());
    }

    /// All three resolve to `<parent>/<stem>_<parser name>_output`, absolute,
    /// and all three honour a `kwargs["output_dir"]` override verbatim (not
    /// made absolute — that is what they have always done).
    #[test]
    fn output_directory_resolution_and_override() {
        fn check<P: ExternalCliParser>(parser: &P, name: &str) {
            let (dir, file) = temp_input();
            let no_kwargs = JsonMap::new();
            let mut with_override = JsonMap::new();
            with_override.insert(
                "output_dir".to_string(),
                JsonValue::from("relative/custom/out"),
            );

            let want = dir
                .path()
                .join(format!("doc_{name}_output"))
                .to_string_lossy()
                .to_string();
            assert_eq!(
                parser.resolve_output_dir(&file, &no_kwargs).unwrap(),
                want,
                "{name} default output dir"
            );
            assert_eq!(
                parser.resolve_output_dir(&file, &with_override).unwrap(),
                "relative/custom/out",
                "{name} output_dir override"
            );
        }

        check(&DoclingParser::new(), "docling");
        check(&PaddleOCRParser::new(), "paddleocr");
        check(&MinerUParser::new(), "mineru");
    }

    /// `serde_json` preserves insertion order, so metadata key order is
    /// observable. MinerU's `parse_method` has always sat between `parser` and
    /// `output_dir`; the shared tail must keep putting it there.
    #[test]
    fn metadata_key_order_is_preserved() {
        let (_dir, file) = temp_input();

        let docling = DoclingParser::new().cli_parse_result(vec![], &file, "D", None);
        assert_eq!(
            docling.metadata.keys().collect::<Vec<_>>(),
            vec!["parser", "output_dir"]
        );
        assert_eq!(docling.metadata["parser"], JsonValue::from("docling"));
        assert_eq!(docling.metadata["output_dir"], JsonValue::from("D"));
        assert_eq!(docling.file_path, file.to_string_lossy().to_string());
        assert_eq!(docling.doc_id, "");

        let mineru = MinerUParser::new().cli_parse_result(
            vec![],
            &file,
            "M",
            Some(("parse_method", JsonValue::from("auto"))),
        );
        assert_eq!(
            mineru.metadata.keys().collect::<Vec<_>>(),
            vec!["parser", "parse_method", "output_dir"]
        );
        assert_eq!(mineru.metadata["parse_method"], JsonValue::from("auto"));
    }

    /// Each argv is spelled verbatim — this is what the shared output-dir and
    /// kwargs plumbing feeds, so a moved flag is caught here.
    #[test]
    fn command_vectors_are_built_verbatim() {
        let (dir, file) = temp_input();
        let abs = file.to_string_lossy().to_string();
        let out = dir.path().join("out").to_string_lossy().to_string();
        let no_kwargs = JsonMap::new();

        assert_eq!(
            DoclingParser::new().build_command(&file, &out, &no_kwargs),
            argv(&["docling", abs.as_str(), "--output", out.as_str(), "--format", "json"])
        );

        let mut kwargs = JsonMap::new();
        kwargs.insert("from_formats".to_string(), JsonValue::from("pdf"));
        kwargs.insert("to_formats".to_string(), JsonValue::from("md"));
        assert_eq!(
            DoclingParser::new().build_command(&file, &out, &kwargs),
            argv(&[
                "docling", abs.as_str(), "--output", out.as_str(), "--format", "json",
                "--from-formats", "pdf", "--to-formats", "md",
            ])
        );

        assert_eq!(
            PaddleOCRParser::new().build_command(&file, &out, &no_kwargs),
            argv(&[
                "python", "-m", "paddleocr", "--input", abs.as_str(), "--output", out.as_str(),
            ])
        );

        let mut kwargs = JsonMap::new();
        kwargs.insert("lang".to_string(), JsonValue::from("ch"));
        kwargs.insert("use_angle_cls".to_string(), JsonValue::Bool(true));
        assert_eq!(
            PaddleOCRParser::new().build_command(&file, &out, &kwargs),
            argv(&[
                "python", "-m", "paddleocr", "--input", abs.as_str(), "--output", out.as_str(),
                "--lang", "ch", "--use_angle_cls",
            ])
        );

        assert_eq!(
            MinerUParser::new().build_command(&file, &out, "auto", &no_kwargs),
            argv(&["magic-pdf", "-p", abs.as_str(), "-o", out.as_str(), "-m", "auto"])
        );
        // Pre-existing quirk, preserved verbatim: the "ocr" method pushes the
        // `-m` flag with no value, unlike every other method.
        assert_eq!(
            MinerUParser::new().build_command(&file, &out, "ocr", &no_kwargs),
            argv(&["magic-pdf", "-p", abs.as_str(), "-o", out.as_str(), "-m"])
        );
        assert_eq!(
            MinerUParser::new().build_command(&file, &out, "OCR", &no_kwargs),
            argv(&["magic-pdf", "-p", abs.as_str(), "-o", out.as_str(), "-m"])
        );

        let mut kwargs = JsonMap::new();
        kwargs.insert("lang".to_string(), JsonValue::from("ch"));
        kwargs.insert("start_page".to_string(), JsonValue::from(2));
        kwargs.insert("end_page".to_string(), JsonValue::from(9));
        assert_eq!(
            MinerUParser::new().build_command(&file, &out, "txt", &kwargs),
            argv(&[
                "magic-pdf", "-p", abs.as_str(), "-o", out.as_str(), "-m", "txt",
                "-l", "ch", "-s", "2", "-e", "9",
            ])
        );
    }

    /// The format lists keep both their members and their concatenation order.
    #[test]
    fn supported_formats_keep_their_exact_order() {
        assert_eq!(
            DoclingParser::new().supported_formats(),
            argv(&[
                ".pdf",
                ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
                ".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
                ".txt", ".md",
            ])
        );
        assert_eq!(
            PaddleOCRParser::new().supported_formats(),
            argv(&[
                ".pdf",
                ".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
            ])
        );
        assert_eq!(
            MinerUParser::new().supported_formats(),
            argv(&[
                ".pdf",
                ".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
                ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
            ])
        );
    }

    /// The shared head of `parse()` rejects a missing file before any process
    /// is spawned, quoting the absolute path.
    #[tokio::test]
    async fn parse_rejects_a_missing_file_with_its_absolute_path() {
        let missing = Path::new("/tmp/easyrag-no-such-input.pdf");
        let parsers: Vec<Arc<dyn MultimodalParser>> = vec![
            Arc::new(DoclingParser::new()),
            Arc::new(PaddleOCRParser::new()),
            Arc::new(MinerUParser::new()),
        ];
        for parser in parsers {
            let got = parser
                .parse(missing, None, &JsonMap::new())
                .await
                .unwrap_err();
            assert_eq!(
                got.to_string(),
                "File not found: /tmp/easyrag-no-such-input.pdf"
            );
        }
    }

    /// Output discovery is a pre-order walk sorted by name, so a JSON file
    /// nested in an earlier-sorted subdirectory wins over a shallower one.
    #[test]
    fn output_json_discovery_is_pre_order() {
        let (dir, file) = temp_input();
        let out = dir.path().join("out");
        std::fs::create_dir_all(out.join("a_nested")).unwrap();
        std::fs::write(
            out.join("a_nested").join("one.json"),
            br#"{"texts":[{"text":"NESTED"}]}"#,
        )
        .unwrap();
        std::fs::write(out.join("z_top.json"), br#"{"texts":[{"text":"TOP"}]}"#).unwrap();

        let items = DoclingParser::new().parse_output(out.to_str().unwrap(), &file);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].primary_text().0, "NESTED");
    }

    /// MinerU looks for `content_list.json` (or `*_content_list.json`) first
    /// and only falls back to "any JSON file", so the named file wins even when
    /// another JSON sorts earlier in the walk.
    #[test]
    fn mineru_prefers_content_list_json_over_the_first_json() {
        let (dir, file) = temp_input();
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(
            out.join("a_earlier.json"),
            br#"[{"type":"text","text":"FROM_ANY"}]"#,
        )
        .unwrap();
        std::fs::write(
            out.join("content_list.json"),
            br#"[{"type":"text","text":"FROM_CONTENT_LIST"}]"#,
        )
        .unwrap();

        let items = MinerUParser::new().parse_output_json(out.to_str().unwrap(), &file);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].primary_text().0, "FROM_CONTENT_LIST");
    }

    /// A missing output directory yields no content rather than an error: all
    /// three warn and carry on, so one bad document cannot fail a batch.
    #[test]
    fn missing_output_directory_yields_no_content() {
        let (dir, file) = temp_input();
        let absent = dir
            .path()
            .join("never-created")
            .to_string_lossy()
            .to_string();
        assert!(DoclingParser::new().parse_output(&absent, &file).is_empty());
        assert!(PaddleOCRParser::new().parse_output(&absent, &file).is_empty());
        assert!(MinerUParser::new()
            .parse_output_json(&absent, &file)
            .is_empty());
    }
}
