//! MinerU document parser.
//! Invokes the `magic-pdf` CLI tool.

use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::{
    run_installation_check, take_last_chars, walk_top_down, MultimodalParser,
    ParserExecutionError, IMAGE_FORMATS, OFFICE_FORMATS, PDF_FORMATS,
};

/// MinerU document parser implementation.
/// Uses the magic-pdf CLI tool.
pub struct MinerUParser {
    timeout_seconds: u64,
}

impl MinerUParser {
    pub fn new() -> Self {
        Self {
            timeout_seconds: 300,
        }
    }

    pub fn with_timeout(timeout_seconds: u64) -> Self {
        Self { timeout_seconds }
    }

    fn build_command(&self, file: &Path, output_dir: &str, method: &str, kwargs: &JsonMap) -> Vec<String> {
        let mut cmd: Vec<String> = vec!["magic-pdf".to_string()];

        // Input file.
        cmd.extend(["-p".to_string(), path_to_absolute(file)]);

        // Output directory.
        cmd.extend(["-o".to_string(), output_dir.to_string()]);

        // Parse method.
        match method.to_lowercase().as_str() {
            "ocr" => cmd.push("-m".to_string()),
            "txt" => cmd.extend(["-m".to_string(), "txt".to_string()]),
            "auto" => cmd.extend(["-m".to_string(), "auto".to_string()]),
            other => cmd.extend(["-m".to_string(), other.to_string()]),
        }

        // Optional parameters.
        if let Some(lang) = kwargs.get("lang").and_then(|v| v.as_str()) {
            cmd.extend(["-l".to_string(), lang.to_string()]);
        }
        if let Some(start) = kwargs.get("start_page").and_then(|v| v.as_i64()) {
            cmd.extend(["-s".to_string(), start.to_string()]);
        }
        if let Some(end) = kwargs.get("end_page").and_then(|v| v.as_i64()) {
            cmd.extend(["-e".to_string(), end.to_string()]);
        }

        cmd
    }

    async fn execute_command(&self, command: &[String], file: &Path) -> easyrag_core::Result<String> {
        let mut cmd = tokio::process::Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(file.parent().unwrap_or_else(|| Path::new(".")))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            CoreError::Internal(format!("Failed to start MinerU process: {e}"))
        })?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_task = tokio::spawn(super::read_stream(stdout));
        let stderr_task = tokio::spawn(super::read_stream(stderr));

        let wait_result =
            tokio::time::timeout(Duration::from_secs(self.timeout_seconds), child.wait()).await;

        let status = match wait_result {
            Ok(Ok(status)) => status,
            Ok(Err(e)) => {
                return Err(CoreError::Internal(format!(
                    "MinerU process error: {e}"
                )))
            }
            Err(_) => {
                let _ = child.start_kill();
                // Reap the killed child to avoid a zombie process.
                let _ = child.wait().await;
                return Err(ParserExecutionError::new(
                    -1,
                    format!("MinerU timed out after {}s", self.timeout_seconds),
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
            let error_msg = take_last_chars(&stderr, 2000);
            tracing::error!("MinerU failed (code={}): {}", code, error_msg);
            return Err(ParserExecutionError::new(code, error_msg, self.name()));
        }
        let _ = stdout;

        // Determine output JSON directory (default layout).
        let stem = file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let output_base = file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{stem}_mineru_output"));
        let json_dir = find_json_output_dir(&output_base);
        Ok(json_dir
            .map(|p| path_to_absolute(&p))
            .unwrap_or_else(|| path_to_absolute(&output_base)))
    }

    fn parse_output_json(&self, json_dir: &str, _original_file: &Path) -> Vec<ContentItem> {
        let dir = Path::new(json_dir);
        if !dir.exists() {
            tracing::warn!("MinerU output directory not found: {}", json_dir);
            return Vec::new();
        }

        // Find the content_list.json file.
        let walked = walk_top_down(dir);
        let content_list_file = walked
            .iter()
            .filter(|p| p.is_file())
            .find(|p| {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                name == "content_list.json" || name.ends_with("_content_list.json")
            })
            .cloned();

        let target = match content_list_file {
            Some(f) => f,
            None => {
                // Fallback: look for any JSON file.
                let Some(any_json) = walked
                    .iter()
                    .filter(|p| p.is_file())
                    .find(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
                    .cloned()
                else {
                    tracing::warn!("No JSON output found in {}", json_dir);
                    return Vec::new();
                };
                any_json
            }
        };

        match std::fs::read_to_string(&target) {
            Ok(text) => self.parse_json_content(&text),
            Err(e) => {
                tracing::error!("Failed to read MinerU output {}: {}", target.display(), e);
                Vec::new()
            }
        }
    }

    fn parse_json_content(&self, json_content: &str) -> Vec<ContentItem> {
        match serde_json::from_str::<Vec<JsonMap>>(json_content) {
            Ok(raw_list) => raw_list
                .into_iter()
                .enumerate()
                .map(|(index, map)| ContentItem::from_raw_map(map, index as i64))
                .collect(),
            Err(e) => {
                tracing::error!("Failed to parse MinerU JSON output: {e}");
                // Fallback: treat entire content as text.
                let mut metadata = JsonMap::new();
                metadata.insert("source".to_string(), JsonValue::from("fallback"));
                vec![ContentItem::Text {
                    text: json_content.to_string(),
                    text_level: 0,
                    content_list_index: 0,
                    page_idx: -1,
                    metadata,
                }]
            }
        }
    }
}

impl Default for MinerUParser {
    fn default() -> Self {
        Self::new()
    }
}

fn find_json_output_dir(base_dir: &Path) -> Option<PathBuf> {
    if !base_dir.exists() {
        return None;
    }
    // MinerU outputs to: output_dir/{filename}/auto/ or output_dir/{filename}/ocr/
    let Ok(entries) = std::fs::read_dir(base_dir) else {
        return None;
    };
    let mut sub_dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    sub_dirs.sort();
    for sub_dir in sub_dirs {
        if let Ok(method_entries) = std::fs::read_dir(&sub_dir) {
            let mut method_dirs: Vec<PathBuf> = method_entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            method_dirs.sort();
            for method_dir in method_dirs {
                if dir_has_json(&method_dir) {
                    return Some(method_dir);
                }
            }
        }
        // Also check direct JSON files.
        if dir_has_json(&sub_dir) {
            return Some(sub_dir);
        }
    }
    Some(base_dir.to_path_buf())
}

fn dir_has_json(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        })
        .unwrap_or(false)
}

/// Closest Rust analogue of (absolute, but not
/// normalized).
pub(super) fn path_to_absolute(path: &Path) -> String {
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

#[async_trait]
impl MultimodalParser for MinerUParser {
    fn name(&self) -> &str {
        "mineru"
    }

    fn supported_formats(&self) -> Vec<String> {
        let mut formats = Vec::new();
        for list in [PDF_FORMATS, IMAGE_FORMATS, OFFICE_FORMATS] {
            formats.extend(list.iter().map(|f| (*f).to_string()));
        }
        formats
    }

    async fn parse(
        &self,
        file: &Path,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
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
            .join(format!("{stem}_mineru_output"));
        let output_dir = kwargs
            .get("output_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| path_to_absolute(&default_output));

        let method = parse_method
            .map(|s| s.to_string())
            .or_else(|| {
                kwargs
                    .get("parse_method")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "auto".to_string());

        tracing::info!(
            "Parsing {} with MinerU (method={}, output={})",
            file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            method,
            output_dir
        );

        // Build command.
        let command = self.build_command(file, &output_dir, &method, kwargs);
        tracing::debug!("MinerU command: {}", command.join(" "));

        // Execute with timeout.
        let output_json_dir = self.execute_command(&command, file).await?;

        // Parse output JSON into content list.
        let content_list = self.parse_output_json(&output_json_dir, file);

        let mut metadata = JsonMap::new();
        metadata.insert("parser".to_string(), JsonValue::from("mineru"));
        metadata.insert("parse_method".to_string(), JsonValue::from(method));
        metadata.insert("output_dir".to_string(), JsonValue::from(output_dir));

        Ok(ParseResult {
            content_list,
            doc_id: String::new(),
            file_path: path_to_absolute(file),
            metadata,
        })
    }

    fn check_installation(&self) -> bool {
        let ok = run_installation_check(&["magic-pdf", "--version"], 10);
        if !ok {
            tracing::debug!("MinerU installation check failed");
        }
        ok
    }
}
