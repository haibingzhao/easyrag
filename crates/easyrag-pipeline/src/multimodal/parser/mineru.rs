//! MinerU document parser.
//! Invokes the `magic-pdf` CLI tool.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::{
    first_json_file, path_to_absolute, walk_top_down, ExternalCliParser, MultimodalParser,
    IMAGE_FORMATS, OFFICE_FORMATS, PDF_FORMATS,
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

    pub(super) fn build_command(&self, file: &Path, output_dir: &str, method: &str, kwargs: &JsonMap) -> Vec<String> {
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

    /// Where MinerU actually left its JSON.
    ///
    /// Recomputed from the file name instead of reusing the resolved output
    /// directory: this has always looked in `<parent>/<stem>_mineru_output`
    /// regardless of a `kwargs["output_dir"]` override, so pointing MinerU at a
    /// custom directory makes it parse nothing. Preserved verbatim — closing
    /// that gap would change behaviour, not refactor it.
    pub(super) fn resolve_json_output_dir(&self, file: &Path) -> String {
        let stem = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let output_base = file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(format!("{stem}_{}_output", self.name()));
        find_json_output_dir(&output_base)
            .map(|p| path_to_absolute(&p))
            .unwrap_or_else(|| path_to_absolute(&output_base))
    }

    pub(super) fn parse_output_json(&self, json_dir: &str, _original_file: &Path) -> Vec<ContentItem> {
        if !self.output_dir_exists(json_dir) {
            return Vec::new();
        }

        // Find the content_list.json file.
        let walked = walk_top_down(Path::new(json_dir));
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
                let Some(any_json) = first_json_file(&walked) else {
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

impl ExternalCliParser for MinerUParser {
    const DISPLAY_NAME: &'static str = "MinerU";
    const INSTALL_CHECK: &'static [&'static str] = &["magic-pdf", "--version"];
    const INSTALL_CHECK_TIMEOUT: u64 = 10;
    const FORMAT_LISTS: &'static [&'static [&'static str]] =
        &[PDF_FORMATS, IMAGE_FORMATS, OFFICE_FORMATS];

    fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds
    }

    /// MinerU quotes stderr alone; the stdout it drained is discarded.
    fn failure_output(&self, _stdout: &str, stderr: &str) -> String {
        stderr.to_string()
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

#[async_trait]
impl MultimodalParser for MinerUParser {
    fn name(&self) -> &str {
        "mineru"
    }

    fn supported_formats(&self) -> Vec<String> {
        self.supported_format_list()
    }

    async fn parse(
        &self,
        file: &Path,
        parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
        let output_dir = self.resolve_output_dir(file, kwargs)?;

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
            "Parsing {} with {} (method={}, output={})",
            file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            Self::DISPLAY_NAME,
            method,
            output_dir
        );

        // Build command.
        let command = self.build_command(file, &output_dir, &method, kwargs);
        tracing::debug!("{} command: {}", Self::DISPLAY_NAME, command.join(" "));

        // Execute with timeout.
        self.execute_command(&command, file).await?;

        // Parse output JSON into content list.
        let output_json_dir = self.resolve_json_output_dir(file);
        let content_list = self.parse_output_json(&output_json_dir, file);

        Ok(self.cli_parse_result(
            content_list,
            file,
            &output_dir,
            Some(("parse_method", JsonValue::from(method))),
        ))
    }

    fn check_installation(&self) -> bool {
        self.installation_ok()
    }
}
