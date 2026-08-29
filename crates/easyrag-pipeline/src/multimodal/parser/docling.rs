//! Docling document parser.
//! Invokes the `docling` CLI tool.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::mineru::path_to_absolute;
use super::{
    run_installation_check, take_last_chars, walk_top_down, MultimodalParser,
    ParserExecutionError, IMAGE_FORMATS, OFFICE_FORMATS, PDF_FORMATS, TEXT_FORMATS,
};

/// Docling document parser implementation.
/// Uses the docling CLI tool for document parsing.
pub struct DoclingParser {
    timeout_seconds: u64,
}

impl DoclingParser {
    pub fn new() -> Self {
        Self {
            timeout_seconds: 300,
        }
    }

    pub fn with_timeout(timeout_seconds: u64) -> Self {
        Self { timeout_seconds }
    }

    fn build_command(&self, file: &Path, output_dir: &str, kwargs: &JsonMap) -> Vec<String> {
        let mut cmd: Vec<String> = vec![
            "docling".to_string(),
            path_to_absolute(file),
            "--output".to_string(),
            output_dir.to_string(),
            "--format".to_string(),
            "json".to_string(),
        ];
        if let Some(from) = kwargs.get("from_formats").and_then(|v| v.as_str()) {
            cmd.extend(["--from-formats".to_string(), from.to_string()]);
        }
        if let Some(to) = kwargs.get("to_formats").and_then(|v| v.as_str()) {
            cmd.extend(["--to-formats".to_string(), to.to_string()]);
        }
        cmd
    }

    async fn execute_command(&self, command: &[String], file: &Path) -> easyrag_core::Result<()> {
        let mut cmd = tokio::process::Command::new(&command[0]);
        cmd.args(&command[1..])
            .current_dir(file.parent().unwrap_or_else(|| Path::new(".")))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| {
            CoreError::Internal(format!("Failed to start Docling process: {e}"))
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
                    "Docling process error: {e}"
                )))
            }
            Err(_) => {
                let _ = child.start_kill();
                // Reap the killed child to avoid a zombie process.
                let _ = child.wait().await;
                return Err(ParserExecutionError::new(
                    -1,
                    format!("Docling timed out after {}s", self.timeout_seconds),
                    self.name(),
                ));
            }
        };

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
            // redirects error stream into stdout; join both for the
            // error message.
            let merged = format!("{stdout}\n{stderr}");
            let error_msg = take_last_chars(&merged, 2000);
            tracing::error!("Docling failed (code={}): {}", code, error_msg);
            return Err(ParserExecutionError::new(code, error_msg, self.name()));
        }
        Ok(())
    }

    fn parse_output(&self, output_dir: &str, _original_file: &Path) -> Vec<ContentItem> {
        let dir = Path::new(output_dir);
        if !dir.exists() {
            tracing::warn!("Docling output directory not found: {}", output_dir);
            return Vec::new();
        }

        let Some(json_file) = walk_top_down(dir)
            .into_iter()
            .filter(|p| p.is_file())
            .find(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        else {
            return Vec::new();
        };

        let Some(root) = std::fs::read_to_string(&json_file)
            .ok()
            .and_then(|text| serde_json::from_str::<JsonMap>(&text).ok())
        else {
            tracing::error!("Failed to parse Docling output: {}", json_file.display());
            return Vec::new();
        };

        // Docling outputs a structured document with "texts", "tables",
        // "figures" etc.
        let mut content_items: Vec<ContentItem> = Vec::new();
        let mut index: i64 = 0;

        // Extract text items.
        let texts = root.get("texts").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        for text in texts {
            let Some(map) = text.as_object().cloned() else { continue };
            let text_value = map.get("text").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            let level = map.get("level").and_then(|v| v.as_i64()).unwrap_or(0);
            content_items.push(ContentItem::Text {
                text: text_value,
                text_level: level,
                content_list_index: index,
                page_idx: -1,
                metadata: map,
            });
            index += 1;
        }

        // Extract table items.
        let tables = root.get("tables").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        for table in tables {
            let Some(map) = table.as_object().cloned() else { continue };
            let table_body = map
                .get("data")
                .and_then(|v| v.as_str())
                .or_else(|| map.get("text").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            let captions = map
                .get("caption")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default();
            content_items.push(ContentItem::Table {
                table_body,
                table_img_path: String::new(),
                captions,
                footnotes: Vec::new(),
                content_list_index: index,
                page_idx: -1,
                metadata: map,
            });
            index += 1;
        }

        // Extract figure items.
        let figures = root.get("figures").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        for figure in figures {
            let Some(map) = figure.as_object().cloned() else { continue };
            let img_path = map
                .get("img_path")
                .and_then(|v| v.as_str())
                .or_else(|| map.get("path").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            let captions = map
                .get("caption")
                .and_then(|v| v.as_str())
                .map(|s| vec![s.to_string()])
                .unwrap_or_default();
            content_items.push(ContentItem::Image {
                img_path,
                captions,
                footnotes: Vec::new(),
                section_path: String::new(),
                neighbor_text: String::new(),
                content_list_index: index,
                page_idx: -1,
                metadata: map,
            });
            index += 1;
        }

        // Extract equations.
        let equations = root.get("equations").and_then(|v| v.as_array()).cloned().unwrap_or_default();
        for eq in equations {
            let Some(map) = eq.as_object().cloned() else { continue };
            let equation_text = map
                .get("latex")
                .and_then(|v| v.as_str())
                .or_else(|| map.get("text").and_then(|v| v.as_str()))
                .unwrap_or("")
                .to_string();
            content_items.push(ContentItem::Equation {
                equation_text,
                equation_format: "latex".to_string(),
                content_list_index: index,
                page_idx: -1,
                metadata: map,
            });
            index += 1;
        }

        content_items
    }
}

impl Default for DoclingParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MultimodalParser for DoclingParser {
    fn name(&self) -> &str {
        "docling"
    }

    fn supported_formats(&self) -> Vec<String> {
        let mut formats = Vec::new();
        for list in [PDF_FORMATS, OFFICE_FORMATS, IMAGE_FORMATS, TEXT_FORMATS] {
            formats.extend(list.iter().map(|f| (*f).to_string()));
        }
        formats
    }

    async fn parse(
        &self,
        file: &Path,
        _parse_method: Option<&str>,
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
            .join(format!("{stem}_docling_output"));
        let output_dir = kwargs
            .get("output_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| path_to_absolute(&default_output));

        tracing::info!(
            "Parsing {} with Docling (output={})",
            file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            output_dir
        );

        let command = self.build_command(file, &output_dir, kwargs);
        tracing::debug!("Docling command: {}", command.join(" "));

        self.execute_command(&command, file).await?;

        let content_list = self.parse_output(&output_dir, file);

        let mut metadata = JsonMap::new();
        metadata.insert("parser".to_string(), JsonValue::from("docling"));
        metadata.insert("output_dir".to_string(), JsonValue::from(output_dir));

        Ok(ParseResult {
            content_list,
            doc_id: String::new(),
            file_path: path_to_absolute(file),
            metadata,
        })
    }

    fn check_installation(&self) -> bool {
        let ok = run_installation_check(&["docling", "--version"], 10);
        if !ok {
            tracing::debug!("Docling installation check failed");
        }
        ok
    }
}
