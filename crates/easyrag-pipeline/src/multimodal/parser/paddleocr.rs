//! PaddleOCR document parser.
//! Invokes the `paddleocr` Python module CLI for OCR-based parsing.

use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::mineru::path_to_absolute;
use super::{
    run_installation_check, take_last_chars, walk_top_down, MultimodalParser,
    ParserExecutionError, IMAGE_FORMATS, PDF_FORMATS,
};

/// PaddleOCR document parser implementation.
/// Uses the paddleocr CLI or Python API for OCR-based parsing.
pub struct PaddleOCRParser {
    timeout_seconds: u64,
}

impl PaddleOCRParser {
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
            "python".to_string(),
            "-m".to_string(),
            "paddleocr".to_string(),
            "--input".to_string(),
            path_to_absolute(file),
            "--output".to_string(),
            output_dir.to_string(),
        ];
        if let Some(lang) = kwargs.get("lang").and_then(|v| v.as_str()) {
            cmd.extend(["--lang".to_string(), lang.to_string()]);
        }
        if kwargs.get("use_angle_cls").and_then(|v| v.as_bool()) == Some(true) {
            cmd.push("--use_angle_cls".to_string());
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
            CoreError::Internal(format!("Failed to start PaddleOCR process: {e}"))
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
                    "PaddleOCR process error: {e}"
                )))
            }
            Err(_) => {
                let _ = child.start_kill();
                // Reap the killed child to avoid a zombie process.
                let _ = child.wait().await;
                return Err(ParserExecutionError::new(
                    -1,
                    format!("PaddleOCR timed out after {}s", self.timeout_seconds),
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
            let merged = format!("{stdout}\n{stderr}");
            let error_msg = take_last_chars(&merged, 2000);
            tracing::error!("PaddleOCR failed (code={}): {}", code, error_msg);
            return Err(ParserExecutionError::new(code, error_msg, self.name()));
        }
        Ok(())
    }

    fn parse_output(&self, output_dir: &str, _original_file: &Path) -> Vec<ContentItem> {
        let dir = Path::new(output_dir);
        if !dir.exists() {
            tracing::warn!("PaddleOCR output directory not found: {}", output_dir);
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
            tracing::error!("Failed to parse PaddleOCR output: {}", json_file.display());
            return Vec::new();
        };

        let mut content_items: Vec<ContentItem> = Vec::new();
        let mut index: i64 = 0;

        // PaddleOCR typically outputs text regions with bounding boxes.
        let regions = root
            .get("regions")
            .and_then(|v| v.as_array())
            .or_else(|| root.get("results").and_then(|v| v.as_array()))
            .cloned()
            .unwrap_or_default();

        for region in regions {
            let Some(map) = region.as_object().cloned() else { continue };
            let text = map.get("text").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
            if !text.is_empty() {
                content_items.push(ContentItem::Text {
                    text,
                    text_level: 0,
                    content_list_index: index,
                    page_idx: -1,
                    metadata: map,
                });
                index += 1;
            }
        }

        // If no structured output, treat as single text block.
        if content_items.is_empty() {
            let all_text = root.get("text").and_then(|v| v.as_str()).unwrap_or("");
            if !all_text.trim().is_empty() {
                let mut metadata = JsonMap::new();
                metadata.insert("source".to_string(), JsonValue::from("paddleocr_raw"));
                content_items.push(ContentItem::Text {
                    text: all_text.to_string(),
                    text_level: 0,
                    content_list_index: 0,
                    page_idx: -1,
                    metadata,
                });
            }
        }

        content_items
    }
}

impl Default for PaddleOCRParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl MultimodalParser for PaddleOCRParser {
    fn name(&self) -> &str {
        "paddleocr"
    }

    fn supported_formats(&self) -> Vec<String> {
        let mut formats = Vec::new();
        for list in [PDF_FORMATS, IMAGE_FORMATS] {
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
            .join(format!("{stem}_paddleocr_output"));
        let output_dir = kwargs
            .get("output_dir")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| path_to_absolute(&default_output));

        tracing::info!(
            "Parsing {} with PaddleOCR (output={})",
            file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            output_dir
        );

        let command = self.build_command(file, &output_dir, kwargs);
        tracing::debug!("PaddleOCR command: {}", command.join(" "));

        self.execute_command(&command, file).await?;

        let content_list = self.parse_output(&output_dir, file);

        let mut metadata = JsonMap::new();
        metadata.insert("parser".to_string(), JsonValue::from("paddleocr"));
        metadata.insert("output_dir".to_string(), JsonValue::from(output_dir));

        Ok(ParseResult {
            content_list,
            doc_id: String::new(),
            file_path: path_to_absolute(file),
            metadata,
        })
    }

    fn check_installation(&self) -> bool {
        let ok = run_installation_check(
            &["python", "-c", "import paddleocr; print(paddleocr.__version__)"],
            15,
        );
        if !ok {
            tracing::debug!("PaddleOCR installation check failed");
        }
        ok
    }
}
