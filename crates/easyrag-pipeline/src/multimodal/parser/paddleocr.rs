//! PaddleOCR document parser.
//! Invokes the `paddleocr` Python module CLI for OCR-based parsing.

use std::path::Path;

use async_trait::async_trait;
use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

use super::super::model::{ContentItem, ParseResult};
use super::{
    first_json_file, path_to_absolute, walk_top_down, ExternalCliParser, MultimodalParser,
    IMAGE_FORMATS, PDF_FORMATS,
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

    pub(super) fn build_command(&self, file: &Path, output_dir: &str, kwargs: &JsonMap) -> Vec<String> {
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

    pub(super) fn parse_output(&self, output_dir: &str, _original_file: &Path) -> Vec<ContentItem> {
        if !self.output_dir_exists(output_dir) {
            return Vec::new();
        }

        let Some(json_file) = first_json_file(&walk_top_down(Path::new(output_dir))) else {
            return Vec::new();
        };

        let Some(root) = self.read_output_json(&json_file) else {
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

impl ExternalCliParser for PaddleOCRParser {
    const DISPLAY_NAME: &'static str = "PaddleOCR";
    const INSTALL_CHECK: &'static [&'static str] =
        &["python", "-c", "import paddleocr; print(paddleocr.__version__)"];
    const INSTALL_CHECK_TIMEOUT: u64 = 15;
    const FORMAT_LISTS: &'static [&'static [&'static str]] = &[PDF_FORMATS, IMAGE_FORMATS];

    fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds
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
        self.supported_format_list()
    }

    async fn parse(
        &self,
        file: &Path,
        _parse_method: Option<&str>,
        kwargs: &JsonMap,
    ) -> easyrag_core::Result<ParseResult> {
        let output_dir = self.resolve_output_dir(file, kwargs)?;

        tracing::info!(
            "Parsing {} with {} (output={})",
            file.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
            Self::DISPLAY_NAME,
            output_dir
        );

        let command = self.build_command(file, &output_dir, kwargs);
        tracing::debug!("{} command: {}", Self::DISPLAY_NAME, command.join(" "));

        self.execute_command(&command, file).await?;

        let content_list = self.parse_output(&output_dir, file);

        Ok(self.cli_parse_result(content_list, file, &output_dir, None))
    }

    fn check_installation(&self) -> bool {
        self.installation_ok()
    }
}
