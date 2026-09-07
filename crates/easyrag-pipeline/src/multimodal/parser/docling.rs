//! Docling document parser.
//! Invokes the `docling` CLI tool.

use std::path::Path;

use async_trait::async_trait;
use easyrag_core::JsonMap;

use super::super::model::{ContentItem, ParseResult};
use super::{
    first_json_file, path_to_absolute, walk_top_down, ExternalCliParser, MultimodalParser,
    IMAGE_FORMATS, OFFICE_FORMATS, PDF_FORMATS, TEXT_FORMATS,
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

    pub(super) fn build_command(&self, file: &Path, output_dir: &str, kwargs: &JsonMap) -> Vec<String> {
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

impl ExternalCliParser for DoclingParser {
    const DISPLAY_NAME: &'static str = "Docling";
    const INSTALL_CHECK: &'static [&'static str] = &["docling", "--version"];
    const INSTALL_CHECK_TIMEOUT: u64 = 10;
    const FORMAT_LISTS: &'static [&'static [&'static str]] =
        &[PDF_FORMATS, OFFICE_FORMATS, IMAGE_FORMATS, TEXT_FORMATS];

    fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds
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
