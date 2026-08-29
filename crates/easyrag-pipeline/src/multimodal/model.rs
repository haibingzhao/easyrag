//! Multimodal content models.

use easyrag_core::JsonMap;
use serde_json::Value as JsonValue;

/// Modal content type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModalType {
    Text,
    Image,
    Table,
    Equation,
    Generic,
}

impl ModalType {
    pub fn value(&self) -> &'static str {
        match self {
            ModalType::Text => "text",
            ModalType::Image => "image",
            ModalType::Table => "table",
            ModalType::Equation => "equation",
            ModalType::Generic => "generic",
        }
    }

    /// Case-insensitive lookup; unknown values fall back to `Generic`
    ///
    pub fn from_value(value: &str) -> Self {
        let lower = value.to_lowercase();
        match lower.as_str() {
            "text" => ModalType::Text,
            "image" => ModalType::Image,
            "table" => ModalType::Table,
            "equation" => ModalType::Equation,
            _ => ModalType::Generic,
        }
    }
}

/// A content item from document parsing. Each variant corresponds to a
/// modality in the parsed content list.
#[derive(Debug, Clone)]
pub enum ContentItem {
    Text {
        text: String,
        text_level: i64,
        content_list_index: i64,
        page_idx: i64,
        metadata: JsonMap,
    },
    Image {
        img_path: String,
        captions: Vec<String>,
        footnotes: Vec<String>,
        section_path: String,
        neighbor_text: String,
        content_list_index: i64,
        page_idx: i64,
        metadata: JsonMap,
    },
    Table {
        table_body: String,
        table_img_path: String,
        captions: Vec<String>,
        footnotes: Vec<String>,
        content_list_index: i64,
        page_idx: i64,
        metadata: JsonMap,
    },
    Equation {
        equation_text: String,
        equation_format: String,
        content_list_index: i64,
        page_idx: i64,
        metadata: JsonMap,
    },
    Generic {
        content: String,
        content_type: String,
        content_list_index: i64,
        page_idx: i64,
        metadata: JsonMap,
    },
}

impl ContentItem {
    /// Original index in the content list.
    pub fn content_list_index(&self) -> i64 {
        match self {
            ContentItem::Text { content_list_index, .. }
            | ContentItem::Image { content_list_index, .. }
            | ContentItem::Table { content_list_index, .. }
            | ContentItem::Equation { content_list_index, .. }
            | ContentItem::Generic { content_list_index, .. } => *content_list_index,
        }
    }

    /// Page index in the source document.
    pub fn page_idx(&self) -> i64 {
        match self {
            ContentItem::Text { page_idx, .. }
            | ContentItem::Image { page_idx, .. }
            | ContentItem::Table { page_idx, .. }
            | ContentItem::Equation { page_idx, .. }
            | ContentItem::Generic { page_idx, .. } => *page_idx,
        }
    }

    /// Raw metadata from the parser.
    pub fn metadata(&self) -> &JsonMap {
        match self {
            ContentItem::Text { metadata, .. }
            | ContentItem::Image { metadata, .. }
            | ContentItem::Table { metadata, .. }
            | ContentItem::Equation { metadata, .. }
            | ContentItem::Generic { metadata, .. } => metadata,
        }
    }

    /// Modal type of this content item.
    pub fn modal_type(&self) -> ModalType {
        match self {
            ContentItem::Text { .. } => ModalType::Text,
            ContentItem::Image { .. } => ModalType::Image,
            ContentItem::Table { .. } => ModalType::Table,
            ContentItem::Equation { .. } => ModalType::Equation,
            ContentItem::Generic { .. } => ModalType::Generic,
        }
    }

    /// Factory method to create a ContentItem from a raw map (parser output).
    /// Handles field aliases from MinerU, Docling, and PaddleOCR parsers.
    pub fn from_raw_map(map: JsonMap, index: i64) -> Self {
        let type_str = map
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("text")
            .to_lowercase();
        let page_idx = map
            .get("page_idx")
            .and_then(|v| v.as_i64())
            .or_else(|| map.get("page_index").and_then(|v| v.as_i64()))
            .unwrap_or(-1);

        match type_str.as_str() {
            "text" => ContentItem::Text {
                text: map
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string(),
                text_level: map
                    .get("text_level")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                content_list_index: index,
                page_idx,
                metadata: map,
            },
            "image" => {
                let img_path = map
                    .get("img_path")
                    .and_then(|v| v.as_str())
                    .or_else(|| map.get("image_path").and_then(|v| v.as_str()))
                    .or_else(|| map.get("path").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                let captions = normalize_caption_list(
                    map.get("img_caption").or_else(|| map.get("captions")),
                );
                let footnotes = normalize_caption_list(
                    map.get("img_footnote").or_else(|| map.get("footnotes")),
                );
                ContentItem::Image {
                    img_path,
                    captions,
                    footnotes,
                    section_path: map
                        .get("_section_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    neighbor_text: map
                        .get("_neighbor_text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    content_list_index: index,
                    page_idx,
                    metadata: map,
                }
            }
            "table" => {
                let table_body = get_table_body(&map);
                let table_img_path = map
                    .get("img_path")
                    .and_then(|v| v.as_str())
                    .or_else(|| map.get("table_img_path").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                let captions = normalize_caption_list(
                    map.get("table_caption").or_else(|| map.get("captions")),
                );
                let footnotes = normalize_caption_list(
                    map.get("table_footnote").or_else(|| map.get("footnotes")),
                );
                ContentItem::Table {
                    table_body,
                    table_img_path,
                    captions,
                    footnotes,
                    content_list_index: index,
                    page_idx,
                    metadata: map,
                }
            }
            "equation" => {
                let (equation_text, equation_format) = get_equation_text_and_format(&map);
                ContentItem::Equation {
                    equation_text,
                    equation_format,
                    content_list_index: index,
                    page_idx,
                    metadata: map,
                }
            }
            _ => {
                let content = map
                    .get("text")
                    .and_then(|v| v.as_str())
                    .or_else(|| map.get("content").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                ContentItem::Generic {
                    content,
                    content_type: type_str,
                    content_list_index: index,
                    page_idx,
                    metadata: map,
                }
            }
        }
    }
}

/// Normalize captions/footnotes to a clean list of strings.
fn normalize_caption_list(value: Option<&JsonValue>) -> Vec<String> {
    match value {
        None => Vec::new(),
        Some(JsonValue::Array(items)) => items
            .iter()
            .filter_map(|v| {
                let s = match v {
                    JsonValue::String(s) => s.trim().to_string(),
                    JsonValue::Null => return None,
                    other => other.to_string(),
                };
                if s.is_empty() {
                    None
                } else {
                    Some(s)
                }
            })
            .collect(),
        Some(JsonValue::String(s)) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Vec::new()
            } else {
                vec![trimmed.to_string()]
            }
        }
        Some(_) => Vec::new(),
    }
}

/// Read table content across common content-list alias fields.
fn get_table_body(item: &JsonMap) -> String {
    if let Some(table_body) = item.get("table_body") {
        if !json_value_to_display(table_body).trim().is_empty() {
            return format_table_body(table_body);
        }
    }
    if let Some(table_data) = item.get("table_data") {
        if !json_value_to_display(table_data).trim().is_empty() {
            return format_table_body(table_data);
        }
    }
    item.get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn json_value_to_display(value: &JsonValue) -> String {
    match value {
        JsonValue::String(s) => s.clone(),
        JsonValue::Null => String::new(),
        other => other.to_string(),
    }
}

/// Format table body for prompts - handles String and List-of-Lists.
/// List-of-Lists becomes a Markdown table with a `| --- |` separator row
/// inserted after the first row.
fn format_table_body(table_body: &JsonValue) -> String {
    match table_body {
        JsonValue::String(s) => s.clone(),
        JsonValue::Array(rows) => {
            if rows.is_empty() {
                return String::new();
            }
            let all_rows_are_lists = rows.iter().all(|r| r.is_array());
            if all_rows_are_lists {
                let rendered: Vec<String> = rows
                    .iter()
                    .map(|row| {
                        let cells = row
                            .as_array()
                            .map(|cells| {
                                cells
                                    .iter()
                                    .map(|cell| match cell {
                                        JsonValue::Null => String::new(),
                                        JsonValue::String(s) => s.clone(),
                                        other => other.to_string(),
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        format!("| {} |", cells.join(" | "))
                    })
                    .collect();
                let column_count = rows
                    .iter()
                    .map(|r| r.as_array().map(|a| a.len()).unwrap_or(0))
                    .max()
                    .unwrap_or(0);
                let separator = format!(
                    "| {} |",
                    vec!["---"; column_count].join(" | ")
                );
                let mut out = Vec::with_capacity(rendered.len() + 1);
                out.push(rendered[0].clone());
                out.push(separator);
                out.extend(rendered.into_iter().skip(1));
                out.join("\n")
            } else {
                rows.iter()
                    .map(|v| match v {
                        JsonValue::Null => String::new(),
                        JsonValue::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        JsonValue::Null => String::new(),
        other => other.to_string(),
    }
}

/// Read equation content preserving LaTeX aliases.
fn get_equation_text_and_format(item: &JsonMap) -> (String, String) {
    let text = item
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let latex = item
        .get("latex")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let equation = item
        .get("equation")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let mut format = item
        .get("text_format")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();

    let equation_text = if !text.is_empty() {
        text
    } else if !latex.is_empty() {
        if format.is_empty() {
            format = "latex".to_string();
        }
        latex
    } else if !equation.is_empty() {
        equation
    } else {
        String::new()
    };
    (equation_text, format)
}

/// Result of document parsing - contains the full content list and metadata.
#[derive(Debug, Clone)]
pub struct ParseResult {
    pub content_list: Vec<ContentItem>,
    pub doc_id: String,
    pub file_path: String,
    pub metadata: JsonMap,
}

impl ParseResult {
    pub fn new(content_list: Vec<ContentItem>, file_path: String, metadata: JsonMap) -> Self {
        Self {
            content_list,
            doc_id: String::new(),
            file_path,
            metadata,
        }
    }

    /// Get only text content items.
    pub fn text_items(&self) -> Vec<&ContentItem> {
        self.content_list
            .iter()
            .filter(|i| matches!(i, ContentItem::Text { .. }))
            .collect()
    }

    /// Get only multimodal (non-text) content items.
    pub fn multimodal_items(&self) -> Vec<&ContentItem> {
        self.content_list
            .iter()
            .filter(|i| !matches!(i, ContentItem::Text { .. }))
            .collect()
    }

    /// Get merged text content as a single string.
    pub fn merged_text(&self) -> String {
        self.content_list
            .iter()
            .filter_map(|item| match item {
                ContentItem::Text { text, .. } if !text.trim().is_empty() => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// Result of modal processing - entity and chunk information for graph
/// insertion.
#[derive(Debug, Clone)]
pub struct ModalProcessResult {
    pub entity_name: String,
    pub entity_type: String,
    pub description: String,
    pub chunk_content: String,
    pub metadata: JsonMap,
    pub success: bool,
    pub error_message: Option<String>,
}

impl ModalProcessResult {
    pub fn failure(error_message: impl Into<String>) -> Self {
        Self {
            entity_name: String::new(),
            entity_type: String::new(),
            description: String::new(),
            chunk_content: String::new(),
            metadata: JsonMap::new(),
            success: false,
            error_message: Some(error_message.into()),
        }
    }
}

/// Result of batch processing operation.
#[derive(Debug, Clone, Default)]
pub struct BatchProcessingResult {
    pub successful_files: Vec<String>,
    pub failed_files: Vec<String>,
    pub total_files: usize,
    pub processing_time_ms: u64,
    pub errors: std::collections::HashMap<String, String>,
    pub output_dir: String,
    pub dry_run: bool,
    pub skipped_files: Vec<String>,
}

impl BatchProcessingResult {
    /// Success rate counts skipped files as successful.
    pub fn success_rate(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            ((self.successful_files.len() + self.skipped_files.len()) as f64
                / self.total_files as f64)
                * 100.0
        }
    }

    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("Batch Processing Summary:\n");
        out.push_str(&format!("  Total files: {}\n", self.total_files));
        out.push_str(&format!("  Successful: {}\n", self.successful_files.len()));
        out.push_str(&format!("  Failed: {}\n", self.failed_files.len()));
        out.push_str(&format!("  Skipped: {}\n", self.skipped_files.len()));
        out.push_str(&format!("  Success rate: {:.1}%\n", self.success_rate()));
        out.push_str(&format!(
            "  Processing time: {}s\n",
            self.processing_time_ms as f64 / 1000.0
        ));
        if !self.output_dir.is_empty() {
            out.push_str(&format!("  Output directory: {}\n", self.output_dir));
        }
        if self.dry_run {
            out.push_str("  Dry run: true\n");
        }
        out
    }
}
