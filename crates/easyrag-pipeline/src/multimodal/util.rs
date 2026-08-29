//! Utility functions for RAGAnything multimodal processing

use std::collections::HashMap;
use std::path::Path;

use base64::Engine;

use super::model::{ContentItem, ModalType};

const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
];

/// Separate text content and multimodal content from a content list.
/// Returns `(text_content, multimodal_items)`.
pub fn separate_content(content_list: Vec<ContentItem>) -> (String, Vec<ContentItem>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut multimodal_items: Vec<ContentItem> = Vec::new();

    for item in content_list {
        match &item {
            ContentItem::Text { text, .. } => {
                if !text.trim().is_empty() {
                    text_parts.push(text.clone());
                }
            }
            _ => multimodal_items.push(item),
        }
    }

    let text_content = text_parts.join("\n\n");

    tracing::info!("Content separation complete:");
    tracing::info!("  - Text content length: {} characters", text_content.len());
    tracing::info!("  - Multimodal items count: {}", multimodal_items.len());

    let mut modal_types: HashMap<&str, usize> = HashMap::new();
    for item in &multimodal_items {
        *modal_types.entry(item.modal_type().value()).or_insert(0) += 1;
    }
    if !modal_types.is_empty() {
        tracing::info!("  - Multimodal type distribution: {modal_types:?}");
    }

    (text_content, multimodal_items)
}

/// Build a hierarchical section path from preceding heading blocks.
/// Maintains a heading level stack; deeper headings nest under shallower
/// ones, joined by " > ".
pub fn extract_section_path(content_list: &[ContentItem], current_index: isize) -> String {
    if content_list.is_empty() || current_index < 0 {
        return String::new();
    }

    let limit = (current_index as usize).min(content_list.len());
    let mut heading_chain: Vec<(i64, String)> = Vec::new();

    for item in content_list.iter().take(limit) {
        let ContentItem::Text {
            text, text_level, ..
        } = item
        else {
            continue;
        };
        if text.trim().is_empty() || *text_level <= 0 {
            continue;
        }

        while heading_chain
            .last()
            .is_some_and(|(level, _)| level >= text_level)
        {
            heading_chain.pop();
        }
        heading_chain.push((*text_level, text.clone()));
    }

    heading_chain
        .iter()
        .map(|(_, title)| title.as_str())
        .collect::<Vec<_>>()
        .join(" > ")
}

/// Collect nearby text blocks around an item index from the content list.
pub fn extract_neighbor_text(
    content_list: &[ContentItem],
    current_index: isize,
    window_size: usize,
) -> String {
    if content_list.is_empty() || current_index < 0 || current_index as usize >= content_list.len()
    {
        return String::new();
    }

    let idx = current_index as usize;
    let start_idx = idx.saturating_sub(window_size);
    let end_idx = (idx + window_size + 1).min(content_list.len());

    let mut parts: Vec<String> = Vec::new();
    for pos in start_idx..end_idx {
        if pos == idx {
            continue;
        }
        if let ContentItem::Text { text, .. } = &content_list[pos] {
            if !text.trim().is_empty() {
                parts.push(text.clone());
            }
        }
    }

    parts.join(" ")
}

/// Encode image file to base64 string. Returns "" on any failure
///
pub fn encode_image_to_base64(image_path: &str) -> String {
    let path = Path::new(image_path);
    if !path.exists() {
        tracing::error!("Image file not found: {}", image_path);
        return String::new();
    }
    match std::fs::read(path) {
        Ok(bytes) => base64::engine::general_purpose::STANDARD.encode(bytes),
        Err(e) => {
            tracing::error!("Failed to encode image {}: {}", image_path, e);
            String::new()
        }
    }
}

/// Validate if a file is a valid image file (extension whitelist + size cap).
pub fn validate_image_file(image_path: &str, max_size_mb: u64) -> bool {
    let path = Path::new(image_path);

    if !path.exists() {
        tracing::warn!("Image file not found: {}", image_path);
        return false;
    }

    // Check file extension (against the lowercased absolute path).
    let path_lower = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf())
        .to_string_lossy()
        .to_lowercase();
    let has_valid_extension = IMAGE_EXTENSIONS.iter().any(|ext| path_lower.ends_with(ext));
    if !has_valid_extension {
        tracing::warn!("File does not appear to be an image: {}", image_path);
        return false;
    }

    // Check file size.
    let max_size = max_size_mb * 1024 * 1024;
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > max_size => {
            tracing::warn!(
                "Image file too large ({} bytes): {}",
                meta.len(),
                image_path
            );
            false
        }
        Err(e) => {
            tracing::error!("Error validating image file {}: {}", image_path, e);
            false
        }
        _ => true,
    }
}

/// Get appropriate processor type based on content type string.
pub fn get_processor_type(content_type: &str) -> ModalType {
    match content_type.to_lowercase().as_str() {
        "image" => ModalType::Image,
        "table" => ModalType::Table,
        "equation" => ModalType::Equation,
        _ => ModalType::Generic,
    }
}

/// Get processor supported features description.
pub fn get_processor_supports(proc_type: &str) -> Vec<&'static str> {
    match proc_type.to_lowercase().as_str() {
        "image" => vec![
            "Image content analysis",
            "Visual understanding",
            "Image description generation",
            "Image entity extraction",
        ],
        "table" => vec![
            "Table structure analysis",
            "Data statistics",
            "Trend identification",
            "Table entity extraction",
        ],
        "equation" => vec![
            "Mathematical formula parsing",
            "Variable identification",
            "Formula meaning explanation",
            "Formula entity extraction",
        ],
        "generic" => vec![
            "General content analysis",
            "Structured processing",
            "Entity extraction",
        ],
        _ => vec!["Basic processing"],
    }
}

/// Get file reference based on `use_full_path` configuration.
pub fn get_file_reference(file_path: &str, use_full_path: bool) -> String {
    if use_full_path {
        file_path.to_string()
    } else {
        Path::new(file_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| file_path.to_string())
    }
}
