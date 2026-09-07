//! Utility functions for RAGAnything multimodal processing

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use base64::Engine;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinHandle;

use super::model::{ContentItem, ModalType};

const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
];

/// First `n` chars of `s` (char-wise, not byte-wise).
///
/// Single source of truth for the four byte-identical copies previously
/// inlined in `processor`, `multimodal_query_engine`, `document_processor`
/// and `context`.
pub(crate) fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Aggregated statistics produced by [`run_batch`].
pub(crate) struct BatchStats {
    pub successful: Vec<String>,
    pub failed: Vec<String>,
    pub errors: HashMap<String, String>,
}

/// Run `f` over `items` with semaphore-bounded concurrency via tokio tasks.
///
/// This is the single source of truth for the two formerly duplicated batch
/// loops (`parser::batch` and `batch_processor`). It preserves their exact
/// structured-concurrency contract:
///
/// * one `Arc<Semaphore>` sized `max_workers.max(1)`, one `acquire_owned` permit
///   per task, awaited at task start and held (`_permit`) for the whole body;
/// * a join fallback that records a panicked/cancelled task as failed with the
///   `"task aborted: {e}"` message so it never vanishes from the stats;
/// * identical stat recording — success pushes the key, failure logs
///   `tracing::error!("{error_msg}")` then pushes the key and inserts the error.
///
/// `key` derives the absolute-path stat key from an item (computed once before
/// spawn, reused by the panic fallback). `f` receives the owned item plus that
/// key, performs the per-item work (including its own success logging, which
/// differs between callers) and returns `Ok(())` or `Err(error_msg)`. Generic
/// with `Fn` bounds so each caller monomorphizes to zero-cost code (no dyn/Box).
pub(crate) async fn run_batch<I, K, F, Fut>(
    items: I,
    max_workers: usize,
    key: K,
    f: F,
) -> BatchStats
where
    I: IntoIterator,
    I::Item: Send + 'static,
    K: Fn(&I::Item) -> String + Send + Sync + 'static,
    F: Fn(I::Item, String) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<(), String>> + Send + 'static,
{
    let semaphore = Arc::new(Semaphore::new(max_workers.max(1)));
    let successful: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let failed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let errors: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

    let f = Arc::new(f);

    let mut handles: Vec<(String, JoinHandle<()>)> = Vec::new();
    for item in items {
        let permit = semaphore.clone().acquire_owned();
        let abs_path = key(&item);
        let successful = Arc::clone(&successful);
        let failed = Arc::clone(&failed);
        let errors = Arc::clone(&errors);
        let f = Arc::clone(&f);
        let task_path = abs_path.clone();

        handles.push((
            abs_path,
            tokio::spawn(async move {
                let Ok(permit) = permit.await else {
                    return;
                };
                let _permit = permit;
                match f(item, task_path.clone()).await {
                    Ok(()) => {
                        successful.lock().await.push(task_path);
                    }
                    Err(error_msg) => {
                        tracing::error!("{error_msg}");
                        failed.lock().await.push(task_path.clone());
                        errors.lock().await.insert(task_path, error_msg);
                    }
                }
            }),
        ));
    }

    for (abs_path, handle) in handles {
        // A panicked/cancelled task must not vanish from the stats:
        // count it as failed so totals stay consistent.
        if let Err(e) = handle.await {
            let error_msg = format!("Failed to process {abs_path}: task aborted: {e}");
            tracing::error!("{error_msg}");
            failed.lock().await.push(abs_path.clone());
            errors.lock().await.insert(abs_path, error_msg);
        }
    }

    let successful = successful.lock().await.clone();
    let failed = failed.lock().await.clone();
    let errors = errors.lock().await.clone();
    BatchStats {
        successful,
        failed,
        errors,
    }
}

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
