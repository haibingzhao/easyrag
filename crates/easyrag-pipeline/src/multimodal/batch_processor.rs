//! Batch processor for folder-level document processing
//!
//! Uses tokio tasks + Semaphore for structured concurrency.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use easyrag_core::{CoreError, JsonMap};
use serde_json::Value as JsonValue;
use tokio::sync::{Mutex, Semaphore};

use super::config::RagAnythingConfig;
use super::document_processor::{absolute_path_string, DocumentProcessor};
use super::model::BatchProcessingResult;
use super::parser::walk_top_down;

/// Batch processor for folder-level document processing.
pub struct BatchProcessor {
    config: RagAnythingConfig,
    document_processor: Arc<DocumentProcessor>,
}

impl BatchProcessor {
    pub fn new(config: RagAnythingConfig, document_processor: Arc<DocumentProcessor>) -> Self {
        Self {
            config,
            document_processor,
        }
    }

    /// Process all supported files in a folder.
    /// `file_extensions`: extensions to include (empty = use config defaults).
    #[allow(clippy::too_many_arguments)]
    pub async fn process_folder(
        &self,
        folder_path: &str,
        output_dir: Option<&str>,
        parse_method: Option<&str>,
        file_extensions: Option<&HashSet<String>>,
        recursive: bool,
        max_workers: usize,
    ) -> easyrag_core::Result<BatchProcessingResult> {
        let start_time = Instant::now();
        let folder = Path::new(folder_path);

        if !folder.exists() || !folder.is_dir() {
            return Err(CoreError::InvalidArgument(format!(
                "Folder not found or not a directory: {folder_path}"
            )));
        }

        // Discover files
        let extensions: HashSet<String> = file_extensions
            .cloned()
            .unwrap_or_else(|| self.config.supported_file_extensions.iter().cloned().collect());
        let files = discover_files(folder, &extensions, recursive);

        if files.is_empty() {
            tracing::warn!("No supported files found in: {}", folder_path);
            return Ok(BatchProcessingResult::default());
        }

        tracing::info!("Found {} files to process in {}", files.len(), folder_path);

        let effective_output_dir = output_dir
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.config.parser_output_dir.clone());
        let _ = std::fs::create_dir_all(&effective_output_dir);

        // Process files with semaphore-controlled concurrency
        let semaphore = Arc::new(Semaphore::new(max_workers.max(1)));
        let successful_files: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let failed_files: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let errors: Arc<Mutex<HashMap<String, String>>> = Arc::new(Mutex::new(HashMap::new()));

        let mut handles: Vec<(String, tokio::task::JoinHandle<()>)> = Vec::new();
        for file in &files {
            let permit = semaphore.clone().acquire_owned();
            let document_processor = Arc::clone(&self.document_processor);
            let successful = successful_files.clone();
            let failed = failed_files.clone();
            let errors = errors.clone();
            let effective_output_dir = effective_output_dir.clone();
            let parse_method = parse_method.map(|s| s.to_string());
            let file = file.clone();

            handles.push((
                absolute_path_string(&file.to_string_lossy()),
                tokio::spawn(async move {
                    let Ok(permit) = permit.await else {
                        return;
                    };
                    let _permit = permit;
                    let abs_path = absolute_path_string(&file.to_string_lossy());

                    let stem = file
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let file_output_dir = Path::new(&effective_output_dir).join(stem);

                    let mut kwargs = JsonMap::new();
                    kwargs.insert(
                        "output_dir".to_string(),
                        JsonValue::from(absolute_path_string(
                            &file_output_dir.to_string_lossy(),
                        )),
                    );
                    if let Some(method) = parse_method.as_deref() {
                        kwargs.insert("parse_method".to_string(), JsonValue::from(method));
                    }

                    match document_processor
                        .process_document_complete(&abs_path, &kwargs)
                        .await
                    {
                        Ok(_) => {
                            successful.lock().await.push(abs_path.clone());
                            tracing::info!(
                                "Successfully processed: {}",
                                file.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_default()
                            );
                        }
                        Err(e) => {
                            let error_msg = format!("Failed to process {abs_path}: {e}");
                            tracing::error!("{error_msg}");
                            failed.lock().await.push(abs_path.clone());
                            errors.lock().await.insert(abs_path, error_msg);
                        }
                    }
                }),
            ));
        }

        for (abs_path, handle) in handles {
            // A panicked/cancelled task must not vanish from the stats.
            if let Err(e) = handle.await {
                let error_msg = format!("Failed to process {abs_path}: task aborted: {e}");
                tracing::error!("{error_msg}");
                failed_files.lock().await.push(abs_path.clone());
                errors.lock().await.insert(abs_path, error_msg);
            }
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        let result = BatchProcessingResult {
            successful_files: successful_files.lock().await.clone(),
            failed_files: failed_files.lock().await.clone(),
            total_files: files.len(),
            processing_time_ms,
            errors: errors.lock().await.clone(),
            output_dir: String::new(),
            dry_run: false,
            skipped_files: Vec::new(),
        };

        tracing::info!("{}", result.summary());
        Ok(result)
    }
}

/// Discover supported files in a directory
///, sorted by name).
fn discover_files(folder: &Path, extensions: &HashSet<String>, recursive: bool) -> Vec<PathBuf> {
    let normalized: HashSet<String> = extensions
        .iter()
        .map(|ext| {
            let lower = ext.to_lowercase();
            if lower.starts_with('.') {
                lower
            } else {
                format!(".{lower}")
            }
        })
        .collect();

    let mut files: Vec<PathBuf> = if recursive {
        walk_top_down(folder)
    } else {
        //: folder itself + direct children
        let mut out = vec![folder.to_path_buf()];
        if let Ok(entries) = std::fs::read_dir(folder) {
            for entry in entries.flatten() {
                out.push(entry.path());
            }
        }
        out
    }
    .into_iter()
    .filter(|p| {
        p.is_file()
            && p.extension()
                .and_then(|e| e.to_str())
                .map(|e| normalized.contains(&format!(".{}", e.to_lowercase())))
                .unwrap_or(false)
    })
    .collect();

    files.sort_by(|a, b| {
        let an = a.file_name().map(|n| n.to_string_lossy().to_string());
        let bn = b.file_name().map(|n| n.to_string_lossy().to_string());
        an.cmp(&bn)
    });
    files
}
