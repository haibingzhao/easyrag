//! Shared directory scanning logic used by both the HTTP
//! `/api/documents/scan` endpoint and the background auto-scanner.

use crate::ingest::IngestionPipeline;
use easyrag_core::storage::traits::DocStatusStorage;
use easyrag_core::util::hash::compute_args_hash;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Result of one directory scan cycle.
#[derive(Clone, Debug, Default)]
pub struct ScanOutcome {
    /// Files matched the extension filter (capped at max_files).
    pub files_seen: usize,
    /// Newly inserted document IDs (new or changed files).
    pub inserted_doc_ids: Vec<String>,
    /// Files skipped because an identical content_hash is already ingested.
    pub skipped_duplicate: usize,
}

/// Collect files recursively, capped at max_files.
pub fn collect_files(dir: &Path, extensions: &[String], max_files: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            if files.len() >= max_files {
                return files;
            }
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if extensions.iter().any(|ext| name.ends_with(ext)) {
                    files.push(path);
                }
            }
        }
    }
    files
}

/// Scan `dir` for matching files and insert new/changed ones as pending
/// documents. Deduplication: a file whose path already exists with an
/// identical content_hash is skipped; changed content is re-inserted
///
pub async fn scan_directory(
    pipeline: &Arc<IngestionPipeline>,
    doc_status: &Arc<dyn DocStatusStorage>,
    dir: &Path,
    extensions: &[String],
    max_files: usize,
    biz_id: &str,
) -> easyrag_core::Result<ScanOutcome> {
    let files = collect_files(dir, extensions, max_files);
    let mut outcome = ScanOutcome {
        files_seen: files.len(),
        ..Default::default()
    };
    for file in &files {
        // Skip unreadable files.
        let Ok(content) = std::fs::read_to_string(file) else {
            continue;
        };
        let absolute = file
            .canonicalize()
            .unwrap_or_else(|_| file.to_path_buf())
            .to_string_lossy()
            .to_string();
        let content_hash = compute_args_hash(std::iter::once(&content));
        // Incremental dedup: same path + same hash -> already ingested.
        if let Some(existing) = doc_status.get_doc_by_file_path(&absolute, biz_id).await? {
            let existing_hash = existing
                .get("content_hash")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if existing_hash == content_hash {
                outcome.skipped_duplicate += 1;
                continue;
            }
            // Content changed: fall through and re-insert the new version.
        }
        if let Ok(doc_id) = pipeline
            .insert_document(&content, &absolute, None, None, biz_id)
            .await
        {
            outcome.inserted_doc_ids.push(doc_id);
        }
    }
    Ok(outcome)
}
