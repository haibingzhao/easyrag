//! Batch document parser with parallel processing

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use easyrag_core::JsonMap;
use md5::{Digest, Md5};
use serde_json::Value as JsonValue;

use super::super::model::BatchProcessingResult;
use super::super::util::{run_batch, BatchStats};
use super::{
    file_mtime_millis, path_to_absolute, walk_top_down, ExternalParserRegistry, MultimodalParser,
};

const MANIFEST_FILE_NAME: &str = ".raganything_batch_manifest.json";

/// Batch document parser with parallel processing capabilities.
/// Uses tokio task spawning + Semaphore for structured concurrency control.
pub struct BatchParser {
    parser: Arc<dyn MultimodalParser>,
    max_workers: usize,
    /// Held for API compatibility (`timeoutPerFile`); the parser instances from
    /// the registry carry their own timeouts.
    #[allow(dead_code)]
    timeout_per_file: u64,
}

/// Signature of a previously processed file for incremental runs.
#[derive(Debug, Clone)]
struct FileSignature {
    /// Persisted in the manifest though not consulted
    /// during filtering; the manifest map key already carries the path.
    #[allow(dead_code)]
    path: String,
    size: u64,
    mtime: i64,
    md5: Option<String>,
}

impl BatchParser {
    /// Create a batch parser for the given parser type.
    /// Errors on unknown parser types.
    pub fn new(
        parser_type: &str,
        max_workers: usize,
        timeout_per_file: u64,
        skip_installation_check: bool,
    ) -> easyrag_core::Result<Self> {
        let parser = ExternalParserRegistry::get_parser(parser_type)?;

        if !skip_installation_check && !parser.check_installation() {
            let mut chars = parser_type.chars();
            let display_name = match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
            };
            tracing::warn!(
                "{} parser installation check failed. \
                 This may be due to package conflicts. \
                 Use skipInstallationCheck=true to bypass this check.",
                display_name
            );
        }

        Ok(Self {
            parser,
            max_workers: max_workers.max(1),
            timeout_per_file,
        })
    }

    /// Get list of supported file extensions.
    pub fn get_supported_extensions(&self) -> Vec<String> {
        self.parser.supported_formats()
    }

    /// Filter file paths to only include supported file types.
    pub fn filter_supported_files(&self, file_paths: &[String], recursive: bool) -> Vec<PathBuf> {
        let supported_extensions = self.parser.supported_formats();
        let mut supported_files: Vec<PathBuf> = Vec::new();

        for path_str in file_paths {
            let path = Path::new(path_str);
            if path.is_file() {
                if is_supported_file(path, &supported_extensions) {
                    supported_files.push(path.to_path_buf());
                } else {
                    tracing::warn!("Unsupported file type: {}", path.display());
                }
            } else if path.is_dir() {
                let candidates: Vec<PathBuf> = if recursive {
                    walk_top_down(path).into_iter().filter(|p| p.is_file()).collect()
                } else {
                    std::fs::read_dir(path)
                        .map(|entries| {
                            entries
                                .filter_map(|e| e.ok())
                                .map(|e| e.path())
                                .filter(|p| p.is_file())
                                .collect()
                        })
                        .unwrap_or_default()
                };
                for candidate in candidates {
                    if is_supported_file(&candidate, &supported_extensions) {
                        supported_files.push(candidate);
                    }
                }
            } else {
                tracing::warn!("Path does not exist: {}", path.display());
            }
        }
        supported_files
    }

    /// Process multiple files in parallel using structured concurrency.
    #[allow(clippy::too_many_arguments)]
    pub async fn process_files(
        &self,
        file_paths: &[String],
        output_dir: &str,
        parse_method: &str,
        recursive: bool,
        dry_run: bool,
        incremental: bool,
        kwargs: &JsonMap,
    ) -> BatchProcessingResult {
        let start_time = Instant::now();

        let supported_files = self.filter_supported_files(file_paths, recursive);
        if supported_files.is_empty() {
            tracing::warn!("No supported files found to process");
            return BatchProcessingResult::default();
        }

        tracing::info!("Found {} files to process", supported_files.len());

        // Create output directory.
        let _ = std::fs::create_dir_all(output_dir);

        // Incremental processing: filter unchanged files.
        let mut files_to_process = supported_files.clone();
        let mut skipped_files: Vec<PathBuf> = Vec::new();

        if incremental {
            let manifest = load_incremental_manifest(output_dir);
            let (changed, unchanged) = filter_incremental_files(&supported_files, &manifest);
            tracing::info!(
                "Incremental scan: {} changed files, {} unchanged files",
                changed.len(),
                unchanged.len()
            );
            files_to_process = changed;
            skipped_files = unchanged;
        }

        if dry_run {
            tracing::info!(
                "Dry run enabled. {} files would be processed.",
                files_to_process.len()
            );
            return BatchProcessingResult {
                successful_files: files_to_process
                    .iter()
                    .map(|f| path_to_absolute(f))
                    .collect(),
                failed_files: Vec::new(),
                total_files: supported_files.len(),
                processing_time_ms: 0,
                errors: HashMap::new(),
                output_dir: String::new(),
                dry_run: false,
                skipped_files: skipped_files.iter().map(|f| path_to_absolute(f)).collect(),
            };
        }

        // Process files with semaphore-controlled concurrency (shared run_batch).
        let parser = Arc::clone(&self.parser);
        let output_dir_str = output_dir.to_string();
        let parse_method_str = parse_method.to_string();
        let kwargs_base = kwargs.clone();
        let stats = run_batch(
            files_to_process.clone(),
            self.max_workers,
            |file: &PathBuf| path_to_absolute(file),
            move |file: PathBuf, abs_path: String| {
                let parser = Arc::clone(&parser);
                let output_dir = output_dir_str.clone();
                let parse_method = parse_method_str.clone();
                let mut parse_kwargs = kwargs_base.clone();
                async move {
                    let stem = file
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_default();
                    let file_output_dir = Path::new(&output_dir).join(stem);
                    if let Err(e) = std::fs::create_dir_all(&file_output_dir) {
                        return Err(format!("Failed to process {abs_path}: mkdir failed: {e}"));
                    }

                    parse_kwargs.insert(
                        "output_dir".to_string(),
                        JsonValue::from(path_to_absolute(&file_output_dir)),
                    );

                    match parser.parse(&file, Some(&parse_method), &parse_kwargs).await {
                        Ok(_) => {
                            tracing::info!(
                                "Successfully processed {}",
                                file.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_default()
                            );
                            Ok(())
                        }
                        Err(e) => Err(format!("Failed to process {abs_path}: {e}")),
                    }
                }
            },
        )
        .await;

        let BatchStats {
            successful,
            failed,
            errors,
        } = stats;

        // Save incremental manifest for successful files.
        if incremental && !successful.is_empty() {
            save_incremental_manifest(output_dir, &successful);
        }

        let processing_time = start_time.elapsed().as_millis() as u64;

        let result = BatchProcessingResult {
            successful_files: successful,
            failed_files: failed,
            total_files: supported_files.len(),
            processing_time_ms: processing_time,
            errors,
            output_dir: String::new(),
            dry_run: false,
            skipped_files: skipped_files.iter().map(|f| path_to_absolute(f)).collect(),
        };

        tracing::info!("{}", result.summary());
        result
    }
}

fn is_supported_file(path: &Path, supported_extensions: &[String]) -> bool {
    let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
        return false;
    };
    supported_extensions.iter().any(|f| *f == format!(".{}", ext.to_lowercase()))
}

// --- Incremental processing support ---

fn load_incremental_manifest(output_dir: &str) -> HashMap<String, FileSignature> {
    let manifest_file = Path::new(output_dir).join(MANIFEST_FILE_NAME);
    if !manifest_file.exists() {
        return HashMap::new();
    }

    let parsed = std::fs::read_to_string(&manifest_file)
        .ok()
        .and_then(|text| serde_json::from_str::<JsonMap>(&text).ok());
    let Some(data) = parsed else {
        tracing::warn!("Could not read incremental manifest");
        return HashMap::new();
    };

    let Some(files) = data.get("files").and_then(|v| v.as_object()) else {
        return HashMap::new();
    };

    files
        .iter()
        .filter_map(|(key, value)| {
            let obj = value.as_object()?;
            Some((
                key.clone(),
                FileSignature {
                    path: obj
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    size: obj
                        .get("size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0),
                    mtime: obj.get("mtime").and_then(|v| v.as_i64()).unwrap_or(0),
                    md5: obj
                        .get("md5")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string()),
                },
            ))
        })
        .collect()
}

fn save_incremental_manifest(output_dir: &str, successful_files: &[String]) {
    let manifest_file = Path::new(output_dir).join(MANIFEST_FILE_NAME);

    let mut files = JsonMap::new();
    for path_str in successful_files {
        let file = Path::new(path_str);
        let mut entry = JsonMap::new();
        entry.insert("path".to_string(), JsonValue::from(path_to_absolute(file)));
        entry.insert(
            "size".to_string(),
            JsonValue::from(std::fs::metadata(file).map(|m| m.len()).unwrap_or(0) as i64),
        );
        entry.insert(
            "mtime".to_string(),
            JsonValue::from(file_mtime_millis(file)),
        );
        entry.insert("md5".to_string(), JsonValue::from(compute_file_md5(file)));
        files.insert(path_str.clone(), JsonValue::Object(entry));
    }

    let mut payload = JsonMap::new();
    payload.insert("version".to_string(), JsonValue::from(1));
    payload.insert(
        "updated_at".to_string(),
        JsonValue::from(chrono::Utc::now().timestamp_millis()),
    );
    payload.insert("files".to_string(), JsonValue::Object(files));

    match serde_json::to_string_pretty(&JsonValue::Object(payload)) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&manifest_file, text) {
                tracing::warn!("Could not save incremental manifest: {e}");
            }
        }
        Err(e) => tracing::warn!("Could not save incremental manifest: {e}"),
    }
}

fn filter_incremental_files(
    files: &[PathBuf],
    manifest: &HashMap<String, FileSignature>,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut changed = Vec::new();
    let mut unchanged = Vec::new();

    for file in files {
        let abs_path = path_to_absolute(file);
        let previous = manifest.get(&abs_path);
        let size = std::fs::metadata(file).map(|m| m.len()).unwrap_or(0);
        let mtime = file_mtime_millis(file);

        if let Some(prev) = previous {
            if prev.size == size && prev.mtime == mtime && prev.md5.is_some() {
                // Fast path: size and mtime match, trust stored signature.
                unchanged.push(file.clone());
                continue;
            }
        }

        // Metadata differs or new file: hash to compare content.
        let current_md5 = compute_file_md5(file);
        if previous.is_some_and(|prev| prev.md5.as_deref() == Some(current_md5.as_str())) {
            unchanged.push(file.clone());
        } else {
            changed.push(file.clone());
        }
    }
    (changed, unchanged)
}

fn compute_file_md5(file: &Path) -> String {
    let Ok(mut input) = std::fs::File::open(file) else {
        return String::new();
    };
    let mut hasher = Md5::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    use std::io::Read;
    loop {
        match input.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => hasher.update(&buffer[..n]),
            Err(_) => return String::new(),
        }
    }
    format!("{:x}", hasher.finalize())
}
