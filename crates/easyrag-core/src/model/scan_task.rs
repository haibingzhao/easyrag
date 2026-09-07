//! Background auto-scan task models (inspired by easy-rss `rag_watermark`
//! + multi-task scheduler design).

use serde::{Deserialize, Serialize};

/// Kind of background scan task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanTaskKind {
    /// Poll pending documents of a workspace/biz scope and trigger the
    /// ingestion pipeline when any are found.
    PendingPoll,
    /// Incrementally scan a local input directory, insert new/changed files
    /// as pending documents and trigger the pipeline.
    DirScan,
}

impl ScanTaskKind {
    pub fn value(&self) -> &'static str {
        match self {
            Self::PendingPoll => "pending_poll",
            Self::DirScan => "dir_scan",
        }
    }

    pub fn from_value(value: &str) -> Result<Self, crate::CoreError> {
        match value {
            "pending_poll" => Ok(Self::PendingPoll),
            "dir_scan" => Ok(Self::DirScan),
            other => Err(crate::CoreError::InvalidArgument(format!(
                "Unknown ScanTaskKind: {other}"
            ))),
        }
    }
}

/// Declarative specification of one background scan task. Each task owns an
/// independent enable switch and interval (easy-rss multi-task pattern).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanTaskSpec {
    pub task_id: String,
    pub kind: ScanTaskKind,
    /// Target workspace (never empty; resolves through WorkspaceResolver).
    pub workspace: String,
    /// Business-line filter; empty string matches all biz lines of the workspace.
    #[serde(default)]
    pub biz_id: String,
    pub interval_secs: u64,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// DirScan only: directory to scan incrementally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_dir: Option<String>,
    /// DirScan only: file extensions to include (e.g. ".txt", ".md").
    #[serde(default)]
    pub extensions: Vec<String>,
    /// DirScan only: per-cycle file cap.
    #[serde(default = "default_max_files")]
    pub max_files_per_cycle: usize,
}

fn default_enabled() -> bool {
    true
}

fn default_max_files() -> usize {
    100
}

/// Persisted scan cursor of one task (equivalent to one `rag_watermark` row
/// in easy-rss). Stored via KVStorage under key `auto_scan:position:{task_id}`;
/// the watermark only moves forward.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScanPosition {
    pub task_id: String,
    /// Epoch milliseconds of the last completed scan cycle.
    #[serde(default)]
    pub watermark_epoch_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_run_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    #[serde(default)]
    pub docs_processed_total: u64,
    #[serde(default)]
    pub docs_inserted_total: u64,
    #[serde(default)]
    pub updated_at: String,
}

impl ScanPosition {
    /// KV storage key convention for this task's position row.
    pub fn storage_key(task_id: &str) -> String {
        format!("auto_scan:position:{task_id}")
    }
}

/// Outcome statistics of one scan cycle (in-memory, exposed by the status endpoint).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScanCycleStats {
    pub task_id: String,
    pub started_at: String,
    pub duration_ms: u64,
    /// Documents seen in the scan scope (pending count or files seen).
    pub docs_seen: usize,
    /// Documents the cycle acted on (pipeline triggered / newly inserted).
    pub docs_triggered_or_inserted: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
