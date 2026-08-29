//! In-memory execution stats for background auto-scan tasks (analog of
//! easy-rss `scheduler_stats.py`): the app-level scanner records each cycle
//! here and the `/api/documents/auto_scan_status` endpoint reads it back.

use std::collections::HashMap;
use std::sync::Mutex;

use easyrag_core::model::{ScanCycleStats, ScanTaskSpec};

/// Shared handle between the background scanner (writer) and the status
/// endpoint (reader).
#[derive(Default)]
pub struct AutoScanStats {
    tasks: Mutex<Vec<ScanTaskSpec>>,
    cycles: Mutex<HashMap<String, ScanCycleStats>>,
}

impl AutoScanStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the task list at scanner startup (drives the status endpoint).
    pub fn register_tasks(&self, tasks: Vec<ScanTaskSpec>) {
        *self.tasks.lock().unwrap() = tasks;
    }

    pub fn task_specs(&self) -> Vec<ScanTaskSpec> {
        self.tasks.lock().unwrap().clone()
    }

    /// Record the outcome of one scan cycle (latest wins per task).
    pub fn record(&self, stats: ScanCycleStats) {
        self.cycles
            .lock()
            .unwrap()
            .insert(stats.task_id.clone(), stats);
    }

    /// Latest cycle stats, sorted by task_id for stable responses.
    pub fn snapshot(&self) -> Vec<ScanCycleStats> {
        let mut cycles: Vec<ScanCycleStats> = self.cycles.lock().unwrap().values().cloned().collect();
        cycles.sort_by(|a, b| a.task_id.cmp(&b.task_id));
        cycles
    }
}
