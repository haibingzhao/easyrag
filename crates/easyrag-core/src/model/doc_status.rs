//! Document processing status models.

use serde::{Deserialize, Serialize};

/// Document processing status.
/// Pipeline order: PENDING -> PARSING -> ANALYZING (optional) -> PROCESSING
/// -> PROCESSED | FAILED.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocStatus {
    Pending,
    Parsing,
    Analyzing,
    Processing,
    Preprocessed,
    Processed,
    Failed,
}

impl DocStatus {
    pub fn value(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Parsing => "parsing",
            Self::Analyzing => "analyzing",
            Self::Processing => "processing",
            Self::Preprocessed => "preprocessed",
            Self::Processed => "processed",
            Self::Failed => "failed",
        }
    }

    pub fn from_value(value: &str) -> Result<Self, crate::CoreError> {
        match value {
            "pending" => Ok(Self::Pending),
            "parsing" => Ok(Self::Parsing),
            "analyzing" => Ok(Self::Analyzing),
            "processing" => Ok(Self::Processing),
            "preprocessed" => Ok(Self::Preprocessed),
            "processed" => Ok(Self::Processed),
            "failed" => Ok(Self::Failed),
            other => Err(crate::CoreError::InvalidArgument(format!(
                "Unknown DocStatus: {other}"
            ))),
        }
    }
}

impl std::fmt::Display for DocStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// Storages status
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoragesStatus {
    NotCreated,
    Created,
    Initialized,
    Finalized,
}

impl StoragesStatus {
    pub fn value(&self) -> &'static str {
        match self {
            Self::NotCreated => "not_created",
            Self::Created => "created",
            Self::Initialized => "initialized",
            Self::Finalized => "finalized",
        }
    }

    pub fn from_value(value: &str) -> Result<Self, crate::CoreError> {
        match value {
            "not_created" => Ok(Self::NotCreated),
            "created" => Ok(Self::Created),
            "initialized" => Ok(Self::Initialized),
            "finalized" => Ok(Self::Finalized),
            other => Err(crate::CoreError::InvalidArgument(format!(
                "Unknown StoragesStatus: {other}"
            ))),
        }
    }
}

/// Document processing status data structure.
/// Field names serialized snake_case (wire-format contract).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocProcessingStatus {
    pub content_summary: String,
    pub content_length: i64,
    pub file_path: String,
    pub status: DocStatus,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunks_count: Option<i64>,
    #[serde(default)]
    pub chunks_list: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_msg: Option<String>,
    #[serde(default)]
    pub metadata: crate::JsonMap,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_processed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_hash: Option<String>,
    #[serde(default)]
    pub has_structure: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
}

impl DocProcessingStatus {
    /// Resolve effective status considering multimodal_processed flag.
    pub fn effective_status(&self) -> DocStatus {
        if self.multimodal_processed == Some(false) && self.status == DocStatus::Processed {
            DocStatus::Preprocessed
        } else {
            self.status
        }
    }
}

/// Represents the result of a deletion operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeletionResult {
    pub status: DeletionStatus,
    pub doc_id: String,
    pub message: String,
    #[serde(default = "default_status_code")]
    pub status_code: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
}

fn default_status_code() -> u16 {
    200
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeletionStatus {
    Success,
    NotFound,
    NotAllowed,
    Fail,
}

impl DeletionStatus {
    pub fn value(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::NotFound => "not_found",
            Self::NotAllowed => "not_allowed",
            Self::Fail => "fail",
        }
    }
}
