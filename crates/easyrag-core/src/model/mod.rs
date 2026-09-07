//! Domain models.

pub mod doc_status;
pub mod document_structure;
pub mod evaluation;
pub mod query_param;
pub mod query_result;
pub mod scan_task;
pub mod scope;
pub mod source_id;
pub mod types;
pub mod verification;

pub use doc_status::{
    DeletionResult, DeletionStatus, DocProcessingStatus, DocStatus, StoragesStatus,
};
pub use document_structure::{DocumentStructureIndex, SectionNode};
pub use evaluation::{
    ConfusionMatrix, EvalReport, GoldenItem, GoldenItemResult, RiskCoveragePoint,
};
pub use query_param::{QueryParam, QueryMode};
pub use query_result::{QueryContextResult, QueryResult, TextChunk};
pub use scan_task::{ScanCycleStats, ScanPosition, ScanTaskKind, ScanTaskSpec};
pub use scope::{AsTenantScope, TenantScope, TenantScopeRef};
pub use source_id::{
    merge_source_ids, parse_source_ids, source_id_intersects, source_ids_intersect,
    source_ids_value,
};
pub use types::{
    EntityExtractionResult, ExtractedEntity, ExtractedRelationship, KnowledgeGraph,
    KnowledgeGraphEdge, KnowledgeGraphNode,
};
pub use verification::{
    CitationCheckResult, ClaimVerification, EvidenceQualityScore, QualityDecision,
    QueryRoute, QueryRouteType, VerificationResult,
};
