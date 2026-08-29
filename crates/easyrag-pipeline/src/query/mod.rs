//! Query subsystem:
//!
//! - [`engine`]: [`QueryEngine`] — unified entry point for all query modes
//!   (bypass/naive/local/global/hybrid/mix/structure) with LRU query cache
//! - [`tree_search`]: [`TreeSearchEngine`] — multi-level LLM-guided tree
//!   navigation over a document's structure index
//! - [`structure`]: [`StructureQueryStrategy`] — dispatches between tree
//!   search and agentic retrieval, then generates the final answer
//! - [`agent`]: [`StructureAgentStrategy`] — PageIndex-style autonomous
//!   document exploration via tool calls
//! - [`citation_check`]: citation validity checking (Feature 3)
//! - [`evidence_scorer`]: CRAG-style evidence quality scoring (Feature 2)
//! - [`router`]: query routing and decomposition (Feature 6)
//! - [`verifier`]: claim verification gate (Feature 1)

pub mod agent;
pub mod citation_check;
pub mod engine;
pub mod evidence_scorer;
pub mod router;
pub mod structure;
pub mod tree_search;
pub mod verifier;

pub use agent::{AgentQueryResult, StructureAgentStrategy, ToolCallRecord};
pub use citation_check::{extract_citations, validate_citations};
pub use engine::{
    KeywordsResult, QueryDataResult, QueryEngine, QueryReference, QueryWithRefsResult,
};
pub use evidence_scorer::EvidenceScorer;
pub use router::QueryRouter;
pub use structure::{StructureQueryResult, StructureQueryStrategy};
pub use tree_search::{ReasoningStep, TreeSearchEngine, TreeSearchResult};
pub use verifier::ClaimVerifier;
