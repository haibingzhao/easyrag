//! Evaluation endpoints for anti-hallucination guardrail assessment.

use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::model::evaluation::GoldenItem;
use easyrag_core::model::QueryParam;
use easyrag_pipeline::evaluation::{GoldenSetEvaluator, Queryable};

use crate::error::AppResult;
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/eval/golden-set", post(eval_golden_set))
        .route("/api/eval/risk-coverage", post(eval_risk_coverage))
}

/// Request body for golden set evaluation.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoldenSetRequest {
    /// The golden set items to evaluate.
    pub items: Vec<GoldenItem>,
    /// Optional query parameters override (mode, top_k, etc.).
    #[serde(default)]
    pub query_param: Option<QueryParam>,
    /// Business line ID.
    #[serde(default = "default_biz_id")]
    pub biz_id: String,
}

fn default_biz_id() -> String {
    "default".to_string()
}

/// Request body for risk-coverage evaluation.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RiskCoverageRequest {
    /// The golden set items to evaluate.
    pub items: Vec<GoldenItem>,
    /// Optional query parameters override.
    #[serde(default)]
    pub query_param: Option<QueryParam>,
    /// Business line ID.
    #[serde(default = "default_biz_id")]
    pub biz_id: String,
}

/// Adapter to use EasyRAG as a Queryable for evaluation.
struct EasyRAGQueryable {
    easy_rag: Arc<easyrag_pipeline::EasyRAG>,
    biz_id: String,
}

#[async_trait::async_trait]
impl Queryable for EasyRAGQueryable {
    async fn eval_query(&self, query: &str, param: &QueryParam) -> easyrag_core::Result<String> {
        let mut eval_param = param.clone();
        eval_param.biz_id = self.biz_id.clone();
        self.easy_rag.query(query, &eval_param).await
    }
}

/// Maximum number of items allowed in a golden set evaluation request.
const MAX_GOLDEN_SET_SIZE: usize = 500;

/// POST /api/eval/golden-set
///
/// Evaluate a golden set of queries against the RAG pipeline and return
/// a report with confusion matrix, hallucination rate, coverage, and
/// faithfulness metrics.
async fn eval_golden_set(
    state: AppState,
    Json(request): Json<GoldenSetRequest>,
) -> AppResult<Json<Value>> {
    if request.items.is_empty() {
        return Err(crate::error::AppError::Core(
            easyrag_core::CoreError::InvalidArgument(
                "Golden set must contain at least one item".to_string(),
            ),
        ));
    }
    if request.items.len() > MAX_GOLDEN_SET_SIZE {
        return Err(crate::error::AppError::Core(
            easyrag_core::CoreError::InvalidArgument(format!(
                "Golden set exceeds maximum size of {MAX_GOLDEN_SET_SIZE} items"
            )),
        ));
    }

    let param = request.query_param.unwrap_or_default();
    let queryable = Arc::new(EasyRAGQueryable {
        easy_rag: Arc::clone(&state.easy_rag),
        biz_id: request.biz_id.clone(),
    });

    let evaluator = GoldenSetEvaluator::new(queryable, param);
    let report = evaluator.evaluate(&request.items).await?;

    Ok(Json(json!({
        "confusion_matrix": {
            "true_positive": report.confusion_matrix.true_positive,
            "false_positive": report.confusion_matrix.false_positive,
            "true_negative": report.confusion_matrix.true_negative,
            "false_negative": report.confusion_matrix.false_negative,
            "total": report.confusion_matrix.total(),
            "precision": report.confusion_matrix.precision(),
            "recall": report.confusion_matrix.recall(),
            "f1": report.confusion_matrix.f1(),
        },
        "hallucination_rate": report.hallucination_rate,
        "coverage": report.coverage,
        "faithfulness": report.faithfulness,
        "item_count": request.items.len(),
        "item_results": report.item_results.iter().map(|r| json!({
            "query": r.query,
            "abstained": r.abstained,
            "correct": r.correct,
            "classification": r.classification,
        })).collect::<Vec<_>>(),
    })))
}

/// POST /api/eval/risk-coverage
///
/// Compute the risk-coverage curve for a golden set. This shows the
/// tradeoff between coverage (fraction of queries answered) and risk
/// (fraction of answers that are incorrect).
async fn eval_risk_coverage(
    state: AppState,
    Json(request): Json<RiskCoverageRequest>,
) -> AppResult<Json<Value>> {
    if request.items.is_empty() {
        return Err(crate::error::AppError::Core(
            easyrag_core::CoreError::InvalidArgument(
                "Golden set must contain at least one item".to_string(),
            ),
        ));
    }
    if request.items.len() > MAX_GOLDEN_SET_SIZE {
        return Err(crate::error::AppError::Core(
            easyrag_core::CoreError::InvalidArgument(format!(
                "Golden set exceeds maximum size of {MAX_GOLDEN_SET_SIZE} items"
            )),
        ));
    }

    let param = request.query_param.unwrap_or_default();
    let queryable = Arc::new(EasyRAGQueryable {
        easy_rag: Arc::clone(&state.easy_rag),
        biz_id: request.biz_id.clone(),
    });

    let evaluator = GoldenSetEvaluator::new(queryable, param);
    let report = evaluator.evaluate(&request.items).await?;

    let curve: Vec<Value> = report
        .risk_coverage_curve
        .iter()
        .map(|p| {
            json!({
                "threshold": p.threshold,
                "coverage": p.coverage,
                "risk": p.risk,
            })
        })
        .collect();

    Ok(Json(json!({
        "risk_coverage_curve": curve,
        "summary": {
            "total_items": request.items.len(),
            "coverage_at_default": report.coverage,
            "hallucination_rate": report.hallucination_rate,
        },
    })))
}
