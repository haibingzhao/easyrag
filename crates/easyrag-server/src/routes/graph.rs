//! Graph CRUD endpoints.
//!
//! P1.12: entity/relation creation also writes to vector storage
//! for searchability (best-effort, failures only logged).
//!
//! All handlers route through the workspace resolver so requests can target
//! a specific (workspace, biz_id) scope; omitted values resolve to defaults.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path as AxumPath, Query};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use easyrag_core::storage::traits::StorageBundle;
use easyrag_core::JsonMap;

use crate::error::AppResult;
use crate::state::{AppServices, AppState};

pub fn router() -> Router<Arc<AppServices>> {
    Router::new()
        .route("/api/graph/labels", get(all_labels))
        .route("/api/graph/search", get(search_labels))
        .route("/api/graph/knowledge", get(knowledge_graph))
        .route("/api/graph/node/{nodeId}", get(get_node))
        .route("/api/graph/stats", get(stats))
        .route("/api/graph/entity/exists", get(entity_exists))
        .route("/api/graph/entity/create", post(create_entity))
        .route("/api/graph/entity/edit", post(edit_entity))
        .route("/api/graph/entity/delete", delete(delete_entity))
        .route("/api/graph/entities/merge", post(merge_entities))
        .route("/api/graph/relation/create", post(create_relation))
        .route("/api/graph/relation/edit", post(edit_relation))
        .route("/api/graph/relation/delete", delete(delete_relation))
}

/// Resolve the storage bundle for a request-level (workspace, biz_id) scope.
/// Returns the bundle plus the normalized biz_id that every storage call
/// must carry as its filter value.
async fn scoped_bundle(
    state: &AppServices,
    workspace: Option<&str>,
    biz_id: Option<&str>,
) -> AppResult<(StorageBundle, String)> {
    let resolved = state.workspace_resolver.resolve(workspace, biz_id).await?;
    Ok((resolved.context.bundle, resolved.biz_id))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateEntityRequest {
    name: String,
    #[serde(default)]
    entity_type: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EditEntityRequest {
    original_name: String,
    #[serde(default)]
    new_name: Option<String>,
    #[serde(default)]
    entity_type: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MergeEntitiesRequest {
    entities: Vec<String>,
    #[serde(default)]
    target_name: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CreateRelationRequest {
    source: String,
    target: String,
    #[serde(default)]
    keywords: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EditRelationRequest {
    source: String,
    target: String,
    #[serde(default)]
    keywords: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    weight: Option<f32>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LabelsQuery {
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SearchQuery {
    query: String,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct KnowledgeQuery {
    #[serde(default)]
    node_label: Option<String>,
    #[serde(default)]
    max_depth: Option<usize>,
    #[serde(default)]
    max_nodes: Option<usize>,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EntityNameQuery {
    name: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RelationQuery {
    source: String,
    target: String,
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

/// Scope-only query for handlers whose meaningful params live in the path.
#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ScopeQuery {
    #[serde(default)]
    workspace: Option<String>,
    #[serde(default)]
    biz_id: Option<String>,
}

async fn all_labels(state: AppState, Query(q): Query<LabelsQuery>) -> AppResult<Json<Value>> {
    let limit = q.limit.unwrap_or(300);
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let labels = bundle.graph_storage.get_popular_labels(limit, &biz_id).await?;
    let total = labels.len();
    Ok(Json(json!({ "labels": labels, "total": total })))
}

async fn search_labels(state: AppState, Query(q): Query<SearchQuery>) -> AppResult<Json<Value>> {
    let limit = q.limit.unwrap_or(50);
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let labels = bundle.graph_storage.search_labels(&q.query, limit, &biz_id).await?;
    let total = labels.len();
    Ok(Json(json!({ "labels": labels, "total": total })))
}

async fn knowledge_graph(
    state: AppState,
    Query(q): Query<KnowledgeQuery>,
) -> AppResult<Json<Value>> {
    let node_label = q.node_label.unwrap_or_else(|| "*".to_string());
    let max_depth = q.max_depth.unwrap_or(3);
    let max_nodes = q.max_nodes.unwrap_or(1000);
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let kg = bundle
        .graph_storage
        .get_knowledge_graph(Some(&node_label), max_depth, max_nodes, &biz_id)
        .await?;
    let nodes: Vec<Value> = kg
        .nodes
        .iter()
        .map(|n| {
            json!({
                "id": n.id,
                "labels": n.labels,
                "properties": n.properties,
            })
        })
        .collect();
    let edges: Vec<Value> = kg
        .edges
        .iter()
        .map(|e| {
            json!({
                "source": e.source,
                "target": e.target,
                "properties": e.properties,
            })
        })
        .collect();
    Ok(Json(json!({
        "nodes": nodes,
        "edges": edges,
        "is_truncated": kg.is_truncated,
    })))
}

async fn get_node(
    state: AppState,
    AxumPath(node_id): AxumPath<String>,
    Query(q): Query<ScopeQuery>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let graph = &bundle.graph_storage;
    let Some(node) = graph.get_node(&node_id, &biz_id).await? else {
        return Ok(Json(json!({ "error": "Node not found" })));
    };
    let edges = graph.get_node_edges(&node_id, &biz_id).await?.unwrap_or_default();
    let degree = graph.node_degree(&node_id, &biz_id).await?;
    let edge_list: Vec<Value> = edges
        .iter()
        .map(|(src, tgt)| json!({ "source": src, "target": tgt }))
        .collect();
    Ok(Json(json!({
        "id": node_id,
        "properties": node,
        "degree": degree,
        "edges": edge_list,
    })))
}

async fn stats(state: AppState, Query(q): Query<ScopeQuery>) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let all_nodes = bundle.graph_storage.get_all_nodes(&biz_id).await?;
    let all_edges = bundle.graph_storage.get_all_edges(&biz_id).await?;
    Ok(Json(json!({
        "node_count": all_nodes.len(),
        "edge_count": all_edges.len(),
    })))
}

async fn entity_exists(
    state: AppState,
    Query(q): Query<EntityNameQuery>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let exists = bundle.graph_storage.has_node(&q.name, &biz_id).await?;
    Ok(Json(json!({ "name": q.name, "exists": exists })))
}

async fn create_entity(
    state: AppState,
    Json(request): Json<CreateEntityRequest>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(
        &state,
        request.workspace.as_deref(),
        request.biz_id.as_deref(),
    )
    .await?;
    let graph = &bundle.graph_storage;
    if graph.has_node(&request.name, &biz_id).await? {
        return Ok(Json(json!({
            "status": "error",
            "message": format!("Entity already exists: {}", request.name),
        })));
    }
    let entity_type = request.entity_type.unwrap_or_else(|| "Other".to_string());
    let description = request.description.unwrap_or_default();
    graph
        .upsert_node(
            &request.name,
            json!({
                "entity_type": entity_type,
                "description": description,
                "source_id": "manual",
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
            &biz_id,
        )
        .await?;
    // P1.12: write entity to vector storage for searchability (best-effort).
    let mut vector_data: HashMap<String, JsonMap> = HashMap::new();
    vector_data.insert(
        request.name.clone(),
        json!({
            "entity_name": request.name,
            "entity_type": entity_type,
            "description": description,
            "source_id": "manual",
            "content": format!("{}: {}", request.name, description),
        })
        .as_object()
        .cloned()
        .unwrap_or_default(),
    );
    if let Err(e) = bundle.vector_storage.upsert(vector_data, &biz_id).await {
        tracing::warn!("Failed to write entity to VDB: {e}");
    }
    Ok(Json(json!({
        "status": "success",
        "message": format!("Entity created: {}", request.name),
    })))
}

async fn edit_entity(
    state: AppState,
    Json(request): Json<EditEntityRequest>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(
        &state,
        request.workspace.as_deref(),
        request.biz_id.as_deref(),
    )
    .await?;
    let graph = &bundle.graph_storage;
    let Some(existing) = graph.get_node(&request.original_name, &biz_id).await? else {
        return Ok(Json(json!({
            "status": "error",
            "message": format!("Entity not found: {}", request.original_name),
        })));
    };
    let mut updated_data = existing;
    if let Some(new_name) = &request.new_name {
        updated_data.insert("entity_name".into(), json!(new_name));
    }
    if let Some(entity_type) = &request.entity_type {
        updated_data.insert("entity_type".into(), json!(entity_type));
    }
    if let Some(description) = &request.description {
        updated_data.insert("description".into(), json!(description));
    }

    // If renaming, delete old node and create the new one.
    let target_name = request
        .new_name
        .clone()
        .unwrap_or_else(|| request.original_name.clone());
    if let Some(new_name) = &request.new_name {
        if new_name != &request.original_name {
            // Reconnect edges to the new name.
            let edges = graph
                .get_node_edges(&request.original_name, &biz_id)
                .await?
                .unwrap_or_default();
            for (src, tgt) in edges {
                if let Some(edge_data) = graph.get_edge(&src, &tgt, &biz_id).await? {
                    let new_src = if src == request.original_name {
                        target_name.clone()
                    } else {
                        src.clone()
                    };
                    let new_tgt = if tgt == request.original_name {
                        target_name.clone()
                    } else {
                        tgt.clone()
                    };
                    graph.upsert_edge(&new_src, &new_tgt, edge_data, &biz_id).await?;
                    graph.remove_edges(&[(src, tgt)], &biz_id).await?;
                }
            }
            graph.delete_node(&request.original_name, &biz_id).await?;
        }
    }

    graph.upsert_node(&target_name, updated_data, &biz_id).await?;
    Ok(Json(json!({
        "status": "success",
        "message": format!("Entity updated: {target_name}"),
    })))
}

async fn delete_entity(
    state: AppState,
    Query(q): Query<EntityNameQuery>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let graph = &bundle.graph_storage;
    if !graph.has_node(&q.name, &biz_id).await? {
        return Ok(Json(json!({
            "status": "error",
            "message": format!("Entity not found: {}", q.name),
        })));
    }
    // Remove all edges connected to this node first.
    let edges = graph.get_node_edges(&q.name, &biz_id).await?.unwrap_or_default();
    if !edges.is_empty() {
        graph.remove_edges(&edges, &biz_id).await?;
    }
    graph.delete_node(&q.name, &biz_id).await?;
    Ok(Json(json!({
        "status": "success",
        "message": format!("Entity deleted: {}", q.name),
        "edges_removed": edges.len(),
    })))
}

async fn merge_entities(
    state: AppState,
    Json(request): Json<MergeEntitiesRequest>,
) -> AppResult<Json<Value>> {
    if request.entities.len() < 2 {
        return Ok(Json(json!({
            "status": "error",
            "message": "Need at least 2 entities to merge",
        })));
    }
    let (bundle, biz_id) = scoped_bundle(
        &state,
        request.workspace.as_deref(),
        request.biz_id.as_deref(),
    )
    .await?;
    let graph = &bundle.graph_storage;
    let target_name = request
        .target_name
        .clone()
        .unwrap_or_else(|| request.entities[0].clone());
    let mut descriptions: Vec<String> = Vec::new();
    let mut entity_type = "Other".to_string();

    // Collect data from all entities.
    for entity_name in &request.entities {
        if let Some(node) = graph.get_node(entity_name, &biz_id).await? {
            if let Some(desc) = node.get("description").and_then(Value::as_str) {
                descriptions.push(desc.to_string());
            }
            if let Some(t) = node.get("entity_type").and_then(Value::as_str) {
                entity_type = t.to_string();
            }
        }
    }

    // Create merged entity.
    graph
        .upsert_node(
            &target_name,
            json!({
                "entity_type": entity_type,
                "description": descriptions.join("\n"),
                "source_id": "merge",
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
            &biz_id,
        )
        .await?;

    // Redirect edges from source entities to target.
    for entity_name in &request.entities {
        if entity_name == &target_name {
            continue;
        }
        let edges = graph.get_node_edges(entity_name, &biz_id).await?.unwrap_or_default();
        for (src, tgt) in edges {
            if let Some(edge_data) = graph.get_edge(&src, &tgt, &biz_id).await? {
                let new_src = if src == *entity_name {
                    target_name.clone()
                } else {
                    src.clone()
                };
                let new_tgt = if tgt == *entity_name {
                    target_name.clone()
                } else {
                    tgt.clone()
                };
                if new_src != new_tgt {
                    // Avoid self-loops.
                    graph.upsert_edge(&new_src, &new_tgt, edge_data, &biz_id).await?;
                }
                graph.remove_edges(&[(src, tgt)], &biz_id).await?;
            }
        }
        graph.delete_node(entity_name, &biz_id).await?;
    }

    Ok(Json(json!({
        "status": "success",
        "message": format!("Merged {} entities into: {target_name}", request.entities.len()),
    })))
}

async fn create_relation(
    state: AppState,
    Json(request): Json<CreateRelationRequest>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(
        &state,
        request.workspace.as_deref(),
        request.biz_id.as_deref(),
    )
    .await?;
    let graph = &bundle.graph_storage;
    if !graph.has_node(&request.source, &biz_id).await?
        || !graph.has_node(&request.target, &biz_id).await?
    {
        return Ok(Json(json!({
            "status": "error",
            "message": "Source or target entity not found",
        })));
    }
    let keywords = request.keywords.unwrap_or_default();
    let description = request.description.unwrap_or_default();
    graph
        .upsert_edge(
            &request.source,
            &request.target,
            json!({
                "keywords": keywords,
                "description": description,
                "source_id": "manual",
                "weight": 1.0,
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
            &biz_id,
        )
        .await?;
    // P1.12: write relation to vector storage for searchability (best-effort).
    let relation_id = format!("{}<SEP>{}", request.source, request.target);
    let mut vector_data: HashMap<String, JsonMap> = HashMap::new();
    vector_data.insert(
        relation_id.clone(),
        json!({
            "entity_name": relation_id,
            "entity_type": "relationship",
            "description": description,
            "keywords": keywords,
            "source_id": "manual",
            "content": format!("{} -> {}: {} {}", request.source, request.target, description, keywords),
        })
        .as_object()
        .cloned()
        .unwrap_or_default(),
    );
    if let Err(e) = bundle.vector_storage.upsert(vector_data, &biz_id).await {
        tracing::warn!("Failed to write relation to VDB: {e}");
    }
    Ok(Json(json!({
        "status": "success",
        "message": format!("Relation created: {} <-> {}", request.source, request.target),
    })))
}

async fn edit_relation(
    state: AppState,
    Json(request): Json<EditRelationRequest>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(
        &state,
        request.workspace.as_deref(),
        request.biz_id.as_deref(),
    )
    .await?;
    let graph = &bundle.graph_storage;
    let Some(existing) = graph.get_edge(&request.source, &request.target, &biz_id).await? else {
        return Ok(Json(json!({ "status": "error", "message": "Relation not found" })));
    };
    let mut updated_data = existing;
    if let Some(keywords) = &request.keywords {
        updated_data.insert("keywords".into(), json!(keywords));
    }
    if let Some(description) = &request.description {
        updated_data.insert("description".into(), json!(description));
    }
    if let Some(weight) = request.weight {
        updated_data.insert("weight".into(), json!(weight));
    }
    graph
        .upsert_edge(&request.source, &request.target, updated_data, &biz_id)
        .await?;
    Ok(Json(json!({
        "status": "success",
        "message": format!("Relation updated: {} <-> {}", request.source, request.target),
    })))
}

async fn delete_relation(
    state: AppState,
    Query(q): Query<RelationQuery>,
) -> AppResult<Json<Value>> {
    let (bundle, biz_id) = scoped_bundle(&state, q.workspace.as_deref(), q.biz_id.as_deref()).await?;
    let graph = &bundle.graph_storage;
    if !graph.has_edge(&q.source, &q.target, &biz_id).await? {
        return Ok(Json(json!({ "status": "error", "message": "Relation not found" })));
    }
    graph
        .remove_edges(&[(q.source.clone(), q.target.clone())], &biz_id)
        .await?;
    Ok(Json(json!({
        "status": "success",
        "message": format!("Relation deleted: {} <-> {}", q.source, q.target),
    })))
}
