//! In-memory graph storage using adjacency list. All edge operations are undirected.

use async_trait::async_trait;
use easyrag_core::llm::EmbeddingFunc;
use easyrag_core::model::{
    source_ids_intersect, KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode,
};
use easyrag_core::storage::traits::{GraphStorage, StorageNameSpace};
use easyrag_core::JsonMap;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::drop_result::drop_success;

const EDGE_SEP: &str = "<SEP>";

/// Business-line tag stored inside each node/edge property map.
fn props_biz(props: &JsonMap) -> Option<&str> {
    props.get("biz_id").and_then(|v| v.as_str())
}

/// Read-path scope predicate for **collection** reads (list/count/search/
/// knowledge-graph). An empty `biz_id` is the wildcard that spans every biz
/// line of the workspace (BC-12/A-5), matching `PgGraphStorage`'s
/// `list_scope_clause`, `Neo4jGraphStorage`'s `node_scope` and the in-memory
/// doc-status backend's `biz_id.is_empty() || ...` filter. Single-record reads
/// (`get_node`/`has_node`/degrees) and every write keep the exact
/// `props_biz(props) == Some(biz_id)` filter, so a status write can never land
/// on the wrong biz line.
fn props_in_list_scope(props: &JsonMap, biz_id: &str) -> bool {
    biz_id.is_empty() || props_biz(props) == Some(biz_id)
}

/// In-memory graph storage. Edge keys are normalized alphabetically so that
/// (a, b) and (b, a) address the same undirected edge.
/// Data is shared across biz lines within a workspace; `biz_id` is a
/// per-record filter rather than instance state.
pub struct InMemoryGraphStorage {
    namespace: String,
    workspace: String,
    global_config: JsonMap,
    embedding_func: Option<EmbeddingFunc>,
    /// node_id -> node properties
    nodes: Mutex<HashMap<String, JsonMap>>,
    /// "src<SEP>tgt" -> edge properties (normalized: src < tgt alphabetically)
    edges: Mutex<HashMap<String, JsonMap>>,
    /// node_id -> set of connected node_ids
    adjacency: Mutex<HashMap<String, HashSet<String>>>,
}

impl InMemoryGraphStorage {
    pub fn new(namespace: impl Into<String>, workspace: impl Into<String>) -> Self {
        Self::with_config(namespace, workspace, JsonMap::new(), None)
    }

    pub fn with_config(
        namespace: impl Into<String>,
        workspace: impl Into<String>,
        global_config: JsonMap,
        embedding_func: Option<EmbeddingFunc>,
    ) -> Self {
        Self {
            namespace: namespace.into(),
            workspace: workspace.into(),
            global_config,
            embedding_func,
            nodes: Mutex::new(HashMap::new()),
            edges: Mutex::new(HashMap::new()),
            adjacency: Mutex::new(HashMap::new()),
        }
    }

    fn edge_key(src: &str, tgt: &str) -> String {
        let (a, b) = if src <= tgt { (src, tgt) } else { (tgt, src) };
        format!("{a}{EDGE_SEP}{b}")
    }

    /// Get total node count.
    pub fn node_count(&self) -> usize {
        self.nodes.lock().len()
    }

    /// Get total edge count.
    pub fn edge_count(&self) -> usize {
        self.edges.lock().len()
    }
}

#[async_trait]
impl StorageNameSpace for InMemoryGraphStorage {
    fn namespace(&self) -> &str {
        &self.namespace
    }

    fn workspace(&self) -> &str {
        &self.workspace
    }

    fn global_config(&self) -> &JsonMap {
        &self.global_config
    }

    async fn drop(&self, biz_id: &str) -> easyrag_core::Result<HashMap<String, String>> {
        let removed_nodes = {
            let mut nodes = self.nodes.lock();
            let before = nodes.len();
            nodes.retain(|_, props| props_biz(props) != Some(biz_id));
            before - nodes.len()
        };
        {
            let mut edges = self.edges.lock();
            edges.retain(|_, props| props_biz(props) != Some(biz_id));
        }
        // Adjacency is shared structural data; pruned lazily by reads.
        Ok(drop_success(format!(
            "data dropped ({removed_nodes} nodes for biz {biz_id})"
        )))
    }
}

#[async_trait]
impl GraphStorage for InMemoryGraphStorage {
    fn embedding_func(&self) -> Option<&EmbeddingFunc> {
        self.embedding_func.as_ref()
    }

    async fn has_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<bool> {
        Ok(self
            .nodes
            .lock()
            .get(node_id)
            .is_some_and(|props| props_biz(props) == Some(biz_id)))
    }

    async fn has_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<bool> {
        Ok(self
            .edges
            .lock()
            .get(&Self::edge_key(source_node_id, target_node_id))
            .is_some_and(|props| props_biz(props) == Some(biz_id)))
    }

    async fn node_degree(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        let nodes = self.nodes.lock();
        let adjacency = self.adjacency.lock();
        Ok(adjacency
            .get(node_id)
            .map(|s| {
                s.iter()
                    .filter(|n| {
                        nodes
                            .get(*n)
                            .is_some_and(|props| props_biz(props) == Some(biz_id))
                    })
                    .count()
            })
            .unwrap_or(0))
    }

    async fn edge_degree(&self, src_id: &str, tgt_id: &str, biz_id: &str) -> easyrag_core::Result<usize> {
        Ok(self.node_degree(src_id, biz_id).await? + self.node_degree(tgt_id, biz_id).await?)
    }

    async fn get_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .nodes
            .lock()
            .get(node_id)
            .filter(|props| props_biz(props) == Some(biz_id))
            .cloned())
    }

    async fn get_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<JsonMap>> {
        Ok(self
            .edges
            .lock()
            .get(&Self::edge_key(source_node_id, target_node_id))
            .filter(|props| props_biz(props) == Some(biz_id))
            .cloned())
    }

    async fn get_node_edges(
        &self,
        source_node_id: &str,
        biz_id: &str,
    ) -> easyrag_core::Result<Option<Vec<(String, String)>>> {
        let nodes = self.nodes.lock();
        if !nodes
            .get(source_node_id)
            .is_some_and(|props| props_biz(props) == Some(biz_id))
        {
            return Ok(None);
        }
        let adjacency = self.adjacency.lock();
        Ok(Some(
            adjacency
                .get(source_node_id)
                .map(|neighbors| {
                    neighbors
                        .iter()
                        .filter(|n| {
                            nodes
                                .get(*n)
                                .is_some_and(|props| props_biz(props) == Some(biz_id))
                        })
                        .map(|n| (source_node_id.to_string(), n.clone()))
                        .collect()
                })
                .unwrap_or_default(),
        ))
    }

    async fn upsert_node(
        &self,
        node_id: &str,
        node_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let mut nodes = self.nodes.lock();
        let entry = nodes.entry(node_id.to_string()).or_default();
        entry.extend(node_data);
        entry.insert("biz_id".to_string(), Value::String(biz_id.to_string()));
        self.adjacency
            .lock()
            .entry(node_id.to_string())
            .or_default();
        Ok(())
    }

    async fn upsert_edge(
        &self,
        source_node_id: &str,
        target_node_id: &str,
        edge_data: JsonMap,
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let key = Self::edge_key(source_node_id, target_node_id);
        {
            let mut edges = self.edges.lock();
            let existing = edges.entry(key).or_default();
            existing.extend(edge_data);
            existing.insert(
                "source".to_string(),
                serde_json::Value::String(source_node_id.to_string()),
            );
            existing.insert(
                "target".to_string(),
                serde_json::Value::String(target_node_id.to_string()),
            );
            existing.insert("biz_id".to_string(), Value::String(biz_id.to_string()));
        }

        // Update adjacency (undirected).
        {
            let mut adjacency = self.adjacency.lock();
            adjacency
                .entry(source_node_id.to_string())
                .or_default()
                .insert(target_node_id.to_string());
            adjacency
                .entry(target_node_id.to_string())
                .or_default()
                .insert(source_node_id.to_string());
        }
        // Ensure both nodes exist (tagged with this biz when freshly created).
        let mut nodes = self.nodes.lock();
        for id in [source_node_id, target_node_id] {
            let entry = nodes.entry(id.to_string()).or_default();
            entry
                .entry("biz_id".to_string())
                .or_insert(Value::String(biz_id.to_string()));
        }
        Ok(())
    }

    async fn delete_node(&self, node_id: &str, biz_id: &str) -> easyrag_core::Result<()> {
        {
            let mut nodes = self.nodes.lock();
            match nodes.get(node_id) {
                Some(props) if props_biz(props) == Some(biz_id) => {
                    nodes.remove(node_id);
                }
                _ => return Ok(()),
            }
        }
        // Remove all edges connected to this node.
        let connected = self.adjacency.lock().remove(node_id).unwrap_or_default();
        let mut edges = self.edges.lock();
        let mut adjacency = self.adjacency.lock();
        for other in connected {
            edges.remove(&Self::edge_key(node_id, &other));
            if let Some(set) = adjacency.get_mut(&other) {
                set.remove(node_id);
            }
        }
        Ok(())
    }

    async fn remove_nodes(&self, node_ids: &[String], biz_id: &str) -> easyrag_core::Result<()> {
        // One pass holding all three locks, in the order the rest of this file
        // already uses (nodes, edges, adjacency). Looping `delete_node`
        // reacquired and released them six times per node, and left the graph
        // observably half-deleted between two of those acquisitions.
        let mut nodes = self.nodes.lock();
        let mut edges = self.edges.lock();
        let mut adjacency = self.adjacency.lock();

        // Nodes on another biz line are left alone, exactly as `delete_node`
        // does, so the single and batch paths cannot disagree about scope.
        let doomed: Vec<&str> = node_ids
            .iter()
            .filter(|id| {
                nodes
                    .get(id.as_str())
                    .is_some_and(|props| props_biz(props) == Some(biz_id))
            })
            .map(|id| id.as_str())
            .collect();

        for node_id in doomed {
            nodes.remove(node_id);
            for other in adjacency.remove(node_id).unwrap_or_default() {
                edges.remove(&Self::edge_key(node_id, &other));
                if let Some(set) = adjacency.get_mut(&other) {
                    set.remove(node_id);
                }
            }
        }
        Ok(())
    }

    async fn remove_edges(
        &self,
        edge_list: &[(String, String)],
        biz_id: &str,
    ) -> easyrag_core::Result<()> {
        let mut edges = self.edges.lock();
        let mut adjacency = self.adjacency.lock();
        for (src, tgt) in edge_list {
            let key = Self::edge_key(src, tgt);
            if edges.get(&key).is_some_and(|props| props_biz(props) == Some(biz_id)) {
                edges.remove(&key);
                if let Some(set) = adjacency.get_mut(src) {
                    set.remove(tgt);
                }
                if let Some(set) = adjacency.get_mut(tgt) {
                    set.remove(src);
                }
            }
        }
        Ok(())
    }

    async fn get_all_labels(&self, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let mut labels: Vec<String> = self
            .nodes
            .lock()
            .iter()
            // EXACT scope, not the list wildcard: `PgGraphStorage::get_all_labels`
            // pins the exact biz line (locked by
            // `bc12_graph_single_record_reads_and_writes_stay_exact`), and no route
            // calls this with a blank biz_id, so memory matches postgres here.
            .filter(|(_, props)| props_biz(props) == Some(biz_id))
            .map(|(id, _)| id.clone())
            .collect();
        labels.sort();
        Ok(labels)
    }

    async fn get_knowledge_graph(
        &self,
        node_label: Option<&str>,
        max_depth: usize,
        max_nodes: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<KnowledgeGraph> {
        let node_label = node_label.unwrap_or("*");
        let mut graph_nodes: Vec<KnowledgeGraphNode> = Vec::new();
        let mut graph_edges: Vec<KnowledgeGraphEdge> = Vec::new();

        let nodes = self.nodes.lock();
        let edges = self.edges.lock();
        let adjacency = self.adjacency.lock();
        let in_biz = |props: &JsonMap| props_in_list_scope(props, biz_id);

        if node_label == "*" {
            // Return all nodes up to max_nodes.
            let visible: Vec<(&String, &JsonMap)> = nodes
                .iter()
                .filter(|(_, props)| in_biz(props))
                .take(max_nodes)
                .collect();
            for (id, props) in visible {
                graph_nodes.push(KnowledgeGraphNode::self_labeled(
                    id.clone(),
                    super::stringify_props(props),
                ));
            }
            for (key, props) in edges.iter().filter(|(_, props)| in_biz(props)) {
                let parts: Vec<&str> = key.split(EDGE_SEP).collect();
                if parts.len() == 2 {
                    graph_edges.push(KnowledgeGraphEdge::untyped(
                        parts[0],
                        parts[1],
                        super::stringify_props(props),
                    ));
                }
            }
            return Ok(KnowledgeGraph {
                nodes: graph_nodes,
                edges: graph_edges,
                // BC-1/A-6: `>=`, not `>`. A graph that exactly fills the
                // budget is reported truncated, matching this backend's BFS
                // path and the Postgres/Neo4j backends. The previous `>` made
                // the two paths of this same backend contradict each other.
                is_truncated: nodes.iter().filter(|(_, p)| in_biz(p)).count() >= max_nodes,
            });
        }

        // BFS from nodeLabel.
        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        let lower_label = node_label.to_lowercase();
        for (key, props) in nodes.iter() {
            if in_biz(props) && key.to_lowercase().contains(&lower_label) {
                queue.push_back((key.clone(), 0));
            }
        }

        while let Some((current, depth)) = queue.pop_front() {
            if visited.len() >= max_nodes {
                break;
            }
            if !visited.insert(current.clone()) {
                continue;
            }

            if let Some(props) = nodes.get(&current).filter(|p| in_biz(p)) {
                graph_nodes.push(KnowledgeGraphNode::self_labeled(
                    current.clone(),
                    super::stringify_props(props),
                ));
            }

            if depth < max_depth {
                let empty = HashSet::new();
                for neighbor in adjacency.get(&current).unwrap_or(&empty) {
                    if visited.contains(neighbor) {
                        continue;
                    }
                    if !nodes.get(neighbor).is_some_and(|p| in_biz(p)) {
                        continue;
                    }
                    queue.push_back((neighbor.clone(), depth + 1));
                    // Add edge.
                    let key = Self::edge_key(&current, neighbor);
                    if let Some(edge_props) = edges.get(&key).filter(|p| in_biz(p)) {
                        graph_edges.push(KnowledgeGraphEdge::untyped(
                            current.clone(),
                            neighbor.clone(),
                            super::stringify_props(edge_props),
                        ));
                    }
                }
            }
        }

        Ok(KnowledgeGraph {
            nodes: graph_nodes,
            edges: graph_edges,
            is_truncated: visited.len() >= max_nodes,
        })
    }

    async fn get_all_nodes(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        let nodes = self.nodes.lock();
        Ok(nodes
            .iter()
            .filter(|(_, props)| props_in_list_scope(props, biz_id))
            .map(|(id, props)| {
                let mut result = props.clone();
                result.insert("id".to_string(), serde_json::Value::String(id.clone()));
                result
            })
            .collect())
    }

    async fn get_all_edges(&self, biz_id: &str) -> easyrag_core::Result<Vec<JsonMap>> {
        Ok(self
            .edges
            .lock()
            .values()
            .filter(|props| props_in_list_scope(props, biz_id))
            .cloned()
            .collect())
    }

    async fn get_nodes_by_source_ids(
        &self,
        chunk_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let wanted: HashSet<String> = chunk_ids.iter().cloned().collect();
        // Filtering happens under the lock, before any clone: the trait default
        // clones every property map in the graph to hand back the handful that
        // match, which is pure waste here where the maps are already local.
        let nodes = self.nodes.lock();
        Ok(nodes
            .iter()
            .filter(|(_, props)| {
                props_in_list_scope(props, biz_id) && source_ids_intersect(props, &wanted)
            })
            .map(|(id, props)| {
                let mut result = props.clone();
                result.insert("id".to_string(), serde_json::Value::String(id.clone()));
                result
            })
            .collect())
    }

    async fn get_edges_by_source_ids(
        &self,
        chunk_ids: &[String],
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<JsonMap>> {
        if chunk_ids.is_empty() {
            return Ok(Vec::new());
        }
        let wanted: HashSet<String> = chunk_ids.iter().cloned().collect();
        Ok(self
            .edges
            .lock()
            .values()
            .filter(|props| {
                props_in_list_scope(props, biz_id) && source_ids_intersect(props, &wanted)
            })
            .cloned()
            .collect())
    }

    async fn get_popular_labels(&self, limit: usize, biz_id: &str) -> easyrag_core::Result<Vec<String>> {
        let nodes = self.nodes.lock();
        let adjacency = self.adjacency.lock();
        let mut degrees: Vec<(String, usize)> = nodes
            .iter()
            .filter(|(_, props)| props_in_list_scope(props, biz_id))
            .map(|(id, _)| {
                let degree = adjacency
                    .get(id)
                    .map(|s| {
                        s.iter()
                            .filter(|n| {
                                nodes
                                    .get(*n)
                                    .is_some_and(|props| props_in_list_scope(props, biz_id))
                            })
                            .count()
                    })
                    .unwrap_or(0);
                (id.clone(), degree)
            })
            .collect();
        degrees.sort_by(|a, b| b.1.cmp(&a.1));
        Ok(degrees.into_iter().take(limit).map(|(id, _)| id).collect())
    }

    async fn search_labels(
        &self,
        query: &str,
        limit: usize,
        biz_id: &str,
    ) -> easyrag_core::Result<Vec<String>> {
        let nodes = self.nodes.lock();
        let lower_query = query.to_lowercase();
        Ok(nodes
            .iter()
            .filter(|(_, props)| props_in_list_scope(props, biz_id))
            .map(|(k, _)| k)
            .filter(|k| k.to_lowercase().contains(&lower_query))
            .take(limit)
            .cloned()
            .collect())
    }
}
