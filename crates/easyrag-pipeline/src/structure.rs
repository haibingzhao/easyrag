//! Document structure tree builder
//!
//! Zero-LLM fast path: parse Markdown headings into a nested
//! [`DocumentStructureIndex`]. Heading-less documents yield `None`
//! (the LLM fallback path is disabled by default and not ported).

use easyrag_core::model::document_structure::{DocumentStructureIndex, SectionNode};

pub struct StructureTreeBuilder {
    max_depth: usize,
}

impl Default for StructureTreeBuilder {
    fn default() -> Self {
        Self::new(4)
    }
}

impl StructureTreeBuilder {
    pub fn new(max_depth: usize) -> Self {
        Self { max_depth }
    }

    /// Build a [`DocumentStructureIndex`] from document content.
    /// Returns `None` when no structure could be determined.
    pub fn build(
        &self,
        content: &str,
        doc_id: &str,
        doc_title: Option<String>,
    ) -> Option<DocumentStructureIndex> {
        if content.trim().is_empty() {
            return None;
        }
        if crate::heading::has_headings(content) {
            let sections = self.build_tree_from_headings(content);
            if !sections.is_empty() {
                tracing::debug!(
                    "Built structure tree with {} top-level sections for document {doc_id}",
                    sections.len()
                );
                return Some(DocumentStructureIndex {
                    doc_id: doc_id.to_string(),
                    doc_title,
                    doc_description: None,
                    sections,
                    // uses the char count; we use the byte length so
                    // this stays consistent with the byte-based
                    // content_start/content_end offsets produced by the
                    // Rust heading parser (consumed only within this runtime).
                    total_chars: content.len(),
                    version: 1,
                });
            }
        }
        tracing::debug!("No structure could be determined for document {doc_id}");
        None
    }

    /// Build a nested tree from Markdown headings using a parent stack.
    fn build_tree_from_headings(&self, content: &str) -> Vec<SectionNode> {
        let headings = crate::heading::parse_headings(content);
        if headings.is_empty() {
            return Vec::new();
        }

        // Which headings are kept (within maxDepth).
        let kept: Vec<bool> = headings.iter().map(|h| h.level <= self.max_depth).collect();

        struct FlatNode {
            title: String,
            level: usize,
            content_start: usize,
            content_end: usize,
            child_indices: Vec<usize>,
        }

        let mut flat_nodes: Vec<FlatNode> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        for (i, h) in headings.iter().enumerate() {
            if !kept[i] {
                continue; // Skip overly deep headings.
            }
            // Pop stack to find parent (strictly lower level).
            while let Some(&top) = stack.last() {
                if flat_nodes[top].level >= h.level {
                    stack.pop();
                } else {
                    break;
                }
            }

            // Content boundaries.
            let content_start = match content[h.offset..].find('\n') {
                Some(pos) => h.offset + pos + 1,
                None => content.len(),
            };
            // Next KEPT heading offset (deep headings are skipped).
            let content_end = ((i + 1)..headings.len())
                .find(|&j| kept[j])
                .map(|j| headings[j].offset)
                .unwrap_or(content.len());

            let node_index = flat_nodes.len();
            flat_nodes.push(FlatNode {
                title: h.title.clone(),
                level: h.level,
                content_start,
                content_end,
                child_indices: Vec::new(),
            });
            if let Some(&parent) = stack.last() {
                flat_nodes[parent].child_indices.push(node_index);
            }
            stack.push(node_index);
        }

        // Root nodes: those not referenced as children.
        let mut is_child = vec![false; flat_nodes.len()];
        for node in &flat_nodes {
            for &child in &node.child_indices {
                is_child[child] = true;
            }
        }

        fn to_section_node(nodes: &[FlatNode], index: usize) -> SectionNode {
            let node = &nodes[index];
            SectionNode {
                title: node.title.clone(),
                level: node.level,
                summary: None,
                content_start: node.content_start,
                content_end: node.content_end,
                children: node
                    .child_indices
                    .iter()
                    .map(|&child| to_section_node(nodes, child))
                    .collect(),
            }
        }

        (0..flat_nodes.len())
            .filter(|&i| !is_child[i])
            .map(|i| to_section_node(&flat_nodes, i))
            .collect()
    }
}
