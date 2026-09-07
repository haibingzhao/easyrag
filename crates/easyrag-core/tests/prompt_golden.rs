//! Phase 0.2 — Prompt render byte-level golden snapshots.
//!
//! Guardrail for the whole refactoring: prompt constants are "string == semantics",
//! so structural assertions are not enough to detect drift. These snapshots pin the
//! EXACT bytes of every core prompt constant (and a couple of rendered examples) so
//! that:
//!   * A-13 (`RAG_RESPONSE` vs `NAIVE_RAG_RESPONSE` drift) is captured as a baseline,
//!   * Phase 4.7 (`PromptBuilder` + segmented constants) can be proven byte-identical,
//!   * any accidental edit to a prompt fails loudly here before it reaches an LLM.
//!
//! Snapshots land in `<workspace>/testdata/golden/prompts/`.
//! Regenerate deliberately with: `INSTA_UPDATE=always cargo test -p easyrag-core --test prompt_golden`.

use std::collections::HashMap;

use easyrag_core::prompt::{
    format_prompt, CLAIM_DECOMPOSITION_PROMPT, CLAIM_VERIFICATION_PROMPT, CONTEXTUAL_CHUNKING_PROMPT,
    DOC_DESCRIPTION_PROMPT, ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT,
    ENTITY_CONTINUE_EXTRACTION_USER_PROMPT, ENTITY_EXTRACTION_EXAMPLES,
    ENTITY_EXTRACTION_JSON_EXAMPLES, ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
    ENTITY_EXTRACTION_JSON_USER_PROMPT, ENTITY_EXTRACTION_SECTION_CONTEXT,
    ENTITY_EXTRACTION_SYSTEM_PROMPT, ENTITY_EXTRACTION_USER_PROMPT, EVIDENCE_QUALITY_PROMPT,
    FAIL_RESPONSE, KEYWORDS_EXTRACTION, KG_QUERY_CONTEXT, NAIVE_QUERY_CONTEXT, NAIVE_RAG_RESPONSE,
    QUERY_ROUTING_PROMPT, RAG_RESPONSE, SECTION_SUMMARY_PROMPT, STRUCTURE_AGENT_SYSTEM_PROMPT,
    STRUCTURE_DRILL_DOWN_PROMPT, STRUCTURE_NAVIGATE_PROMPT, STRUCTURE_QUERY_RESPONSE,
    SUMMARIZE_ENTITY_DESCRIPTIONS,
};

/// Absolute path of the golden snapshot directory for this suite.
///
/// Resolved from `CARGO_MANIFEST_DIR` (…/crates/easyrag-core) up to the workspace
/// root, then into `testdata/golden/prompts`. An absolute path is used so insta
/// honours it regardless of how it joins the base directory.
fn golden_dir() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root above crates/easyrag-core");
    root.join("testdata").join("golden").join("prompts")
}

/// Snapshot a raw prompt constant's exact bytes under a stable name.
fn snap_const(name: &str, value: &str) {
    insta::with_settings!({
        snapshot_path => golden_dir(),
        prepend_module_to_snapshot => false,
        omit_expression => true,
    }, {
        insta::assert_snapshot!(name, value);
    });
}

#[test]
fn golden_response_prompts() {
    // The A-13 pair: RAG (KG + chunks) vs NAIVE (chunks only). Only their
    // intentional differences remain (KG wording, {context_data} vs {content_data});
    // `[BC-2]` aligned the accidental `references` vs `**References**` drift, which
    // is why RAG_RESPONSE.snap and RENDERED_RAG_RESPONSE.snap changed by one line.
    snap_const("RAG_RESPONSE", &RAG_RESPONSE);
    snap_const("NAIVE_RAG_RESPONSE", &NAIVE_RAG_RESPONSE);
    snap_const("FAIL_RESPONSE", FAIL_RESPONSE);
}

#[test]
fn golden_query_context_prompts() {
    snap_const("KG_QUERY_CONTEXT", &KG_QUERY_CONTEXT);
    snap_const("NAIVE_QUERY_CONTEXT", &NAIVE_QUERY_CONTEXT);
}

#[test]
fn golden_entity_extraction_prompts() {
    snap_const("ENTITY_EXTRACTION_SECTION_CONTEXT", ENTITY_EXTRACTION_SECTION_CONTEXT);
    snap_const("ENTITY_EXTRACTION_SYSTEM_PROMPT", &ENTITY_EXTRACTION_SYSTEM_PROMPT);
    snap_const("ENTITY_EXTRACTION_USER_PROMPT", &ENTITY_EXTRACTION_USER_PROMPT);
    snap_const(
        "ENTITY_CONTINUE_EXTRACTION_USER_PROMPT",
        ENTITY_CONTINUE_EXTRACTION_USER_PROMPT,
    );
    snap_const(
        "ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT",
        &ENTITY_EXTRACTION_JSON_SYSTEM_PROMPT,
    );
    snap_const("ENTITY_EXTRACTION_JSON_USER_PROMPT", &ENTITY_EXTRACTION_JSON_USER_PROMPT);
    snap_const(
        "ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT",
        ENTITY_CONTINUE_EXTRACTION_JSON_USER_PROMPT,
    );
    snap_const("ENTITY_EXTRACTION_EXAMPLES", &ENTITY_EXTRACTION_EXAMPLES.join("\n"));
    snap_const(
        "ENTITY_EXTRACTION_JSON_EXAMPLES",
        &ENTITY_EXTRACTION_JSON_EXAMPLES.join("\n"),
    );
    snap_const("SUMMARIZE_ENTITY_DESCRIPTIONS", SUMMARIZE_ENTITY_DESCRIPTIONS);
}

#[test]
fn golden_keyword_and_routing_prompts() {
    snap_const("KEYWORDS_EXTRACTION", KEYWORDS_EXTRACTION);
    snap_const("QUERY_ROUTING_PROMPT", QUERY_ROUTING_PROMPT);
    snap_const("EVIDENCE_QUALITY_PROMPT", EVIDENCE_QUALITY_PROMPT);
    snap_const("CLAIM_DECOMPOSITION_PROMPT", CLAIM_DECOMPOSITION_PROMPT);
    snap_const("CLAIM_VERIFICATION_PROMPT", CLAIM_VERIFICATION_PROMPT);
}

#[test]
fn golden_structure_and_chunking_prompts() {
    snap_const("DOC_DESCRIPTION_PROMPT", DOC_DESCRIPTION_PROMPT);
    snap_const("SECTION_SUMMARY_PROMPT", SECTION_SUMMARY_PROMPT);
    snap_const("STRUCTURE_NAVIGATE_PROMPT", STRUCTURE_NAVIGATE_PROMPT);
    snap_const("STRUCTURE_DRILL_DOWN_PROMPT", STRUCTURE_DRILL_DOWN_PROMPT);
    snap_const("STRUCTURE_QUERY_RESPONSE", STRUCTURE_QUERY_RESPONSE);
    snap_const("STRUCTURE_AGENT_SYSTEM_PROMPT", STRUCTURE_AGENT_SYSTEM_PROMPT);
    snap_const("CONTEXTUAL_CHUNKING_PROMPT", CONTEXTUAL_CHUNKING_PROMPT);
}

#[test]
fn golden_rendered_examples() {
    // End-to-end rendered bytes: format_prompt substitutes {placeholders}. Unknown
    // placeholders are intentionally left literal (matches format_prompt contract).
    let mut vars = HashMap::new();
    vars.insert("response_type".to_string(), "Multiple Paragraphs".to_string());
    vars.insert("user_prompt".to_string(), "Answer concisely.".to_string());
    vars.insert("context_data".to_string(), "<CONTEXT_DATA>".to_string());
    snap_const("RENDERED_RAG_RESPONSE", &format_prompt(&RAG_RESPONSE, &vars));

    let mut naive_vars = HashMap::new();
    naive_vars.insert("response_type".to_string(), "Multiple Paragraphs".to_string());
    naive_vars.insert("user_prompt".to_string(), "Answer concisely.".to_string());
    naive_vars.insert("content_data".to_string(), "<CONTENT_DATA>".to_string());
    snap_const(
        "RENDERED_NAIVE_RAG_RESPONSE",
        &format_prompt(&NAIVE_RAG_RESPONSE, &naive_vars),
    );
}
