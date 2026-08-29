//! StructureTreeBuilder tests (6 cases).

use easyrag_pipeline::StructureTreeBuilder;

#[test]
fn build_empty_content_returns_none() {
    let builder = StructureTreeBuilder::new(4);
    assert!(builder.build("", "doc1", None).is_none());
}

#[test]
fn build_no_headings_returns_none() {
    let builder = StructureTreeBuilder::new(4);
    let content = "This is plain text without any headings.\nJust paragraphs.";
    assert!(builder.build(content, "doc1", None).is_none());
}

#[test]
fn build_simple_headings_builds_tree() {
    let builder = StructureTreeBuilder::new(4);
    let content = "# Chapter 1\nContent of chapter 1.\n\n# Chapter 2\nContent of chapter 2.";

    let result = builder
        .build(content, "doc1", Some("Test Doc".to_string()))
        .expect("structure expected");
    assert_eq!("doc1", result.doc_id);
    assert_eq!(Some("Test Doc".to_string()), result.doc_title);
    assert_eq!(2, result.sections.len());
    assert_eq!("Chapter 1", result.sections[0].title);
    assert_eq!("Chapter 2", result.sections[1].title);
    assert_eq!(1, result.sections[0].level);
}

#[test]
fn build_nested_headings_builds_hierarchy() {
    let builder = StructureTreeBuilder::new(4);
    let content = "# Part 1\n## Section 1.1\nContent 1.1\n## Section 1.2\nContent 1.2\n\
                   # Part 2\nContent 2";

    let result = builder.build(content, "doc1", None).expect("structure expected");
    assert_eq!(2, result.sections.len());

    let part1 = &result.sections[0];
    assert_eq!("Part 1", part1.title);
    assert_eq!(2, part1.children.len());
    assert_eq!("Section 1.1", part1.children[0].title);
    assert_eq!("Section 1.2", part1.children[1].title);

    let part2 = &result.sections[1];
    assert_eq!("Part 2", part2.title);
    assert!(part2.children.is_empty());
}

#[test]
fn build_deep_nesting_respects_max_depth() {
    let content = "# L1\n## L2\n### L3\n#### L4\n##### L5\nContent at level 5";

    let builder = StructureTreeBuilder::new(3);
    let result = builder.build(content, "doc1", None).expect("structure expected");
    // L4 and L5 should be skipped (level > maxDepth).
    assert_eq!(1, result.sections.len());
    assert_eq!("L1", result.sections[0].title);
}

#[test]
fn build_content_offsets_are_correct() {
    let content = "# Title\nSome content here";
    let builder = StructureTreeBuilder::new(4);
    let result = builder.build(content, "doc1", None).expect("structure expected");
    let section = &result.sections[0];
    assert!(section.content_start > 0);
    assert!(section.content_end <= content.len());
}
