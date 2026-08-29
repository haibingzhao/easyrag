//! Ported from LuceneKeywordSearchTest.kt (10 cases) against the tantivy
//! replacement backend. Run with: cargo test -p easyrag-storage
//! --features tantivy-backend

#![cfg(feature = "tantivy-backend")]

use easyrag_core::storage::keyword::KeywordSearchable;
use easyrag_core::JsonMap;
use easyrag_storage::tantivy_search::TantivyKeywordSearch;
use serde_json::json;

/// Default biz line used to pass the biz_id filter argument.
const BIZ: &str = "default";
use std::collections::HashMap;

fn json_map(value: serde_json::Value) -> JsonMap {
    match value {
        serde_json::Value::Object(m) => m,
        _ => panic!("expected object"),
    }
}

fn create_search(temp_dir: &tempfile::TempDir) -> TantivyKeywordSearch {
    TantivyKeywordSearch::new(
        "test",
        temp_dir.path().to_str().unwrap(),
        16,
    )
    .unwrap()
}

fn ids(results: &[JsonMap]) -> Vec<String> {
    results
        .iter()
        .map(|r| r["id"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn index_and_query_basic() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([
            (
                "doc1".to_string(),
                json_map(json!({"content": "The quick brown fox jumps over the lazy dog", "file_path": "a.txt"})),
            ),
            (
                "doc2".to_string(),
                json_map(json!({"content": "A fast red car drives on the highway", "file_path": "b.txt"})),
            ),
            (
                "doc3".to_string(),
                json_map(json!({"content": "The fox and the hound are friends", "file_path": "c.txt"})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    let results = search.keyword_query("fox", 10, None, None, BIZ).await.unwrap();
    assert!(!results.is_empty(), "Should find documents containing 'fox'");
    let ids = ids(&results);
    assert!(ids.contains(&"doc1".to_string()), "doc1 contains 'fox'");
    assert!(ids.contains(&"doc3".to_string()), "doc3 contains 'fox'");
    assert!(!ids.contains(&"doc2".to_string()), "doc2 does not contain 'fox'");
    search.close().unwrap();
}

#[tokio::test]
async fn query_top_k_limits_results() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    let docs: HashMap<String, JsonMap> = (1..=20)
        .map(|i| {
            (
                format!("doc{i}"),
                json_map(json!({"content": format!("common keyword text number {i}"), "file_path": format!("f{i}.txt")})),
            )
        })
        .collect();
    search.index_documents(docs, BIZ).await.unwrap();

    let results = search
        .keyword_query("common keyword", 5, None, None, BIZ)
        .await
        .unwrap();
    assert!(results.len() <= 5, "Should return at most 5 results");
    search.close().unwrap();
}

#[tokio::test]
async fn remove_documents_no_longer_matched() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([
            (
                "doc1".to_string(),
                json_map(json!({"content": "unique elephant document", "file_path": "a.txt"})),
            ),
            (
                "doc2".to_string(),
                json_map(json!({"content": "another elephant reference", "file_path": "b.txt"})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    let mut results = search.keyword_query("elephant", 10, None, None, BIZ).await.unwrap();
    assert_eq!(2, results.len());

    search.remove_documents(&["doc1".to_string()], BIZ).await.unwrap();

    results = search.keyword_query("elephant", 10, None, None, BIZ).await.unwrap();
    assert_eq!(1, results.len());
    assert_eq!("doc2", results.first().unwrap()["id"].as_str().unwrap());
    search.close().unwrap();
}

#[tokio::test]
async fn update_document_idempotent() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({"content": "original content about cats", "file_path": "a.txt"})),
        )]), BIZ)
        .await
        .unwrap();

    // Update same ID with new content.
    search
        .index_documents(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({"content": "updated content about dogs", "file_path": "a.txt"})),
        )]), BIZ)
        .await
        .unwrap();

    let cat_results = search.keyword_query("cats", 10, None, None, BIZ).await.unwrap();
    assert!(cat_results.is_empty(), "Old content should not match after update");

    let dog_results = search.keyword_query("dogs", 10, None, None, BIZ).await.unwrap();
    assert_eq!(1, dog_results.len());
    assert_eq!("doc1", dog_results.first().unwrap()["id"].as_str().unwrap());
    search.close().unwrap();
}

#[tokio::test]
async fn commit_index_persists() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({"content": "persistent data test", "file_path": "a.txt"})),
        )]), BIZ)
        .await
        .unwrap();
    search.commit_index().await.unwrap();
    search.close().unwrap();

    // Reopen and verify data persists.
    let search2 = create_search(&temp_dir);
    let results = search2
        .keyword_query("persistent", 10, None, None, BIZ)
        .await
        .unwrap();
    assert_eq!(1, results.len());
    assert_eq!("doc1", results.first().unwrap()["id"].as_str().unwrap());
    search2.close().unwrap();
}

#[tokio::test]
async fn query_empty_query_returns_empty() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({"content": "some content", "file_path": "a.txt"})),
        )]), BIZ)
        .await
        .unwrap();
    let results = search.keyword_query("", 10, None, None, BIZ).await.unwrap();
    assert!(results.is_empty());
    search.close().unwrap();
}

#[tokio::test]
async fn query_result_contains_expected_fields() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([(
            "doc1".to_string(),
            json_map(json!({"content": "hello world test", "file_path": "test.txt"})),
        )]), BIZ)
        .await
        .unwrap();
    let results = search.keyword_query("hello", 10, None, None, BIZ).await.unwrap();
    assert_eq!(1, results.len());
    let result = &results[0];
    assert_eq!("doc1", result["id"].as_str().unwrap());
    assert_eq!("hello world test", result["content"].as_str().unwrap());
    assert_eq!("test.txt", result["file_path"].as_str().unwrap());
    assert!(result["score"].as_f64().unwrap() > 0.0);
    search.close().unwrap();
}

#[tokio::test]
async fn query_multi_term_or_semantics() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([
            (
                "doc1".to_string(),
                json_map(json!({"content": "apple banana cherry", "file_path": "a.txt"})),
            ),
            (
                "doc2".to_string(),
                json_map(json!({"content": "banana smoothie recipe", "file_path": "b.txt"})),
            ),
            (
                "doc3".to_string(),
                json_map(json!({"content": "chocolate cake recipe", "file_path": "c.txt"})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    // OR semantics: should match docs with "banana" OR "chocolate".
    let results = search
        .keyword_query("banana chocolate", 10, None, None, BIZ)
        .await
        .unwrap();
    let ids = ids(&results);
    assert!(
        ids.contains(&"doc1".to_string()) || ids.contains(&"doc2".to_string()),
        "Should match banana docs"
    );
    assert!(ids.contains(&"doc3".to_string()), "Should match chocolate doc");
    search.close().unwrap();
}

#[tokio::test]
async fn query_time_range_filters_results() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    let now = chrono::Utc::now().timestamp();
    search
        .index_documents(HashMap::from([
            (
                "old-doc".to_string(),
                json_map(json!({"content": "ancient history document", "file_path": "a.txt", "create_time": now - 86400})),
            ),
            (
                "new-doc".to_string(),
                json_map(json!({"content": "recent history document", "file_path": "b.txt", "create_time": now})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    // Only recent docs (last hour).
    let results = search
        .keyword_query("history document", 10, Some(now - 3600), None, BIZ)
        .await
        .unwrap();
    let ids = ids(&results);
    assert!(ids.contains(&"new-doc".to_string()), "Recent doc should be included");
    assert!(!ids.contains(&"old-doc".to_string()), "Old doc should be filtered out");
    search.close().unwrap();
}

#[tokio::test]
async fn query_time_range_end_filters_future_docs() {
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    let now = chrono::Utc::now().timestamp();
    search
        .index_documents(HashMap::from([
            (
                "past-doc".to_string(),
                json_map(json!({"content": "science experiment results", "file_path": "a.txt", "create_time": now - 7200})),
            ),
            (
                "future-doc".to_string(),
                json_map(json!({"content": "science fiction novel", "file_path": "b.txt", "create_time": now + 86400})),
            ),
        ]), BIZ)
        .await
        .unwrap();

    // Only docs before now.
    let results = search
        .keyword_query("science", 10, None, Some(now), BIZ)
        .await
        .unwrap();
    let ids = ids(&results);
    assert!(ids.contains(&"past-doc".to_string()), "Past doc should be included");
    assert!(!ids.contains(&"future-doc".to_string()), "Future doc should be filtered out");
    search.close().unwrap();
}

#[tokio::test]
async fn hangul_is_split_into_single_char_tokens_like_lucene() {
    // Lucene StandardAnalyzer treats Hangul syllables as CJK-class scripts
    // (single-character tokens); SimpleTokenizer must match that behavior.
    let temp_dir = tempfile::tempdir().unwrap();
    let search = create_search(&temp_dir);
    search
        .index_documents(HashMap::from([(
            "kr-doc".to_string(),
            json_map(json!({"content": "한국어 텍스트 검색", "file_path": "k.txt"})),
        )]), BIZ)
        .await
        .unwrap();

    // A substring of the original syllable run must match (would not if the
    // run were kept as one token).
    let results = search.keyword_query("한국", 10, None, None, BIZ).await.unwrap();
    assert_eq!(vec!["kr-doc"], ids(&results));
    search.close().unwrap();
}

#[test]
fn new_rejects_legacy_lucene_dir_without_tantivy_metadata() {
    // A non-empty index dir lacking meta.json is a legacy Lucene index;
    // creating tantivy files inside it must fail fast instead of mixing.
    let temp_dir = tempfile::tempdir().unwrap();
    let legacy = temp_dir
        .path()
        .join("lucene_index")
        .join("legacy-ws")
        .join("default");
    std::fs::create_dir_all(&legacy).unwrap();
    std::fs::write(legacy.join("segments_1"), b"lucene-ish").unwrap();

    let result = TantivyKeywordSearch::new(
        "legacy-ws",
        temp_dir.path().to_str().unwrap(),
        16,
    );
    assert!(result.is_err(), "must refuse to write into a legacy Lucene dir");
}

#[tokio::test]
async fn shared_index_filters_by_biz_argument() {
    // One workspace owns one shared index dir (lucene_index/{workspace});
    // biz_id is a stored field + query filter, not a directory dimension.
    // Documents of two biz lines must not leak into each other, and a
    // remove in one biz line must not destroy a same-id doc of the other.
    const BIZ_A: &str = "biz_a";
    const BIZ_B: &str = "biz_b";
    let temp_dir = tempfile::tempdir().unwrap();
    let dir = temp_dir.path().to_str().unwrap();
    let search = TantivyKeywordSearch::new("wsx", dir, 16).unwrap();
    assert!(temp_dir
        .path()
        .join("lucene_index")
        .join("wsx")
        .join("meta.json")
        .exists());

    search
        .index_documents(HashMap::from([(
            "docA".to_string(),
            json_map(json!({"content": "zebra crossing report", "file_path": "a.txt"})),
        )]), BIZ_A)
        .await
        .unwrap();
    // Same id in biz_b with different content.
    search
        .index_documents(HashMap::from([(
            "docA".to_string(),
            json_map(json!({"content": "zebra finance ledger", "file_path": "b.txt"})),
        )]), BIZ_B)
        .await
        .unwrap();

    let in_a = search.keyword_query("zebra", 10, None, None, BIZ_A).await.unwrap();
    assert_eq!(1, in_a.len(), "biz_a must see only its own document");
    assert_eq!("a.txt", in_a[0]["file_path"].as_str().unwrap());
    let in_b = search.keyword_query("zebra", 10, None, None, BIZ_B).await.unwrap();
    assert_eq!(1, in_b.len(), "biz_b must see only its own document");
    assert_eq!("b.txt", in_b[0]["file_path"].as_str().unwrap());

    // Removing docA in biz_a must not touch biz_b's same-id document.
    search.remove_documents(&["docA".to_string()], BIZ_A).await.unwrap();
    let after_a = search.keyword_query("zebra", 10, None, None, BIZ_A).await.unwrap();
    assert!(after_a.is_empty(), "biz_a doc must be gone");
    let after_b = search.keyword_query("zebra", 10, None, None, BIZ_B).await.unwrap();
    assert_eq!(1, after_b.len(), "biz_b doc must survive cross-biz remove");

    search.close().unwrap();
}
