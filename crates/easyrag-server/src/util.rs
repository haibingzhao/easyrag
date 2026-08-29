//! API utility functions.

use chrono::{DateTime, Local, NaiveDateTime, TimeZone, Utc};
use serde_json::{json, Value};

const TIMESTAMP_FORMAT: &str = "%Y-%m-%d %H:%M:%S";

/// Format an instant to a human-readable string in the system's local zone
///` with
/// `ZoneId.systemDefault`). Null maps to the empty string.
pub fn format_timestamp(instant: Option<DateTime<Utc>>) -> String {
    match instant {
        Some(dt) => dt.with_timezone(&Local).format(TIMESTAMP_FORMAT).to_string(),
        None => String::new(),
    }
}

/// Parse a timestamp string to a UTC datetime. Tries RFC 3339 first, then
/// the local `yyyy-MM-dd HH:mm:ss` pattern; returns `None` on failure
///
pub fn parse_timestamp(value: Option<&str>) -> Option<DateTime<Utc>> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Some(dt.with_timezone(&Utc));
    }
    NaiveDateTime::parse_from_str(value, TIMESTAMP_FORMAT)
        .ok()
        .and_then(|naive| Local.from_local_datetime(&naive).single())
        .map(|dt| dt.with_timezone(&Utc))
}

/// Truncate a string to a maximum length with ellipsis. Counts characters
///, never bytes. Null maps to the empty string.
pub fn truncate_string(text: Option<&str>, max_length: usize) -> String {
    match text {
        None => String::new(),
        Some(t) if t.chars().count() <= max_length => t.to_string(),
        Some(t) => t.chars().take(max_length).collect::<String>() + "...",
    }
}

/// Build a paginated response wrapper.
pub fn paginated_response(items: Value, total: u64, page: u64, page_size: u64) -> Value {
    let total_pages = if page_size == 0 {
        0
    } else {
        (total + page_size - 1) / page_size
    };
    json!({
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    })
}

/// Extract bearer token from an Authorization header. Requires exactly two
/// whitespace-separated parts with a case-insensitive `Bearer` scheme.
pub fn extract_bearer_token(auth_header: Option<&str>) -> Option<&str> {
    let header = auth_header?;
    let mut parts = header.split(' ');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(scheme), Some(token), None) if scheme.eq_ignore_ascii_case("Bearer") => {
            Some(token)
        }
        _ => None,
    }
}

/// Generate a standard error response body (optional `detail` field).
pub fn error_response(status: u16, message: &str, detail: Option<&str>) -> Value {
    match detail {
        Some(detail) => json!({ "status": status, "message": message, "detail": detail }),
        None => json!({ "status": status, "message": message }),
    }
}

/// Generate a standard success response body (optional `data` field).
pub fn success_response(message: &str, data: Option<Value>) -> Value {
    match data {
        Some(data) => json!({ "status": "success", "message": message, "data": data }),
        None => json!({ "status": "success", "message": message }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_string_counts_chars_not_bytes() {
        assert_eq!(truncate_string(None, 200), "");
        assert_eq!(truncate_string(Some("short"), 200), "short");
        assert_eq!(truncate_string(Some("abcdef"), 3), "abc...");
        // Multibyte: 200 CJK chars must not be split mid-codepoint.
        let cjk = "中".repeat(250);
        let out = truncate_string(Some(&cjk), 200);
        assert_eq!(out.chars().count(), 203);
        assert!(out.ends_with("..."));
    }

    #[test]
    fn bearer_token_parsing() {
        assert_eq!(extract_bearer_token(None), None);
        assert_eq!(extract_bearer_token(Some("Bearer abc")), Some("abc"));
        assert_eq!(extract_bearer_token(Some("bearer abc")), Some("abc"));
        assert_eq!(extract_bearer_token(Some("Basic abc")), None);
        assert_eq!(extract_bearer_token(Some("Bearer a b")), None);
        assert_eq!(extract_bearer_token(Some("Bearer")), None);
    }

    #[test]
    fn paginated_response_computes_total_pages() {
        let v = paginated_response(json!([]), 10, 1, 3);
        assert_eq!(v["total"], 10);
        assert_eq!(v["page"], 1);
        assert_eq!(v["page_size"], 3);
        assert_eq!(v["total_pages"], 4);
        // (0 + 3 - 1) / 3 == 0
        assert_eq!(paginated_response(json!([]), 0, 1, 3)["total_pages"], 0);
    }

    #[test]
    fn success_and_error_response_shapes() {
        let s = success_response("ok", None);
        assert_eq!(s["status"], "success");
        assert_eq!(s["message"], "ok");
        assert!(s.get("data").is_none());

        let s = success_response("ok", Some(json!({"k": 1})));
        assert_eq!(s["data"]["k"], 1);

        let e = error_response(400, "bad", None);
        assert_eq!(e["status"], 400);
        assert!(e.get("detail").is_none());

        let e = error_response(500, "boom", Some("stack"));
        assert_eq!(e["detail"], "stack");
    }

    #[test]
    fn timestamp_round_trip() {
        assert_eq!(format_timestamp(None), "");
        let now = Utc::now();
        let formatted = format_timestamp(Some(now));
        assert_eq!(formatted.len(), 19);
        // Local-format parse must succeed and land within a second.
        let parsed = parse_timestamp(Some(&formatted)).expect("local format parse");
        assert!((parsed - now).num_seconds().abs() <= 1);
        // RFC3339 parse path.
        assert_eq!(
            parse_timestamp(Some("2024-01-01T00:00:00Z")),
            Some(DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z").unwrap().with_timezone(&Utc))
        );
        assert_eq!(parse_timestamp(Some("")), None);
        assert_eq!(parse_timestamp(Some("not-a-date")), None);
    }
}
