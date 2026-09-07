//! Shared `{status, message}` result envelope for `StorageNameSpace::drop`.
//!
//! Every backend used to hand-build the same two-entry `HashMap`, seven copies
//! of the same literal keys spread over postgres / neo4j / json and the four
//! in-memory storages. These helpers are the single source of that shape; the
//! contract itself is unchanged (`status` is `"success"` or `"error"`,
//! `message` carries the human-readable detail).

use std::collections::HashMap;

/// Successful drop: `{"status": "success", "message": <detail>}`.
pub(crate) fn drop_success(message: impl Into<String>) -> HashMap<String, String> {
    HashMap::from([
        ("status".to_string(), "success".to_string()),
        ("message".to_string(), message.into()),
    ])
}

/// Failed drop: `{"status": "error", "message": <error text>}`.
///
/// Only the graph backends report a failed drop through the envelope (the
/// in-memory and JSON backends cannot fail), hence the feature gate.
#[cfg_attr(
    not(any(feature = "neo4j", feature = "postgres")),
    allow(dead_code)
)]
pub(crate) fn drop_error(error: impl std::fmt::Display) -> HashMap<String, String> {
    HashMap::from([
        ("status".to_string(), "error".to_string()),
        ("message".to_string(), error.to_string()),
    ])
}

/// Fold a fallible drop into the envelope: `Ok(())` reports `success_message`,
/// `Err` reports the error text as an `"error"` status (never as a propagated
/// error — `drop` answers with the envelope either way).
///
/// Postgres is the only backend whose drop is fallible in four places.
#[cfg_attr(not(feature = "postgres"), allow(dead_code))]
pub(crate) fn drop_result(
    outcome: easyrag_core::Result<()>,
    success_message: String,
) -> HashMap<String, String> {
    match outcome {
        Ok(()) => drop_success(success_message),
        Err(e) => drop_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use easyrag_core::CoreError;

    /// The envelope is a wire contract: exactly two keys, exact literal values.
    #[test]
    fn success_envelope_has_exactly_status_and_message() {
        let envelope = drop_success("data dropped (2 records for biz default)");
        assert_eq!(2, envelope.len());
        assert_eq!("success", envelope["status"]);
        assert_eq!("data dropped (2 records for biz default)", envelope["message"]);
    }

    #[test]
    fn error_envelope_renders_the_error_display_text() {
        let envelope = drop_error(CoreError::InvalidArgument("boom".to_string()));
        assert_eq!(2, envelope.len());
        assert_eq!("error", envelope["status"]);
        assert_eq!("boom", envelope["message"]);
    }

    #[test]
    fn result_folds_ok_to_success_and_err_to_error() {
        let ok = drop_result(
            Ok(()),
            "All nodes in workspace 'ws', biz_id 'b' deleted".to_string(),
        );
        assert_eq!("success", ok["status"]);
        assert_eq!("All nodes in workspace 'ws', biz_id 'b' deleted", ok["message"]);

        let err = drop_result(
            Err(CoreError::InvalidArgument("boom".to_string())),
            "unused".to_string(),
        );
        assert_eq!("error", err["status"]);
        assert_eq!("boom", err["message"]);
    }
}
