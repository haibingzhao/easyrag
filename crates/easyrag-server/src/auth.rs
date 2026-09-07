//! Security components.
//!
//!  blueprint: `JwtAuthProvider` (HMAC-SHA, key = secret bytes padded
//! to 32), `LoginRateLimiter` (sliding window + LRU eviction), and a
//! security filter chain whitelisting `/api/auth/login`, `/api/health`,
//! `/api/version`, `/api/auth-status` while requiring authentication for
//! every other `/api/**` route when auth is enabled.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::Request;
use axum::http::header;
use axum::middleware::Next;
use axum::response::Response;
use jsonwebtoken::errors::ErrorKind;
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::util::extract_bearer_token;

/// Paths that stay public when authentication is enabled
///
pub const AUTH_WHITELIST: [&str; 4] =
    ["/api/auth/login", "/api/health", "/api/version", "/api/auth-status"];

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// HMAC key derivation matching:
/// truncate to 32 bytes or zero-pad up to 32.
fn derive_key(secret: &str) -> [u8; 32] {
    let bytes = secret.as_bytes();
    let mut key = [0u8; 32];
    let n = bytes.len().min(32);
    key[..n].copy_from_slice(&bytes[..n]);
    key
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    iat: i64,
    exp: i64,
}

/// JWT authentication provider.
#[derive(Clone)]
pub struct JwtAuthProvider {
    key: [u8; 32],
    expiration_hours: u64,
}

impl JwtAuthProvider {
    pub fn new(secret: &str, expiration_hours: u64) -> Self {
        Self {
            key: derive_key(secret),
            expiration_hours,
        }
    }

    pub fn generate_token(&self, username: &str) -> Result<String, jsonwebtoken::errors::Error> {
        let now_seconds = now_millis() / 1000;
        let claims = Claims {
            sub: username.to_string(),
            iat: now_seconds,
            exp: now_seconds + (self.expiration_hours as i64) * 3600,
        };
        encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(&self.key),
        )
    }

    /// Returns the subject on success; every failure maps to `None`
    ///
    pub fn validate_token(&self, token: &str) -> Option<String> {
        self.validate_token_detailed(token).ok()
    }

    /// Like `validate_token` but classifies the failure reason for the
    /// auth middleware's diagnostic log ( swallows the exception;
    /// the classification only lands in server-side logs).
    pub fn validate_token_detailed(&self, token: &str) -> Result<String, String> {
        let mut validation = Validation::new(Algorithm::HS256);
        validation.required_spec_claims.clear();
        decode::<Claims>(
            token,
            &DecodingKey::from_secret(&self.key),
            &validation,
        )
        .map(|data| data.claims.sub)
        .map_err(|e| match e.kind() {
            ErrorKind::ExpiredSignature => "token expired".to_string(),
            ErrorKind::InvalidSignature | ErrorKind::Crypto(_) => {
                "signature verification failed (signing key mismatch?)".to_string()
            }
            ErrorKind::InvalidAlgorithm => "unexpected signing algorithm".to_string(),
            ErrorKind::InvalidToken
            | ErrorKind::Base64(_)
            | ErrorKind::Json(_)
            | ErrorKind::Utf8(_) => "malformed token".to_string(),
            other => format!("{other:?}"),
        })
    }

    /// Short hex fingerprint of the signing key: the 32-byte key is folded
    /// into 8 bytes via XOR (every byte influences the result), for
    /// diagnosing cross-process/restart secret mismatches in deployments;
    /// never exposes the key itself.
    pub fn key_fingerprint(&self) -> String {
        let mut folded = [0u8; 8];
        for (i, b) in self.key.iter().enumerate() {
            folded[i % 8] ^= b;
        }
        folded.iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[derive(Default)]
struct RateLimitEntry {
    attempts: Vec<i64>,
    last_access: i64,
}

/// Login rate limiter preventing brute-force attacks: sliding 60s window, LRU eviction once the tracked
/// key count exceeds the cap.
pub struct LoginRateLimiter {
    max_attempts_per_minute: usize,
    max_tracked_keys: usize,
    attempts: Mutex<HashMap<String, RateLimitEntry>>,
}

impl LoginRateLimiter {
    pub fn new(max_attempts_per_minute: usize, max_tracked_keys: usize) -> Self {
        Self {
            max_attempts_per_minute,
            max_tracked_keys,
            attempts: Mutex::new(HashMap::new()),
        }
    }

    /// Sanitize identifier to prevent log injection
    ///
    fn sanitize(identifier: &str) -> String {
        identifier
            .chars()
            .map(|c| match c {
                '\r' | '\n' | '\t' => '_',
                other => other,
            })
            .take(256)
            .collect()
    }

    pub fn is_allowed(&self, identifier: &str) -> bool {
        let safe_id = Self::sanitize(identifier);
        let now = now_millis();
        let window_start = now - 60_000;
        let mut map = self.attempts.lock().unwrap();
        let entry = map.entry(safe_id).or_default();
        entry.last_access = now;
        entry.attempts.retain(|t| *t >= window_start);
        entry.attempts.len() < self.max_attempts_per_minute
    }

    pub fn record_attempt(&self, identifier: &str) {
        self.record_attempt_at(identifier, now_millis());
    }

    fn record_attempt_at(&self, identifier: &str, now: i64) {
        let safe_id = Self::sanitize(identifier);
        let mut map = self.attempts.lock().unwrap();
        let entry = map.entry(safe_id).or_default();
        entry.last_access = now;
        entry.attempts.push(now);
        // LRU eviction: drop the least-recently-accessed key once the cap
        // is exceeded.
        if map.len() > self.max_tracked_keys {
            if let Some(eldest) = map
                .iter()
                .min_by_key(|(_, e)| e.last_access)
                .map(|(k, _)| k.clone())
            {
                map.remove(&eldest);
            }
        }
    }

    pub fn reset(&self, identifier: &str) {
        let safe_id = Self::sanitize(identifier);
        self.attempts.lock().unwrap().remove(&safe_id);
    }
}

/// Axum middleware enforcing the security filter chain: public
/// whitelist + `/api/**` authenticated, everything else permitted.
///
/// Rejections are logged with a classified reason (missing/malformed
/// header, expired token, signature mismatch, ...) plus the request path
/// and client IP, so 401 incidents can be attributed to a signing-key
/// change/restart vs. an expired or malformed token. Only the first 16
/// chars of the token are logged (never the full token).
pub async fn auth_middleware(
    axum::extract::State(jwt): axum::extract::State<std::sync::Arc<JwtAuthProvider>>,
    req: Request,
    next: Next,
) -> Result<Response, AppError> {
    let path = req.uri().path().to_string();
    if !path.starts_with("/api/") || AUTH_WHITELIST.contains(&path.as_str()) {
        return Ok(next.run(req).await);
    }
    let client_ip = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|c| c.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let header = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    let Some(header) = header else {
        tracing::warn!("[auth] 401 {path} from {client_ip}: missing Authorization header");
        return Err(AppError::unauthorized("Unauthorized"));
    };
    let Some(token) = extract_bearer_token(Some(header)) else {
        tracing::warn!("[auth] 401 {path} from {client_ip}: malformed Authorization header");
        return Err(AppError::unauthorized("Unauthorized"));
    };
    match jwt.validate_token_detailed(token) {
        Ok(_) => Ok(next.run(req).await),
        Err(reason) => {
            let prefix: String = token.chars().take(16).collect();
            tracing::warn!(
                "[auth] 401 {path} from {client_ip}: token {prefix}... rejected: {reason}"
            );
            Err(AppError::unauthorized("Unauthorized"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_derivation_zero_pads_and_truncates() {
        let short = derive_key("abc");
        assert_eq!(&short[..3], b"abc");
        assert!(short[3..].iter().all(|b| *b == 0));
        let long = derive_key(&"x".repeat(64));
        assert_eq!(long, [b'x'; 32]);
    }

    #[test]
    fn jwt_round_trip_and_rejections() {
        let provider = JwtAuthProvider::new("some-secret", 24);
        let token = provider.generate_token("admin").unwrap();
        assert_eq!(provider.validate_token(&token).as_deref(), Some("admin"));
        assert_eq!(provider.validate_token("garbage"), None);
        // Token signed with a different key must be rejected.
        let other = JwtAuthProvider::new("other-secret", 24);
        assert_eq!(other.validate_token(&token), None);
    }

    #[test]
    fn expired_token_is_rejected() {
        let provider = JwtAuthProvider::new("secret", 24);
        let now_seconds = now_millis() / 1000;
        let claims = Claims {
            sub: "admin".into(),
            iat: now_seconds - 7200,
            exp: now_seconds - 3600,
        };
        let token = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(&provider.key),
        )
        .unwrap();
        assert_eq!(provider.validate_token(&token), None);
    }

    #[test]
    fn detailed_validation_classifies_rejection_reasons() {
        let provider = JwtAuthProvider::new("some-secret", 24);
        let token = provider.generate_token("admin").unwrap();
        assert_eq!(
            provider.validate_token_detailed(&token).as_deref(),
            Ok("admin")
        );

        // Token signed with a different key -> signature failure, the
        // class that points at a signing-key mismatch across processes.
        let other = JwtAuthProvider::new("other-secret", 24);
        assert!(matches!(
            other.validate_token_detailed(&token),
            Err(reason) if reason.contains("signature")
        ));

        // Garbage input -> malformed.
        assert!(matches!(
            provider.validate_token_detailed("garbage"),
            Err(reason) if reason.contains("malformed")
        ));

        // Expired beyond the 60s leeway -> expired.
        let now_seconds = now_millis() / 1000;
        let claims = Claims {
            sub: "admin".into(),
            iat: now_seconds - 7200,
            exp: now_seconds - 3600 - 61,
        };
        let expired = encode(
            &Header::new(Algorithm::HS256),
            &claims,
            &EncodingKey::from_secret(&provider.key),
        )
        .unwrap();
        assert!(matches!(
            provider.validate_token_detailed(&expired),
            Err(reason) if reason.contains("expired")
        ));
    }

    #[test]
    fn key_fingerprint_is_stable_and_secret_dependent() {
        let a = JwtAuthProvider::new("secret-a-abcdefghijklmnopqrstuvwxyz", 24);
        let b = JwtAuthProvider::new("secret-b-abcdefghijklmnopqrstuvwxyz", 24);
        assert_eq!(a.key_fingerprint(), a.key_fingerprint());
        assert_eq!(a.key_fingerprint().len(), 16);
        assert_ne!(a.key_fingerprint(), b.key_fingerprint());
    }

    #[test]
    fn rate_limiter_sliding_window() {
        let limiter = LoginRateLimiter::new(2, 10);
        assert!(limiter.is_allowed("user"));
        limiter.record_attempt("user");
        limiter.record_attempt("user");
        assert!(!limiter.is_allowed("user"));
        // Distinct identifier unaffected.
        assert!(limiter.is_allowed("other"));
        limiter.reset("user");
        assert!(limiter.is_allowed("user"));
    }

    #[test]
    fn rate_limiter_evicts_least_recently_accessed() {
        let limiter = LoginRateLimiter::new(5, 2);
        limiter.record_attempt_at("a", 1000);
        limiter.record_attempt_at("b", 2000);
        limiter.record_attempt_at("c", 3000); // evicts "a" (oldest access)
        let map = limiter.attempts.lock().unwrap();
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key("a"));
        assert!(map.contains_key("b") && map.contains_key("c"));
    }

    #[test]
    fn rate_limiter_sanitizes_identifiers() {
        // Wide cap: this test only checks identifier sanitization, not LRU
        // eviction (sub-millisecond runs make eviction order nondeterministic).
        let limiter = LoginRateLimiter::new(5, 10);
        // Injection chars collapse to underscores and share one bucket.
        limiter.record_attempt("x\n1");
        limiter.record_attempt("x\r1");
        limiter.record_attempt("x\t1");
        let map = limiter.attempts.lock().unwrap();
        assert_eq!(map["x_1"].attempts.len(), 3);
        // Long identifiers truncate at 256 chars.
        drop(map);
        limiter.record_attempt(&"y".repeat(300));
        let map = limiter.attempts.lock().unwrap();
        assert!(map.keys().any(|k| k.len() == 256));
    }
}
