//! Tokenizer interface and implementations.
//! Uses tiktoken-rs (Rust port of tiktoken, same cl100k_base BPE as jtokkit).

use crate::CoreError;
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};
use tiktoken_rs::CoreBPE;

/// Tokenizer interface equivalent (encode/decode).
pub trait TokenizerInterface: Send + Sync {
    fn encode(&self, content: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> Result<String, CoreError>;

    /// Decode a window that may be cut inside a character — see
    /// [`Tokenizer::decode_window`].
    ///
    /// The default keeps strict [`decode`](TokenizerInterface::decode)
    /// semantics and reports nothing for a window it cannot decode; an encoder
    /// that can reach its raw token bytes overrides this to drop the partial
    /// character instead.
    fn decode_window(&self, tokens: &[u32]) -> String {
        self.decode(tokens).unwrap_or_default()
    }
}

/// A wrapper around a tokenizer providing a consistent interface.
pub struct Tokenizer {
    pub model_name: String,
    inner: Box<dyn TokenizerInterface>,
}

impl Tokenizer {
    pub fn new(model_name: impl Into<String>, inner: Box<dyn TokenizerInterface>) -> Self {
        Self {
            model_name: model_name.into(),
            inner,
        }
    }

    pub fn encode(&self, content: &str) -> Vec<u32> {
        self.inner.encode(content)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String, CoreError> {
        self.inner.decode(tokens)
    }

    /// Decode a token window that may be cut inside a character.
    ///
    /// Byte-level BPE splits one multi-byte character across several tokens, so
    /// a window sliced at an arbitrary token boundary routinely ends — and, for
    /// the window that follows it, begins — inside a character. Strict
    /// [`decode`](Tokenizer::decode) rejects those bytes
    /// (`incomplete utf-8 byte sequence`), which turned a chunk boundary into a
    /// whole-document ingestion failure.
    ///
    /// The partial character is *dropped* here rather than replaced: it cannot
    /// be represented in a `String` without inventing characters, and the
    /// overlapping neighbouring window still carries it. Text whose token
    /// boundaries are all character boundaries (ASCII, and any window that
    /// happens not to cut) decodes byte-identically to
    /// [`decode`](Tokenizer::decode).
    ///
    /// Never fails: a caller sliding a fixed-size window over a document must
    /// not lose that document to a boundary that is off by a byte.
    pub fn decode_window(&self, tokens: &[u32]) -> String {
        self.inner.decode_window(tokens)
    }

    pub fn count_tokens(&self, content: &str) -> usize {
        self.encode(content).len()
    }
}

/// tiktoken-based tokenizer implementation.
/// Default model: gpt-4o-mini (uses cl100k_base encoding), mirroring the
/// jtokkit fallback behaviour of the original implementation.
pub struct TiktokenEncoder {
    bpe: CoreBPE,
}

impl TiktokenEncoder {
    pub fn new(model_name: &str) -> Self {
        Self {
            bpe: resolve_encoding(model_name),
        }
    }

    /// The bytes behind `tokens`, without the UTF-8 validation
    /// [`decode`](TokenizerInterface::decode) applies.
    ///
    /// `_decode_native_and_split` is tiktoken-rs's per-token byte accessor —
    /// the underscore is the crate's own "not the tidy surface" marker, but it
    /// is public and it is the only way to reach the bytes of a token whose
    /// sequence is not valid UTF-8 on its own. `tokens` must therefore come
    /// from [`encode`](TokenizerInterface::encode): an id outside the vocabulary
    /// is not an error here, it panics inside tiktoken-rs.
    fn token_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.bpe
            ._decode_native_and_split(tokens.to_vec())
            .flatten()
            .collect()
    }
}

/// Longest valid UTF-8 slice of `bytes`, dropping a partial character at either
/// edge.
///
/// The head can only be a character whose leading bytes belong to an earlier
/// window (so it starts with continuation bytes); the tail can only be a
/// character that continues into the next window. Both are unrepresentable, and
/// both are recovered by the overlapping neighbouring window.
fn trim_partial_chars(bytes: &[u8]) -> String {
    // Continuation bytes (0b10xxxxxx) with no lead byte in front of them.
    let head = bytes
        .iter()
        .position(|byte| byte & 0xC0 != 0x80)
        .unwrap_or(bytes.len());
    let rest = &bytes[head..];
    match std::str::from_utf8(rest) {
        Ok(text) => text.to_string(),
        // `valid_up_to` is always a character boundary, so this slice is valid
        // by construction. A window of `encode` output can only fail at its
        // tail; trimming there keeps a corrupt window from taking the document
        // down with it either way.
        Err(e) => std::str::from_utf8(&rest[..e.valid_up_to()])
            .expect("valid_up_to is a character boundary")
            .to_string(),
    }
}

fn resolve_encoding(model_name: &str) -> CoreBPE {
    // Try to resolve the encoding for the model; fall back to cl100k_base on
    // any failure (same as JtokkitEncoder.resolveEncoding).
    tiktoken_rs::get_bpe_from_model(model_name)
        .or_else(|_| tiktoken_rs::get_bpe_from_model("gpt-4o-mini"))
        .unwrap_or_else(|_| tiktoken_rs::cl100k_base().expect("cl100k_base must be available"))
}

impl TokenizerInterface for TiktokenEncoder {
    fn encode(&self, content: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(content)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, CoreError> {
        self.bpe
            .decode(tokens.to_vec())
            .map_err(|e| CoreError::InvalidArgument(format!("tokenizer decode failed: {e}")))
    }

    fn decode_window(&self, tokens: &[u32]) -> String {
        match self.decode(tokens) {
            Ok(text) => text,
            Err(_) => trim_partial_chars(&self.token_bytes(tokens)),
        }
    }
}

/// Construct the default tokenizer (gpt-4o-mini → cl100k_base).
pub fn default_tokenizer() -> Tokenizer {
    Tokenizer::new("gpt-4o-mini", Box::new(TiktokenEncoder::new("gpt-4o-mini")))
}

/// Default tokenizer provider for the application (thread-safe singleton,
///).
static PROVIDER: Lazy<RwLock<Option<Arc<Tokenizer>>>> = Lazy::new(|| RwLock::new(None));

pub struct TokenizerProvider;

impl TokenizerProvider {
    pub fn get_tokenizer() -> Arc<Tokenizer> {
        {
            let guard = PROVIDER.read().unwrap();
            if let Some(t) = guard.as_ref() {
                return Arc::clone(t);
            }
        }
        let mut guard = PROVIDER.write().unwrap();
        if let Some(t) = guard.as_ref() {
            return Arc::clone(t);
        }
        let t = Arc::new(default_tokenizer());
        *guard = Some(Arc::clone(&t));
        t
    }

    pub fn set_tokenizer(tokenizer: Tokenizer) {
        let mut guard = PROVIDER.write().unwrap();
        *guard = Some(Arc::new(tokenizer));
    }
}

#[cfg(test)]
mod tests {
    // Ported from TokenizerTest.kt (10 cases).
    use super::*;

    #[test]
    fn encode_non_empty_text_returns_non_empty_tokens() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let tokens = tokenizer.encode("Hello, world!");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn encode_empty_text_returns_empty_tokens() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let tokens = tokenizer.encode("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn encode_decode_round_trip_preserves_text() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let original = "The quick brown fox jumps over the lazy dog.";
        let tokens = tokenizer.encode(original);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn encode_decode_round_trip_chinese_text() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let original = "你好世界，这是一个测试。";
        let tokens = tokenizer.encode(original);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn count_tokens_non_empty_text_returns_positive() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let count = tokenizer.count_tokens("Hello, world!");
        assert!(count > 0);
    }

    #[test]
    fn count_tokens_empty_text_returns_zero() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        assert_eq!(0, tokenizer.count_tokens(""));
    }

    #[test]
    fn count_tokens_longer_text_has_more_tokens() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let short_count = tokenizer.count_tokens("Hi");
        let long_count =
            tokenizer.count_tokens("This is a much longer sentence with many more words in it.");
        assert!(long_count > short_count);
    }

    #[test]
    fn encode_same_text_returns_same_tokens() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let text = "Deterministic test";
        let tokens1 = tokenizer.encode(text);
        let tokens2 = tokenizer.encode(text);
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn encode_different_text_returns_different_tokens() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let tokens1 = tokenizer.encode("Hello");
        let tokens2 = tokenizer.encode("Goodbye");
        assert_ne!(tokens1, tokens2);
    }

    #[test]
    fn tokenizer_provider_returns_same_instance() {
        let t1 = TokenizerProvider::get_tokenizer();
        let t2 = TokenizerProvider::get_tokenizer();
        assert!(Arc::ptr_eq(&t1, &t2));
    }

    // ===== decode_window: windows cut inside a character =====
    //
    // Byte-level BPE splits one CJK character across several tokens, so a
    // window sliced at an arbitrary token boundary routinely lands inside a
    // character. Strict `decode` rejects those bytes (`incomplete utf-8 byte
    // sequence`); that rejection used to abort the whole document at
    // `TokenSizeChunker`, which is what these tests pin down.

    /// A CJK paragraph taken from the document that hit this in production
    /// (`easyai/market_insights/stage1-2026-09-28-gold-crash-verdict.md`):
    /// cl100k_base splits characters like 慌 / 挤 / 兑 / 崩 across two tokens, so
    /// a window boundary can and does land inside one of them. A paragraph of
    /// only "whole-token" characters (e.g. 标签体系) would not reproduce it.
    fn cjk_text() -> String {
        "市场恐慌挤兑风险与金价崩盘的分母冲击".repeat(8)
    }

    #[test]
    fn strict_decode_rejects_a_window_that_cuts_a_character() {
        // Characterisation of the hazard `decode_window` exists for: this is
        // *not* a bug in `decode`, and it must keep behaving this way.
        let tokenizer = TokenizerProvider::get_tokenizer();
        let tokens = tokenizer.encode(&cjk_text());
        let cut = (1..tokens.len())
            .find(|&end| tokenizer.decode(&tokens[..end]).is_err())
            .expect("a CJK document must have a token boundary that cuts a character");

        let error = tokenizer.decode(&tokens[..cut]).unwrap_err();
        assert!(
            error.to_string().contains("utf-8"),
            "expected a UTF-8 boundary error, got: {error}"
        );
    }

    #[test]
    fn decode_window_recovers_a_window_ending_mid_character() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let original = cjk_text();
        let tokens = tokenizer.encode(&original);
        let cut = (1..tokens.len())
            .find(|&end| tokenizer.decode(&tokens[..end]).is_err())
            .expect("must find a cutting boundary");

        let text = tokenizer.decode_window(&tokens[..cut]);

        // Only the partial trailing character is dropped: the window still
        // decodes to a non-empty, replacement-character-free prefix.
        assert!(!text.is_empty(), "a cut window must not decode to nothing");
        assert!(!text.contains('\u{FFFD}'), "no replacement characters");
        assert!(original.starts_with(&text), "must be a prefix of the original");

        // The one character missing is the one the boundary landed in, and it
        // is missing precisely because it did not fit: the first `cut` tokens
        // stop inside it.
        let kept = text.chars().count();
        let through_dropped: String = original.chars().take(kept + 1).collect();
        assert_eq!(
            kept + 1,
            through_dropped.chars().count(),
            "the window must not have consumed the whole document"
        );
        assert!(
            tokenizer.encode(&through_dropped).len() > cut,
            "the dropped character must be the one the boundary cut into"
        );
    }

    #[test]
    fn decode_window_recovers_a_window_starting_mid_character() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        let original = cjk_text();
        let tokens = tokenizer.encode(&original);
        // A window that begins on a continuation byte fails at index 0.
        let start = (1..tokens.len())
            .find(|&start| tokenizer.decode(&tokens[start..]).is_err())
            .expect("must find a start that lands inside a character");

        let text = tokenizer.decode_window(&tokens[start..]);

        assert!(!text.is_empty(), "a leading cut must not empty the window");
        assert!(!text.contains('\u{FFFD}'), "no replacement characters");
        assert!(
            original.contains(&text),
            "the recovered window must be a substring of the original"
        );
    }

    #[test]
    fn decode_window_matches_strict_decode_when_no_character_is_cut() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        // ASCII text: every token boundary is a character boundary, so the
        // window decode must be byte-identical to the strict path — this is
        // what keeps existing chunking behaviour untouched.
        let original = "The quick brown fox.";
        let tokens = tokenizer.encode(original);
        for end in 1..=tokens.len() {
            assert_eq!(
                tokenizer.decode(&tokens[..end]).unwrap(),
                tokenizer.decode_window(&tokens[..end])
            );
        }
    }

    #[test]
    fn decode_window_on_empty_tokens_is_empty() {
        let tokenizer = TokenizerProvider::get_tokenizer();
        assert!(tokenizer.decode_window(&[]).is_empty());
    }
}
