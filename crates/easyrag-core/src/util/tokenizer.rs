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
}
