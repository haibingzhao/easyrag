//! Shared HTTP transport for the LLM clients (Phase 4.1 / 4.2).
//!
//! `openai.rs`, `ollama.rs` and `rerank.rs` each built a byte-identical
//! `reqwest::Client`, each hand-rolled the same newline-splitting stream
//! skeleton, each mapped `reqwest::Error` the same way, and both chat clients
//! each spelled out the same system+history message loop. This module
//! single-sources exactly those things.
//!
//! # What is deliberately NOT abstracted
//!
//! Provider-specific shapes stay provider-specific:
//!
//! - request body shape (Ollama's nested `options`/`num_predict` vs OpenAI's
//!   top-level `max_tokens`/`response_format`/`enable_thinking`),
//! - tool-call pairing (OpenAI by synthesized `tool_call_id`, Ollama by
//!   `function.index`) and `arguments` encoding (JSON string vs JSON object),
//! - response pointers (`/choices/0/...` vs `/message/...`),
//! - framing (SSE `data: ` prefix + `[DONE]` sentinel vs bare NDJSON),
//! - embedding response shape (`data[].embedding`, which *errors* when absent,
//!   vs `embeddings[]`, which silently yields nothing).
//!
//! Two POST flavours remain, and they now differ *only* in the auth header:
//! [`LlmTransport::post_bearer`] and [`LlmTransport::post_anonymous`] share one
//! status check, so a non-2xx response becomes `ApiStatus` whose message embeds
//! the code — which is what lets [`is_retryable_message`] see "502"/"503".
//!
//! They were not always alike. The Ollama path used to skip the status check
//! entirely, so a 503 body was parsed as ordinary text and degraded to empty
//! content: a visibly failing server produced `Ok("")` for chat and
//! `Ok(vec![])` for embeddings. Making the two flavours agree is the behaviour
//! change `[BC-4]` exists to make (defect A-7), and it is deliberately kept out
//! of the `[equivalent]` Phase 4.1/4.2 work — which is also why the retry loop
//! and circuit breaker live here rather than in `openai.rs`: adopting them for
//! a second provider must not mean re-deriving them.
//!
//! # Dispatch discipline
//!
//! [`ChatWireFormat`] is a *generic* parameter ([`stream_chat`] is
//! monomorphized per client), never `dyn`: the provider is fixed at
//! construction, so per-line stream parsing costs zero vtable hops. The plan
//! sketched this as `LlmTransport<W: ChatWireFormat>`; the generic lives on the
//! free function instead because [`LlmTransport`] is also shared by the rerank
//! client, which has no chat wire format at all and would otherwise need a
//! phantom marker type. Same static dispatch, no dead type parameter.

use easyrag_core::concurrency::{breaker_result, CircuitBreaker};
use easyrag_core::llm::service::{LlmRequest, MessageRole};
use easyrag_core::util::text::truncate_with_ellipsis;
use futures::stream::BoxStream;
use futures::StreamExt;
use serde_json::{json, Value as JsonValue};
use std::time::Duration;

// ===== client construction =====

/// The three fields every LLM client config duplicates verbatim. Deriving an
/// endpoint from a config and building the client from the endpoint is what
/// removes the triplicated `reqwest::Client::builder()` block (Phase 4.4).
#[derive(Clone, Debug)]
pub struct HttpEndpointConfig {
    pub base_url: String,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Connection timeout.
    pub connect_timeout: Duration,
}

impl HttpEndpointConfig {
    pub fn new(
        base_url: impl Into<String>,
        timeout: Duration,
        connect_timeout: Duration,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            timeout,
            connect_timeout,
        }
    }

    /// Build the [`LlmTransport`] for this endpoint.
    pub fn build_transport(&self) -> easyrag_core::Result<LlmTransport> {
        LlmTransport::new(self.clone())
    }
}

/// Single source for the shared `reqwest::Client`.
///
/// The error wording (`http client: {e}`) is the one all three clients already
/// produced, so a mis-configured TLS backend still fails identically.
pub fn build_http_client(
    timeout: Duration,
    connect_timeout: Duration,
) -> easyrag_core::Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .timeout(timeout)
        .build()
        .map_err(|e| easyrag_core::CoreError::Internal(format!("http client: {e}")))
}

/// Map a `reqwest` send/receive failure onto the crate's error type.
///
/// Timeouts stay distinguishable from connection failures because the retry
/// classifier keys on the variant (and on the word "timeout" in the message).
pub fn map_reqwest_error(e: reqwest::Error) -> easyrag_core::CoreError {
    if e.is_timeout() {
        easyrag_core::CoreError::ApiTimeout
    } else {
        easyrag_core::CoreError::ApiConnection(e.to_string())
    }
}

/// Owned HTTP layer: one pooled `reqwest::Client` plus the base URL every path
/// is appended to. Cloning is cheap (`reqwest::Client` is an `Arc` inside),
/// which the streaming path relies on to move the client into a `'static`
/// stream.
#[derive(Clone)]
pub struct LlmTransport {
    http: reqwest::Client,
    base_url: String,
}

impl LlmTransport {
    pub fn new(endpoint: HttpEndpointConfig) -> easyrag_core::Result<Self> {
        Ok(Self {
            http: build_http_client(endpoint.timeout, endpoint.connect_timeout)?,
            base_url: endpoint.base_url,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Escape hatch for callers that need to build a request themselves
    /// (currently only the rerank client, whose `Authorization` header is
    /// conditional on a non-empty key).
    pub fn client(&self) -> &reqwest::Client {
        &self.http
    }

    pub fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Shared body of both checked POST flavours: send, then turn a non-2xx
    /// response into `ApiStatus` whose message embeds the code and the first
    /// 500 bytes of the body.
    ///
    /// `auth` decides the header set, and nothing else: `Some` sends
    /// `Authorization: Bearer` (OpenAI-compatible), `None` sends an explicit
    /// `Content-Type` and no credentials (Ollama, a local daemon). `.json()`
    /// already sets `Content-Type`, but the explicit header is kept on the
    /// anonymous path because that is byte-for-byte what the Ollama client sent
    /// before Phase 4.1.
    ///
    /// `provider` only feeds the message (`"{provider} API returned status …"`),
    /// so an Azure/Bailian deployment keeps the wording its callers match on.
    async fn post_checked(
        &self,
        path: &str,
        auth: Option<&str>,
        body: &JsonValue,
        provider: &str,
    ) -> easyrag_core::Result<String> {
        let mut request = self.http.post(self.url(path)).json(body);
        request = match auth {
            Some(key) => request.bearer_auth(key),
            None => request.header("Content-Type", "application/json"),
        };
        let response = request.send().await.map_err(map_reqwest_error)?;
        let status = response.status();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            return Err(easyrag_core::CoreError::api_status(
                status.as_u16(),
                format!(
                    "{provider} API returned status {}: {}",
                    status.as_u16(),
                    truncate_with_ellipsis(&text, 500)
                ),
                None,
            ));
        }
        Ok(text)
    }

    /// OpenAI-compatible flavour: bearer auth plus the shared status check.
    pub async fn post_bearer(
        &self,
        path: &str,
        api_key: &str,
        body: &JsonValue,
        provider: &str,
    ) -> easyrag_core::Result<String> {
        self.post_checked(path, Some(api_key), body, provider).await
    }

    /// Ollama flavour: no credentials, same status check.
    ///
    /// This used to be `post_plain`, which skipped the status check entirely —
    /// a 503 body was returned as ordinary text and the parsers degraded it to
    /// empty content, so a visibly failing server produced `Ok("")` (and, for
    /// embeddings, `Ok(vec![])`). `[BC-4]` removes that: the check is what makes
    /// the failure visible to [`is_retryable_message`] and therefore retryable,
    /// and without it the retry loop and circuit breaker added in the same
    /// change could never trigger.
    pub async fn post_anonymous(
        &self,
        path: &str,
        body: &JsonValue,
        provider: &str,
    ) -> easyrag_core::Result<String> {
        self.post_checked(path, None, body, provider).await
    }

    /// Rerank flavour: `Authorization` only when a key is configured, and the
    /// raw `reqwest::Error` handed back so the caller can log it and degrade to
    /// the original document order.
    pub async fn post_optional_bearer(
        &self,
        path: &str,
        api_key: Option<&str>,
        body: &JsonValue,
    ) -> Result<String, reqwest::Error> {
        let mut request = self
            .http
            .post(self.url(path))
            .header("Content-Type", "application/json")
            .json(body);
        if let Some(key) = api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }
        let response = request.send().await?;
        response.text().await
    }

    /// Dispatch to the POST flavour a client is configured for. One `match` per
    /// attempt, not per stream line, so the branch costs nothing measurable and
    /// the resilience helpers below can stay provider-agnostic.
    pub async fn post(
        &self,
        flavour: &PostFlavour,
        path: &str,
        body: &JsonValue,
    ) -> easyrag_core::Result<String> {
        match flavour {
            PostFlavour::Bearer {
                api_key,
                provider,
            } => self.post_bearer(path, api_key, body, provider).await,
            PostFlavour::Anonymous { provider } => self.post_anonymous(path, body, provider).await,
        }
    }

    /// Circuit breaker + retry around a single POST-and-parse (Phase 4.3).
    ///
    /// `complete`, `complete_with_tools` and `embed` on the OpenAI client each
    /// hand-wrote the same two-level nesting — an outer `breaker.execute(|| {
    /// let body = body.clone(); async move { … } })` around an inner retry loop
    /// cloning the body again — and differed only in path, log label and
    /// response parser. Both clones are still required (the breaker re-runs its
    /// block per trial, the retry loop re-runs its block per attempt, and each
    /// `async move` consumes the body), but they now live in exactly one place,
    /// and the `BreakerError`→`CoreError` collapse is `core::breaker_result`.
    ///
    /// `parse` is a bare `fn` pointer rather than a closure: it captures
    /// nothing, so this stays monomorphized with no allocation per attempt.
    /// Borrowing `self`/`path`/`flavour` into the `async move` blocks is sound
    /// because neither `execute` nor [`execute_with_retry`] requires a `'static`
    /// future — they are awaited inside this call.
    #[allow(clippy::too_many_arguments)]
    pub async fn execute_resilient<T>(
        &self,
        flavour: &PostFlavour,
        breaker: &CircuitBreaker,
        policy: &RetryPolicy,
        request_name: &str,
        path: &str,
        body: JsonValue,
        parse: fn(&str) -> easyrag_core::Result<T>,
    ) -> easyrag_core::Result<T> {
        let result = breaker
            .execute(|| {
                let body = body.clone();
                async move {
                    execute_with_retry(policy, request_name, || {
                        let body = body.clone();
                        async move {
                            let response = self.post(flavour, path, &body).await?;
                            parse(&response)
                        }
                    })
                    .await
                }
            })
            .await;
        breaker_result(result)
    }
}

// ===== resilience =====

/// Which POST flavour a client uses, and the credentials it needs. Owned so it
/// can be handed to [`LlmTransport::execute_resilient`] alongside the body.
#[derive(Clone, Debug)]
pub enum PostFlavour {
    /// Bearer auth, status checked, non-2xx surfaced as `ApiStatus`. `provider`
    /// only feeds that error message.
    Bearer {
        api_key: String,
        provider: &'static str,
    },
    /// No credentials, status still checked (Ollama, `[BC-4]`).
    Anonymous { provider: &'static str },
}

/// Exponential base for the retry backoff. Matches
/// `core::RetryConfig::default().exponential_base`, so the LLM clients and the
/// storage-layer retries introduced in Phase 3.10 grow at the same rate.
const EXPONENTIAL_BASE: u64 = 2;

/// Ceiling on a single backoff, in milliseconds. Matches
/// `core::RetryConfig::default().max_delay` (60s). Without it a large attempt
/// count would saturate `u64` and then sleep essentially forever.
const MAX_DELAY_MS: u64 = 60_000;

/// Retry budget and backoff base for LLM HTTP calls.
///
/// Extracted from `OpenAiConfig`'s two fields so the same curve can be applied
/// to any client rather than re-implemented per provider.
#[derive(Clone, Copy, Debug)]
pub struct RetryPolicy {
    /// Attempt budget. Values below 1 are treated as 1 (a single try).
    pub max_retries: u32,
    /// Base of the backoff, in milliseconds.
    pub delay_base_ms: u64,
}

impl RetryPolicy {
    pub fn max_attempts(&self) -> u64 {
        self.max_retries.max(1) as u64
    }

    /// The deterministic half of the curve: `base * 2^(attempt-1)`, capped at
    /// [`MAX_DELAY_MS`].
    ///
    /// `[BC-3]`. This used to be `base * attempt^2` — 1000/4000/9000ms at the
    /// default base — which grows faster than exponential early on and is what
    /// the OpenAI client shipped with. Both saturating operations matter:
    /// `saturating_pow` for a large attempt count and `saturating_mul` for a
    /// large configured base, so a misconfigured policy degrades to the cap
    /// instead of wrapping around to a tiny (or zero) delay.
    pub fn backoff_ms(&self, attempt: u64) -> u64 {
        // 2^32 already exceeds any useful delay, so clamping the exponent keeps
        // the arithmetic in range without changing the capped result.
        let exponent = attempt.saturating_sub(1).min(32) as u32;
        let factor = EXPONENTIAL_BASE.saturating_pow(exponent);
        self.delay_base_ms
            .saturating_mul(factor)
            .min(MAX_DELAY_MS)
    }

    /// Apply *equal jitter*: half the backoff is fixed, the other half is scaled
    /// by `fraction` (clamped to `0.0..=1.0`).
    ///
    /// Equal rather than full jitter because full jitter (`rand(0, backoff)`) can
    /// draw a near-zero delay and immediately re-hit an upstream that is already
    /// shedding load; keeping a fixed floor preserves the point of backing off
    /// while still de-synchronizing concurrent clients.
    ///
    /// Taking the fraction as a parameter is what makes the curve testable —
    /// `0.0` is the floor and `1.0` the full backoff, both exactly.
    pub fn delay_ms_with_jitter(&self, attempt: u64, fraction: f64) -> u64 {
        let backoff = self.backoff_ms(attempt);
        let floor = backoff / 2;
        let span = backoff - floor;
        floor + (span as f64 * fraction.clamp(0.0, 1.0)) as u64
    }

    /// Backoff before the next attempt, in milliseconds, including jitter.
    pub fn delay_ms(&self, attempt: u64) -> u64 {
        self.delay_ms_with_jitter(attempt, rand::random::<f64>())
    }
}

/// Keyword-based transient classification for LLM failures.
///
/// Matches on the rendered error rather than the variant because the useful
/// signal often arrives inside an `ApiStatus` message ("503", "rate limit")
/// produced upstream by [`LlmTransport::post_bearer`].
pub fn is_retryable_message(message: &str) -> bool {
    let msg = message.to_lowercase();
    msg.contains("rate limit")
        || msg.contains("timeout")
        || msg.contains("connection")
        || msg.contains("503")
        || msg.contains("502")
}

/// Retry `block` while it fails with a transient error and attempts remain.
///
/// The warn line and the "no attempts left" fallback message are reproduced
/// verbatim from `openai.rs`, which operators and the retry tests key on.
pub async fn execute_with_retry<T, F, Fut>(
    policy: &RetryPolicy,
    request_name: &str,
    mut block: F,
) -> easyrag_core::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = easyrag_core::Result<T>>,
{
    let max = policy.max_attempts();
    let mut last_error: Option<easyrag_core::CoreError> = None;
    for attempt in 1..=max {
        match block().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                last_error = Some(e);
                let err = last_error.as_ref().unwrap();
                if is_retryable_message(&err.to_string()) && attempt < max {
                    let delay = policy.delay_ms(attempt);
                    tracing::warn!(
                        "{request_name} request failed (attempt {attempt}/{max}), \
                         retrying in {delay}ms: {err}"
                    );
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                } else {
                    break;
                }
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        easyrag_core::CoreError::Internal(format!("{request_name} request failed"))
    }))
}

// ===== request assembly shared by the chat clients =====

/// Wire name of a chat role. Spelled out four times across the two clients
/// before Phase 4.1 (twice in the completion builders, twice in the tool-call
/// builders).
pub fn role_str(role: MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
    }
}

/// Build the message list: system prompt (when present) + history + the
/// caller-supplied final user message.
///
/// The last message is a parameter rather than always `request.prompt` because
/// the vision path substitutes a content *array* (text + `image_url`) for the
/// plain string. Key order inside each object is preserved from the original
/// builders, which the request-body goldens lock byte for byte.
pub fn build_chat_messages(request: &LlmRequest, user_message: JsonValue) -> Vec<JsonValue> {
    let mut messages = Vec::new();
    if let Some(sp) = &request.system_prompt {
        messages.push(json!({ "role": "system", "content": sp }));
    }
    for msg in &request.history {
        messages.push(json!({ "role": role_str(msg.role), "content": msg.content }));
    }
    messages.push(user_message);
    messages
}

/// Convert a JSON array of numbers into `f32`s, dropping anything non-numeric.
///
/// Both embedding paths applied this same filter-map to their own differently
/// named array (`data[].embedding` vs `embeddings[]`), and both *dropped*
/// non-numeric entries rather than zero-filling — so a malformed element
/// shortens the vector instead of holding its position. That is the pre-Phase
/// 4.2 behaviour on both sides and is reproduced verbatim here; changing it
/// would shift every stored embedding and is not this PR's business.
pub fn f32_array(values: &[JsonValue]) -> Vec<f32> {
    values
        .iter()
        .filter_map(|v| v.as_f64())
        .map(|f| f as f32)
        .collect()
}

// ===== streaming =====

/// How a streamed chat request authenticates. The two variants produce the
/// exact header sets the clients produced before Phase 4.1: OpenAI relied on
/// `.bearer_auth()` (which also sets `Content-Type` via `.json()`), Ollama set
/// `Content-Type` explicitly and sent no credentials.
#[derive(Clone, Debug)]
pub enum StreamAuth {
    /// `Authorization: Bearer <key>`.
    Bearer(String),
    /// No credentials; an explicit `Content-Type: application/json` header.
    ContentType,
}

/// The per-provider half of a streamed chat call: where to POST, how to
/// authenticate, what body to send, and how to turn one raw line of the
/// response into content.
///
/// Implemented by the clients themselves (not by a separate config newtype) so
/// `&self` gives the impls access to the model name and API key they already
/// own, and so [`stream_chat`] monomorphizes on the concrete client type.
pub trait ChatWireFormat {
    /// Chat endpoint, appended to the transport's base URL.
    const CHAT_PATH: &'static str;
    /// JSON pointer at the completion text in a non-streaming response.
    const CONTENT_POINTER: &'static str;
    /// Prefix of the warn line emitted when the stream connection fails. Kept
    /// per provider because operators grep for the existing wording
    /// (`"SSE connection failed"` / `"Ollama stream connection failed"`).
    const STREAM_FAILURE_LOG: &'static str;

    fn transport(&self) -> &LlmTransport;

    fn stream_auth(&self) -> StreamAuth;

    /// Provider-specific request body. Not unified — see the module docs.
    fn build_chat_body(&self, request: &LlmRequest, stream: bool) -> JsonValue;

    /// Turn one raw line (or, when `is_tail`, the unterminated remainder of the
    /// buffer) into a content delta, or `None` to skip it.
    ///
    /// The line arrives **unnormalized**, trailing `\n` included: OpenAI trims
    /// only `['\r', '\n']` and requires the `data: ` prefix, Ollama trims all
    /// whitespace and skips blanks. Normalizing here would change what either
    /// provider accepts.
    ///
    /// `is_tail` exists to preserve one asymmetry: Ollama logs unparseable
    /// newline-delimited lines at `debug` but stays silent about an unparseable
    /// tail. OpenAI ignores the flag.
    ///
    /// No `&self` receiver — the parsers are pure, and keeping them associated
    /// functions lets [`stream_chat`] call them from inside a `'static` stream.
    fn parse_stream_line(line: &str, is_tail: bool) -> Option<String>;

    /// Extract the completion text from a non-streaming response. The default
    /// is the four steps both clients already spelled out, parameterized only
    /// by [`Self::CONTENT_POINTER`]; malformed JSON degrades to an empty string
    /// rather than an error.
    fn parse_completion(response: &str) -> String {
        serde_json::from_str::<JsonValue>(response)
            .ok()
            .and_then(|json| {
                json
                    .pointer(Self::CONTENT_POINTER)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_default()
    }
}

/// The streaming skeleton both chat clients duplicated: POST, warn-and-return
/// on a connection failure, then split `bytes_stream()` on `\n` and hand each
/// line to the provider's parser, finishing with the unterminated remainder.
///
/// Everything the `'static` stream needs is cloned out of `wire` up front, so
/// the returned stream borrows nothing.
pub fn stream_chat<W: ChatWireFormat>(wire: &W, request: LlmRequest) -> BoxStream<'static, String> {
    let transport = wire.transport().clone();
    let http = transport.client().clone();
    let url = transport.url(W::CHAT_PATH);
    let auth = wire.stream_auth();
    let body = wire.build_chat_body(&request, true);
    let failure_log = W::STREAM_FAILURE_LOG;
    Box::pin(async_stream::stream! {
        let builder = http.post(&url).json(&body);
        let builder = match auth {
            StreamAuth::Bearer(key) => builder.bearer_auth(key),
            StreamAuth::ContentType => builder.header("Content-Type", "application/json"),
        };
        let response = match builder.send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("{failure_log}: {e}");
                return;
            }
        };
        let mut chunks = response.bytes_stream();
        let mut buffer = String::new();
        while let Some(chunk) = chunks.next().await {
            let Ok(bytes) = chunk else { break };
            buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..=pos).collect();
                if let Some(content) = W::parse_stream_line(&line, false) {
                    yield content;
                }
            }
        }
        // Tail without trailing newline.
        if let Some(content) = W::parse_stream_line(&buffer, true) {
            yield content;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use easyrag_core::CoreError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ===== RetryPolicy =====

    /// `[BC-3]` Pins the deterministic backoff curve. The pre-change shape was
    /// `base * attempt^2` (1000/4000/9000 at the default base); this asserts the
    /// exponential one, so it is the second half of the red-first evidence — it
    /// passed before the change only in its quadratic form, which the companion
    /// `bc3_total_backoff_is_exponential_with_equal_jitter` caught failing.
    #[test]
    fn bc3_backoff_doubles_per_attempt() {
        let policy = RetryPolicy {
            max_retries: 4,
            delay_base_ms: 1000,
        };
        assert_eq!(1000, policy.backoff_ms(1));
        assert_eq!(2000, policy.backoff_ms(2));
        assert_eq!(4000, policy.backoff_ms(3));
        assert_eq!(8000, policy.backoff_ms(4));
    }

    /// The cap is what keeps a large attempt count (or a large configured base)
    /// from saturating `u64` and sleeping forever. 60s matches
    /// `core::RetryConfig::default().max_delay`.
    #[test]
    fn bc3_backoff_is_capped_and_never_overflows() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        assert_eq!(60_000, policy.backoff_ms(10));
        assert_eq!(60_000, policy.backoff_ms(1000));

        // A misconfigured base saturates to the cap instead of wrapping to a
        // tiny delay, which would turn a backoff into a hot retry loop.
        let huge = RetryPolicy { max_retries: 3, delay_base_ms: u64::MAX };
        assert_eq!(60_000, huge.backoff_ms(1));
        assert_eq!(60_000, huge.backoff_ms(3));
    }

    /// Equal jitter: the floor is exactly half the backoff and the ceiling is
    /// the whole of it. Passing the fraction in is what makes this assertable
    /// despite the production path drawing a random one.
    #[test]
    fn bc3_equal_jitter_spans_half_to_all_of_the_backoff() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        // attempt 3 -> backoff 4000, so the band is 2000..=4000.
        assert_eq!(2000, policy.delay_ms_with_jitter(3, 0.0));
        assert_eq!(3000, policy.delay_ms_with_jitter(3, 0.5));
        assert_eq!(4000, policy.delay_ms_with_jitter(3, 1.0));
        // Out-of-range fractions are clamped, not extrapolated.
        assert_eq!(2000, policy.delay_ms_with_jitter(3, -1.0));
        assert_eq!(4000, policy.delay_ms_with_jitter(3, 2.0));
    }

    /// The production entry point really does randomize — and stays inside the
    /// band. Sampled rather than asserted once, so a `delay_ms` that ignored the
    /// fraction (or returned the floor every time) fails here.
    #[test]
    fn bc3_delay_ms_stays_in_band_and_varies() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        let samples: Vec<u64> = (0..64).map(|_| policy.delay_ms(3)).collect();
        for s in &samples {
            assert!((2000..=4000).contains(s), "{s} outside the jitter band");
        }
        assert!(
            samples.windows(2).any(|w| w[0] != w[1]),
            "64 draws produced no variation: {:?}",
            &samples[..8]
        );
    }

    /// A budget below one attempt is clamped, not allowed to mean "never call".
    #[test]
    fn max_attempts_never_drops_below_one() {
        assert_eq!(1, RetryPolicy { max_retries: 0, delay_base_ms: 1000 }.max_attempts());
        assert_eq!(1, RetryPolicy { max_retries: 1, delay_base_ms: 1000 }.max_attempts());
        assert_eq!(5, RetryPolicy { max_retries: 5, delay_base_ms: 1000 }.max_attempts());
    }

    // ===== is_retryable_message =====

    #[test]
    fn retryable_keywords_match_case_insensitively() {
        for message in [
            "rate limit exceeded",
            "Rate Limit",
            "request TIMEOUT",
            "Connection reset by peer",
            "OpenAI API returned status 503: busy",
            "OpenAI API returned status 502: bad gateway",
        ] {
            assert!(is_retryable_message(message), "expected retryable: {message}");
        }
    }

    #[test]
    fn non_transient_failures_are_not_retried() {
        for message in [
            "OpenAI API returned status 401: invalid api key",
            "OpenAI API returned status 400: model not found",
            "Embedding API returned empty data: {}",
        ] {
            assert!(!is_retryable_message(message), "expected fatal: {message}");
        }
    }

    // ===== execute_with_retry =====

    /// A counter shared with the block so the tests can assert how many times
    /// the HTTP call was actually made — retry counts are the observable
    /// behaviour here, not just the final `Result`.
    fn counter() -> Arc<AtomicUsize> {
        Arc::new(AtomicUsize::new(0))
    }

    fn hits(calls: &Arc<AtomicUsize>) -> usize {
        calls.load(Ordering::SeqCst)
    }

    /// `ApiStatus` renders as its `message` verbatim, and `post_bearer` is what
    /// puts the status code into that message — the retry matcher downstream
    /// relies on that chain, so the tests build errors the same way.
    fn status(code: u16, message: &str) -> CoreError {
        CoreError::ApiStatus {
            message: message.to_string(),
            status_code: code,
            request_id: None,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn retries_a_transient_failure_then_returns_the_success() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        let calls = counter();
        let inner = Arc::clone(&calls);
        let result = execute_with_retry(&policy, "OpenAI", move || {
            let inner = Arc::clone(&inner);
            async move {
                let n = inner.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(CoreError::ApiConnection("connection reset".to_string()))
                } else {
                    Ok("ok".to_string())
                }
            }
        })
        .await;
        assert_eq!("ok", result.expect("third attempt must succeed"));
        assert_eq!(3, hits(&calls));
    }

    #[tokio::test(start_paused = true)]
    async fn a_fatal_failure_is_returned_without_a_second_attempt() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        let calls = counter();
        let inner = Arc::clone(&calls);
        let result = execute_with_retry::<String, _, _>(&policy, "OpenAI", move || {
            let inner = Arc::clone(&inner);
            async move {
                inner.fetch_add(1, Ordering::SeqCst);
                Err(status(401, "OpenAI API returned status 401: invalid api key"))
            }
        })
        .await;
        let err = result.expect_err("must fail");
        assert!(matches!(err, CoreError::ApiStatus { status_code: 401, .. }), "got {err:?}");
        assert_eq!(1, hits(&calls));
    }

    /// The budget is exhausted exactly once and the *last* error survives —
    /// callers and logs key on the upstream wording, not on a synthesized one.
    #[tokio::test(start_paused = true)]
    async fn exhausting_the_budget_returns_the_last_error() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        let calls = counter();
        let inner = Arc::clone(&calls);
        let result = execute_with_retry::<String, _, _>(&policy, "OpenAI", move || {
            let inner = Arc::clone(&inner);
            async move {
                let n = inner.fetch_add(1, Ordering::SeqCst) + 1;
                Err(status(503, &format!("OpenAI API returned status 503: busy #{n}")))
            }
        })
        .await;
        assert_eq!(
            "OpenAI API returned status 503: busy #3",
            result.expect_err("must fail").to_string()
        );
        assert_eq!(3, hits(&calls));
    }

    #[tokio::test(start_paused = true)]
    async fn a_zero_budget_still_makes_exactly_one_attempt() {
        let policy = RetryPolicy { max_retries: 0, delay_base_ms: 1000 };
        let calls = counter();
        let inner = Arc::clone(&calls);
        let result = execute_with_retry::<String, _, _>(&policy, "OpenAI", move || {
            let inner = Arc::clone(&inner);
            async move {
                inner.fetch_add(1, Ordering::SeqCst);
                Err(CoreError::ApiConnection("timeout".to_string()))
            }
        })
        .await;
        assert!(result.is_err());
        assert_eq!(1, hits(&calls));
    }

    /// Total elapsed time under the paused clock equals the sum of the sleeps,
    /// which is the curve observed from outside.
    ///
    /// `[BC-3]` red-first probe, written before the curve changed: with the
    /// quadratic backoff this measured exactly 5000ms (1000 + 4000, no sleep
    /// after the final attempt). Exponential backoff with equal jitter must land
    /// in 1500..=3000ms instead — attempt 1 contributes 500..=1000 and attempt 2
    /// contributes 1000..=2000. Asserting the new band therefore fails against
    /// the old curve, which is the evidence that the behaviour really changed.
    #[tokio::test(start_paused = true)]
    async fn bc3_total_backoff_is_exponential_with_equal_jitter() {
        let policy = RetryPolicy { max_retries: 3, delay_base_ms: 1000 };
        let start = tokio::time::Instant::now();
        let result = execute_with_retry::<String, _, _>(&policy, "OpenAI", || async {
            Err(CoreError::ApiConnection("connection refused".to_string()))
        })
        .await;
        assert!(result.is_err());
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(1500) && elapsed <= Duration::from_millis(3000),
            "expected the exponential+jitter band 1500..=3000ms, got {elapsed:?}"
        );
    }

    // ===== request assembly =====

    #[test]
    fn role_str_covers_every_variant() {
        assert_eq!("system", role_str(MessageRole::System));
        assert_eq!("user", role_str(MessageRole::User));
        assert_eq!("assistant", role_str(MessageRole::Assistant));
    }

    #[test]
    fn build_chat_messages_orders_system_history_then_the_supplied_message() {
        let mut request = LlmRequest::new("ignored here");
        request.system_prompt = Some("be brief".to_string());
        request.history = vec![
            easyrag_core::llm::service::ChatMessage {
                role: MessageRole::User,
                content: "earlier".to_string(),
            },
            easyrag_core::llm::service::ChatMessage {
                role: MessageRole::Assistant,
                content: "answer".to_string(),
            },
        ];
        // The final message arrives pre-shaped: the vision path substitutes a
        // content *array* for the string, which is why it is a parameter.
        let messages = build_chat_messages(
            &request,
            json!({ "role": "user", "content": "the actual question" }),
        );
        assert_eq!(
            vec![
                json!({ "role": "system", "content": "be brief" }),
                json!({ "role": "user", "content": "earlier" }),
                json!({ "role": "assistant", "content": "answer" }),
                json!({ "role": "user", "content": "the actual question" }),
            ],
            messages
        );
    }

    /// A request with no system prompt yields no system message — the field is
    /// `Option` precisely so absent stays absent on the wire.
    #[test]
    fn build_chat_messages_omits_an_absent_system_prompt() {
        let request = LlmRequest::new("only this");
        let messages = build_chat_messages(&request, json!({ "role": "user", "content": "only this" }));
        assert_eq!(vec![json!({ "role": "user", "content": "only this" })], messages);
    }

    /// Non-numeric entries are dropped, not zero-filled (the pre-existing
    /// behaviour on both providers, see [`f32_array`]).
    #[test]
    fn f32_array_converts_in_order_and_drops_non_numbers() {
        let values = vec![json!(1.5), json!(2), json!("nope"), json!(null), json!(-0.25)];
        assert_eq!(vec![1.5f32, 2.0, -0.25], f32_array(&values));
        assert!(f32_array(&[]).is_empty());
    }

    #[test]
    fn url_joins_base_and_path_without_a_separator_assumption() {
        // The base URL carries any trailing `/v1` and the path its own leading
        // slash, so joining is plain concatenation — the shape all three
        // clients already had.
        let config = HttpEndpointConfig::new(
            "https://api.example.com/v1",
            Duration::from_secs(1),
            Duration::from_secs(1),
        );
        let transport = LlmTransport::new(config).expect("client builds");
        assert_eq!(
            "https://api.example.com/v1/chat/completions",
            transport.url("/chat/completions")
        );
        assert_eq!("https://api.example.com/v1", transport.base_url());
    }
}
