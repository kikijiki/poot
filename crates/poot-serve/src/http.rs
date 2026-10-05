//! The HTTP layer: rate limiting, request read/parse with size and slowloris budgets, routing, and response writing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::admin::{
    LoraAdminBackendResolver, LoraAdminPolicy, handle_lora_admin_with_backend,
    respond_lora_admin_denial,
};
use crate::api::{ChatReq, CompletionReq, classify_handler_error};
use crate::handlers::{
    handle_chat, handle_completion, handle_embeddings_encoder, handle_rerank_cross,
    handle_rerank_encoder, stream_chat, stream_completion,
};
use crate::metrics::Metrics;
use crate::types::{Backend, Job};

/// Per-key request-rate limiter (card 058). Fixed 60-second window per key: the first request starts the
/// window, later ones increment until `rpm` is reached, then the window rejects until it rolls over.
/// Generation endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`) are limited;
/// `/health` and `/metrics` are not. `rpm = 0` (`POOT_RATE_LIMIT_RPM` unset or 0, the default) disables it.
///
/// The key is the client's peer IP, not the `Authorization` value. Inference has no auth layer (only LoRA
/// administration reads that header), so keying on it would let a client reset its budget by varying the
/// header and grow `windows` without bound. `check_at` also evicts long-stale entries to bound the map.
pub(crate) struct RateLimiter {
    pub(crate) rpm: usize,
    pub(crate) windows: Mutex<HashMap<String, (Instant, usize)>>,
}

/// A window untouched for this long is safe to evict. Well above the 60s window so a client mid-window
/// is never evicted.
const STALE_WINDOW_AGE: Duration = Duration::from_secs(300);

impl RateLimiter {
    pub(crate) fn from_env() -> Self {
        let rpm = std::env::var("POOT_RATE_LIMIT_RPM")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        Self {
            rpm,
            windows: Mutex::new(HashMap::new()),
        }
    }

    /// `Ok(())` if the request is allowed; `Err(retry_after_secs)` if `key` has hit its per-minute limit.
    /// `now` is injected so the window logic is unit-testable without sleeping.
    pub(crate) fn check_at(&self, key: &str, now: Instant) -> Result<(), u64> {
        if self.rpm == 0 {
            return Ok(());
        }
        let mut windows = self.windows.lock().unwrap();
        // Evict on every call (no background thread) so `windows` holds roughly the keys seen in the
        // last STALE_WINDOW_AGE.
        windows.retain(|_, (started, _)| now.duration_since(*started) < STALE_WINDOW_AGE);
        let entry = windows.entry(key.to_string()).or_insert((now, 0));
        if now.duration_since(entry.0) >= Duration::from_secs(60) {
            *entry = (now, 0); // window rolled over: reset.
        }
        if entry.1 >= self.rpm {
            let elapsed = now.duration_since(entry.0).as_secs();
            return Err(60u64.saturating_sub(elapsed).max(1));
        }
        entry.1 += 1;
        Ok(())
    }

    pub(crate) fn check(&self, key: &str) -> Result<(), u64> {
        self.check_at(key, Instant::now())
    }
}

/// Max simultaneous connection-handler threads. The accept loop spawns one OS thread per connection, so
/// many idle connections could exhaust thread/memory limits before the rate limiter or size caps run.
/// `POOT_MAX_CONNECTIONS=0` means unlimited; unset defaults to 4096.
pub(crate) fn max_connections() -> usize {
    max_connections_from(std::env::var("POOT_MAX_CONNECTIONS").ok().as_deref())
}

/// Pure core of [`max_connections`]. An unparseable value falls back to 4096; `0` disables the cap.
pub(crate) fn max_connections_from(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse().ok()).unwrap_or(4096)
}

/// RAII decrement for the live-connection counter. Uses `Drop` so the count still falls if `handle`
/// panics; otherwise the cap would eventually wedge shut.
pub(crate) struct ConnGuard(pub(crate) Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Upper bound on a request body: room for a base64-encoded VLM image, checked before the body buffer
/// is allocated so `Content-Length` cannot size an allocation.
pub(crate) const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Cap on the cumulative bytes of the request line and all headers, so a newline-free header cannot grow
/// a `String` without limit. 64 KiB is far above a legitimate header block.
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;

/// True when the declared `Content-Length` fits within [`MAX_BODY_BYTES`].
pub(crate) fn body_len_ok(content_length: usize) -> bool {
    content_length <= MAX_BODY_BYTES
}

/// Idle read timeout for the request head and body (slowloris bound); a stalled client would otherwise
/// pin a `poot-conn` thread forever. It never cuts an active generation, since streaming responses are
/// writes. `POOT_READ_TIMEOUT_SECS=0` disables it; default 30s.
pub(crate) fn request_read_timeout() -> Option<Duration> {
    request_read_timeout_from(std::env::var("POOT_READ_TIMEOUT_SECS").ok().as_deref())
}

/// Pure core of [`request_read_timeout`]. An unparseable value falls back to 30s; `0` disables it.
pub(crate) fn request_read_timeout_from(raw: Option<&str>) -> Option<Duration> {
    let secs = raw.and_then(|s| s.trim().parse::<u64>().ok()).unwrap_or(30);
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// A rejected explicit `POOT_WRITE_TIMEOUT_SECS` value. Variants are distinct so malformed
/// configuration is never silently replaced by the default.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ResponseWriteTimeoutError {
    #[error("POOT_WRITE_TIMEOUT_SECS must be valid Unicode, got {value:?}")]
    NotUnicode { value: std::ffi::OsString },
    #[error("POOT_WRITE_TIMEOUT_SECS must be greater than zero, got 0")]
    Zero,
    #[error("POOT_WRITE_TIMEOUT_SECS must be a positive whole number of seconds, got {value:?}")]
    Negative { value: String },
    #[error("POOT_WRITE_TIMEOUT_SECS must be a positive whole number of seconds, got {value:?}")]
    Malformed { value: String },
    #[error("POOT_WRITE_TIMEOUT_SECS is too large to represent as seconds, got {value:?}")]
    Overflow { value: String },
}

/// Idle write timeout for every response write, so one blocked socket write cannot hold a handler
/// indefinitely. Missing selects 30 seconds; an explicit value must be a positive `u64`.
pub(crate) fn response_write_timeout() -> std::result::Result<Duration, ResponseWriteTimeoutError> {
    match std::env::var("POOT_WRITE_TIMEOUT_SECS") {
        Ok(raw) => response_write_timeout_from(Some(&raw)),
        Err(std::env::VarError::NotPresent) => response_write_timeout_from(None),
        Err(std::env::VarError::NotUnicode(value)) => {
            Err(ResponseWriteTimeoutError::NotUnicode { value })
        }
    }
}

/// Pure core of [`response_write_timeout`].
pub(crate) fn response_write_timeout_from(
    raw: Option<&str>,
) -> std::result::Result<Duration, ResponseWriteTimeoutError> {
    let Some(raw) = raw else {
        return Ok(Duration::from_secs(30));
    };
    let value = raw.trim();
    if value.starts_with('-') {
        return Err(ResponseWriteTimeoutError::Negative {
            value: raw.to_string(),
        });
    }
    let secs = value.parse::<u64>().map_err(|error| match error.kind() {
        std::num::IntErrorKind::PosOverflow => ResponseWriteTimeoutError::Overflow {
            value: raw.to_string(),
        },
        _ => ResponseWriteTimeoutError::Malformed {
            value: raw.to_string(),
        },
    })?;
    if secs == 0 {
        return Err(ResponseWriteTimeoutError::Zero);
    }
    Ok(Duration::from_secs(secs))
}

/// Configure the accepted socket before a handler owns it. Callers must close the socket if this fails,
/// or the handler would keep unbounded writes.
pub(crate) fn configure_response_writes(
    stream: &TcpStream,
    timeout: Duration,
) -> std::io::Result<()> {
    stream.set_write_timeout(Some(timeout))
}

/// The outcome of reading a request: a parsed request, or a "too large" rejection (413 body, 431 headers).
#[cfg(test)]
pub(crate) enum RequestRead {
    Parsed(ParsedRequest),
    BodyTooLarge,
    HeadersTooLarge,
}

pub(crate) enum RequestHeadRead {
    Parsed(ParsedRequestHead),
    HeadersTooLarge,
}

/// Authorization fields are counted while the head is read. On a duplicate both values are discarded, so
/// neither wins by ordering and duplicated secrets do not stay in the parsed request.
pub(crate) enum AuthorizationHeader {
    Missing,
    Single(String),
    Duplicate,
}

#[cfg(test)]
impl AuthorizationHeader {
    pub(crate) fn as_deref(&self) -> Option<&str> {
        match self {
            Self::Single(value) => Some(value),
            Self::Missing | Self::Duplicate => None,
        }
    }
}

/// `read_line` bounded by a shared header-byte budget: reads at most `*budget` more bytes, decrements it,
/// and returns `false` once the budget is exhausted (the caller then rejects with 431). A clean EOF
/// leaves the budget untouched and returns `true`.
pub(crate) fn read_capped_line<R: BufRead>(
    reader: &mut R,
    buf: &mut String,
    budget: &mut usize,
) -> std::io::Result<bool> {
    let n = reader.by_ref().take(*budget as u64).read_line(buf)?;
    *budget -= n;
    Ok(*budget > 0)
}

/// The bounded request line and headers. The body stays unread until the head is classified and, for
/// a protected admin target, authorized.
pub(crate) struct ParsedRequestHead {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) authorization: AuthorizationHeader,
    pub(crate) content_length: usize,
}

/// The request line + headers + body, parsed off the wire by `read_request`.
pub(crate) struct ParsedRequest {
    pub(crate) method: String,
    pub(crate) path: String,
    #[cfg(test)]
    pub(crate) authorization: AuthorizationHeader,
    pub(crate) body: String,
}

/// Read the request line, headers (capturing `Content-Length` and `Authorization`), and the body.
/// Returns `Ok(None)` when the declared `Content-Length` exceeds [`MAX_BODY_BYTES`] (caller responds
/// 413), before any allocation sized off `content_length`.
#[cfg(test)]
pub(crate) fn read_request(stream: &mut TcpStream) -> Result<RequestRead> {
    let mut reader = BufReader::new(stream.try_clone()?);
    read_request_from(&mut reader)
}

/// Socket-independent request parser. Reads the request line and headers under a shared
/// [`MAX_HEADER_BYTES`] budget, then the body under the [`MAX_BODY_BYTES`] cap.
#[cfg(test)]
pub(crate) fn read_request_from<R: BufRead>(reader: &mut R) -> Result<RequestRead> {
    let head = match read_request_head_from(reader)? {
        RequestHeadRead::Parsed(head) => head,
        RequestHeadRead::HeadersTooLarge => return Ok(RequestRead::HeadersTooLarge),
    };
    let Some(body) = read_request_body_from(reader, head.content_length)? else {
        return Ok(RequestRead::BodyTooLarge);
    };
    Ok(RequestRead::Parsed(ParsedRequest {
        method: head.method,
        path: head.path,
        authorization: head.authorization,
        body,
    }))
}

/// Read only the bounded request line and headers. Must not apply the body cap or read any body byte:
/// protected targets authorize right after it returns.
pub(crate) fn read_request_head_from<R: BufRead>(reader: &mut R) -> Result<RequestHeadRead> {
    let mut header_budget = MAX_HEADER_BYTES;

    // Request line: bounded like the headers.
    let mut request_line = String::new();
    if !read_capped_line(reader, &mut request_line, &mut header_budget)? {
        return Ok(RequestHeadRead::HeadersTooLarge);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // Headers: read until the blank line under the shared budget, capturing Content-Length and the raw Authorization value.
    let mut content_length = 0usize;
    let mut authorization = AuthorizationHeader::Missing;
    loop {
        let mut header = String::new();
        if !read_capped_line(reader, &mut header, &mut header_budget)? {
            return Ok(RequestHeadRead::HeadersTooLarge);
        }
        if header.is_empty() || header == "\r\n" || header == "\n" {
            break; // blank line (end of headers) or clean EOF
        }
        let line = header
            .strip_suffix("\r\n")
            .or_else(|| header.strip_suffix('\n'))
            .unwrap_or(&header);
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.eq_ignore_ascii_case("content-length") {
            content_length = value.trim().parse().unwrap_or(0);
        } else if name.eq_ignore_ascii_case("authorization") {
            // Strip one OWS byte after the colon; keep every other byte so the admin parser can
            // reject stray whitespace and HTAB separators.
            let value = value
                .strip_prefix(' ')
                .or_else(|| value.strip_prefix('\t'))
                .unwrap_or(value);
            authorization = match authorization {
                AuthorizationHeader::Missing => AuthorizationHeader::Single(value.to_string()),
                AuthorizationHeader::Single(_) | AuthorizationHeader::Duplicate => {
                    AuthorizationHeader::Duplicate
                }
            };
        }
    }
    Ok(RequestHeadRead::Parsed(ParsedRequestHead {
        method,
        path,
        authorization,
        content_length,
    }))
}

/// Read and convert a body only after protected-head authorization. `None` is the deferred 413 signal.
pub(crate) fn read_request_body_from<R: BufRead>(
    reader: &mut R,
    content_length: usize,
) -> Result<Option<String>> {
    if !body_len_ok(content_length) {
        return Ok(None);
    }
    // Read incrementally with `take(content_length)` instead of pre-allocating the declared size: a
    // client can declare up to MAX_BODY_BYTES and send nothing, so pre-allocation would let bursts of
    // such connections amplify memory. A short body yields a truncated body that fails JSON parsing with a 400.
    let mut body = Vec::new();
    reader
        .by_ref()
        .take(content_length as u64)
        .read_to_end(&mut body)?;
    Ok(Some(String::from_utf8_lossy(&body).into_owned()))
}

pub(crate) fn request_target_path(target: &str) -> &str {
    target.split_once('?').map_or(target, |(path, _)| path)
}

pub(crate) fn is_lora_admin_target(target: &str) -> bool {
    let path = request_target_path(target);
    path == "/v1/lora_adapters" || path.starts_with("/v1/lora_adapters/")
}

/// Production request staging. A protected target is identified from the URI path and authorized after
/// the bounded head, before any body handling or lazy backend resolution.
pub(crate) fn read_request_for_routing(
    stream: &mut TcpStream,
    backend: &dyn LoraAdminBackendResolver,
    policy: &LoraAdminPolicy,
) -> Result<Option<ParsedRequest>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let head = match read_request_head_from(&mut reader)? {
        RequestHeadRead::Parsed(head) => head,
        RequestHeadRead::HeadersTooLarge => {
            respond_error(
                stream,
                431,
                "request headers exceed the maximum allowed size",
                "invalid_request_error",
            )?;
            return Ok(None);
        }
    };

    if is_lora_admin_target(&head.path) {
        if let Err(denial) = policy.authorize(&head.authorization) {
            respond_lora_admin_denial(stream, denial)?;
            return Ok(None);
        }
        let Some(body) = read_request_body_from(&mut reader, head.content_length)? else {
            respond_error(
                stream,
                413,
                "request body exceeds the maximum allowed size",
                "invalid_request_error",
            )?;
            return Ok(None);
        };
        let admin = backend.resolve_lora_admin_backend();
        handle_lora_admin_with_backend(stream, admin, &head.method, &head.path, &body)?;
        return Ok(None);
    }

    let Some(body) = read_request_body_from(&mut reader, head.content_length)? else {
        respond_error(
            stream,
            413,
            "request body exceeds the maximum allowed size",
            "invalid_request_error",
        )?;
        return Ok(None);
    };
    Ok(Some(ParsedRequest {
        method: head.method,
        path: head.path,
        #[cfg(test)]
        authorization: head.authorization,
        body,
    }))
}

/// Read one HTTP request, route it, and write the response. Runs on a per-connection thread; generation
/// is submitted to the engine's job queue when there is one. `engine` is `None` while no serving loop
/// exists: a valid generation request is then refused with a typed 503.
pub(crate) fn handle(
    mut stream: TcpStream,
    engine: Option<&Sender<Job>>,
    backend: &Backend,
    metrics: &Metrics,
    limiter: &RateLimiter,
    lora_admin_policy: &LoraAdminPolicy,
) -> Result<()> {
    let Some(ParsedRequest {
        method, path, body, ..
    }) = read_request_for_routing(&mut stream, backend, lora_admin_policy)?
    else {
        return Ok(());
    };

    if method == "GET" && path.starts_with("/health") {
        let body = serde_json::json!({ "status": "ok", "engine_loaded": engine.is_some() });
        return respond(&mut stream, 200, &body.to_string());
    }
    // OpenAI /v1/models: list and retrieve (`GET /v1/models/{id}`). The path may carry a query string.
    if method == "GET" {
        let route = path.split('?').next().unwrap_or(path.as_str());
        if route == "/v1/models" {
            let body =
                serde_json::json!({ "object": "list", "data": [model_object(&metrics.model)] })
                    .to_string();
            return respond(&mut stream, 200, &body);
        }
        if let Some(id) = route.strip_prefix("/v1/models/") {
            // Retrieve: poot serves one model, else 404.
            if id == metrics.model {
                return respond(&mut stream, 200, &model_object(&metrics.model).to_string());
            }
            return respond_error(
                &mut stream,
                404,
                format_args!("model '{id}' not found"),
                "invalid_request_error",
            );
        }
    }
    // Prometheus text format (checked before the generic /metrics JSON route).
    if method == "GET" && path.starts_with("/metrics/prometheus") {
        return respond_typed(
            &mut stream,
            200,
            "text/plain; version=0.0.4",
            &metrics.prometheus(),
        );
    }
    if method == "GET" && path.starts_with("/metrics") {
        let body = metrics.json();
        return respond(&mut stream, 200, &body);
    }
    if method != "POST" {
        return respond_error(
            &mut stream,
            405,
            "method not allowed",
            "invalid_request_error",
        );
    }

    // Rate limit on the generation endpoints: 429 + Retry-After over budget. Keyed on peer IP, not `auth_key` (see RateLimiter).
    let rate_key = stream
        .peer_addr()
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    if let Err(retry_after) = limiter.check(&rate_key) {
        let body = serde_json::json!({
            "error": { "message": "rate limit exceeded", "type": "rate_limit_exceeded", "code": 429 }
        })
        .to_string();
        return respond_typed_with_header(
            &mut stream,
            429,
            "application/json",
            &format!("Retry-After: {retry_after}"),
            &body,
        );
    }

    // /v1/embeddings: a BERT encoder. A cross-encoder cannot produce sentence embeddings (it scores
    // pairs), and a causal decoder serves none.
    if path.starts_with("/v1/embeddings") {
        return match backend {
            Backend::Encoder(enc) => handle_embeddings_encoder(&mut stream, enc, metrics, &body),
            Backend::Decoder(_) => respond_unsupported(
                &mut stream,
                "this model is a causal decoder; embeddings come from an encoder model.",
            ),
            Backend::CrossEncoder(_) => respond_unsupported(
                &mut stream,
                "this model is a cross-encoder reranker; it has no embedding output. Use /v1/rerank.",
            ),
        };
    }
    // /v1/rerank (Cohere/Jina-shaped): a cross-encoder scores pairs jointly; an encoder ranks by
    // bi-encoder cosine; a causal decoder serves none.
    if path.starts_with("/v1/rerank") {
        return match backend {
            Backend::CrossEncoder(ce) => handle_rerank_cross(&mut stream, ce, &body),
            Backend::Encoder(enc) => handle_rerank_encoder(&mut stream, enc, &body),
            Backend::Decoder(_) => respond_unsupported(
                &mut stream,
                "this model is a causal decoder; reranking comes from an encoder or cross-encoder model.",
            ),
        };
    }

    // Generation endpoints below need a causal decoder. An encoder / cross-encoder has no LM head.
    let runner = match backend {
        Backend::Decoder(runner) => runner.as_ref(),
        Backend::Encoder(_) | Backend::CrossEncoder(_) => {
            return respond_unsupported(
                &mut stream,
                "this model is an encoder (no LM head); generation is not supported. \
                 Use /v1/embeddings or /v1/rerank.",
            );
        }
    };

    let result = if path.starts_with("/v1/completions") {
        let req: CompletionReq = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                return respond_error(&mut stream, 400, e, "invalid_request_error");
            }
        };
        if req.stream {
            // SSE: one chunk per token, then a throughput receipt and [DONE].
            return stream_completion(&mut stream, engine, runner, metrics, &req);
        }
        handle_completion(engine, runner, metrics, &req)
    } else if path.starts_with("/v1/chat/completions") {
        let req: ChatReq = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                return respond_error(&mut stream, 400, e, "invalid_request_error");
            }
        };
        if req.stream {
            // SSE chat: a chat.completion.chunk delta per token, then finish, receipt and [DONE].
            return stream_chat(&mut stream, engine, runner, metrics, &req);
        }
        handle_chat(engine, runner, metrics, &req)
    } else {
        return respond_error(&mut stream, 404, "unknown path", "invalid_request_error");
    };

    match result {
        Ok(json) => respond(&mut stream, 200, &json),
        Err(e) => {
            // A malformed guided_regex/guided_json/guided_choice/response_format is a 400, not a 500
            // (see classify_handler_error).
            let (status, type_) = classify_handler_error(&e);
            respond_error(&mut stream, status, format_args!("{e:#}"), type_)
        }
    }
}

pub(crate) fn respond(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    respond_typed(stream, status, "application/json", body)
}

/// Write an OpenAI-shaped error: `{"error": {"message", "type", "code"}}`. Clients read `error.message`;
/// a flat string makes them fall back to a generic message. `type` is "invalid_request_error" for 4xx,
/// "server_error" for 5xx.
pub(crate) fn respond_error(
    stream: &mut TcpStream,
    status: u16,
    message: impl std::fmt::Display,
    type_: &str,
) -> Result<()> {
    respond(stream, status, &error_body(status, message, type_))
}

/// One OpenAI `model` object (`{id, object, created, owned_by}`), shared by the `/v1/models` list and
/// `/v1/models/{id}` retrieve routes.
pub(crate) fn model_object(id: &str) -> serde_json::Value {
    serde_json::json!({ "id": id, "object": "model", "created": 0, "owned_by": "poot" })
}

/// The OpenAI-shaped error JSON body (pure; the unit-test surface for [`respond_error`]).
pub(crate) fn error_body(status: u16, message: impl std::fmt::Display, type_: &str) -> String {
    serde_json::json!({
        "error": { "message": message.to_string(), "type": type_, "code": status }
    })
    .to_string()
}

/// 400 for an endpoint the loaded backend cannot serve (e.g. generation on an encoder). `message` names
/// what to use instead.
pub(crate) fn respond_unsupported(stream: &mut TcpStream, message: &str) -> Result<()> {
    respond_error(stream, 400, message, "invalid_request_error")
}

/// Write an HTTP response with an explicit Content-Type (JSON for the API, `text/plain` for Prometheus).
pub(crate) fn respond_typed(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &str,
) -> Result<()> {
    respond_typed_with_header(stream, status, content_type, "", body)
}

/// Like [`respond_typed`] with one extra header line (e.g. `Retry-After: 30`). `extra` is the full
/// header without its trailing CRLF; `""` for none.
pub(crate) fn respond_typed_with_header(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra: &str,
    body: &str,
) -> Result<()> {
    let reason = match status {
        200 => "OK",
        401 => "Unauthorized",
        403 => "Forbidden",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Internal Server Error",
    };
    let extra = if extra.is_empty() {
        String::new()
    } else {
        format!("{extra}\r\n")
    };
    let resp = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.flush()?;
    Ok(())
}
