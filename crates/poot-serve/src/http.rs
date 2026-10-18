//! The HTTP layer: admission (rate limit, request queue, origin, framing), request read/parse under size caps and
//! a whole-request deadline, routing, and response writing.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpStream};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::admin::{LoraAdminPolicy, handle_lora_admin, respond_lora_admin_denial};
use crate::api::{ChatReq, CompletionReq, classify_handler_error};
use crate::engine_plan::{EngineHealth, EngineState};
use crate::handlers::{
    handle_chat, handle_completion, handle_embeddings, handle_rerank, stream_chat,
    stream_completion,
};
use crate::lifecycle::deadline_after;
use crate::metrics::Metrics;
use crate::types::Served;

/// Per-key request-rate limiter (card 058). Fixed 60-second window per key: the first request starts the
/// window, later ones increment until `rpm` is reached, then the window rejects until it rolls over.
/// Model endpoints (`/v1/completions`, `/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`) are limited;
/// `/health` and `/metrics` are not. `rpm = 0` (`--rate-limit-rpm` unset or 0, the default) disables it.
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
    /// A limiter of `rpm` requests per minute per client; `0` disables it.
    pub(crate) fn new(rpm: usize) -> Self {
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
        // A panic elsewhere must not turn every later request into a panic: the map holds only counters.
        let mut windows = self.windows.lock().unwrap_or_else(|e| e.into_inner());
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

/// RAII decrement for the live-connection counter. Uses `Drop` so the count still falls if `handle`
/// panics; otherwise the cap would eventually wedge shut.
pub(crate) struct ConnGuard(pub(crate) Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The bound on model requests in the server at once, running or waiting for the engine. A request past
/// the bound is refused on the spot, before its body is read, so neither the queue nor the memory behind
/// it grows with the number of clients.
pub(crate) struct RequestQueue {
    cap: NonZeroUsize,
    admitted: AtomicUsize,
}

/// One admitted request's place in the [`RequestQueue`], released when it drops.
pub(crate) struct QueueSlot<'a>(&'a RequestQueue);

impl RequestQueue {
    pub(crate) fn new(cap: NonZeroUsize) -> Self {
        Self {
            cap,
            admitted: AtomicUsize::new(0),
        }
    }

    /// A slot for one more request, or `None` when the queue is full.
    pub(crate) fn admit(&self) -> Option<QueueSlot<'_>> {
        self.admitted
            .try_update(Ordering::SeqCst, Ordering::SeqCst, |admitted| {
                (admitted < self.cap.get()).then_some(admitted + 1)
            })
            .ok()
            .map(|_| QueueSlot(self))
    }
}

impl Drop for QueueSlot<'_> {
    fn drop(&mut self) {
        self.0.admitted.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Cap on the cumulative bytes of the request line and all headers, so a newline-free header cannot grow
/// a `String` without limit. 64 KiB is far above a legitimate header block.
pub(crate) const MAX_HEADER_BYTES: usize = 64 * 1024;

/// The largest request body each route takes, in bytes, checked against `Content-Length` before the body
/// is read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BodyCaps {
    /// Every route but a chat that may carry images.
    pub(crate) text: usize,
    /// A chat on a server that takes images, whose messages may hold base64 images.
    pub(crate) images: usize,
}

impl BodyCaps {
    /// The cap of `route` on a server that takes images, or not.
    pub(crate) fn of(self, route: ModelRoute, takes_images: bool) -> usize {
        match route {
            ModelRoute::Chat if takes_images => self.images,
            _ => self.text,
        }
    }
}

/// The `Host` values a request may carry, from the address the listener bound: a loopback listener is
/// reached as `localhost` or its loopback address, a listener on one address as that address, and a
/// wildcard listener has no one name, so only the port is checked. A page on another origin that is
/// rebound to a loopback address still names its own host, and is refused here.
pub(crate) struct HostPolicy {
    /// The names a request may use, lowercase; `None` accepts any name.
    names: Option<Vec<String>>,
    port: u16,
}

impl HostPolicy {
    pub(crate) fn for_listener(bound: SocketAddr) -> Self {
        let ip = bound.ip();
        let names = if ip.is_loopback() {
            Some(
                ["localhost", "127.0.0.1", "[::1]"]
                    .map(String::from)
                    .to_vec(),
            )
        } else if ip.is_unspecified() {
            None
        } else {
            Some(vec![match ip {
                IpAddr::V4(ip) => ip.to_string(),
                IpAddr::V6(ip) => format!("[{ip}]"),
            }])
        };
        Self {
            names,
            port: bound.port(),
        }
    }

    /// `Ok` when `host` (the header's value) names this listener. A port, when given, must be the bound one.
    fn check(&self, host: Option<&str>) -> Result<(), Refusal> {
        let host = host.ok_or(Refusal::MissingHost)?;
        let (name, port) = match host.strip_prefix('[') {
            Some(rest) => {
                let (inner, after) = rest.split_once(']').ok_or(Refusal::ForeignHost)?;
                (format!("[{inner}]"), after.strip_prefix(':'))
            }
            None => match host.rsplit_once(':') {
                Some((name, port)) => (name.to_string(), Some(port)),
                None => (host.to_string(), None),
            },
        };
        if let Some(port) = port
            && port.parse::<u16>().ok() != Some(self.port)
        {
            return Err(Refusal::ForeignHost);
        }
        match &self.names {
            Some(names) if !names.contains(&name.to_ascii_lowercase()) => Err(Refusal::ForeignHost),
            _ => Ok(()),
        }
    }
}

/// Everything `handle` bounds a request by, built once from the configuration and the bound address.
pub(crate) struct Limits {
    pub(crate) limiter: RateLimiter,
    pub(crate) queue: RequestQueue,
    /// The admitted requests that are still reading their body: a smaller bound than `queue`, so
    /// clients that send slowly cannot hold every slot while no work reaches the engine.
    pub(crate) receiving: RequestQueue,
    pub(crate) bodies: BodyCaps,
    pub(crate) hosts: HostPolicy,
    /// Time from accept in which the request head and body must have arrived.
    pub(crate) request_timeout: Duration,
}

/// Why a request is refused before any handler runs, each with its own status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Refusal {
    #[error("request headers exceed the maximum allowed size")]
    HeadersTooLarge,
    #[error("Content-Length must be a whole number of bytes")]
    InvalidContentLength,
    #[error("Content-Length appears more than once")]
    DuplicateContentLength,
    #[error("the {0} header appears more than once")]
    DuplicateHeader(&'static str),
    #[error("Transfer-Encoding is not supported: send the body with a Content-Length")]
    TransferEncoding,
    #[error("a POST needs a Content-Length")]
    LengthRequired,
    #[error("the request body ended before its Content-Length")]
    TruncatedBody,
    #[error("the Content-Type must be application/json")]
    NotJson,
    #[error("a POST needs a Host header")]
    MissingHost,
    #[error("the Host header does not name this server")]
    ForeignHost,
    #[error("request body exceeds the maximum allowed size")]
    BodyTooLarge,
    #[error("the request body is not valid UTF-8")]
    BodyNotUtf8,
    #[error("the request did not arrive within the request timeout")]
    Timeout,
}

impl Refusal {
    fn status(self) -> u16 {
        match self {
            Self::HeadersTooLarge => 431,
            Self::InvalidContentLength
            | Self::DuplicateContentLength
            | Self::DuplicateHeader(_)
            | Self::TruncatedBody
            | Self::MissingHost
            | Self::BodyNotUtf8 => 400,
            Self::TransferEncoding | Self::LengthRequired => 411,
            Self::NotJson => 415,
            Self::ForeignHost => 403,
            Self::BodyTooLarge => 413,
            Self::Timeout => 408,
        }
    }
}

/// Why reading a request stopped: a refusal the client earned, or the socket failing.
#[derive(Debug, thiserror::Error)]
pub(crate) enum ReadFailure {
    #[error(transparent)]
    Refused(#[from] Refusal),
    #[error("read the request: {0}")]
    Io(#[from] std::io::Error),
}

/// A socket reader that stops at a deadline, however the client paces its bytes: each read waits at most
/// what is left of the time since accept, so a client sending a byte at a time cannot hold the connection
/// past it.
struct DeadlineStream {
    stream: TcpStream,
    deadline: Instant,
}

impl Read for DeadlineStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return Err(std::io::ErrorKind::TimedOut.into());
        }
        self.stream.set_read_timeout(Some(left))?;
        self.stream.read(buf)
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

/// Authorization fields are counted while the head is read. On a duplicate both values are discarded, so
/// neither wins by ordering and duplicated secrets do not stay in the parsed request.
pub(crate) enum AuthorizationHeader {
    Missing,
    Single(String),
    Duplicate,
}

/// The bounded request line and headers. The body stays unread until the head is classified and, for
/// a protected admin target, authorized.
pub(crate) struct ParsedRequestHead {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) authorization: AuthorizationHeader,
    /// The one `Content-Length`, when the head has one.
    pub(crate) content_length: Option<usize>,
    pub(crate) content_type: Option<String>,
    pub(crate) host: Option<String>,
}

/// The value of a `Content-Length` header: ASCII digits only, so a sign, a fraction or a list is not a
/// length that reads as another.
fn parse_content_length(value: &str) -> Result<usize, Refusal> {
    let value = value.trim();
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(Refusal::InvalidContentLength);
    }
    value.parse().map_err(|_| Refusal::InvalidContentLength)
}

/// Keep the single value of a header that must not repeat.
fn single_header(
    slot: &mut Option<String>,
    name: &'static str,
    value: &str,
) -> Result<(), Refusal> {
    if slot.replace(value.trim().to_string()).is_some() {
        return Err(Refusal::DuplicateHeader(name));
    }
    Ok(())
}

/// Read only the bounded request line and headers. Must not apply the body cap or read any body byte:
/// protected targets authorize right after it returns. Framing the server cannot honor (a malformed or
/// repeated `Content-Length`, any `Transfer-Encoding`) is refused here, since a body read on a guess at
/// where it ends is how requests get smuggled.
pub(crate) fn read_request_head_from<R: BufRead>(
    reader: &mut R,
) -> Result<ParsedRequestHead, ReadFailure> {
    let mut header_budget = MAX_HEADER_BYTES;

    // Request line: bounded like the headers.
    let mut request_line = String::new();
    if !read_capped_line(reader, &mut request_line, &mut header_budget)? {
        return Err(Refusal::HeadersTooLarge.into());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    // Headers: read until the blank line under the shared budget.
    let mut content_length = None;
    let mut content_type = None;
    let mut host = None;
    let mut authorization = AuthorizationHeader::Missing;
    loop {
        let mut header = String::new();
        if !read_capped_line(reader, &mut header, &mut header_budget)? {
            return Err(Refusal::HeadersTooLarge.into());
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
            let length = parse_content_length(value)?;
            if content_length.replace(length).is_some() {
                return Err(Refusal::DuplicateContentLength.into());
            }
        } else if name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(Refusal::TransferEncoding.into());
        } else if name.eq_ignore_ascii_case("content-type") {
            single_header(&mut content_type, "Content-Type", value)?;
        } else if name.eq_ignore_ascii_case("host") {
            single_header(&mut host, "Host", value)?;
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
    Ok(ParsedRequestHead {
        method,
        path,
        authorization,
        content_length,
        content_type,
        host,
    })
}

/// Read a body of `content_length` bytes, after checking the declaration against `cap`: a declared body
/// over the cap is refused with no byte read. The buffer grows with the bytes that arrive, not with the
/// declaration, so a client that declares a large body and sends none costs nothing.
pub(crate) fn read_request_body_from<R: BufRead>(
    reader: &mut R,
    content_length: usize,
    cap: usize,
) -> Result<String, ReadFailure> {
    if content_length > cap {
        return Err(Refusal::BodyTooLarge.into());
    }
    let mut body = Vec::new();
    reader
        .by_ref()
        .take(content_length as u64)
        .read_to_end(&mut body)?;
    if body.len() < content_length {
        return Err(Refusal::TruncatedBody.into());
    }
    String::from_utf8(body).map_err(|_| Refusal::BodyNotUtf8.into())
}

/// A POST must declare a JSON body: a browser form cannot send `application/json` cross-origin without a
/// preflight the server never answers, so requiring it keeps another origin's page out. Parameters such
/// as `charset` are allowed.
fn require_json(content_type: Option<&str>) -> Result<(), Refusal> {
    let media = content_type
        .and_then(|value| value.split(';').next())
        .map(str::trim);
    match media {
        Some(media) if media.eq_ignore_ascii_case("application/json") => Ok(()),
        _ => Err(Refusal::NotJson),
    }
}

/// The routes that run the model: each is a message on the engine's one queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelRoute {
    Completions,
    Chat,
    Embeddings,
    Rerank,
}

impl ModelRoute {
    pub(crate) fn of(target: &str) -> Option<Self> {
        match request_target_path(target) {
            "/v1/completions" => Some(Self::Completions),
            "/v1/chat/completions" => Some(Self::Chat),
            "/v1/embeddings" => Some(Self::Embeddings),
            "/v1/rerank" => Some(Self::Rerank),
            _ => None,
        }
    }
}

pub(crate) fn request_target_path(target: &str) -> &str {
    target.split_once('?').map_or(target, |(path, _)| path)
}

pub(crate) fn is_lora_admin_target(target: &str) -> bool {
    let path = request_target_path(target);
    path == "/v1/lora_adapters" || path.starts_with("/v1/lora_adapters/")
}

/// The `/health` answer for the engine's state: 200 `serving` with the backend, rows and positions of the
/// engine plan once it is `Serving`, else 503 with the state's name. Reads the state the plan publishes;
/// it never answers from a constant.
fn health_response(health: &EngineHealth) -> (u16, String) {
    let engine_loaded = health.engine_loaded();
    match health.state() {
        EngineState::Serving {
            backend,
            rows,
            max_seq,
        } => (
            200,
            serde_json::json!({
                "status": "serving",
                "backend": backend,
                "rows": rows,
                "max_seq": max_seq,
                "engine_loaded": engine_loaded,
            })
            .to_string(),
        ),
        EngineState::Starting => (
            503,
            serde_json::json!({ "status": "starting", "engine_loaded": engine_loaded }).to_string(),
        ),
        EngineState::Draining => (
            503,
            serde_json::json!({ "status": "draining", "engine_loaded": engine_loaded }).to_string(),
        ),
        EngineState::Dead(death) => (
            503,
            serde_json::json!({
                "status": "dead",
                "cause": death.kind(),
                "reason": death.reason(),
                "engine_loaded": engine_loaded,
            })
            .to_string(),
        ),
    }
}

/// Configure the accepted socket before a handler owns it. Callers must close the socket if this fails,
/// or the handler would keep unbounded writes.
pub(crate) fn configure_response_writes(
    stream: &TcpStream,
    timeout: Duration,
) -> std::io::Result<()> {
    stream.set_write_timeout(Some(timeout))
}

/// Write all of `bytes` within the socket's write timeout counted over the whole write. The timeout
/// itself bounds one `write` call, so a reader that drains a few bytes just inside each window would
/// hold the connection for as long as the response is long.
pub(crate) fn write_all_within(stream: &mut TcpStream, bytes: &[u8]) -> std::io::Result<()> {
    let Some(window) = stream.write_timeout()? else {
        return stream.write_all(bytes);
    };
    let deadline = Instant::now().checked_add(window);
    let mut rest = bytes;
    let written = loop {
        if rest.is_empty() {
            break Ok(());
        }
        let left = deadline.map_or(window, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        });
        if left.is_zero() {
            break Err(std::io::ErrorKind::TimedOut.into());
        }
        if let Err(error) = stream.set_write_timeout(Some(left)) {
            break Err(error);
        }
        match stream.write(rest) {
            Ok(0) => break Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => rest = &rest[n..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                break Err(std::io::ErrorKind::TimedOut.into());
            }
            Err(error) => break Err(error),
        }
    };
    stream.set_write_timeout(Some(window))?;
    written
}

/// Answer a request that failed to read: a refusal by its status, a read that ran out of time as 408,
/// and any other socket failure as the error.
fn refuse(stream: &mut TcpStream, failure: ReadFailure) -> Result<()> {
    let refusal = match failure {
        ReadFailure::Refused(refusal) => refusal,
        ReadFailure::Io(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
            ) =>
        {
            Refusal::Timeout
        }
        ReadFailure::Io(error) => return Err(error.into()),
    };
    respond_refusal(stream, refusal)
}

fn respond_refusal(stream: &mut TcpStream, refusal: Refusal) -> Result<()> {
    respond_error(stream, refusal.status(), refusal, "invalid_request_error")
}

/// Unread request bytes a closing connection takes in, so the kernel does not answer them with a reset
/// that destroys the response still in flight to the client.
const LINGER_BYTES: u64 = 256 * 1024;

/// Close a connection whose request body was not read in full: end the response, then take in what the
/// client is still sending, up to [`LINGER_BYTES`] and the request deadline. A refusal answers before the
/// body arrives; closing a socket with unread input resets the connection, and a reset can discard a
/// response the client has not read yet. Half-closing first sends the response and the end of it, and the
/// client's own close ends the wait.
fn close_with_body_unread(stream: &TcpStream, reader: &mut impl Read) {
    let _ = stream.shutdown(Shutdown::Write);
    let _ = std::io::copy(&mut reader.take(LINGER_BYTES), &mut std::io::sink());
}

/// Read one HTTP request, route it, and write the response. Runs on a per-connection thread; every route
/// that runs the model sends the engine thread a message on the one queue in `served`. The request head
/// and body must arrive within `limits.request_timeout` of `accepted`.
pub(crate) fn handle(
    mut stream: TcpStream,
    accepted: Instant,
    served: &Served,
    health: &EngineHealth,
    metrics: &Metrics,
    limits: &Limits,
    lora_admin_policy: &LoraAdminPolicy,
) -> Result<()> {
    let mut reader = BufReader::new(DeadlineStream {
        stream: stream.try_clone()?,
        deadline: deadline_after(accepted, limits.request_timeout),
    });
    // Set once the request's body (if it declared one) has been read in full.
    let mut body_unread = true;
    let answered = serve(
        &mut stream,
        &mut reader,
        &mut body_unread,
        served,
        health,
        metrics,
        limits,
        lora_admin_policy,
    );
    if body_unread {
        close_with_body_unread(&stream, &mut reader);
    }
    answered
}

fn serve(
    stream: &mut TcpStream,
    reader: &mut BufReader<DeadlineStream>,
    body_unread: &mut bool,
    served: &Served,
    health: &EngineHealth,
    metrics: &Metrics,
    limits: &Limits,
    lora_admin_policy: &LoraAdminPolicy,
) -> Result<()> {
    let head = match read_request_head_from(reader) {
        Ok(head) => head,
        Err(failure) => return refuse(stream, failure),
    };
    if is_lora_admin_target(&head.path) {
        return serve_lora_admin(
            stream,
            reader,
            body_unread,
            &head,
            served,
            limits,
            lora_admin_policy,
        );
    }
    match head.method.as_str() {
        "GET" => {
            *body_unread = head.content_length.is_some_and(|length| length > 0);
            serve_get(stream, &head.path, health, metrics)
        }
        "POST" => serve_model(stream, reader, body_unread, head, served, metrics, limits),
        _ => respond_error(stream, 405, "method not allowed", "invalid_request_error"),
    }
}

/// A LoRA administration request: authorized from the head alone, then checked for the origin it came
/// from, and only then is its body read.
fn serve_lora_admin<R: BufRead>(
    stream: &mut TcpStream,
    reader: &mut R,
    body_unread: &mut bool,
    head: &ParsedRequestHead,
    served: &Served,
    limits: &Limits,
    policy: &LoraAdminPolicy,
) -> Result<()> {
    if let Err(denial) = policy.authorize(&head.authorization) {
        return respond_lora_admin_denial(stream, denial);
    }
    let body = match read_admin_body(reader, head, limits) {
        Ok(body) => body,
        Err(failure) => return refuse(stream, failure),
    };
    *body_unread = false;
    handle_lora_admin(stream, served, &head.method, &head.path, &body)
}

/// The origin checks of an administration request, and the JSON body of its POST.
fn read_admin_body<R: BufRead>(
    reader: &mut R,
    head: &ParsedRequestHead,
    limits: &Limits,
) -> Result<String, ReadFailure> {
    limits.hosts.check(head.host.as_deref())?;
    if head.method != "POST" {
        return Ok(String::new());
    }
    require_json(head.content_type.as_deref())?;
    let length = head.content_length.ok_or(Refusal::LengthRequired)?;
    read_request_body_from(reader, length, limits.bodies.text)
}

/// The read-only routes. A path none of them takes is a method the server does not answer there.
fn serve_get(
    stream: &mut TcpStream,
    path: &str,
    health: &EngineHealth,
    metrics: &Metrics,
) -> Result<()> {
    if path.starts_with("/health") {
        let (status, body) = health_response(health);
        return respond(stream, status, &body);
    }
    // OpenAI /v1/models: list and retrieve (`GET /v1/models/{id}`). The path may carry a query string.
    let route = request_target_path(path);
    if route == "/v1/models" {
        let body = serde_json::json!({ "object": "list", "data": [model_object(&metrics.model)] })
            .to_string();
        return respond(stream, 200, &body);
    }
    if let Some(id) = route.strip_prefix("/v1/models/") {
        // Retrieve: poot serves one model, else 404.
        if id == metrics.model {
            return respond(stream, 200, &model_object(&metrics.model).to_string());
        }
        return respond_error(
            stream,
            404,
            format_args!("model '{id}' not found"),
            "invalid_request_error",
        );
    }
    // Prometheus text format (checked before the generic /metrics JSON route).
    if path.starts_with("/metrics/prometheus") {
        return respond_typed(
            stream,
            200,
            "text/plain; version=0.0.4",
            &metrics.prometheus(),
        );
    }
    if path.starts_with("/metrics") {
        return respond(stream, 200, &metrics.json());
    }
    respond_error(stream, 405, "method not allowed", "invalid_request_error")
}

/// A POST to a model route. Everything that can refuse it from its head does so before the body is read:
/// the origin, the declared type and length, the client's rate and the queue. Only an admitted request's
/// body is read, and only up to its route's cap.
fn serve_model<R: BufRead>(
    stream: &mut TcpStream,
    reader: &mut R,
    body_unread: &mut bool,
    head: ParsedRequestHead,
    served: &Served,
    metrics: &Metrics,
    limits: &Limits,
) -> Result<()> {
    let Some(route) = ModelRoute::of(&head.path) else {
        return respond_error(stream, 404, "unknown path", "invalid_request_error");
    };
    let declared = limits
        .hosts
        .check(head.host.as_deref())
        .and_then(|()| require_json(head.content_type.as_deref()))
        .and_then(|()| head.content_length.ok_or(Refusal::LengthRequired));
    let content_length = match declared {
        Ok(length) => length,
        Err(refusal) => return respond_refusal(stream, refusal),
    };

    // Rate limit on the model routes: 429 + Retry-After over budget. Keyed on peer IP (see RateLimiter).
    let rate_key = stream
        .peer_addr()
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    if let Err(retry_after) = limits.limiter.check(&rate_key) {
        return respond_busy(stream, "rate limit exceeded", retry_after);
    }
    let Some(_slot) = limits.queue.admit() else {
        return respond_busy(
            stream,
            "the server is at its request limit: retry shortly",
            1,
        );
    };

    // Only so many admitted requests may be reading their body at once, so slow senders cannot hold every
    // slot of the queue while nothing reaches the engine.
    let Some(receiving) = limits.receiving.admit() else {
        return respond_busy(
            stream,
            "the server is receiving as many request bodies as it allows: retry shortly",
            1,
        );
    };
    let body = read_request_body_from(
        reader,
        content_length,
        limits.bodies.of(route, served.images.is_some()),
    );
    drop(receiving);
    let body = match body {
        Ok(body) => body,
        Err(failure) => return refuse(stream, failure),
    };
    *body_unread = false;

    // A second handle on the socket for the hang-up probe: a non-streaming response writes nothing until
    // the request ends, so only a read tells the engine the client left.
    let probe = stream.try_clone().ok();
    let client_gone = || probe.as_ref().is_some_and(crate::lifecycle::peer_closed);
    let result = match route {
        ModelRoute::Embeddings => handle_embeddings(served, metrics, &body, &client_gone),
        ModelRoute::Rerank => handle_rerank(served, metrics, &body, &client_gone),
        ModelRoute::Completions => {
            let req: CompletionReq = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => return respond_error(stream, 400, e, "invalid_request_error"),
            };
            if req.stream {
                // SSE: one chunk per token, then a throughput receipt and [DONE].
                return stream_completion(stream, served, metrics, &req);
            }
            handle_completion(served, metrics, &req, &client_gone)
        }
        ModelRoute::Chat => {
            let req: ChatReq = match serde_json::from_str(&body) {
                Ok(r) => r,
                Err(e) => return respond_error(stream, 400, e, "invalid_request_error"),
            };
            if req.stream {
                // SSE chat: a chat.completion.chunk delta per token, then finish, receipt and [DONE].
                return stream_chat(stream, served, metrics, &req);
            }
            handle_chat(served, metrics, &req, &client_gone)
        }
    };

    match result {
        Ok(json) => respond(stream, 200, &json),
        Err(e) => {
            // A malformed guided_regex/guided_json/guided_choice/response_format is a 400, not a 500
            // (see classify_handler_error).
            let (status, type_) = classify_handler_error(&e);
            respond_error(stream, status, format_args!("{e:#}"), type_)
        }
    }
}

/// A 429 with `Retry-After`: the client's rate or the server's request limit.
fn respond_busy(stream: &mut TcpStream, message: &str, retry_after: u64) -> Result<()> {
    respond_typed_with_header(
        stream,
        429,
        "application/json",
        &format!("Retry-After: {retry_after}"),
        &error_body(429, message, "rate_limit_exceeded"),
    )
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
        409 => "Conflict",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        415 => "Unsupported Media Type",
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
    write_all_within(stream, resp.as_bytes())?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    /// A request the way the socket-independent parser returns it: the head's `Authorization` kept beside
    /// the method, path and body. Production authorizes on the head before the body is read, so no
    /// production type carries both.
    pub(crate) struct ParsedRequestWithHead {
        pub(crate) method: String,
        pub(crate) path: String,
        pub(crate) authorization: AuthorizationHeader,
        pub(crate) body: String,
    }

    impl AuthorizationHeader {
        pub(crate) fn as_deref(&self) -> Option<&str> {
            match self {
                Self::Single(value) => Some(value),
                Self::Missing | Self::Duplicate => None,
            }
        }
    }

    /// Read the request line, headers and the body of at most `body_cap` bytes from a socket.
    pub(crate) fn read_request(
        stream: &mut TcpStream,
        body_cap: usize,
    ) -> Result<ParsedRequestWithHead, ReadFailure> {
        let mut reader = BufReader::new(stream.try_clone()?);
        read_request_from(&mut reader, body_cap)
    }

    /// Socket-independent request parser. Reads the request line and headers under a shared
    /// [`MAX_HEADER_BYTES`] budget, then the body under `body_cap`.
    pub(crate) fn read_request_from<R: BufRead>(
        reader: &mut R,
        body_cap: usize,
    ) -> Result<ParsedRequestWithHead, ReadFailure> {
        let head = read_request_head_from(reader)?;
        let body = read_request_body_from(reader, head.content_length.unwrap_or(0), body_cap)?;
        Ok(ParsedRequestWithHead {
            method: head.method,
            path: head.path,
            authorization: head.authorization,
            body,
        })
    }
}
