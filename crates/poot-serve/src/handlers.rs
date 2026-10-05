//! Endpoint handlers for the Runner-backed decoder/encoder paths: completions, chat, streaming, embeddings, rerank.

use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant};

use anyhow::Result;
use poot_llm::encoder::{CrossEncoderRunner, EncoderPooling, EncoderRunner};
use poot_llm::{GenerationControl, Runner};

use crate::api::{
    ChatChoice, ChatReq, ChatResp, ChatRespMsg, Choice, CompletionReq, CompletionResp,
    GuidedDecodeError, LoraAdapterError, Stop, Usage, apply_guided, apply_stop, clamp_logprobs,
    clamp_n, logprobs_json, parse_tool_calls, resolve_lora_adapter, sampler_of,
    suppress_stream_usage, tool_call_stream_deltas,
};
use crate::http::{error_body, respond, respond_error};
use crate::lifecycle::{require_engine, submit};
use crate::metrics::{InFlight, Metrics};
use crate::types::{GenerationError, Job};
/// Count only response-side transport failures. Explicit `Failed`, `Done`, and receiver cancellation are
/// already counted at their engine terminal boundary; a dead engine/job channel has no producer left.
pub(crate) fn account_generation_error(metrics: &Metrics, error: &GenerationError) {
    if error.needs_response_error_metric() {
        metrics.requests_errored.fetch_add(1, Ordering::Relaxed);
    }
}

/// Answer a valid generation request that no engine can take, before any header is written: the
/// refusal's own status (503) with an OpenAI-shaped `server_error` naming the missing engine. A request
/// that was queued is never answered this way, and no completion is fabricated for one that was not.
pub(crate) fn respond_generation_refusal(
    stream: &mut TcpStream,
    error: &GenerationError,
) -> Result<()> {
    respond_error(stream, error.stream_status(), error, "server_error")
}

/// Emit the terminal response after SSE headers when generation did not reach `Done`: an OpenAI-shaped
/// error object as a data event (the status is already 200). No finish chunk, usage receipt, or `[DONE]`,
/// since those signal success. Cancellation writes nothing; the client transport is already gone.
pub(crate) fn stream_generation_error(
    stream: &mut TcpStream,
    metrics: &Metrics,
    error: &GenerationError,
) -> Result<()> {
    account_generation_error(metrics, error);
    if matches!(
        error,
        GenerationError::ClientCancelled | GenerationError::ServerDrainExpired
    ) {
        tracing::debug!(terminal = error.kind(), "generation stream cancelled");
        return Ok(());
    }

    let status = error.stream_status();
    tracing::error!(
        terminal = error.kind(),
        status,
        error = %error,
        "generation stream failed"
    );
    let body = error_body(status, error, "server_error");
    write_sse_data(stream, &body)?;
    Ok(())
}

/// Write and flush one SSE data event as a single fallible operation, so a timeout or error returns
/// immediately and no later finalization frame is attempted. One call is one write-deadline window;
/// under a test build each call is also recorded on `sse_write_probe`, so "no later frame is
/// attempted" is counted rather than inferred from wall time.
pub(crate) fn write_sse_data(
    stream: &mut TcpStream,
    data: &impl std::fmt::Display,
) -> std::io::Result<()> {
    #[cfg(test)]
    let result = sse_write_probe::attempt(|| write_sse_frame(stream, data));
    #[cfg(not(test))]
    let result = write_sse_frame(stream, data);
    result
}

/// The one blocking socket write behind [`write_sse_data`]: the frame bytes, then the flush that
/// carries them. Split out so a test build can record the attempt around the whole operation.
fn write_sse_frame(stream: &mut TcpStream, data: &impl std::fmt::Display) -> std::io::Result<()> {
    write!(stream, "data: {data}\n\n")?;
    stream.flush()
}

/// An SSE comment sent while tool output is buffered. It carries no model output, but its write/flush
/// detects disconnect and backpressure at each generated token.
pub(crate) fn write_tool_stream_heartbeat(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b": poot-tool-buffering\n\n")?;
    stream.flush()
}

/// Gate streaming success finalization on an explicit `Done` from [`submit`], before flushing held text
/// or building finish/usage metadata.
pub(crate) fn require_stream_done<T>(
    stream: &mut TcpStream,
    metrics: &Metrics,
    result: std::result::Result<T, GenerationError>,
) -> Result<Option<T>> {
    match result {
        Ok(done) => Ok(Some(done)),
        Err(error) => {
            stream_generation_error(stream, metrics, &error)?;
            Ok(None)
        }
    }
}

/// Stream a completion as SSE: headers, a `data:` chunk per token, then a throughput-receipt chunk
/// (usage + timing) and `data: [DONE]`. Once generation starts the headers are sent, so a mid-stream
/// failure emits an SSE error event and closes without success metadata. The receipt reports
/// time-to-first-token, total time, and decode tok/s.
pub(crate) fn stream_completion(
    stream: &mut TcpStream,
    engine: Option<&Sender<Job>>,
    runner: &Runner,
    metrics: &Metrics,
    req: &CompletionReq,
) -> Result<()> {
    let _inflight = InFlight::enter(metrics);
    // Validate before sending the 200 + SSE headers: a bad `guided_json`/`guided_regex` or failed encode
    // must be a 400, not a 200 followed by a broken stream.
    let prompt_tokens = match runner.encode(&req.prompt) {
        Ok(t) => t.len(),
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    let sampler = match apply_guided(
        sampler_of(&req.sampling),
        runner,
        &req.guided_choice,
        &req.guided_regex,
        &req.guided_grammar,
        &req.guided_json,
        &req.response_format,
        &None, // completions carry no tools
        &None,
    ) {
        Ok(s) => s,
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    // Acquire the `lora_adapter` lease before committing to the stream; an unknown name is a 400.
    let lora_adapter = match resolve_lora_adapter(runner, &req.lora_adapter) {
        Ok(lease) => lease,
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    // The bound comes before the engine lookup, and the lookup before the 200: a valid request that no
    // engine can take is a 503 status, not a broken stream.
    let max_new = req.effective_max_tokens();
    let tx = match require_engine(engine) {
        Ok(tx) => tx,
        Err(error) => return respond_generation_refusal(stream, &error),
    };
    // Validation passed: commit to streaming.
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    let start = Instant::now();
    let mut ttft: Option<Duration> = None;
    let stops = Stop::list(&req.stop);
    // Hold back any pending stop-sequence suffix so the stop marker never leaks; `buf.hit` records a match.
    let mut buf = StreamStops::new(stops.clone());
    let mut full_text = String::new();
    let res = submit(
        tx,
        &req.prompt,
        max_new,
        sampler,
        stops,
        true,
        lora_adapter,
        |piece, info| {
            let emit = buf.push(piece);
            if emit.is_empty() {
                // Nothing safe to send yet (held-back partial stop marker); keep going.
                return GenerationControl::Continue(());
            }
            ttft.get_or_insert_with(|| start.elapsed());
            // Inter-token latency from the engine's `since_prev_ms`; skipped for the first token (0.0).
            if info.since_prev_ms > 0.0 {
                metrics.itl.observe(info.since_prev_ms);
            }
            tracing::trace!(
                token_idx = info.token_idx,
                since_prev_ms = format!("{:.3}", info.since_prev_ms),
                cumulative_ms = format!("{:.3}", info.cumulative_ms),
                kv_used_ratio = format!("{:.3}", info.kv_used_ratio),
                "token"
            );
            full_text.push_str(&emit);
            let chunk = serde_json::json!({
                "object": "text_completion",
                "choices": [{ "index": 0, "text": emit, "finish_reason": serde_json::Value::Null }],
            })
            .to_string();
            // A failed write means the client is gone: stop.
            if write_sse_data(stream, &chunk).is_ok() {
                GenerationControl::Continue(())
            } else {
                GenerationControl::Break(())
            }
        },
    );
    let Some((_tokens, _logprobs, natural)) = require_stream_done(stream, metrics, res)? else {
        return Ok(());
    };
    // Flush any held-back tail that was not a stop, then a final chunk carrying finish_reason.
    let tail = buf.finish();
    if !tail.is_empty() {
        full_text.push_str(&tail);
        let chunk = serde_json::json!({
            "object": "text_completion",
            "choices": [{ "index": 0, "text": tail, "finish_reason": serde_json::Value::Null }],
        });
        write_sse_data(stream, &chunk)?;
    }
    let finish = if buf.hit || natural { "stop" } else { "length" };
    let final_chunk = serde_json::json!({
        "object": "text_completion",
        "choices": [{ "index": 0, "text": "", "finish_reason": finish }],
    });
    write_sse_data(stream, &final_chunk)?;
    let completion_tokens = runner.token_count(&full_text).unwrap_or(0);
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(total_ms);
    // Record TTFT only when a token streamed.
    metrics.record_request(
        total_ms,
        prompt_tokens,
        ttft.map(|d| d.as_secs_f64() * 1000.0),
    );
    tracing::debug!(
        prompt_tokens,
        completion_tokens,
        ttft_ms = format!("{ttft_ms:.1}"),
        total_ms = format!("{total_ms:.1}"),
        finish,
        "stream completion done"
    );
    if !suppress_stream_usage(&req.stream_options) {
        let receipt = stream_receipt(
            "text_completion",
            prompt_tokens,
            completion_tokens,
            ttft_ms,
            total_ms,
        );
        write_sse_data(stream, &receipt)?;
    }
    write_sse_data(stream, &"[DONE]")?;
    Ok(())
}

/// Build the final SSE throughput-receipt chunk: OpenAI-shaped `usage` (with `choices: []`) plus a poot
/// `timing` extension (time-to-first-token, total wall time, decode tokens/sec). `object` matches the
/// stream (`text_completion` or `chat.completion.chunk`).
pub(crate) fn stream_receipt(
    object: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    ttft_ms: f64,
    total_ms: f64,
) -> String {
    let round2 = |x: f64| (x * 100.0).round() / 100.0;
    // tok/s over total wall time (prefill + decode); 0 when nothing was generated.
    let tps = if total_ms > 0.0 && completion_tokens > 0 {
        completion_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    serde_json::json!({
        "object": object,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "timing": {
            "ttft_ms": round2(ttft_ms),
            "total_ms": round2(total_ms),
            "tokens_per_second": round2(tps),
        },
    })
    .to_string()
}

/// A stop-aware streaming buffer: emits only text that cannot be part of a stop string, holding back any
/// suffix that could start one. This suppresses end markers such as `<|im_end|>`, which the engine streams
/// before halting and which may be split across token boundaries.
pub(crate) struct StreamStops {
    pub(crate) stops: Vec<String>,
    /// Text seen but not yet safe to emit (a possible stop prefix, or past a hit stop).
    pub(crate) pending: String,
    /// Set once a full stop string has been seen; no further text is emitted.
    pub(crate) hit: bool,
}

impl StreamStops {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        StreamStops {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
            hit: false,
        }
    }

    /// Feed a generated piece; return the text now safe to emit. Once a stop is hit, returns "" forever.
    pub(crate) fn push(&mut self, piece: &str) -> String {
        if self.hit {
            return String::new();
        }
        self.pending.push_str(piece);
        // A full stop in the buffer: emit everything before it and stop for good.
        if let Some(i) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
        {
            self.hit = true;
            let out = self.pending[..i].to_string();
            self.pending.clear();
            return out;
        }
        // No full stop: hold back the longest pending suffix that is a prefix of some stop; emit the rest.
        let hold = self.max_held_suffix();
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        out
    }

    /// Flush any held-back text at end of generation (no stop was hit on this tail).
    pub(crate) fn finish(&mut self) -> String {
        if self.hit {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    /// Length (bytes) of the longest suffix of `pending` that is a strict prefix of some stop string.
    /// Respects UTF-8 char boundaries.
    pub(crate) fn max_held_suffix(&self) -> usize {
        let max_len = self.stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let plen = self.pending.len();
        let upper = max_len.min(plen);
        for k in (1..=upper).rev() {
            let start = plen - k;
            if !self.pending.is_char_boundary(start) {
                continue;
            }
            let suffix = &self.pending[start..];
            if self
                .stops
                .iter()
                .any(|s| s.len() > suffix.len() && s.starts_with(suffix))
            {
                return k;
            }
        }
        0
    }
}

/// Encode one embedding vector per OpenAI `encoding_format`: a JSON number array (`"float"`, default) or
/// a base64 string of little-endian f32 bytes (`"base64"`). openai-python requests base64 by default.
pub(crate) fn embedding_value(emb: &[f32], base64_fmt: bool) -> serde_json::Value {
    if base64_fmt {
        use base64::Engine;
        let mut bytes = Vec::with_capacity(emb.len() * 4);
        for f in emb {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(&bytes))
    } else {
        serde_json::json!(emb)
    }
}

/// Resolve the encoder `/v1/embeddings` pooling override (poot extension): "mean"/"cls" (case-insensitive
/// for cls) pick that mode; any other value or absent falls back to the model's `1_Pooling` mode.
pub(crate) fn encoder_pooling_override(
    req: Option<&str>,
    model_default: EncoderPooling,
) -> EncoderPooling {
    match req {
        Some(s) if s.eq_ignore_ascii_case("cls") => EncoderPooling::Cls,
        Some(s) if s.eq_ignore_ascii_case("mean") => EncoderPooling::Mean,
        _ => model_default,
    }
}

/// `/v1/embeddings` on a BERT-class encoder: L2-normalized sentence embeddings via `EncoderRunner::embed`,
/// same request/response shape as `handle_embeddings`. Runs on the connection thread's CPU eval.
pub(crate) fn handle_embeddings_encoder(
    stream: &mut TcpStream,
    enc: &EncoderRunner,
    metrics: &Metrics,
    body: &str,
) -> Result<()> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum EmbedInput {
        One(String),
        Many(Vec<String>),
    }
    #[derive(serde::Deserialize)]
    struct EmbedReq {
        input: EmbedInput,
        /// poot extension: override the encoder pooling, "mean" or "cls". Defaults to the model's `1_Pooling` mode.
        #[serde(default)]
        pooling: Option<String>,
        /// OpenAI `encoding_format`: "float" (default) or "base64" (openai-python's default).
        #[serde(default)]
        encoding_format: Option<String>,
    }
    let req: EmbedReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, e, "invalid_request_error");
        }
    };
    // Per-request pooling override, else the model's configured mode.
    let pooling = encoder_pooling_override(req.pooling.as_deref(), enc.pooling());
    let base64_fmt = req.encoding_format.as_deref() == Some("base64");
    let inputs = match req.input {
        EmbedInput::One(s) => vec![s],
        EmbedInput::Many(v) => v,
    };
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;
    for (i, text) in inputs.iter().enumerate() {
        match enc.embed_with(text, pooling) {
            Ok(emb) => {
                total_tokens += enc.token_count(text).unwrap_or(0);
                data.push(serde_json::json!({ "object": "embedding", "index": i, "embedding": embedding_value(&emb, base64_fmt) }));
            }
            Err(e) => {
                return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
            }
        }
    }
    let resp = serde_json::json!({
        "object": "list",
        "data": data,
        "model": metrics.model,
        "usage": { "prompt_tokens": total_tokens, "total_tokens": total_tokens },
    })
    .to_string();
    respond(stream, 200, &resp)
}

/// `POST /v1/rerank` on a BERT-class encoder: order `documents` by cosine similarity to `query` via
/// `EncoderRunner::rerank`. Same response shape as `handle_rerank`.
pub(crate) fn handle_rerank_encoder(
    stream: &mut TcpStream,
    enc: &EncoderRunner,
    body: &str,
) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct RerankReq {
        query: String,
        documents: Vec<String>,
        #[serde(default)]
        top_n: Option<usize>,
    }
    let req: RerankReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, e, "invalid_request_error");
        }
    };
    let docs: Vec<&str> = req.documents.iter().map(String::as_str).collect();
    let ranked = match enc.rerank(&req.query, &docs) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    let n = req.top_n.unwrap_or(ranked.len()).min(ranked.len());
    let results: Vec<_> = ranked[..n]
        .iter()
        .map(|&(index, score)| serde_json::json!({ "index": index, "relevance_score": score }))
        .collect();
    let resp = serde_json::json!({ "results": results }).to_string();
    respond(stream, 200, &resp)
}

/// `POST /v1/rerank` on a cross-encoder: score each `documents` entry against `query` jointly through
/// `BertForSequenceClassification`. Same response shape as `handle_rerank`; `relevance_score` is the raw
/// classifier logit (higher is more relevant, not bounded to [0, 1]).
pub(crate) fn handle_rerank_cross(
    stream: &mut TcpStream,
    ce: &CrossEncoderRunner,
    body: &str,
) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct RerankReq {
        query: String,
        documents: Vec<String>,
        #[serde(default)]
        top_n: Option<usize>,
    }
    let req: RerankReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, e, "invalid_request_error");
        }
    };
    let docs: Vec<&str> = req.documents.iter().map(String::as_str).collect();
    let ranked = match ce.rerank(&req.query, &docs) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    let n = req.top_n.unwrap_or(ranked.len()).min(ranked.len());
    let results: Vec<_> = ranked[..n]
        .iter()
        .map(|&(index, score)| serde_json::json!({ "index": index, "relevance_score": score }))
        .collect();
    let resp = serde_json::json!({ "results": results }).to_string();
    respond(stream, 200, &resp)
}

/// `/v1/completions` (non-streaming): greedy-continue `prompt`, return the text (OpenAI text_completion).
pub(crate) fn handle_completion(
    engine: Option<&Sender<Job>>,
    runner: &Runner,
    metrics: &Metrics,
    req: &CompletionReq,
) -> Result<String> {
    let _inflight = InFlight::enter(metrics);
    let start = Instant::now();
    let prompt_len = runner.encode(&req.prompt)?.len();
    let stops = Stop::list(&req.stop);
    let n = clamp_n(req.n);
    // Acquire once, outside the `n`-choices loop; every choice shares the lease.
    let lora_adapter = resolve_lora_adapter(runner, &req.lora_adapter).map_err(LoraAdapterError)?;
    // Bound `max_tokens` once, before any engine is asked to size an allocation from it.
    let max_new = req.effective_max_tokens();
    // `n` choices: one sequential submission each (so a request never exhausts the engine's slots), with
    // a per-choice seed so sampled choices differ (greedy choices are identical).
    let mut choices = Vec::with_capacity(n);
    let mut total_completion = 0usize;
    for i in 0..n {
        let mut sampler =
            sampler_of(&req.sampling).reseed(req.sampling.seed.wrapping_add(i as u64));
        if let Some(k) = req.logprobs {
            sampler = sampler.with_logprobs(clamp_logprobs(k));
        }
        sampler = apply_guided(
            sampler,
            runner,
            &req.guided_choice,
            &req.guided_regex,
            &req.guided_grammar,
            &req.guided_json,
            &req.response_format,
            &None, // completions carry no tools
            &None,
        )
        .map_err(GuidedDecodeError)?;
        // The engine lookup comes after every validation of the request, so a request that is invalid on
        // its face is a 400 whether or not an engine exists.
        let tx = match require_engine(engine) {
            Ok(tx) => tx,
            Err(error) => {
                account_generation_error(metrics, &error);
                return Err(error.into());
            }
        };
        let (tokens, logprobs, natural) = match submit(
            tx,
            &req.prompt,
            max_new,
            sampler,
            stops.clone(),
            false,
            lora_adapter.clone(),
            |_, _| GenerationControl::Continue(()),
        ) {
            Ok(done) => done,
            Err(error) => {
                account_generation_error(metrics, &error);
                return Err(error.into());
            }
        };
        let generated = tokens.len() - prompt_len.min(tokens.len());
        let mut text = runner.decode(&tokens[prompt_len.min(tokens.len())..])?;
        let trimmed = apply_stop(&mut text, &stops);
        // A stop sequence trims the returned text, so usage counts only what is returned.
        let completion_tokens = if trimmed {
            runner.token_count(&text)?
        } else {
            generated
        };
        total_completion += completion_tokens;
        choices.push(Choice {
            index: i,
            text,
            // EOS or a stop sequence -> "stop"; max_tokens / KV cap -> "length". `natural` comes from the
            // engine (EOS is not in `tokens`).
            finish_reason: if natural { "stop" } else { "length" }.into(),
            logprobs: req
                .logprobs
                .map(|_| logprobs_json(runner, &logprobs, completion_tokens)),
        });
    }
    let resp = CompletionResp {
        id: "cmpl-poot".into(),
        object: "text_completion".into(),
        model: metrics.model.clone(),
        choices,
        // OpenAI counts the shared prompt once; completion is summed across choices.
        usage: Usage::new(prompt_len, total_completion),
    };
    // Metrics reflect the actual prefill work (n re-prefills); TTFT is not observable when not streaming.
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics.record_request(latency_ms, prompt_len * n, None);
    tracing::debug!(
        prompt_tokens = prompt_len,
        completion_tokens = total_completion,
        latency_ms = format!("{latency_ms:.1}"),
        "completion done"
    );
    Ok(serde_json::to_string(&resp)?)
}

/// Render a chat request into its prompt and the assistant turn-end stop markers: the model's jinja
/// `chat_template` when it ships one, else the per-arch `ChatFormat`. Shared by the stream and non-stream
/// paths. The prompt must come from the model's own template; wrong markup degrades output and the
/// turn-end never matches.
pub(crate) fn chat_prompt_and_stops(runner: &Runner, req: &ChatReq) -> (String, Vec<String>) {
    // Pass the full message objects (role, content, tool_calls, tool_call_id, name) so multi-turn tool
    // conversations render through the template's tool-history branch.
    let msgs = serde_json::to_value(&req.messages).unwrap_or(serde_json::Value::Null);
    let rendered = runner.render_chat_value(&msgs, req.tools.as_ref());
    (rendered.prompt, rendered.stops)
}

/// `/v1/chat/completions` (non-streaming): render the conversation, generate the assistant turn, and trim
/// at the turn end.
pub(crate) fn handle_chat(
    engine: Option<&Sender<Job>>,
    runner: &Runner,
    metrics: &Metrics,
    req: &ChatReq,
) -> Result<String> {
    // A modality the loaded model has no front end for is refused before any generation work.
    crate::api::reject_unsupported_modality(runner, req)?;
    let _inflight = InFlight::enter(metrics);
    let start = Instant::now();
    let (prompt, turn_stops) = chat_prompt_and_stops(runner, req);

    let prompt_len = runner.encode(&prompt)?.len();
    // Stop at the template's end marker(s) as well as user stop sequences.
    let mut stops = Stop::list(&req.stop);
    stops.extend(turn_stops.iter().cloned());
    let user_stops = Stop::list(&req.stop);
    let n = clamp_n(req.n);
    // Acquire once, outside the `n`-choices loop (as in `handle_completion`).
    let lora_adapter = resolve_lora_adapter(runner, &req.lora_adapter).map_err(LoraAdapterError)?;
    // Bound `max_tokens` once (as in `handle_completion`).
    let max_new = req.effective_max_tokens();
    // `n` choices: one submission each (sequential), with a per-choice seed so sampled choices differ.
    let mut choices = Vec::with_capacity(n);
    let mut total_completion = 0usize;
    for i in 0..n {
        let mut sampler =
            sampler_of(&req.sampling).reseed(req.sampling.seed.wrapping_add(i as u64));
        if req.logprobs {
            sampler = sampler.with_logprobs(clamp_logprobs(req.top_logprobs.unwrap_or(0)));
        }
        sampler = apply_guided(
            sampler,
            runner,
            &req.guided_choice,
            &req.guided_regex,
            &req.guided_grammar,
            &req.guided_json,
            &req.response_format,
            &req.tools,
            &req.tool_choice,
        )
        .map_err(GuidedDecodeError)?;
        // The engine lookup comes after every validation of the request, so a request that is invalid on
        // its face is a 400 whether or not an engine exists.
        let tx = match require_engine(engine) {
            Ok(tx) => tx,
            Err(error) => {
                account_generation_error(metrics, &error);
                return Err(error.into());
            }
        };
        let (tokens, logprobs, natural) = match submit(
            tx,
            &prompt,
            max_new,
            sampler,
            stops.clone(),
            false,
            lora_adapter.clone(),
            |_, _| GenerationControl::Continue(()),
        ) {
            Ok(done) => done,
            Err(error) => {
                account_generation_error(metrics, &error);
                return Err(error.into());
            }
        };
        let generated = tokens.len() - prompt_len.min(tokens.len());
        let mut content = runner.decode(&tokens[prompt_len.min(tokens.len())..])?;
        // The turn ends at the earliest template end marker; user stops truncate further.
        let turn_end = apply_stop(&mut content, &turn_stops);
        let user_stopped = apply_stop(&mut content, &user_stops);
        let content = content.trim_end().to_string();
        let completion_tokens = if turn_end || user_stopped {
            runner.token_count(&content)?
        } else {
            generated
        };
        total_completion += completion_tokens;
        // If the model emitted <tool_call> markup, surface it as message.tool_calls with content=null and
        // finish_reason "tool_calls". Only when the request offered tools; otherwise it stays plain text.
        let tool_calls = if req.tools.is_some() {
            let tc = parse_tool_calls(&content);
            (!tc.is_empty()).then_some(tc)
        } else {
            None
        };
        let (message, finish_reason) = if let Some(tc) = tool_calls {
            (
                ChatRespMsg {
                    role: "assistant".into(),
                    content: None,
                    tool_calls: Some(tc),
                },
                "tool_calls".to_string(),
            )
        } else {
            // EOS (`natural`) or a template/user stop -> "stop"; only a max_tokens/cap cutoff is "length".
            let reason = if natural || turn_end || user_stopped {
                "stop"
            } else {
                "length"
            };
            (
                ChatRespMsg {
                    role: "assistant".into(),
                    content: Some(content),
                    tool_calls: None,
                },
                reason.to_string(),
            )
        };
        choices.push(ChatChoice {
            index: i,
            message,
            finish_reason,
            logprobs: req
                .logprobs
                .then(|| logprobs_json(runner, &logprobs, completion_tokens)),
        });
    }
    let resp = ChatResp {
        id: "chatcmpl-poot".into(),
        object: "chat.completion".into(),
        model: metrics.model.clone(),
        choices,
        usage: Usage::new(prompt_len, total_completion),
    };
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    metrics.record_request(latency_ms, prompt_len * n, None);
    tracing::debug!(
        prompt_tokens = prompt_len,
        completion_tokens = total_completion,
        latency_ms = format!("{latency_ms:.1}"),
        "chat completion done"
    );
    Ok(serde_json::to_string(&resp)?)
}

/// `/v1/chat/completions` with `stream: true`: SSE `chat.completion.chunk` events. A role delta, then a
/// content delta per token (turn-end / stop markers suppressed by `StreamStops`), then a `finish_reason`
/// chunk, the throughput receipt, and `[DONE]`. Headers are sent before generation, so a mid-stream
/// failure emits an error event and no success metadata.
pub(crate) fn stream_chat(
    stream: &mut TcpStream,
    engine: Option<&Sender<Job>>,
    runner: &Runner,
    metrics: &Metrics,
    req: &ChatReq,
) -> Result<()> {
    // Refuse an input the loaded model cannot take before the 200 headers.
    if let Err(e) = crate::api::reject_unsupported_modality(runner, req) {
        let (status, type_) = crate::api::classify_handler_error(&e);
        return respond_error(stream, status, format_args!("{e:#}"), type_);
    }
    let _inflight = InFlight::enter(metrics);
    let model = &metrics.model;
    // Validate before sending the 200 + SSE headers (as in stream_completion). The guided constraint is
    // rebuilt per choice below; this call only rejects an invalid one before the 200.
    let (prompt, turn_stops) = chat_prompt_and_stops(runner, req);
    let prompt_tokens = match runner.encode(&prompt) {
        Ok(t) => t.len(),
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    if let Err(e) = apply_guided(
        sampler_of(&req.sampling),
        runner,
        &req.guided_choice,
        &req.guided_regex,
        &req.guided_grammar,
        &req.guided_json,
        &req.response_format,
        &req.tools,
        &req.tool_choice,
    ) {
        return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
    }
    // Acquire before the streaming response so an unknown adapter is a 400.
    let lora_adapter = match resolve_lora_adapter(runner, &req.lora_adapter) {
        Ok(lease) => lease,
        Err(e) => {
            return respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error");
        }
    };
    // Bound, then engine lookup, then the 200 (as in `stream_completion`).
    let max_new = req.effective_max_tokens();
    let tx = match require_engine(engine) {
        Ok(tx) => tx,
        Err(error) => return respond_generation_refusal(stream, &error),
    };
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()?;
    let mut all_stops = Stop::list(&req.stop);
    all_stops.extend(turn_stops);
    let tools_mode = req.tools.is_some();
    let n = clamp_n(req.n);

    let start = Instant::now();
    let mut ttft: Option<Duration> = None;
    let mut total_completion = 0usize;

    // OpenAI `n` over the stream: each choice in turn (sequential submissions), every chunk carrying its
    // own `index`. A per-choice reseed makes sampled choices differ (greedy choices are identical).
    for choice_idx in 0..n {
        let sampler = {
            let s =
                sampler_of(&req.sampling).reseed(req.sampling.seed.wrapping_add(choice_idx as u64));
            apply_guided(
                s,
                runner,
                &req.guided_choice,
                &req.guided_regex,
                &req.guided_grammar,
                &req.guided_json,
                &req.response_format,
                &req.tools,
                &req.tool_choice,
            )?
        };
        // The first chunk of a choice announces the assistant role.
        let role = serde_json::json!({
            "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": choice_idx, "delta": { "role": "assistant" }, "finish_reason": serde_json::Value::Null }],
        });
        if write_sse_data(stream, &role).is_err() {
            return Ok(()); // client gone before this choice started.
        }

        // When the request offers `tools`, buffer the output instead of streaming content deltas, since
        // `<tool_call>` markup must not reach the client as content. At choice end: parsed calls become a
        // `delta.tool_calls` chunk; otherwise the buffered text is flushed as one content delta.
        let mut buf = StreamStops::new(all_stops.clone());
        let mut full_content = String::new();
        let res = submit(
            tx,
            &prompt,
            max_new,
            sampler,
            all_stops.clone(),
            true,
            lora_adapter.clone(),
            |piece, info| {
                let emit = buf.push(piece);
                if tools_mode && write_tool_stream_heartbeat(stream).is_err() {
                    // Probe before the empty-emission return: `StreamStops` may hold a long stop prefix
                    // while every piece stays private.
                    return GenerationControl::Break(());
                }
                if emit.is_empty() {
                    // Held-back partial stop marker; keep going.
                    return GenerationControl::Continue(());
                }
                ttft.get_or_insert_with(|| start.elapsed());
                // Inter-token latency from the engine's `since_prev_ms`.
                if info.since_prev_ms > 0.0 {
                    metrics.itl.observe(info.since_prev_ms);
                }
                tracing::trace!(
                    token_idx = info.token_idx,
                    since_prev_ms = format!("{:.3}", info.since_prev_ms),
                    cumulative_ms = format!("{:.3}", info.cumulative_ms),
                    kv_used_ratio = format!("{:.3}", info.kv_used_ratio),
                    "token"
                );
                full_content.push_str(&emit);
                if tools_mode {
                    // Output stays buffered; the heartbeat above already probed transport.
                    return GenerationControl::Continue(());
                }
                let chunk = serde_json::json!({
                    "object": "chat.completion.chunk", "model": model,
                    "choices": [{ "index": choice_idx, "delta": { "content": emit }, "finish_reason": serde_json::Value::Null }],
                });
                // A failed write means the client is gone: stop.
                if write_sse_data(stream, &chunk).is_ok() {
                    GenerationControl::Continue(())
                } else {
                    GenerationControl::Break(())
                }
            },
        );
        let Some((_tokens, _logprobs, natural)) = require_stream_done(stream, metrics, res)? else {
            return Ok(());
        };
        // Flush any held-back tail that turned out not to be a stop.
        let tail = buf.finish();
        if !tail.is_empty() {
            full_content.push_str(&tail);
            if !tools_mode {
                let chunk = serde_json::json!({
                    "object": "chat.completion.chunk", "model": model,
                    "choices": [{ "index": choice_idx, "delta": { "content": tail }, "finish_reason": serde_json::Value::Null }],
                });
                write_sse_data(stream, &chunk)?;
            }
        }
        // In tools mode, parse the buffered output: tool calls -> a tool_calls delta + "tool_calls" finish;
        // otherwise flush the buffer as one content delta and finish normally.
        let tool_calls = if tools_mode {
            parse_tool_calls(&full_content)
        } else {
            Vec::new()
        };
        let finish = if !tool_calls.is_empty() {
            let deltas = tool_call_stream_deltas(&tool_calls);
            let chunk = serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{ "index": choice_idx, "delta": { "tool_calls": deltas }, "finish_reason": serde_json::Value::Null }],
            });
            write_sse_data(stream, &chunk)?;
            "tool_calls"
        } else {
            if tools_mode && !full_content.is_empty() {
                // Tools offered but no call emitted: deliver the buffered reply as one content delta.
                let chunk = serde_json::json!({
                    "object": "chat.completion.chunk", "model": model,
                    "choices": [{ "index": choice_idx, "delta": { "content": full_content.clone() }, "finish_reason": serde_json::Value::Null }],
                });
                write_sse_data(stream, &chunk)?;
            }
            // A stop sequence (buf.hit) or EOS (`natural`) -> "stop"; only a max_tokens/cap cutoff is "length".
            if buf.hit || natural { "stop" } else { "length" }
        };
        let done = serde_json::json!({
            "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": choice_idx, "delta": {}, "finish_reason": finish }],
        });
        write_sse_data(stream, &done)?;
        total_completion += runner.token_count(full_content.trim_end()).unwrap_or(0);
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.map(|d| d.as_secs_f64() * 1000.0).unwrap_or(total_ms);
    metrics.record_request(
        total_ms,
        prompt_tokens,
        ttft.map(|d| d.as_secs_f64() * 1000.0),
    );
    tracing::debug!(
        prompt_tokens,
        completion_tokens = total_completion,
        ttft_ms = format!("{ttft_ms:.1}"),
        total_ms = format!("{total_ms:.1}"),
        "stream chat done"
    );
    if !suppress_stream_usage(&req.stream_options) {
        let receipt = stream_receipt(
            "chat.completion.chunk",
            prompt_tokens,
            total_completion,
            ttft_ms,
            total_ms,
        );
        write_sse_data(stream, &receipt)?;
    }
    write_sse_data(stream, &"[DONE]")?;
    Ok(())
}

/// Test-only count of response-frame write attempts, so a test can assert how many frames a handler
/// tried after a write deadline fired instead of inferring it from wall time (which also carries work
/// the deadline does not govern). Like `startup::select_serve_route_probe`, it is compiled out of
/// production builds, so production keeps no seam. Thread-local because the connection handler both
/// records and reads it, which also keeps tests that share a process under `cargo test` from seeing
/// each other's attempts. Declared last so `clippy::items_after_test_module` stays quiet.
#[cfg(test)]
pub(crate) mod sse_write_probe {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    #[derive(Default)]
    struct Record {
        /// frame writes recorded on this thread.
        attempts: usize,
        /// cumulative time inside those writes; a saturated attempt costs one write deadline.
        blocked: Duration,
    }

    thread_local! {
        static RECORD: RefCell<Record> = RefCell::new(Record::default());
    }

    /// Run one frame write, recording its duration on this thread, and return its result unchanged.
    pub(crate) fn attempt(write: impl FnOnce() -> std::io::Result<()>) -> std::io::Result<()> {
        let started = Instant::now();
        let result = write();
        RECORD.with(|record| {
            let mut record = record.borrow_mut();
            record.attempts += 1;
            record.blocked += started.elapsed();
        });
        result
    }

    /// `(attempts, blocked)` recorded so far on the calling thread.
    pub(crate) fn summary() -> (usize, Duration) {
        RECORD.with(|record| (record.borrow().attempts, record.borrow().blocked))
    }
}
