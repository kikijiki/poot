//! Endpoint handlers: completions and chat (streamed or not), embeddings and rerank. Each validates the
//! request, turns it into jobs on the one engine queue and turns their events into the response. The
//! request-to-job conversion is [`Shaped`]; `n` choices are `n` concurrent jobs, all waited on from the
//! connection's one thread.

use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use anyhow::Result;
use poot_llm::GenerationControl;
use poot_llm::Sampler;
use poot_llm::driver::{AdapterRef, Encoded, ModelHandle};
use poot_llm::encoders::{Pooling, pool, rank};
use poot_models::model::ModelOutput;

use crate::api::{
    ChatChoice, ChatReq, ChatResp, ChatRespMsg, Choice, CompletionReq, CompletionResp,
    GuidedDecodeError, LoraAdapterError, RefusedRequest, Stop, Usage, apply_guided, check_choices,
    clamp_logprobs, clamp_n, classify_handler_error, is_image_part, logprobs_json,
    parse_tool_calls, reject_unsupported_modality, resolve_lora_adapter, sampler_of,
    suppress_stream_usage, tool_call_stream_deltas,
};
use crate::http::{error_body, respond_error, write_all_within};
use crate::lifecycle::{ClientGone, GenerationRequest, peer_closed, submit, submit_encode};
use crate::metrics::{InFlight, Metrics};
use crate::types::{Completion, EncodeRow, GenerationError, PromptInput, Served};
use crate::vision::{ImageClient, ImageError, ImageTurn};

/// Count only response-side transport failures. Explicit `Failed`, `Done`, and receiver cancellation are
/// already counted at their engine terminal boundary; a dead engine/job channel has no producer left.
pub(crate) fn account_generation_error(metrics: &Metrics, error: &GenerationError) {
    if error.needs_response_error_metric() {
        metrics.requests_errored.fetch_add(1, Ordering::Relaxed);
    }
}

/// Answer a request that failed validation, before any header is written: the status and OpenAI error
/// type its kind decides ([`classify_handler_error`]).
fn respond_shaping_error(stream: &mut TcpStream, error: &anyhow::Error) -> Result<()> {
    let (status, type_) = classify_handler_error(error);
    respond_error(stream, status, format_args!("{error:#}"), type_)
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
    let body = error_body(status, error, error.error_type());
    write_sse_data(stream, &body)?;
    Ok(())
}

/// A recorder around one frame write: runs `write` and returns its result. A test installs one on its
/// handler thread to count attempts and time them; none is installed in a served build.
type FrameRecorder = fn(&mut dyn FnMut() -> std::io::Result<()>) -> std::io::Result<()>;

thread_local! {
    static FRAME_RECORDER: std::cell::Cell<Option<FrameRecorder>> = const { std::cell::Cell::new(None) };
}

/// Write and flush one SSE data event as a single fallible operation, so a timeout or error returns
/// immediately and no later finalization frame is attempted. One call is one write-deadline window; a
/// recorder installed on the thread sees each call, so "no later frame is attempted" is counted rather
/// than inferred from wall time.
pub(crate) fn write_sse_data(
    stream: &mut TcpStream,
    data: &impl std::fmt::Display,
) -> std::io::Result<()> {
    match FRAME_RECORDER.get() {
        Some(record) => record(&mut || write_sse_frame(stream, data)),
        None => write_sse_frame(stream, data),
    }
}

/// The one blocking socket write behind [`write_sse_data`]: the frame bytes, within the write deadline
/// as a whole. Split out so a recorder can wrap the whole operation.
fn write_sse_frame(stream: &mut TcpStream, data: &impl std::fmt::Display) -> std::io::Result<()> {
    write_all_within(stream, format!("data: {data}\n\n").as_bytes())
}

/// An SSE comment sent while tool output is buffered. It carries no model output, but its write/flush
/// detects disconnect and backpressure at each generated token.
pub(crate) fn write_tool_stream_heartbeat(stream: &mut TcpStream) -> std::io::Result<()> {
    write_all_within(stream, b": poot-tool-buffering\n\n")
}

const SSE_HEADERS: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";

/// A request validated and converted: what every one of its `n` jobs shares. Everything that can refuse
/// the request happens in the constructors, before the first response byte.
struct Shaped {
    prompt: PromptInput,
    /// User stop sequences, and a chat's turn-end markers.
    stops: Vec<String>,
    max_new: usize,
    /// The choices: one concurrent job each.
    n: usize,
    /// The request's seed (its own, else one drawn from the OS); choice `i` samples from `seed + i`.
    seed: u64,
    lora_adapter: AdapterRef,
}

impl Shaped {
    /// The job of one choice.
    fn request(&self, sampler: Sampler, stream: bool) -> GenerationRequest {
        GenerationRequest {
            prompt: self.prompt.clone(),
            max_new: self.max_new,
            sampler,
            stop: self.stops.clone(),
            stream,
            lora_adapter: self.lora_adapter,
        }
    }
}

/// The prompt as token ids; a prompt the tokenizer refuses is the client's.
fn tokenize(model: &ModelHandle, prompt: &str) -> Result<Vec<u32>> {
    model
        .text()
        .encode(prompt)
        .map_err(|e| anyhow::Error::new(RefusedRequest(format!("{e:#}"))))
}

fn shape_completion(served: &Served, req: &CompletionReq) -> Result<Shaped> {
    if req.stream && req.logprobs.is_some() {
        return Err(RefusedRequest(
            "logprobs are not streamed: send the request without `stream`".into(),
        )
        .into());
    }
    req.sampling.check()?;
    let stops = Stop::list(&req.stop)?;
    let seed = req.sampling.resolve_seed()?;
    let caps = served.view.caps();
    let tokens = tokenize(&served.model, &req.prompt)?;
    let n = clamp_n(req.n);
    check_choices(n, &caps)?;
    let lora_adapter =
        resolve_lora_adapter(&served.view, &req.lora_adapter).map_err(LoraAdapterError)?;
    Ok(Shaped {
        prompt: PromptInput::text(tokens),
        stops,
        // Bound once, before any engine is asked to size an allocation from it.
        max_new: req.effective_max_tokens(),
        n,
        seed,
        lora_adapter,
    })
}

/// One choice's sampler: reseeded per choice so sampled choices differ (greedy ones are identical), with
/// the request's logprobs and guided constraint. A constraint that cannot be built is the client's.
fn completion_sampler(
    model: &ModelHandle,
    req: &CompletionReq,
    seed: u64,
    choice: usize,
) -> Result<Sampler> {
    let mut sampler = sampler_of(&req.sampling).reseed(seed.wrapping_add(choice as u64));
    if let Some(k) = req.logprobs {
        sampler = sampler.with_logprobs(clamp_logprobs(k));
    }
    apply_guided(
        sampler,
        model,
        &req.guided_choice,
        &req.guided_regex,
        &req.guided_grammar,
        &req.guided_json,
        &req.response_format,
        &None, // completions carry no tools
        &None,
    )
    .map_err(|e| anyhow::Error::new(GuidedDecodeError(e)))
}

fn shape_chat(served: &Served, req: &ChatReq, client_gone: ClientGone<'_>) -> Result<Shaped> {
    if req.stream && req.logprobs {
        return Err(RefusedRequest(
            "logprobs are not streamed: send the request without `stream`".into(),
        )
        .into());
    }
    req.sampling.check()?;
    let user_stops = Stop::list(&req.stop)?;
    let seed = req.sampling.resolve_seed()?;
    let caps = served.view.caps();
    // A modality nothing here can take is refused before any generation work.
    reject_unsupported_modality(served.images.is_some(), req)?;
    let (prompt, turn_stops) = match &served.images {
        Some(images) if req.messages.iter().any(|msg| !msg.image_urls.is_empty()) => {
            (image_prompt(images, req, client_gone)?, Vec::new())
        }
        _ => {
            let (tokens, turn_stops) = chat_tokens_and_stops(&served.model, req)?;
            (PromptInput::text(tokens), turn_stops)
        }
    };
    let n = clamp_n(req.n);
    check_choices(n, &caps)?;
    let lora_adapter =
        resolve_lora_adapter(&served.view, &req.lora_adapter).map_err(LoraAdapterError)?;
    // Stop at the template's end marker(s) as well as user stop sequences.
    let mut stops = user_stops;
    stops.extend(turn_stops);
    Ok(Shaped {
        prompt,
        stops,
        max_new: req.effective_max_tokens(),
        n,
        seed,
        lora_adapter,
    })
}

/// The prompt of a chat request whose messages carry images: the image front end renders the whole
/// conversation through the checkpoint's chat template, with an image part where each image sits, and
/// lowers the images to the embedding rows at their placeholders. Each message's images come first in
/// its content, then its text; the request's images are those of its messages in order.
fn image_prompt(
    images: &ImageClient,
    req: &ChatReq,
    client_gone: ClientGone<'_>,
) -> Result<PromptInput> {
    let refuse = |reason: &str| anyhow::Error::new(RefusedRequest(reason.into()));
    // The tools go into the chat template, whose tool branch the image template does not have;
    // `stop`, `response_format` and the guided fields act on the sampler and the stop scanner, so they
    // apply unchanged.
    if req.tools.is_some() || req.tool_choice.is_some() {
        return Err(refuse(
            "`tools` and `tool_choice` are not supported with images: the image chat template has no tool branch",
        ));
    }
    let mut urls = Vec::new();
    let mut turns = Vec::with_capacity(req.messages.len());
    for message in &req.messages {
        if message.role == "tool" || message.tool_calls.is_some() {
            return Err(refuse(
                "tool calls and tool results are not supported with images: the image chat template has no tool branch",
            ));
        }
        let image_parts = message
            .non_text_parts
            .iter()
            .filter(|part| is_image_part(part))
            .count();
        if image_parts != message.image_urls.len() {
            return Err(refuse("an image part carries no `image_url`"));
        }
        urls.extend(message.image_urls.iter().cloned());
        turns.push(ImageTurn {
            role: message.role.clone(),
            images: message.image_urls.len(),
            text: message.content.clone().unwrap_or_default(),
        });
    }
    let prepared = images
        .prepare(&urls, turns, client_gone)
        .map_err(|error| match error {
            ImageError::Refused(refused) => anyhow::Error::new(refused),
            ImageError::Interrupted(interrupted) => anyhow::Error::new(interrupted),
            failed @ ImageError::Failed(_) => anyhow::Error::new(failed),
        })?;
    Ok(PromptInput {
        tokens: prepared.prompt,
        inputs: prepared.inputs,
    })
}

fn chat_sampler(model: &ModelHandle, req: &ChatReq, seed: u64, choice: usize) -> Result<Sampler> {
    let mut sampler = sampler_of(&req.sampling).reseed(seed.wrapping_add(choice as u64));
    if req.logprobs {
        sampler = sampler.with_logprobs(clamp_logprobs(req.top_logprobs.unwrap_or(0)));
    }
    apply_guided(
        sampler,
        model,
        &req.guided_choice,
        &req.guided_regex,
        &req.guided_grammar,
        &req.guided_json,
        &req.response_format,
        &req.tools,
        &req.tool_choice,
    )
    .map_err(|e| anyhow::Error::new(GuidedDecodeError(e)))
}

/// Every choice's sampler, built before any job is submitted so a bad constraint is refused up front.
fn samplers(n: usize, build: impl Fn(usize) -> Result<Sampler>) -> Result<Vec<Sampler>> {
    (0..n).map(build).collect()
}

/// The completion's generated tokens as the response counts them: a stop string cuts the text, so the
/// tokens are those of the cut text; any other ending counts the tokens generated.
fn completion_tokens(model: &ModelHandle, completion: &Completion, text: &str) -> Result<usize> {
    if completion.finish == crate::types::FinishReason::Stop {
        Ok(model.text().token_count(text)?)
    } else {
        Ok(completion.tokens.len())
    }
}

/// Stream a completion as SSE: headers, a `data:` chunk per released piece of every choice, then each
/// choice's finish chunk, a throughput-receipt chunk (usage + timing) and `data: [DONE]`. Once
/// generation starts the headers are sent, so a mid-stream failure emits an SSE error event and closes
/// without success metadata. The receipt reports time-to-first-token, total time, and decode tok/s.
pub(crate) fn stream_completion(
    stream: &mut TcpStream,
    served: &Served,
    metrics: &Metrics,
    req: &CompletionReq,
) -> Result<()> {
    let _inflight = InFlight::enter(metrics);
    // Validate before sending the 200 + SSE headers: a bad `guided_json`/`guided_regex`, a failed encode
    // or a refused request must be a 4xx, not a 200 followed by a broken stream.
    let prepared = shape_completion(served, req).and_then(|shaped| {
        let choices = samplers(shaped.n, |i| {
            completion_sampler(&served.model, req, shaped.seed, i)
        })?;
        Ok((shaped, choices))
    });
    let (shaped, choices) = match prepared {
        Ok(prepared) => prepared,
        Err(e) => return respond_shaping_error(stream, &e),
    };
    write_all_within(stream, SSE_HEADERS)?;
    let start = Instant::now();
    let mut ttft: Option<Duration> = None;
    // A second handle on the socket, for the hang-up probe: the token closure owns the writes.
    let probe = stream.try_clone().ok();
    let gone = || probe.as_ref().is_some_and(peer_closed);
    let submitted = submit(
        &served.engine,
        metrics,
        served.max_queue_wait,
        choices
            .into_iter()
            .map(|sampler| shaped.request(sampler, true))
            .collect(),
        &gone,
        |index, piece, info| {
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
            let chunk = serde_json::json!({
                "object": "text_completion",
                "choices": [{ "index": index, "text": piece, "finish_reason": serde_json::Value::Null }],
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
    let completions = match submitted {
        Ok(completions) => completions,
        Err(error) => return stream_generation_error(stream, metrics, &error),
    };
    let mut total_completion = 0usize;
    for (index, completion) in completions.iter().enumerate() {
        let final_chunk = serde_json::json!({
            "object": "text_completion",
            "choices": [{ "index": index, "text": "", "finish_reason": completion.finish.api_name() }],
        });
        write_sse_data(stream, &final_chunk)?;
        total_completion += completion_tokens(&served.model, completion, &completion.text)?;
    }
    let prompt_tokens = completions[0].prompt_tokens;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.map_or(total_ms, |d| d.as_secs_f64() * 1000.0);
    tracing::debug!(
        prompt_tokens,
        completion_tokens = total_completion,
        ttft_ms = format!("{ttft_ms:.1}"),
        total_ms = format!("{total_ms:.1}"),
        "stream completion done"
    );
    if !suppress_stream_usage(&req.stream_options) {
        let receipt = stream_receipt(
            "text_completion",
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

/// `/v1/completions` (non-streaming): greedy-continue `prompt`, return the text (OpenAI text_completion).
pub(crate) fn handle_completion(
    served: &Served,
    metrics: &Metrics,
    req: &CompletionReq,
    client_gone: ClientGone<'_>,
) -> Result<String> {
    let _inflight = InFlight::enter(metrics);
    let shaped = shape_completion(served, req)?;
    let choices = samplers(shaped.n, |i| {
        completion_sampler(&served.model, req, shaped.seed, i)
    })?;
    let completions = submit(
        &served.engine,
        metrics,
        served.max_queue_wait,
        choices
            .into_iter()
            .map(|sampler| shaped.request(sampler, false))
            .collect(),
        client_gone,
        |_, _, _| GenerationControl::Continue(()),
    )
    .inspect_err(|error| account_generation_error(metrics, error))?;
    let mut total_completion = 0usize;
    let mut choices = Vec::with_capacity(completions.len());
    for (index, completion) in completions.iter().enumerate() {
        // A stop sequence trims the returned text, so usage counts only what is returned.
        let completion_tokens = completion_tokens(&served.model, completion, &completion.text)?;
        total_completion += completion_tokens;
        choices.push(Choice {
            index,
            text: completion.text.clone(),
            finish_reason: completion.finish.api_name().into(),
            logprobs: req.logprobs.map(|_| {
                logprobs_json(served.model.text(), &completion.logprobs, completion_tokens)
            }),
        });
    }
    let prompt_tokens = completions[0].prompt_tokens;
    tracing::debug!(
        prompt_tokens,
        completion_tokens = total_completion,
        "completion done"
    );
    let resp = CompletionResp {
        id: "cmpl-poot".into(),
        object: "text_completion".into(),
        model: metrics.model.clone(),
        choices,
        // OpenAI counts the shared prompt once; completion is summed across choices.
        usage: Usage::new(prompt_tokens, total_completion),
    };
    Ok(serde_json::to_string(&resp)?)
}

/// Render a chat request into its prompt tokens and the assistant turn-end stop markers: the jinja chat template
/// (the checkpoint's, or the `--chat-template` given at startup) when there is one, else the per-arch
/// `ChatFormat`. Shared by the stream and non-stream paths. The prompt must come from the template; wrong
/// markup degrades output and the turn-end never matches. A template that fails to render the conversation
/// refuses the request rather than answering in another format. Message contents are tokenized as plain text:
/// a message cannot spell a control token.
pub(crate) fn chat_tokens_and_stops(
    handle: &ModelHandle,
    req: &ChatReq,
) -> Result<(Vec<u32>, Vec<String>)> {
    // Pass the full message objects (role, content, tool_calls, tool_call_id, name) so multi-turn tool
    // conversations render through the template's tool-history branch.
    let msgs = serde_json::to_value(&req.messages).unwrap_or(serde_json::Value::Null);
    // The conversation is the client's: a template that refuses it (it raises, or spends its budget) is a 400.
    // The tokenizer failing on the rendered prompt is not the client's doing.
    let rendered = handle
        .text()
        .render_chat_value(&msgs, req.tools.as_ref())
        .map_err(|e| anyhow::Error::new(RefusedRequest(format!("{e:#}"))))?;
    let tokens = handle.text().encode_chat(&rendered)?;
    Ok((tokens, rendered.stops))
}

/// `/v1/chat/completions` (non-streaming): render the conversation, generate the assistant turn, and trim
/// at the turn end.
pub(crate) fn handle_chat(
    served: &Served,
    metrics: &Metrics,
    req: &ChatReq,
    client_gone: ClientGone<'_>,
) -> Result<String> {
    let _inflight = InFlight::enter(metrics);
    let shaped = shape_chat(served, req, client_gone)?;
    let choices = samplers(shaped.n, |i| {
        chat_sampler(&served.model, req, shaped.seed, i)
    })?;
    let completions = submit(
        &served.engine,
        metrics,
        served.max_queue_wait,
        choices
            .into_iter()
            .map(|sampler| shaped.request(sampler, false))
            .collect(),
        client_gone,
        |_, _, _| GenerationControl::Continue(()),
    )
    .inspect_err(|error| account_generation_error(metrics, error))?;
    let mut total_completion = 0usize;
    let mut choices = Vec::with_capacity(completions.len());
    for (index, completion) in completions.iter().enumerate() {
        let content = completion.text.trim_end().to_string();
        let completion_tokens = completion_tokens(&served.model, completion, &content)?;
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
            (
                ChatRespMsg {
                    role: "assistant".into(),
                    content: Some(content),
                    tool_calls: None,
                },
                completion.finish.api_name().to_string(),
            )
        };
        choices.push(ChatChoice {
            index,
            message,
            finish_reason,
            logprobs: req.logprobs.then(|| {
                logprobs_json(served.model.text(), &completion.logprobs, completion_tokens)
            }),
        });
    }
    let prompt_tokens = completions[0].prompt_tokens;
    tracing::debug!(
        prompt_tokens,
        completion_tokens = total_completion,
        "chat completion done"
    );
    let resp = ChatResp {
        id: "chatcmpl-poot".into(),
        object: "chat.completion".into(),
        model: metrics.model.clone(),
        choices,
        usage: Usage::new(prompt_tokens, total_completion),
    };
    Ok(serde_json::to_string(&resp)?)
}

/// `/v1/chat/completions` with `stream: true`: SSE `chat.completion.chunk` events. A role delta per
/// choice, then a content delta per released piece (turn-end / stop markers never released), then each
/// choice's `finish_reason` chunk, the throughput receipt, and `[DONE]`. Headers are sent before
/// generation, so a mid-stream failure emits an error event and no success metadata.
pub(crate) fn stream_chat(
    stream: &mut TcpStream,
    served: &Served,
    metrics: &Metrics,
    req: &ChatReq,
) -> Result<()> {
    let _inflight = InFlight::enter(metrics);
    let model = &metrics.model;
    // A second handle on the socket, for the hang-up probe: the shaping of a request with images waits on
    // the vision thread, and the token closure owns the writes.
    let probe = stream.try_clone().ok();
    let gone = || probe.as_ref().is_some_and(peer_closed);
    // Validate before sending the 200 + SSE headers (as in stream_completion).
    let prepared = shape_chat(served, req, &gone).and_then(|shaped| {
        let choices = samplers(shaped.n, |i| {
            chat_sampler(&served.model, req, shaped.seed, i)
        })?;
        Ok((shaped, choices))
    });
    let (shaped, choices) = match prepared {
        Ok(prepared) => prepared,
        Err(e) => return respond_shaping_error(stream, &e),
    };
    write_all_within(stream, SSE_HEADERS)?;
    // When the request offers `tools`, the output is buffered instead of streamed as content deltas,
    // since `<tool_call>` markup must not reach the client as content. At choice end: parsed calls become
    // a `delta.tool_calls` chunk; otherwise the text is flushed as one content delta.
    let tools_mode = req.tools.is_some();
    let start = Instant::now();
    let mut ttft: Option<Duration> = None;
    // The first chunk of each choice announces the assistant role.
    for index in 0..shaped.n {
        let role = serde_json::json!({
            "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": index, "delta": { "role": "assistant" }, "finish_reason": serde_json::Value::Null }],
        });
        if write_sse_data(stream, &role).is_err() {
            // The client is gone before any choice started.
            return stream_generation_error(stream, metrics, &GenerationError::ClientCancelled);
        }
    }
    let submitted = submit(
        &served.engine,
        metrics,
        served.max_queue_wait,
        choices
            .into_iter()
            .map(|sampler| shaped.request(sampler, true))
            .collect(),
        &gone,
        |index, piece, info| {
            if tools_mode && write_tool_stream_heartbeat(stream).is_err() {
                return GenerationControl::Break(());
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
            if tools_mode {
                // Output stays buffered; the heartbeat above already probed transport.
                return GenerationControl::Continue(());
            }
            let chunk = serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{ "index": index, "delta": { "content": piece }, "finish_reason": serde_json::Value::Null }],
            });
            // A failed write means the client is gone: stop.
            if write_sse_data(stream, &chunk).is_ok() {
                GenerationControl::Continue(())
            } else {
                GenerationControl::Break(())
            }
        },
    );
    let completions = match submitted {
        Ok(completions) => completions,
        Err(error) => return stream_generation_error(stream, metrics, &error),
    };
    let mut total_completion = 0usize;
    for (index, completion) in completions.iter().enumerate() {
        // In tools mode, parse the buffered output: tool calls -> a tool_calls delta + "tool_calls"
        // finish; otherwise flush the buffer as one content delta and finish normally.
        let tool_calls = if tools_mode {
            parse_tool_calls(&completion.text)
        } else {
            Vec::new()
        };
        let finish = if !tool_calls.is_empty() {
            let deltas = tool_call_stream_deltas(&tool_calls);
            let chunk = serde_json::json!({
                "object": "chat.completion.chunk", "model": model,
                "choices": [{ "index": index, "delta": { "tool_calls": deltas }, "finish_reason": serde_json::Value::Null }],
            });
            write_sse_data(stream, &chunk)?;
            "tool_calls"
        } else {
            if tools_mode && !completion.text.is_empty() {
                // Tools offered but no call emitted: deliver the buffered reply as one content delta.
                let chunk = serde_json::json!({
                    "object": "chat.completion.chunk", "model": model,
                    "choices": [{ "index": index, "delta": { "content": completion.text }, "finish_reason": serde_json::Value::Null }],
                });
                write_sse_data(stream, &chunk)?;
            }
            completion.finish.api_name()
        };
        let done = serde_json::json!({
            "object": "chat.completion.chunk", "model": model,
            "choices": [{ "index": index, "delta": {}, "finish_reason": finish }],
        });
        write_sse_data(stream, &done)?;
        total_completion +=
            completion_tokens(&served.model, completion, completion.text.trim_end())?;
    }
    let prompt_tokens = completions[0].prompt_tokens;
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = ttft.map_or(total_ms, |d| d.as_secs_f64() * 1000.0);
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

/// `input` of an embeddings request: one text or several.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum EmbedInput {
    One(String),
    Many(Vec<String>),
}

#[derive(serde::Deserialize)]
struct EmbedReq {
    input: EmbedInput,
    /// poot extension: override the pooling, "mean" or "cls". Defaults to the model's `1_Pooling` mode.
    #[serde(default)]
    pooling: Option<String>,
    /// OpenAI `encoding_format`: "float" (default) or "base64" (openai-python's default).
    #[serde(default)]
    encoding_format: Option<String>,
}

#[derive(serde::Deserialize)]
struct RerankReq {
    query: String,
    documents: Vec<String>,
    #[serde(default)]
    top_n: Option<usize>,
}

/// An embeddings or rerank body, parsed: malformed JSON is the client's.
fn parse_body<T: serde::de::DeserializeOwned>(body: &str) -> Result<T> {
    serde_json::from_str(body).map_err(|e| anyhow::Error::new(RefusedRequest(e.to_string())))
}

/// The tokens of `text` as an encoder takes them: its special tokens added, and an empty text refused.
fn encoder_row(model: &ModelHandle, first: &str, second: Option<&str>) -> Result<EncodeRow> {
    let tokens = model
        .text()
        .encode_for_encoder(first, second)
        .map_err(|e| anyhow::Error::new(RefusedRequest(format!("{e:#}"))))?;
    if tokens.ids.is_empty() {
        return Err(RefusedRequest("cannot embed empty text".into()).into());
    }
    Ok(EncodeRow {
        ids: tokens.ids,
        types: tokens.types,
    })
}

/// Refuse `rows` the engine is not fitted for, naming the cap and the flag that sets it: the fitted
/// shapes are the only ones prepared, and a step never compiles another.
fn check_encode_fit(served: &Served, rows: &[EncodeRow]) -> Result<()> {
    let Some(fitted) = served.view.caps().encode else {
        return Err(RefusedRequest(
            "this model has no embedding pass: it serves no embeddings or rerank".into(),
        )
        .into());
    };
    if rows.len() > fitted.rows.get() {
        return Err(RefusedRequest(format!(
            "{} texts exceed the {} rows the engine is fitted for (--rows)",
            rows.len(),
            fitted.rows
        ))
        .into());
    }
    let longest = fitted.lengths.last().map_or(0, |len| len.get());
    if let Some((index, row)) = rows
        .iter()
        .enumerate()
        .find(|(_, row)| row.ids.len() > longest)
    {
        return Err(RefusedRequest(format!(
            "text {index} has {} tokens, over the {longest} the engine is fitted for (--max-seq)",
            row.ids.len()
        ))
        .into());
    }
    Ok(())
}

/// One `Encode` job over `rows`, answered by the engine thread; a failure is accounted here when it is a
/// transport loss.
fn encode(
    served: &Served,
    metrics: &Metrics,
    rows: Vec<EncodeRow>,
    client_gone: ClientGone<'_>,
) -> Result<Encoded> {
    check_encode_fit(served, &rows)?;
    submit_encode(&served.engine, metrics, rows, client_gone).map_err(|error| {
        account_generation_error(metrics, &error);
        error.into()
    })
}

/// The unit vectors of `encoded`'s rows, pooled as `pooling` says.
fn sentence_vectors(encoded: &Encoded, pooling: Pooling) -> Vec<Vec<f32>> {
    encoded
        .rows
        .iter()
        .map(|row| pool(row, encoded.width, pooling))
        .collect()
}

/// An embedding as the response carries it: a float array, or the base64 of its little-endian f32 bytes
/// (`"base64"`, which openai-python requests by default).
fn embedding_value(embedding: &[f32], base64_format: bool) -> serde_json::Value {
    if base64_format {
        use base64::Engine;
        let bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
    } else {
        serde_json::json!(embedding)
    }
}

/// `POST /v1/embeddings`: each input's tokens in one `Encode` step on the engine thread, pooled on the
/// host into a unit vector.
pub(crate) fn handle_embeddings(
    served: &Served,
    metrics: &Metrics,
    body: &str,
    client_gone: ClientGone<'_>,
) -> Result<String> {
    let _inflight = InFlight::enter(metrics);
    let req: EmbedReq = parse_body(body)?;
    let pooling = match req.pooling.as_deref() {
        None => served.pooling,
        Some(mode) if mode.eq_ignore_ascii_case("cls") => Pooling::Cls,
        Some(mode) if mode.eq_ignore_ascii_case("mean") => Pooling::Mean,
        Some(mode) => {
            return Err(RefusedRequest(format!(
                "unknown pooling {mode:?}: expected \"mean\" or \"cls\""
            ))
            .into());
        }
    };
    let base64_format = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return Err(RefusedRequest(format!(
                "unknown encoding_format {other:?}: expected \"float\" or \"base64\""
            ))
            .into());
        }
    };
    let inputs = match req.input {
        EmbedInput::One(text) => vec![text],
        EmbedInput::Many(texts) => texts,
    };
    if inputs.is_empty() {
        return Err(RefusedRequest("`input` holds no text".into()).into());
    }
    let rows = inputs
        .iter()
        .map(|text| encoder_row(&served.model, text, None))
        .collect::<Result<Vec<_>>>()?;
    let total_tokens: usize = rows.iter().map(|row| row.ids.len()).sum();
    let encoded = encode(served, metrics, rows, client_gone)?;
    let data: Vec<serde_json::Value> = sentence_vectors(&encoded, pooling)
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            serde_json::json!({
                "object": "embedding",
                "index": index,
                "embedding": embedding_value(embedding, base64_format),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "object": "list",
        "data": data,
        "model": metrics.model,
        "usage": { "prompt_tokens": total_tokens, "total_tokens": total_tokens },
    })
    .to_string())
}

/// `POST /v1/rerank` (Cohere/Jina-shaped): order `documents` by relevance to `query`. A model whose
/// output is classifier scores (a cross-encoder) scores each pair jointly, and `relevance_score` is the
/// raw logit (higher is more relevant, not bounded to [0, 1]); any other model ranks by the cosine of
/// the pooled sentence vectors of the query and each document.
pub(crate) fn handle_rerank(
    served: &Served,
    metrics: &Metrics,
    body: &str,
    client_gone: ClientGone<'_>,
) -> Result<String> {
    let _inflight = InFlight::enter(metrics);
    let req: RerankReq = parse_body(body)?;
    let ranked = if req.documents.is_empty() {
        Vec::new()
    } else if served
        .model
        .schema()
        .encode()
        .is_some_and(|program| matches!(program.output(), ModelOutput::Scores { .. }))
    {
        let rows = req
            .documents
            .iter()
            .map(|doc| encoder_row(&served.model, &req.query, Some(doc)))
            .collect::<Result<Vec<_>>>()?;
        let encoded = encode(served, metrics, rows, client_gone)?;
        rank(encoded.rows.iter().map(|row| row[0]))
    } else {
        let rows = std::iter::once(req.query.as_str())
            .chain(req.documents.iter().map(String::as_str))
            .map(|text| encoder_row(&served.model, text, None))
            .collect::<Result<Vec<_>>>()?;
        let encoded = encode(served, metrics, rows, client_gone)?;
        let vectors = sentence_vectors(&encoded, served.pooling);
        let (query, docs) = vectors.split_first().expect("the query is one of the rows");
        rank(
            docs.iter()
                .map(|doc| query.iter().zip(doc).map(|(a, b)| a * b).sum::<f32>()),
        )
    };
    let n = req.top_n.unwrap_or(ranked.len()).min(ranked.len());
    let results: Vec<_> = ranked[..n]
        .iter()
        .map(|&(index, score)| serde_json::json!({ "index": index, "relevance_score": score }))
        .collect();
    Ok(serde_json::json!({ "results": results }).to_string())
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

/// Test-only count of response-frame write attempts, so a test can assert how many frames a handler
/// tried after a write deadline fired instead of inferring it from wall time (which also carries work
/// the deadline does not govern). Installed on the handler's own thread, so tests that share a process
/// never see each other's attempts. Declared last so `clippy::items_after_test_module` stays quiet.
#[cfg(test)]
pub(crate) mod sse_write_probe {
    use std::cell::Cell;
    use std::time::{Duration, Instant};

    thread_local! {
        /// `(attempts, blocked)`: frame writes on this thread and the time inside them; a saturated
        /// attempt costs one write deadline.
        static RECORD: Cell<(usize, Duration)> = const { Cell::new((0, Duration::ZERO)) };
    }

    fn attempt(write: &mut dyn FnMut() -> std::io::Result<()>) -> std::io::Result<()> {
        let started = Instant::now();
        let result = write();
        RECORD.with(|record| {
            let (attempts, blocked) = record.get();
            record.set((attempts + 1, blocked + started.elapsed()));
        });
        result
    }

    /// Record the frame writes the calling thread makes from now on.
    pub(crate) fn install() {
        super::FRAME_RECORDER.set(Some(attempt));
    }

    /// `(attempts, blocked)` recorded so far on the calling thread.
    pub(crate) fn summary() -> (usize, Duration) {
        RECORD.with(|record| record.get())
    }
}
