//! OpenAI-compatible API types (request/response structs, serde) and request-shaping helpers.

use anyhow::{Context, Result, bail};
use poot_llm::driver::{AdapterRef, DriverCaps};
use poot_llm::{Sampler, TokenLogprob};

use crate::modality::Modality;
use poot_text::TextCodec;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};

use crate::types::{EngineView, GenerationError};
use crate::vision::ImageError;

/// Sampling knobs shared by `/v1/completions` and `/v1/chat/completions`, flattened into both
/// request structs. All optional; defaults are greedy with no penalties.
#[derive(Deserialize, Default)]
pub(crate) struct SamplingParams {
    #[serde(default)]
    pub(crate) temperature: f32,
    #[serde(default)]
    pub(crate) top_k: usize,
    #[serde(default = "default_top_p")]
    pub(crate) top_p: f32,
    /// The sampling seed. Absent, the request draws one from the OS ([`SamplingParams::resolve_seed`]);
    /// a given seed makes the request reproducible.
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) min_p: f32,
    /// HF/CTRL multiplicative repetition penalty (1.0 = off).
    #[serde(default = "default_rep")]
    pub(crate) repetition_penalty: f32,
    #[serde(default)]
    pub(crate) presence_penalty: f32,
    #[serde(default)]
    pub(crate) frequency_penalty: f32,
    /// OpenAI `logit_bias`: token id (as a JSON string key, per the OpenAI API) -> additive bias.
    #[serde(default)]
    pub(crate) logit_bias: std::collections::HashMap<String, f32>,
}

/// Most `logit_bias` entries a request may carry: the bias is applied to the logits at every token on
/// the engine thread, so its size is paid per token.
pub(crate) const MAX_LOGIT_BIAS_ENTRIES: usize = 10_000;

impl SamplingParams {
    /// Refuse a request whose sampling knobs cost more per token than the cap allows.
    pub(crate) fn check(&self) -> Result<(), RefusedRequest> {
        let entries = self.logit_bias.len();
        if entries > MAX_LOGIT_BIAS_ENTRIES {
            return Err(RefusedRequest(format!(
                "`logit_bias` has {entries} entries: the limit is {MAX_LOGIT_BIAS_ENTRIES}"
            )));
        }
        Ok(())
    }

    /// The seed of the request: the one it gave, else one drawn from the OS, so two unseeded requests
    /// sample differently. Draw once per request; its choices are offset from it.
    pub(crate) fn resolve_seed(&self) -> Result<u64> {
        if let Some(seed) = self.seed {
            return Ok(seed);
        }
        let mut bytes = [0u8; 8];
        SystemRandom::new()
            .fill(&mut bytes)
            .map_err(|_| anyhow::anyhow!("the OS random source failed"))?;
        Ok(u64::from_le_bytes(bytes))
    }
}

/// Build a sampler from the request's optional fields (default greedy). A positive temperature
/// enables the temperature/top-k/top-p/min-p draw; penalties and logit_bias apply on either path. The
/// caller seeds each choice's stream with [`Sampler::reseed`] from [`SamplingParams::resolve_seed`].
pub(crate) fn sampler_of(p: &SamplingParams) -> Sampler {
    let base = if p.temperature <= 0.0 {
        Sampler::greedy()
    } else {
        Sampler::new(p.temperature, p.top_k, p.top_p, 0)
    };
    // OpenAI sends logit_bias keys as token-id strings; parse them to ids and drop any non-numeric key.
    let bias = p
        .logit_bias
        .iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|id| (id, v)))
        .collect();
    base.with_penalties(
        p.repetition_penalty,
        p.presence_penalty,
        p.frequency_penalty,
    )
    .with_min_p(p.min_p)
    .with_logit_bias(bias)
}

/// Resolve a request's `lora_adapter` extension field (see [`CompletionReq::lora_adapter`]/
/// [`ChatReq::lora_adapter`]) at the HTTP layer, before the `Job` reaches the engine. `None` (absent or
/// `null`) is [`AdapterRef::Base`], so such a request behaves the same whether or not adapters are
/// loaded. A named adapter is a hard error unless the engine has loaded it: a typo must not silently
/// serve the un-adapted model, and a driver with no adapter generation (`caps.lora` is `None`) has
/// nothing to select from.
pub(crate) fn resolve_lora_adapter(view: &EngineView, name: &Option<String>) -> Result<AdapterRef> {
    let Some(name) = name.as_deref() else {
        return Ok(AdapterRef::Base);
    };
    if view.caps().lora.is_none() {
        bail!("unknown lora_adapter {name:?}: no adapter is loaded");
    }
    view.adapter(name)
        .with_context(|| format!("unknown lora_adapter {name:?}: no loaded adapter has that name"))
}

/// Build the OpenAI-style `logprobs` object from the per-token records, giving each token id its
/// surface bytes (the vocabulary piece, so a word-initial SentencePiece token keeps its leading
/// space) and the text of those bytes. Uses the chat-completions `content` array shape for both
/// endpoints. Records are truncated to `limit` so they align with the returned (possibly
/// stop-trimmed) text.
pub(crate) fn logprobs_json(
    text: &TextCodec,
    records: &[TokenLogprob],
    limit: usize,
) -> serde_json::Value {
    let entry = |id: u32, logprob: f32| -> serde_json::Value {
        let bytes = text.token_bytes(id);
        let piece = String::from_utf8_lossy(&bytes);
        serde_json::json!({ "token": piece, "logprob": logprob, "bytes": bytes })
    };
    let content: Vec<serde_json::Value> = records
        .iter()
        .take(limit)
        .map(|r| {
            let mut e = entry(r.token, r.logprob);
            let top: Vec<serde_json::Value> = r.top.iter().map(|&(id, lp)| entry(id, lp)).collect();
            e["top_logprobs"] = serde_json::Value::Array(top);
            e
        })
        .collect();
    serde_json::json!({ "content": content })
}

/// Marks a [`Result`] error as a guided-decode/structured-output constraint-compile failure: a
/// malformed `guided_regex`, `guided_json`, `guided_choice`, or `response_format` (card 224).
/// poot-llm's constraint builders return a plain `anyhow`-compatible error with no variant for "bad
/// client input" vs "internal fault", so `handle_completion`/`handle_chat` wrap the error at the one
/// call site that can only fail on client input ([`apply_guided`]) and [`classify_handler_error`]
/// reads the wrapper back via `downcast_ref`. The streaming paths (`stream_completion` /
/// `stream_chat`) validate upfront and match `apply_guided`'s `Err` directly.
#[derive(Debug)]
pub(crate) struct GuidedDecodeError(pub(crate) anyhow::Error);

impl std::fmt::Display for GuidedDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for GuidedDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

/// Spec 248 (epic 129 B8): the same "bad client input" wrapper as [`GuidedDecodeError`], for
/// [`resolve_lora_adapter`]'s unknown-adapter error, a second client-input failure
/// `handle_completion`/`handle_chat` can hit before submitting a `Job`. A separate type because
/// `GuidedDecodeError` is specifically about guided decoding; [`classify_handler_error`] checks both.
#[derive(Debug)]
pub(crate) struct LoraAdapterError(pub(crate) anyhow::Error);

impl std::fmt::Display for LoraAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for LoraAdapterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

/// A content part the loaded model cannot take: a modality it has no front end for or a part type this
/// server does not know. An explicit refusal, not a silent
/// ignore: the part would otherwise be dropped and the model would answer a different request than the
/// one sent.
#[derive(Debug)]
pub(crate) struct UnsupportedCapabilityError {
    /// The modality name (`image`, `video`, `audio`), or `content` for an unknown part type.
    pub(crate) modality: &'static str,
    pub(crate) part_type: String,
    /// What is missing.
    pub(crate) missing: &'static str,
}

impl std::fmt::Display for UnsupportedCapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported capability: {} input (content part {:?}) is not supported by the loaded model: {}",
            self.modality, self.part_type, self.missing
        )
    }
}

impl std::error::Error for UnsupportedCapabilityError {}

/// A request the engine refuses before it runs: valid JSON the server cannot serve as asked (more
/// choices than the engine has rows). A typed 400, never a silently dropped field.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct RefusedRequest(pub(crate) String);

/// The modality an OpenAI content-part `type` needs a front end for. `None` for a part type this server
/// does not know.
fn part_modality(part_type: &str) -> Option<Modality> {
    match part_type {
        "image_url" | "image" | "input_image" => Some(Modality::Image),
        "video_url" | "video" | "input_video" => Some(Modality::Video),
        "input_audio" | "audio_url" | "audio" => Some(Modality::Audio),
        _ => None,
    }
}

/// Whether a content-part `type` names an image.
pub(crate) fn is_image_part(part_type: &str) -> bool {
    part_modality(part_type) == Some(Modality::Image)
}

/// Refuse a chat request that carries a content part nothing here can take, before any generation work.
/// An image needs the image front end of a vision-language checkpoint (`image_front_end`); the other
/// modalities and an unknown part type are refused for every model.
pub(crate) fn reject_unsupported_modality(
    image_front_end: bool,
    req: &ChatReq,
) -> anyhow::Result<()> {
    for part_type in req.messages.iter().flat_map(|msg| &msg.non_text_parts) {
        let modality = part_modality(part_type);
        let missing = match modality {
            Some(Modality::Image) if image_front_end => continue,
            Some(Modality::Image) => "the loaded model has no image front end",
            Some(_) => "this server has no front end for it",
            None => "the part type is not one this server knows",
        };
        return Err(anyhow::Error::new(UnsupportedCapabilityError {
            modality: modality.map_or("content", Modality::name),
            part_type: part_type.clone(),
            missing,
        }));
    }
    Ok(())
}

/// Refuse `n` choices the engine has no rows to run at once: each choice is one job, and a request is
/// answered only when every choice is, so more choices than rows could never all be served together.
pub(crate) fn check_choices(n: usize, caps: &DriverCaps) -> anyhow::Result<()> {
    if caps.row_counts.is_empty() {
        return Err(anyhow::Error::new(RefusedRequest(
            "this model serves embeddings and rerank only: it generates no text".into(),
        )));
    }
    if n > caps.rows.get() {
        return Err(anyhow::Error::new(RefusedRequest(format!(
            "n = {n} choices exceed the engine's {} rows: each choice is one concurrent job",
            caps.rows
        ))));
    }
    Ok(())
}

/// Classify a `handle_completion`/`handle_chat` error into the HTTP status + OpenAI error `type`
/// (card 224): a [`GuidedDecodeError`] or [`LoraAdapterError`] is bad client input (400
/// `invalid_request_error`), as in the streaming routes; an [`UnsupportedCapabilityError`] is 400
/// `unsupported_capability`; a [`GenerationError`] is answered by its own status (an engine failure by
/// its class, R484-007); anything else (encode/decode failures) is a server fault (500 `server_error`).
pub(crate) fn classify_handler_error(e: &anyhow::Error) -> (u16, &'static str) {
    if e.downcast_ref::<GuidedDecodeError>().is_some()
        || e.downcast_ref::<LoraAdapterError>().is_some()
        || e.downcast_ref::<RefusedRequest>().is_some()
    {
        (400, "invalid_request_error")
    } else if e.downcast_ref::<UnsupportedCapabilityError>().is_some() {
        (400, "unsupported_capability")
    } else if let Some(error) = e.downcast_ref::<GenerationError>() {
        (error.stream_status(), error.error_type())
    } else if e.downcast_ref::<ImageError>().is_some() {
        (500, "image_front_end_error")
    } else {
        (500, "server_error")
    }
}

/// Longest `guided_regex` pattern accepted, in bytes. The compiled automaton is bounded separately
/// (`poot-text`), so this only keeps the request itself small.
const MAX_GUIDED_PATTERN_BYTES: usize = 4096;
/// Longest `guided_grammar` (GBNF source) accepted, in bytes.
const MAX_GUIDED_GRAMMAR_BYTES: usize = 16 * 1024;
/// Longest `guided_json` / `response_format` schema accepted, in bytes of its JSON text.
const MAX_GUIDED_SCHEMA_BYTES: usize = 16 * 1024;

/// Refuse a guided-decoding input longer than its cap, before any compile work.
fn check_guided_len(field: &str, len: usize, max: usize) -> Result<()> {
    if len > max {
        bail!("{field} is {len} bytes long: the limit is {max}");
    }
    Ok(())
}

/// Apply a guided-decoding constraint to the sampler from the request (specs 040 / 043 / 044).
/// `guided_choice` (a non-empty string list) constrains output to one of those strings;
/// `guided_regex` to a regex; `guided_grammar` to a GBNF grammar; `guided_json` to a JSON Schema;
/// OpenAI `response_format` (`{type: json_schema, json_schema: {schema}}`) to the same JSON-Schema
/// constraint, and `{type: json_object}` to a free-JSON constraint. Precedence: choice, regex,
/// grammar, guided_json, then response_format.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_guided(
    sampler: Sampler,
    text: &poot_text::TextCodec,
    guided_choice: &Option<Vec<String>>,
    guided_regex: &Option<String>,
    guided_grammar: &Option<String>,
    guided_json: &Option<serde_json::Value>,
    response_format: &Option<serde_json::Value>,
    tools: &Option<serde_json::Value>,
    tool_choice: &Option<serde_json::Value>,
) -> Result<Sampler> {
    // OpenAI `tool_choice` forcing (highest precedence): when the client requires a specific tool (or
    // any tool), the model's tool-call codec constrains the output to exactly one schema-conforming call
    // in its own markup. A model whose codec cannot be forced falls through to the model + template.
    if let Some(forced) = resolve_forced_tools(tools, tool_choice)?
        && let Some(constraint) = text.forced_tool_constraint(&forced)?
    {
        return Ok(sampler.with_constraint(constraint));
    }
    if let Some(choices) = guided_choice.as_ref().filter(|c| !c.is_empty()) {
        return Ok(sampler.with_constraint(text.build_choice_constraint(choices)?));
    }
    if let Some(pattern) = guided_regex.as_ref().filter(|p| !p.is_empty()) {
        check_guided_len("guided_regex", pattern.len(), MAX_GUIDED_PATTERN_BYTES)?;
        return Ok(sampler.with_constraint(text.build_regex_constraint(pattern)?));
    }
    if let Some(gbnf) = guided_grammar.as_ref().filter(|g| !g.is_empty()) {
        check_guided_len("guided_grammar", gbnf.len(), MAX_GUIDED_GRAMMAR_BYTES)?;
        return Ok(sampler.with_constraint(text.build_grammar_constraint(gbnf)?));
    }
    if let Some(schema) = guided_json {
        check_guided_len(
            "guided_json",
            schema.to_string().len(),
            MAX_GUIDED_SCHEMA_BYTES,
        )?;
        return Ok(sampler.with_constraint(text.build_json_constraint(schema)?));
    }
    // OpenAI structured outputs (the standard field clients send, vs poot's `guided_json` extension).
    if let Some(rf) = response_format {
        match response_format_constraint(rf)? {
            ResponseFormatConstraint::Schema(schema) => {
                check_guided_len(
                    "response_format schema",
                    schema.to_string().len(),
                    MAX_GUIDED_SCHEMA_BYTES,
                )?;
                return Ok(sampler.with_constraint(text.build_json_constraint(schema)?));
            }
            ResponseFormatConstraint::JsonObject => {
                return Ok(sampler.with_constraint(text.build_json_object_constraint()?));
            }
            ResponseFormatConstraint::Unconstrained => {}
        }
    }
    Ok(sampler)
}

/// What an OpenAI `response_format` object asks the sampler to enforce.
#[derive(Debug)]
pub(crate) enum ResponseFormatConstraint<'a> {
    /// `{"type":"text"}` or an absent type: no constraint.
    Unconstrained,
    /// `{"type":"json_schema", ...}`: constrain to this JSON Schema (a byte-DFA via `guided_json`).
    Schema(&'a serde_json::Value),
    /// `{"type":"json_object"}`: constrain to any well-formed JSON value of arbitrary depth (the
    /// pushdown acceptor; the regex/DFA engine cannot count brackets).
    JsonObject,
}

/// Resolve an OpenAI `response_format` object into the constraint it asks for. Errors on a
/// malformed `json_schema` (missing inner schema) and unknown types.
pub(crate) fn response_format_constraint(
    rf: &serde_json::Value,
) -> Result<ResponseFormatConstraint<'_>> {
    match rf.get("type").and_then(|t| t.as_str()) {
        Some("json_schema") => {
            let schema = rf
                .get("json_schema")
                .and_then(|j| j.get("schema"))
                .context("response_format json_schema needs a `json_schema.schema` object")?;
            Ok(ResponseFormatConstraint::Schema(schema))
        }
        // Arbitrary-depth JSON: served by the pushdown acceptor rather than the regex/DFA engine.
        Some("json_object") => Ok(ResponseFormatConstraint::JsonObject),
        Some("text") | None => Ok(ResponseFormatConstraint::Unconstrained), // explicit/implicit unconstrained
        Some(other) => bail!("unknown response_format type {other:?}"),
    }
}

/// Resolve OpenAI `tool_choice` + `tools` into the `(name, parameter-schema)` list the reply is
/// forced to call, or `None` when no forcing applies (`"auto"`/`"none"`/absent). `"required"` forces
/// any one of the tools; `{"type":"function","function":{"name":X}}` forces that named tool. Errors
/// (client-side) when forcing is requested but no matching tool is provided. A `tools` entry is
/// `{"type":"function","function":{"name":..,"parameters":..}}`; a bare
/// `{"name":..,"parameters":..}` is also accepted.
pub(crate) fn resolve_forced_tools(
    tools: &Option<serde_json::Value>,
    tool_choice: &Option<serde_json::Value>,
) -> Result<Option<Vec<(String, serde_json::Value)>>> {
    let Some(choice) = tool_choice else {
        return Ok(None);
    };
    // Forcing mode: all tools (`"required"`) or one named function.
    let named: Option<&str> = match choice {
        serde_json::Value::String(s) => match s.as_str() {
            "required" => None,
            "auto" | "none" => return Ok(None),
            other => bail!("unknown tool_choice {other:?}"),
        },
        serde_json::Value::Object(_) => Some(
            choice
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .context("tool_choice object needs a `function.name`")?,
        ),
        _ => bail!("tool_choice must be a string or a function object"),
    };
    let arr = tools
        .as_ref()
        .and_then(|t| t.as_array())
        .filter(|a| !a.is_empty())
        .context("tool_choice forces a tool call but no tools were provided")?;
    let mut out = Vec::new();
    for t in arr {
        let f = t.get("function").unwrap_or(t); // accept the wrapped or a bare function object
        let Some(name) = f.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        if let Some(want) = named
            && name != want
        {
            continue;
        }
        let params = f
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
        out.push((name.to_string(), params));
    }
    if out.is_empty() {
        bail!("tool_choice names a function that is not present in `tools`");
    }
    Ok(Some(out))
}

pub(crate) fn default_max() -> usize {
    64
}
/// Upper bound on requested generation length. An engine sizes a KV-cache allocation as
/// `prompt_len + max_new + 1` from the request, so an unbounded value is an allocation abort
/// (`max_tokens = 999999999999` asks for a multi-TB buffer) or, near `usize::MAX`, an overflow of that
/// arithmetic into an undersized buffer.
pub(crate) const MAX_NEW_TOKENS_CAP: usize = 32_768;

/// The one bound on a requested generation length: the request value, clamped to
/// [`MAX_NEW_TOKENS_CAP`]. Clamped, not rejected, like `clamp_n`. Both `ChatReq` and `CompletionReq`
/// resolve their `effective_max_tokens` through it, so the value handed to `submit` is bounded before
/// any engine sizes a KV allocation from it.
pub(crate) fn bound_max_tokens(requested: usize) -> usize {
    requested.min(MAX_NEW_TOKENS_CAP)
}

pub(crate) fn default_top_p() -> f32 {
    1.0
}
pub(crate) fn default_rep() -> f32 {
    1.0
}
pub(crate) fn default_n() -> usize {
    1
}

/// Upper bound on OpenAI `n` (card 224): the fan-out loop in `handle_completion`/`handle_chat`/
/// `stream_chat` runs `n` sequential generations and pre-sizes a `Vec` to `n`, so an unclamped `n`
/// is an unbounded-resource DoS.
pub(crate) const MAX_CHOICES: usize = 128;

/// Clamp OpenAI `n` to `[1, MAX_CHOICES]`. Clamped, not rejected, matching OpenAI's silent capping.
pub(crate) fn clamp_n(n: usize) -> usize {
    n.clamp(1, MAX_CHOICES)
}

/// Upper bound on `logprobs` / `top_logprobs` (card 224). The fields are documented "(0-20)" as in
/// OpenAI but were not enforced, and `record_logprobs` (poot-llm) allocates
/// `Vec::with_capacity(k + 2)` per generated token, so an unclamped `k` is a per-token allocation DoS.
pub(crate) const MAX_LOGPROBS: usize = 20;

/// Clamp `logprobs`/`top_logprobs` to `[0, MAX_LOGPROBS]` (OpenAI's own documented range).
pub(crate) fn clamp_logprobs(k: usize) -> usize {
    k.min(MAX_LOGPROBS)
}

/// Upper bound on client-supplied `stop` sequences. Each generated token runs an `O(stops * text)`
/// scan per active slot (the loop's `StopScanner`), so an unbounded `stop` array is a per-token CPU DoS: a ~64MB body of short strings
/// pins a worker doing millions of `contains` per token for up to `MAX_NEW_TOKENS_CAP` tokens. OpenAI
/// caps `stop` at 4; 16 leaves headroom. Truncated, not rejected, like `clamp_n`/`clamp_logprobs`.
pub(crate) const MAX_STOP_SEQUENCES: usize = 16;
/// Longest single `stop` string, in bytes; a longer one is a 400. The scan is `O(bytes)` per token.
pub(crate) const MAX_STOP_BYTES: usize = 1024;
/// Most bytes all of a request's kept `stop` strings may hold together; more is a 400.
pub(crate) const MAX_TOTAL_STOP_BYTES: usize = 4096;

/// OpenAI's `stop` is a string or an array of strings; accept either.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Stop {
    One(String),
    Many(Vec<String>),
}
impl Stop {
    /// The untrusted-input chokepoint for every `stop` consumer: truncate to `MAX_STOP_SEQUENCES` and
    /// refuse a string over `MAX_STOP_BYTES` or kept strings over `MAX_TOTAL_STOP_BYTES` here, so the
    /// per-token scan is bounded once at the boundary (engine-derived chat-template stops appended
    /// downstream are few and trusted).
    pub(crate) fn list(opt: &Option<Stop>) -> Result<Vec<String>, RefusedRequest> {
        let mut v = match opt {
            Some(Stop::One(s)) => vec![s.clone()],
            Some(Stop::Many(v)) => v.clone(),
            None => Vec::new(),
        };
        v.truncate(MAX_STOP_SEQUENCES);
        if let Some(long) = v.iter().find(|stop| stop.len() > MAX_STOP_BYTES) {
            return Err(RefusedRequest(format!(
                "a `stop` string is {} bytes long: the limit is {MAX_STOP_BYTES}",
                long.len()
            )));
        }
        let total: usize = v.iter().map(String::len).sum();
        if total > MAX_TOTAL_STOP_BYTES {
            return Err(RefusedRequest(format!(
                "`stop` strings total {total} bytes: the limit is {MAX_TOTAL_STOP_BYTES}"
            )));
        }
        Ok(v)
    }
}

/// OpenAI `stream_options`: opt-in streaming extras. The only supported field is `include_usage`,
/// which controls whether the stream carries a final usage/receipt chunk before `[DONE]`.
/// `#[serde(default)]` here and on its callers means `{"stream_options":{}}` parses to
/// `include_usage: None` (not treated as opted out).
#[derive(Deserialize, Default)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: Option<bool>,
}

/// poot streams a final usage/throughput-receipt chunk (`stream_receipt`) by default. This predates
/// OpenAI's `stream_options.include_usage` and nothing internal consumes it, so the default stays
/// "always emit" for backward compatibility. Only an explicit `include_usage: false` suppresses it;
/// absent `stream_options` or `include_usage: true` both emit. Card 220 Bug C.
pub(crate) fn suppress_stream_usage(stream_options: &Option<StreamOptions>) -> bool {
    stream_options.as_ref().and_then(|o| o.include_usage) == Some(false)
}

#[derive(Deserialize)]
pub(crate) struct CompletionReq {
    pub(crate) prompt: String,
    #[serde(default = "default_max")]
    pub(crate) max_tokens: usize,
    #[serde(default)]
    pub(crate) stream: bool,
    /// OpenAI streaming extras (card 220 Bug C): `{"include_usage": false}` suppresses the final
    /// usage/receipt chunk. Absent or `include_usage: true` keeps the default (always emit).
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub(crate) sampling: SamplingParams,
    #[serde(default)]
    pub(crate) stop: Option<Stop>,
    /// legacy completions `logprobs`: when set, include that many top alternatives per token.
    #[serde(default)]
    pub(crate) logprobs: Option<usize>,
    /// guided decoding (vLLM extension): constrain the completion to be exactly one of these strings.
    #[serde(default)]
    pub(crate) guided_choice: Option<Vec<String>>,
    /// guided decoding (vLLM extension): constrain the completion to match this regex.
    #[serde(default)]
    pub(crate) guided_regex: Option<String>,
    /// guided decoding (vLLM extension): constrain the completion to a GBNF context-free grammar.
    #[serde(default)]
    pub(crate) guided_grammar: Option<String>,
    /// guided decoding (vLLM extension): constrain the completion to this JSON Schema.
    #[serde(default)]
    pub(crate) guided_json: Option<serde_json::Value>,
    /// OpenAI structured outputs: `{"type":"json_schema","json_schema":{"schema":{..}}}` enforces that
    /// schema (same constraint as `guided_json`); `{"type":"text"}` is unconstrained. Lower precedence
    /// than the explicit `guided_*` fields.
    #[serde(default)]
    pub(crate) response_format: Option<serde_json::Value>,
    /// OpenAI `n`: how many independent completions to return (default 1).
    #[serde(default = "default_n")]
    pub(crate) n: usize,
    /// poot extension (spec 248, epic 129 B8): select a loaded LoRA adapter by name. Resolved to an
    /// adapter selection by [`resolve_lora_adapter`] before the job reaches the engine; an unknown name
    /// is a 400, not a fallback to the un-adapted base (and no adapter is loaded yet). Absent or `null`
    /// means no adapter.
    #[serde(default)]
    pub(crate) lora_adapter: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct ChatReq {
    pub(crate) messages: Vec<ChatMsg>,
    #[serde(default = "default_max")]
    pub(crate) max_tokens: usize,
    /// OpenAI's newer chat field, superseding the deprecated `max_tokens` for chat completions. When
    /// present it wins; see [`ChatReq::effective_max_tokens`].
    #[serde(default)]
    pub(crate) max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: bool,
    /// OpenAI streaming extras (card 220 Bug C): `{"include_usage": false}` suppresses the final
    /// usage/receipt chunk. Absent or `include_usage: true` keeps the default (always emit).
    #[serde(default)]
    pub(crate) stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub(crate) sampling: SamplingParams,
    #[serde(default)]
    pub(crate) stop: Option<Stop>,
    /// OpenAI chat `logprobs`: when true, return the chosen tokens' log-probabilities.
    #[serde(default)]
    pub(crate) logprobs: bool,
    /// OpenAI chat `top_logprobs` (0-20): how many alternatives to include per token (requires `logprobs`).
    #[serde(default)]
    pub(crate) top_logprobs: Option<usize>,
    /// guided decoding (vLLM extension): constrain the reply to be exactly one of these strings.
    #[serde(default)]
    pub(crate) guided_choice: Option<Vec<String>>,
    /// guided decoding (vLLM extension): constrain the reply to match this regex.
    #[serde(default)]
    pub(crate) guided_regex: Option<String>,
    /// guided decoding (vLLM extension): constrain the reply to a GBNF context-free grammar.
    #[serde(default)]
    pub(crate) guided_grammar: Option<String>,
    /// guided decoding (vLLM extension): constrain the reply to this JSON Schema.
    #[serde(default)]
    pub(crate) guided_json: Option<serde_json::Value>,
    /// OpenAI structured outputs: `{"type":"json_schema","json_schema":{"schema":{..}}}` enforces that
    /// schema (same constraint as `guided_json`); `{"type":"text"}` is unconstrained. Lower precedence
    /// than the explicit `guided_*` fields.
    #[serde(default)]
    pub(crate) response_format: Option<serde_json::Value>,
    /// OpenAI `n`: how many independent completions to return (default 1).
    #[serde(default = "default_n")]
    pub(crate) n: usize,
    /// OpenAI tool / function calling (spec 046): the function definitions the model may call, passed
    /// verbatim into the model's jinja `chat_template` (its tool-use branch renders the schemas and
    /// instruction). The model's tool-call output is parsed back into `message.tool_calls`.
    #[serde(default)]
    pub(crate) tools: Option<serde_json::Value>,
    /// OpenAI `tool_choice` ("auto" / "none" / "required" / a named function). For "required" or a named
    /// function on a model whose tool-call codec can be forced, generation is constrained to exactly one
    /// schema-conforming call in that codec's markup (see `resolve_forced_tools` / `apply_guided`);
    /// "auto"/"none" and a codec that cannot be forced leave the decision to the template + model.
    #[serde(default)]
    pub(crate) tool_choice: Option<serde_json::Value>,
    /// poot extension (spec 248, epic 129 B8): the same field and resolution path as
    /// [`CompletionReq::lora_adapter`], for chat completions.
    #[serde(default)]
    pub(crate) lora_adapter: Option<String>,
}

impl CompletionReq {
    /// The generation cap: `max_tokens` (else the default), bounded by [`bound_max_tokens`].
    pub(crate) fn effective_max_tokens(&self) -> usize {
        bound_max_tokens(self.max_tokens)
    }
}

impl ChatReq {
    /// The generation cap: `max_completion_tokens` (the current OpenAI field) wins over the deprecated
    /// `max_tokens` when both are sent; otherwise whichever was provided, else the default. Bounded by
    /// [`bound_max_tokens`].
    pub(crate) fn effective_max_tokens(&self) -> usize {
        bound_max_tokens(self.max_completion_tokens.unwrap_or(self.max_tokens))
    }
}

/// The model's tool calls in `text` as OpenAI `tool_calls` entries, parsed by the model's own codec;
/// empty when it has none or `text` carries no tool-call markup (the normal text reply).
pub(crate) fn parse_tool_calls(codec: &poot_text::TextCodec, text: &str) -> Vec<ToolCallOut> {
    codec
        .parse_tool_calls(text)
        .into_iter()
        .enumerate()
        .map(|(idx, call)| ToolCallOut {
            id: format!("call_{idx}"),
            kind: "function",
            function: ToolCallFn {
                name: call.name,
                arguments: call.arguments,
            },
        })
        .collect()
}

/// Build the OpenAI streaming `delta.tool_calls` array (spec 046 slice 2): each parsed call becomes
/// `{index, id, type, function:{name, arguments}}` with `index` its position in the stream's
/// tool-call list. The whole call (name + full arguments string) ships in one delta; argument
/// fragments are not streamed.
pub(crate) fn tool_call_stream_deltas(calls: &[ToolCallOut]) -> Vec<serde_json::Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            serde_json::json!({
                "index": i, "id": c.id, "type": c.kind,
                "function": { "name": c.function.name, "arguments": c.function.arguments },
            })
        })
        .collect()
}

/// OpenAI-shaped token accounting.
#[derive(Serialize)]
pub(crate) struct Usage {
    pub(crate) prompt_tokens: usize,
    pub(crate) completion_tokens: usize,
    pub(crate) total_tokens: usize,
}
impl Usage {
    pub(crate) fn new(prompt: usize, completion: usize) -> Self {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(from = "ChatMsgRaw")]
pub(crate) struct ChatMsg {
    pub(crate) role: String,
    // Optional so an assistant tool-call turn (`content: null`) and the full OpenAI message shape
    // round-trip (spec 046 multi-turn); serialized as `null` when absent, which templates treat as
    // falsy. Deserialization accepts the content-part array form (text parts joined; other part types
    // recorded in `non_text_parts` so a model without that front end can refuse them), so clients that
    // always send `content: [{type:text,...}]` do not 400.
    #[serde(default)]
    pub(crate) content: Option<String>,
    /// Non-text content-part `type` strings seen while flattening `content` (e.g. `image_url`).
    /// Empty for plain strings, null, and text-only arrays. Not re-serialized: templates and OpenAI
    /// round-trips see the flattened text form only.
    #[serde(default, skip_serializing)]
    pub(crate) non_text_parts: Vec<String>,
    /// The `url` of each image part that carries one (`image_url`, `input_image`), in order. Not
    /// re-serialized: the chat template sees the flattened text, and an image reaches the model through
    /// the image front end.
    #[serde(default, skip_serializing)]
    pub(crate) image_urls: Vec<String>,
    /// Assistant turn's prior tool calls (OpenAI history echo), passed to the template so it re-renders
    /// the `<tool_call>` markup for a multi-turn tool conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<serde_json::Value>,
    /// `role: "tool"` result messages (OpenAI `tool_call_id`) + the function name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
}

/// Intermediate form so one `content` value can flatten to text and record non-text part types without a
/// second pass over the raw request body.
#[derive(Deserialize)]
struct ChatMsgRaw {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

impl From<ChatMsgRaw> for ChatMsg {
    fn from(raw: ChatMsgRaw) -> Self {
        let (content, non_text_parts, image_urls) = flatten_chat_content(&raw.content);
        ChatMsg {
            role: raw.role,
            content,
            non_text_parts,
            image_urls,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
        }
    }
}

/// Flatten one OpenAI `content` value to `(text, non_text_part_types, image_urls)`. Text parts are
/// concatenated; every non-`text` part's `type` string is recorded, and the `url` of each image part
/// (`image_url` as a string or as `{url}`).
fn flatten_chat_content(v: &serde_json::Value) -> (Option<String>, Vec<String>, Vec<String>) {
    match v {
        serde_json::Value::String(s) => (Some(s.clone()), Vec::new(), Vec::new()),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            let mut non_text_parts = Vec::new();
            let mut image_urls = Vec::new();
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            text.push_str(t);
                        }
                    }
                    Some(other) => {
                        if is_image_part(other) {
                            let url = part.get("image_url").and_then(|image| {
                                image.as_str().or_else(|| image.get("url")?.as_str())
                            });
                            image_urls.extend(url.map(str::to_string));
                        }
                        non_text_parts.push(other.to_string());
                    }
                    None => {}
                }
            }
            (Some(text), non_text_parts, image_urls)
        }
        _ => (None, Vec::new(), Vec::new()),
    }
}

#[derive(Serialize)]
pub(crate) struct Choice {
    pub(crate) index: usize,
    pub(crate) text: String,
    pub(crate) finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<serde_json::Value>,
}
#[derive(Serialize)]
pub(crate) struct CompletionResp {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) model: String,
    pub(crate) choices: Vec<Choice>,
    pub(crate) usage: Usage,
}
#[derive(Serialize)]
pub(crate) struct ChatChoice {
    pub(crate) index: usize,
    pub(crate) message: ChatRespMsg,
    pub(crate) finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logprobs: Option<serde_json::Value>,
}
#[derive(Serialize)]
pub(crate) struct ChatRespMsg {
    pub(crate) role: String,
    // OpenAI sends `content: null` on a tool-call turn, so this is always present (Some or null).
    pub(crate) content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_calls: Option<Vec<ToolCallOut>>,
}
/// One OpenAI `tool_calls` entry (spec 046): `{id, type:"function", function:{name, arguments}}`,
/// where `arguments` is a JSON string (OpenAI serializes the args object as a string).
#[derive(Serialize)]
pub(crate) struct ToolCallOut {
    pub(crate) id: String,
    #[serde(rename = "type")]
    pub(crate) kind: &'static str,
    pub(crate) function: ToolCallFn,
}
#[derive(Serialize)]
pub(crate) struct ToolCallFn {
    pub(crate) name: String,
    pub(crate) arguments: String,
}
#[derive(Serialize)]
pub(crate) struct ChatResp {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) model: String,
    pub(crate) choices: Vec<ChatChoice>,
    pub(crate) usage: Usage,
}
