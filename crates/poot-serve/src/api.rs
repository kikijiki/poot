//! OpenAI-compatible API types (request/response structs, serde) and request-shaping helpers.

use anyhow::{Context, Result, bail};
use poot_llm::{LoraAdapterLease, Modality, Runner, Sampler, TokenLogprob};
use serde::{Deserialize, Serialize};

use crate::types::GenerationError;

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
    #[serde(default)]
    pub(crate) seed: u64,
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

/// Build a sampler from the request's optional fields (default greedy). A positive temperature
/// enables the temperature/top-k/top-p/min-p draw; penalties and logit_bias apply on either path.
pub(crate) fn sampler_of(p: &SamplingParams) -> Sampler {
    let base = if p.temperature <= 0.0 {
        Sampler::greedy()
    } else {
        Sampler::new(p.temperature, p.top_k, p.top_p, p.seed)
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
/// [`ChatReq::lora_adapter`]) and acquire its request-owned lease at the HTTP layer, before the `Job`
/// reaches the engine. Resolution and acquisition are atomic under the Runner's pool lock, so an
/// unload cannot reuse the selected slot while the request waits for admission. `None` (absent or
/// `null`) is an inert no-adapter lease with no pool lookup, so such a request behaves the same
/// whether or not the server was started with `--lora-adapter` flags. A named adapter that is not
/// registered is a hard error (as in `poot_load::lora::LoraAdapterConfig::unsupported_targets`): a
/// typo must not silently serve the un-adapted model.
pub(crate) fn resolve_lora_adapter(
    runner: &Runner,
    name: &Option<String>,
) -> Result<LoraAdapterLease> {
    runner
        .acquire_lora_adapter(name.as_deref())
        .map_err(anyhow::Error::from)
}

/// Build the OpenAI-style `logprobs` object from the per-token records, decoding each token id to
/// its text piece (and UTF-8 bytes). Uses the chat-completions `content` array shape for both
/// endpoints. Records are truncated to `limit` so they align with the returned (possibly
/// stop-trimmed) text.
pub(crate) fn logprobs_json(
    runner: &Runner,
    records: &[TokenLogprob],
    limit: usize,
) -> serde_json::Value {
    let entry = |id: u32, logprob: f32| -> serde_json::Value {
        let piece = runner.decode(&[id]).unwrap_or_default();
        serde_json::json!({ "token": piece, "logprob": logprob, "bytes": piece.as_bytes() })
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

/// A content part the loaded model cannot take: a modality it has no front end for (see
/// [`Runner::accepts_modality`]) or a part type this server does not know. An explicit refusal, not a
/// silent ignore: the part would otherwise be dropped and the model would answer a different request than
/// the one sent.
#[derive(Debug)]
pub(crate) struct UnsupportedCapabilityError {
    /// The modality name (`image`, `video`, `audio`), or `content` for an unknown part type.
    pub(crate) modality: &'static str,
    pub(crate) part_type: String,
}

impl std::fmt::Display for UnsupportedCapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported capability: {} input (content part {:?}) is not supported by the loaded model",
            self.modality, self.part_type
        )
    }
}

impl std::error::Error for UnsupportedCapabilityError {}

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

/// Refuse a chat request that carries a content part the loaded model cannot take, before any generation
/// work. The decision is the model's capability ([`Runner::accepts_modality`]), never its family; a part
/// type nothing can take is refused for every model.
pub(crate) fn reject_unsupported_modality(runner: &Runner, req: &ChatReq) -> anyhow::Result<()> {
    for part_type in req.messages.iter().flat_map(|msg| &msg.non_text_parts) {
        let modality = part_modality(part_type);
        if modality.is_some_and(|modality| runner.accepts_modality(modality)) {
            continue;
        }
        return Err(anyhow::Error::new(UnsupportedCapabilityError {
            modality: modality.map_or("content", Modality::name),
            part_type: part_type.clone(),
        }));
    }
    Ok(())
}

/// Classify a `handle_completion`/`handle_chat` error into the HTTP status + OpenAI error `type`
/// (card 224): a [`GuidedDecodeError`] or [`LoraAdapterError`] is bad client input (400
/// `invalid_request_error`), as in the streaming routes; an [`UnsupportedCapabilityError`] is 400
/// `unsupported_capability`; a valid request with no engine loaded is a 503 `server_error`; anything
/// else (encode/decode/engine failures) is a server fault (500 `server_error`).
pub(crate) fn classify_handler_error(e: &anyhow::Error) -> (u16, &'static str) {
    if e.downcast_ref::<GuidedDecodeError>().is_some()
        || e.downcast_ref::<LoraAdapterError>().is_some()
    {
        (400, "invalid_request_error")
    } else if e.downcast_ref::<UnsupportedCapabilityError>().is_some() {
        (400, "unsupported_capability")
    } else if let Some(refusal @ GenerationError::NoEngine) = e.downcast_ref::<GenerationError>() {
        (refusal.stream_status(), "server_error")
    } else {
        (500, "server_error")
    }
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
    runner: &Runner,
    guided_choice: &Option<Vec<String>>,
    guided_regex: &Option<String>,
    guided_grammar: &Option<String>,
    guided_json: &Option<serde_json::Value>,
    response_format: &Option<serde_json::Value>,
    tools: &Option<serde_json::Value>,
    tool_choice: &Option<serde_json::Value>,
) -> Result<Sampler> {
    // OpenAI `tool_choice` forcing (highest precedence): when the client requires a specific tool (or
    // any tool), constrain the output to exactly one schema-conforming ChatML tool call. Only models
    // that emit ChatML `<tool_call>` markup are constrained; other tool-call formats fall through to
    // the model + template.
    if let Some(forced) = resolve_forced_tools(tools, tool_choice)?
        && matches!(
            runner.chat_format(),
            poot_models::chat::ChatFormat::ChatML | poot_models::chat::ChatFormat::Granite
        )
    {
        return Ok(sampler.with_constraint(runner.build_tool_call_constraint(&forced)?));
    }
    if let Some(choices) = guided_choice.as_ref().filter(|c| !c.is_empty()) {
        return Ok(sampler.with_constraint(runner.build_choice_constraint(choices)?));
    }
    if let Some(pattern) = guided_regex.as_ref().filter(|p| !p.is_empty()) {
        return Ok(sampler.with_constraint(runner.build_regex_constraint(pattern)?));
    }
    if let Some(gbnf) = guided_grammar.as_ref().filter(|g| !g.is_empty()) {
        return Ok(sampler.with_constraint(runner.build_grammar_constraint(gbnf)?));
    }
    if let Some(schema) = guided_json {
        return Ok(sampler.with_constraint(runner.build_json_constraint(schema)?));
    }
    // OpenAI structured outputs (the standard field clients send, vs poot's `guided_json` extension).
    if let Some(rf) = response_format {
        match response_format_constraint(rf)? {
            ResponseFormatConstraint::Schema(schema) => {
                return Ok(sampler.with_constraint(runner.build_json_constraint(schema)?));
            }
            ResponseFormatConstraint::JsonObject => {
                return Ok(sampler.with_constraint(runner.build_json_object_constraint()?));
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
/// scan per active slot (batch loop `hit_stop`), and `StreamStops`/`apply_stop` iterate every stop per
/// streamed piece, so an unbounded `stop` array is a per-token CPU DoS: a ~64MB body of short strings
/// pins a worker doing millions of `contains` per token for up to `MAX_NEW_TOKENS_CAP` tokens. OpenAI
/// caps `stop` at 4; 16 leaves headroom. Truncated, not rejected, like `clamp_n`/`clamp_logprobs`.
pub(crate) const MAX_STOP_SEQUENCES: usize = 16;

/// OpenAI's `stop` is a string or an array of strings; accept either.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Stop {
    One(String),
    Many(Vec<String>),
}
impl Stop {
    /// The untrusted-input chokepoint for every `stop` consumer: truncate to `MAX_STOP_SEQUENCES` here
    /// so the count is bounded once at the boundary, before any per-token scan (engine-derived
    /// chat-template stops appended downstream are few and trusted).
    pub(crate) fn list(opt: &Option<Stop>) -> Vec<String> {
        let mut v = match opt {
            Some(Stop::One(s)) => vec![s.clone()],
            Some(Stop::Many(v)) => v.clone(),
            None => Vec::new(),
        };
        v.truncate(MAX_STOP_SEQUENCES);
        v
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
    /// poot extension (spec 248, epic 129 B8): select a LoRA adapter registered at startup via
    /// `--lora-adapter name=path` (see `main.rs`), by name. Resolved to a
    /// `poot_load::lora::LoraAdapterPool` index by [`resolve_lora_adapter`] before the job reaches the
    /// engine; an unknown name is a 400, not a fallback to the un-adapted base. Absent or `null` means
    /// no adapter.
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
    /// function on a ChatML tool-format model, generation is constrained to exactly one schema-conforming
    /// `<tool_call>` block (see `resolve_forced_tools` / `apply_guided`); "auto"/"none" and non-ChatML
    /// formats leave the decision to the template + model.
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

/// Convert one model-emitted call object into an OpenAI `tool_calls` entry. The object must carry a
/// string `name`; args come from `arguments` (ChatML/Mistral) or `parameters` (Llama-3.1),
/// re-serialized as a JSON string (OpenAI's shape), defaulting to `{}`. Returns `None` (skipped, not
/// fatal) when there is no name: a garbled call must not 500 the request.
pub(crate) fn tool_call_from_value(v: &serde_json::Value, idx: usize) -> Option<ToolCallOut> {
    let name = v.get("name").and_then(|n| n.as_str())?;
    let arguments = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .map(|a| a.to_string())
        .unwrap_or_else(|| "{}".to_string());
    Some(ToolCallOut {
        id: format!("call_{idx}"),
        kind: "function",
        function: ToolCallFn {
            name: name.to_string(),
            arguments,
        },
    })
}

/// Parse the model's tool-call markup (spec 046) into OpenAI `tool_calls`, dispatching on the marker
/// present:
/// - ChatML/Hermes (qwen2.5, hermes, granite): `<tool_call>\n{json}\n</tool_call>` blocks.
/// - Mistral: `[TOOL_CALLS]` followed by a JSON array of call objects.
/// - Llama-3.1: `<|python_tag|>` followed by one or more `;`-separated `{name, parameters}` objects
///   (terminated by `<|eom_id|>` / `<|eot_id|>` when present).
///
/// Returns empty when there is no tool-call markup (the normal text reply). Malformed bodies are
/// skipped, never fatal.
pub(crate) fn parse_tool_calls(text: &str) -> Vec<ToolCallOut> {
    if text.contains("<tool_call>") {
        return parse_chatml_tool_calls(text);
    }
    if let Some(after) = text.split("[TOOL_CALLS]").nth(1) {
        return parse_mistral_tool_calls(after);
    }
    if let Some(after) = text.split("<|python_tag|>").nth(1) {
        return parse_llama_tool_calls(after);
    }
    Vec::new()
}

/// ChatML/Hermes `<tool_call>{json}</tool_call>` blocks.
pub(crate) fn parse_chatml_tool_calls(text: &str) -> Vec<ToolCallOut> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(open) = rest.find("<tool_call>") {
        let after = &rest[open + "<tool_call>".len()..];
        let Some(close) = after.find("</tool_call>") else {
            break;
        };
        let body = after[..close].trim();
        rest = &after[close + "</tool_call>".len()..];
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(body)
            && let Some(call) = tool_call_from_value(&v, calls.len())
        {
            calls.push(call);
        }
    }
    calls
}

/// Mistral `[TOOL_CALLS]` form: the text after the marker is a JSON array of `{name, arguments}` objects.
pub(crate) fn parse_mistral_tool_calls(after_marker: &str) -> Vec<ToolCallOut> {
    let mut calls = Vec::new();
    // The array may be followed by other tokens (e.g. `</s>`); parse the first JSON value.
    let trimmed = after_marker.trim_start();
    let mut de = serde_json::Deserializer::from_str(trimmed).into_iter::<serde_json::Value>();
    if let Some(Ok(serde_json::Value::Array(items))) = de.next() {
        for item in &items {
            if let Some(call) = tool_call_from_value(item, calls.len()) {
                calls.push(call);
            }
        }
    }
    calls
}

/// Llama-3.1 `<|python_tag|>` form: one or more `;`-separated `{name, parameters}` objects,
/// terminated by an end marker when present.
pub(crate) fn parse_llama_tool_calls(after_marker: &str) -> Vec<ToolCallOut> {
    let mut calls = Vec::new();
    // Strip the Llama turn-end markers.
    let body = after_marker
        .split("<|eom_id|>")
        .next()
        .unwrap_or(after_marker)
        .split("<|eot_id|>")
        .next()
        .unwrap_or(after_marker);
    // Parse consecutive JSON objects, tolerating `;`/whitespace separators between them. A naive
    // `body.split(';')` corrupts a call whose arguments contain a semicolon (e.g. `{"cmd":"a;b"}`),
    // both halves fail to parse and the call is silently dropped. Instead parse one JSON value at a time
    // with the streaming deserializer and skip separators by byte offset, as the Mistral parser does.
    let mut rest = body.trim_start();
    while !rest.is_empty() {
        let (val, consumed) = {
            let mut stream =
                serde_json::Deserializer::from_str(rest).into_iter::<serde_json::Value>();
            match stream.next() {
                Some(Ok(v)) => (v, stream.byte_offset()),
                _ => break, // no further parseable JSON value: stop (skip any trailing junk)
            }
        };
        if let Some(call) = tool_call_from_value(&val, calls.len()) {
            calls.push(call);
        }
        rest = rest[consumed..].trim_start_matches(|c: char| c == ';' || c.is_whitespace());
    }
    calls
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

/// Truncate `text` at the earliest occurrence of any stop string; returns whether one matched.
pub(crate) fn apply_stop(text: &mut String, stops: &[String]) -> bool {
    let cut = stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min();
    if let Some(i) = cut {
        text.truncate(i);
        true
    } else {
        false
    }
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
        let (content, non_text_parts) = flatten_chat_content(&raw.content);
        ChatMsg {
            role: raw.role,
            content,
            non_text_parts,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
        }
    }
}

/// Flatten one OpenAI `content` value to `(text, non_text_part_types)`. Text parts are concatenated; every
/// non-`text` part's `type` string is recorded.
fn flatten_chat_content(v: &serde_json::Value) -> (Option<String>, Vec<String>) {
    match v {
        serde_json::Value::String(s) => (Some(s.clone()), Vec::new()),
        serde_json::Value::Array(parts) => {
            let mut text = String::new();
            let mut non_text_parts = Vec::new();
            for part in parts {
                match part.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            text.push_str(t);
                        }
                    }
                    Some(other) => non_text_parts.push(other.to_string()),
                    None => {}
                }
            }
            (Some(text), non_text_parts)
        }
        _ => (None, Vec::new()),
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
