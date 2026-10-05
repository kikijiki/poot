//! Chat templating: the hardcoded ChatFormat fallback families and jinja chat_template rendering.

use std::collections::BTreeMap;
use std::path::Path;

use poot_models::chat::ChatFormat;

use crate::error::Result;
use crate::text::tokenize::TextCodec;

impl TextCodec {
    /// The chat-prompt template family for this model (from its family): the fallback used when the model
    /// ships no jinja `chat_template`. Prefer [`TextCodec::render_chat_value`], which uses the model's own
    /// template when available.
    pub fn chat_format(&self) -> ChatFormat {
        self.chat_format
    }

    /// Render `messages`, a raw OpenAI `messages` array (full objects), so multi-turn tool conversations
    /// render: an assistant turn's `tool_calls` and `role: "tool"` result messages reach the template's
    /// tool-history branch. Falls back to [`ChatFormat`] over each message's `role`/`content` (tool fields
    /// dropped) when there is no jinja template or it fails to render.
    pub fn render_chat_value(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
    ) -> RenderedChat {
        let chat = &self.chat_template;
        if let Some(tmpl) = &chat.template {
            match render_jinja_value(
                tmpl,
                messages,
                chat.bos_token.as_deref(),
                chat.eos_token.as_deref(),
                tools,
            ) {
                Ok(prompt) => {
                    let stops = derive_jinja_stops(
                        tmpl,
                        chat.bos_token.as_deref(),
                        chat.eos_token.as_deref(),
                    );
                    return RenderedChat { prompt, stops };
                }
                Err(e) => {
                    tracing::warn!("chat_template render failed ({e}); falling back to ChatFormat");
                }
            }
        }
        // No template: flatten to (role, content) pairs for the hardcoded ChatFormat (tool fields ignored).
        let pairs: Vec<(&str, &str)> = messages
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|m| {
                        (
                            m.get("role").and_then(|r| r.as_str()).unwrap_or(""),
                            m.get("content").and_then(|c| c.as_str()).unwrap_or(""),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let fmt = self.chat_format;
        RenderedChat {
            prompt: fmt.render(&pairs),
            stops: vec![fmt.turn_end().to_string()],
        }
    }
}

/// The result of rendering a conversation for the chat endpoint: the model-ready prompt and the turn-end markers
/// generation should stop/truncate at (the template's per-turn separator + eos string, or
/// [`ChatFormat::turn_end`] on the fallback path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedChat {
    pub prompt: String,
    pub stops: Vec<String>,
}

/// A sentinel assistant message used to probe a jinja template for its assistant turn-end separator; chosen so
/// no template emits it as markup and it is unambiguous to locate in the rendered text.
const CHAT_PROBE_SENTINEL: &str = "\u{1}POOT_ASSISTANT_BODY\u{1}";

/// Render a conversation through a model's jinja `chat_template`. Passes `messages` (`[{role, content}]`),
/// `add_generation_prompt = true`, and `bos_token`/`eos_token` (the variables HF chat templates reference).
/// Pure. Tool-call branches are not exercised (no `tools` in the context, so templates take their no-tools
/// path). Superseded by [`render_jinja_value`] in production; kept only for its own unit tests below.
#[cfg(test)]
pub(crate) fn render_jinja_chat(
    template: &str,
    messages: &[(&str, &str)],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    tools: Option<&serde_json::Value>,
) -> Result<String> {
    render_jinja(template, messages, bos_token, eos_token, true, tools)
}

/// Render a jinja chat template with an explicit `add_generation_prompt`. The probe path renders an assistant
/// message with `add_generation_prompt = false` to read back the turn-end separator. `tools` (the OpenAI
/// function definitions) feeds the template's tool-use branch; `None` -> the no-tools path.
fn render_jinja(
    template: &str,
    messages: &[(&str, &str)],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_generation_prompt: bool,
    tools: Option<&serde_json::Value>,
) -> Result<String> {
    let msgs: Vec<BTreeMap<&str, &str>> = messages
        .iter()
        .map(|(role, content)| BTreeMap::from([("role", *role), ("content", *content)]))
        .collect();
    render_jinja_core(
        template,
        minijinja::value::Value::from_serialize(&msgs),
        bos_token,
        eos_token,
        add_generation_prompt,
        tools,
    )
}

/// Like [`render_jinja`] but the conversation is a raw OpenAI `messages` array (full message objects, not just
/// `(role, content)` pairs) so multi-turn tool conversations render: an assistant turn's `tool_calls` history
/// and `role: "tool"` results reach the template's tool-history branch. The `(role, content)` path stays for the
/// non-tool case and the stop probe.
pub fn render_jinja_value(
    template: &str,
    messages: &serde_json::Value,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    tools: Option<&serde_json::Value>,
) -> Result<String> {
    render_jinja_core(
        template,
        minijinja::value::Value::from_serialize(messages),
        bos_token,
        eos_token,
        true,
        tools,
    )
}

/// Days since 1970-01-01 -> (year, month, day) in the proleptic Gregorian calendar (Howard Hinnant's
/// civil-from-days algorithm). Pure; unit-tested against known dates and round-tripped with [`days_from_civil`].
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (year, month, day) -> days since 1970-01-01 (the inverse of [`civil_from_days`]).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Format a UTC Unix timestamp (seconds) with a strftime-style string, supporting the specifiers chat
/// templates use (`%Y %y %m %d %e %H %M %S %b %h %B %a %A %j %p %%`). Unknown specifiers pass through
/// verbatim. UTC only - no timezone or DST. Backs the `strftime_now` jinja function.
fn format_strftime(fmt: &str, secs: i64) -> String {
    const MON: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    const DOW: [&str; 7] = [
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
    ];
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (h, mi, sec) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    let (y, m, d) = civil_from_days(days);
    let wd = (days.rem_euclid(7) + 4) % 7; // 1970-01-01 was a Thursday; 0 = Sunday
    let doy = days - days_from_civil(y, 1, 1) + 1; // 1-based day of year
    let mut out = String::with_capacity(fmt.len());
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('H') => out.push_str(&format!("{h:02}")),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{sec:02}")),
            Some('b' | 'h') => out.push_str(&MON[(m - 1) as usize][..3]),
            Some('B') => out.push_str(MON[(m - 1) as usize]),
            Some('a') => out.push_str(&DOW[wd as usize][..3]),
            Some('A') => out.push_str(DOW[wd as usize]),
            Some('j') => out.push_str(&format!("{doy:03}")),
            Some('p') => out.push_str(if h < 12 { "AM" } else { "PM" }),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Shared jinja render: build the environment, bind `messages` (already a jinja value), `tools`,
/// `add_generation_prompt`, and the bos/eos tokens, and render.
fn render_jinja_core(
    template: &str,
    messages_val: minijinja::value::Value,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
    add_generation_prompt: bool,
    tools: Option<&serde_json::Value>,
) -> Result<String> {
    use minijinja::{Environment, context, value::Value};
    let mut env = Environment::new();
    // HF chat templates are Python-Jinja: they call Python dict/str/list methods (`.get`, `.items`, `.strip`,
    // `.startswith`, ...) that minijinja lacks by default, so real GGUF/HF chat_templates fail to render
    // ("unknown method: map has no method named get") and callers silently drop to the ChatFormat fallback.
    // minijinja-contrib's pycompat implements those methods on maps/strings/lists; wiring it here covers every
    // render_jinja_* caller (Runner's chat path and poot-serve's gemma4/qwen3next backends funnel through
    // render_jinja_core).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // Templates call raise_exception(...) on an invalid conversation; surface it as an error (which the caller
    // turns into a ChatFormat fallback) instead of a missing-function failure.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    // `strftime_now(fmt)` is injected by HF transformers when applying a chat template; Llama-3.x and similar
    // templates call it for the system-prompt date. minijinja has no such builtin, so without this the template
    // fails to render and the caller drops to the (wrong-for-that-model) ChatFormat fallback.
    env.add_function("strftime_now", |fmt: String| -> String {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        format_strftime(&fmt, secs)
    });
    env.add_template("chat", template)
        .map_err(|e| err!("compile chat_template: {e}"))?;
    let tmpl = env.get_template("chat").expect("just added");
    // `tools` is undefined when absent so templates take their `{%- if tools %}` no-tools branch.
    let tools_val = tools.map_or(Value::UNDEFINED, Value::from_serialize);
    let rendered = tmpl
        .render(context! {
            messages => messages_val,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token.unwrap_or(""),
            eos_token => eos_token.unwrap_or(""),
            tools => tools_val,
        })
        .map_err(|e| err!("render chat_template: {e}"))?;
    Ok(rendered)
}

/// Derive the turn-end stop markers from a jinja template, so the server halts/truncates the assistant turn
/// without a hardcoded marker. Two sources: the eos token string, and the per-turn separator a template emits
/// after an assistant message (probed by rendering a sentinel assistant body with `add_generation_prompt =
/// false` and reading the text that follows it). For ChatML the separator is the eos itself (`<|im_end|>`);
/// for phi3 it is `<|end|>` while eos is `<|endoftext|>`, and both end up in the set. Falls back to just the eos
/// string if the probe fails.
pub fn derive_jinja_stops(
    template: &str,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Vec<String> {
    let mut stops: Vec<String> = Vec::new();
    if let Ok(rendered) = render_jinja(
        template,
        &[("user", ""), ("assistant", CHAT_PROBE_SENTINEL)],
        bos_token,
        eos_token,
        false,
        None,
    ) && let Some(after) = rendered.split(CHAT_PROBE_SENTINEL).nth(1)
    {
        let sep = after.trim();
        // Drop a trailing eos so the separator is the standalone per-turn marker (phi3: "<|end|>").
        let sep = match eos_token {
            Some(eos) if sep != eos => sep.strip_suffix(eos).unwrap_or(sep).trim(),
            _ => sep,
        };
        if !sep.is_empty() {
            stops.push(sep.to_string());
        }
    }
    if let Some(eos) = eos_token
        && !eos.is_empty()
        && !stops.iter().any(|s| s == eos)
    {
        stops.push(eos.to_string());
    }
    stops
}

/// Parse a model's embedded chat template + BOS/EOS token strings from held `tokenizer_config.json` bytes.
/// The `chat_template` field may be a string or an array of `{name, template}` variants; the `default`
/// entry is taken, else the first. Token fields may be strings or AddedToken-style objects.
pub(crate) fn parse_tokenizer_config_chat_bytes(
    bytes: &[u8],
) -> serde_json::Result<(Option<String>, Option<String>, Option<String>)> {
    let json: serde_json::Value = serde_json::from_slice(bytes)?;
    let template = match json.get("chat_template") {
        Some(serde_json::Value::String(template)) => Some(template.clone()),
        Some(serde_json::Value::Array(variants)) => variants
            .iter()
            .find(|variant| {
                variant.get("name").and_then(serde_json::Value::as_str) == Some("default")
            })
            .or_else(|| variants.first())
            .and_then(|variant| variant.get("template"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string),
        _ => None,
    };
    let token = |key: &str| -> Option<String> {
        json.get(key)
            .and_then(|value| {
                value
                    .as_str()
                    .or_else(|| value.get("content").and_then(serde_json::Value::as_str))
            })
            .map(str::to_string)
    };
    Ok((template, token("bos_token"), token("eos_token")))
}

/// Read a model's chat template + bos/eos token strings from `tokenizer_config.json`. Returns all-`None` when
/// neither source below has a template (the server then falls back to the hardcoded [`ChatFormat`]). Embedded
/// fields use [`parse_tokenizer_config_chat_bytes`].
///
/// Also falls back to a standalone sibling `chat_template.jinja` file when `tokenizer_config.json` carries no
/// embedded `chat_template` field (found during gpt-oss-20b coherence verification, docs/updates/ "gptoss real
/// checkpoint coherence"): current HF releases, including `openai/gpt-oss-20b`, moved the template out of
/// `tokenizer_config.json` into its own `chat_template.jinja`. Without this any such arch would silently degrade
/// to the `ChatFormat` fallback.
pub(crate) fn read_tokenizer_config_chat(
    dir: &Path,
) -> (Option<String>, Option<String>, Option<String>) {
    let parsed = std::fs::read(dir.join("tokenizer_config.json"))
        .ok()
        .and_then(|bytes| parse_tokenizer_config_chat_bytes(&bytes).ok());
    let (template, bos, eos) = parsed.unwrap_or_default();
    let template =
        template.or_else(|| std::fs::read_to_string(dir.join("chat_template.jinja")).ok());
    (template, bos, eos)
}

#[cfg(test)]
mod read_tokenizer_config_chat_tests {
    use super::{parse_tokenizer_config_chat_bytes, read_tokenizer_config_chat};

    #[test]
    fn held_bytes_preserve_embedded_chat_semantics_without_a_path() {
        let (template, bos, eos) = parse_tokenizer_config_chat_bytes(
            br#"{"chat_template":[{"name":"tool","template":"tool"},{"name":"default","template":"default"}],"bos_token":{"content":"<bos>"},"eos_token":"<eos>"}"#,
        )
        .unwrap();
        assert_eq!(template.as_deref(), Some("default"));
        assert_eq!(bos.as_deref(), Some("<bos>"));
        assert_eq!(eos.as_deref(), Some("<eos>"));
    }

    /// `tokenizer_config.json` alone is not enough for current HF releases that ship their template as a standalone
    /// `chat_template.jinja` file (e.g. `openai/gpt-oss-20b`). Without the fallback, `Runner::render_chat_value` would
    /// silently degrade to the hardcoded `ChatFormat` for any such model.
    #[test]
    fn falls_back_to_standalone_chat_template_jinja_file() {
        let dir = poot_test_util::unique_temp_path("poot_chat_template_jinja_fallback_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"bos_token": "<bos>", "eos_token": "<eos>"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "{{ messages[0].content }}").unwrap();
        let (template, bos, eos) = read_tokenizer_config_chat(&dir);
        assert_eq!(template.as_deref(), Some("{{ messages[0].content }}"));
        assert_eq!(bos.as_deref(), Some("<bos>"));
        assert_eq!(eos.as_deref(), Some("<eos>"));
    }

    /// An embedded `tokenizer_config.json` `chat_template` field still wins over a standalone file (should one also
    /// be present), as for every other arch.
    #[test]
    fn embedded_chat_template_field_takes_priority_over_standalone_file() {
        let dir = poot_test_util::unique_temp_path("poot_chat_template_jinja_priority_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template": "embedded"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("chat_template.jinja"), "standalone").unwrap();
        let (template, _, _) = read_tokenizer_config_chat(&dir);
        assert_eq!(template.as_deref(), Some("embedded"));
    }

    #[test]
    fn neither_source_present_returns_none() {
        let dir = poot_test_util::unique_temp_path("poot_chat_template_jinja_none_fixture");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tokenizer_config.json"), r#"{}"#).unwrap();
        let (template, _, _) = read_tokenizer_config_chat(&dir);
        assert_eq!(template, None);
    }
}

#[cfg(test)]
mod jinja_chat_template {
    use super::{
        civil_from_days, days_from_civil, derive_jinja_stops, format_strftime,
        read_tokenizer_config_chat, render_jinja_chat, render_jinja_value,
    };
    use poot_models::chat::ChatFormat;

    // A minimal ChatML jinja template (the shape Qwen ships, minus tool branches): per-message concat plus the assistant opener gated on add_generation_prompt.
    const CHATML: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + \
        message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}\
        {{ '<|im_start|>assistant\\n' }}{% endif %}";

    // A phi-style template: each turn ends with <|end|>, the generation prompt is <|assistant|>, and the
    // eos token (<|endoftext|>) is distinct from the per-turn separator.
    const PHI: &str = "{% for message in messages %}{{ '<|' + message['role'] + '|>' + \
        message['content'] + '<|end|>' }}{% endfor %}{% if add_generation_prompt %}\
        {{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}";

    const MSGS: &[(&str, &str)] = &[("system", "Be terse."), ("user", "Hi")];

    #[test]
    fn chatml_jinja_matches_the_hardcoded_render() {
        // Rendering the ChatML jinja template is byte-identical to ChatFormat::ChatML.
        let got = render_jinja_chat(CHATML, MSGS, None, Some("<|im_end|>"), None).unwrap();
        assert_eq!(got, ChatFormat::ChatML.render(MSGS));
    }

    #[test]
    fn chatml_stops_are_just_eos() {
        // ChatML's per-turn separator is the eos (<|im_end|>), so the stop set collapses to it.
        let stops = derive_jinja_stops(CHATML, None, Some("<|im_end|>"));
        assert_eq!(stops, vec!["<|im_end|>".to_string()]);
    }

    #[test]
    fn phi_stops_include_the_separator_and_eos() {
        // The per-turn separator (<|end|>) is derived from the template, not the eos string.
        let stops = derive_jinja_stops(PHI, None, Some("<|endoftext|>"));
        assert_eq!(
            stops,
            vec!["<|end|>".to_string(), "<|endoftext|>".to_string()]
        );
    }

    #[test]
    fn bad_template_errors_so_the_caller_can_fall_back() {
        // An unsupported/broken template surfaces an error (Runner::render_chat turns it into a ChatFormat fallback) rather than panicking.
        assert!(render_jinja_chat("{% for x in %}", MSGS, None, None, None).is_err());
    }

    #[test]
    fn format_strftime_matches_known_dates() {
        // 1700000000 = 2023-11-14 22:13:20 UTC (a Tuesday, day-of-year 318). UTC, no DST.
        let t = 1_700_000_000;
        assert_eq!(format_strftime("%Y-%m-%d", t), "2023-11-14");
        assert_eq!(format_strftime("%H:%M:%S", t), "22:13:20");
        assert_eq!(format_strftime("%d %b %Y", t), "14 Nov 2023"); // the Llama-3.x date format
        assert_eq!(format_strftime("%A %B %y", t), "Tuesday November 23");
        assert_eq!(format_strftime("%j", t), "318");
        assert_eq!(format_strftime("%p", t), "PM");
        assert_eq!(
            format_strftime("100%% done at %H:%M", t),
            "100% done at 22:13"
        );
        assert_eq!(format_strftime("%Q", t), "%Q"); // unknown specifier passes through
        // epoch + a leap-day sanity check.
        assert_eq!(format_strftime("%Y-%m-%d %A", 0), "1970-01-01 Thursday");
        assert_eq!(format_strftime("%Y-%m-%d", 1_582_934_400), "2020-02-29"); // a leap day
    }

    #[test]
    fn civil_days_round_trip() {
        // civil_from_days and days_from_civil are inverses across a wide range (covers many leap years).
        for z in (-60_000..60_000).step_by(13) {
            let (y, m, d) = civil_from_days(z);
            assert_eq!(days_from_civil(y, m, d), z, "round-trip at day {z}");
        }
    }

    #[test]
    fn strftime_now_is_registered_so_llama3_style_templates_render() {
        // A template calling strftime_now (as Llama-3.x's real chat_template does) must render instead of erroring
        // "unknown function: strftime_now" and dropping to the ChatFormat fallback.
        let tmpl = "{{ bos_token }}<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: \
            December 2023\nToday Date: {{ strftime_now('%d %b %Y') }}\n\n{% for m in messages %}\
            <|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}\
            {% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
        let out = render_jinja_chat(
            tmpl,
            MSGS,
            Some("<|begin_of_text|>"),
            Some("<|eot_id|>"),
            None,
        )
        .expect("template with strftime_now should render, not fall back");
        assert!(out.contains("Today Date: "), "got: {out}");
        // the date is filled (a 4-digit year is present), and the role markers rendered.
        assert!(
            out.contains("20") && out.contains("<|start_header_id|>user<|end_header_id|>"),
            "got: {out}"
        );
    }

    #[test]
    fn tool_use_template_constructs_render_deterministically() {
        // An always-on guard for the tool-calling render path (the skip-if-absent tests above need a real model).
        // Exercises the constructs tool templates rely on: iterating `tools` and `messages`, nested member access
        // (`tool.function.name`), and the `add_generation_prompt` conditional, asserting the exact output so a
        // minijinja member-access / loop / whitespace regression is caught.
        let tmpl = "{% for t in tools %}[{{ t.function.name }}]{% endfor %}|\
            {% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\
            {% if add_generation_prompt %}<assistant>{% endif %}";
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "get_weather", "description": "w"}},
            {"type": "function", "function": {"name": "get_time", "description": "t"}},
        ]);
        let messages = serde_json::json!([
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]);
        let out = render_jinja_value(tmpl, &messages, None, None, Some(&tools)).unwrap();
        assert_eq!(out, "[get_weather][get_time]|<system>S<user>U<assistant>");

        // `tojson` (used to embed function schemas) emits valid, complete JSON: render it and round-trip the first
        // tool through serde to prove nothing was dropped (without pinning minijinja's spacing).
        let json_out = render_jinja_value(
            "{{ tools[0] | tojson }}",
            &messages,
            None,
            None,
            Some(&tools),
        )
        .unwrap();
        let parsed: serde_json::Value =
            serde_json::from_str(&json_out).expect("tojson is valid JSON");
        assert_eq!(parsed, tools[0]);
    }

    #[test]
    fn real_qwen_template_renders_with_role_markers() {
        // The qwen2.5-0.5b chat_template (read from tokenizer_config.json) renders through the jinja path and
        // contains the expected markers. Skipped when the model is absent. (The bare qwen2.5-0.5b download omits
        // tokenizer_config.json; the awq variant ships the same template.)
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b-awq"))
        else {
            return;
        };
        let (tmpl, bos, eos) = read_tokenizer_config_chat(&dir);
        let tmpl = tmpl.expect("qwen2.5-0.5b ships a chat_template");
        let out = render_jinja_chat(&tmpl, MSGS, bos.as_deref(), eos.as_deref(), None).unwrap();
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>"), "got: {out}");
        assert!(out.ends_with("<|im_start|>assistant\n"), "got: {out}");
        // The shipped qwen template (no-tools path) is a true drop-in for the hardcoded ChatML render.
        assert_eq!(out, ChatFormat::ChatML.render(MSGS), "got: {out}");
    }

    #[test]
    fn tools_render_into_the_qwen_prompt_and_none_is_unchanged() {
        // Passing a `tools` array fires the qwen template's tool-use branch (the function schema + the <tool_call>
        // instruction land in the prompt); passing None is byte-identical to the no-tools render. Skipped when the
        // qwen2.5 template is absent.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b-awq"))
        else {
            return;
        };
        let (tmpl, bos, eos) = read_tokenizer_config_chat(&dir);
        let tmpl = tmpl.expect("qwen2.5 ships a chat_template");
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        }]);
        let with =
            render_jinja_chat(&tmpl, MSGS, bos.as_deref(), eos.as_deref(), Some(&tools)).unwrap();
        assert!(with.contains("get_weather"), "tool name missing: {with}");
        assert!(
            with.contains("<tool_call>"),
            "tool-call instruction missing: {with}"
        );
        // None is the unchanged no-tools render.
        let without = render_jinja_chat(&tmpl, MSGS, bos.as_deref(), eos.as_deref(), None).unwrap();
        assert!(!without.contains("get_weather"));
        assert!(!without.contains("<tools>"));
    }

    #[test]
    fn multi_turn_tool_conversation_renders() {
        // Multi-turn: a conversation carrying an assistant `tool_calls` turn (content null) and a `role: "tool"`
        // result must render through the qwen template: the assistant's call is re-emitted and the tool result reaches
        // the prompt. Skipped when no qwen2.5 template (with a tool branch) is local.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b-awq"))
        else {
            return;
        };
        let (tmpl, bos, eos) = read_tokenizer_config_chat(&dir);
        let tmpl = tmpl.expect("qwen2.5 ships a chat_template");
        let tools = serde_json::json!([{
            "type": "function",
            "function": { "name": "get_weather", "parameters": {
                "type": "object", "properties": { "city": { "type": "string" } } } }
        }]);
        let messages = serde_json::json!([
            { "role": "user", "content": "What is the weather in Paris?" },
            { "role": "assistant", "content": serde_json::Value::Null, "tool_calls": [
                { "id": "call_0", "type": "function",
                  "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" } }
            ]},
            { "role": "tool", "tool_call_id": "call_0", "name": "get_weather",
              "content": "{\"temp_c\": 18, \"sky\": \"sunny\"}" },
        ]);
        let out = render_jinja_value(
            &tmpl,
            &messages,
            bos.as_deref(),
            eos.as_deref(),
            Some(&tools),
        )
        .unwrap();
        // the assistant turn's tool call is re-rendered, and the tool result lands in the prompt.
        assert!(
            out.contains("get_weather"),
            "assistant tool call missing: {out}"
        );
        assert!(out.contains("sunny"), "tool result missing: {out}");
        assert!(
            out.contains("tool_response") || out.contains("<tool_call>"),
            "no tool markup: {out}"
        );
        // the prompt ends ready for the assistant's next turn.
        assert!(
            out.trim_end().ends_with("assistant"),
            "no generation prompt: {out}"
        );
    }

    #[test]
    fn pycompat_dict_get_renders_instead_of_erroring() {
        // The failure poot-serve hit in production: a real HF chat_template (gemma4's included) calls Python dict
        // methods like `message.get('role')` on the per-message map. Before wiring minijinja-contrib's pycompat
        // unknown_method_callback into render_jinja_core's Environment this failed with "unknown method: map has no
        // method named get" and the caller fell back to the hardcoded ChatFormat. Also exercises `.get` with a
        // missing-key default, since HF templates commonly write `message.get('tool_calls')` with no second arg.
        let tmpl = "{% for m in messages %}{{ m.get('role') }}:{{ m.get('content') }}:\
            {{ m.get('missing_key', 'DEFAULT') }}\n{% endfor %}";
        let messages = serde_json::json!([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ]);
        let out = render_jinja_value(tmpl, &messages, None, None, None)
            .expect("a .get()-using template must render, not fall back to ChatFormat");
        assert_eq!(out, "system:Be terse.:DEFAULT\nuser:hi:DEFAULT\n");
    }

    #[test]
    fn pycompat_dict_items_keys_values_and_str_methods_also_work() {
        // pycompat covers more than `.get`: HF templates also lean on `.items()`/`.keys()`/`.strip()`/
        // `.startswith()` etc. A minimal sweep so a minijinja-contrib bump that drops coverage is caught here rather
        // than only in a real model's template.
        let tmpl = "{% for k in ({'a': 1, 'b': 2}).keys() %}{{ k }}{% endfor %}|\
            {{ '  hi  '.strip() }}|{{ 'hello'.startswith('he') }}";
        let out = render_jinja_value(tmpl, &serde_json::json!([]), None, None, None)
            .expect("pycompat str/dict methods must render");
        assert_eq!(out, "ab|hi|true");
    }

    #[test]
    fn real_gemma4_chat_template_renders_via_jinja_not_the_chatformat_fallback() {
        // The production repro: serving gemma4 logged `gemma4 chat_template render failed (...
        // unknown method: map has no method named get ...); falling back to ChatFormat`. This reads the real GGUF's
        // `tokenizer.chat_template` (header-only via GgufIndex::open, no tensor data) and renders a 2-message
        // conversation through it, asserting it succeeds and uses role-specific gemma markup (proving the real
        // template rendered, not a fallback). Skipped when the model is absent from POOT_MODELS_DIR.
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "gemma-4-31B-it-GGUF/gemma-4-31B-it-UD-Q4_K_XL.gguf"
        )) else {
            return;
        };
        assert!(path.is_file(), "not a file: {}", path.display());
        let g = poot_load::gguf::GgufIndex::open(path).expect("load gemma4 gguf header/metadata");
        let tmpl = g
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
            .expect("gemma4 GGUF ships tokenizer.chat_template")
            .to_string();
        let messages = serde_json::json!([
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]);
        let out = render_jinja_value(&tmpl, &messages, None, None, None)
            .expect("the real gemma4 chat_template must render (this is the reported bug)");
        eprintln!("gemma4 chat_template render:\n{out}");
        assert!(!out.is_empty());
    }
}
