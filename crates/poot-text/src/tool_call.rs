//! Tool-call codecs: how a model spells a function call in its output, and the grammar that forces one.
//!
//! The model names its codec (an open [`ToolCallCodec`], or none when its format has no tool markup):
//! the server asks [`TextCodec::parse_tool_calls`] and [`TextCodec::forced_tool_constraint`] and never
//! looks at the chat format itself.

use std::sync::Arc;

use crate::error::Result;
use crate::guided::{Constraint, tool_call_regex};
use crate::tokenize::TextCodec;

/// A tool call a model emitted: the function name and its arguments as a JSON string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    /// The function the model called.
    pub name: String,
    /// The arguments object re-serialized as JSON text (OpenAI's shape); `{}` when the call carried none.
    pub arguments: String,
}

/// One model family's tool-call output format.
pub trait ToolCallCodec: Send + Sync + std::fmt::Debug {
    /// The calls in a completion's `text`, in order. Empty when the text carries no tool-call markup (a
    /// normal reply); a malformed call is skipped, never fatal.
    fn parse(&self, text: &str) -> Vec<ParsedToolCall>;

    /// A regex matching exactly one well-formed call of one of `forced` (name, parameter schema), in this
    /// format's markup, or `None` when the format cannot be forced by a grammar (the model and its
    /// template then decide).
    fn forced_pattern(&self, forced: &[(String, serde_json::Value)]) -> Result<Option<String>>;
}

/// A call object `{name, arguments | parameters}`. `None` (skipped) when there is no string `name`: a
/// garbled call must not fail the request.
fn call_from_value(v: &serde_json::Value) -> Option<ParsedToolCall> {
    let name = v.get("name").and_then(|n| n.as_str())?;
    let arguments = v
        .get("arguments")
        .or_else(|| v.get("parameters"))
        .map(|a| a.to_string())
        .unwrap_or_else(|| "{}".to_string());
    Some(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

/// ChatML/Hermes (Qwen2.5, Hermes, Granite): `<tool_call>\n{json}\n</tool_call>` blocks.
#[derive(Debug, Clone, Copy)]
pub struct HermesToolCodec;

impl ToolCallCodec for HermesToolCodec {
    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        let mut calls = Vec::new();
        let mut rest = text;
        while let Some(open) = rest.find(OPEN) {
            let after = &rest[open + OPEN.len()..];
            let Some(close) = after.find(CLOSE) else {
                break;
            };
            let body = after[..close].trim();
            rest = &after[close + CLOSE.len()..];
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(body)
                && let Some(call) = call_from_value(&v)
            {
                calls.push(call);
            }
        }
        calls
    }

    fn forced_pattern(&self, forced: &[(String, serde_json::Value)]) -> Result<Option<String>> {
        tool_call_regex(forced).map(Some)
    }
}

/// Mistral: `[TOOL_CALLS]` followed by a JSON array of `{name, arguments}` objects.
#[derive(Debug, Clone, Copy)]
pub struct MistralToolCodec;

impl ToolCallCodec for MistralToolCodec {
    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        let Some(after_marker) = text.split("[TOOL_CALLS]").nth(1) else {
            return Vec::new();
        };
        // The array may be followed by other tokens (e.g. `</s>`); parse the first JSON value.
        let trimmed = after_marker.trim_start();
        let mut de = serde_json::Deserializer::from_str(trimmed).into_iter::<serde_json::Value>();
        match de.next() {
            Some(Ok(serde_json::Value::Array(items))) => {
                items.iter().filter_map(call_from_value).collect()
            }
            _ => Vec::new(),
        }
    }

    fn forced_pattern(&self, _forced: &[(String, serde_json::Value)]) -> Result<Option<String>> {
        Ok(None)
    }
}

/// Llama 3.1: `<|python_tag|>` followed by one or more `;`-separated `{name, parameters}` objects,
/// terminated by `<|eom_id|>` / `<|eot_id|>` when present.
#[derive(Debug, Clone, Copy)]
pub struct LlamaToolCodec;

impl ToolCallCodec for LlamaToolCodec {
    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        let Some(after_marker) = text.split("<|python_tag|>").nth(1) else {
            return Vec::new();
        };
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
        // `body.split(';')` corrupts a call whose arguments contain a semicolon (e.g. `{"cmd":"a;b"}`):
        // both halves fail to parse and the call is silently dropped. Parse one JSON value at a time and
        // skip separators by byte offset instead.
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
            if let Some(call) = call_from_value(&val) {
                calls.push(call);
            }
            rest = rest[consumed..].trim_start_matches(|c: char| c == ';' || c.is_whitespace());
        }
        calls
    }

    fn forced_pattern(&self, _forced: &[(String, serde_json::Value)]) -> Result<Option<String>> {
        Ok(None)
    }
}

impl TextCodec {
    /// This codec with `codec` as the model's tool-call format (`None`: the model has no tool-call
    /// markup). The model's configuration names it (`ModelConfig::tool_codec`), so a new family needs no
    /// chat-format variant.
    pub fn with_tool_codec(mut self, codec: Option<Arc<dyn ToolCallCodec>>) -> Self {
        self.tool_codec = codec;
        self
    }

    /// The tool calls in `text` as this model spells them; empty when the model has no tool-call format
    /// or `text` carries none.
    pub fn parse_tool_calls(&self, text: &str) -> Vec<ParsedToolCall> {
        self.tool_codec
            .as_ref()
            .map(|codec| codec.parse(text))
            .unwrap_or_default()
    }

    /// The constraint forcing exactly one call of one of `forced` (the (name, parameter schema) list the
    /// request permits), or `None` when this model's tool-call format cannot be forced by a grammar.
    pub fn forced_tool_constraint(
        &self,
        forced: &[(String, serde_json::Value)],
    ) -> Result<Option<Constraint>> {
        let Some(codec) = &self.tool_codec else {
            return Ok(None);
        };
        codec
            .forced_pattern(forced)?
            .map(|pattern| self.build_regex_constraint(&pattern))
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(call: &ParsedToolCall) -> serde_json::Value {
        serde_json::from_str(&call.arguments).unwrap()
    }

    #[test]
    fn hermes_single_multiple_and_none() {
        let one = HermesToolCodec.parse(
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
        );
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].name, "get_weather");
        assert_eq!(args(&one[0])["city"], "Paris");

        let two = HermesToolCodec.parse(
            "<tool_call>\n{\"name\": \"a\", \"arguments\": {}}\n</tool_call><tool_call>\n{\"name\": \"b\", \"arguments\": {\"x\": 1}}\n</tool_call>",
        );
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].name, "b");

        assert!(
            HermesToolCodec
                .parse("The weather in Paris is sunny.")
                .is_empty()
        );
        assert!(
            HermesToolCodec
                .parse("<tool_call>\nnot json\n</tool_call>")
                .is_empty()
        );
        assert!(
            HermesToolCodec
                .parse("<tool_call>\n{\"arguments\": {}}\n</tool_call>")
                .is_empty()
        );
    }

    #[test]
    fn mistral_array_form() {
        let m = MistralToolCodec.parse(
            "[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}, {\"name\": \"now\", \"arguments\": {}}]</s>",
        );
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].name, "get_weather");
        assert_eq!(m[1].name, "now");
        assert_eq!(args(&m[0])["city"], "Paris");
        assert!(MistralToolCodec.parse("[TOOL_CALLS]not json").is_empty());
        assert!(MistralToolCodec.parse("plain answer").is_empty());
    }

    #[test]
    fn llama_python_tag_form() {
        let l = LlamaToolCodec.parse(
            "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}<|eom_id|>",
        );
        assert_eq!(l.len(), 1);
        assert_eq!(args(&l[0])["city"], "Rome");

        let two = LlamaToolCodec.parse(
            "<|python_tag|>{\"name\": \"a\", \"parameters\": {}}; {\"name\": \"b\", \"parameters\": {\"x\": 1}}",
        );
        assert_eq!(two.len(), 2);
        assert_eq!(two[1].name, "b");

        // a semicolon inside a string argument must not split the call (a naive `split(';')` drops it).
        let semi = LlamaToolCodec.parse(
            "<|python_tag|>{\"name\": \"run\", \"parameters\": {\"cmd\": \"echo a; echo b\"}}<|eom_id|>",
        );
        assert_eq!(semi.len(), 1, "a semicolon in a string arg must not split");
        assert_eq!(args(&semi[0])["cmd"], "echo a; echo b");

        let mixed = LlamaToolCodec.parse(
            "<|python_tag|>{\"name\": \"a\", \"parameters\": {\"q\": \"x;y\"}}; {\"name\": \"b\", \"parameters\": {}}",
        );
        assert_eq!(mixed.len(), 2);
        assert_eq!(args(&mixed[0])["q"], "x;y");

        assert!(LlamaToolCodec.parse("<|python_tag|>not json").is_empty());
        assert!(LlamaToolCodec.parse("just a normal answer").is_empty());
    }

    #[test]
    fn a_codec_parses_only_its_own_markup() {
        let hermes = "<tool_call>\n{\"name\":\"c\",\"arguments\":{}}\n</tool_call>";
        assert!(MistralToolCodec.parse(hermes).is_empty());
        assert!(LlamaToolCodec.parse(hermes).is_empty());
        assert_eq!(HermesToolCodec.parse(hermes).len(), 1);
    }
}
