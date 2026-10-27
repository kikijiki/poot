//! The model supplies its tool-call codec: serve parses and forces tool calls through it and never
//! looks at the chat format.

use std::sync::Arc;

use poot_text::{ChatFormat, ParsedToolCall, ToolCallCodec};

use super::*;

/// A tool-call format no built-in knows: `@call <name> <arguments>`, forced as `@call <name>`.
#[derive(Debug)]
struct AtCallCodec;

impl ToolCallCodec for AtCallCodec {
    fn parse(&self, text: &str) -> Vec<ParsedToolCall> {
        text.strip_prefix("@call ")
            .and_then(|rest| rest.split_once(' '))
            .map(|(name, arguments)| ParsedToolCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            })
            .into_iter()
            .collect()
    }

    fn forced_pattern(
        &self,
        forced: &[(String, serde_json::Value)],
    ) -> poot_text::Result<Option<String>> {
        Ok(Some(
            forced
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join("|"),
        ))
    }
}

fn fixture() -> poot_text::TextCodec {
    // Gemma names no tool codec of its own: every call below comes from the fixture's codec.
    spm_codec(ChatFormat::Gemma).with_tool_codec(Some(Arc::new(AtCallCodec)))
}

/// SC-002: a fixture model with a custom tool codec gets its calls parsed by `parse_tool_calls`, with no
/// serve change. Mutation: parse with the Hermes codec in `parse_tool_calls`; the call is not found.
#[test]
fn a_model_with_a_custom_tool_codec_has_its_calls_parsed() {
    let calls = parse_tool_calls(&fixture(), "@call get_weather {\"city\":\"Paris\"}");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    assert_eq!(calls[0].function.arguments, "{\"city\":\"Paris\"}");
    assert!(parse_tool_calls(&fixture(), "plain reply").is_empty());
}

/// SC-002: a forced tool call is constrained by the fixture codec's own grammar. The sampler is handed
/// logits that favour `a`; the forced tool is named `b`, so a constrained pick is `b`. Mutation: force
/// with the Hermes grammar in `apply_guided`; nothing starts with `<tool_call>` and the pick is not `b`.
#[test]
fn a_forced_call_is_constrained_by_the_models_own_grammar() {
    let text = fixture();
    let tools = Some(serde_json::json!([
        {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}}
    ]));
    let mut sampler = apply_guided(
        Sampler::greedy(),
        &text,
        &None,
        &None,
        &None,
        &None,
        &None,
        &tools,
        &Some(serde_json::json!("required")),
    )
    .expect("a forced call builds");
    let b = text.encode("b").expect("encode b").pop().expect("a token");
    let pick = sampler
        .pick(&[0.0, 0.0, 0.0, 9.0, 1.0, 0.0])
        .expect("a finite pick");
    assert_eq!(pick as u32, b, "the codec's grammar reached the sampler");
}

/// A family with no tool codec is not forced: the request falls through to the model and its template.
#[test]
fn a_model_with_no_tool_codec_is_not_forced() {
    let text = spm_codec(ChatFormat::Gemma);
    let tools = Some(serde_json::json!([
        {"type": "function", "function": {"name": "b", "parameters": {"type": "object"}}}
    ]));
    let mut sampler = apply_guided(
        Sampler::greedy(),
        &text,
        &None,
        &None,
        &None,
        &None,
        &None,
        &tools,
        &Some(serde_json::json!("required")),
    )
    .expect("an unforceable family is not an error");
    let pick = sampler
        .pick(&[0.0, 0.0, 0.0, 9.0, 1.0, 0.0])
        .expect("a finite pick");
    assert_eq!(pick, 3, "unconstrained greedy takes the largest logit");
    assert!(parse_tool_calls(&text, "<tool_call>\n{\"name\":\"x\"}\n</tool_call>").is_empty());
}

/// SC-001: serve never names the chat format: no `ChatFormat::` match, no `use ChatFormat::*` glob, no
/// `chat_format()` branch; the model's codec decides. Mutation: add any of those to a serve source file.
#[test]
fn serve_never_branches_on_the_chat_format() {
    let src = poot_test_util::manifest_dir().join("src");
    // Split so this file does not match itself.
    let needles = [["Chat", "Format"].concat(), ["chat_", "format"].concat()];
    let mut stack = vec![src.clone()];
    let mut hits = Vec::new();
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read the serve sources") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.starts_with(src.join("tests"))
            {
                let text = std::fs::read_to_string(&path).expect("read a source file");
                if needles.iter().any(|needle| text.contains(needle.as_str())) {
                    hits.push(path);
                }
            }
        }
    }
    assert!(hits.is_empty(), "serve names the chat format in {hits:?}");
}
