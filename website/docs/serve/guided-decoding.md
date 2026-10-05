---
id: guided-decoding
title: Guided decoding
sidebar_position: 6
---

# Guided decoding

:::caution[Being restored]
poot is being refactored. This is not available yet and is planned to return.
:::

Today a valid guided request returns `503` naming the missing engine. The server still validates the constraint first: an invalid `guided_regex`, `guided_json`, `guided_choice`, `guided_grammar` or `response_format` is a `400` either way.

The completion and chat endpoints can constrain the output so it is guaranteed well-formed. The
engine compiles the constraint to a per-step token mask and the sampler only ever picks an allowed
token. Four request fields, checked in this order (`guided_choice`, then `guided_regex`, then
`guided_grammar`, then `guided_json`):

- `guided_choice: ["yes", "no", "maybe"]`, the output is exactly one of the listed strings.
- `guided_regex: "[0-9]{4}-[0-9]{2}-[0-9]{2}"`, the output matches the regex (a byte-level anchored
  DFA).
- `guided_grammar: "root ::= ..."`, the output is a sentence of a context-free grammar written in
  GBNF (the llama.cpp grammar syntax). Because it is a pushdown engine, not a regex, the grammar can
  be recursive and balanced (nested brackets, arithmetic expressions, a small DSL). Rules use `::=`,
  `|` for alternation, `[...]` char classes, `.` for any character, `"..."` literals, `()` groups,
  and `* + ? {m} {m,} {m,n}` repetition; the entry rule is `root`.
- `guided_json: <JSON Schema>`, the output is JSON conforming to the schema.

The `guided_json` compiler supports the regex-expressible subset of JSON Schema: `object` (with
`properties` and a `required` list; listed keys are mandatory, the rest optional), `array` (`items`,
`minItems`/`maxItems`), `string` (`pattern` to a regex, or `minLength`/`maxLength`), `integer`
(`minimum`/`maximum`, inclusive or exclusive, two-sided or one-sided, constrained to the exact
range), `number`, `boolean`, `enum` and `const` (scalar OR composite; a fixed object/array is matched
structurally with sorted keys and optional whitespace), `oneOf`/`anyOf`, and local `$ref` (a
`"#/$defs/Name"` reference into the same document, resolved and inlined, the shape Pydantic and the
OpenAI SDK emit). Object keys are emitted in sorted order (still valid JSON). Unsupported constructs
(recursive or external `$ref`, `number` bounds) return an error rather than silently
under-constraining.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Make a character."}],
    "max_tokens": 64,
    "guided_json": {
      "type": "object",
      "properties": {
        "name":  {"type": "string"},
        "age":   {"type": "integer", "minimum": 1, "maximum": 120},
        "class": {"enum": ["warrior", "mage", "rogue"]}
      },
      "required": ["name", "class"]
    }
  }'
```

The response content is then guaranteed to parse and conform, e.g.
`{"age":42,"class":"mage","name":"Bel"}`.

OpenAI clients usually send the standard `response_format` field instead of poot's `guided_json`
extension; poot accepts it and maps it to the same constraint:

- `response_format: {"type": "json_schema", "json_schema": {"name": "...", "schema": <JSON Schema>}}`,
  the output conforms to the inner `schema` (the same subset `guided_json` supports). The
  `name`/`strict` keys are accepted and ignored.
- `response_format: {"type": "text"}`, unconstrained (the default).
- `response_format: {"type": "json_object"}`, the output is a single well-formed JSON value of any
  shape and any nesting depth. This uses a pushdown (stack) acceptor, not the regex/DFA engine, so it
  counts brackets and never blocks a deeply nested value. It constrains only the JSON grammar, not
  the keys or types; pass a concrete `json_schema` when you need those constrained too.

When both are present `guided_json` wins. The explicit `guided_choice`/`guided_regex`/
`guided_grammar`/`guided_json` fields all take precedence over `response_format`.

## Forcing a tool call

On the chat endpoint, OpenAI `tool_choice` can require the model to call a tool, and poot then
constrains generation to a valid, schema-conforming call (it reuses the guided-decoding engine, so
the reply is guaranteed to parse):

- `tool_choice: "required"`, the reply must be a call to one of the `tools`.
- `tool_choice: {"type": "function", "function": {"name": "get_weather"}}`, the reply must call that
  specific tool, with `arguments` conforming to its `parameters` schema.
- `tool_choice: "auto"` / `"none"` (or omitted), the model and template decide, unconstrained.

Forcing applies to models that emit the ChatML `<tool_call>` markup (Qwen2.5, Hermes, Granite);
other tool-call formats are left to the template as before. This takes precedence over the
`guided_*` fields.

Design notes: [Architecture: Serving design](../architecture/serving-design.mdx#the-design-the-scheduling-loop-builds-on).
