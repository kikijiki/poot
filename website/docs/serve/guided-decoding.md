---
id: guided-decoding
title: Guided decoding
sidebar_position: 6
---

# Guided decoding

The completion and chat endpoints can constrain the output so it is guaranteed well-formed. The engine compiles the constraint to a per-step token mask, and the sampler only picks an allowed token. Every one of a request's `n` choices carries the constraint.

## Request fields

Four fields, checked in this order:

- `guided_choice: ["yes", "no", "maybe"]`: the output is exactly one of the strings.
- `guided_regex: "[0-9]{4}-[0-9]{2}-[0-9]{2}"`: the output matches the regex.
- `guided_grammar: "root ::= ..."`: the output is a sentence of a context-free grammar in GBNF (the llama.cpp grammar syntax). It can be recursive. Rules use `::=`, `|`, `[...]`, `.`, `"..."`, `()` groups and `* + ? {m} {m,} {m,n}`; the entry rule is `root`.
- `guided_json: <JSON Schema>`: the output is JSON conforming to the schema.

An invalid constraint is a `400` before the engine is asked.

## Limits

Each of these is a `400`:

| Input                                                            | Limit      |
| ---------------------------------------------------------------- | ---------- |
| `guided_regex`                                                   | 4096 bytes |
| `guided_grammar`                                                 | 16 KiB     |
| `guided_json` or `response_format` schema, as JSON text          | 16 KiB     |
| A regex whose automaton needs more working memory                | 256 KiB    |
| A `guided_json` schema whose compiled regex is larger            | 256 KiB    |

The error names the regex and the limit it hit. A large automaton, such as a counted repetition over an alternation like `(a|b)*a(a|b){17}`, is refused rather than slowly built. Deeply nested arrays or many optional properties can pass the schema bound.

## The `guided_json` subset

| Construct           | Support                                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| `object`            | `properties` and `required`; listed keys are mandatory, the rest optional  |
| `array`             | `items`, `minItems`/`maxItems`                                             |
| `string`            | `pattern`, or `minLength`/`maxLength`                                      |
| `integer`           | `minimum`/`maximum`, inclusive or exclusive, one- or two-sided             |
| `number`, `boolean` | Plain types                                                                |
| `enum`, `const`     | Scalar or composite                                                        |
| `oneOf`, `anyOf`    | Alternation                                                                |
| local `$ref`        | `"#/$defs/Name"` into the same document, inlined                           |

Object keys are emitted in sorted order. Unsupported constructs (recursive or external `$ref`, `number` bounds) return an error rather than silently under-constraining.

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

The content is then guaranteed to parse and conform, for example `{"age":42,"class":"mage","name":"Bel"}`.

## `response_format`

OpenAI clients send `response_format` instead of `guided_json`. poot maps it to the same constraint:

- `{"type": "json_schema", "json_schema": {"name": "...", "schema": <JSON Schema>}}`: the output conforms to `schema`. `name` and `strict` are accepted and ignored.
- `{"type": "json_object"}`: the output is one well-formed JSON value of any shape and depth. Keys and types are not constrained; use `json_schema` for that.
- `{"type": "text"}`: unconstrained (the default).

The explicit `guided_*` fields take precedence over `response_format`.

## Token bytes

The constraint applies to the bytes each token adds to the output.

- A word-initial token of a SentencePiece model counts as its leading space plus the word. A pattern that starts with a space, such as `" the [a-z]+"`, constrains the tokens the model emits.
- The `bytes` of a logprob entry include that space.
- A whole (non-streamed) reply is decoded as one sequence, which drops the single leading space a SentencePiece model adds to the start of text.

## Forcing a tool call

On the chat endpoint, `tool_choice` can require a tool call. poot then constrains generation to a valid call whose `arguments` conform to the tool's `parameters` schema.

| `tool_choice`                                        | Effect                                     |
| ---------------------------------------------------- | ------------------------------------------ |
| `"required"`                                         | The reply must call one of the `tools`     |
| `{"type": "function", "function": {"name": "..."}}`  | The reply must call that tool              |
| `"auto"`, `"none"`, or omitted                       | The model and template decide, unconstrained |

Forcing applies to models whose tool-call codec emits the ChatML `<tool_call>` markup. The `[TOOL_CALLS]` and `<|python_tag|>` formats are parsed but left to the template when forced, and a model with no codec returns plain content. Forcing takes precedence over the `guided_*` fields.

Design notes: [Serving design](../architecture/serving-design.mdx#the-scheduling-loop).
