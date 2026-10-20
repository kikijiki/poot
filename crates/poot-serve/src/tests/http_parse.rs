use super::*;

// `stream_options.include_usage` is OpenAI's opt-in for the streamed usage chunk. With no `stream_options` poot still emits it (backward compat); only an explicit `include_usage: false` suppresses it.

#[test]
fn completion_req_parses_stream_options_include_usage_false() {
    let req: CompletionReq =
        serde_json::from_str(r#"{"prompt":"hi","stream_options":{"include_usage":false}}"#)
            .unwrap();
    assert_eq!(req.stream_options.unwrap().include_usage, Some(false));
}

#[test]
fn completion_req_defaults_stream_options_to_none() {
    // no `stream_options` at all -> None (not Some(default)); also proves unknown/omitted fields
    // are still tolerated (no `deny_unknown_fields`).
    let req: CompletionReq = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
    assert!(req.stream_options.is_none());
}

#[test]
fn completion_req_parses_empty_stream_options_object() {
    // `{"stream_options":{}}` -> Some(StreamOptions{include_usage: None}), distinct from absent.
    let req: CompletionReq =
        serde_json::from_str(r#"{"prompt":"hi","stream_options":{}}"#).unwrap();
    let so = req.stream_options.unwrap();
    assert_eq!(so.include_usage, None);
}

#[test]
fn chat_req_parses_stream_options_include_usage_false() {
    let req: ChatReq =
        serde_json::from_str(r#"{"messages":[],"stream_options":{"include_usage":false}}"#)
            .unwrap();
    assert_eq!(req.stream_options.unwrap().include_usage, Some(false));
}

#[test]
fn chat_req_defaults_stream_options_to_none() {
    let req: ChatReq = serde_json::from_str(r#"{"messages":[]}"#).unwrap();
    assert!(req.stream_options.is_none());
}

#[test]
fn chat_req_parses_empty_stream_options_object() {
    let req: ChatReq = serde_json::from_str(r#"{"messages":[],"stream_options":{}}"#).unwrap();
    assert_eq!(req.stream_options.unwrap().include_usage, None);
}

#[test]
fn request_parses_richer_sampling_params() {
    // spec 037: the flattened SamplingParams accept the OpenAI knobs and configure the sampler. A
    // logit_bias (string token-id keys) is enough to force the host-side readback path (non-greedy).
    let body = r#"{"prompt":"hi","max_tokens":8,"temperature":0,
            "repetition_penalty":1.3,"presence_penalty":0.5,"frequency_penalty":0.2,
            "min_p":0.05,"logit_bias":{"42":-5.0}}"#;
    let req: CompletionReq = serde_json::from_str(body).unwrap();
    assert_eq!(req.sampling.repetition_penalty, 1.3);
    assert_eq!(req.sampling.presence_penalty, 0.5);
    assert_eq!(req.sampling.logit_bias.get("42"), Some(&-5.0));
    let s = sampler_of(&req.sampling);
    assert!(
        !s.is_greedy(),
        "penalties/bias must force the readback path"
    );
}

#[test]
fn request_parses_guided_choice() {
    // spec 040: the vLLM `guided_choice` extension parses on both endpoints.
    let c: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","guided_choice":["yes","no"]}"#).unwrap();
    assert_eq!(
        c.guided_choice.as_deref(),
        Some(&["yes".to_string(), "no".to_string()][..])
    );
    let ch: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":"hi"}],"guided_choice":["a","b","c"]}"#,
    )
    .unwrap();
    assert_eq!(ch.guided_choice.map(|v| v.len()), Some(3));
    // absent -> None (unconstrained).
    let plain: CompletionReq = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
    assert!(plain.guided_choice.is_none());
}

#[test]
fn request_parses_guided_grammar() {
    // the vLLM `guided_grammar` extension parses a GBNF string on both endpoints.
    let c: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","guided_grammar":"root ::= \"yes\" | \"no\""}"#)
            .unwrap();
    assert_eq!(
        c.guided_grammar.as_deref(),
        Some("root ::= \"yes\" | \"no\"")
    );
    let ch: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":"hi"}],"guided_grammar":"root ::= [0-9]+"}"#,
    )
    .unwrap();
    assert_eq!(ch.guided_grammar.as_deref(), Some("root ::= [0-9]+"));
    let plain: CompletionReq = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
    assert!(plain.guided_grammar.is_none());
}

#[test]
fn request_parses_guided_json() {
    // spec 044: the vLLM `guided_json` extension parses a JSON Schema object on both endpoints.
    let c: CompletionReq = serde_json::from_str(
        r#"{"prompt":"x","guided_json":{"type":"object","properties":{"a":{"type":"integer"}}}}"#,
    )
    .unwrap();
    assert_eq!(c.guided_json.as_ref().unwrap()["type"], "object");
    let plain: CompletionReq = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
    assert!(plain.guided_json.is_none());
}

#[test]
fn request_parses_response_format() {
    // OpenAI structured outputs: `response_format` parses on both endpoints.
    let c: CompletionReq = serde_json::from_str(
            r#"{"prompt":"x","response_format":{"type":"json_schema","json_schema":{"name":"r","schema":{"type":"integer"}}}}"#,
        )
        .unwrap();
    assert_eq!(c.response_format.as_ref().unwrap()["type"], "json_schema");
    let ch: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}"#,
    )
    .unwrap();
    assert_eq!(ch.response_format.as_ref().unwrap()["type"], "text");
    let plain: CompletionReq = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
    assert!(plain.response_format.is_none());
}

#[test]
fn response_format_maps_to_the_inner_schema() {
    // A `json_schema` response_format yields exactly its `json_schema.schema` - the same value a client
    // could have passed via `guided_json`, so both routes constrain identically.
    let rf = serde_json::json!({
        "type": "json_schema",
        "json_schema": {"name": "person", "strict": true,
            "schema": {"type": "object", "properties": {"age": {"type": "integer"}}}},
    });
    let schema = match response_format_constraint(&rf).unwrap() {
        ResponseFormatConstraint::Schema(s) => s,
        _ => panic!("json_schema response_format must resolve to a Schema"),
    };
    assert_eq!(
        schema,
        &serde_json::json!({"type":"object","properties":{"age":{"type":"integer"}}})
    );
    // `text` (and an absent type) impose no constraint.
    assert!(matches!(
        response_format_constraint(&serde_json::json!({"type":"text"})).unwrap(),
        ResponseFormatConstraint::Unconstrained
    ));
    assert!(matches!(
        response_format_constraint(&serde_json::json!({})).unwrap(),
        ResponseFormatConstraint::Unconstrained
    ));
}

#[test]
fn response_format_json_object_resolves_and_garbage_rejected() {
    // `json_object` (arbitrary-depth JSON) resolves to the JsonObject constraint (served by the pushdown acceptor) rather than erroring.
    assert!(matches!(
        response_format_constraint(&serde_json::json!({"type":"json_object"})).unwrap(),
        ResponseFormatConstraint::JsonObject
    ));
    // a `json_schema` type missing its inner schema is still rejected.
    assert!(response_format_constraint(&serde_json::json!({"type":"json_schema"})).is_err());
    // an unknown type is still rejected rather than ignored.
    let err = response_format_constraint(&serde_json::json!({"type":"yaml"})).unwrap_err();
    assert!(err.to_string().contains("unknown response_format type"));
}

#[test]
fn request_parses_tools() {
    // spec 046 FR-001: the OpenAI `tools` array parses on the chat endpoint.
    let ch: ChatReq = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],"tool_choice":"auto"}"#,
        )
        .unwrap();
    assert_eq!(
        ch.tools.as_ref().unwrap()[0]["function"]["name"],
        "get_weather"
    );
    // absent -> None (a plain chat request, no tool branch).
    let plain: ChatReq =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert!(plain.tools.is_none());
}

#[test]
fn request_parses_content_array() {
    // card 032/054: OpenAI clients increasingly send `content` as a PART ARRAY (the default for image
    // inputs, and for some text-only SDKs). The chat endpoint accepts it and flattens text parts; a
    // plain string and `null` (tool-call turn) still work.
    let arr: ChatReq = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}]}]}"#,
        )
        .expect("content array parses");
    assert_eq!(arr.messages[0].content.as_deref(), Some("Hello world"));
    // an image_url part parses and is recorded (text parts still flatten); a text-only model refuses it
    // at the handler (`text_only_model_refuses_an_image_audio_video_or_unknown_part_with_a_typed_error`).
    let img: ChatReq = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}"#,
        )
        .expect("image part parses");
    assert_eq!(img.messages[0].content.as_deref(), Some("describe"));
    assert_eq!(
        img.messages[0].non_text_parts,
        vec!["image_url".to_string()]
    );
    // plain string and absent/null still work.
    let s: ChatReq =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert_eq!(s.messages[0].content.as_deref(), Some("hi"));
    let n: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#,
    )
    .unwrap();
    assert!(n.messages[0].content.is_none());
}

/// A codec-parsed call becomes an OpenAI `tool_calls` entry: ordered ids, the `function` kind, the
/// arguments kept as a JSON string. The markup formats themselves are pinned in poot-text.
#[test]
fn parse_tool_calls_shapes_the_models_calls_as_openai_entries() {
    let codec = chatml_codec();
    let calls = parse_tool_calls(
        &codec,
        "<tool_call>\n{\"name\": \"a\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>",
    );
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].kind, "function");
    assert_eq!(calls[0].id, "call_0");
    assert_eq!(calls[1].id, "call_1");
    assert_eq!(calls[1].function.name, "b");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
    assert!(parse_tool_calls(&codec, "The weather in Paris is sunny.").is_empty());
}

#[test]
fn tool_call_stream_deltas_carry_index_and_function() {
    // spec 046 slice 2 (FR-005): the streaming `delta.tool_calls` shape - each call gets its stream
    // `index`, the OpenAI `{id, type, function:{name, arguments}}` fields, and the arguments stay a
    // JSON string.
    let calls = parse_tool_calls(
        &chatml_codec(),
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><tool_call>\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>",
    );
    let deltas = tool_call_stream_deltas(&calls);
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["index"], 0);
    assert_eq!(deltas[0]["type"], "function");
    assert_eq!(deltas[0]["id"], "call_0");
    assert_eq!(deltas[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(deltas[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(deltas[1]["index"], 1);
    assert_eq!(deltas[1]["function"]["name"], "now");
    // no calls -> empty delta list (the normal content-streaming path).
    assert!(tool_call_stream_deltas(&[]).is_empty());
}

#[test]
fn request_parses_guided_regex() {
    // spec 043: the vLLM `guided_regex` extension parses on both endpoints.
    let c: CompletionReq =
        serde_json::from_str(r#"{"prompt":"x","guided_regex":"[0-9]+"}"#).unwrap();
    assert_eq!(c.guided_regex.as_deref(), Some("[0-9]+"));
    let ch: ChatReq = serde_json::from_str(
        r#"{"messages":[{"role":"user","content":"hi"}],"guided_regex":"\\d{3}-\\d{4}"}"#,
    )
    .unwrap();
    assert_eq!(ch.guided_regex.as_deref(), Some(r"\d{3}-\d{4}"));
}

#[test]
fn request_parses_n_choices() {
    // OpenAI `n` parses and defaults to 1.
    let c: CompletionReq = serde_json::from_str(r#"{"prompt":"x","n":3}"#).unwrap();
    assert_eq!(c.n, 3);
    let plain: CompletionReq = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
    assert_eq!(plain.n, 1, "n defaults to 1");
    let ch: ChatReq =
        serde_json::from_str(r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#).unwrap();
    assert_eq!(ch.n, 2);
}

#[test]
fn request_defaults_stay_greedy() {
    // With no sampling fields the request defaults to a plain greedy sampler (no regression).
    let req: CompletionReq = serde_json::from_str(r#"{"prompt":"hi"}"#).unwrap();
    assert_eq!(req.sampling.repetition_penalty, 1.0);
    assert!(sampler_of(&req.sampling).is_greedy());
}

/// An oversized `Content-Length` must be rejected by `read_request` before the body buffer is allocated.
/// The client sends only headers, so if the cap check moved after the read, this test would hang waiting
/// for bytes (caught by nextest's per-test timeout) rather than fail fast.
#[test]
fn read_request_rejects_oversized_content_length_without_reading_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();

    let too_big = BODY_CAP + 1;
    write!(
        client,
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {too_big}\r\n\r\n"
    )
    .unwrap();
    client.flush().unwrap();
    // no body bytes sent: read_request must return before trying to read any of them.

    let result = read_request(&mut server, BODY_CAP);
    assert!(
        matches!(result, Err(ReadFailure::Refused(Refusal::BodyTooLarge))),
        "an oversized Content-Length must be rejected, not parsed into a request"
    );
}

/// Loopback check that the `read_request` extraction (card 224) kept behavior for a normal within-bound request: method/path/body/authorization parse as before.
#[test]
fn read_request_parses_a_normal_small_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).unwrap();
    let (mut server, _) = listener.accept().unwrap();

    let body = r#"{"prompt":"hi"}"#;
    write!(
            client,
            "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\nAuthorization: Bearer secret\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
    client.flush().unwrap();

    let parsed = read_request(&mut server, BODY_CAP).expect("a normal-sized body must parse");
    assert_eq!(parsed.method, "POST");
    assert_eq!(parsed.path, "/v1/completions");
    assert_eq!(parsed.authorization.as_deref(), Some("Bearer secret"));
    assert_eq!(parsed.body, body);
}

/// card 224 follow-on: the socket-independent parser accepts a normal request over in-memory bytes,
/// extracting method/path/auth/body identically to the socket path.
#[test]
fn read_request_from_parses_a_normal_request() {
    let body = r#"{"prompt":"hi"}"#;
    let raw = format!(
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\nAuthorization: Bearer secret\r\n\r\n{}",
        body.len(),
        body
    );
    let mut cur = std::io::Cursor::new(raw.into_bytes());
    let parsed = read_request_from(&mut cur, BODY_CAP).expect("normal request must parse");
    assert_eq!(parsed.method, "POST");
    assert_eq!(parsed.path, "/v1/completions");
    assert_eq!(parsed.authorization.as_deref(), Some("Bearer secret"));
    assert_eq!(parsed.body, body);
}

/// A body that ends before its declared `Content-Length` is a truncated request, refused rather than
/// handed to the JSON parser as if it were whole. The body is read as it arrives, not allocated at the
/// declared size, so a client that declares a large body and sends a few bytes costs only those bytes.
#[test]
fn read_request_from_refuses_a_body_shorter_than_its_declared_length() {
    let sent = r#"{"prompt":"hi"}"#; // far less than the declared Content-Length below
    let raw = format!("POST /v1/completions HTTP/1.1\r\nContent-Length: 1000000\r\n\r\n{sent}");
    let mut cur = std::io::Cursor::new(raw.into_bytes());
    assert!(matches!(
        read_request_from(&mut cur, BODY_CAP),
        Err(ReadFailure::Refused(Refusal::TruncatedBody))
    ));
}

/// card 224 follow-on: a single header line longer than `MAX_HEADER_BYTES` (no newline) must be rejected as HeadersTooLarge without the reader allocating past the cap (closing the unbounded-`read_line` DoS on the request-head side).
#[test]
fn read_request_from_rejects_an_oversized_header_line() {
    let mut raw = b"GET / HTTP/1.1\r\nX-Huge: ".to_vec();
    raw.extend(std::iter::repeat_n(b'A', MAX_HEADER_BYTES + 1)); // no CRLF: an unterminated giant line
    let mut cur = std::io::Cursor::new(raw);
    assert!(
        matches!(
            read_request_from(&mut cur, BODY_CAP),
            Err(ReadFailure::Refused(Refusal::HeadersTooLarge))
        ),
        "an over-long header line must be rejected as HeadersTooLarge"
    );
}

/// card 224 follow-on: many small headers whose CUMULATIVE size exceeds `MAX_HEADER_BYTES` are also
/// rejected (the budget is shared across the whole header block, not per-line).
#[test]
fn read_request_from_rejects_oversized_cumulative_headers() {
    let mut raw = b"GET / HTTP/1.1\r\n".to_vec();
    // Each "X: y\r\n" is small; enough of them together blow the budget before the blank line.
    let one = b"X-Pad-Header-Name: paddingpaddingpadding\r\n";
    while raw.len() <= MAX_HEADER_BYTES + one.len() {
        raw.extend_from_slice(one);
    }
    raw.extend_from_slice(b"\r\n"); // terminator (never reached under budget)
    let mut cur = std::io::Cursor::new(raw);
    assert!(
        matches!(
            read_request_from(&mut cur, BODY_CAP),
            Err(ReadFailure::Refused(Refusal::HeadersTooLarge))
        ),
        "cumulative headers over the budget must be rejected"
    );
}

#[test]
fn reason_phrase_covers_the_status_codes_this_server_emits() {
    for (status, want) in [
        (200u16, "OK"),
        (400, "Bad Request"),
        (404, "Not Found"),
        (405, "Method Not Allowed"),
        (413, "Payload Too Large"),
        (429, "Too Many Requests"),
        (500, "Internal Server Error"),
        (503, "Service Unavailable"),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        respond_typed(&mut server, status, "text/plain", "").unwrap();
        drop(server);
        let mut resp = String::new();
        client.read_to_string(&mut resp).unwrap();
        let line = resp.lines().next().unwrap_or("");
        assert!(
            line.starts_with(&format!("HTTP/1.1 {status} {want}")),
            "status {status}: got {line:?}"
        );
    }
}
