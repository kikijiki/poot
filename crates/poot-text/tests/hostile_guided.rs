//! POOT-1221 (R2-C5-001): guided-decoding compile on hostile client input. A `guided_regex`,
//! `guided_json`, `guided_choice`, `guided_grammar` or forced tool call is request text, so building
//! its constraint must stay bounded in time and memory and must answer with a value or a typed error,
//! never a panic or a stack overflow. Every row drives the production builders on `TextCodec` over a
//! small SentencePiece vocabulary. Model-free.
//!
//! The size properties are the ones POOT-1200 put in place: a pattern whose automaton would pass the
//! size limits is refused rather than built (the DFA, determinization and NFA caps), and a refusal
//! arrives quickly rather than after the work it was meant to avoid.
//!
//! Mutations (applied by hand, shown red, restored; recorded in the POOT-1221 report):
//! HG-1 `MAX_SCHEMA_REGEX_BYTES` raised to 2 GiB (the nested-schema row goes red on its memory bound);
//! HG-2 `DETERMINIZE_SIZE_LIMIT` raised back to 1 MiB (the random-pattern row goes red on its time
//! budget: one pattern takes 7 s to refuse).

use std::sync::Arc;
use std::time::{Duration, Instant};

use poot_text::{ChatFormat, Constraint, GgufVocab, HermesToolCodec, TextCodec};

/// Longest a build or a refusal may take. The bounded builds finish in milliseconds; the margin
/// absorbs a loaded box without letting an unbounded build (seconds to minutes) pass.
const BUDGET: Duration = Duration::from_secs(5);

fn codec() -> TextCodec {
    let tokens: Vec<String> = [
        "<unk>",
        "<s>",
        "</s>",
        "\u{2581}the",
        "\u{2581}cat",
        "\u{2581}",
        "a",
        "b",
        "c",
        "0",
        "1",
        "{",
        "}",
        "\"",
        "<0x41>",
    ]
    .map(String::from)
    .to_vec();
    let mut token_types = vec![1u32; tokens.len()];
    token_types[0] = 2;
    token_types[1] = 3;
    token_types[2] = 3;
    *token_types.last_mut().unwrap() = 6;
    let vocab = GgufVocab {
        model: Some("llama".into()),
        scores: Some(vec![0.0; tokens.len()]),
        token_types: Some(token_types),
        tokens: Some(tokens),
        unknown_token_id: Some(0),
        add_space_prefix: Some(true),
        ..GgufVocab::default()
    };
    TextCodec::from_gguf(&vocab, 2, ChatFormat::ChatML)
        .expect("build the SPM codec")
        .with_tool_codec(Some(Arc::new(HermesToolCodec)))
}

/// A deterministic xorshift stream (the repository's tests carry no rng dependency).
fn stream(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed
        .wrapping_mul(0x100000001B3)
        .wrapping_add(0x9E3779B97F4A7C15)
        | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// Most resident memory (bytes) the whole test process may have used after any build: the high-water
/// mark, so a build that allocated and freed still counts. The builds here need a few megabytes; a
/// build that sizes its work from the client's text instead of from a limit needs gigabytes (the
/// regex text of a nested schema doubled per level, and one test binary reached 115 GiB).
const MEMORY_BOUND: u64 = 2 << 30;

/// `VmHWM` of this process, in bytes.
fn peak_resident_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let kib: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kib| kib.trim().parse().ok())
        .expect("VmHWM in /proc/self/status");
    kib * 1024
}

/// Run `build`, require it to return inside [`BUDGET`] and the process to stay under [`MEMORY_BOUND`],
/// and give back what it answered.
fn bounded<T>(what: &str, build: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let answer = build();
    let took = started.elapsed();
    assert!(
        took < BUDGET,
        "{what}: took {took:?}, over the {BUDGET:?} budget"
    );
    let peak = peak_resident_bytes();
    assert!(
        peak < MEMORY_BOUND,
        "{what}: the process reached {peak} resident bytes, over the {MEMORY_BOUND} bound"
    );
    answer
}

/// A built constraint answers its first mask and survives a few steps along the lowest allowed ids.
fn walk_a_few_steps(constraint: &mut Constraint) {
    for _ in 0..4 {
        let (allowed, eos_ok) = constraint.allowed_next();
        if eos_ok {
            return;
        }
        let Some(next) = allowed.into_iter().min() else {
            return;
        };
        constraint.advance(next);
    }
}

// ---------------------------------------------------------------------------------------------
// regex
// ---------------------------------------------------------------------------------------------

/// The patterns that would build an automaton past the limits are refused by the builder, quickly,
/// with the error that names the pattern. Each is a class of blow-up: determinization (a counter after
/// an unanchored star), nested counted repetition, a wide Unicode class repeated, and a large
/// alternation of long words.
#[test]
fn patterns_that_blow_up_the_automaton_are_refused_quickly() {
    let codec = codec();
    let wide_alternation = (0..4000)
        .map(|i| format!("word{i}abcdefghij{i}"))
        .collect::<Vec<_>>()
        .join("|");
    let hostile = [
        ("determinization", "(a|b)*a(a|b){17}".to_string()),
        ("nested counters", "((a{1000}){1000}){1000}".to_string()),
        ("unicode class repeated", r"\p{L}{200}".to_string()),
        ("wide alternation", wide_alternation),
        (
            "case-folded unicode counter",
            r"(?i)\w{500}\w{500}".to_string(),
        ),
    ];
    for (name, pattern) in hostile {
        let error = bounded(name, || codec.build_regex_constraint(&pattern))
            .err()
            .unwrap_or_else(|| panic!("{name}: the pattern must be refused"));
        assert!(
            error.to_string().contains("guided regex"),
            "{name}: the refusal names the guided regex, got: {error}"
        );
    }
}

/// The refusal row's other side: a pattern inside the limits builds and constrains.
#[test]
fn a_pattern_inside_the_limits_builds() {
    let codec = codec();
    let mut constraint = bounded("modest pattern", || {
        codec.build_regex_constraint("(a|b)*a(a|b){6}")
    })
    .expect("a modest pattern builds");
    walk_a_few_steps(&mut constraint);
}

#[test]
fn regex_syntax_the_engine_refuses_is_an_error_not_a_panic() {
    let codec = codec();
    let deep_open = "(".repeat(200_000);
    let deep_balanced = format!("{}a{}", "(".repeat(100_000), ")".repeat(100_000));
    for (name, pattern) in [
        ("unclosed group", "(a".to_string()),
        ("stray close", "a)".to_string()),
        ("dangling repeat", "*a".to_string()),
        ("reversed counter", "a{5,2}".to_string()),
        ("counter over u32", "a{99999999999}".to_string()),
        ("bad class", "[z-a]".to_string()),
        ("lookahead", "(?=a)a".to_string()),
        ("backreference", r"(a)\1".to_string()),
        ("unknown property", r"\p{NotAProperty}".to_string()),
        ("invalid escape", r"\q".to_string()),
        ("lone backslash", "\\".to_string()),
        ("200k open parens", deep_open),
        ("100k nested groups", deep_balanced),
    ] {
        let outcome = bounded(name, || codec.build_regex_constraint(&pattern));
        if name != "100k nested groups" {
            assert!(outcome.is_err(), "{name}: must be refused");
        }
    }
}

/// Seeded random patterns over a small alphabet of operators, groups and counters: each builds or is
/// refused inside the budget, and a built constraint answers masks.
#[test]
fn random_patterns_build_or_are_refused_inside_the_budget() {
    let codec = codec();
    let mut next = stream(0x0BAD_5EED);
    let pieces = [
        "a", "b", "c", ".", "[ab]", "[^a]", "\\d", "\\w", "(", ")", "(?:", "|", "*", "+", "?",
        "{2}", "{1,3}", "{0,40}", "{30,}", "\\p{L}", "$", "^",
    ];
    let mut built = 0usize;
    for round in 0..1500 {
        let length = 1 + (next() % 24) as usize;
        let pattern: String = (0..length)
            .map(|_| pieces[(next() % pieces.len() as u64) as usize])
            .collect();
        match bounded(&format!("round {round} {pattern:?}"), || {
            codec.build_regex_constraint(&pattern)
        }) {
            Ok(mut constraint) => {
                built += 1;
                walk_a_few_steps(&mut constraint);
            }
            Err(error) => assert!(
                !error.to_string().is_empty(),
                "round {round} {pattern:?}: an empty refusal"
            ),
        }
    }
    assert!(
        built > 50,
        "only {built} of 1500 random patterns built: the generator is too hostile to cover the build path"
    );
}

// ---------------------------------------------------------------------------------------------
// JSON schema
// ---------------------------------------------------------------------------------------------

#[test]
fn schema_size_bombs_are_refused_or_built_inside_the_budget() {
    let codec = codec();
    let big_enum: Vec<_> = (0..20_000).map(|i| format!("choice-number-{i}")).collect();
    // $defs where each level references the one below twice: 2^24 expansion when inlined.
    let mut defs = serde_json::Map::new();
    defs.insert("d0".into(), serde_json::json!({"type": "integer"}));
    for level in 1..24 {
        defs.insert(
            format!("d{level}"),
            serde_json::json!({
                "type": "object",
                "properties": {
                    "l": {"$ref": format!("#/$defs/d{}", level - 1)},
                    "r": {"$ref": format!("#/$defs/d{}", level - 1)},
                },
                "required": ["l", "r"]
            }),
        );
    }
    let schemas = [
        ("enum of 20k strings", serde_json::json!({"enum": big_enum})),
        (
            "array of a billion items",
            serde_json::json!({"type": "array", "items": {"type": "integer"}, "minItems": 1_000_000_000u64, "maxItems": 1_000_000_001u64}),
        ),
        (
            "string of a billion characters",
            serde_json::json!({"type": "string", "minLength": 1_000_000_000u64, "maxLength": 1_000_000_001u64}),
        ),
        (
            "integer bound of 10^18 digits wide",
            serde_json::json!({"type": "integer", "minimum": i64::MIN, "maximum": i64::MAX}),
        ),
        (
            "doubling $ref chain",
            serde_json::json!({"$defs": defs, "$ref": "#/$defs/d23"}),
        ),
        (
            "hundreds of required properties",
            serde_json::json!({
                "type": "object",
                "properties": (0..400).map(|i| (format!("p{i}"), serde_json::json!({"type": "integer"}))).collect::<serde_json::Map<_, _>>(),
                "required": (0..400).map(|i| format!("p{i}")).collect::<Vec<_>>(),
            }),
        ),
    ];
    for (name, schema) in schemas {
        // Either answer is acceptable; what is not: a panic, or a build past the budget.
        match bounded(name, || codec.build_json_constraint(&schema)) {
            Ok(mut constraint) => walk_a_few_steps(&mut constraint),
            Err(error) => assert!(!error.to_string().is_empty(), "{name}: empty refusal"),
        }
    }
}

/// A schema 50 levels deep fits any request (`serde_json` stops parsing at 128 levels), yet an array
/// repeats its item in the regex, so the text doubles per level. The size bound refuses it while the
/// text is built (a test binary once reached 115 GiB here).
#[test]
fn a_deeply_nested_schema_is_refused_by_the_regex_size_bound() {
    let codec = codec();
    let mut schema = serde_json::json!({"type": "integer"});
    for _ in 0..50 {
        schema = serde_json::json!({"type": "array", "items": schema});
    }
    let text = schema.to_string();
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("50 levels parse");
    let error = bounded("50-level array schema", || {
        codec.build_json_constraint(&parsed)
    })
    .err()
    .expect("an array repeats its item, so 50 levels name 2^50 bytes of regex");
    assert!(error.to_string().contains("bytes of regex"), "{error}");

    let mut nested = serde_json::json!({"type": "string"});
    for _ in 0..50 {
        nested =
            serde_json::json!({"type": "object", "properties": {"k": nested}, "required": ["k"]});
    }
    let parsed: serde_json::Value = serde_json::from_str(&nested.to_string()).unwrap();
    bounded("50-level object schema", || {
        codec.build_json_constraint(&parsed)
    })
    .ok();
}

#[test]
fn schemas_of_the_wrong_shape_are_errors_not_panics() {
    let codec = codec();
    for (name, schema) in [
        ("null", serde_json::Value::Null),
        ("a string", serde_json::json!("integer")),
        ("a number", serde_json::json!(7)),
        ("an array", serde_json::json!([{"type": "integer"}])),
        ("unknown type", serde_json::json!({"type": "quaternion"})),
        ("type is a number", serde_json::json!({"type": 5})),
        (
            "properties is a list",
            serde_json::json!({"type": "object", "properties": []}),
        ),
        (
            "required names a missing property",
            serde_json::json!({"type": "object", "properties": {}, "required": ["ghost"]}),
        ),
        ("enum is empty", serde_json::json!({"enum": []})),
        ("enum is not a list", serde_json::json!({"enum": 3})),
        (
            "ref to nowhere",
            serde_json::json!({"$ref": "#/$defs/missing"}),
        ),
        (
            "remote ref",
            serde_json::json!({"$ref": "https://example.com/s.json"}),
        ),
        (
            "recursive ref",
            serde_json::json!({"$defs": {"n": {"type": "object", "properties": {"next": {"$ref": "#/$defs/n"}}, "required": ["next"]}}, "$ref": "#/$defs/n"}),
        ),
        (
            "pattern is invalid",
            serde_json::json!({"type": "string", "pattern": "("}),
        ),
        (
            "bounds reversed",
            serde_json::json!({"type": "integer", "minimum": 5, "maximum": 2}),
        ),
    ] {
        let outcome = bounded(name, || codec.build_json_constraint(&schema));
        // These shapes are tolerated today (a missing property is skipped, an empty enum matches the
        // empty text); the row asserts only that they do not panic or run away. Every other shape
        // must be refused.
        let tolerated = [
            "required names a missing property",
            "enum is empty",
            "enum is not a list",
            "type is a number",
            "properties is a list",
        ];
        if !tolerated.contains(&name) {
            assert!(outcome.is_err(), "{name}: must be refused");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// choice, grammar, tool call
// ---------------------------------------------------------------------------------------------

#[test]
fn a_choice_list_of_any_size_builds_inside_the_budget() {
    let codec = codec();
    let choices: Vec<String> = (0..50_000).map(|i| format!("choice {i}")).collect();
    let mut constraint = bounded("50k choices", || codec.build_choice_constraint(&choices))
        .expect("choices are encoded, not compiled");
    walk_a_few_steps(&mut constraint);
    let mut empty =
        bounded("no choices", || codec.build_choice_constraint(&[])).expect("an empty list builds");
    walk_a_few_steps(&mut empty);
    bounded("an empty choice", || {
        codec.build_choice_constraint(&[String::new()])
    })
    .ok();
}

#[test]
fn grammars_the_parser_cannot_use_are_errors_not_panics() {
    let codec = codec();
    let deep = format!("root ::= {}\"a\"{}", "(".repeat(5_000), ")".repeat(5_000));
    for (name, gbnf) in [
        ("empty", String::new()),
        ("no root", "other ::= \"a\"".to_string()),
        ("undefined rule", "root ::= missing".to_string()),
        ("left recursion", "root ::= root \"a\" | \"a\"".to_string()),
        ("unterminated string", "root ::= \"abc".to_string()),
        ("bad class", "root ::= [z-a]".to_string()),
        ("unclosed group", "root ::= (\"a\"".to_string()),
        ("binary junk", "\u{0}\u{1}\u{2}root ::=".to_string()),
        ("5k nested groups", deep),
    ] {
        bounded(name, || codec.build_grammar_constraint(&gbnf)).ok();
    }
}

#[test]
fn random_grammar_text_never_panics() {
    let codec = codec();
    let mut next = stream(0x6BAD_F00D);
    let pieces = [
        "root", "::=", "\"a\"", "\"bc\"", "[a-c]", "[^a]", "(", ")", "|", "*", "+", "?", "x", "y",
        "\n", " ", "{2}", "{1,3}", "\"", "[", "]", "-",
    ];
    for round in 0..2000 {
        let length = 1 + (next() % 20) as usize;
        let gbnf: String = (0..length)
            .map(|_| pieces[(next() % pieces.len() as u64) as usize])
            .collect::<Vec<_>>()
            .join(" ");
        if let Ok(mut constraint) = bounded(&format!("round {round} {gbnf:?}"), || {
            codec.build_grammar_constraint(&gbnf)
        }) {
            walk_a_few_steps(&mut constraint);
        }
    }
}

#[test]
fn forced_tool_calls_of_any_count_build_inside_the_budget() {
    let codec = codec();
    let tools: Vec<(String, serde_json::Value)> = (0..300)
        .map(|i| {
            (
                format!("tool_{i}"),
                serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
            )
        })
        .collect();
    bounded("300 forced tools", || codec.forced_tool_constraint(&tools)).ok();
    let hostile_name = vec![(
        "na\"me\\(".to_string(),
        serde_json::json!({"type": "integer"}),
    )];
    let built = bounded("a name that is regex syntax", || {
        codec.forced_tool_constraint(&hostile_name)
    });
    if let Ok(Some(mut constraint)) = built {
        walk_a_few_steps(&mut constraint);
    }
    assert!(
        bounded("no tools", || codec.forced_tool_constraint(&[])).is_err(),
        "a forced call with no tool to force has no constraint"
    );
}
