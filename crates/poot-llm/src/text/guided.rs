//! Guided decoding: Constraint (choice/regex/JSON-schema/grammar/tool-call) and token masks.

use crate::error::{OptionExt, Result};
use crate::text::grammar;
use crate::text::tokenize::TextCodec;

impl TextCodec {
    /// Build a `guided_choice` constraint: each choice is encoded to its token sequence (no BOS, no
    /// special tokens). Choices that encode to nothing are dropped.
    pub fn build_choice_constraint(&self, choices: &[String]) -> Result<Constraint> {
        let mut seqs = Vec::with_capacity(choices.len());
        for c in choices {
            let enc = self
                .tokenizer
                .encode(c.as_str(), false)
                .map_err(|e| err!("encode choice {c:?}: {e}"))?;
            seqs.push(enc.get_ids().to_vec());
        }
        Ok(Constraint::choices(seqs, self.eos))
    }

    /// Build a `guided_regex` constraint: an anchored byte-level DFA for `pattern` plus a per-token byte
    /// table. Tokens are allowed while the output stays a prefix of some full match.
    pub fn build_regex_constraint(&self, pattern: &str) -> Result<Constraint> {
        use regex_automata::dfa::Automaton;
        use regex_automata::{Anchored, Input};
        let dfa = self.guided_dfa(pattern)?;
        let start = dfa
            .start_state_forward(&Input::new("").anchored(Anchored::Yes))
            .map_err(|e| err!("guided regex start state: {e}"))?;
        Ok(Constraint::regex(
            dfa,
            start,
            self.token_byte_table(),
            self.eos,
        ))
    }

    /// The anchored byte-DFA for `pattern`, cached by pattern. Building it is the dominant per-request
    /// cost of `guided_json` (~14ms for a small schema) and clients reuse schemas. The cache is bounded
    /// (it clears past a cap).
    fn guided_dfa(
        &self,
        pattern: &str,
    ) -> Result<std::sync::Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>> {
        use regex_automata::dfa::{StartKind, dense};
        const CAP: usize = 64;
        if let Some(dfa) = self.guided_dfa_cache.lock().unwrap().get(pattern) {
            return Ok(dfa.clone());
        }
        // The whole output must match: anchor the start and append an end anchor so a state is dead
        // exactly when the prefix can no longer extend to a full match (without it the DFA matches a
        // prefix early and never dead-ends on trailing garbage). Group the pattern so `$` binds after
        // all alternatives.
        let whole = format!("(?:{pattern})$");
        let dfa = std::sync::Arc::new(
            dense::Builder::new()
                .configure(dense::Config::new().start_kind(StartKind::Anchored))
                .build(&whole)
                .map_err(|e| err!("compile guided regex {pattern:?}: {e}"))?,
        );
        let mut cache = self.guided_dfa_cache.lock().unwrap();
        if cache.len() >= CAP {
            cache.clear(); // bounded: drop the cold set rather than grow without limit
        }
        cache.insert(pattern.to_string(), dfa.clone());
        Ok(dfa)
    }

    /// The per-token UTF-8 byte table (token id -> decoded bytes), built once and cached (~50ms for a
    /// 150k vocab). Byte-fallback tokens decode to U+FFFD and so are masked out of ASCII/JSON patterns.
    fn token_byte_table(&self) -> std::sync::Arc<Vec<Vec<u8>>> {
        self.guided_byte_table
            .get_or_init(|| {
                let vocab = self.tokenizer.get_vocab_size(true);
                let mut bytes = Vec::with_capacity(vocab);
                for id in 0..vocab as u32 {
                    let s = self.tokenizer.decode(&[id], false).unwrap_or_default();
                    bytes.push(s.into_bytes());
                }
                std::sync::Arc::new(bytes)
            })
            .clone()
    }

    /// Build a `guided_json` constraint: compile a JSON Schema to a regex and constrain decoding to it.
    /// See [`json_schema_to_regex`] for the supported subset.
    pub fn build_json_constraint(&self, schema: &serde_json::Value) -> Result<Constraint> {
        let pattern = json_schema_to_regex(schema)?;
        self.build_regex_constraint(&pattern)
    }

    /// Build a `json_object` constraint (OpenAI `response_format: {"type":"json_object"}`): one well-formed
    /// JSON value of any shape and depth. Uses the pushdown [`Constraint::JsonValue`] acceptor, so nesting
    /// is unbounded. Prefer `build_json_constraint` when the schema is known.
    pub fn build_json_object_constraint(&self) -> Result<Constraint> {
        Ok(Constraint::json_value(self.token_byte_table(), self.eos))
    }

    /// Build a `guided_grammar` constraint from GBNF source. The output is constrained to a sentence of
    /// the grammar, so unlike the regex path it can express balanced/recursive structure. Parsing is cheap
    /// and not cached. See [`grammar::Grammar`].
    pub fn build_grammar_constraint(&self, gbnf: &str) -> Result<Constraint> {
        let g = grammar::Grammar::parse(gbnf).map_err(|e| err!("guided_grammar: {e}"))?;
        Ok(Constraint::grammar(
            std::sync::Arc::new(g),
            self.token_byte_table(),
            self.eos,
        ))
    }

    /// Build a forced tool-call constraint (OpenAI `tool_choice`): exactly one ChatML
    /// `<tool_call>{json}</tool_call>` block with `{"name": <forced>, "arguments": <args>}`, arguments
    /// conforming to the tool's parameter schema. `forced` is the (name, parameter-schema) list the
    /// request permits (one for a named tool, all for `"required"`). Only valid for models that emit
    /// ChatML tool-call markup (the caller gates on chat format).
    pub fn build_tool_call_constraint(
        &self,
        forced: &[(String, serde_json::Value)],
    ) -> Result<Constraint> {
        let pattern = tool_call_regex(forced)?;
        self.build_regex_constraint(&pattern)
    }
}

/// A guided-decoding constraint that masks each decode step to the tokens it permits. Stateful: it
/// advances as each generated token is observed.
///
/// - `Choices`: the completion must be one of a fixed set of strings (token-level). Built by
///   [`Runner::build_choice_constraint`].
/// - `Regex`: the completion must match a regex (byte-level DFA; a token is allowed if its bytes avoid
///   the dead state, and EOS is allowed in a match state). Built by [`TextCodec::build_regex_constraint`].
#[derive(Clone)]
pub enum Constraint {
    Choices {
        /// the allowed choices, each as its token-id sequence (no BOS).
        choices: Vec<Vec<u32>>,
        /// the generated tokens so far (advances as tokens are observed).
        generated: Vec<u32>,
        /// the model's EOS id, permitted once the generated tokens complete a choice.
        eos: u32,
    },
    Regex {
        dfa: std::sync::Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>,
        /// the current DFA state (advances as tokens are observed).
        state: regex_automata::util::primitives::StateID,
        /// per-token UTF-8 bytes, indexed by token id (shared from the runner).
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        eos: u32,
    },
    /// The completion must be a single well-formed JSON value of any shape and nesting depth
    /// (`response_format: json_object`). Unlike `Regex`, this is a pushdown acceptor (a stack of open
    /// containers), so it can count brackets. A token is allowed if its bytes advance the acceptor; EOS is
    /// allowed once a complete top-level value is held. See [`JsonAcceptor`] and
    /// [`TextCodec::build_json_object_constraint`].
    JsonValue {
        /// the incremental JSON acceptor (advances as tokens are observed).
        acc: JsonAcceptor,
        /// per-token UTF-8 bytes, indexed by token id (shared from the runner).
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        eos: u32,
    },
    /// The completion must be a sentence of a user-supplied GBNF grammar (`guided_grammar`). Like
    /// `JsonValue` a pushdown engine, but general: it walks a set of parser stacks. A token is allowed if
    /// its decoded characters keep some parse alive; EOS is allowed once some parse is complete. See
    /// [`grammar::GrammarState`] and [`TextCodec::build_grammar_constraint`].
    Grammar {
        /// the incremental grammar parser state (advances as tokens are observed).
        state: grammar::GrammarState,
        /// per-token UTF-8 bytes, indexed by token id (shared from the runner).
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        eos: u32,
    },
}

impl Constraint {
    /// Build a `guided_choice` constraint from pre-encoded token sequences. Empty sequences are dropped.
    pub fn choices(choices: Vec<Vec<u32>>, eos: u32) -> Self {
        Constraint::Choices {
            choices: choices.into_iter().filter(|c| !c.is_empty()).collect(),
            generated: Vec::new(),
            eos,
        }
    }

    /// Build a `guided_regex` constraint from a compiled anchored byte-DFA, its start state, the per-token
    /// byte table, and the EOS id.
    pub fn regex(
        dfa: std::sync::Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>,
        start: regex_automata::util::primitives::StateID,
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        eos: u32,
    ) -> Self {
        Constraint::Regex {
            dfa,
            state: start,
            bytes,
            eos,
        }
    }

    /// Build a `json_object` constraint from the per-token byte table and the EOS id.
    pub fn json_value(bytes: std::sync::Arc<Vec<Vec<u8>>>, eos: u32) -> Self {
        Constraint::JsonValue {
            acc: JsonAcceptor::new(),
            bytes,
            eos,
        }
    }

    /// Build a `guided_grammar` constraint from a compiled grammar, the per-token byte table, and the EOS id.
    pub fn grammar(
        grammar: std::sync::Arc<grammar::Grammar>,
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        eos: u32,
    ) -> Self {
        Constraint::Grammar {
            state: grammar.start_state(),
            bytes,
            eos,
        }
    }

    /// The set of allowed next token ids and whether EOS is permitted at the current state.
    pub(crate) fn allowed_next(&self) -> (std::collections::HashSet<u32>, bool) {
        use regex_automata::dfa::Automaton;
        let mut allowed = std::collections::HashSet::new();
        match self {
            Constraint::Choices {
                choices, generated, ..
            } => {
                let mut eos_ok = false;
                for c in choices {
                    if c.len() >= generated.len() && c[..generated.len()] == *generated {
                        if c.len() == generated.len() {
                            eos_ok = true;
                        } else {
                            allowed.insert(c[generated.len()]);
                        }
                    }
                }
                (allowed, eos_ok)
            }
            Constraint::Regex {
                dfa, state, bytes, ..
            } => {
                // a token is allowed if its bytes walk the DFA from `state` without hitting a dead/quit
                // state. EOS is allowed when the current state matches (the output so far is a full match).
                for (id, tok_bytes) in bytes.iter().enumerate() {
                    if tok_bytes.is_empty() {
                        continue;
                    }
                    let mut s = *state;
                    let mut alive = true;
                    for &b in tok_bytes {
                        s = dfa.next_state(s, b);
                        if dfa.is_dead_state(s) || dfa.is_quit_state(s) {
                            alive = false;
                            break;
                        }
                    }
                    if alive {
                        allowed.insert(id as u32);
                    }
                }
                let eos_ok = dfa.is_match_state(dfa.next_eoi_state(*state));
                (allowed, eos_ok)
            }
            Constraint::JsonValue { acc, bytes, .. } => {
                // a token is allowed if its bytes drive the JSON acceptor from the current state without a
                // rejection (simulated on a clone, so the live state is untouched). EOS is allowed once the
                // acceptor holds a complete top-level value.
                for (id, tok_bytes) in bytes.iter().enumerate() {
                    if tok_bytes.is_empty() {
                        continue;
                    }
                    let mut a = acc.clone();
                    let mut alive = true;
                    for &b in tok_bytes {
                        if !a.step(b) {
                            alive = false;
                            break;
                        }
                    }
                    if alive {
                        allowed.insert(id as u32);
                    }
                }
                (allowed, acc.is_complete())
            }
            Constraint::Grammar { state, bytes, .. } => {
                // a token is allowed if its decoded characters keep some parse alive (simulated on a clone).
                // Tokens that do not decode to valid UTF-8 cannot feed the char-level engine and are masked.
                for (id, tok_bytes) in bytes.iter().enumerate() {
                    if tok_bytes.is_empty() {
                        continue;
                    }
                    let Ok(text) = std::str::from_utf8(tok_bytes) else {
                        continue;
                    };
                    let mut s = state.clone();
                    let mut alive = true;
                    for ch in text.chars() {
                        if !s.accept_char(ch) {
                            alive = false;
                            break;
                        }
                    }
                    if alive {
                        allowed.insert(id as u32);
                    }
                }
                (allowed, state.is_complete())
            }
        }
    }

    /// Advance the constraint by one generated token.
    pub(crate) fn advance(&mut self, token: u32) {
        use regex_automata::dfa::Automaton;
        match self {
            Constraint::Choices { generated, .. } => generated.push(token),
            Constraint::Regex {
                dfa, state, bytes, ..
            } => {
                if let Some(tok_bytes) = bytes.get(token as usize) {
                    for &b in tok_bytes {
                        *state = dfa.next_state(*state, b);
                    }
                }
            }
            Constraint::JsonValue { acc, bytes, .. } => {
                if let Some(tok_bytes) = bytes.get(token as usize) {
                    for &b in tok_bytes {
                        // the chosen token was in the allowed set, so every byte steps cleanly.
                        let _ = acc.step(b);
                    }
                }
            }
            Constraint::Grammar { state, bytes, .. } => {
                if let Some(tok_bytes) = bytes.get(token as usize)
                    && let Ok(text) = std::str::from_utf8(tok_bytes)
                {
                    for ch in text.chars() {
                        // the chosen token was in the allowed set, so every char steps cleanly.
                        let _ = state.accept_char(ch);
                    }
                }
            }
        }
    }

    pub(crate) fn eos(&self) -> u32 {
        match self {
            Constraint::Choices { eos, .. }
            | Constraint::Regex { eos, .. }
            | Constraint::JsonValue { eos, .. }
            | Constraint::Grammar { eos, .. } => *eos,
        }
    }
}

/// One open container on the acceptor's stack.
#[derive(Clone, Copy, PartialEq, Eq)]
enum JsonCtr {
    Obj,
    Arr,
}

/// Where the acceptor is within the byte stream. A scalar (string / number / literal) that is not closed by
/// a delimiter (a number) stays in its own sub-state until a terminator byte is seen.
#[derive(Clone, Copy)]
enum JsonMode {
    /// expecting a value to begin (root, after `[`, after `,` in an array, or after `:` in an object).
    Value,
    /// just after `{`: expecting the first key string or `}` (empty object).
    ObjFirstKey,
    /// just after `,` in an object: expecting a key string (a close is not allowed - no trailing comma).
    ObjKey,
    /// just after `[`: expecting the first value or `]` (empty array).
    ArrFirstValue,
    /// inside a string; `is_key` marks a string that is an object key (so its close expects `:`).
    Str { is_key: bool },
    /// inside a string, just after a `\`.
    StrEsc { is_key: bool },
    /// inside a `\u` escape; `left` hex digits remain (1..=4).
    StrU { is_key: bool, left: u8 },
    /// a key string just closed: expecting `:`.
    Colon,
    /// just consumed a leading `-`: need the first digit.
    NumMinus,
    /// the integer part is exactly `0` (no more integer digits may follow).
    NumZero,
    /// the integer part has >= 1 digit and a nonzero lead.
    NumInt,
    /// just consumed `.`: need >= 1 fraction digit.
    NumDot,
    /// >= 1 fraction digit consumed.
    NumFrac,
    /// just consumed `e`/`E`: need a sign or a digit.
    NumExp,
    /// consumed a sign after `e`/`E`: need a digit.
    NumExpSign,
    /// >= 1 exponent digit consumed.
    NumExpDig,
    /// matching a literal (`true` / `false` / `null`); `i` bytes of `s` already matched.
    Lit { s: &'static [u8], i: usize },
    /// a value just completed; the next byte is decided by the current container (`,`/close/etc.).
    AfterValue,
    /// the root value is complete (stack empty): only trailing whitespace may follow, and EOS is allowed.
    Done,
}

/// Incremental byte-level pushdown acceptor for a single JSON value of any nesting depth. A stack of
/// open containers (`JsonCtr`) requires every `{`/`[` to be matched before the value is complete.
/// `step` feeds one byte and returns whether it was accepted; `is_complete` reports whether the bytes
/// seen so far form a whole top-level value (EOS allowed).
#[derive(Clone)]
pub struct JsonAcceptor {
    stack: Vec<JsonCtr>,
    mode: JsonMode,
}

impl JsonAcceptor {
    fn new() -> Self {
        JsonAcceptor {
            stack: Vec::new(),
            mode: JsonMode::Value,
        }
    }

    /// JSON insignificant whitespace.
    fn is_ws(b: u8) -> bool {
        matches!(b, b' ' | b'\t' | b'\n' | b'\r')
    }

    fn is_hex(b: u8) -> bool {
        b.is_ascii_digit() || (b'a'..=b'f').contains(&b) || (b'A'..=b'F').contains(&b)
    }

    /// Pop the just-closed container; the container is itself a value, so we land in `AfterValue`.
    fn pop_container(&mut self) {
        self.stack.pop();
        self.mode = JsonMode::AfterValue;
    }

    /// Feed one byte. Returns false (leaving the state unspecified, so callers clone before probing) if
    /// the byte is not acceptable. The inner loop re-dispatches only when a number or completed root
    /// value is terminated by a byte that belongs to the surrounding context; each re-dispatch advances
    /// the mode toward a returning state, so it terminates.
    fn step(&mut self, b: u8) -> bool {
        loop {
            match self.mode {
                JsonMode::Done => return Self::is_ws(b),
                JsonMode::Value => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    match b {
                        b'{' => {
                            self.stack.push(JsonCtr::Obj);
                            self.mode = JsonMode::ObjFirstKey;
                        }
                        b'[' => {
                            self.stack.push(JsonCtr::Arr);
                            self.mode = JsonMode::ArrFirstValue;
                        }
                        b'"' => self.mode = JsonMode::Str { is_key: false },
                        b'-' => self.mode = JsonMode::NumMinus,
                        b'0' => self.mode = JsonMode::NumZero,
                        b'1'..=b'9' => self.mode = JsonMode::NumInt,
                        b't' => self.mode = JsonMode::Lit { s: b"true", i: 1 },
                        b'f' => self.mode = JsonMode::Lit { s: b"false", i: 1 },
                        b'n' => self.mode = JsonMode::Lit { s: b"null", i: 1 },
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::ObjFirstKey => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    match b {
                        b'"' => self.mode = JsonMode::Str { is_key: true },
                        b'}' => self.pop_container(), // empty object
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::ObjKey => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    match b {
                        b'"' => self.mode = JsonMode::Str { is_key: true },
                        _ => return false, // a close is not allowed right after `,`
                    }
                    return true;
                }
                JsonMode::ArrFirstValue => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    if b == b']' {
                        self.pop_container(); // empty array
                        return true;
                    }
                    // otherwise a value begins here: reduce to Value and re-dispatch this byte.
                    self.mode = JsonMode::Value;
                }
                JsonMode::Str { is_key } => {
                    match b {
                        b'"' => {
                            self.mode = if is_key {
                                JsonMode::Colon
                            } else {
                                JsonMode::AfterValue
                            };
                        }
                        b'\\' => self.mode = JsonMode::StrEsc { is_key },
                        0x00..=0x1f => return false, // control chars must be escaped
                        _ => {} // any other byte (incl. UTF-8 continuation) stays in the string
                    }
                    return true;
                }
                JsonMode::StrEsc { is_key } => {
                    match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                            self.mode = JsonMode::Str { is_key }
                        }
                        b'u' => self.mode = JsonMode::StrU { is_key, left: 4 },
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::StrU { is_key, left } => {
                    if Self::is_hex(b) {
                        self.mode = if left <= 1 {
                            JsonMode::Str { is_key }
                        } else {
                            JsonMode::StrU {
                                is_key,
                                left: left - 1,
                            }
                        };
                        return true;
                    }
                    return false;
                }
                JsonMode::Colon => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    if b == b':' {
                        self.mode = JsonMode::Value;
                        return true;
                    }
                    return false;
                }
                JsonMode::NumMinus => {
                    match b {
                        b'0' => self.mode = JsonMode::NumZero,
                        b'1'..=b'9' => self.mode = JsonMode::NumInt,
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::NumZero => match b {
                    b'.' => {
                        self.mode = JsonMode::NumDot;
                        return true;
                    }
                    b'e' | b'E' => {
                        self.mode = JsonMode::NumExp;
                        return true;
                    }
                    _ => self.mode = JsonMode::AfterValue, // terminator: reduce and re-dispatch
                },
                JsonMode::NumInt => match b {
                    b'0'..=b'9' => return true,
                    b'.' => {
                        self.mode = JsonMode::NumDot;
                        return true;
                    }
                    b'e' | b'E' => {
                        self.mode = JsonMode::NumExp;
                        return true;
                    }
                    _ => self.mode = JsonMode::AfterValue,
                },
                JsonMode::NumDot => {
                    match b {
                        b'0'..=b'9' => self.mode = JsonMode::NumFrac,
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::NumFrac => match b {
                    b'0'..=b'9' => return true,
                    b'e' | b'E' => {
                        self.mode = JsonMode::NumExp;
                        return true;
                    }
                    _ => self.mode = JsonMode::AfterValue,
                },
                JsonMode::NumExp => {
                    match b {
                        b'+' | b'-' => self.mode = JsonMode::NumExpSign,
                        b'0'..=b'9' => self.mode = JsonMode::NumExpDig,
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::NumExpSign => {
                    match b {
                        b'0'..=b'9' => self.mode = JsonMode::NumExpDig,
                        _ => return false,
                    }
                    return true;
                }
                JsonMode::NumExpDig => match b {
                    b'0'..=b'9' => return true,
                    _ => self.mode = JsonMode::AfterValue,
                },
                JsonMode::Lit { s, i } => {
                    if i < s.len() && b == s[i] {
                        self.mode = if i + 1 == s.len() {
                            JsonMode::AfterValue
                        } else {
                            JsonMode::Lit { s, i: i + 1 }
                        };
                        return true;
                    }
                    return false;
                }
                JsonMode::AfterValue => {
                    if Self::is_ws(b) {
                        return true;
                    }
                    match self.stack.last() {
                        None => {
                            // the root value is complete: only whitespace/EOS may follow.
                            self.mode = JsonMode::Done;
                            // re-dispatch this byte in Done (accepts ws, rejects anything else).
                        }
                        Some(JsonCtr::Arr) => {
                            match b {
                                b',' => self.mode = JsonMode::Value,
                                b']' => self.pop_container(),
                                _ => return false,
                            }
                            return true;
                        }
                        Some(JsonCtr::Obj) => {
                            match b {
                                b',' => self.mode = JsonMode::ObjKey,
                                b'}' => self.pop_container(),
                                _ => return false,
                            }
                            return true;
                        }
                    }
                }
            }
        }
    }

    /// True when the bytes seen so far form one complete top-level JSON value (stack empty), where EOS is
    /// permitted. Numbers complete without a closing delimiter, so their complete sub-states count.
    fn is_complete(&self) -> bool {
        self.stack.is_empty()
            && matches!(
                self.mode,
                JsonMode::Done
                    | JsonMode::AfterValue
                    | JsonMode::NumZero
                    | JsonMode::NumInt
                    | JsonMode::NumFrac
                    | JsonMode::NumExpDig
            )
    }
}

/// A regex matching exactly the integers within the given (possibly one-sided or absent) bounds, in
/// canonical JSON form (no leading zeros, optional `-`). A one-sided bound is still a regular language,
/// so it is expressed exactly. `None`/`None` is the unbounded integer.
pub(crate) fn int_bound_regex(lo: Option<i64>, hi: Option<i64>) -> Result<String> {
    fn pow10(c: u32) -> u128 {
        10u128.pow(c)
    }
    // `num` with its last `c` decimal digits set to 9 / to 0.
    fn fill_nines(num: u128, c: u32) -> u128 {
        num - (num % pow10(c)) + (pow10(c) - 1)
    }
    fn fill_zeros(num: u128, c: u32) -> u128 {
        num - (num % pow10(c))
    }
    // boundary "stops" splitting [a, b] into power-of-10-aligned sub-ranges.
    fn split(a: u128, b: u128) -> Vec<u128> {
        let mut stops = std::collections::BTreeSet::new();
        stops.insert(b);
        let mut c = 1;
        loop {
            let s = fill_nines(a, c);
            if s >= b {
                break;
            }
            stops.insert(s);
            c += 1;
        }
        let mut c = 1;
        loop {
            let base = fill_zeros(b + 1, c);
            if base == 0 {
                break;
            }
            let s = base - 1;
            if s <= a {
                break;
            }
            stops.insert(s);
            c += 1;
        }
        stops.into_iter().collect()
    }
    // per-digit pattern for [start, stop] (same digit count - the split guarantees alignment).
    fn to_pattern(start: u128, stop: u128) -> String {
        let (ss, st) = (start.to_string(), stop.to_string());
        let mut pat = String::new();
        let mut anyc = 0;
        for (cs, ct) in ss.chars().zip(st.chars()) {
            if cs == ct {
                pat.push(cs);
            } else if cs == '0' && ct == '9' {
                anyc += 1;
            } else {
                pat.push_str(&format!("[{cs}-{ct}]"));
            }
        }
        if anyc > 0 {
            pat.push_str("[0-9]");
            if anyc > 1 {
                pat.push_str(&format!("{{{anyc}}}"));
            }
        }
        pat
    }
    // [a, b] for non-negative magnitudes a <= b.
    fn pos(a: u128, b: u128) -> String {
        let mut start = a;
        let mut parts = Vec::new();
        for stop in split(a, b) {
            parts.push(to_pattern(start, stop));
            start = stop + 1;
        }
        if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            format!("(?:{})", parts.join("|"))
        }
    }
    // non-negative integers (magnitudes) >= k, with no leading zeros - the open-ended upper range. Numbers
    // with more digits than k always qualify; numbers with the same digit count must be >= k.
    fn mag_ge(k: u128) -> String {
        if k == 0 {
            return "(?:0|[1-9][0-9]*)".to_string();
        }
        let l = k.to_string().len() as u32; // k has l digits
        let same = pos(k, pow10(l) - 1); // [k, 10^l - 1], same digit count
        let longer = format!("[1-9][0-9]{{{l},}}"); // strictly more than l digits
        format!("(?:{same}|{longer})")
    }
    let body = match (lo, hi) {
        (Some(lo), Some(hi)) => {
            if lo > hi {
                bail!("guided_json: integer minimum {lo} > maximum {hi}");
            }
            // i128 so i64::MIN negates safely.
            let (lo, hi) = (lo as i128, hi as i128);
            if lo >= 0 {
                pos(lo as u128, hi as u128)
            } else if hi < 0 {
                // all negative: magnitudes |hi|..|lo|, prefixed "-".
                format!("-{}", pos((-hi) as u128, (-lo) as u128))
            } else {
                // straddles zero: negatives -1..lo (magnitudes 1..|lo|) | non-negatives 0..hi.
                format!("(?:-{}|{})", pos(1, (-lo) as u128), pos(0, hi as u128))
            }
        }
        (Some(lo), None) => {
            // integers >= lo.
            if lo >= 0 {
                mag_ge(lo as u128)
            } else {
                // negatives [lo, -1] (finite magnitudes 1..|lo|) | all non-negatives.
                format!("(?:-{}|{})", pos(1, (-(lo as i128)) as u128), mag_ge(0))
            }
        }
        (None, Some(hi)) => {
            // integers <= hi.
            if hi < 0 {
                // all <= hi < 0: negative, magnitude >= |hi|.
                format!("-{}", mag_ge((-(hi as i128)) as u128))
            } else {
                // all negatives (magnitude >= 1) | non-negatives [0, hi].
                format!("(?:-{}|{})", mag_ge(1), pos(0, hi as u128))
            }
        }
        (None, None) => "-?(?:0|[1-9][0-9]*)".to_string(),
    };
    Ok(format!("(?:{body})"))
}

/// Map a JSON Schema string `format` to a regex matching the string content (between the quotes).
/// Dates/times are RFC 3339; `date-time` requires an offset (`Z` or `+hh:mm`), `time`'s is optional. An
/// unknown format returns `None` and is ignored, since `format` is advisory.
fn json_format_regex(name: &str) -> Option<&'static str> {
    let date = "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])";
    Some(match name {
        "uuid" => "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "date" => date,
        "time" => {
            "(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
        }
        "date-time" => {
            "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
        }
        "email" => "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}",
        "ipv4" => {
            "(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])"
        }
        _ => return None,
    })
}

/// Build a regex matching exactly a forced ChatML tool call (OpenAI `tool_choice`): `<tool_call>` markup
/// around `{"name": <forced name>, "arguments": <that tool's parameter schema>}`. `forced` is
/// `(name, parameter-schema)` per permitted tool; multiple entries become an alternation. The body reuses
/// [`json_schema_to_regex`]. The markup allows at most one optional newline on each side (the ChatML
/// convention), not unbounded whitespace: `[ \t\r\n]*` lets a greedy model emit newlines forever instead
/// of entering the JSON. The parser trims the body, so a single newline round-trips.
pub fn tool_call_regex(forced: &[(String, serde_json::Value)]) -> Result<String> {
    if forced.is_empty() {
        bail!("tool_call: no tools to force");
    }
    let mut alts = Vec::with_capacity(forced.len());
    for (name, params) in forced {
        let call_schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "const": name }, "arguments": params },
            "required": ["name", "arguments"],
        });
        alts.push(json_schema_to_regex(&call_schema)?);
    }
    let body = if alts.len() == 1 {
        alts.pop().unwrap()
    } else {
        format!("(?:{})", alts.join("|"))
    };
    Ok(format!(r"<tool_call>\n?{body}\n?</tool_call>"))
}

/// Compile a concrete JSON Schema to a regex matching exactly the conforming JSON values (a schema with
/// fixed structure is regular, so it fits the byte-DFA engine). Optional whitespace (`[ \t\n\r]*`) is
/// allowed at structural points. Supported subset:
///
/// - `{"type":"object","properties":{...},"required":[...]}`: listed properties in schema (sorted-key)
///   order, comma-separated; `required` selects which must appear (the rest may be omitted, any in-order
///   subset). With no `required` field every property is required; pass `"required": []` for an
///   all-optional object.
/// - `{"type":"string"}`: `pattern` constrains the content to a regex as a full match, else a known
///   `format` (`uuid`, `date`, `time`, `date-time`, `email`, `ipv4`), else `minLength`/`maxLength` bound
///   its length (escapes not modeled). `{"type":"integer"}` with `minimum`/`maximum` (or exclusive forms)
///   bounds it to an exact range, two-sided or one-sided. Also `{"type":"number"}`, `{"type":"boolean"}`.
/// - `{"type":"array","items":<schema>}`; `minItems`/`maxItems` bound the count.
/// - `{"enum":[..]}` and `{"const":v}`: scalars match their canonical JSON, composites match structurally
///   with sorted keys and optional whitespace.
/// - `{"oneOf":[..]}` / `{"anyOf":[..]}`: a union; `oneOf`'s exactly-one is treated as at-least-one.
/// - `{"$ref":"#/$defs/Name"}`: a local JSON-Pointer reference (`$defs`, `definitions`, or any path) is
///   inlined. A recursive ref or an external ref (not starting with `#`) errors, since unbounded
///   recursion is not regular.
///
/// Unsupported constructs (unbounded objects, `number` bounds, object/array `const`/`enum`, etc.) fall
/// back to the unconstrained type or error rather than silently mis-constraining. Object keys are
/// emitted in sorted order (serde_json sorts them), which is valid JSON.
pub fn json_schema_to_regex(schema: &serde_json::Value) -> Result<String> {
    // optional JSON whitespace between structural tokens.
    const WS: &str = "[ \\t\\n\\r]*";
    // escape a literal string for use inside a regex.
    fn esc(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            if "\\.+*?()|[]{}^$".contains(c) {
                out.push('\\');
            }
            out.push(c);
        }
        out
    }
    // the regex matching exactly one JSON value (for `const` / `enum`). A scalar is its canonical JSON
    // serialization, regex-escaped (e.g. "a", 42, 4.2 -> 4\.2, true, null). A composite is matched
    // structurally with optional whitespace and object keys in sorted order, as in the object-schema path.
    fn literal_regex(v: &serde_json::Value) -> Result<String> {
        use serde_json::Value;
        match v {
            Value::String(_) | Value::Number(_) | Value::Bool(_) | Value::Null => {
                Ok(esc(&serde_json::to_string(v).expect("scalar serializes")))
            }
            Value::Array(items) => {
                let mut parts = Vec::with_capacity(items.len());
                for it in items {
                    parts.push(literal_regex(it)?);
                }
                Ok(format!(
                    "\\[{WS}{}{WS}\\]",
                    parts.join(&format!("{WS},{WS}"))
                ))
            }
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort(); // deterministic, matches the object-schema path
                let mut members = Vec::with_capacity(keys.len());
                for k in keys {
                    let key = esc(&serde_json::to_string(k).expect("key serializes"));
                    members.push(format!("{key}{WS}:{WS}{}", literal_regex(&map[k])?));
                }
                Ok(format!(
                    "\\{{{WS}{}{WS}\\}}",
                    members.join(&format!("{WS},{WS}"))
                ))
            }
        }
    }
    // Resolve a local JSON-Pointer `$ref` ("#/$defs/Name", "#/definitions/Name", ...) against the root
    // document. Only same-document refs are supported; an external/remote ref (not starting with `#`) errors.
    fn resolve_ref<'a>(root: &'a serde_json::Value, r: &str) -> Result<&'a serde_json::Value> {
        let ptr = r.strip_prefix('#').context(
            "guided_json: only local \"#/...\" $ref is supported (no external/remote refs)",
        )?;
        root.pointer(ptr)
            .with_context(|| format!("guided_json: $ref {r:?} does not resolve in the schema"))
    }
    // `root` is the whole schema document ($ref targets resolve against it); `active` is the stack of refs
    // currently being expanded, so a recursive ref (one already on its own path) is rejected, not looped.
    fn value_regex(
        schema: &serde_json::Value,
        ws: &str,
        root: &serde_json::Value,
        active: &mut Vec<String>,
    ) -> Result<String> {
        // `$ref`: inline the referenced subschema (taking precedence over sibling keywords, the common
        // generator shape). A ref already on the active path is a true cycle - not expressible as a regex.
        if let Some(serde_json::Value::String(r)) = schema.get("$ref") {
            if active.iter().any(|a| a == r) {
                bail!("guided_json: recursive $ref {r:?} is not expressible as a finite regex");
            }
            let target = resolve_ref(root, r)?;
            active.push(r.clone());
            let out = value_regex(target, ws, root, active);
            active.pop();
            return out;
        }
        // `const`: a single fixed JSON value.
        if let Some(c) = schema.get("const") {
            return Ok(format!("(?:{})", literal_regex(c)?));
        }
        // `oneOf` / `anyOf`: an alternation of subschemas. oneOf's "exactly one" is treated as "at
        // least one" since a regex cannot cheaply forbid a value matching two subschemas.
        if let Some(serde_json::Value::Array(subs)) =
            schema.get("oneOf").or_else(|| schema.get("anyOf"))
        {
            if subs.is_empty() {
                bail!("guided_json: oneOf/anyOf needs at least one subschema");
            }
            let mut alts: Vec<String> = Vec::with_capacity(subs.len());
            for s in subs {
                alts.push(value_regex(s, ws, root, active)?);
            }
            return Ok(format!("(?:{})", alts.join("|")));
        }
        // an `enum` of scalar literals: an alternation of their regex-escaped JSON serializations.
        if let Some(serde_json::Value::Array(variants)) = schema.get("enum") {
            let alts: Vec<String> = variants.iter().map(literal_regex).collect::<Result<_>>()?;
            return Ok(format!("(?:{})", alts.join("|")));
        }
        let ty = schema
            .get("type")
            .and_then(|t| t.as_str())
            .context("guided_json: each schema node needs a \"type\" (or \"enum\")")?;
        // a `{min,max}` regex quantifier (max optional -> unbounded); errors if min > max.
        fn rep(min: u64, max: Option<u64>, what: &str) -> Result<String> {
            if let Some(m) = max {
                if min > m {
                    bail!("guided_json: {what} min {min} > max {m}");
                }
                Ok(format!("{{{min},{m}}}"))
            } else {
                Ok(format!("{{{min},}}"))
            }
        }
        let u64_of = |k: &str| schema.get(k).and_then(|v| v.as_u64());
        Ok(match ty {
            "string" => {
                if let Some(serde_json::Value::String(p)) = schema.get("pattern") {
                    // a regex `pattern` on the string content as a full match (the quotes anchor it; a
                    // surrounding `^`/`$` is stripped). It matches raw JSON bytes, so escape sequences are
                    // not modeled and it must not contain an unescaped quote. If `minLength`/`maxLength` are
                    // also set the pattern governs.
                    let p = p.strip_prefix('^').unwrap_or(p);
                    let p = p.strip_suffix('$').unwrap_or(p);
                    format!("\"(?:{p})\"")
                } else if let Some(re) = schema
                    .get("format")
                    .and_then(|v| v.as_str())
                    .and_then(json_format_regex)
                {
                    // a known `format` constrains the content to its regex (full match); `pattern` governs
                    // if also present, and length bounds are not enforced. An unknown format is ignored.
                    format!("\"(?:{re})\"")
                } else {
                    let (min, max) = (u64_of("minLength"), u64_of("maxLength"));
                    if min.is_none() && max.is_none() {
                        // unbounded: quote, any run of non-quote/backslash chars OR escapes, quote.
                        "\"(?:\\\\.|[^\"\\\\])*\"".to_string()
                    } else {
                        // length-bounded: a run of non-escape chars (escapes are not modeled here).
                        let q = rep(min.unwrap_or(0), max, "minLength/maxLength")?;
                        format!("\"[^\"\\\\]{q}\"")
                    }
                }
            }
            "integer" => {
                // inclusive bounds, applying the exclusive forms (+1 / -1 for integers). Both, one-sided, or
                // neither are all expressible (a one-sided bound is infinite but still regular).
                let i64_of = |k: &str| schema.get(k).and_then(|v| v.as_i64());
                let lo = i64_of("minimum").or_else(|| i64_of("exclusiveMinimum").map(|m| m + 1));
                let hi = i64_of("maximum").or_else(|| i64_of("exclusiveMaximum").map(|m| m - 1));
                int_bound_regex(lo, hi)?
            }
            "number" => {
                // Float range bounds are not modeled. Per the documented contract (usage.md), error
                // rather than silently emit an unbounded number that ignores the caller's bounds; use
                // `integer` for a bounded whole number, or omit the bounds.
                if ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
                    .iter()
                    .any(|k| schema.get(k).is_some())
                {
                    bail!(
                        "guided_json: `number` range bounds (minimum/maximum) are not supported; use \
                         `integer` bounds or omit them"
                    );
                }
                "-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?".to_string()
            }
            "boolean" => "(?:true|false)".to_string(),
            "array" => {
                let items = schema
                    .get("items")
                    .context("guided_json: array needs \"items\"")?;
                let item = value_regex(items, ws, root, active)?;
                let (min, max) = (u64_of("minItems").unwrap_or(0), u64_of("maxItems"));
                // [ ws (items, comma-separated, count in [min,max]) ws ]. The inner (,item) group repeats
                // (count-1) times, so its bound is [min-1, max-1] (or [0, max-1] when min==0, the first item
                // being optional). maxItems==0 -> the empty array only.
                if max == Some(0) {
                    format!("\\[{ws}\\]")
                } else if min == 0 {
                    let inner = rep(0, max.map(|m| m - 1), "maxItems")?;
                    format!("\\[{ws}(?:{item}(?:{ws},{ws}{item}){inner})?{ws}\\]")
                } else {
                    let inner = rep(min - 1, max.map(|m| m - 1), "minItems/maxItems")?;
                    format!("\\[{ws}{item}(?:{ws},{ws}{item}){inner}{ws}\\]")
                }
            }
            "object" => object_regex(schema, ws, root, active)?,
            other => bail!("guided_json: unsupported type {other:?}"),
        })
    }
    fn object_regex(
        schema: &serde_json::Value,
        ws: &str,
        root: &serde_json::Value,
        active: &mut Vec<String>,
    ) -> Result<String> {
        let props = schema
            .get("properties")
            .and_then(|p| p.as_object())
            .context("guided_json: object needs \"properties\"")?;
        if props.is_empty() {
            return Ok(format!("\\{{{ws}\\}}"));
        }
        // `required`: listed keys must appear, the rest may be omitted. With no "required" field every
        // property is required (tighter than JSON Schema's "all optional", so a bare `properties`
        // schema pins every field); pass `"required": []` for an all-optional object.
        let required: Option<std::collections::HashSet<&str>> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect());
        // each property: ("key" ws : ws <value>, is_required). Keys are in serde_json's sorted order.
        let mut entries: Vec<(String, bool)> = Vec::with_capacity(props.len());
        for (key, sub) in props {
            let part = format!(
                "\"{}\"{ws}:{ws}{}",
                esc(key),
                value_regex(sub, ws, root, active)?
            );
            let req = required
                .as_ref()
                .is_none_or(|set| set.contains(key.as_str()));
            entries.push((part, req));
        }
        // A conforming object is any in-order subset of the properties that includes every required
        // one. Only the first present property has no leading comma, so enumerate by which property is
        // first-present: `first` is valid iff every property before it is optional. Each later property
        // carries its own leading comma (mandatory if required, optional otherwise), keeping commas
        // well-formed for any subset. All-required reduces to the single `p0,p1,...` alternative.
        let n = entries.len();
        let sep = format!("{ws},{ws}");
        let any_required = entries.iter().any(|(_, r)| *r);
        let mut alts: Vec<String> = Vec::new();
        for first in 0..n {
            if entries[..first].iter().any(|(_, r)| *r) {
                break; // a required property before `first` cannot be skipped.
            }
            let mut s = entries[first].0.clone();
            for (part, req) in &entries[first + 1..] {
                if *req {
                    s.push_str(&format!("{sep}{part}"));
                } else {
                    s.push_str(&format!("(?:{sep}{part})?"));
                }
            }
            alts.push(s);
        }
        if !any_required {
            alts.push(String::new()); // the empty object {} (every property optional).
        }
        Ok(format!("\\{{{ws}(?:{}){ws}\\}}", alts.join("|")))
    }
    value_regex(schema, WS, schema, &mut Vec::new())
}

#[cfg(test)]
mod guided_byte_table_cache {
    use crate::core::runner::Runner;
    use std::sync::Arc;

    #[test]
    fn byte_table_is_built_once_and_reused() {
        // The per-token byte table is cached on the Runner: repeated builds return the same Arc.
        // Skipped if the tokenizer model is absent.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b"))
        else {
            return;
        };
        let r = Runner::load(&dir).expect("load qwen2.5-0.5b");
        let a = r.token_byte_table();
        let b = r.token_byte_table();
        assert!(
            Arc::ptr_eq(&a, &b),
            "the byte table must be cached (same Arc)"
        );
        assert_eq!(
            a.len(),
            r.tokenizer.get_vocab_size(true),
            "one byte entry per vocab id"
        );
        // a guided constraint builds and shares that cached table.
        let cons = r
            .build_regex_constraint("[0-9]{3}")
            .expect("build a guided_regex constraint");
        drop(cons);
        // and the table is still the same Arc afterwards.
        let c = r.token_byte_table();
        assert!(
            Arc::ptr_eq(&a, &c),
            "still the same cached Arc after a build"
        );
    }

    #[test]
    fn guided_dfa_is_cached_per_pattern_and_bounded() {
        // The compiled DFA is cached by pattern: same pattern returns the same Arc, a different pattern a
        // different one, and the cache stays bounded. Skipped if the tokenizer model is absent.
        let Some(dir) = poot_test_util::model_path(poot_test_util::checkpoint!("qwen2.5-0.5b"))
        else {
            return;
        };
        let r = Runner::load(&dir).expect("load qwen2.5-0.5b");
        let d1 = r.guided_dfa("[0-9]{3}").unwrap();
        let d2 = r.guided_dfa("[0-9]{3}").unwrap();
        assert!(
            Arc::ptr_eq(&d1, &d2),
            "same pattern must reuse the cached DFA"
        );
        let other = r.guided_dfa("[a-z]+").unwrap();
        assert!(
            !Arc::ptr_eq(&d1, &other),
            "a different pattern is a distinct DFA"
        );
        // many distinct patterns must not grow the cache past its cap.
        for i in 0..200 {
            let _ = r.guided_dfa(&format!("lit{i}")).unwrap();
        }
        let len = r.guided_dfa_cache.lock().unwrap().len();
        assert!(len <= 64, "cache must stay bounded, got {len}");
    }
}

#[cfg(test)]
mod guided_mask_fuzz {
    use super::Constraint;
    use regex_automata::dfa::{StartKind, dense};
    use regex_automata::meta::Regex as MetaRegex;
    use regex_automata::{Anchored, Input};
    use std::collections::HashSet;
    use std::sync::Arc;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n.max(1)
        }
    }

    // a random regex over {a,b,c} from bounded constructs only (no `*`/`+`), so its language is finite
    // and enumerable by brute force (the independent oracle).
    fn gen_regex(rng: &mut Rng, depth: u32) -> String {
        const CH: &[u8] = b"abc";
        if depth == 0 {
            return match rng.below(2) {
                0 => (CH[rng.below(3) as usize] as char).to_string(),
                _ => {
                    // a 2-char class like [ab]
                    let (a, b) = (rng.below(3), rng.below(3));
                    format!("[{}{}]", CH[a as usize] as char, CH[b as usize] as char)
                }
            };
        }
        match rng.below(4) {
            0 => format!("{}{}", gen_regex(rng, depth - 1), gen_regex(rng, depth - 1)),
            1 => format!(
                "(?:{}|{})",
                gen_regex(rng, depth - 1),
                gen_regex(rng, depth - 1)
            ),
            2 => {
                let m = rng.below(2);
                let n = m + rng.below(2); // {0,0}, {0,1}, {1,1}, {1,2}
                format!("(?:{}){{{m},{n}}}", gen_regex(rng, depth - 1))
            }
            _ => gen_regex(rng, depth - 1),
        }
    }

    // all strings over {a,b,c} of length <= max_len that fully match `re`. Returns None if a match of the
    // maximum length is found (the language may extend past the window, making the oracle unsound).
    fn enumerate(re: &MetaRegex, max_len: usize) -> Option<Vec<Vec<u8>>> {
        const CH: &[u8] = b"abc";
        let mut matches = Vec::new();
        let mut frontier: Vec<Vec<u8>> = vec![vec![]];
        for len in 0..=max_len {
            let mut next = Vec::new();
            for s in &frontier {
                if re.is_match(s.as_slice()) {
                    if len == max_len {
                        return None; // possibly-truncated language; skip
                    }
                    matches.push(s.clone());
                }
                for &c in CH {
                    let mut t = s.clone();
                    t.push(c);
                    next.push(t);
                }
            }
            frontier = next;
        }
        Some(matches)
    }

    #[test]
    fn token_mask_allows_exactly_the_extendable_tokens() {
        // At every step the regex constraint allows a token iff appending its bytes keeps the output a
        // prefix of some full match, and allows EOS iff the output is a full match. Checked against a
        // brute-force language-enumeration oracle over random bounded regexes and a multi-byte token table.
        const MAX_LEN: usize = 6;
        // token id -> bytes: single chars plus multi-char tokens; the last id is EOS (empty bytes,
        // handled via the eos flag).
        let toks: Vec<&[u8]> = vec![b"a", b"b", b"c", b"aa", b"ab", b"bc", b"cc", b"abc", b""];
        let eos = (toks.len() - 1) as u32;
        let bytes: Arc<Vec<Vec<u8>>> = Arc::new(toks.iter().map(|t| t.to_vec()).collect());

        let mut checked = 0usize;
        for seed in 0..400u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1);
            let r = gen_regex(&mut rng, 3);
            let meta = match MetaRegex::new(&format!(r"\A(?:{r})\z")) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let Some(matches) = enumerate(&meta, MAX_LEN) else {
                continue; // language may extend past the window
            };
            let match_set: HashSet<Vec<u8>> = matches.iter().cloned().collect();

            // the DFA-under-test, built exactly as Runner::build_regex_constraint does.
            let whole = format!("(?:{r})$");
            let dfa = match dense::Builder::new()
                .configure(dense::Config::new().start_kind(StartKind::Anchored))
                .build(&whole)
            {
                Ok(d) => Arc::new(d),
                Err(_) => continue,
            };
            use regex_automata::dfa::Automaton;
            let start = dfa
                .start_state_forward(&Input::new("").anchored(Anchored::Yes))
                .unwrap();
            let mut cons = Constraint::regex(dfa, start, bytes.clone(), eos);

            // walk: at each live prefix, compare the constraint's mask to the oracle, then take a step.
            let mut prefix: Vec<u8> = Vec::new();
            for _ in 0..(MAX_LEN + 2) {
                let (allowed_c, eos_c) = cons.allowed_next();
                // oracle: token allowed iff prefix+bytes is a prefix of some match; eos iff prefix is a match.
                let mut allowed_o = HashSet::new();
                for (id, tb) in toks.iter().enumerate() {
                    if tb.is_empty() {
                        continue;
                    }
                    let mut pb = prefix.clone();
                    pb.extend_from_slice(tb);
                    if matches.iter().any(|m| m.starts_with(&pb)) {
                        allowed_o.insert(id as u32);
                    }
                }
                let eos_o = match_set.contains(&prefix);
                assert_eq!(
                    allowed_c, allowed_o,
                    "seed {seed} regex {r:?} prefix {prefix:?}: token mask mismatch"
                );
                assert_eq!(
                    eos_c, eos_o,
                    "seed {seed} regex {r:?} prefix {prefix:?}: eos flag mismatch"
                );
                if allowed_c.is_empty() {
                    break; // dead-end or only-eos; both sides agree
                }
                // step by a random allowed token.
                let pick = *allowed_c
                    .iter()
                    .nth(rng.below(allowed_c.len() as u64) as usize)
                    .unwrap();
                prefix.extend_from_slice(toks[pick as usize]);
                cons.advance(pick);
            }
            checked += 1;
        }
        assert!(checked > 200, "too many regexes skipped ({checked}/400)");
    }

    // ---- json_object pushdown acceptor (Constraint::JsonValue) ----

    use super::JsonAcceptor;

    /// Feed a byte string into a fresh acceptor; return (all_bytes_accepted, complete_at_end).
    fn drive(s: &[u8]) -> (bool, bool) {
        let mut a = JsonAcceptor::new();
        for &b in s {
            if !a.step(b) {
                return (false, false);
            }
        }
        (true, a.is_complete())
    }

    /// The index of the first byte rejected, or None if the whole string is accepted.
    fn first_reject(s: &[u8]) -> Option<usize> {
        let mut a = JsonAcceptor::new();
        for (i, &b) in s.iter().enumerate() {
            if !a.step(b) {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn json_acceptor_accepts_valid_values_and_completes_only_at_the_end() {
        // Each is a JSON value with no complete proper prefix (a container/string closed only by the
        // final byte), so is_complete() is false at every proper prefix and true exactly at the end.
        // serde_json cross-checks that each is valid JSON.
        let valid = [
            r#"{}"#,
            r#"[]"#,
            r#""hi""#,
            r#"{"a":1}"#,
            r#"[1,2,3]"#,
            r#"{"a":[{"b":[1,2]},{"c":{"d":true}}]}"#,
            r#"{"k":"v\né\"end","n":-12.5e+3,"b":false,"z":null,"arr":[[],[1],[[2]]]}"#,
            "{ \"a\" : 1 , \"b\" : [ true , null ] }", // insignificant whitespace everywhere
        ];
        for v in valid {
            serde_json::from_str::<serde_json::Value>(v)
                .unwrap_or_else(|e| panic!("test battery item is not valid JSON: {v:?}: {e}"));
            let bytes = v.as_bytes();
            for cut in 1..bytes.len() {
                let (ok, done) = drive(&bytes[..cut]);
                assert!(ok, "valid prefix rejected: {:?}", &v[..cut]);
                assert!(!done, "completed at a proper prefix: {:?}", &v[..cut]);
            }
            let (ok, done) = drive(bytes);
            assert!(ok && done, "full valid value not accepted+complete: {v:?}");
            // trailing whitespace after a complete value is fine and stays complete.
            let mut with_ws = v.to_string();
            with_ws.push_str("  \n\t");
            let (ok2, done2) = drive(with_ws.as_bytes());
            assert!(ok2 && done2, "trailing ws broke completeness: {v:?}");
            // a second value after a complete one is rejected (exactly one value).
            let two = format!("{v} {v}");
            assert!(
                first_reject(two.as_bytes()).is_some(),
                "accepted two values: {v:?}"
            );
        }
    }

    #[test]
    fn json_acceptor_counts_brackets() {
        // A value is complete only when every opened bracket has been closed.
        assert_eq!(drive(br#"{"a":1"#), (true, false)); // object still open -> not complete
        assert_eq!(drive(br#"{"a":1}"#), (true, true)); // closed -> complete
        assert_eq!(drive(br#"[[]"#), (true, false)); // one array still open
        assert_eq!(drive(br#"[[]]"#), (true, true)); // both closed
        assert_eq!(drive(br#"[1,[2,[3"#), (true, false)); // three arrays open
        assert_eq!(drive(br#"[1,[2,[3]]]"#), (true, true)); // all closed
        // an unmatched or extra close is rejected at that byte.
        assert_eq!(first_reject(b"}"), Some(0));
        assert_eq!(first_reject(b"]"), Some(0));
        assert_eq!(first_reject(br#"{}}"#), Some(2)); // extra `}` after the object closed
        assert_eq!(first_reject(br#"[]]"#), Some(2));
    }

    #[test]
    fn json_acceptor_handles_arbitrary_depth() {
        // No fixed depth cap: 200 levels of nesting stays alive and completes when unwound.
        const D: usize = 200;
        let mut a = JsonAcceptor::new();
        for _ in 0..D {
            assert!(a.step(b'['));
        }
        assert!(!a.is_complete(), "unclosed deep nest must not be complete");
        for _ in 0..D {
            assert!(a.step(b']'));
        }
        assert!(a.is_complete(), "fully closed deep nest must be complete");
        // the same for objects: {"a":{"a":{...}}}.
        let mut s = String::new();
        for _ in 0..50 {
            s.push_str(r#"{"a":"#);
        }
        s.push('1');
        for _ in 0..50 {
            s.push('}');
        }
        assert_eq!(drive(s.as_bytes()), (true, true));
    }

    #[test]
    fn json_acceptor_rejects_malformed_at_the_offending_byte() {
        assert_eq!(first_reject(b"01"), Some(1)); // leading zero: no digit may follow a lone 0
        assert_eq!(first_reject(br#"[1,]"#), Some(3)); // trailing comma in array
        assert_eq!(first_reject(br#"{"a":1,}"#), Some(7)); // trailing comma in object
        assert_eq!(first_reject(br#"{a:1}"#), Some(1)); // unquoted key
        assert_eq!(first_reject(br#""\x""#), Some(2)); // bad string escape
        assert_eq!(first_reject(br#"{"a" 1}"#), Some(5)); // missing colon
        assert_eq!(first_reject(b"tru3"), Some(3)); // broken literal
        assert_eq!(first_reject(b"nul"), None); // a prefix of `null` is not yet rejected...
        assert!(!drive(b"nul").1, "...but it is not complete either");
        // a bare `-` or `1.` needs more: accepted so far, but not complete.
        assert_eq!(drive(b"-"), (true, false));
        assert_eq!(drive(b"1."), (true, false));
        assert_eq!(drive(b"1e"), (true, false));
        // a bare number IS a complete top-level value.
        assert_eq!(drive(b"42"), (true, true));
        assert_eq!(drive(b"-0.5e10"), (true, true));
    }

    #[test]
    fn json_value_constraint_mask_matches_acceptor_and_serde_eos() {
        // Wrap the acceptor in Constraint::JsonValue over a single-byte-per-token table and walk several
        // target values, checking at every prefix that the token mask equals the raw acceptor's per-byte
        // verdict and the EOS flag equals serde_json's verdict on the emitted prefix.
        let toks: Vec<Vec<u8>> = (0u32..128).map(|b| vec![b as u8]).collect();
        let table = Arc::new(toks.clone());
        let eos = 200u32;
        let targets = [
            r#"{"a":[1,-2.5,true,null],"b":{"c":"dA"}}"#,
            r#"[[[[]]]]"#,
            r#""just a string""#,
            r#"12345"#,
        ];
        for target in targets {
            let mut cons = Constraint::json_value(table.clone(), eos);
            let tb = target.as_bytes();
            for cut in 0..tb.len() {
                let prefix = &tb[..cut];
                let (allowed, eos_ok) = cons.allowed_next();
                // independent EOS oracle: the prefix is a complete value iff serde_json parses it.
                let serde_ok = serde_json::from_slice::<serde_json::Value>(prefix).is_ok();
                assert_eq!(
                    eos_ok,
                    serde_ok,
                    "target {target:?} prefix {:?}: eos flag vs serde_json",
                    std::str::from_utf8(prefix).unwrap()
                );
                // the byte we are about to emit must be in the mask.
                let next = tb[cut];
                assert!(
                    allowed.contains(&(next as u32)),
                    "target {target:?} prefix {:?}: next byte {:?} not allowed",
                    std::str::from_utf8(prefix).unwrap(),
                    next as char
                );
                // the mask must equal the raw acceptor's verdict for every candidate byte.
                for (id, one) in toks.iter().enumerate().take(128) {
                    let mut probe = {
                        let mut a = JsonAcceptor::new();
                        for &pb in prefix {
                            a.step(pb);
                        }
                        a
                    };
                    let raw_ok = probe.step(one[0]);
                    assert_eq!(
                        allowed.contains(&(id as u32)),
                        raw_ok,
                        "target {target:?} prefix {:?}: mask/raw disagree on byte {:?}",
                        std::str::from_utf8(prefix).unwrap(),
                        one[0] as char
                    );
                }
                cons.advance(next as u32);
            }
            // after the whole value, EOS is allowed and serde agrees.
            let (_allowed, eos_ok) = cons.allowed_next();
            assert!(eos_ok, "target {target:?}: EOS not allowed at end");
            assert!(serde_json::from_slice::<serde_json::Value>(tb).is_ok());
        }
    }

    #[test]
    fn grammar_constraint_masks_tokens_through_the_byte_table() {
        // Wrap a GBNF grammar in Constraint::Grammar over a single-byte-per-token table and walk a valid
        // sentence, checking the mask permits only grammar-legal characters and EOS toggles exactly at
        // complete sentences. Grammar: balanced parens of any depth around "x".
        let toks: Vec<Vec<u8>> = (0u32..128).map(|b| vec![b as u8]).collect();
        let table = Arc::new(toks);
        let eos = 200u32;
        let g = std::sync::Arc::new(
            super::grammar::Grammar::parse(r#"root ::= "(" root ")" | "x""#).unwrap(),
        );

        // walk "((x))": at the start only '(' or 'x' are legal; EOS is never legal mid-parse; at the end EOS
        // is legal and no further char keeps a complete parse re-closable.
        let mut cons = Constraint::grammar(g.clone(), table.clone(), eos);
        let (allowed, eos_ok) = cons.allowed_next();
        assert!(allowed.contains(&(b'(' as u32)) && allowed.contains(&(b'x' as u32)));
        assert!(!allowed.contains(&(b')' as u32)) && !allowed.contains(&(b'y' as u32)));
        assert!(!eos_ok, "empty output is not a complete sentence");

        for &b in b"((x))" {
            let (allowed, _) = cons.allowed_next();
            assert!(
                allowed.contains(&(b as u32)),
                "byte {:?} blocked mid-parse",
                b as char
            );
            cons.advance(b as u32);
        }
        let (_allowed, eos_ok) = cons.allowed_next();
        assert!(eos_ok, "a balanced sentence must permit EOS");

        // a shorter unbalanced prefix must NOT permit EOS.
        let mut c2 = Constraint::grammar(g, table, eos);
        for &b in b"((x)" {
            c2.advance(b as u32);
        }
        let (_a, eos_ok) = c2.allowed_next();
        assert!(!eos_ok, "an unbalanced prefix must not permit EOS");
    }

    #[test]
    fn tool_call_regex_forces_a_valid_schema_conforming_call() {
        // The forced tool-call regex accepts exactly a ChatML <tool_call> block with the forced name and
        // schema-conforming arguments, and rejects a wrong name, a missing required arg, or plain text.
        // Compiled as an anchored whole-string match, as build_regex_constraint does.
        let params = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        });
        let forced = vec![("get_weather".to_string(), params)];
        let pat = super::tool_call_regex(&forced).unwrap();
        let re = MetaRegex::new(&format!("^(?:{pat})$")).unwrap();

        // json_schema_to_regex emits object keys in sorted order (arguments before name).
        assert!(re.is_match(
            r#"<tool_call>{"arguments":{"city":"NYC"},"name":"get_weather"}</tool_call>"#
        ));
        // surrounding whitespace inside the markup is allowed (the parser trims it).
        assert!(re.is_match(
            "<tool_call>\n{\"arguments\":{\"city\":\"x\"},\"name\":\"get_weather\"}\n</tool_call>"
        ));
        // wrong function name.
        assert!(
            !re.is_match(r#"<tool_call>{"arguments":{"city":"NYC"},"name":"other"}</tool_call>"#)
        );
        // missing required argument.
        assert!(!re.is_match(r#"<tool_call>{"arguments":{},"name":"get_weather"}</tool_call>"#));
        // not wrapped as a tool call.
        assert!(!re.is_match(r#"{"name":"get_weather","arguments":{"city":"NYC"}}"#));

        // "required" over multiple tools becomes an alternation accepting either.
        let two = vec![
            (
                "a".to_string(),
                serde_json::json!({"type":"object","properties":{},"required":[]}),
            ),
            (
                "b".to_string(),
                serde_json::json!({"type":"object","properties":{},"required":[]}),
            ),
        ];
        let pat2 = super::tool_call_regex(&two).unwrap();
        let re2 = MetaRegex::new(&format!("^(?:{pat2})$")).unwrap();
        assert!(re2.is_match(r#"<tool_call>{"arguments":{},"name":"a"}</tool_call>"#));
        assert!(re2.is_match(r#"<tool_call>{"arguments":{},"name":"b"}</tool_call>"#));
        assert!(!re2.is_match(r#"<tool_call>{"arguments":{},"name":"c"}</tool_call>"#));

        assert!(super::tool_call_regex(&[]).is_err());
    }
}
