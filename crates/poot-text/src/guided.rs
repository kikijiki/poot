//! Guided decoding: Constraint (choice/regex/JSON-schema/grammar/tool-call) and token masks.

use crate::error::{OptionExt, Result};
use crate::grammar;
use crate::tokenize::TextCodec;

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
    /// (it clears past a cap). The build is size-bounded ([`DFA_SIZE_LIMIT`] and friends): a pattern
    /// whose automaton would exceed the limits is refused rather than built, so one request cannot
    /// exhaust the server.
    fn guided_dfa(
        &self,
        pattern: &str,
    ) -> Result<std::sync::Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>> {
        use regex_automata::dfa::{StartKind, dense};
        use regex_automata::nfa::thompson;
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
                .configure(
                    dense::Config::new()
                        .start_kind(StartKind::Anchored)
                        .dfa_size_limit(Some(DFA_SIZE_LIMIT))
                        .determinize_size_limit(Some(DETERMINIZE_SIZE_LIMIT)),
                )
                .thompson(thompson::Config::new().nfa_size_limit(Some(NFA_SIZE_LIMIT)))
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

    /// The per-token surface byte table (token id -> the bytes the token contributes to the output),
    /// built once from the vocabulary's token pieces and cached (~50ms for a 150k vocab). See
    /// [`TextCodec::token_bytes`].
    fn token_byte_table(&self) -> std::sync::Arc<Vec<Vec<u8>>> {
        self.guided_byte_table
            .get_or_init(|| {
                let kind = PieceKind::of(self.tokenizer.get_decoder());
                let added = self.tokenizer.get_added_tokens_decoder();
                let vocab = self.tokenizer.get_vocab_size(true);
                let reverse = (kind == PieceKind::ByteLevel).then(byte_level_reverse);
                let bytes = (0..vocab as u32)
                    .map(|id| {
                        let Some(piece) = self.tokenizer.id_to_token(id) else {
                            return Vec::new();
                        };
                        if added.contains_key(&id) {
                            return piece.into_bytes(); // an added token is its literal text
                        }
                        match (kind, &reverse) {
                            (PieceKind::ByteLevel, Some(reverse)) => {
                                byte_level_surface(&piece, reverse)
                            }
                            (PieceKind::SentencePiece, _) => sentencepiece_surface(&piece),
                            (PieceKind::WordPiece, _) => piece
                                .strip_prefix("##")
                                .unwrap_or(&piece)
                                .as_bytes()
                                .to_vec(),
                            _ => piece.into_bytes(),
                        }
                    })
                    .collect();
                std::sync::Arc::new(bytes)
            })
            .clone()
    }

    /// The bytes token `id` contributes to the output text, read from the vocabulary's token piece, not
    /// from decoding the token alone: a word-initial SentencePiece token (`\u{2581}the`) keeps its
    /// leading space (0x20) where an isolated decode strips it, and a byte-fallback token (`<0x41>`) is
    /// its one byte. Guided masks and logprob `bytes` both read this. Empty for an id outside the
    /// vocabulary.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.token_byte_table()
            .get(id as usize)
            .cloned()
            .unwrap_or_default()
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

/// Largest compiled DFA (bytes) a client pattern may build. A pattern past it is refused with a 400.
const DFA_SIZE_LIMIT: usize = 1 << 20;
/// Largest working set (bytes) the DFA determinization may use before it gives up.
const DETERMINIZE_SIZE_LIMIT: usize = 1 << 20;
/// Largest Thompson NFA (bytes) a client pattern may build, which bounds the work before determinization.
const NFA_SIZE_LIMIT: usize = 1 << 20;
/// Most per-state token masks one request keeps ([`StateMasks`]); the cache clears past it.
const STATE_MASK_CAP: usize = 1024;

/// How a tokenizer's vocabulary pieces spell their bytes, read from its decoder: the decoder is what
/// turns pieces back into text, so it names the spelling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PieceKind {
    /// GPT-2 byte-level BPE: each piece character stands for one byte.
    ByteLevel,
    /// SentencePiece (Metaspace or ByteFallback decoder): U+2581 is a space, `<0xNN>` is a byte.
    SentencePiece,
    /// WordPiece: a `##` prefix marks a continuation and is not text.
    WordPiece,
    /// No decoder, or one that rewrites pieces in a way poot does not model (CTC, BPEDecoder): the piece
    /// is taken as its own text. Such tokenizers are unsupported for exact surface bytes.
    Verbatim,
}

impl PieceKind {
    fn of(decoder: Option<&tokenizers::DecoderWrapper>) -> Self {
        use tokenizers::DecoderWrapper;
        match decoder {
            Some(DecoderWrapper::ByteLevel(_)) => Self::ByteLevel,
            Some(DecoderWrapper::Metaspace(_) | DecoderWrapper::ByteFallback(_)) => {
                Self::SentencePiece
            }
            Some(DecoderWrapper::WordPiece(_)) => Self::WordPiece,
            Some(DecoderWrapper::Sequence(seq)) => {
                // the most specific member names the spelling: byte-level, then SentencePiece, then WordPiece.
                let kinds: Vec<Self> = seq
                    .get_decoders()
                    .iter()
                    .map(|d| Self::of(Some(d)))
                    .collect();
                [Self::ByteLevel, Self::SentencePiece, Self::WordPiece]
                    .into_iter()
                    .find(|k| kinds.contains(k))
                    .unwrap_or(Self::Verbatim)
            }
            _ => Self::Verbatim,
        }
    }
}

/// The inverse of GPT-2's `bytes_to_unicode`: the byte each byte-level BPE piece character stands for.
/// Printable Latin-1 bytes are themselves; the other bytes took the code points from U+0100 up, in order.
fn byte_level_reverse() -> std::collections::HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut next = 0x100u32;
    let mut reverse = std::collections::HashMap::with_capacity(256);
    for b in 0..=255u8 {
        let c = if printable(b) {
            char::from(b)
        } else {
            let c = char::from_u32(next).expect("U+0100..U+0143 are scalar values");
            next += 1;
            c
        };
        reverse.insert(c, b);
    }
    reverse
}

/// The bytes of a byte-level BPE piece: each character is the byte it encodes (a character outside the
/// alphabet is its own UTF-8).
fn byte_level_surface(piece: &str, reverse: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(piece.len());
    for c in piece.chars() {
        match reverse.get(&c) {
            Some(&b) => bytes.push(b),
            None => bytes.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
        }
    }
    bytes
}

/// The bytes of a SentencePiece-style piece: the space marker U+2581 is a space (0x20), and a byte-fallback
/// piece `<0xNN>` is the byte NN. Any other piece is its own text.
fn sentencepiece_surface(piece: &str) -> Vec<u8> {
    if let Some(byte) = piece
        .strip_prefix("<0x")
        .and_then(|hex| hex.strip_suffix('>'))
        .filter(|hex| hex.len() == 2)
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
    {
        return vec![byte];
    }
    piece.replace('\u{2581}', " ").into_bytes()
}

/// One DFA state's allowed tokens as a bitmask over the vocabulary.
struct TokenMask {
    words: Vec<u64>,
}

impl TokenMask {
    fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().enumerate().flat_map(|(w, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 {
                    return None;
                }
                let bit = rest.trailing_zeros();
                rest &= rest - 1;
                Some(w as u32 * 64 + bit)
            })
        })
    }
}

/// The allowed-token mask of each DFA state a request has visited. A state's mask is computed by walking
/// the whole vocabulary once; every later visit reads the cached bitmask, so a step costs the allowed
/// count, not the vocabulary. Shared by clones of one constraint.
#[derive(Default)]
pub struct StateMasks {
    by_state: std::sync::Mutex<
        std::collections::HashMap<
            regex_automata::util::primitives::StateID,
            std::sync::Arc<TokenMask>,
        >,
    >,
    /// Token byte-walks done computing masks: grows by the vocabulary size per state computed, never on a
    /// cache hit.
    walked: std::sync::atomic::AtomicUsize,
}

impl StateMasks {
    fn mask(
        &self,
        dfa: &regex_automata::dfa::dense::DFA<Vec<u32>>,
        state: regex_automata::util::primitives::StateID,
        bytes: &[Vec<u8>],
    ) -> std::sync::Arc<TokenMask> {
        use regex_automata::dfa::Automaton;
        use std::sync::atomic::Ordering;
        if let Some(mask) = self.by_state.lock().unwrap().get(&state) {
            return mask.clone();
        }
        let mut words = vec![0u64; bytes.len().div_ceil(64)];
        for (id, tok_bytes) in bytes.iter().enumerate() {
            if tok_bytes.is_empty() {
                continue;
            }
            let mut s = state;
            let alive = tok_bytes.iter().all(|&b| {
                s = dfa.next_state(s, b);
                !(dfa.is_dead_state(s) || dfa.is_quit_state(s))
            });
            if alive {
                words[id / 64] |= 1 << (id % 64);
            }
        }
        self.walked.fetch_add(bytes.len(), Ordering::Relaxed);
        let mask = std::sync::Arc::new(TokenMask { words });
        let mut cache = self.by_state.lock().unwrap();
        if cache.len() >= STATE_MASK_CAP {
            cache.clear(); // bounded: drop the cold set rather than grow without limit
        }
        cache.insert(state, mask.clone());
        mask
    }
}

/// A guided-decoding constraint that masks each decode step to the tokens it permits. Stateful: it
/// advances as each generated token is observed.
///
/// - `Choices`: the completion must be one of a fixed set of strings (token-level). Built by
///   [`TextCodec::build_choice_constraint`].
/// - `Regex`: the completion must match a regex (byte-level DFA; a token is allowed if its bytes avoid
///   the dead state, and EOS is allowed in a match state). Built by [`TextCodec::build_regex_constraint`].
#[derive(Clone)]
pub enum Constraint {
    /// The completion is one of a fixed set of token sequences.
    Choices {
        /// the allowed choices, each as its token-id sequence (no BOS).
        choices: Vec<Vec<u32>>,
        /// the generated tokens so far (advances as tokens are observed).
        generated: Vec<u32>,
        /// the model's EOS id, permitted once the generated tokens complete a choice.
        eos: u32,
    },
    /// The completion matches a regex, tracked by a byte-level DFA.
    Regex {
        /// the compiled byte-level DFA of the pattern (shared across requests with the same pattern).
        dfa: std::sync::Arc<regex_automata::dfa::dense::DFA<Vec<u32>>>,
        /// the current DFA state (advances as tokens are observed).
        state: regex_automata::util::primitives::StateID,
        /// per-token surface bytes, indexed by token id (see [`TextCodec::token_bytes`]).
        bytes: std::sync::Arc<Vec<Vec<u8>>>,
        /// the allowed-token bitmask of every state this constraint has visited.
        masks: std::sync::Arc<StateMasks>,
        /// the model's EOS id, permitted in a match state.
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
        /// the model's EOS id, permitted once a complete top-level value is held.
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
        /// the model's EOS id, permitted once some parse is complete.
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
            masks: std::sync::Arc::default(),
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
    pub fn allowed_next(&self) -> (std::collections::HashSet<u32>, bool) {
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
                dfa,
                state,
                bytes,
                masks,
                ..
            } => {
                // a token is allowed if its bytes walk the DFA from `state` without hitting a dead/quit
                // state (the state's cached mask). EOS is allowed when the current state matches (the
                // output so far is a full match).
                allowed.extend(masks.mask(dfa, *state, bytes).ids());
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
    pub fn advance(&mut self, token: u32) {
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

    /// The model's EOS id this constraint permits once the completion is complete.
    pub fn eos(&self) -> u32 {
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
pub(crate) fn tool_call_regex(forced: &[(String, serde_json::Value)]) -> Result<String> {
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
pub(crate) fn json_schema_to_regex(schema: &serde_json::Value) -> Result<String> {
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
    use crate::chat_format::ChatFormat;
    use crate::tokenize::{ChatTemplate, TextCodec};
    use std::sync::Arc;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;

    /// A codec over a small word-level vocabulary: the caches under test depend on the tokenizer's size,
    /// not on a checkpoint.
    fn codec() -> TextCodec {
        let vocab = (0..64u32)
            .map(|i| (format!("w{i}"), i))
            .chain([("<unk>".to_string(), 64u32)])
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel model");
        TextCodec::new(
            Tokenizer::new(model),
            None,
            None,
            0,
            ChatFormat::ChatML,
            ChatTemplate::default(),
        )
    }

    #[test]
    fn byte_table_is_built_once_and_reused() {
        // The per-token byte table is cached on the codec: repeated builds return the same Arc.
        let r = codec();
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
        // different one, and the cache stays bounded.
        let r = codec();
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

            // the DFA-under-test, built exactly as TextCodec::build_regex_constraint does.
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

#[cfg(test)]
mod schema_regex_tests {
    use super::{int_bound_regex, json_schema_to_regex};

    #[test]
    fn json_schema_to_regex_accepts_and_rejects() {
        // full-match helper via regex-automata's meta engine (\A..\z = whole string).
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "active": {"type": "boolean"},
                "kind": {"enum": ["a", "b"]}
            }
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        // keys are emitted in sorted order (serde_json sorts object keys): active, age, kind, name, tags.
        // conforming (compact and with whitespace) -> match.
        assert!(full_match(
            &rx,
            r#"{"active":true,"age":3,"kind":"a","name":"bob","tags":["x","y"]}"#
        ));
        assert!(full_match(
            &rx,
            "{ \"active\": false, \"age\": 3, \"kind\": \"b\", \"name\": \"bob\", \"tags\": [] }"
        ));
        // non-conforming -> no match.
        assert!(
            !full_match(&rx, r#"{"active":true,"age":3}"#),
            "missing properties"
        );
        assert!(
            !full_match(
                &rx,
                r#"{"active":true,"age":3,"kind":"a","name":3,"tags":[]}"#
            ),
            "name must be a string"
        );
        assert!(
            !full_match(
                &rx,
                r#"{"active":true,"age":3,"kind":"c","name":"b","tags":[]}"#
            ),
            "kind must be one of the enum"
        );
        // an unsupported construct errors rather than mis-constraining.
        assert!(json_schema_to_regex(&serde_json::json!({"type": "null"})).is_err());
    }

    #[test]
    fn json_schema_to_regex_optional_properties_unions_const_and_ref() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }

        // Optional properties: `required` selects which keys must appear (the rest are any in-order subset).
        // Exercises object_regex's alternation-by-first-present-property and comma placement, which the
        // all-required test never hits and which emit malformed JSON (stray/double commas, a skippable
        // required field) if wrong. Props sorted: a, b, c; only b is required.
        let rx = json_schema_to_regex(&serde_json::json!({
            "type":"object",
            "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"}},
            "required":["b"]
        }))
        .unwrap();
        for s in [
            r#"{"b":1}"#,
            r#"{"b":1,"c":2}"#,
            r#"{"a":0,"b":1}"#,
            r#"{"a":0,"b":1,"c":2}"#,
        ] {
            assert!(full_match(&rx, s), "required-subset should accept {s}");
        }
        for s in [
            r#"{}"#,            // b (required) missing
            r#"{"a":0}"#,       // b missing
            r#"{"a":0,"c":2}"#, // b missing
            r#"{"c":2,"b":1}"#, // wrong (non-sorted) order
            r#"{"b":1,}"#,      // trailing comma
        ] {
            assert!(!full_match(&rx, s), "required-subset should reject {s}");
        }

        // ALL-OPTIONAL object (`required: []`) accepts the empty object and any in-order subset, and never a
        // leading/trailing comma.
        let rx = json_schema_to_regex(&serde_json::json!({
            "type":"object",
            "properties":{"a":{"type":"integer"},"b":{"type":"integer"}},
            "required":[]
        }))
        .unwrap();
        for s in [r#"{}"#, r#"{"a":1}"#, r#"{"b":2}"#, r#"{"a":1,"b":2}"#] {
            assert!(full_match(&rx, s), "all-optional should accept {s}");
        }
        for s in [r#"{"b":2,"a":1}"#, r#"{"a":1,}"#, r#"{,"a":1}"#] {
            assert!(!full_match(&rx, s), "all-optional should reject {s}");
        }

        // oneOf/anyOf: a union of subschemas -> an alternation (oneOf's exactly-one relaxed to at-least-one).
        let rx = json_schema_to_regex(&serde_json::json!({
            "oneOf":[{"type":"integer"},{"type":"boolean"}]
        }))
        .unwrap();
        assert!(full_match(&rx, "42") && full_match(&rx, "true"));
        assert!(!full_match(&rx, "\"x\"") && !full_match(&rx, "1.5"));

        // const: a single fixed scalar value (its escaped canonical JSON serialization).
        let rx = json_schema_to_regex(&serde_json::json!({"const":"hello"})).unwrap();
        assert!(full_match(&rx, "\"hello\"") && !full_match(&rx, "\"world\""));
        let rx = json_schema_to_regex(&serde_json::json!({"const":42})).unwrap();
        assert!(full_match(&rx, "42") && !full_match(&rx, "43"));

        // $ref: a local JSON-Pointer ref is resolved + inlined (here into a bounded integer, so the bound is
        // enforced through the ref). External and recursive refs error rather than loop/mis-constrain.
        let rx = json_schema_to_regex(&serde_json::json!({
            "$defs":{"Id":{"type":"integer","minimum":1}},
            "type":"object",
            "properties":{"id":{"$ref":"#/$defs/Id"}},
            "required":["id"]
        }))
        .unwrap();
        assert!(
            full_match(&rx, r#"{"id":5}"#),
            "ref resolves to the bounded integer"
        );
        assert!(
            !full_match(&rx, r#"{"id":0}"#),
            "ref's minimum:1 is enforced"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref":"http://x/y"})).is_err(),
            "external $ref errors"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({
                "$defs":{"Node":{"type":"object","properties":{"next":{"$ref":"#/$defs/Node"}},"required":["next"]}},
                "$ref":"#/$defs/Node"
            }))
            .is_err(),
            "recursive $ref errors (not expressible as a finite regex)"
        );
    }

    #[test]
    fn json_schema_string_format_constrains_the_content() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // (format, a conforming value, a non-conforming value) - each tested as the bare quoted string.
        let cases = [
            ("uuid", "550e8400-e29b-41d4-a716-446655440000", "not-a-uuid"),
            ("date", "2026-06-24", "2026-13-40"),
            ("date-time", "2026-06-24T14:30:00Z", "2026-06-24 14:30"),
            ("time", "14:30:00", "25:61:00"),
            ("email", "bob@example.com", "bob@@no"),
            ("ipv4", "192.168.0.1", "999.1.1.1"),
        ];
        for (fmt, good, bad) in cases {
            let rx =
                json_schema_to_regex(&serde_json::json!({"type":"string","format":fmt})).unwrap();
            assert!(
                full_match(&rx, &format!("\"{good}\"")),
                "{fmt}: should accept {good}"
            );
            assert!(
                !full_match(&rx, &format!("\"{bad}\"")),
                "{fmt}: should reject {bad}"
            );
        }
        // pattern governs over format when both are present.
        let rx = json_schema_to_regex(
            &serde_json::json!({"type":"string","format":"uuid","pattern":"^x+$"}),
        )
        .unwrap();
        assert!(full_match(&rx, "\"xxx\""), "pattern governs over format");
        // an UNKNOWN format is ignored (advisory) - the string stays generically constrained, not an error.
        let rx = json_schema_to_regex(&serde_json::json!({"type":"string","format":"hostname"}))
            .unwrap();
        assert!(
            full_match(&rx, "\"anything goes\""),
            "unknown format ignored"
        );
    }

    #[test]
    fn int_range_regex_matches_exactly_the_range() {
        let int_range_regex = |lo: i64, hi: i64| int_bound_regex(Some(lo), Some(hi));
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // EXHAUSTIVELY verify the regex accepts exactly the integers in [lo,hi] (the only safe check for a
        // range-to-regex). Small ranges so the brute force over a +-8 margin stays fast.
        let small = [
            (0i64, 9),
            (0, 0),
            (5, 5),
            (1, 12),
            (8, 123),
            (0, 255),
            (100, 150),
            (-5, 5),
            (-128, 127),
            (-99, -10),
            (-1, 1),
            (3, 3),
        ];
        for (lo, hi) in small {
            let rx = int_range_regex(lo, hi).unwrap();
            for v in (lo - 8)..=(hi + 8) {
                let want = v >= lo && v <= hi;
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    "range [{lo},{hi}] on {v}: expected {want} (regex {rx})"
                );
            }
        }
        // a large range: boundary off-by-ones + interior/exterior samples.
        let rx = int_range_regex(0, 1_000_000).unwrap();
        for (v, want) in [
            (-1i64, false),
            (0, true),
            (1, true),
            (500_000, true),
            (999_999, true),
            (1_000_000, true),
            (1_000_001, false),
        ] {
            assert_eq!(full_match(&rx, &v.to_string()), want, "1e6 range on {v}");
        }
        // canonical-form rejects: no leading zeros, digits only, non-empty.
        let rx = int_range_regex(0, 100).unwrap();
        assert!(!full_match(&rx, "007") && !full_match(&rx, "1a") && !full_match(&rx, ""));
        assert!(int_range_regex(5, 2).is_err(), "min > max errors");
    }

    #[test]
    fn int_bound_regex_handles_one_sided_and_open() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // One-sided bounds are infinite but regular. Verify exactly over a window wide enough to span the
        // bound and several digit-count boundaries (1, 2, 3 digits) on both sides.
        let lows = [
            Some(0i64),
            Some(1),
            Some(5),
            Some(10),
            Some(100),
            Some(-7),
            Some(-100),
        ];
        let highs = [
            Some(0i64),
            Some(9),
            Some(12),
            Some(100),
            Some(-1),
            Some(-50),
        ];
        for lo in lows {
            let rx = int_bound_regex(lo, None).unwrap();
            for v in -250i64..=250 {
                let want = v >= lo.unwrap();
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    ">= {lo:?} on {v} (regex {rx})"
                );
            }
        }
        for hi in highs {
            let rx = int_bound_regex(None, hi).unwrap();
            for v in -250i64..=250 {
                let want = v <= hi.unwrap();
                assert_eq!(
                    full_match(&rx, &v.to_string()),
                    want,
                    "<= {hi:?} on {v} (regex {rx})"
                );
            }
        }
        // No bounds: any canonical integer (positive, negative, zero), no leading zeros.
        let rx = int_bound_regex(None, None).unwrap();
        for s in ["0", "-1", "42", "-1000", "999999999999"] {
            assert!(full_match(&rx, s), "unbounded should accept {s}");
        }
        assert!(!full_match(&rx, "007") && !full_match(&rx, "1.5") && !full_match(&rx, ""));
    }

    #[test]
    fn json_schema_to_regex_integer_bounds() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // inclusive minimum/maximum.
        let rx =
            json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":1,"maximum":5}))
                .unwrap();
        for v in 1..=5 {
            assert!(full_match(&rx, &v.to_string()));
        }
        assert!(!full_match(&rx, "0") && !full_match(&rx, "6"));
        // exclusive forms (draft-07 numeric): >0 and <5 -> [1,4].
        let rx = json_schema_to_regex(
            &serde_json::json!({"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":5}),
        )
        .unwrap();
        assert!(full_match(&rx, "1") && full_match(&rx, "4"));
        assert!(!full_match(&rx, "0") && !full_match(&rx, "5"));
        // one-sided `minimum` -> integers >= 3 (an open upper range, now enforced exactly).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":3})).unwrap();
        assert!(
            !full_match(&rx, "1") && !full_match(&rx, "2"),
            "below the minimum"
        );
        assert!(
            full_match(&rx, "3") && full_match(&rx, "1000"),
            "at/above the minimum"
        );
        // one-sided `maximum` -> integers <= 10 (all negatives included, nothing above 10).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"integer","maximum":10})).unwrap();
        assert!(
            full_match(&rx, "10") && full_match(&rx, "-99"),
            "at/below the maximum"
        );
        assert!(!full_match(&rx, "11"), "above the maximum");
    }

    #[test]
    fn json_schema_number_bounds_error_not_silently_ignored() {
        // `number` with NO bounds is fine (an unbounded JSON number).
        assert!(json_schema_to_regex(&serde_json::json!({"type":"number"})).is_ok());
        // But range bounds on a `number` must ERROR (the documented contract - float-range regex is not
        // modeled, so we fail loud instead of silently emitting an unbounded number that ignores the bounds).
        for k in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
            let err = json_schema_to_regex(&serde_json::json!({"type":"number", k: 1.5}))
                .expect_err(&format!(
                    "number + {k} must error, not silently under-constrain"
                ));
            assert!(
                format!("{err:#}").contains("number"),
                "error should name the number-bounds limitation, got: {err:#}"
            );
        }
        // `integer` bounds are still supported (a different, expressible path) - sanity that we didn't break it.
        assert!(
            json_schema_to_regex(&serde_json::json!({"type":"integer","minimum":1,"maximum":5}))
                .is_ok()
        );
    }

    #[test]
    fn json_schema_to_regex_string_pattern() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // a string field constrained to a 4-digit code; the pattern is a FULL match (anchored by the quotes).
        let rx = json_schema_to_regex(&serde_json::json!({"type":"string","pattern":"[0-9]{4}"}))
            .unwrap();
        assert!(full_match(&rx, r#""1234""#));
        assert!(!full_match(&rx, r#""123""#), "too short");
        assert!(!full_match(&rx, r#""12345""#), "too long (full match)");
        assert!(!full_match(&rx, r#""abcd""#), "wrong char class");
        assert!(!full_match(&rx, "1234"), "must be a quoted JSON string");

        // surrounding ^...$ anchors are stripped (the quotes already anchor).
        let rx2 =
            json_schema_to_regex(&serde_json::json!({"type":"string","pattern":"^[A-Z]{2}$"}))
                .unwrap();
        assert!(full_match(&rx2, r#""US""#) && !full_match(&rx2, r#""USA""#));

        // a nested pattern field inside an object property.
        let obj = serde_json::json!({
            "type":"object",
            "properties":{"date":{"type":"string","pattern":"[0-9]{4}-[0-9]{2}-[0-9]{2}"}},
            "required":["date"]
        });
        let rxo = json_schema_to_regex(&obj).unwrap();
        assert!(full_match(&rxo, r#"{"date":"2026-06-23"}"#));
        assert!(!full_match(&rxo, r#"{"date":"2026/06/23"}"#));
    }

    #[test]
    fn json_schema_to_regex_size_bounds() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // array minItems/maxItems: 1..2 integers.
        let arr = serde_json::json!({
            "type": "array", "items": {"type": "integer"}, "minItems": 1, "maxItems": 2
        });
        let ra = json_schema_to_regex(&arr).unwrap();
        assert!(full_match(&ra, "[5]") && full_match(&ra, "[5,6]"));
        assert!(full_match(&ra, "[5, 6]"), "whitespace allowed");
        assert!(!full_match(&ra, "[]"), "below minItems");
        assert!(!full_match(&ra, "[5,6,7]"), "above maxItems");

        // maxItems:0 -> empty array only.
        let empty = serde_json::json!({"type":"array","items":{"type":"integer"},"maxItems":0});
        let re0 = json_schema_to_regex(&empty).unwrap();
        assert!(full_match(&re0, "[]") && !full_match(&re0, "[1]"));

        // string minLength/maxLength: exactly 2..4 non-escape chars.
        let s = serde_json::json!({"type":"string","minLength":2,"maxLength":4});
        let rs = json_schema_to_regex(&s).unwrap();
        assert!(full_match(&rs, r#""ab""#) && full_match(&rs, r#""abcd""#));
        assert!(!full_match(&rs, r#""a""#) && !full_match(&rs, r#""abcde""#));

        // unbounded string is unchanged (escapes still allowed).
        let su = json_schema_to_regex(&serde_json::json!({"type":"string"})).unwrap();
        assert!(
            full_match(&su, r#""a\"b""#),
            "escapes allowed when unbounded"
        );

        // invalid bounds error rather than producing a bad regex.
        assert!(
            json_schema_to_regex(&serde_json::json!({"type":"string","minLength":5,"maxLength":2}))
                .is_err()
        );
    }

    #[test]
    fn json_schema_to_regex_const_enum_oneof() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // non-string enum (integers).
        let ints = serde_json::json!({ "enum": [1, 2, 30] });
        let rx = json_schema_to_regex(&ints).unwrap();
        assert!(full_match(&rx, "2") && full_match(&rx, "30"));
        assert!(!full_match(&rx, "3") && !full_match(&rx, "\"2\""));

        // const (a fixed scalar; a `.` in the number must be regex-escaped, not a wildcard).
        let c = serde_json::json!({ "const": 4.2 });
        let rxc = json_schema_to_regex(&c).unwrap();
        assert!(full_match(&rxc, "4.2"));
        assert!(
            !full_match(&rxc, "412"),
            "the dot is a literal, not a wildcard"
        );

        // oneOf: a property that is a string OR null (a common "nullable" union).
        let union = serde_json::json!({
            "type": "object",
            "properties": { "v": { "oneOf": [ {"type": "string"}, {"const": null} ] } },
            "required": ["v"]
        });
        let rxu = json_schema_to_regex(&union).unwrap();
        assert!(full_match(&rxu, r#"{"v":"hi"}"#));
        assert!(full_match(&rxu, r#"{"v":null}"#));
        assert!(!full_match(&rxu, r#"{"v":3}"#), "neither string nor null");

        // composite const: a fixed object is matched structurally with SORTED keys and optional whitespace.
        let cobj = serde_json::json!({ "const": {"b": 2, "a": 1} });
        let rxo = json_schema_to_regex(&cobj).unwrap();
        assert!(
            full_match(&rxo, r#"{"a":1,"b":2}"#),
            "sorted-key canonical form"
        );
        assert!(
            full_match(&rxo, "{ \"a\" : 1 , \"b\" : 2 }"),
            "insignificant whitespace ok"
        );
        assert!(!full_match(&rxo, r#"{"a":1,"b":3}"#), "wrong value");
        assert!(!full_match(&rxo, r#"{"a":1}"#), "missing key");

        // composite enum: an array literal, and an enum over composites.
        let carr = serde_json::json!({ "const": [1, "x", true] });
        let rxa = json_schema_to_regex(&carr).unwrap();
        assert!(full_match(&rxa, r#"[1,"x",true]"#));
        assert!(full_match(&rxa, "[ 1 , \"x\" , true ]"));
        assert!(!full_match(&rxa, r#"[1,"x",false]"#));
        assert!(!full_match(&rxa, r#"[1,"x"]"#), "wrong length");

        let cenum = serde_json::json!({ "enum": [ {"a": 1}, [2, 3] ] });
        let rxe = json_schema_to_regex(&cenum).unwrap();
        assert!(full_match(&rxe, r#"{"a":1}"#) && full_match(&rxe, r#"[2,3]"#));
        assert!(!full_match(&rxe, r#"{"a":2}"#) && !full_match(&rxe, r#"[2,4]"#));
    }

    #[test]
    fn json_schema_to_regex_honors_required_optional_properties() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // sorted keys: age, name. required = [name] -> `name` mandatory, `age` optional (and age sorts FIRST,
        // so it exercises the first-present alternation: age may lead, or name may be the first present).
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "age": {"type": "integer"}, "name": {"type": "string"} },
            "required": ["name"]
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        assert!(full_match(&rx, r#"{"name":"bob"}"#), "optional age omitted");
        assert!(
            full_match(&rx, r#"{"age":3,"name":"bob"}"#),
            "optional age present (in sorted order)"
        );
        assert!(
            !full_match(&rx, r#"{"age":3}"#),
            "required name must appear"
        );
        assert!(
            !full_match(&rx, r#"{}"#),
            "required name must appear (empty)"
        );
        assert!(
            !full_match(&rx, r#"{"name":"bob","age":3}"#),
            "properties must be in sorted-key order"
        );

        // required:[] -> a fully optional object: every subset (incl. {}) conforms, in sorted order.
        let opt = serde_json::json!({
            "type": "object",
            "properties": { "a": {"type": "boolean"}, "b": {"type": "integer"} },
            "required": []
        });
        let rxo = json_schema_to_regex(&opt).unwrap();
        for s in [
            r#"{}"#,
            r#"{"a":true}"#,
            r#"{"b":7}"#,
            r#"{"a":false,"b":7}"#,
        ] {
            assert!(full_match(&rxo, s), "all-optional should accept {s}");
        }
        assert!(!full_match(&rxo, r#"{"b":7,"a":true}"#), "wrong order");

        // no `required` field -> backward-compatible all-required behavior (a missing property is rejected).
        let allreq = serde_json::json!({
            "type": "object",
            "properties": { "a": {"type": "boolean"}, "b": {"type": "integer"} }
        });
        let rxr = json_schema_to_regex(&allreq).unwrap();
        assert!(full_match(&rxr, r#"{"a":true,"b":7}"#));
        assert!(
            !full_match(&rxr, r#"{"a":true}"#),
            "all required by default"
        );
    }

    #[test]
    fn json_schema_to_regex_resolves_local_refs() {
        fn full_match(pattern: &str, s: &str) -> bool {
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z")).unwrap();
            re.is_match(s)
        }
        // The Pydantic/OpenAI-SDK shape: properties point at `$defs` via `$ref`. A `$ref` is inlined as the
        // referenced subschema, so the constraint is identical to writing the subschema inline.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "home": { "$ref": "#/$defs/Address" },
                "work": { "$ref": "#/$defs/Address" }
            },
            "required": ["home", "work"],
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": { "city": {"type": "string"}, "zip": {"type": "integer"} },
                    "required": ["city", "zip"]
                }
            }
        });
        let rx = json_schema_to_regex(&schema).unwrap();
        // sorted keys throughout: home<work, city<zip. The same $def is inlined twice (a diamond, not a cycle).
        assert!(full_match(
            &rx,
            r#"{"home":{"city":"AMS","zip":1011},"work":{"city":"BER","zip":10115}}"#
        ));
        assert!(
            !full_match(
                &rx,
                r#"{"home":{"city":"AMS"},"work":{"city":"BER","zip":10115}}"#
            ),
            "home.zip is required by the referenced subschema"
        );

        // `definitions` (draft-7 spelling) and a ref at the document root also resolve.
        let root_ref = serde_json::json!({
            "$ref": "#/definitions/N",
            "definitions": { "N": {"type": "integer", "minimum": 1, "maximum": 9} }
        });
        let rxn = json_schema_to_regex(&root_ref).unwrap();
        assert!(full_match(&rxn, "7"));
        assert!(!full_match(&rxn, "0"), "below the referenced range");

        // A recursive ref (a node referring to itself) is unbounded - rejected, not looped forever.
        let recursive = serde_json::json!({
            "$ref": "#/$defs/Tree",
            "$defs": {
                "Tree": {
                    "type": "object",
                    "properties": { "child": { "$ref": "#/$defs/Tree" } },
                    "required": ["child"]
                }
            }
        });
        let err = json_schema_to_regex(&recursive).unwrap_err().to_string();
        assert!(err.contains("recursive"), "got: {err}");

        // An unresolvable ref and an external ref both error rather than silently dropping the constraint.
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref": "#/$defs/Missing"})).is_err(),
            "dangling ref"
        );
        assert!(
            json_schema_to_regex(&serde_json::json!({"$ref": "https://example.com/s.json"}))
                .is_err(),
            "external ref"
        );
    }

    // Property fuzzer: a random schema from the supported subset, paired with a conforming-by-construction
    // instance, must be accepted by the compiled regex in both compact and pretty-printed form (the
    // latter also stresses optional-whitespace handling). Targets over-constraining (a conforming value
    // wrongly rejected), the dominant compiler-bug class.
    #[test]
    fn json_schema_to_regex_accepts_generated_conforming_json() {
        use serde_json::{Value, json};

        // deterministic xorshift64 PRNG (seeded per iteration; no external rng dep, replayable).
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
            fn coin(&mut self) -> bool {
                self.below(2) == 0
            }
        }

        // a random schema node; `depth` bounds nesting (scalars only at depth 0, keeping it regular + small).
        fn gen_schema(rng: &mut Rng, depth: u32) -> Value {
            let kinds = if depth == 0 { 4 } else { 7 };
            match rng.below(kinds) {
                0 => {
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("integer"));
                    let a = rng.below(40) as i64 - 20;
                    let b = a + rng.below(40) as i64;
                    match rng.below(4) {
                        1 => {
                            m.insert("minimum".into(), json!(a));
                        }
                        2 => {
                            m.insert("maximum".into(), json!(b));
                        }
                        3 => {
                            m.insert("minimum".into(), json!(a));
                            m.insert("maximum".into(), json!(b));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                1 => json!({"type": "boolean"}),
                2 => {
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("string"));
                    let min = rng.below(4);
                    let max = min + rng.below(4);
                    match rng.below(3) {
                        1 => {
                            m.insert("minLength".into(), json!(min));
                        }
                        2 => {
                            m.insert("minLength".into(), json!(min));
                            m.insert("maxLength".into(), json!(max));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                3 => {
                    let pool = [
                        json!("x"),
                        json!("y"),
                        json!("z"),
                        json!(1),
                        json!(2),
                        json!(true),
                    ];
                    let n = (2 + rng.below(3) as usize).min(pool.len());
                    let mut idxs: Vec<usize> = (0..pool.len()).collect();
                    let mut vars = Vec::new();
                    for _ in 0..n {
                        let j = rng.below(idxs.len() as u64) as usize;
                        vars.push(pool[idxs.remove(j)].clone());
                    }
                    json!({ "enum": vars })
                }
                4 => {
                    let item = gen_schema(rng, depth.saturating_sub(1));
                    let mut m = serde_json::Map::new();
                    m.insert("type".into(), json!("array"));
                    m.insert("items".into(), item);
                    let min = rng.below(3);
                    let max = min + rng.below(3);
                    match rng.below(3) {
                        1 => {
                            m.insert("minItems".into(), json!(min));
                        }
                        2 => {
                            m.insert("minItems".into(), json!(min));
                            m.insert("maxItems".into(), json!(max));
                        }
                        _ => {}
                    }
                    Value::Object(m)
                }
                5 => {
                    let names = ["a", "b", "c"];
                    let nprops = 1 + rng.below(3) as usize;
                    let mut props = serde_json::Map::new();
                    let mut req = Vec::new();
                    for &name in &names[..nprops] {
                        props.insert(name.into(), gen_schema(rng, depth.saturating_sub(1)));
                        if rng.coin() {
                            req.push(json!(name));
                        }
                    }
                    json!({ "type": "object", "properties": props, "required": req })
                }
                _ => {
                    let n = 2 + rng.below(2) as usize;
                    let subs: Vec<Value> = (0..n)
                        .map(|_| gen_schema(rng, depth.saturating_sub(1)))
                        .collect();
                    json!({ "oneOf": subs })
                }
            }
        }

        fn rand_string(rng: &mut Rng, len: u64) -> String {
            // no quote/backslash: fits both the length-bounded `[^"\\]{m,n}` and unbounded string patterns.
            const CH: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
            (0..len)
                .map(|_| CH[rng.below(CH.len() as u64) as usize] as char)
                .collect()
        }

        // a value that conforms to `schema` by construction.
        fn gen_instance(rng: &mut Rng, schema: &Value) -> Value {
            if let Some(Value::Array(vars)) = schema.get("enum") {
                return vars[rng.below(vars.len() as u64) as usize].clone();
            }
            if let Some(Value::Array(subs)) = schema.get("oneOf") {
                let pick = rng.below(subs.len() as u64) as usize;
                return gen_instance(rng, &subs[pick]);
            }
            match schema.get("type").and_then(|t| t.as_str()).unwrap_or("") {
                "boolean" => json!(rng.coin()),
                "integer" => {
                    let lo = schema.get("minimum").and_then(|v| v.as_i64());
                    let hi = schema.get("maximum").and_then(|v| v.as_i64());
                    let v = match (lo, hi) {
                        (Some(lo), Some(hi)) => lo + rng.below((hi - lo + 1) as u64) as i64,
                        (Some(lo), None) => lo + rng.below(50) as i64,
                        (None, Some(hi)) => hi - rng.below(50) as i64,
                        (None, None) => rng.below(100) as i64 - 50,
                    };
                    json!(v)
                }
                "string" => {
                    let min = schema
                        .get("minLength")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let max = schema
                        .get("maxLength")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(min + 3);
                    let len = min + rng.below(max - min + 1);
                    json!(rand_string(rng, len))
                }
                "array" => {
                    let items = schema.get("items").unwrap();
                    let min = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0);
                    let max = schema
                        .get("maxItems")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(min + 2);
                    let k = min + rng.below(max - min + 1);
                    Value::Array((0..k).map(|_| gen_instance(rng, items)).collect())
                }
                "object" => {
                    let props = schema
                        .get("properties")
                        .and_then(|p| p.as_object())
                        .unwrap();
                    let req: std::collections::HashSet<&str> = schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();
                    let mut obj = serde_json::Map::new();
                    for (name, psch) in props {
                        if req.contains(name.as_str()) || rng.coin() {
                            obj.insert(name.clone(), gen_instance(rng, psch));
                        }
                    }
                    Value::Object(obj)
                }
                _ => json!(null),
            }
        }

        let mut failures = 0usize;
        for i in 0..400u64 {
            let seed =
                0x9E37_79B9_7F4A_7C15u64 ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(1);
            let mut rng = Rng(seed | 1);
            let schema = gen_schema(&mut rng, 3);
            // every generated construct is in the supported subset, so compilation must succeed.
            let pattern = json_schema_to_regex(&schema)
                .unwrap_or_else(|e| panic!("schema did not compile: {e}\nschema: {schema}"));
            let re = regex_automata::meta::Regex::new(&format!(r"\A(?:{pattern})\z"))
                .unwrap_or_else(|e| panic!("emitted an invalid regex: {e}\npattern: {pattern}"));
            for _ in 0..4 {
                let inst = gen_instance(&mut rng, &schema);
                let compact = serde_json::to_string(&inst).unwrap();
                if !re.is_match(&compact) {
                    eprintln!(
                        "REJECTED conforming (compact): {compact}\n  schema:  {schema}\n  pattern: {pattern}"
                    );
                    failures += 1;
                }
                let pretty = serde_json::to_string_pretty(&inst).unwrap();
                if !re.is_match(&pretty) {
                    eprintln!(
                        "REJECTED conforming (pretty): {pretty:?}\n  schema:  {schema}\n  pattern: {pattern}"
                    );
                    failures += 1;
                }
            }
        }
        assert_eq!(
            failures, 0,
            "{failures} conforming instances were wrongly rejected"
        );
    }
}

#[cfg(test)]
mod surface_bytes_and_limits {
    use super::Constraint;
    use crate::chat_format::ChatFormat;
    use crate::tokenize::{GgufVocab, TextCodec};
    use regex_automata::dfa::Automaton;

    /// A SentencePiece GGUF vocabulary with `add_space_prefix` (the llama default): word-initial pieces
    /// carry the U+2581 marker, which an isolated decode strips from the first piece of its output.
    fn spm_codec() -> TextCodec {
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
            "<0x41>",
        ]
        .map(String::from)
        .to_vec();
        let vocab = GgufVocab {
            model: Some("llama".into()),
            scores: Some(vec![0.0; tokens.len()]),
            token_types: Some(vec![2, 3, 3, 1, 1, 1, 1, 1, 1, 6]),
            tokens: Some(tokens),
            unknown_token_id: Some(0),
            add_space_prefix: Some(true),
            ..GgufVocab::default()
        };
        TextCodec::from_gguf(&vocab, 2, ChatFormat::ChatML).expect("build the SPM codec")
    }

    /// Greedy lowest-id walk of a constraint to its first accepting state: the token ids it picks.
    fn walk(constraint: &mut Constraint) -> Vec<u32> {
        let mut picked = Vec::new();
        for _ in 0..16 {
            let (allowed, eos_ok) = constraint.allowed_next();
            if eos_ok {
                return picked;
            }
            let next = allowed
                .into_iter()
                .min()
                .expect("a live state allows some token");
            constraint.advance(next);
            picked.push(next);
        }
        panic!("the walk never reached an accepting state: {picked:?}");
    }

    #[test]
    fn word_initial_spm_token_keeps_its_leading_space_in_guided_output() {
        let codec = spm_codec();
        let mut constraint = codec
            .build_regex_constraint("^ the [a-z]+$")
            .expect("build the constraint");
        let picked = walk(&mut constraint);
        let text: Vec<u8> = picked
            .iter()
            .flat_map(|&id| codec.token_bytes(id))
            .collect();
        assert_eq!(
            picked.first(),
            Some(&3),
            "the first token must be the word-initial `the`"
        );
        let re = regex_automata::meta::Regex::new("^ the [a-z]+$").unwrap();
        assert!(
            re.is_match(&text),
            "guided output {:?} must match the regex",
            String::from_utf8_lossy(&text)
        );
    }

    #[test]
    fn token_bytes_come_from_the_vocabulary_piece() {
        let codec = spm_codec();
        assert_eq!(
            codec.token_bytes(3),
            b" the",
            "word-initial piece starts with 0x20"
        );
        assert_eq!(codec.token_bytes(5), b" ", "the bare marker is a space");
        assert_eq!(codec.token_bytes(6), b"a");
        assert_eq!(
            codec.token_bytes(9),
            b"A",
            "a byte-fallback piece is its byte"
        );
        assert_eq!(
            codec.token_bytes(1),
            b"<s>",
            "a control token is its literal text"
        );
        assert!(codec.token_bytes(99).is_empty(), "outside the vocabulary");
    }

    #[test]
    fn byte_level_pieces_map_back_to_their_bytes() {
        use tokenizers::Tokenizer;
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::models::wordlevel::WordLevel;
        // `\u{0120}` is GPT-2's space; `\u{00c3}\u{00a9}` is the two bytes of U+00E9.
        let vocab = [("\u{0120}the", 0u32), ("\u{00c3}\u{00a9}", 1), ("x", 2)]
            .map(|(t, i)| (t.to_string(), i))
            .into();
        let mut tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("x".into())
                .build()
                .unwrap(),
        );
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
        let codec = TextCodec::new(
            tokenizer,
            None,
            None,
            0,
            ChatFormat::ChatML,
            crate::tokenize::ChatTemplate::default(),
        );
        assert_eq!(codec.token_bytes(0), b" the");
        assert_eq!(codec.token_bytes(1), "\u{e9}".as_bytes());
    }

    /// A codec over a word-level vocabulary with the given decoder (`None` is no decoder at all).
    fn word_codec(decoder: Option<tokenizers::DecoderWrapper>) -> TextCodec {
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;
        let vocab = [("##ing", 0u32), ("the", 1), ("\u{2581}x", 2), ("<unk>", 3)]
            .map(|(t, i)| (t.to_string(), i))
            .into();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(decoder);
        TextCodec::new(
            tokenizer,
            None,
            None,
            3,
            ChatFormat::ChatML,
            crate::tokenize::ChatTemplate::default(),
        )
    }

    #[test]
    fn a_wordpiece_continuation_loses_its_marker_and_other_decoders_keep_the_piece() {
        use tokenizers::DecoderWrapper;
        use tokenizers::decoders::wordpiece::WordPiece;
        let wordpiece = word_codec(Some(DecoderWrapper::WordPiece(WordPiece::default())));
        assert_eq!(
            wordpiece.token_bytes(0),
            b"ing",
            "`##` is a marker, not text"
        );
        assert_eq!(wordpiece.token_bytes(1), b"the");
        assert_eq!(
            wordpiece.token_bytes(2),
            "\u{2581}x".as_bytes(),
            "WordPiece has no space marker"
        );
        let verbatim = word_codec(None);
        assert_eq!(
            verbatim.token_bytes(0),
            b"##ing",
            "no decoder: the piece is the text"
        );
        assert_eq!(verbatim.token_bytes(2), "\u{2581}x".as_bytes());
    }

    #[test]
    fn a_state_explosion_pattern_is_refused() {
        let codec = spm_codec();
        // `(a|b)*a(a|b){17}` determinizes to 2^18 states.
        let started = std::time::Instant::now();
        let err = codec
            .build_regex_constraint("(a|b)*a(a|b){17}")
            .err()
            .expect("the DFA size limit must refuse the pattern");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "refusal took {:?}",
            started.elapsed()
        );
        assert!(err.to_string().contains("guided regex"), "{err}");
        // a pattern within the limits still builds.
        codec
            .build_regex_constraint("(a|b)*a(a|b){6}")
            .expect("a modest pattern builds");
    }

    /// A 32k-token numeric vocabulary: every token walks the `[0-9]+` DFA, so a mask is a full-vocab scan.
    fn numeric_codec() -> TextCodec {
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;
        let vocab = (0..32_000u32)
            .map(|i| (i.to_string(), i))
            .chain([("<unk>".to_string(), 32_000u32)])
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        TextCodec::new(
            Tokenizer::new(model),
            None,
            None,
            32_000,
            ChatFormat::ChatML,
            crate::tokenize::ChatTemplate::default(),
        )
    }

    /// Guided decode-rate row: the host cost of one mask step on a 32k vocabulary where about 12% of the
    /// tokens are allowed. Prints ns/step; run by name with `--run-ignored` and compare a build with the
    /// per-state cache against one without it (see the card). A timing is not an assertion.
    #[test]
    #[ignore = "timing row, not a correctness test: prints ns/step"]
    fn guided_mask_step_rate_row() {
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;
        let letters = b"abcdefghijklmnopqrstuvwxyz";
        let vocab = (0..32_000u32)
            .map(|i| (format!("{}{i}", letters[i as usize % 26] as char), i))
            .chain([("<unk>".to_string(), 32_000u32)])
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let codec = TextCodec::new(
            Tokenizer::new(model),
            None,
            None,
            32_000,
            ChatFormat::ChatML,
            crate::tokenize::ChatTemplate::default(),
        );
        let mut constraint = codec.build_regex_constraint("([a-c][0-9]+)+").unwrap();
        let steps = 300u32;
        let started = std::time::Instant::now();
        let mut allowed_len = 0;
        for _ in 0..steps {
            let (allowed, _) = constraint.allowed_next();
            allowed_len = allowed.len();
            constraint.advance(0); // token `a0`
        }
        let per_step = started.elapsed() / steps;
        eprintln!("guided mask step: {per_step:?}/step, {allowed_len} of 32000 tokens allowed");
    }

    #[test]
    fn a_revisited_state_reads_its_cached_mask() {
        let codec = numeric_codec();
        let mut constraint = codec.build_regex_constraint("[0-9]+").unwrap();
        let Constraint::Regex { masks, .. } = &constraint else {
            unreachable!("a regex constraint");
        };
        let masks = masks.clone();
        let walked = || masks.walked.load(std::sync::atomic::Ordering::Relaxed);
        let (first, _) = constraint.allowed_next();
        assert_eq!(first.len(), 32_000);
        assert_eq!(
            walked(),
            32_001,
            "the first visit scans the vocabulary once"
        );
        constraint.advance(7);
        let (_, _) = constraint.allowed_next();
        let after_second_state = walked();
        assert_eq!(after_second_state, 2 * 32_001, "a new state scans once");
        for token in [1, 22, 333, 4444] {
            let (allowed, _) = constraint.allowed_next();
            assert_eq!(allowed.len(), 32_000);
            constraint.advance(token);
        }
        assert_eq!(
            walked(),
            after_second_state,
            "revisiting the loop state must not scan the vocabulary again"
        );
        // the cached mask is the same set the scan produced.
        let Constraint::Regex { dfa, state, .. } = &constraint else {
            unreachable!()
        };
        assert!(!dfa.is_dead_state(*state));
    }
}
