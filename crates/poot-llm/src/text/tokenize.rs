//! Tokenizer family: GGUF SPM/BPE construction, encode/decode, streaming detokenization, and the
//! [`TextCodec`] those services hang off.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use poot_load::gguf::GgufIndex;
use poot_models::chat::ChatFormat;
use regex_automata::dfa::dense;
use tokenizers::Tokenizer;

use crate::error::{OptionExt, Result};

/// A checkpoint's jinja chat template and the bos/eos token strings it references (`{{ bos_token }}`,
/// `{{ eos_token }}`). All `None` for a checkpoint that ships none: rendering then falls back to the
/// family's [`ChatFormat`].
#[derive(Clone, Debug, Default)]
pub struct ChatTemplate {
    pub template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

/// One encoder row's tokens: the ids and each token's segment id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderTokens {
    pub ids: Vec<u32>,
    pub types: Vec<u32>,
}

/// Compiled guided-decoding DFAs, keyed by regex pattern.
pub(crate) type DfaCache = HashMap<String, Arc<dense::DFA<Vec<u32>>>>;

/// Everything text-side a checkpoint ships: tokenization, chat rendering and the guided constraint
/// builders. Held by `ModelHandle`. The services are `impl TextCodec` blocks in this directory's files; none needs a model or a device.
pub struct TextCodec {
    pub(crate) tokenizer: Tokenizer,
    /// For SentencePiece GGUFs (`tokenizer.ggml.model = "llama"|"gemma4"`), the SP-BPE encoder
    /// [`Self::encode`] uses instead of the HF `tokenizer` (which only approximated SP-BPE and
    /// over-segmented small-vocab non-vocab words). Decode still uses `tokenizer`. `None` for
    /// byte-level BPE and safetensors checkpoints.
    pub(crate) spm: Option<SpmData>,
    /// The BOS id [`Self::encode`] prepends: `Some` only for a checkpoint trained with a mandatory BOS.
    pub(crate) bos: Option<u32>,
    /// The EOS id [`Self::encode`] appends: `Some` only for a GGUF whose `add_eos_token` is set.
    pub(crate) eos_suffix: Option<u32>,
    /// The id a guided constraint permits once its pattern is complete.
    pub(crate) eos: u32,
    pub(crate) chat_format: ChatFormat,
    pub(crate) chat_template: ChatTemplate,
    /// Whether `chat_template` was given at load in place of the checkpoint's own: a template the caller
    /// asked for does not fall back to `chat_format` when it fails to render.
    pub(crate) chat_template_overridden: bool,
    /// Per-token UTF-8 bytes (token id -> decoded bytes), built once on first use: it depends only on
    /// the static tokenizer (~50ms for a 150k vocab).
    pub(crate) guided_byte_table: OnceLock<Arc<Vec<Vec<u8>>>>,
    /// Bounded cache of compiled guided-decoding DFAs, keyed by regex pattern: the DFA build
    /// dominates a guided request's cost.
    pub(crate) guided_dfa_cache: Mutex<DfaCache>,
}

impl TextCodec {
    pub(crate) fn new(
        tokenizer: Tokenizer,
        spm: Option<SpmData>,
        bos: Option<u32>,
        eos: u32,
        chat_format: ChatFormat,
        chat_template: ChatTemplate,
    ) -> Self {
        Self {
            tokenizer,
            spm,
            bos,
            eos_suffix: None,
            eos,
            chat_format,
            chat_template,
            chat_template_overridden: false,
            guided_byte_table: OnceLock::new(),
            guided_dfa_cache: Mutex::new(HashMap::new()),
        }
    }
}

impl TextCodec {
    /// The text services of an HF checkpoint directory: `tokenizer.json`, and the chat template and
    /// bos/eos strings of `tokenizer_config.json` (or a sibling `chat_template.jinja`). `encode`
    /// prepends `prompt_bos`, the family's mandatory BOS (`ModelConfig::prompt_bos`), when it has one.
    pub(crate) fn from_hf_dir(
        dir: &std::path::Path,
        eos: u32,
        prompt_bos: Option<u32>,
        chat_format: ChatFormat,
    ) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| err!("load tokenizer: {e}"))?;
        let (template, bos_token, eos_token) = crate::text::chat::read_tokenizer_config_chat(dir);
        Ok(Self::new(
            tokenizer,
            None,
            prompt_bos,
            eos,
            chat_format,
            ChatTemplate {
                template,
                bos_token,
                eos_token,
            },
        ))
    }

    /// The text services of a GGUF checkpoint, read from the same header index its weights were:
    /// the tokenizer (SentencePiece or byte-level BPE), the chat template and the bos/eos strings
    /// looked up in the token table. `encode` adds BOS and EOS as llama.cpp does
    /// ([`gguf_special_policy`]).
    pub(crate) fn from_gguf(g: &GgufIndex, eos: u32, chat_format: ChatFormat) -> Result<Self> {
        let tokenizer = gguf_tokenizer(g)?;
        let token_id = |key: &str| g.get(key).and_then(|v| v.as_u64()).map(|v| v as u32);
        let (add_bos, add_eos) = gguf_special_policy(g);
        let bos = token_id("tokenizer.ggml.bos_token_id").filter(|_| add_bos);
        let eos_suffix = token_id("tokenizer.ggml.eos_token_id").filter(|_| add_eos);
        let token_str = |key: &str| -> Option<String> {
            let id = g.get(key).and_then(|v| v.as_u64())? as usize;
            g.get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())?
                .get(id)
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        let chat_template = ChatTemplate {
            template: g
                .get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            bos_token: token_str("tokenizer.ggml.bos_token_id"),
            eos_token: token_str("tokenizer.ggml.eos_token_id"),
        };
        Ok(Self::new(
            tokenizer,
            gguf_spm_data(g),
            bos,
            eos,
            chat_format,
            chat_template,
        )
        .with_eos_suffix(eos_suffix))
    }

    /// This codec with `encode` appending `eos_suffix` (a GGUF's `add_eos_token`).
    fn with_eos_suffix(mut self, eos_suffix: Option<u32>) -> Self {
        self.eos_suffix = eos_suffix;
        self
    }
}

impl std::fmt::Debug for TextCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextCodec")
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .field("chat_format", &self.chat_format)
            .field("spm", &self.spm.is_some())
            .finish_non_exhaustive()
    }
}

/// UTF-8-boundary-safe streaming diff: the stable new content of `full` beyond `prev`, holding back an
/// in-progress multi-byte codepoint. Shared by every `stream_piece`. The `tokenizers` decode is lossy UTF-8,
/// so a token that only partly completes a codepoint decodes its tail as U+FFFD of a different byte length than the final char, and slicing at
/// the raw `prev.len()` can land off a char boundary. Trimming any trailing U+FFFD run from both `prev`
/// and `full` keeps `stable_prev` a byte-prefix of `stable_full`, so the slice is char-boundary-safe.
pub(crate) fn stream_piece_delta(prev: &str, full: &str) -> String {
    let stable_prev = prev.trim_end_matches('\u{FFFD}');
    let stable_full = full.trim_end_matches('\u{FFFD}');
    stable_full
        .get(stable_prev.len()..)
        .unwrap_or("")
        .to_string()
}

impl TextCodec {
    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        // SentencePiece GGUFs use the correct SP-BPE encoder; everything else the HF tokenizer.
        let mut ids = match &self.spm {
            Some(spm) => spm.encode(prompt),
            None => self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| err!("encode: {e}"))?
                .get_ids()
                .to_vec(),
        };
        if let Some(bos) = self.bos {
            // Guard against a double BOS: a chat template that renders `{{ bos_token }}` already tokenizes to a
            // leading BOS id (see `render_chat_value`), so inserting again would give `[BOS, BOS, ...]`. Without a
            // leading BOS in the prompt the insert still runs, so there is always exactly one.
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        if let Some(eos) = self.eos_suffix {
            ids.push(eos);
        }
        Ok(ids)
    }

    /// `first` (and `second`, a cross-encoder's document) as an encoder takes it: the tokenizer's own
    /// special tokens (`[CLS]`, `[SEP]`) added, and each token's segment id (0 for the first sentence,
    /// 1 for the second). A checkpoint's tokenizer may ship fixed padding; only the real tokens are
    /// kept, so a row's length is its own and the encoder's mask does the padding.
    pub fn encode_for_encoder(&self, first: &str, second: Option<&str>) -> Result<EncoderTokens> {
        let encoded = match second {
            Some(second) => self.tokenizer.encode((first, second), true),
            None => self.tokenizer.encode(first, true),
        }
        .map_err(|e| err!("encode: {e}"))?;
        let real = |i: &usize| encoded.get_attention_mask()[*i] == 1;
        let keep: Vec<usize> = (0..encoded.get_ids().len()).filter(real).collect();
        Ok(EncoderTokens {
            ids: keep.iter().map(|&i| encoded.get_ids()[i]).collect(),
            types: keep.iter().map(|&i| encoded.get_type_ids()[i]).collect(),
        })
    }

    /// Number of tokens in `text` with no special tokens added (no BOS), for usage accounting when the
    /// returned text was trimmed (e.g. at a stop sequence) and the generated token count overcounts.
    pub fn token_count(&self, text: &str) -> Result<usize> {
        Ok(match &self.spm {
            Some(spm) => spm.encode(text).len(),
            None => self
                .tokenizer
                .encode(text, false)
                .map_err(|e| err!("encode: {e}"))?
                .get_ids()
                .len(),
        })
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| err!("decode: {e}"))
    }

    /// Incremental detokenization for streaming: the text contributed by the last token of `tokens`, with
    /// leading spaces preserved. Per-token `decode(&[t])` loses leading spaces on SentencePiece/Metaspace
    /// tokenizers (an isolated token decodes as the first piece, stripping its space marker). Decoding the
    /// running sequence and taking the suffix keeps them. One token of left context (the token before
    /// `gen_start`, if any) anchors the decode so the first generated token keeps its leading space.
    ///
    /// UTF-8 boundary safety: see [`stream_piece_delta`]. Trailing U+FFFD runs are stripped from `prev` and
    /// `full` before diffing, so an in-progress multi-byte codepoint is held back (nothing emitted) until a
    /// later token completes it, then emitted whole; a replacement char never leaks. Stateless. Call after
    /// pushing the new token, once per token.
    pub fn stream_piece(&self, tokens: &[u32], gen_start: usize) -> Result<String> {
        let anchor = gen_start.saturating_sub(1); // one left-context token (none if the prompt is empty)
        let prev = self.decode(&tokens[anchor..tokens.len() - 1])?; // context + all but the new token
        let full = self.decode(&tokens[anchor..])?; // context + the new token
        Ok(stream_piece_delta(&prev, &full))
    }
}

/// SentencePiece-BPE encode of one already-normalized text segment (U+2581 for spaces), a port of
/// llama.cpp's `llm_tokenizer_spm::tokenize`: over a doubly-linked list of the text's chars, repeatedly
/// merge the adjacent symbol pair whose concatenation is a vocab token with the highest score, until no
/// valid merge remains; then map each symbol to its id, using byte-fallback (`<0xNN>`) tokens for chars
/// not in vocab. `vocab` maps token string -> id; `scores[id]` is the SentencePiece log-prob; `unk_id`
/// is the fallback when byte-fallback tokens are absent. Appends ids to `out`.
pub(crate) fn spm_encode(
    text: &str,
    vocab: &HashMap<String, u32>,
    scores: &[f32],
    unk_id: u32,
    out: &mut Vec<u32>,
) {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    // One symbol = a byte range [start, start+n) into `text`. A merged-away symbol has n == 0.
    #[derive(Clone, Copy)]
    struct Sym {
        start: usize,
        n: usize,
        prev: i32,
        next: i32,
    }
    let chars: Vec<(usize, usize)> = text
        .char_indices()
        .map(|(i, c)| (i, c.len_utf8()))
        .collect();
    if chars.is_empty() {
        return;
    }
    let mut syms: Vec<Sym> = chars
        .iter()
        .enumerate()
        .map(|(k, &(start, n))| Sym {
            start,
            n,
            prev: k as i32 - 1,
            next: if k + 1 < chars.len() {
                k as i32 + 1
            } else {
                -1
            },
        })
        .collect();

    // A candidate merge of adjacent symbols `left`+`right`, prioritized by the formed token's score. `size`
    // is the combined byte length at push time, a staleness check when popped.
    struct Bigram {
        score: f32,
        left: i32,
        right: i32,
        size: usize,
    }
    impl PartialEq for Bigram {
        fn eq(&self, o: &Self) -> bool {
            self.score == o.score && self.left == o.left
        }
    }
    impl Eq for Bigram {}
    impl Ord for Bigram {
        fn cmp(&self, o: &Self) -> Ordering {
            // Max-heap by score; ties go to the leftmost pair (smaller left index), like llama.cpp's
            // `std::priority_queue` comparator.
            self.score
                .partial_cmp(&o.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| o.left.cmp(&self.left))
        }
    }
    impl PartialOrd for Bigram {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }

    let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
    let try_add = |heap: &mut BinaryHeap<Bigram>, syms: &[Sym], left: i32, right: i32| {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (&syms[left as usize], &syms[right as usize]);
        let s = &text[l.start..r.start + r.n];
        if let Some(&id) = vocab.get(s) {
            heap.push(Bigram {
                score: scores.get(id as usize).copied().unwrap_or(0.0),
                left,
                right,
                size: l.n + r.n,
            });
        }
    };
    // Seed every adjacent (prev, cur) pair once (cur=0 has prev -1, skipped).
    for k in 0..syms.len() {
        try_add(&mut heap, &syms, syms[k].prev, k as i32);
    }

    while let Some(bg) = heap.pop() {
        let (li, ri) = (bg.left as usize, bg.right as usize);
        // Stale if either symbol was already merged away, or their combined length no longer matches.
        if syms[li].n == 0 || syms[ri].n == 0 || syms[li].n + syms[ri].n != bg.size {
            continue;
        }
        // Merge right into left; right becomes empty; relink. (Fields are read into locals first: Rust rejects
        // `syms[li].n += syms[ri].n` as a simultaneous mut+shared borrow of the same Vec.)
        let rn = syms[ri].n;
        let rnext = syms[ri].next;
        syms[li].n += rn;
        syms[ri].n = 0;
        syms[li].next = rnext;
        if rnext >= 0 {
            syms[rnext as usize].prev = bg.left;
        }
        let lprev = syms[li].prev;
        try_add(&mut heap, &syms, lprev, bg.left);
        try_add(&mut heap, &syms, bg.left, rnext);
    }

    // Walk the final linked list from the head (prev == -1) and emit ids.
    let mut i = syms
        .iter()
        .position(|s| s.prev == -1)
        .map(|p| p as i32)
        .unwrap_or(-1);
    while i >= 0 {
        let s = syms[i as usize];
        if s.n > 0 {
            let piece = &text[s.start..s.start + s.n];
            match vocab.get(piece) {
                Some(&id) => out.push(id),
                None => {
                    // byte-fallback: emit each byte as a `<0xNN>` token, else <unk>.
                    for &b in piece.as_bytes() {
                        let bt = format!("<0x{b:02X}>");
                        out.push(vocab.get(&bt).copied().unwrap_or(unk_id));
                    }
                }
            }
        }
        i = s.next;
    }
}

/// The SentencePiece tokenizer data for a `tokenizer.ggml.model = "llama"|"gemma4"` GGUF (gemma, phi3,
/// llama, mistral). `TextCodec::encode` uses it via [`spm_encode`] (bit-exact vs llama.cpp) instead of the
/// HF-BPE reconstructed-merge model, which over-segmented small-vocab non-vocab words. Decode still uses
/// the HF `Tokenizer`. `specials` are the CONTROL/USER_DEFINED atomic tokens the input is split on before
/// SP-BPE (chat turn markers), longest first for greedy matching.
pub(crate) struct SpmData {
    pub(crate) vocab: HashMap<String, u32>,
    pub(crate) scores: Vec<f32>,
    pub(crate) unk_id: u32,
    pub(crate) add_prefix: bool,
    pub(crate) specials: Vec<(String, u32)>,
}

impl SpmData {
    /// Normalizes one text fragment (optional leading U+2581 dummy prefix, then space -> U+2581) and SP-BPEs it.
    fn seg(&self, s: &str, add_prefix: bool, out: &mut Vec<u32>) {
        if s.is_empty() {
            return;
        }
        let mut norm = String::new();
        if add_prefix {
            norm.push('\u{2581}');
        }
        for ch in s.chars() {
            norm.push(if ch == ' ' { '\u{2581}' } else { ch });
        }
        spm_encode(&norm, &self.vocab, &self.scores, self.unk_id, out);
    }

    /// Encodes `text`: splits on the registered special tokens (earliest occurrence, longest match at a tie),
    /// emits each special's id atomically, and SP-BPEs every text fragment. The SentencePiece dummy prefix is
    /// applied to every non-empty fragment, as llama.cpp does: a special token resets the start of text, so
    /// e.g. the `\n` after each phi3 `<|end|>` gets its own leading U+2581. BOS is added by the caller
    /// (`TextCodec::encode`).
    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Earliest special occurrence; ties go to the longest special.
            let mut hit: Option<(usize, usize, u32)> = None; // (byte_pos, special_len, id)
            for (s, id) in &self.specials {
                if let Some(pos) = rest.find(s.as_str()) {
                    let better = match hit {
                        None => true,
                        Some((hp, hlen, _)) => pos < hp || (pos == hp && s.len() > hlen),
                    };
                    if better {
                        hit = Some((pos, s.len(), *id));
                    }
                }
            }
            match hit {
                Some((pos, len, id)) => {
                    self.seg(&rest[..pos], self.add_prefix, &mut out);
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.seg(rest, self.add_prefix, &mut out);
                    rest = "";
                }
            }
        }
        out
    }
}

/// The GGUF's CONTROL (`token_type` 3) + USER_DEFINED (4) tokens as tokenizer added tokens: the special
/// tokens (BOS/EOS, chat turn markers like `<start_of_turn>`, `<|im_start|>`, `<|user|>`). They must be
/// registered so the added vocabulary matches them atomically, before the normalizer/pre-tokenizer/BPE;
/// otherwise a special adjacent to text (as in every chat template, e.g. gemma's
/// `<start_of_turn>model\n`) is shredded into byte/sub-word pieces instead of its single trained id.
/// llama.cpp registers exactly these as special. `normalized(false)` so gemma's space->U+2581 normalizer
/// never touches them. Absent `token_type` -> no specials.
fn gguf_special_tokens(g: &GgufIndex) -> Vec<tokenizers::AddedToken> {
    let (Some(tokens), Some(types)) = (
        g.get("tokenizer.ggml.tokens").and_then(|v| v.as_array()),
        g.get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array()),
    ) else {
        return vec![];
    };
    tokens
        .iter()
        .zip(types.iter())
        .filter_map(|(t, ty)| {
            let s = t.as_str()?;
            // llama_token_type: 3 = CONTROL, 4 = USER_DEFINED (the atomic special/added tokens).
            match ty.as_u64()? {
                3 | 4 => Some(tokenizers::AddedToken::from(s.to_string(), true).normalized(false)),
                _ => None,
            }
        })
        .collect()
}

/// Builds [`SpmData`] for a SentencePiece GGUF (`tokenizer.ggml.model = "llama"|"gemma4"`), else `None`.
/// Reads tokens, scores, unk, add_space_prefix, and the CONTROL/USER_DEFINED specials as (string, id)
/// pairs. `TextCodec::encode` uses it for SP-BPE; decode stays on the HF `Tokenizer`.
pub(crate) fn gguf_spm_data(g: &GgufIndex) -> Option<SpmData> {
    if !matches!(
        g.get("tokenizer.ggml.model").and_then(|v| v.as_str()),
        Some("llama") | Some("gemma4")
    ) {
        return None;
    }
    // Enumerate before filter_map (see `gguf_tokenizer_spm_bpe`): `vocab`'s ids must be the token's true
    // index into `tokenizer.ggml.tokens`/`.scores` (`scores` below is an unfiltered `.map()`, and
    // `spm_encode` reads `scores[id]` by that id), not its position after dropping a non-Str element.
    let token_strs: Vec<(usize, String)> = g
        .get("tokenizer.ggml.tokens")?
        .as_array()?
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (i, s.to_string())))
        .collect();
    let scores: Vec<f32> = g
        .get("tokenizer.ggml.scores")?
        .as_array()?
        .iter()
        .map(|v| v.as_f32().unwrap_or(0.0))
        .collect();
    let vocab: HashMap<String, u32> = token_strs
        .iter()
        .map(|(i, s)| (s.clone(), *i as u32))
        .collect();
    let unk_id = g
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let add_prefix = g
        .get("tokenizer.ggml.add_space_prefix")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    // CONTROL(3) + USER_DEFINED(4) tokens are the atomic special tokens to split on. Longest-first so the
    // greedy match prefers a longer special over a shorter prefix of it.
    let mut specials: Vec<(String, u32)> = Vec::new();
    if let Some(types) = g
        .get("tokenizer.ggml.token_type")
        .and_then(|v| v.as_array())
    {
        for &(i, ref s) in &token_strs {
            let Some(ty) = types.get(i) else { continue };
            if matches!(ty.as_u64(), Some(3) | Some(4)) && !s.is_empty() {
                specials.push((s.clone(), i as u32));
            }
        }
    }
    specials.sort_by_key(|b| std::cmp::Reverse(b.0.len()));
    Some(SpmData {
        vocab,
        scores,
        unk_id,
        add_prefix,
        specials,
    })
}

/// Whether llama.cpp adds BOS and EOS around a GGUF's text: the file's `tokenizer.ggml.add_bos_token` /
/// `add_eos_token` when present, else the vocab family's default (`llama_vocab::impl::load`): SentencePiece
/// adds BOS, byte-level BPE adds none except the llama3 pre-tokenizer, and none adds EOS.
fn gguf_special_policy(g: &GgufIndex) -> (bool, bool) {
    let default_bos = match g.get("tokenizer.ggml.model").and_then(|v| v.as_str()) {
        Some("llama" | "gemma4") => true,
        _ => BpePre::from_gguf(g) == BpePre::Llama3,
    };
    let flag = |key: &str, default: bool| g.get(key).and_then(|v| v.as_bool()).unwrap_or(default);
    (
        flag("tokenizer.ggml.add_bos_token", default_bos),
        flag("tokenizer.ggml.add_eos_token", false),
    )
}

/// llama.cpp's pre-tokenizer families for byte-level BPE GGUFs, selected by `tokenizer.ggml.pre`. The
/// regexes are llama.cpp's `LLAMA_VOCAB_PRE_TYPE_*` split patterns: they differ in how digit runs group
/// (llama3 in runs of up to three from the left, qwen2 one digit at a time, gpt2 the whole run).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BpePre {
    /// `llama-bpe` (and its `llama3`/`llama-v3` aliases): Llama-3 family.
    Llama3,
    /// `qwen2`: Qwen2/2.5/3.
    Qwen2,
    /// Anything else (including an absent key): the GPT-2 split, the `tokenizers` ByteLevel default.
    Gpt2,
}

impl BpePre {
    const LLAMA3_REGEX: &'static str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const QWEN2_REGEX: &'static str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    fn from_gguf(g: &GgufIndex) -> Self {
        match g.get("tokenizer.ggml.pre").and_then(|v| v.as_str()) {
            Some("llama-bpe" | "llama3" | "llama-v3") => Self::Llama3,
            Some("qwen2") => Self::Qwen2,
            _ => Self::Gpt2,
        }
    }

    /// llama.cpp sets `ignore_merges` for llama3: a whole pre-token that is a vocab entry is emitted
    /// as that id without running the merges.
    fn ignore_merges(self) -> bool {
        self == Self::Llama3
    }

    fn pre_tokenizer(self) -> Result<tokenizers::PreTokenizerWrapper> {
        use tokenizers::SplitDelimiterBehavior;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::pre_tokenizers::sequence::Sequence;
        use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
        let regex = match self {
            Self::Llama3 => Self::LLAMA3_REGEX,
            Self::Qwen2 => Self::QWEN2_REGEX,
            // No prefix space, trim offsets, GPT-2 regex split.
            Self::Gpt2 => return Ok(ByteLevel::new(false, true, true).into()),
        };
        let split = Split::new(
            SplitPattern::Regex(regex.to_string()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| err!("split: {e}"))?;
        Ok(Sequence::new(vec![
            split.into(),
            ByteLevel::new(false, true, false).into(),
        ])
        .into())
    }
}

/// Builds a byte-level BPE [`Tokenizer`] from a GGUF's `tokenizer.ggml.tokens` + `.merges` (qwen2 is
/// gpt2-style byte-level BPE). Special tokens already live in the vocab; the ByteLevel
/// pre-tokenizer/decoder (no prefix space) handles plain text.
pub(crate) fn gguf_tokenizer(g: &GgufIndex) -> Result<Tokenizer> {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;

    // SentencePiece archs (gemma3: `tokenizer.ggml.model="llama"`; gemma4: "gemma4", same SPM-derived
    // tokens+scores shape, card 162) ship tokens+scores, not merges.
    if matches!(
        g.get("tokenizer.ggml.model").and_then(|v| v.as_str()),
        Some("llama") | Some("gemma4")
    ) {
        return gguf_tokenizer_spm_bpe(g);
    }

    let tokens = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .context("gguf missing tokenizer.ggml.tokens")?;
    let merges = g
        .get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .context("gguf missing tokenizer.ggml.merges")?;

    let vocab: std::collections::HashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (s.to_string(), i as u32)))
        .collect();
    // Drop any merge rule referencing a non-vocab token, either an orphan piece (`a` or `b`) or an orphan
    // result (`a+b`). Such a rule is dead: a byte-level BPE merge `(a,b)->ab` fires only when both inputs
    // and the merged token exist in the vocab, so removing it changes nothing. This tolerates a
    // GGUF-conversion quirk in some gpt2-BPE checkpoints (e.g. OLMo-2): a few merges reference a
    // double-UTF-8 mojibake of U+FFFD ("ï¿½", bytes c3af c2bf c2bd) as a piece or as a two-piece result
    // "ï¿" + "½", while the vocab holds the correct U+FFFD (bytes ef bf bd). HuggingFace `BPE::build`
    // rejects the whole tokenizer on the first such orphan (`Token ... out of vocabulary`); llama.cpp is
    // lenient. Filtering matches that and is a no-op for clean checkpoints (qwen2 etc. drop zero).
    let total_merges = merges.len();
    let merges: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| m.as_str())
        .filter_map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .filter(|(a, b)| {
            vocab.contains_key(a) && vocab.contains_key(b) && vocab.contains_key(&format!("{a}{b}"))
        })
        .collect();
    if merges.len() != total_merges {
        tracing::debug!(
            dropped = total_merges - merges.len(),
            kept = merges.len(),
            "gguf bpe: dropped merge rules referencing non-vocab pieces (dead rules; llama.cpp-lenient)"
        );
    }

    let pre = BpePre::from_gguf(g);
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(pre.ignore_merges())
        .build()
        .map_err(|e| err!("build bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(pre.pre_tokenizer()?));
    tok.with_decoder(Some(ByteLevel::new(false, true, true)));
    // Register control/user-defined tokens as atomic added-tokens (see `gguf_special_tokens`).
    tok.add_special_tokens(&gguf_special_tokens(g));
    Ok(tok)
}

/// Builds a BPE [`Tokenizer`] for a SentencePiece GGUF (gemma: `tokenizer.ggml.model="llama"`). The GGUF
/// carries tokens + per-token scores but no merge list. Merges are reconstructed from the vocab (the
/// standard sentencepiece->BPE extraction): a token splits into (left,right) both in vocab, picking the
/// split whose later-merged piece has the best (lowest-id = most frequent) rank, and merges are ordered
/// by the formed token's id. Then a `tokenizers::BPE` with byte_fallback + ignore_merges, gemma's
/// normalizer (space->U+2581), and decoder.
fn gguf_tokenizer_spm_bpe(g: &GgufIndex) -> Result<Tokenizer> {
    use tokenizers::decoders::DecoderWrapper;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecSequence;
    use tokenizers::decoders::strip::Strip;
    use tokenizers::models::bpe::BPE;
    use tokenizers::normalizers::replace::Replace;
    use tokenizers::normalizers::utils::Sequence as NormSequence;
    use tokenizers::normalizers::{NormalizerWrapper, Prepend};

    // SentencePiece add_dummy_prefix: prepend one U+2581 to the start of the text (default true; gemma-3
    // sets it false). Without it a leading word tokenizes as `The` instead of `▁The`, diverging from
    // llama.cpp for phi3/llama/mistral (which omit the key). gemma keeps `add_space_prefix=false`, so this
    // is inert there.
    let add_prefix = g
        .get("tokenizer.ggml.add_space_prefix")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let tokens = g
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .context("gguf missing tokenizer.ggml.tokens")?;
    // Enumerate before filter_map, as in the byte-level path (`gguf_tokenizer` above): filtering first would
    // drop a non-Str element and shift every later token's id. In practice `Cursor::value` (poot-load)
    // decodes a whole array through one element-type tag, so a parsed `tokenizer.ggml.tokens` array cannot
    // contain a non-Str element; this is defensive consistency with the sibling path.
    let token_strs: Vec<(usize, &str)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| t.as_str().map(|s| (i, s)))
        .collect();
    let vocab: std::collections::HashMap<String, u32> = token_strs
        .iter()
        .map(|&(i, s)| (s.to_string(), i as u32))
        .collect();

    // Reconstruct merges. For each multi-char token (by id = rank order), find the best split into two
    // vocab tokens; merge priority is the formed token's id (lower first). Skip byte tokens (<0xNN>) and
    // special tokens, which are atomic.
    let mut merges: Vec<(String, String)> = Vec::new();
    for &(_, s) in &token_strs {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() < 2 || (s.starts_with("<") && s.ends_with(">")) {
            continue;
        }
        let mut best: Option<(u32, String, String)> = None;
        let mut acc = String::new();
        for k in 0..chars.len() - 1 {
            acc.push(chars[k]);
            let right: String = chars[k + 1..].iter().collect();
            if let (Some(&lr), Some(&rr)) = (vocab.get(&acc), vocab.get(&right)) {
                let rank = lr.max(rr); // the piece merged last gates the merge
                if best.as_ref().map(|b| rank < b.0).unwrap_or(true) {
                    best = Some((rank, acc.clone(), right));
                }
            }
        }
        if let Some((_, l, r)) = best {
            merges.push((l, r));
        }
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .ignore_merges(true)
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| err!("build gemma bpe: {e}"))?;
    let mut tok = Tokenizer::new(bpe);
    // Normalizer: [Prepend("▁") when add_dummy_prefix] then space->▁. The Prepend gives a leading word its
    // SP-correct `▁The` id (as llama.cpp) instead of bare `The`.
    let mut norms: Vec<NormalizerWrapper> = Vec::new();
    if add_prefix {
        norms.push(Prepend::new("\u{2581}".to_string()).into());
    }
    norms.push(
        Replace::new(" ", "\u{2581}")
            .map_err(|e| err!("replace: {e}"))?
            .into(),
    );
    tok.with_normalizer(Some(NormSequence::new(norms)));
    // Split on the U+2581 space-marker (kept attached to the following word) so that with ignore_merges
    // each whole word (e.g. "The", "U+2581capital") matches its vocab id directly, as gemma does.
    {
        use tokenizers::SplitDelimiterBehavior;
        use tokenizers::pre_tokenizers::split::Split;
        let split = Split::new("\u{2581}", SplitDelimiterBehavior::MergedWithNext, false)
            .map_err(|e| err!("split: {e}"))?;
        tok.with_pre_tokenizer(Some(split));
    }
    let mut decs: Vec<DecoderWrapper> = vec![
        DecoderWrapper::Replace(Replace::new("\u{2581}", " ").map_err(|e| err!("replace: {e}"))?),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
    ];
    if add_prefix {
        // undo the dummy prefix: strip the single leading space it decodes back into.
        decs.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
    }
    tok.with_decoder(Some(DecSequence::new(decs)));
    // Register control/user-defined tokens as atomic added-tokens (see `gguf_special_tokens`); required for
    // the SPM path, where a special adjacent to text otherwise shreds via byte_fallback.
    tok.add_special_tokens(&gguf_special_tokens(g));
    Ok(tok)
}

/// Unit tests of the shared `stream_piece_delta` diff primitive, the one place the U+FFFD-trim logic
/// lives (called by `TextCodec::stream_piece`). They call the pure function
/// directly, simulating `tokenizers`' lossy-UTF-8 decode (`String::from_utf8_lossy`) over a byte buffer growing one byte at a time.
#[cfg(test)]
mod stream_piece_delta_tests {
    use super::stream_piece_delta;

    /// A 4-byte codepoint (U+1F600: F0 9F 98 80) fed one byte at a time must be held back while incomplete
    /// (each intermediate `full` ends in a U+FFFD of a different byte length than the final char) and then
    /// emitted whole, exactly once, when the last byte completes it.
    ///
    /// Without the U+FFFD trim (`full.get(prev.len()..).unwrap_or("")`) it fails: after byte 1 the raw
    /// replacement char leaks, and at the final byte `prev.len()` (3) is off the emoji's char boundary, so
    /// `full.get(3..)` returns `None` and the emoji is dropped.
    #[test]
    fn split_codepoint_holds_back_then_emits_whole_char() {
        let bytes: [u8; 4] = [0xF0, 0x9F, 0x98, 0x80]; // U+1F600
        let mut buf: Vec<u8> = Vec::new();
        let mut prev = String::new();
        let mut emitted = String::new();
        for &b in &bytes {
            buf.push(b);
            let full = String::from_utf8_lossy(&buf).to_string();
            let delta = stream_piece_delta(&prev, &full);
            if buf.len() < bytes.len() {
                assert_eq!(
                    delta, "",
                    "an incomplete codepoint must emit nothing prematurely (buf={buf:?})"
                );
            }
            emitted.push_str(&delta);
            prev = full;
        }
        assert_eq!(
            emitted, "\u{1F600}",
            "the whole emoji must be emitted exactly once, only once fully formed"
        );
    }

    /// Plain ASCII (no codepoint is ever incomplete, so the U+FFFD trim is a no-op) streams every character
    /// immediately with no holdback.
    #[test]
    fn ascii_streams_immediately_no_holdback() {
        let full_text = "hello world";
        let mut prev = String::new();
        let mut emitted = String::new();
        for ch in full_text.chars() {
            let mut full = prev.clone();
            full.push(ch);
            let delta = stream_piece_delta(&prev, &full);
            assert_eq!(delta, ch.to_string(), "ascii must never be held back");
            emitted.push_str(&delta);
            prev = full;
        }
        assert_eq!(emitted, full_text);
    }
}

#[cfg(test)]
mod stream_piece_tests {
    use super::{ChatTemplate, TextCodec};
    use poot_load::gguf::{GgufIndex, GgufValue, write_gguf};
    use poot_models::chat::ChatFormat;

    // Standard GPT-2 byte-to-unicode mapping (HF's `bytes_to_unicode()`). Every raw byte 0..256 gets its
    // own single-char token string, so a GGUF vocab built from it is a byte-level tokenizer with token id
    // == raw byte, as in a real qwen2/qwen3 GGUF (`gguf_tokenizer`'s non-SPM branch, taken since no
    // `tokenizer.ggml.model` key is set). Only for building a synthetic fixture.
    fn byte_to_unicode() -> [char; 256] {
        let mut printable: Vec<u16> = Vec::new();
        printable.extend(33u16..=126u16);
        printable.extend(161u16..=172u16);
        printable.extend(174u16..=255u16);
        let mut table = ['\u{0}'; 256];
        for &b in &printable {
            table[b as usize] = char::from_u32(b as u32).unwrap();
        }
        let mut next = 256u32;
        for slot in table.iter_mut() {
            if *slot == '\u{0}' {
                *slot = char::from_u32(next).unwrap();
                next += 1;
            }
        }
        table
    }

    /// A text codec over a GGUF vocab that is the full byte-level alphabet (token id == raw byte, no
    /// merges), built through the production `gguf_tokenizer` path. Token ids are built directly from
    /// UTF-8 bytes (bypassing `encode()`), so a test can choose exactly where a multi-byte codepoint is
    /// split across "generated" tokens.
    fn byte_level_codec() -> TextCodec {
        // The 256-byte alphabet plus one extra 2-char vocab entry (`!"`, id 256), needed only so a single
        // (never-decoded) merge rule is buildable: `BPE::builder` requires a merge's combined form to be
        // in-vocab, and `write_gguf`'s array encoder (`gguf_encode_value`) needs a non-empty
        // `tokenizer.ggml.merges` list to infer its element type. No decoded id is >= 256, so it is inert.
        let alphabet = byte_to_unicode();
        let mut tokens: Vec<GgufValue> = alphabet
            .iter()
            .map(|c| GgufValue::Str(c.to_string()))
            .collect();
        tokens.push(GgufValue::Str("!\"".to_string()));
        let kvs = vec![
            ("general.architecture", GgufValue::Str("qwen2".into())),
            ("tokenizer.ggml.tokens", GgufValue::Array(tokens)),
            (
                "tokenizer.ggml.merges",
                GgufValue::Array(vec![GgufValue::Str("! \"".to_string())]),
            ),
        ];
        let bytes = write_gguf(&kvs, &[]);
        let index = GgufIndex::parse(&poot_quant::StoredBytes::from(bytes))
            .expect("parse the byte-level fixture");
        TextCodec::from_gguf(&index, u32::MAX, ChatFormat::ChatML)
            .expect("build the byte-level fixture's codec")
    }

    /// Simulates a streaming decode loop: calls `stream_piece` once per appended token for the generated ids
    /// `all[gen_start..]` (the shape every `generate_kv_*` loop uses) and concatenates the pieces, like a
    /// client accumulating SSE `delta.content` chunks.
    fn simulate_stream(runner: &TextCodec, all: &[u32], gen_start: usize) -> String {
        let mut out = String::new();
        for k in (gen_start + 1)..=all.len() {
            out.push_str(&runner.stream_piece(&all[..k], gen_start).unwrap());
        }
        out
    }

    /// An emoji's multi-byte UTF-8 encoding split across the last two generated tokens must still stream
    /// out complete: the streamed concatenation equals the non-streaming `decode(&tokens[gen_start..])`
    /// byte for byte, even though the codepoint is incomplete for intermediate steps. Without the trim,
    /// `prev`'s trailing U+FFFD (3 bytes) has a different length than the completed char, so the slice
    /// lands off a char boundary and either drops the char or leaks the raw U+FFFD.
    #[test]
    fn stream_piece_reassembles_emoji_split_across_tokens() {
        let runner = byte_level_codec();
        let prompt = "Hi ";
        let generated = "\u{1F600} there!"; // grinning face U+1F600 = F0 9F 98 80 (4 bytes)
        let emoji_bytes: Vec<u8> = "\u{1F600}".bytes().collect();
        assert_eq!(
            emoji_bytes.len(),
            4,
            "test assumption: a 4-byte UTF-8 codepoint"
        );
        let prompt_ids: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        let gen_ids: Vec<u32> = generated.bytes().map(|b| b as u32).collect();
        let gen_start = prompt_ids.len();
        let all: Vec<u32> = prompt_ids.into_iter().chain(gen_ids).collect();

        let streamed = simulate_stream(&runner, &all, gen_start);
        let full = runner.decode(&all[gen_start..]).unwrap();
        assert_eq!(
            streamed, full,
            "streamed concatenation must equal the non-streaming decode, including the split emoji"
        );
        assert_eq!(
            streamed, generated,
            "must also match the original generated text exactly"
        );
        assert!(
            !streamed.contains('\u{FFFD}'),
            "no replacement char may leak into the finished stream: {streamed:?}"
        );
    }

    /// A 3-byte CJK codepoint split across tokens (a different byte count than the 4-byte emoji above).
    #[test]
    fn stream_piece_reassembles_cjk_split_across_tokens() {
        let runner = byte_level_codec();
        let prompt = "translate: ";
        let generated = "\u{4E2D}\u{6587}"; // "china script" (Chinese), 3 bytes each codepoint
        let prompt_ids: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        let gen_ids: Vec<u32> = generated.bytes().map(|b| b as u32).collect();
        let gen_start = prompt_ids.len();
        let all: Vec<u32> = prompt_ids.into_iter().chain(gen_ids).collect();

        let streamed = simulate_stream(&runner, &all, gen_start);
        let full = runner.decode(&all[gen_start..]).unwrap();
        assert_eq!(streamed, full);
        assert_eq!(streamed, generated);
    }

    /// A plain-ASCII generation streams identically to the non-streaming decode; the U+FFFD trim is a
    /// no-op on ASCII.
    #[test]
    fn stream_piece_ascii_regression() {
        let runner = byte_level_codec();
        let prompt = "The capital of France is";
        let generated = " Paris, a city in Europe.";
        let prompt_ids: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        let gen_ids: Vec<u32> = generated.bytes().map(|b| b as u32).collect();
        let gen_start = prompt_ids.len();
        let all: Vec<u32> = prompt_ids.into_iter().chain(gen_ids).collect();

        let streamed = simulate_stream(&runner, &all, gen_start);
        let full = runner.decode(&all[gen_start..]).unwrap();
        assert_eq!(streamed, full);
        assert_eq!(
            streamed, generated,
            "byte-level decode must roundtrip plain ASCII exactly"
        );
    }

    /// A minimal Metaspace/SentencePiece-style `TextCodec` (WordLevel model over whole ▁-prefixed words),
    /// constructed directly (private fields are visible in this module tree) since this tokenizer shape has
    /// no GGUF fixture path in `gguf_tokenizer`. Only `tokenizer` matters for `decode`/`stream_piece`; every
    /// other field is an inert default.
    fn metaspace_codec() -> TextCodec {
        use tokenizers::Tokenizer;
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};

        let words = ["The", "capital", "of", "France", "is", "Paris"];
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("<unk>".to_string(), 0u32);
        for (i, w) in words.iter().enumerate() {
            vocab.insert(format!("\u{2581}{w}"), (i + 1) as u32);
        }
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel model");
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Metaspace::new(
            '\u{2581}',
            PrependScheme::Always,
            true,
        )));
        tok.with_decoder(Some(Metaspace::new(
            '\u{2581}',
            PrependScheme::Always,
            true,
        )));

        TextCodec::new(
            tok,
            None,
            None,
            u32::MAX,
            ChatFormat::ChatML,
            ChatTemplate::default(),
        )
    }

    /// Leading spaces stay preserved on a SentencePiece/Metaspace tokenizer (the original reason
    /// `stream_piece` anchors on one token of left context); the U+FFFD-trim fix must not regress this.
    #[test]
    fn stream_piece_preserves_leading_space_metaspace() {
        let runner = metaspace_codec();
        let prompt_ids: Vec<u32> = vec![1, 2, 3, 4, 5]; // "The capital of France is"
        let paris_id: u32 = 6;
        let gen_start = prompt_ids.len();
        let mut tokens = prompt_ids;
        tokens.push(paris_id);

        // Sanity: the isolated-token decode this function avoids does lose the Metaspace marker
        // (`Decoder::decode_chain` drops a leading replacement char at index 0).
        let naive = runner.decode(&[paris_id]).unwrap();
        assert_eq!(
            naive, "Paris",
            "sanity: isolated single-token decode loses the Metaspace leading space"
        );

        let piece = runner.stream_piece(&tokens, gen_start).unwrap();
        assert_eq!(
            piece, " Paris",
            "stream_piece must keep the leading space via its left-context anchor"
        );
        // `decode(&tokens[gen_start..])` is not the reference here: with one generated token it is itself the
        // isolated single-token decode and drops the marker like `naive` above. The multi-token byte-level
        // tests above check the equivalence with left context.
    }
}

/// Card 221: `TextCodec::encode` must not double-prepend BOS when the rendered chat prompt already starts
/// with `{{ bos_token }}` (real Gemma templates do). Synthetic end-to-end repro through
/// `TextCodec::render_chat_value` -> `TextCodec::encode`, the production path (`render_chat_value` passes
/// the chat template's `bos_token` into the jinja render; `encode` re-tokenizes the rendered text).
#[cfg(test)]
mod double_bos_tests {
    use super::{ChatTemplate, TextCodec};
    use poot_models::chat::ChatFormat;
    use tokenizers::Tokenizer;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;

    /// A minimal `TextCodec`, constructed directly like `stream_piece_tests::metaspace_codec` (only
    /// `tokenizer`/`bos`/`bos_token`/`chat_template` matter). A GGUF fixture is not possible here:
    /// `poot_load::gguf::write_gguf` has no `GgufValue::Bool` case, so `tokenizer.ggml.add_bos_token` cannot
    /// be written. It still runs a real `tokenizers::Tokenizer` (WordLevel + whitespace splitting) through
    /// `TextCodec::render_chat_value` (jinja) and `TextCodec::encode`, with `add_bos_token=true` and a
    /// `{{ bos_token }}`-leading chat_template like a real Gemma GGUF.
    fn bos_template_codec() -> TextCodec {
        const BOS_ID: u32 = 0;
        let vocab: std::collections::HashMap<String, u32> = [
            ("BOS".to_string(), BOS_ID),
            ("user".to_string(), 1u32),
            ("hi".to_string(), 2u32),
            ("hello".to_string(), 3u32),
            ("<unk>".to_string(), 4u32),
        ]
        .into_iter()
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("build wordlevel model");
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(WhitespaceSplit));

        TextCodec::new(
            tok,
            None,
            // mirrors a GGUF with tokenizer.ggml.add_bos_token=true
            Some(BOS_ID),
            u32::MAX,
            ChatFormat::ChatML,
            ChatTemplate {
                // A real Gemma chat_template starts with `{{ bos_token }}`; this is that shape reduced to
                // what `render_jinja_core` needs (messages/bos_token context vars).
                template: Some(
                    "{{ bos_token }} {% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
                        .to_string(),
                ),
                bos_token: Some("BOS".to_string()),
                eos_token: None,
            },
        )
    }

    /// End-to-end repro: render a one-message chat through the `{{ bos_token }}`-leading template, then
    /// `encode` the rendered prompt as the production chat path does (`poot-serve` calls `render_chat` then
    /// `encode` on `.prompt`). The guard in `encode` must yield exactly one leading BOS id, not two.
    #[test]
    fn render_chat_then_encode_has_exactly_one_leading_bos() {
        let runner = bos_template_codec();
        let rendered = runner
            .render_chat_value(
                &serde_json::json!([{"role": "user", "content": "hi"}]),
                None,
            )
            .unwrap();
        assert!(
            rendered.prompt.trim_start().starts_with("BOS"),
            "template must have rendered the bos_token literal first: {:?}",
            rendered.prompt
        );
        let ids = runner
            .encode(&rendered.prompt)
            .expect("encode rendered chat prompt");
        let bos = runner.bos.expect("fixture sets add_bos_token=true");
        let leading_bos = ids.iter().take_while(|&&id| id == bos).count();
        assert_eq!(
            leading_bos, 1,
            "expected exactly one leading BOS id, got {leading_bos} in {ids:?}"
        );
        assert_eq!(ids, vec![0u32, 1, 2], "BOS user hi, exactly once each");
    }

    /// Restoring the unconditional `ids.insert(0, bos)` (dropping the `if ids.first() != Some(&bos)` guard)
    /// makes `render_chat_then_encode_has_exactly_one_leading_bos` see `leading_bos == 2` and
    /// `ids == [0, 0, 1, 2]` instead of `[0, 1, 2]`, so the fixture reproduces the bug and the guard fixes it.
    #[test]
    fn encode_without_a_chat_template_still_prepends_bos_once() {
        // Sanity: the guard must not regress the plain raw-prompt case; a prompt not rendered through
        // `{{ bos_token }}` still gets exactly one BOS prepended.
        let runner = bos_template_codec();
        let bos = runner.bos.expect("fixture sets add_bos_token=true");
        let ids = runner.encode("hello").expect("encode raw prompt");
        assert_eq!(
            ids.first(),
            Some(&bos),
            "BOS must still be prepended when absent"
        );
        assert_eq!(
            ids.iter().take_while(|&&id| id == bos).count(),
            1,
            "exactly one BOS, not a double insert"
        );
        assert_eq!(ids, vec![0u32, 3], "BOS hello");
    }
}

#[cfg(test)]
mod phi3_gguf_inspect_legacy {
    use super::gguf_tokenizer;
    use poot_load::gguf::GgufIndex;
    use poot_models::chat::ChatFormat;

    // The Phi3 chat template emits `<|user|>`/`<|end|>`/`<|assistant|>`. For the server's chat path,
    // poot's GGUF tokenizer must encode each as a single special-token id, not literal-text pieces. This
    // loads only the phi3 GGUF tokenizer (no weights) and checks that (spec 034's Phi3 arm; full coherence
    // is covered by the phi3 GGUF generate test and the ChatFormat render unit tests).
    #[test]
    #[ignore = "needs the phi-3-mini GGUF under POOT_MODELS_DIR"]
    fn phi3_gguf_chat_special_tokens_are_single_ids() {
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "phi-3-mini-gguf/Phi-3-mini-4k-instruct-q4.gguf"
        )) else {
            return;
        };
        let g = GgufIndex::parse(
            &poot_quant::StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap(),
        )
        .expect("load phi3 gguf");
        let tok = gguf_tokenizer(&g).expect("build phi3 gguf tokenizer");
        for marker in ["<|user|>", "<|assistant|>", "<|end|>", "<|system|>"] {
            let ids = tok
                .encode(marker, false)
                .expect("encode")
                .get_ids()
                .to_vec();
            eprintln!("{marker} -> {ids:?}");
            assert_eq!(
                ids.len(),
                1,
                "phi3 chat marker {marker} must be ONE special-token id, got {ids:?} (it would be \
                 split into literal text otherwise, corrupting the chat prompt)"
            );
        }
        // Sanity: the full Phi3-rendered conversation round-trips back to text containing the markers.
        let prompt = ChatFormat::Phi3.render(&[("user", "Hi")]);
        let ids = tok
            .encode(prompt.as_str(), false)
            .expect("encode prompt")
            .get_ids()
            .to_vec();
        let back = tok.decode(&ids, false).expect("decode");
        assert!(
            back.contains("<|user|>") && back.contains("<|assistant|>"),
            "roundtrip: {back:?}"
        );
    }

    // Feasibility probe for a GGUF phi3 path: dumps the arch, phi3.* / rope metadata, and per-layer tensor
    // names (fused vs split qkv/gate_up, how LongRoPE factors are carried). Run with:
    //   cargo test -p poot-llm --release phi3_gguf_metadata -- --ignored --nocapture
    #[test]
    #[ignore = "needs the phi-3-mini GGUF under POOT_MODELS_DIR"]
    fn phi3_gguf_metadata() {
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "phi-3-mini-gguf/Phi-3-mini-4k-instruct-q4.gguf"
        )) else {
            return;
        };
        let g = GgufIndex::parse(
            &poot_quant::StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap(),
        )
        .expect("load phi3 gguf");
        eprintln!("arch = {:?}", g.architecture());
        let mut keys: Vec<&String> = g
            .metadata
            .keys()
            .filter(|k| {
                let k = k.as_str();
                k.contains("rope")
                    || k.contains("rotary")
                    || k.contains("phi")
                    || k.contains("scal")
                    || k.contains("head")
                    || k.contains("context")
                    || k.contains("embedding")
            })
            .collect();
        keys.sort();
        eprintln!("-- selected metadata --");
        for k in keys {
            eprintln!("  {k} = {:?}", g.metadata.get(k));
        }
        // layer-0 tensor names (fused vs split), plus any rope_factors tensors.
        let mut tnames: Vec<&String> = g
            .tensors
            .keys()
            .filter(|t| t.starts_with("blk.0.") || t.contains("rope") || !t.starts_with("blk."))
            .collect();
        tnames.sort();
        eprintln!("-- tensors (blk.0 + non-block + rope) --");
        for t in tnames {
            eprintln!("  {t}  dims={:?}", g.tensors.get(t).map(|i| &i.dims));
        }
    }
}

#[cfg(test)]
mod gemma_tokenizer {
    use super::{GgufIndex, gguf_tokenizer};
    use tokenizers::Tokenizer;

    #[test]
    #[ignore = "needs the phi-4-mini tokenizer.json under POOT_MODELS_DIR"]
    fn phi3_tokenizer_roundtrips() {
        // phi-4-mini uses the o200k (vocab 200064) BPE tokenizer; confirm the tokenizers-crate path encodes a
        // plain prompt to a sane, non-empty id list that decodes back (a mis-encode would feed the model
        // garbage).
        let Some(tjson) =
            poot_test_util::model_path(poot_test_util::checkpoint!("phi-4-mini/tokenizer.json"))
        else {
            return;
        };
        let tok = Tokenizer::from_file(tjson).expect("load phi tokenizer.json");
        let prompt = "The capital of France is";
        let enc = tok.encode(prompt, false).expect("encode");
        let ids = enc.get_ids();
        eprintln!("phi3 encode {prompt:?} -> {ids:?}");
        assert!(!ids.is_empty(), "empty encoding");
        assert!(
            ids.iter().all(|&i| i < 200064),
            "id out of vocab range: {ids:?}"
        );
        let back = tok.decode(ids, false).expect("decode");
        eprintln!("phi3 decode -> {back:?}");
        assert_eq!(back.trim(), prompt, "roundtrip mismatch");
    }

    #[test]
    #[ignore = "needs the gemma-3-1b GGUF + safetensors tokenizer.json under POOT_MODELS_DIR"]
    fn gguf_spm_matches_hf_tokenizer() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "gemma-3-1b-gguf/gemma-3-1b-it-Q4_K_M.gguf"
        )) else {
            return;
        };
        let Some(tjson) =
            poot_test_util::model_path(poot_test_util::checkpoint!("gemma-3-1b-it/tokenizer.json"))
        else {
            return;
        };
        let g = GgufIndex::parse(
            &poot_quant::StoredBytes::map(std::fs::File::open(gguf).unwrap()).unwrap(),
        )
        .unwrap();
        let ours = gguf_tokenizer(&g).expect("build gguf spm-bpe tokenizer");
        let hf = Tokenizer::from_file(tjson).expect("load tokenizer.json");
        // Validated: whole-word text (space-separated, no attached punctuation) tokenizes bit-exactly vs HF
        // via the ignore_merges + U+2581-split path, including the SentencePiece dummy-space prefix (a leading
        // word -> "▁The" when add_space_prefix defaults true; gemma sets it false); bit-exact vs `llama-tokenize`
        // for gemma3 plain+chat and olmo2 (2026-07-28). Open: chunks that fall to the reconstructed BPE merges
        // (a non-vocab word such as "Testing"/"fox" in a small-vocab model like phi3, or attached punctuation
        // "Hello,") still differ from HF's SP-BPE merge order. This asserts the validated cases and reports
        // subword.
        let exact = ["The capital of France is", "def add a b return a plus b"];
        let wip = ["Hello, world!", "Multi-token   spaces and mixed text."];
        for s in exact {
            let a = ours.encode(s, false).unwrap().get_ids().to_vec();
            let b = hf.encode(s, false).unwrap().get_ids().to_vec();
            assert_eq!(a, b, "whole-word sample should match HF exactly: {s:?}");
            eprintln!("exact {s:?} -> {} ids", a.len());
        }
        for s in wip {
            let a = ours.encode(s, false).unwrap().get_ids().to_vec();
            let b = hf.encode(s, false).unwrap().get_ids().to_vec();
            eprintln!("wip   {s:?} match={}\n  gguf={a:?}\n  hf  ={b:?}", a == b);
        }
    }
}

/// `TextCodec::encode` against llama.cpp's own token ids on the Llama-3.2-1B vocabulary (the llama-bpe
/// pre-tokenizer: digits in runs of up to three from the left, the llama3 regex splits, BOS from the
/// file's defaults). Skips when the checkpoint is absent from the models directory.
#[cfg(test)]
mod llama3_golden_tests {
    use super::*;

    /// `(text, ids)`, each recorded with
    /// `llama-tokenize -m Llama-3.2-1B-Instruct-Q4_K_M.gguf -f <file holding exactly the text> --ids`
    /// (llama.cpp build of the dev box, 2026-10-06). BOS 128000 leads every row: llama.cpp adds it for
    /// the llama-bpe pre-tokenizer.
    const GOLDEN: &[(&str, &[u32])] = &[
        ("", &[128000]),
        ("x", &[128000, 87]),
        (
            "The number 4719 and 1234567 and 0042",
            &[
                128000, 791, 1396, 220, 20617, 24, 323, 220, 4513, 10961, 22, 323, 220, 8759, 17,
            ],
        ),
        (
            "Total: 3.14159, 2024-10-06, 1000000",
            &[
                128000, 7749, 25, 220, 18, 13, 9335, 2946, 11, 220, 2366, 19, 12, 605, 12, 2705,
                11, 220, 1041, 931, 15,
            ],
        ),
        (
            "héllo wörld, 你好世界, 😀 café",
            &[
                128000, 71, 19010, 385, 289, 9603, 509, 11, 118195, 53901, 102616, 11, 91416, 53050,
            ],
        ),
        ("a  b   c    d", &[128000, 64, 220, 293, 256, 272, 262, 294]),
        (
            "line one\n\n\nline two\r\n  indented",
            &[128000, 1074, 832, 1432, 1074, 1403, 319, 220, 1280, 16243],
        ),
        (
            "fn main() {\n    println!(\"{}\", x[0] + 12);\n}",
            &[
                128000, 8998, 1925, 368, 341, 262, 14069, 80978, 865, 58, 15, 60, 489, 220, 717,
                317, 92,
            ],
        ),
        (
            "don't I'll WE'RE can't",
            &[128000, 15357, 956, 358, 3358, 20255, 95253, 649, 956],
        ),
        (
            "hi<|eot_id|>x<|start_header_id|>user<|end_header_id|>",
            &[128000, 6151, 128009, 87, 128006, 882, 128007],
        ),
        (
            "  leading and trailing   ",
            &[128000, 220, 6522, 323, 28848, 262],
        ),
    ];

    /// The qwen2 pre-tokenizer on the Qwen2.5-0.5B vocabulary (one digit per token, no BOS), recorded as
    /// above with `qwen2.5-0.5b-instruct-q4_k_m.gguf`. The upper-case contractions and the newline runs
    /// are where the qwen2 regex differs from the GPT-2 split, so a wrong selector changes these ids.
    #[test]
    fn qwen2_encode_equals_llama_cpp_token_ids() {
        const GOLDEN: &[(&str, &[u32])] = &[
            (
                "The number 4719 and 1234567, don't  go\nx",
                &[
                    785, 1372, 220, 19, 22, 16, 24, 323, 220, 16, 17, 18, 19, 20, 21, 22, 11, 1513,
                    944, 220, 728, 198, 87,
                ],
            ),
            ("WE'RE I'LL", &[12457, 94153, 358, 6, 4086]),
            ("a \n\nb  \n  c", &[64, 4710, 65, 2303, 220, 272]),
        ];
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        )) else {
            return;
        };
        let g = GgufIndex::parse(
            &poot_quant::StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap(),
        )
        .expect("open the Qwen2.5 GGUF");
        assert_eq!(
            BpePre::from_gguf(&g),
            BpePre::Qwen2,
            "tokenizer.ggml.pre must select the qwen2 split"
        );
        let codec = TextCodec::from_gguf(&g, 151645, ChatFormat::ChatML).expect("build codec");
        for (text, want) in GOLDEN {
            assert_eq!(
                &codec.encode(text).expect("encode"),
                want,
                "token ids of {text:?} differ from llama.cpp"
            );
        }
    }

    #[test]
    fn encode_equals_llama_cpp_token_ids() {
        let Some(path) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "llama-3.2-1b-gguf/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )) else {
            return;
        };
        let g = GgufIndex::parse(
            &poot_quant::StoredBytes::map(std::fs::File::open(&path).unwrap()).unwrap(),
        )
        .expect("open the Llama-3.2-1B GGUF");
        assert_eq!(
            BpePre::from_gguf(&g),
            BpePre::Llama3,
            "tokenizer.ggml.pre must select the llama3 split"
        );
        let codec = TextCodec::from_gguf(&g, 128009, ChatFormat::ChatML).expect("build codec");
        for (text, want) in GOLDEN {
            assert_eq!(
                &codec.encode(text).expect("encode"),
                want,
                "token ids of {text:?} differ from llama.cpp"
            );
        }
    }
}

/// The BOS/EOS rows of [`gguf_special_policy`] through `TextCodec::from_gguf`/`encode`, on synthetic GGUF
/// vocabularies: llama.cpp's per-family defaults and the explicit `add_bos_token` / `add_eos_token` keys.
#[cfg(test)]
mod gguf_special_policy_tests {
    use super::*;
    use poot_load::gguf::{GgufValue, write_gguf};

    const BOS: u32 = 3;
    const EOS: u32 = 4;

    /// A byte-level BPE vocab (`a`=0, `b`=1, `ab`=2, `<s>`=3, `</s>`=4): `encode("ab")` is `[2]` before
    /// BOS/EOS. `pre` is `tokenizer.ggml.pre`; `flags` the add_bos/add_eos keys under test.
    fn bpe_codec(pre: &str, flags: &[(&'static str, bool)]) -> TextCodec {
        let str_array = |items: &[&str]| {
            GgufValue::Array(
                items
                    .iter()
                    .map(|s| GgufValue::Str(s.to_string()))
                    .collect(),
            )
        };
        let mut kvs = vec![
            ("general.architecture", GgufValue::Str("llama".into())),
            ("tokenizer.ggml.model", GgufValue::Str("gpt2".into())),
            ("tokenizer.ggml.pre", GgufValue::Str(pre.into())),
            (
                "tokenizer.ggml.tokens",
                str_array(&["a", "b", "ab", "<s>", "</s>"]),
            ),
            ("tokenizer.ggml.merges", str_array(&["a b"])),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(BOS)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(EOS)),
        ];
        kvs.extend(flags.iter().map(|&(k, v)| (k, GgufValue::Bool(v))));
        let index = GgufIndex::parse(&poot_quant::StoredBytes::from(write_gguf(&kvs, &[])))
            .expect("parse the BPE fixture");
        TextCodec::from_gguf(&index, EOS, ChatFormat::ChatML).expect("build the BPE codec")
    }

    /// A SentencePiece vocab (`<unk>`=0, `<s>`=1, `</s>`=2, `▁a`=3, `a`=4): `encode("a")` is `[3]`.
    fn spm_codec(flags: &[(&'static str, bool)]) -> TextCodec {
        let tokens = ["<unk>", "<s>", "</s>", "\u{2581}a", "a"];
        let mut kvs = vec![
            ("general.architecture", GgufValue::Str("llama".into())),
            ("tokenizer.ggml.model", GgufValue::Str("llama".into())),
            (
                "tokenizer.ggml.tokens",
                GgufValue::Array(
                    tokens
                        .iter()
                        .map(|s| GgufValue::Str(s.to_string()))
                        .collect(),
                ),
            ),
            (
                "tokenizer.ggml.scores",
                GgufValue::Array([0.0, 0.0, 0.0, -1.0, -2.0].map(GgufValue::F32).to_vec()),
            ),
            ("tokenizer.ggml.bos_token_id", GgufValue::U32(1)),
            ("tokenizer.ggml.eos_token_id", GgufValue::U32(2)),
        ];
        kvs.extend(flags.iter().map(|&(k, v)| (k, GgufValue::Bool(v))));
        let index = GgufIndex::parse(&poot_quant::StoredBytes::from(write_gguf(&kvs, &[])))
            .expect("parse the SPM fixture");
        TextCodec::from_gguf(&index, 2, ChatFormat::ChatML).expect("build the SPM codec")
    }

    const ADD_BOS: &str = "tokenizer.ggml.add_bos_token";
    const ADD_EOS: &str = "tokenizer.ggml.add_eos_token";

    #[test]
    fn byte_level_bpe_adds_no_bos_by_default() {
        assert_eq!(bpe_codec("gpt-2", &[]).encode("ab").unwrap(), [2]);
    }

    #[test]
    fn the_llama3_pre_tokenizer_adds_bos_by_default() {
        assert_eq!(bpe_codec("llama-bpe", &[]).encode("ab").unwrap(), [BOS, 2]);
    }

    #[test]
    fn an_explicit_add_bos_false_overrides_the_llama3_default() {
        let codec = bpe_codec("llama-bpe", &[(ADD_BOS, false)]);
        assert_eq!(codec.encode("ab").unwrap(), [2]);
    }

    #[test]
    fn an_explicit_add_bos_true_overrides_the_bpe_default() {
        let codec = bpe_codec("gpt-2", &[(ADD_BOS, true)]);
        assert_eq!(codec.encode("ab").unwrap(), [BOS, 2]);
    }

    #[test]
    fn add_eos_token_appends_eos() {
        let codec = bpe_codec("gpt-2", &[(ADD_EOS, true)]);
        assert_eq!(codec.encode("ab").unwrap(), [2, EOS]);
    }

    #[test]
    fn add_bos_and_add_eos_wrap_the_text() {
        let codec = bpe_codec("gpt-2", &[(ADD_BOS, true), (ADD_EOS, true)]);
        assert_eq!(codec.encode("ab").unwrap(), [BOS, 2, EOS]);
    }

    #[test]
    fn no_eos_is_appended_by_default() {
        assert_eq!(bpe_codec("llama-bpe", &[]).encode("ab").unwrap(), [BOS, 2]);
    }

    #[test]
    fn sentencepiece_adds_bos_by_default() {
        assert_eq!(spm_codec(&[]).encode("a").unwrap(), [1, 3]);
    }

    #[test]
    fn an_explicit_add_bos_false_overrides_the_sentencepiece_default() {
        assert_eq!(spm_codec(&[(ADD_BOS, false)]).encode("a").unwrap(), [3]);
    }

    #[test]
    fn sentencepiece_honours_add_eos_token() {
        let codec = spm_codec(&[(ADD_EOS, true)]);
        assert_eq!(codec.encode("a").unwrap(), [1, 3, 2]);
    }
}
