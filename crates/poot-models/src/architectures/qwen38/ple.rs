use super::*;
use poot_graph_ir::ops::silu;

/// `Qwen4ExpTextNGramEmbedding`/`Qwen4ExpTextPLELayer` shape (the "N-gram Embedding / PLE layer"). Real
/// `Qwen/Qwen3.8-Flash-Next` `config.json` `text_config` values: `ngram_size: 3`, `heads_per_ngram: 8`,
/// `ple_layer_ids: [2]`, `ple_embed_dim: 2560`, `ple_conv_kernel_size: 4`,
/// `ngram_vocab_size_base: 20000000`, `make_ngram_vocab_size_divisible_by: 128`, `vocab_size: 248320`,
/// `eos_token_id: 248044`; `seed` is absent, so it takes `configuration_qwen4_exp.py:156`'s default `1234`.
///
/// # `ple_layer_ids` is one-indexed
///
/// `ple_layer_ids: [2]` does not mean decoder layer index 2. `Qwen4ExpTextDecoderLayer.__init__` selects
/// the PLE layer with `config.ple_layer_ids.index(layer_idx + 1) if layer_idx + 1 in config.ple_layer_ids
/// else None` (`modeling_qwen4_exp.py` line 1275), and `Qwen4ExpTextConfig.__post_init__` validates
/// "`ple_layer_ids` must contain ONE-INDEXED ids in `[1, num_hidden_layers]`"
/// (`configuration_qwen4_exp.py:240-246`). So `[2]` puts the PLE on 0-based decoder layer 1, matching the
/// checkpoint's `model.safetensors.index.json` (which has `model.language_model.layers.1.ple.*` and no
/// `layers.2.ple.*`). The config also requires every PLE layer to be a `linear_attention` layer
/// (`configuration_qwen4_exp.py:248-254`), which layer 1 is.
///
/// [`Self::ple_layer_ids`] keeps the one-indexed convention verbatim so a config parsed from
/// `config.json` needs no adjustment; [`Qwen4ExpPleConfig::ple_layer_index`] does the `+1` conversion at
/// the single point of use.
///
/// # The N-gram embedding table is enormous (see [`qwen38_ngram_tables`])
///
/// Real `ngram_heads = (ngram_size - 1) * heads_per_ngram = 16`, each with its own ~20M-entry prime vocab
/// slice, so the padded table is `[320001536, 160]` BF16 (about 102 GB) across 128
/// `ngram_embedding.shard_*.weight` tensors (real `split_ngram_parts`; each `[2500012, 160]` per the
/// `model-00005-of-00131.safetensors` header). Real checkpoint loading is out of scope; see
/// [`Qwen4ExpModelConfig::ple`].
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpPleConfig {
    /// Real `ple_layer_ids`, kept one-indexed as the real config stores it (real value: `[2]`).
    pub ple_layer_ids: Vec<usize>,
    /// Real `ple_embed_dim` (2560): the width of the concatenated per-head n-gram embedding and the input
    /// width of `key_proj`/`value_proj`. Must be divisible by [`Self::ngram_heads`].
    pub ple_embed_dim: usize,
    /// Real `ple_conv_kernel_size` (4): the PLE short conv's tap count.
    pub ple_conv_kernel_size: usize,
    /// Real `ngram_size` (3): the largest n-gram order. Also the short conv's dilation (real
    /// `conv_dilation = config.ngram_size`, `Qwen4ExpTextPLELayer.__init__`).
    pub ngram_size: usize,
    /// Real `ngram_vocab_size_base` (20000000): the search floor for each hash head's prime vocab slice
    /// size (see [`qwen38_ngram_tables`]).
    pub ngram_vocab_size_base: u64,
    /// Real `heads_per_ngram` (8): hash heads per n-gram order.
    pub heads_per_ngram: usize,
    /// Real `make_ngram_vocab_size_divisible_by` (128): the embedding table's row-count padding.
    pub make_ngram_vocab_size_divisible_by: usize,
    /// Real `seed` (`configuration_qwen4_exp.py`'s default `1234`; absent from `config.json`).
    pub seed: u64,
    /// Real `eos_token_id` (248044): the fill value for out-of-segment shifted tokens and the
    /// pre-sequence context (see [`qwen38_ngram_ids`]).
    pub eos_token_id: i32,
    /// Real `vocab_size` (248320); only used to derive [`Qwen4ExpNgramTables::layer_multipliers`].
    pub vocab_size: usize,
}

impl Qwen4ExpPleConfig {
    /// Real `self.ngram_heads = (ngram_size - 1) * heads_per_ngram` (16): one hash head per (n-gram order
    /// in `2..=ngram_size`, head within that order) pair.
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Real `self.context_len = ngram_size - 1` (2): how many tokens before the current window the hash
    /// needs (the real implementation carries them as a third conv-state slot).
    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }

    /// Real `head_dim_per_ngram = embedding_dim // self.ngram_heads` (160): each hash head contributes this
    /// many features to [`Self::ple_embed_dim`].
    pub fn head_dim_per_ngram(&self) -> usize {
        self.ple_embed_dim / self.ngram_heads()
    }

    /// Real `conv_dilation = config.ngram_size` (3): the short conv's dilation, not its kernel size.
    pub fn conv_dilation(&self) -> usize {
        self.ngram_size
    }

    /// Real `self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation` (9): the dilated short
    /// conv's causal left-pad width, hence the decode-side conv cache depth.
    pub fn conv_state_len(&self) -> usize {
        (self.ple_conv_kernel_size - 1) * self.conv_dilation()
    }

    /// Real `Qwen4ExpTextDecoderLayer.__init__` selector: `Some(position_in_ple_layer_ids)` if this 0-based
    /// decoder layer carries a PLE layer, `None` otherwise (the test is `layer_idx + 1`; see the struct docs).
    pub fn ple_layer_index(&self, layer_idx: usize) -> Option<usize> {
        self.ple_layer_ids
            .iter()
            .position(|&id| id == layer_idx + 1)
    }
}

/// SplitMix64 constants from `modeling_qwen4_exp.py`'s `_SPLITMIX_GAMMA`/`_SPLITMIX_M1`/`_SPLITMIX_M2`/
/// `_PRIME_1` (lines 1046-1050).
pub(crate) const NGRAM_SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

pub(crate) const NGRAM_SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;

pub(crate) const NGRAM_SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

pub(crate) const NGRAM_PRIME_1: u64 = 10007;

/// Real `_splitmix64` (`modeling_qwen4_exp.py` lines 1053-1057). The real code masks to 64 bits after
/// every step (`& _MASK64`), which is Rust's `u64` wrapping arithmetic.
pub(crate) fn ngram_splitmix64(value: u64) -> u64 {
    let value = value.wrapping_add(NGRAM_SPLITMIX_GAMMA);
    let value = (value ^ (value >> 30)).wrapping_mul(NGRAM_SPLITMIX_M1);
    let value = (value ^ (value >> 27)).wrapping_mul(NGRAM_SPLITMIX_M2);
    value ^ (value >> 31)
}

/// Real `_is_prime` (lines 1071-1080).
pub(crate) fn ngram_is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut d = 3u64;
    while d * d <= value {
        if value.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Real `_find_nth_prime_after(start, count)` (lines 1083-1089): the `count`-th prime strictly greater
/// than `start`.
pub(crate) fn ngram_nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !ngram_is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// Host-side derived tables `Qwen4ExpTextNGramEmbedding.__init__` builds once per PLE layer
/// (`modeling_qwen4_exp.py` lines 1106-1124). All are pure functions of the config. The checkpoint also
/// ships `ngram_heads_vocab_sizes`/`ngram_heads_offsets`/`layer_multipliers` as `nn.Buffer` tensors
/// (I64 `[16]`/`[16]`/`[3]`), but they are re-derived here rather than loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen4ExpNgramTables {
    /// Per hash head, the prime modulus its id is reduced by. Real: 16 consecutive primes above
    /// `ngram_vocab_size_base - 1 = 19999999`.
    pub head_vocab_sizes: Vec<i64>,
    /// Per hash head, the running start of its slice inside the shared embedding table.
    pub head_offsets: Vec<i64>,
    /// `sum(head_vocab_sizes)`, the unpadded row count.
    pub total_vocab_size: i64,
    /// `ceil(total_vocab_size / make_ngram_vocab_size_divisible_by) * make_ngram_vocab_size_divisible_by`,
    /// the real `nn.Embedding`'s row count (real value: 320001536).
    pub padded_vocab_size: usize,
    /// Real `_build_layer_multipliers` output, `[ngram_size]` odd 64-bit multipliers.
    pub layer_multipliers: Vec<i64>,
}

/// Build [`Qwen4ExpNgramTables`] for one PLE layer, following `Qwen4ExpTextNGramEmbedding.__init__`/
/// `_build_layer_multipliers` (`modeling_qwen4_exp.py` lines 1059-1124). `ple_layer_index` is the layer's
/// position inside `ple_layer_ids` (0 for the real single-PLE-layer config); the real code folds it into
/// both the per-head global prime index and the multiplier seed, so two PLE layers hash differently.
///
/// Cost: the prime search is `ngram_heads` trial-division scans just above `ngram_vocab_size_base`; at
/// the real base of 20000000 that is ~4473 divisions per candidate, a few hundred microseconds in total,
/// cheap enough to run at trace time.
pub fn qwen38_ngram_tables(ple: &Qwen4ExpPleConfig, ple_layer_index: usize) -> Qwen4ExpNgramTables {
    let heads = ple.ngram_heads();
    let mut head_vocab_sizes = Vec::with_capacity(heads);
    let mut head_offsets = Vec::with_capacity(heads);
    let mut total: i64 = 0;
    for head_idx in 0..heads {
        let global_head_idx = ple_layer_index * heads + head_idx;
        let size = ngram_nth_prime_after(ple.ngram_vocab_size_base - 1, global_head_idx + 1);
        head_vocab_sizes.push(size as i64);
        head_offsets.push(total);
        total += size as i64;
    }
    let divisor = ple.make_ngram_vocab_size_divisible_by;
    let padded_vocab_size = (total as usize).div_ceil(divisor) * divisor;

    // Real `_build_layer_multipliers` (lines 1059-1068).
    let max_long = i64::MAX as u64;
    let multiplier_max = max_long / (ple.vocab_size.max(1) as u64);
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = ple
        .seed
        .wrapping_add(NGRAM_PRIME_1.wrapping_mul(ple_layer_index as u64));
    let layer_multipliers = (0..ple.ngram_size)
        .map(|index| {
            let value = base_seed.wrapping_add(NGRAM_SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            (2 * (ngram_splitmix64(value) % half_bound) + 1) as i64
        })
        .collect();

    Qwen4ExpNgramTables {
        head_vocab_sizes,
        head_offsets,
        total_vocab_size: total,
        padded_vocab_size,
        layer_multipliers,
    }
}

// Card 638: this whole cluster is a CPU reference oracle with no production caller -
// `qwen38_ngram_ids` is checked bit-exactly against the real in-graph composition
// (`qwen38_ngram_ids_graph`/`trace_qwen38_ngram_ids`, below, outside this module) by many tests. A
// production-shaped item must not be hidden behind its own `#[cfg(test)]`; a genuine test oracle like this
// one belongs in a test module instead, so it lives here rather than at `ple`'s top level.
#[cfg(test)]
pub(crate) mod ngram_oracle {
    use super::*;

    /// Real `Qwen4ExpTextNGramEmbedding._shift_right_ignore_eos` (`modeling_qwen4_exp.py` lines 1126-1139):
    /// shift `tokens` right by `shift`, filling with `eos` wherever the source position would fall before the
    /// current EOS-delimited segment's start (so an n-gram never straddles a document boundary) or before the
    /// start of the buffer.
    ///
    /// The real code uses `cummax` over `where(token == eos, position, -1)`, then a one-position right shift
    /// so the boundary token itself belongs to the old segment (`previous_eos =
    /// cat([-1], previous_eos_inclusive[:-1])`). This port scans forward once, updating `seg_start` only after
    /// position `t` is emitted (the same "exclusive of `t`" rule).
    pub(crate) fn ngram_shift_right_ignore_eos(tokens: &[i64], shift: usize, eos: i64) -> Vec<i64> {
        if shift == 0 {
            return tokens.to_vec();
        }
        let mut out = Vec::with_capacity(tokens.len());
        let mut seg_start = 0usize;
        for (t, &tok) in tokens.iter().enumerate() {
            let in_segment = t >= shift && t - shift >= seg_start;
            out.push(if in_segment { tokens[t - shift] } else { eos });
            if tok == eos {
                seg_start = t + 1;
            }
        }
        out
    }

    /// Host-side n-gram id hashing: `Qwen4ExpTextNGramEmbedding.forward` (`modeling_qwen4_exp.py` lines
    /// 1141-1187), everything up to but excluding the embedding lookup. Returns the
    /// `[input_ids.len() * ngram_heads]` row-major id block the graph then `Gather`s (see `qwen38_ple_core_with_embeddings`).
    ///
    /// # Why host-side
    ///
    /// The real hash is 64-bit integer arithmetic: each shifted token is multiplied by an odd ~2^45
    /// multiplier (wrapping at 64 bits), XORed together, then reduced modulo a ~20M prime. `poot_graph_ir`
    /// has no i64 dtype, no bitwise-XOR `OpKind`, and no integer modulo. The result is a pure function of the
    /// token ids, like the tokenizer output and the causal/RoPE tables, so the ids are computed here and bound
    /// as an `I32` step input (`{layer}.ple.ngram_ids`, card 550a). No new `OpKind` is needed.
    ///
    /// # Arguments
    ///
    /// `previous_context` is the source's `previous_context`: exactly [`Qwen4ExpPleConfig::context_len`] token ids
    /// preceding `input_ids`. For a from-scratch prefill the real code fills it with `eos_token_id`
    /// (`input_ids.new_full((batch, context_len), self.eos_token_id)`); for a decode step it is the last
    /// `context_len` generated tokens (the real code carries them in the cache's third conv-state slot). The
    /// caller builds it (see `qwen38_ngram_previous_context`) or, preferably, threads it through
    /// `Qwen4ExpNgramHistory`, which owns the trailing window and advances it per chunk.
    ///
    /// A `previous_context` whose length is not `context_len` is caller-controlled input, so this returns
    /// [`Qwen4ExpNgramError`] rather than panicking.
    ///
    /// # Signed 64-bit wrapping and Python's `remainder`
    ///
    /// `shifted_tokens[i] * layer_multipliers[i]` overflows `int64` for realistic vocab sizes and PyTorch
    /// wraps (two's complement), so this uses `wrapping_mul` on `i64`. `torch.remainder` is Python-style (the
    /// result takes the divisor's sign, always non-negative here), not Rust's `%`, so this uses `rem_euclid`;
    /// plain `%` on a negative `mixed_ids` would give a negative row index.
    ///
    /// No caller outside this crate's own tests, where it is the CPU oracle
    /// [`qwen38_ngram_ids_graph`]/[`trace_qwen38_ngram_ids`] (the real in-graph, production path) is checked
    /// against.
    pub(crate) fn qwen38_ngram_ids(
        ple: &Qwen4ExpPleConfig,
        tables: &Qwen4ExpNgramTables,
        previous_context: &[i32],
        input_ids: &[i32],
    ) -> Result<Vec<i32>, Qwen4ExpNgramError> {
        let want = ple.context_len();
        if previous_context.len() != want {
            return Err(Qwen4ExpNgramError::ContextLength {
                got: previous_context.len(),
                want,
            });
        }
        let eos = ple.eos_token_id as i64;
        let heads = ple.ngram_heads();
        let per = ple.heads_per_ngram;

        // Real `token_history = torch.cat([previous_context, input_ids], dim=-1)`.
        let history: Vec<i64> = previous_context
            .iter()
            .chain(input_ids.iter())
            .map(|&v| v as i64)
            .collect();
        let hist_len = history.len();
        let shifted: Vec<Vec<i64>> = (0..ple.ngram_size)
            .map(|shift| ngram_shift_right_ignore_eos(&history, shift, eos))
            .collect();

        // Real: one `blocks` entry per n-gram order in `2..=ngram_size`, concatenated along the head axis.
        let mut ids = vec![0i32; hist_len * heads];
        for ngram in 2..=ple.ngram_size {
            let start_idx = (ngram - 2) * per;
            for t in 0..hist_len {
                let mut mixed = shifted[0][t].wrapping_mul(tables.layer_multipliers[0]);
                for (position, row) in shifted.iter().enumerate().take(ngram).skip(1) {
                    mixed ^= row[t].wrapping_mul(tables.layer_multipliers[position]);
                }
                for j in 0..per {
                    let head = start_idx + j;
                    let id =
                        mixed.rem_euclid(tables.head_vocab_sizes[head]) + tables.head_offsets[head];
                    ids[t * heads + head] = id as i32;
                }
            }
        }

        // Real `[:, -input_ids.shape[1]:]`: drop the `context_len` history-only rows.
        let drop = hist_len - input_ids.len();
        Ok(ids[drop * heads..].to_vec())
    }

    /// Caller-controlled input failures from [`qwen38_ngram_ids`] / `Qwen4ExpNgramHistory`. Internal
    /// invariants (a carrier window it advanced itself, a non-empty shard list in the sharded lookup) stay
    /// `assert!`s; only shapes the caller supplies get here.
    ///
    /// No caller outside this crate's own tests.
    #[derive(Debug, thiserror::Error, PartialEq, Eq)]
    pub(crate) enum Qwen4ExpNgramError {
        /// `previous_context` must be exactly `context_len = ngram_size - 1` tokens.
        #[error(
            "previous_context must be exactly context_len = ngram_size - 1 tokens; got {got}, want {want}"
        )]
        ContextLength { got: usize, want: usize },
    }

    /// Build the `previous_context` argument [`qwen38_ngram_ids`] expects from a token history: the last
    /// [`Qwen4ExpPleConfig::context_len`] entries of `prior_tokens`, left-padded with `eos_token_id` when there
    /// are fewer (the real code's two fill sites: the no-cache `new_full(.., eos_token_id)` initializer and
    /// the `F.pad(.., value=self.eos_token_id)` for a first call shorter than `context_len`,
    /// `modeling_qwen4_exp.py` lines 1149-1163). An empty slice gives the from-scratch prefill case.
    ///
    /// Prefer [`Qwen4ExpNgramHistory`] when feeding a sequence in more than one call: it owns this window and
    /// advances it, so chunked prefill and decode reproduce one full prefill's ids.
    pub(crate) fn qwen38_ngram_previous_context(
        ple: &Qwen4ExpPleConfig,
        prior_tokens: &[i32],
    ) -> Vec<i32> {
        let n = ple.context_len();
        let mut ctx = vec![ple.eos_token_id; n];
        let take = prior_tokens.len().min(n);
        ctx[n - take..].copy_from_slice(&prior_tokens[prior_tokens.len() - take..]);
        ctx
    }

    /// Carried n-gram history for chunked prefill and token-by-token decode: owns exactly
    /// [`Qwen4ExpPleConfig::context_len`] trailing tokens and yields the next chunk's ids through
    /// [`qwen38_ngram_ids`], then advances the window so the following call sees this chunk as prior
    /// context. One full prefill over a sequence and any chunked/decode split of the same sequence must
    /// therefore produce the same concatenated ids (Card 449 SC-003 foundation, Card 363 FR-006/FR-012
    /// two-token prior carry). This is host-side id derivation only; it adds no graph `Slot`.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct Qwen4ExpNgramHistory {
        /// Trailing window handed to [`qwen38_ngram_ids`] as `previous_context`. Invariant: always exactly
        /// `context_len` tokens (enforced by construction and by `ids_for_chunk`).
        trailing: Vec<i32>,
        context_len: usize,
    }

    impl Qwen4ExpNgramHistory {
        /// From-scratch prefill history: `context_len` copies of `eos_token_id` (equivalent to
        /// [`qwen38_ngram_previous_context`] on an empty prior).
        pub fn new(ple: &Qwen4ExpPleConfig) -> Self {
            Self::from_prior(ple, &[])
        }

        /// Resume over `prior_tokens` with the same last-`context_len` / EOS-pad rule as
        /// [`qwen38_ngram_previous_context`].
        pub fn from_prior(ple: &Qwen4ExpPleConfig, prior_tokens: &[i32]) -> Self {
            Self {
                trailing: qwen38_ngram_previous_context(ple, prior_tokens),
                context_len: ple.context_len(),
            }
        }

        /// The window that would be (and is) passed to [`qwen38_ngram_ids`] for the next chunk.
        pub fn previous_context(&self) -> &[i32] {
            &self.trailing
        }

        /// Ids for `input_ids` against the carried window, then advance the window to the last
        /// `context_len` tokens of the combined history (via [`qwen38_ngram_previous_context`], so the
        /// EOS-pad rule stays in one place). `input_ids` may be empty (ids are empty, window unchanged).
        pub fn ids_for_chunk(
            &mut self,
            ple: &Qwen4ExpPleConfig,
            tables: &Qwen4ExpNgramTables,
            input_ids: &[i32],
        ) -> Result<Vec<i32>, Qwen4ExpNgramError> {
            let ids = qwen38_ngram_ids(ple, tables, &self.trailing, input_ids)?;
            // Advance: next previous_context = last context_len of (trailing ++ input_ids). `trailing` is
            // already context_len long, so the combined slice never needs an EOS pad, but going through
            // the helper keeps the two fill rules identical.
            if !input_ids.is_empty() {
                let combined: Vec<i32> = self.trailing.iter().chain(input_ids).copied().collect();
                self.trailing = qwen38_ngram_previous_context(ple, &combined);
            }
            // True internal invariant: only this type mutates `trailing`, and both constructors and the
            // advance path preserve `context_len`.
            assert_eq!(
                self.trailing.len(),
                self.context_len,
                "Qwen4ExpNgramHistory window must stay context_len long"
            );
            Ok(ids)
        }
    }
} // mod ngram_oracle

#[cfg(test)]
pub(crate) use ngram_oracle::*;

/// In-graph n-gram id derivation: the composition counterpart of [`qwen38_ngram_ids`], built from
/// `poot_graph_ir::ops` primitives (I32 `Mul`/`Xor`/`RemU` over the little-endian u64 hash helpers).
///
/// # Input domain
///
/// - `tokens`: `[L]` I32 token ids in `[0, vocab)`. Products `token * multiplier` are then below
///   `vocab * 2^46 < 2^63` for the real multipliers, so the u64 schoolbook never wraps the
///   mathematical product the host uses `wrapping_mul` for (it does not wrap at those bounds).
///   An out-of-domain id (negative or `>= vocab`) still produces a defined graph value, but it
///   diverges from the host oracle. The composition applies [`Builder::guard_index_bounds`] with
///   `upper_exclusive = vocab_size`, clamps out-of-range lanes to zero for the hash, and returns
///   the witness so the tracer can fail closed.
/// - `ngram_size` must be in `2..=3` (the real config is `3`); larger orders are not composed.
/// - `history`: the carried `[1, context_len]` I32 state, **XOR-encoded** with `eos_token_id`
///   (`stored = tok ^ eos`). A zeroed device buffer therefore decodes as EOS-filled history, which is
///   exactly the from-scratch prefill initializer. The graph both decodes on read and re-encodes on
///   write, so the host only ever seeds zeros (or resumes from a prior step's state).
///
/// Returns `(ids [L, ngram_heads], history_out [1, context_len], token_guard_witness)`. The witness
/// is a keepdim last-axis sum of the out-of-range token count (F32); declare it as a validation
/// output so a device run publishes nothing when a token left `[0, vocab)`.
///
/// # Semantics
///
/// Matches [`qwen38_ngram_ids`] bit-exactly at the same constants: `s0 = H[t]`, `s1 = H[t-1]`, and
/// `s2 = (H[t-1] == eos) ? eos : H[t-2]` for the n=3 order (the only segment-reset that can hide a
/// token behind a two-token context window; see the host function's `_shift_right_ignore_eos`).
/// Each order's mixed XOR of `s_i * m_i` is reduced per head by `RemU` and offset into the head's
/// prime slice. The host function remains the reference oracle.
pub fn qwen38_ngram_ids_graph(
    b: &Builder,
    tokens: Traced,
    history: Traced,
    ple: &Qwen4ExpPleConfig,
    tables: &Qwen4ExpNgramTables,
) -> (Traced, Traced, Traced) {
    assert!(
        (2..=3).contains(&ple.ngram_size),
        "ngram_size must be in 2..=3; got {}",
        ple.ngram_size
    );
    let ctx = ple.context_len();
    let heads = ple.ngram_heads();
    let per = ple.heads_per_ngram;
    let token_ty = b.aval(tokens);
    assert_eq!(token_ty.dtype, DType::I32, "tokens must be I32");
    assert_eq!(
        token_ty.shape.len(),
        1,
        "tokens must be a [L] vector, got {:?}",
        token_ty.shape
    );
    let seq_len = token_ty.shape[0];
    let hist_ty = b.aval(history);
    assert_eq!(hist_ty.dtype, DType::I32, "history state must be I32");
    assert_eq!(
        hist_ty.shape,
        vec![1, ctx],
        "history state must be [1, context_len={ctx}], got {:?}",
        hist_ty.shape
    );
    assert!(
        tables.layer_multipliers.len() >= ple.ngram_size,
        "tables.layer_multipliers must cover ngram_size"
    );
    assert_eq!(tables.head_vocab_sizes.len(), heads);
    assert_eq!(tables.head_offsets.len(), heads);
    if ple.ngram_size >= 3 {
        assert!(
            ctx >= 2,
            "ngram_size >= 3 needs context_len >= 2 for the s2 lookback; got {ctx}"
        );
    }

    let eos = ple.eos_token_id;
    let eos_lit = Scalar::I32(eos);

    // Guard token ids into [0, vocab); out-of-range lanes hash as 0 and the witness fails closed.
    let guard = b
        .guard_index_bounds(tokens, ple.vocab_size)
        .map_err(|e| panic!("token bounds guard failed: {e}"))
        .expect("tokens are I32 ranked");
    let tokens = guard.guarded;
    let token_witness = guard.witness;

    // Decode the XOR-encoded state: raw = stored ^ eos. A zero buffer becomes EOS-filled history.
    let hist_flat = b.reshape(history, vec![ctx]);
    let hist_raw = b.binary_scalar(BinOp::Xor, hist_flat, eos_lit);

    // H = previous_context ++ tokens, length ctx + L.
    let h = b.concat(0, &[hist_raw, tokens]);
    let h_len = ctx + seq_len;

    // Token positions only (drop the context_len history-only rows, as the host does).
    let s0 = b.slice(h, 0, ctx, h_len);
    let s1 = b.slice(h, 0, ctx - 1, ctx - 1 + seq_len);
    let s2_raw = if ctx >= 2 {
        Some(b.slice(h, 0, ctx - 2, ctx - 2 + seq_len))
    } else {
        None
    };

    // s2 = (s1 == eos) ? eos : s2_raw. Equality is And(GeU(s1, eos), GeU(eos_like, s1)); the
    // literal is materialized to s1's shape because the planner rejects a literal on the left.
    let s2 = s2_raw.map(|raw| {
        let eos_like = b.binary_scalar(
            BinOp::Or,
            b.binary_scalar(BinOp::And, s1, Scalar::I32(0)),
            eos_lit,
        );
        let ge = b.binary_scalar(BinOp::GeU, s1, eos_lit);
        let le = b.binary(BinOp::GeU, eos_like, s1);
        let eq = b.binary(BinOp::And, ge, le);
        b.select(eq, eos_like, raw)
    });

    let m: Vec<u64> = tables.layer_multipliers.iter().map(|&v| v as u64).collect();
    let p0 = u64_mul_i32_const(b, s0, m[0]);
    let p1 = u64_mul_i32_const(b, s1, m[1]);
    let mixed2 = u64_xor(b, p0, p1);
    let mixed3 = if ple.ngram_size >= 3 {
        let s2 = s2.expect("ngram_size >= 3 provides s2");
        let p2 = u64_mul_i32_const(b, s2, m[2]);
        Some(u64_xor(b, mixed2, p2))
    } else {
        None
    };

    let mut columns: Vec<Traced> = Vec::with_capacity(heads);
    for head in 0..heads {
        let n = if head < per { 2 } else { 3 };
        let mixed = if n == 2 {
            mixed2
        } else {
            mixed3.expect("n=3 heads exist only when ngram_size >= 3")
        };
        let modulus =
            u32::try_from(tables.head_vocab_sizes[head]).expect("prime head vocab fits u32");
        assert!(modulus > 0, "head {head} modulus must be positive");
        let rem = u64_rem_u32(b, mixed, modulus)
            .unwrap_or_else(|e| panic!("head {head} prime modulus out of range: {e}"));
        let offset = i32::try_from(tables.head_offsets[head]).expect("head offset fits i32");
        let id = b.binary_scalar(BinOp::Add, rem, Scalar::I32(offset));
        // [L] -> [L, 1] so a last-axis concat stacks heads into [L, ngram_heads].
        columns.push(b.reshape(id, vec![seq_len, 1]));
    }
    let ids = b.concat(1, &columns);

    // New state: last ctx tokens of H = H[L .. L+ctx], re-encoded with eos.
    let new_raw = b.slice(h, 0, seq_len, seq_len + ctx);
    let encoded = b.binary_scalar(BinOp::Xor, new_raw, eos_lit);
    let history_out = b.reshape(encoded, vec![1, ctx]);
    (ids, history_out, token_witness)
}

/// `Slot::Token [L]` plus the carried XOR-encoded
/// history state, finishing with that state pair so chunked prefill and decode thread the window.
#[cfg(test)]
pub(crate) fn trace_qwen38_ngram_ids(
    ple: &Qwen4ExpPleConfig,
    tables: &Qwen4ExpNgramTables,
    seq_len: usize,
) -> Graph {
    let ctx = ple.context_len();
    let b = Builder::new();
    let tokens = b.slot(Slot::Token, TensorType::new(vec![seq_len], DType::I32));
    let history = b.state_input(
        "ple.ngram_history",
        TensorType::new(vec![1, ctx], DType::I32),
        StateRole::Recurrent,
    );
    // The composition returns the token-bounds witness for a caller that declares it (whole-model
    // tracer). This standalone graph finishes without validations: the guard still clamps out-of-
    // range lanes to zero for the hash.
    let (ids, history_out, _token_guard) = qwen38_ngram_ids_graph(&b, tokens, history, ple, tables);
    b.finish_with_state(ids, &[(history, history_out)])
}

/// Shared core of [`qwen38_ple_prefill`] and [`qwen38_ple_decode`]: `Qwen4ExpTextPLELayer.forward`
/// (`modeling_qwen4_exp.py` lines 1241-1261) up to its dilated short conv (the only piece whose prefill
/// and decode forms differ):
///
/// ```text
/// embeddings   = ngram_embedding(ngram_ids).flatten(-2)              // [1, L, ple_embed_dim]
/// key_normed   = norm_key(key_proj(embeddings)).unflatten(-1, (hc, H))
/// value        = value_proj(embeddings)                              // [1, L, H]
/// query_normed = norm_query(hidden_states).unflatten(-1, (hc, H))
/// gate         = (key_normed * query_normed).sum(-1, keepdim=True) / sqrt(H)
/// gate         = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()     // SIGNED sqrt
/// gated_value  = sigmoid(gate) * value.unsqueeze(-2)                 // [1, L, hc, H]
/// ```
///
/// Returns `(gated_value.flatten(-2), norm_conv(gated_value.flatten(-2)))`: the real forward feeds the
/// normed copy into the conv and adds the un-normed one back (`output = gated_value +
/// self._short_conv(gated_value_normed, ..)`), so both are needed.
///
/// `hidden_states` is the widened `[1, L, hc_count*H]` Hyper-Connections stream (the real PLE runs before
/// `attn_hyper_connection`, on the multi-stream state; see [`qwen38_ple_prefill`]). The three norms are
/// `Qwen4ExpTextRMSNorm(hc_count*H, group_size=H)` grouped RMSNorms, so they reuse
/// [`qwen38_grouped_rmsnorm`] unchanged, including its `+1.0`-is-the-loader's-job convention.
///
/// # The signed square root
///
/// `poot_graph_ir` has no `Abs`, `Sign`, or `Min`/`clamp` unary op, so: `abs(x) = max(x, -x)`
/// (`BinOp::Max`), `clamp_min(x, m) = max(x, m)` (`binary_scalar`), and `sign(x) = ge(x, 0) - ge(-x, 0)`
/// (`BinOp::Ge`, exactly `0.0`/`1.0`), which reproduces `torch.sign`'s `sign(0) == 0` since both `Ge`
/// terms fire at zero and cancel. Deriving sign as `x / max(|x|, 1e-6)` would be wrong for
/// `0 < |x| < 1e-6`, where it decays toward zero instead of saturating at +-1 while the real `clamp_min`
/// floors the magnitude at `sqrt(1e-6)`.
///
/// The `emb` argument is the already-looked-up `[1, L, ple_embed_dim]` n-gram embedding: the dense
/// synthetic path gathers it from `ple_embedding.ngram_embedding.weight`, and the exact path
/// substitutes [`qwen4exp_ple_scaled_row`] over the sharded E4M3 tables.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_core_with_embeddings(
    b: &Builder,
    hidden_states: Traced,
    emb: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> (Traced, Traced) {
    let shape = b.aval(hidden_states).shape; // [1, L, hc_count*H]
    let l = shape[1];
    let hc_h = hc_count * h;
    let e = ple.ple_embed_dim;

    let key_w = b.constant(
        &format!("{p}.ple.key_proj.weight"),
        TensorType::f32(vec![e, hc_h]),
    );
    let key = linear(b, emb, key_w, None); // [1, L, hc*H]
    let key_norm_w = b.constant(
        &format!("{p}.ple.norm_key.weight"),
        TensorType::f32(vec![hc_h]),
    );
    let key_normed = qwen38_grouped_rmsnorm(b, key, key_norm_w, hc_count, h, eps);

    let value_w = b.constant(
        &format!("{p}.ple.value_proj.weight"),
        TensorType::f32(vec![e, h]),
    );
    let value = linear(b, emb, value_w, None); // [1, L, H]

    let query_norm_w = b.constant(
        &format!("{p}.ple.norm_query.weight"),
        TensorType::f32(vec![hc_h]),
    );
    let query_normed = qwen38_grouped_rmsnorm(b, hidden_states, query_norm_w, hc_count, h, eps);

    let grouped = vec![1, l, hc_count, h];
    let key_g = b.reshape(key_normed, grouped.clone());
    let query_g = b.reshape(query_normed, grouped.clone());
    let prod = b.binary(BinOp::Mul, key_g, query_g);
    let dot = b.reduce(RedOp::Sum, prod, 3, true); // [1, L, hc, 1]
    let gate = b.binary_scalar(BinOp::Mul, dot, Scalar::F32(1.0 / (h as f32).sqrt()));

    // `gate.abs().clamp_min(1e-6).sqrt() * gate.sign()`; see this function's doc comment.
    let neg = b.unary(UnOp::Neg, gate);
    let abs_gate = b.binary(BinOp::Max, gate, neg);
    let clamped = b.binary_scalar(BinOp::Max, abs_gate, Scalar::F32(1e-6));
    let magnitude = b.unary(UnOp::Sqrt, clamped);
    let ge_pos = b.binary_scalar(BinOp::Ge, gate, Scalar::F32(0.0));
    let ge_neg = b.binary_scalar(BinOp::Ge, neg, Scalar::F32(0.0));
    let sign = b.binary(BinOp::Sub, ge_pos, ge_neg);
    let gate_signed = b.binary(BinOp::Mul, magnitude, sign);

    let gate_sig = sigmoid(b, gate_signed); // [1, L, hc, 1]
    let value_row = b.reshape(value, vec![1, l, 1, h]);
    let gated = b.binary(BinOp::Mul, gate_sig, value_row); // [1, L, hc, H], broadcast
    let gated_flat = b.reshape(gated, vec![1, l, hc_h]);

    let conv_norm_w = b.constant(
        &format!("{p}.ple.norm_conv.weight"),
        TensorType::f32(vec![hc_h]),
    );
    let gated_normed = qwen38_grouped_rmsnorm(b, gated_flat, conv_norm_w, hc_count, h, eps);

    (gated_flat, gated_normed)
}

/// One PLE layer's prefill forward: `qwen38_ple_core_with_embeddings` plus the real `_short_conv`
/// (`silu(conv1d(gated_value_normed))`, `modeling_qwen4_exp.py` lines 1223-1239) and the residual form
/// `output = gated_value + short_conv(gated_value_normed)`.
///
/// The conv is depthwise (real `groups=hc_hidden_size`) and dilated (real `dilation=config.ngram_size`),
/// which is [`poot_graph_ir::ops::causal_conv1d_prefill_dilated`] (the non-dilated
/// `causal_conv1d_prefill` delegates to it). The real `F.pad(.., (short_conv_state_len, 0))` plus
/// trailing slice is that function's own causal left-pad.
///
/// `conv_mask` (real `apply_mask_to_padding_states`) has no counterpart: this tracer has no padded batch
/// rows, like every other prefill tracer here.
///
/// Returns the PLE contribution `[1, L, hc_count*H]`; the caller adds it to the widened stream (real
/// `hidden_states = hidden_states + self.ple(..)`, `Qwen4ExpTextDecoderLayer.forward` lines 1291-1293).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_prefill(
    b: &Builder,
    hidden_states: Traced,
    ngram_ids: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    tables: &Qwen4ExpNgramTables,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> Traced {
    let emb = dense_ngram_embeddings(b, hidden_states, ngram_ids, p, ple, tables);
    qwen38_ple_prefill_with_embeddings(b, hidden_states, emb, p, ple, hc_count, h, eps)
}

/// Dense-table n-gram embedding lookup shared by the synthetic PLE path: gather
/// `[padded_vocab, head_dim_per_ngram]` with `[L, ngram_heads]` ids and flatten to
/// `[1, L, ple_embed_dim]`.
#[allow(clippy::too_many_arguments)]
fn dense_ngram_embeddings(
    b: &Builder,
    hidden_states: Traced,
    ngram_ids: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    tables: &Qwen4ExpNgramTables,
) -> Traced {
    let shape = b.aval(hidden_states).shape;
    let l = shape[1];
    let table = b.constant(
        &format!("{p}.ple.ple_embedding.ngram_embedding.weight"),
        TensorType::f32(vec![tables.padded_vocab_size, ple.head_dim_per_ngram()]),
    );
    let looked = b.gather(table, 0, ngram_ids);
    b.reshape(looked, vec![1, l, ple.ple_embed_dim])
}

/// [`qwen38_ple_prefill`] with the n-gram embedding already looked up (exact sharded PLE path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_prefill_with_embeddings(
    b: &Builder,
    hidden_states: Traced,
    emb: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> Traced {
    qwen38_ple_prefill_with_embeddings_and_cache(b, hidden_states, emb, p, ple, hc_count, h, eps).0
}

/// [`qwen38_ple_prefill_with_embeddings`] plus the final short-conv cache so prefill's carried
/// state layout matches decode's (Card 449 H0b). The cache holds the last
/// `(K-1)*dilation` raw (pre-conv) `gated_normed` positions, the same window
/// [`causal_conv1d_decode_dilated`] consumes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_prefill_with_embeddings_and_cache(
    b: &Builder,
    hidden_states: Traced,
    emb: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> (Traced, Traced) {
    let (gated, gated_normed) =
        qwen38_ple_core_with_embeddings(b, hidden_states, emb, p, ple, hc_count, h, eps);
    let k = ple.ple_conv_kernel_size;
    let dilation = ple.conv_dilation();
    let pad = (k - 1) * dilation;
    let conv_w = b.constant(
        &format!("{p}.ple.conv1d.weight"),
        TensorType::f32(vec![k, hc_count * h]),
    );
    let conv = causal_conv1d_prefill_dilated(b, gated_normed, conv_w, k, dilation);
    let act = silu(b, conv);
    let ple_out = b.binary(BinOp::Add, gated, act);
    let hc_h = hc_count * h;
    let l = b.aval(gated_normed).shape[1];
    let cc_out = if pad == 0 {
        // Degenerate K=1: the decode cache window is empty. Emit a length-0 slice so the state
        // pair still exists and matches `conv_state_len() == 0`.
        b.slice(gated_normed, 1, 0, 0)
    } else {
        let first = b.slice(gated_normed, 1, 0, 1);
        let zero_slice = b.binary_scalar(BinOp::Mul, first, Scalar::F32(0.0));
        let zeros = b.broadcast(zero_slice, vec![1, pad, hc_h]);
        let x_pad = b.concat(1, &[zeros, gated_normed]);
        b.slice(x_pad, 1, l, l + pad)
    };
    (ple_out, cc_out)
}

/// [`qwen38_ple_prefill`]'s decode counterpart: one token against the PLE short conv's carried cache
/// (real `past_key_values.update_conv_state(.., state_idx=1, conv_kernel_size=self.short_conv_state_len)`;
/// the cache depth is `short_conv_state_len = (K-1)*dilation = 9`, not `K-1 = 3`, because the kernel is
/// dilated). Everything before the conv is `qwen38_ple_core_with_embeddings` unchanged, at `L = 1`.
///
/// Returns `(ple_contribution [1,1,hc_count*H], conv_cache_out [1, conv_state_len, hc_count*H])`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_decode(
    b: &Builder,
    hidden_states: Traced,
    ngram_ids: Traced,
    conv_cache_in: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    tables: &Qwen4ExpNgramTables,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> (Traced, Traced) {
    let emb = dense_ngram_embeddings(b, hidden_states, ngram_ids, p, ple, tables);
    qwen38_ple_decode_with_embeddings(
        b,
        hidden_states,
        emb,
        conv_cache_in,
        p,
        ple,
        hc_count,
        h,
        eps,
    )
}

/// [`qwen38_ple_decode`] with the n-gram embedding already looked up (exact sharded PLE path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_ple_decode_with_embeddings(
    b: &Builder,
    hidden_states: Traced,
    emb: Traced,
    conv_cache_in: Traced,
    p: &str,
    ple: &Qwen4ExpPleConfig,
    hc_count: usize,
    h: usize,
    eps: f32,
) -> (Traced, Traced) {
    let (gated, gated_normed) =
        qwen38_ple_core_with_embeddings(b, hidden_states, emb, p, ple, hc_count, h, eps);
    let k = ple.ple_conv_kernel_size;
    let conv_w = b.constant(
        &format!("{p}.ple.conv1d.weight"),
        TensorType::f32(vec![k, hc_count * h]),
    );
    let (conv, cache_out) = causal_conv1d_decode_dilated(
        b,
        gated_normed,
        conv_w,
        conv_cache_in,
        k,
        ple.conv_dilation(),
    );
    let act = silu(b, conv);
    (b.binary(BinOp::Add, gated, act), cache_out)
}

/// The real PLE embedding partition (`Qwen/Qwen3.8-Flash-Next-FP8`'s
/// `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.*`, pinned by
/// `model.safetensors.index.json` SHA-256 `0419e2c2...`, 17,410,140 bytes; see the card's spec, "Exact PLE
/// source inventory"). These are the checkpoint's declared metadata: 128 equal E4M3 shards tile
/// `[0, 320001536)` exactly, each `[2500012, 160]`.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_SHARDS: usize = 128;

#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_ROWS_PER_SHARD: usize = 2_500_012;

#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_ROW_WIDTH: usize = 160;

/// `QWEN4EXP_PLE_SHARDS * QWEN4EXP_PLE_ROWS_PER_SHARD`, computed so a drift between the two pinned
/// constants above is caught by arithmetic.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_TOTAL_ROWS: usize = QWEN4EXP_PLE_SHARDS * QWEN4EXP_PLE_ROWS_PER_SHARD;

/// `QWEN4EXP_PLE_TOTAL_ROWS * QWEN4EXP_PLE_ROW_WIDTH` (1 byte per E4M3 element); checked against the
/// spec's pinned `51_200_245_760` in `qwen4exp_ple_inventory_matches_exact_partition`.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_E4M3_BYTES: usize = QWEN4EXP_PLE_TOTAL_ROWS * QWEN4EXP_PLE_ROW_WIDTH;

/// `weight_scale` + `key_proj.weight` + `value_proj.weight` + `norm_key`/`norm_query`/`norm_conv.weight`
/// + `conv1d.weight`: the seven BF16 tensors in the spec's inventory table.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_DENSE_BF16_TENSOR_COUNT: usize = 7;

/// Element count of the seven BF16 tensors combined (spec: "The seven BF16 tensors contain 32,839,681
/// elements").
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_DENSE_BF16_ELEMENTS: usize = 32_839_681;

/// `ngram_heads_vocab_sizes` + `ngram_heads_offsets` + `layer_multipliers`: the three I64 tensors.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_I64_TENSOR_COUNT: usize = 3;

#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_I64_ELEMENTS: usize = 35;

/// `QWEN4EXP_PLE_SHARDS + QWEN4EXP_PLE_DENSE_BF16_TENSOR_COUNT + QWEN4EXP_PLE_I64_TENSOR_COUNT` = 138,
/// checked against the spec's pinned literal in `qwen4exp_ple_inventory_matches_exact_partition`.
#[cfg(test)]
pub(crate) const QWEN4EXP_PLE_TOTAL_TENSORS: usize =
    QWEN4EXP_PLE_SHARDS + QWEN4EXP_PLE_DENSE_BF16_TENSOR_COUNT + QWEN4EXP_PLE_I64_TENSOR_COUNT;

/// The contiguous, non-overlapping row range shard `shard` of `QWEN4EXP_PLE_SHARDS` covers within the
/// combined `[0, QWEN4EXP_PLE_TOTAL_ROWS)` table. Source shard order (filesystem/inventory order) is not
/// assumed to equal logical embedding-row order (spec: "It shall not assume filesystem shard order
/// equals embedding shard order"); every caller derives the tiling here.
#[cfg(test)]
pub(crate) fn qwen4exp_ple_shard_row_range(shard: usize) -> std::ops::Range<usize> {
    assert!(
        shard < QWEN4EXP_PLE_SHARDS,
        "shard {shard} is out of range 0..{QWEN4EXP_PLE_SHARDS}"
    );
    let start = shard * QWEN4EXP_PLE_ROWS_PER_SHARD;
    start..(start + QWEN4EXP_PLE_ROWS_PER_SHARD)
}

/// One PLE embedding shard's Card 359 E4M3 owner and the logical row range it covers, in logical shard
/// order. `rows` is `qwen4exp_ple_shard_row_range(shard)` on the production path; a bounded test fixture
/// uses its own small contiguous ranges (see the sharded-lookup tests).
#[derive(Clone, Debug)]
pub struct Qwen4ExpPleShardRange {
    /// The graph constant name this shard's E4M3 table is bound under.
    pub linear_id: String,
    pub rows: std::ops::Range<usize>,
}

/// Card 363's sharded E4M3 PLE embedding lookup for one n-gram id (FR-009, SC-005): for each shard in
/// `shards`, compute a masked local row index that is always in range for that shard's table (clamped to
/// `0` when the shard does not own `id`), decode that row through a `Gather{axis: 0}` + `Cast{to: F32}`
/// pair, zero every non-owning branch's decoded row, and sum across shards. Exactly one branch survives
/// for an id a shard owns.
///
/// **Why `Gather` + `Cast`, not `OpKind::DenseRowGather`.** `DenseRowGather` is compiler-only: `Builder`
/// exposes no method for it (`op.rs`'s doc comment; Card 405). A tracer builds the ordinary primitive
/// chain a table lookup is, and the generic pass `poot_graph_plan::fold_dense_bf16_row_gathers`
/// (`transform.rs:653`) rewrites it into `OpKind::DenseRowGather`'s E4M3 row once the graph reaches the
/// production planner (it reads `DENSE_ROW_GATHER_SOURCE_DTYPES` generically, `transform.rs:701-702`,
/// which Card 405 extended to admit `DType::E4M3FN`). This function's bounded oracle test calls
/// `fold_dense_bf16_row_gathers` explicitly before evaluating, so it exercises `DenseRowGather`'s E4M3
/// eval arm rather than the generic `Gather` dispatch (AGENTS.md: consuming a production-produced value
/// is not exercising the production code).
///
/// Returns the decoded, unscaled `[row_width]` row; see [`qwen4exp_ple_scaled_row`] for the scale multiply.
pub(crate) fn qwen4exp_ple_sharded_row_unscaled(
    b: &Builder,
    id: Traced,
    shards: &[Qwen4ExpPleShardRange],
    row_width: usize,
) -> Traced {
    assert!(!shards.is_empty(), "at least one shard is required");
    let mut total: Option<Traced> = None;
    for shard in shards {
        let rows_in_shard = shard.rows.len();
        assert!(
            rows_in_shard > 0,
            "shard {} has an empty row range",
            shard.linear_id
        );
        let offset = i32::try_from(shard.rows.start)
            .expect("shard row offset fits i32 for this bounded/real row count");
        let last = i32::try_from(rows_in_shard - 1).expect("shard row count fits i32");

        // local_row = id - offset; in range iff 0 <= local_row <= last.
        let local_row = b.binary_scalar(BinOp::Sub, id, Scalar::I32(offset));
        let lo_ok = b.binary_scalar(BinOp::Ge, local_row, Scalar::I32(0));
        // hi_ok = (last - local_row) >= 0, computed without UnOp::Neg (I32 rejects it; op.rs's Unary infer
        // admits only Not/Clz for I32): negate via `* -1` instead, which I32 Mul admits.
        let over = b.binary_scalar(BinOp::Sub, local_row, Scalar::I32(last));
        let neg_over = b.binary_scalar(BinOp::Mul, over, Scalar::I32(-1));
        let hi_ok = b.binary_scalar(BinOp::Ge, neg_over, Scalar::I32(0));
        let mask = b.binary(BinOp::Mul, lo_ok, hi_ok);
        // Clamp to a row that exists in this shard's table whether or not it owns `id`: when mask=0 the
        // safe row is 0 (valid for a nonempty shard); when mask=1, lo_ok/hi_ok already proved local_row
        // is in [0, rows_in_shard).
        let safe_row = b.binary(BinOp::Mul, local_row, mask);

        let table = b.constant(
            &shard.linear_id,
            TensorType::new(vec![rows_in_shard, row_width], DType::E4M3FN),
        );
        let gathered = b.gather(table, 0, safe_row);
        let decoded = b.cast(gathered, DType::F32);

        // Expand the ownership mask across the row width. `id` may be a scalar (one lookup) or a
        // `[n]` batch (the whole-model PLE path); either way `mask` matches `id`'s shape and
        // `decoded` is `id.shape ++ [row_width]`.
        let decoded_shape = b.aval(decoded).shape;
        let mut mask_unsq = b.aval(mask).shape;
        mask_unsq.push(1);
        let mask_col = b.reshape(b.cast(mask, DType::F32), mask_unsq);
        let mask_row = b.broadcast(mask_col, decoded_shape);
        let masked = b.binary(BinOp::Mul, decoded, mask_row);
        total = Some(match total {
            None => masked,
            Some(acc) => b.binary(BinOp::Add, acc, masked),
        });
    }
    total.expect("shards is non-empty, asserted above")
}

/// [`qwen4exp_ple_sharded_row_unscaled`] plus the shared BF16-decoded scale, applied as an ordinary `Mul`
/// after the gather and never fused into the gather primitive (Card 405's scope). A gather that reads
/// the right row with the wrong scale and one that reads the wrong row with the right scale are different
/// bugs; separating this function from [`qwen4exp_ple_sharded_row_unscaled`] lets a test tell them apart
/// (see the sharded-lookup tests).
pub(crate) fn qwen4exp_ple_scaled_row(
    b: &Builder,
    id: Traced,
    shards: &[Qwen4ExpPleShardRange],
    row_width: usize,
    scale: Traced,
) -> Traced {
    let row = qwen4exp_ple_sharded_row_unscaled(b, id, shards, row_width);
    // Broadcast the scalar/table scale across the full row shape so a batched `[n, row_width]`
    // lookup and the original scalar-id `[row_width]` path share one multiply.
    let scale_row = b.broadcast(scale, b.aval(row).shape);
    b.binary(BinOp::Mul, row, scale_row)
}
