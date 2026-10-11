//! The per-layer embedding (PLE) block of Qwen3.8-Flash-Next (`Qwen4ExpTextPLELayer`): a hashed
//! n-gram lookup of the step's tokens, gated into the multi-stream residual by a query over the
//! stream, smoothed by a dilated depthwise causal short conv, and added back to the stream before
//! the layer's own blocks.
//!
//! ```text
//! ids          = hash(tokens, carried history)                       // [rows, tokens, heads]
//! emb          = table[ids].flatten(-2)                              // [rows, tokens, embed_dim]
//! key          = grouped_norm(key_proj(emb))                         // [rows, tokens, streams * width]
//! value        = value_proj(emb)                                     // [rows, tokens, width]
//! query        = grouped_norm(x)
//! gate         = sum_width(key * query) / sqrt(width)                // per stream
//! gated        = sigmoid(sign(gate) * sqrt(max(|gate|, 1e-6))) * value
//! out          = gated + silu(conv(grouped_norm(gated)))
//! ```
//!
//! The hash reads the previous `ngram_size - 1` tokens, which the step carries as a recurrent
//! history (XOR-encoded with the end-of-sequence id, so a zeroed state decodes as an
//! end-of-sequence prefix, the start of a sequence), and the conv carries its own window: a prompt
//! prefilled in chunks continues where the previous chunk stopped, and one token is a decode step.
//! A token outside the vocabulary leaves a [`PleWitness`] non-zero, so the executor refuses the
//! step before it reads any output, and the hash reads a zero id in its place.
//!
//! The table is one dense `[rows, head_dim]` weight, or row-range shards of it (the checkpoint's E4M3
//! shards: one buffer cannot hold the whole table). A sharded lookup gathers from every shard at the
//! id's shard-local row, clamped into range where the shard does not own the id, and keeps the owning
//! shard's row with a select, so the ids stay I32 end to end (exact above `2^24`) and a non-owning
//! row never reaches the result. A sharded table carries one scale the rows are multiplied by.

use poot_graph_ir::ops::{self, sigmoid, silu, u64_mul_i32_const, u64_rem_u32, u64_xor};
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, UnOp, ValidationId};
use poot_quant::weights::{PleRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::conv::{ConvParams, ConvWeights, causal_conv};
use super::linear::{Weight, WeightError, linear};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;

/// SplitMix64's constants and the multiplier seed's prime, from the reference model's hash.
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const PRIME_1: u64 = 10007;

/// The first validation id of the PLE witnesses: above every router witness id
/// ([`WitnessKind::id`](super::witness::WitnessKind::id) is `layer * 3 + kind`, far below `2^31`).
const WITNESS_BASE: u32 = 1 << 31;

/// The run-time check of a PLE layer: the step's token ids are in `0..vocab`. Its validation id is
/// `2^31 + layer`, so the id alone names the layer ([`PleWitness::decode`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleWitness;

impl PleWitness {
    /// The validation id of the token check of the PLE on `layer`.
    pub fn id(layer: usize) -> ValidationId {
        let layer = u32::try_from(layer)
            .ok()
            .filter(|&layer| layer < WITNESS_BASE)
            .expect("a layer index below 2^31: no checkpoint has that many layers");
        ValidationId(WITNESS_BASE + layer)
    }

    /// The layer of an id [`Self::id`] minted; `None` for any other id.
    pub fn decode(id: ValidationId) -> Option<usize> {
        id.0.checked_sub(WITNESS_BASE).map(|layer| layer as usize)
    }
}

/// What a PLE layer's config fixes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PleSpec {
    /// The model's hidden width, and how many parallel streams the residual carries.
    pub width: usize,
    pub streams: usize,
    /// The width of the concatenated per-head n-gram embedding.
    pub embed_dim: usize,
    /// The short conv's tap count; its dilation is `ngram_size`.
    pub conv_kernel: usize,
    /// The largest n-gram order (2 or 3).
    pub ngram_size: usize,
    /// Hash heads per n-gram order.
    pub heads_per_ngram: usize,
    /// The floor of each head's prime vocabulary slice.
    pub ngram_vocab_base: u64,
    /// The table's row count is padded up to a multiple of this.
    pub vocab_divisible_by: usize,
    /// The seed of the hash multipliers.
    pub seed: u64,
    pub vocab: usize,
    /// The end-of-sequence id: the fill of the hash's out-of-segment context.
    pub eos: u32,
    pub eps: f32,
    /// This layer's position among the model's PLE layers: it salts the head primes and the
    /// multipliers, so two PLE layers hash differently.
    pub ordinal: usize,
}

/// The host-side tables the reference model derives once per PLE layer: all pure functions of the
/// spec.
#[derive(Clone, Debug, PartialEq, Eq)]
struct NgramTables {
    /// Per hash head, the prime its id is reduced by, and the start of its slice of the table.
    head_vocab: Vec<u32>,
    head_offset: Vec<i32>,
    /// The table's row count: the slices' total padded to `vocab_divisible_by`.
    padded_rows: usize,
    /// `ngram_size` odd 64-bit multipliers.
    multipliers: Vec<u64>,
}

/// Checked PLE parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PleParams {
    spec: PleSpec,
    eos: i32,
    conv: ConvParams,
    tables: NgramTables,
}

impl PleParams {
    pub fn new(spec: PleSpec) -> Result<Self, ParamError> {
        let zero = |field| ParamError::new(field, ConfigReason::Zero);
        for (field, value) in [
            ("hidden_size", spec.width),
            ("hc_count", spec.streams),
            ("ple_embed_dim", spec.embed_dim),
            ("heads_per_ngram", spec.heads_per_ngram),
            ("vocab_size", spec.vocab),
            (
                "make_ngram_vocab_size_divisible_by",
                spec.vocab_divisible_by,
            ),
        ] {
            if value == 0 {
                return Err(zero(field));
            }
        }
        if !(2..=3).contains(&spec.ngram_size) {
            return Err(ParamError::new("ngram_size", ConfigReason::Unsupported));
        }
        if spec.ngram_vocab_base < 2 {
            return Err(ParamError::new(
                "ngram_vocab_size_base",
                ConfigReason::Unsupported,
            ));
        }
        let heads = (spec.ngram_size - 1) * spec.heads_per_ngram;
        if !spec.embed_dim.is_multiple_of(heads) {
            return Err(ParamError::new(
                "ple_embed_dim",
                ConfigReason::NotDivisible { by: heads },
            ));
        }
        if !(spec.eps.is_finite() && spec.eps > 0.0) {
            return Err(ParamError::new(
                "rms_norm_eps",
                ConfigReason::NotFinitePositive,
            ));
        }
        let eos = i32::try_from(spec.eos).map_err(|_| {
            ParamError::new(
                "eos_token_id",
                ConfigReason::Exceeds {
                    max: i32::MAX as usize,
                },
            )
        })?;
        // A token id enters the hash as an I32 and the n-gram ids index the table as I32.
        let vocab_ok = i32::try_from(spec.vocab).is_ok();
        if !vocab_ok {
            return Err(ParamError::new(
                "vocab_size",
                ConfigReason::Exceeds {
                    max: i32::MAX as usize,
                },
            ));
        }
        let conv = ConvParams::new(spec.streams * spec.width, spec.conv_kernel, spec.ngram_size)?;
        let tables = ngram_tables(&spec, heads)?;
        Ok(Self {
            spec,
            eos,
            conv,
            tables,
        })
    }

    /// One hash head per (order in `2..=ngram_size`, head within the order).
    pub fn heads(&self) -> usize {
        (self.spec.ngram_size - 1) * self.spec.heads_per_ngram
    }

    /// The tokens before a step the hash reads: `ngram_size - 1`.
    pub fn context(&self) -> usize {
        self.spec.ngram_size - 1
    }

    /// Each head's share of the embedding width.
    pub fn head_dim(&self) -> usize {
        self.spec.embed_dim / self.heads()
    }

    /// The table's row count.
    pub fn rows(&self) -> usize {
        self.tables.padded_rows
    }

    pub fn spec(&self) -> &PleSpec {
        &self.spec
    }

    fn stream_width(&self) -> usize {
        self.spec.streams * self.spec.width
    }
}

fn splitmix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(SPLITMIX_GAMMA);
    z = (z ^ (z >> 30)).wrapping_mul(SPLITMIX_M1);
    z = (z ^ (z >> 27)).wrapping_mul(SPLITMIX_M2);
    z ^ (z >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value < 4 {
        return true;
    }
    if value.is_multiple_of(2) {
        return false;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

/// The `count`-th prime strictly greater than `start`.
fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut found = 0;
    let mut candidate = start;
    while found < count {
        candidate += 1;
        if is_prime(candidate) {
            found += 1;
        }
    }
    candidate
}

fn ngram_tables(spec: &PleSpec, heads: usize) -> Result<NgramTables, ParamError> {
    // The head slices stack into one table whose row ids are I32, and each prime reduces a packed
    // u64 through `u64_rem_u32`.
    let too_big = |field| {
        ParamError::new(
            field,
            ConfigReason::Exceeds {
                max: i32::MAX as usize,
            },
        )
    };
    let mut head_vocab = Vec::with_capacity(heads);
    let mut head_offset = Vec::with_capacity(heads);
    let mut total: u64 = 0;
    for head in 0..heads {
        let global = spec.ordinal * heads + head;
        let prime = nth_prime_after(spec.ngram_vocab_base - 1, global + 1);
        head_vocab.push(u32::try_from(prime).map_err(|_| too_big("ngram_vocab_size_base"))?);
        head_offset.push(i32::try_from(total).map_err(|_| too_big("ngram_vocab_size_base"))?);
        total += prime;
    }
    let total = usize::try_from(total).map_err(|_| too_big("ngram_vocab_size_base"))?;
    let padded_rows = total.div_ceil(spec.vocab_divisible_by) * spec.vocab_divisible_by;
    if i32::try_from(padded_rows).is_err() {
        return Err(too_big("ngram_vocab_size_base"));
    }
    let multiplier_max = i64::MAX as u64 / spec.vocab as u64;
    let half_bound = (multiplier_max / 2).max(1);
    let base_seed = spec
        .seed
        .wrapping_add(PRIME_1.wrapping_mul(spec.ordinal as u64));
    let multipliers = (0..spec.ngram_size)
        .map(|index| {
            let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index as u64 + 1));
            2 * (splitmix64(value) % half_bound) + 1
        })
        .collect();
    Ok(NgramTables {
        head_vocab,
        head_offset,
        padded_rows,
        multipliers,
    })
}

/// The n-gram table: one dense weight, or consecutive row-range shards under one scale.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Table {
    Dense(Weight),
    Sharded { shards: Vec<Shard>, scale: Weight },
}

/// One shard of a sharded table: rows `start..start + rows` of the logical table.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Shard {
    weight: Weight,
    start: i32,
    rows: i32,
}

/// A PLE layer's weights: the n-gram table, the key and value projections, the three grouped norm
/// scales and the short conv.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PleWeights {
    table: Table,
    key: Weight,
    value: Weight,
    key_norm: Weight,
    query_norm: Weight,
    conv_norm: Weight,
    conv: ConvWeights,
}

impl PleWeights {
    /// The layer's weights from `map`. The table is the mapped [`PleRole::Table`], or, when that is
    /// absent, the mapped [`PleRole::TableShard`]s in ordinal order (shard `i` follows shards `0..i`)
    /// with their [`PleRole::TableScale`]; the shards must tile exactly `p.rows()` rows.
    pub fn new(map: &WeightMap, layer: usize, p: &PleParams) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Ple(role));
        let weight = |role, shape: &[usize]| Weight::new(map, id(role), shape);
        let (stream, width, embed) = (p.stream_width(), p.spec.width, p.spec.embed_dim);
        let table_shape = [p.rows(), p.head_dim()];
        let table = if map.handle(id(PleRole::Table)).is_none()
            && map.handle(id(PleRole::TableShard(0))).is_some()
        {
            Table::sharded(map, layer, p)?
        } else {
            Table::Dense(Weight::lookup_table(map, id(PleRole::Table), &table_shape)?)
        };
        Ok(Self {
            table,
            key: weight(PleRole::Key, &[stream, embed])?,
            value: weight(PleRole::Value, &[width, embed])?,
            key_norm: weight(PleRole::KeyNorm, &[stream])?,
            query_norm: weight(PleRole::QueryNorm, &[stream])?,
            conv_norm: weight(PleRole::ConvNorm, &[stream])?,
            conv: ConvWeights::new(map, id(PleRole::Conv), None, &p.conv)?,
        })
    }
}

impl Table {
    fn sharded(map: &WeightMap, layer: usize, p: &PleParams) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Ple(role));
        let width = p.head_dim();
        let mut shards = Vec::new();
        let mut start = 0usize;
        while let Some(handle) = map.handle(id(PleRole::TableShard(shards.len() as u32))) {
            let rows = handle.shape.first().copied().unwrap_or(0);
            let weight = Weight::lookup_table(
                map,
                id(PleRole::TableShard(shards.len() as u32)),
                &[rows, width],
            )?;
            if start + rows > p.rows() {
                // A shard past the table's end: report the table the shards were meant to tile.
                return Err(WeightError::Shape {
                    id: id(PleRole::Table),
                    expected: vec![p.rows(), width],
                    found: vec![start + rows, width],
                });
            }
            let fits = "the table's rows fit i32: checked by PleParams::new";
            shards.push(Shard {
                weight,
                start: i32::try_from(start).expect(fits),
                rows: i32::try_from(rows).expect(fits),
            });
            start += rows;
        }
        if start != p.rows() {
            return Err(WeightError::Shape {
                id: id(PleRole::Table),
                expected: vec![p.rows(), width],
                found: vec![start, width],
            });
        }
        let scale = Weight::new(map, id(PleRole::TableScale), &[])?;
        Ok(Table::Sharded { shards, scale })
    }

    /// The rows `ids` (`[..]` I32, every id below the table's row count) select, widened to F32:
    /// `[.., head_dim]`.
    fn lookup(&self, b: &Builder, ids: Traced) -> Traced {
        let widen = |rows: Traced| {
            if b.aval(rows).dtype == DType::F32 {
                rows
            } else {
                b.cast(rows, DType::F32)
            }
        };
        match self {
            Table::Dense(table) => widen(b.gather(table.declare(b), 0, ids)),
            Table::Sharded { shards, scale } => {
                let mut picked: Option<Traced> = None;
                for shard in shards {
                    // The shard-local row, and whether this shard does not own it: an unsigned
                    // compare sees a negative local row as past the end.
                    let local = b.binary_scalar(BinOp::Sub, ids, Scalar::I32(shard.start));
                    let foreign = b.binary_scalar(BinOp::GeU, local, Scalar::I32(shard.rows));
                    let owned = b.binary_scalar(BinOp::Xor, foreign, Scalar::I32(1));
                    // A foreign id reads row 0, which every shard has.
                    let row = b.binary(BinOp::Mul, local, owned);
                    let rows = widen(b.gather(shard.weight.declare(b), 0, row));
                    picked = Some(match picked {
                        None => rows,
                        Some(before) => {
                            let shape = b.aval(rows).shape;
                            let mut flag = b.aval(foreign).shape;
                            flag.push(1);
                            let foreign = b.broadcast(b.reshape(foreign, flag), shape);
                            b.select(foreign, before, rows)
                        }
                    });
                }
                let picked = picked.expect("a sharded table has at least one shard");
                let scale = b.broadcast(scale.declare_f32(b), b.aval(picked).shape);
                b.binary(BinOp::Mul, picked, scale)
            }
        }
    }
}

/// A grouped RMSNorm of `x` (`[.., groups * width]`): each of the `groups` runs of `width` values
/// is normalized by its own mean square, then the whole row is scaled by `1 + scale` (the
/// reference model's `Qwen4ExpTextRMSNorm(dim, group_size)`: the scale has no group structure and is
/// stored as an offset from one).
pub fn grouped_norm(b: &Builder, x: Traced, scale: &Weight, groups: usize, eps: f32) -> Traced {
    let shape = b.aval(x).shape;
    let width = shape[shape.len() - 1] / groups;
    let mut grouped = shape[..shape.len() - 1].to_vec();
    grouped.extend([groups, width]);
    let normed = b.reshape(
        ops::rmsnorm_no_weight(b, b.reshape(x, grouped), eps),
        shape.clone(),
    );
    let scale = b.binary_scalar(BinOp::Add, scale.declare_f32(b), Builder::f32(1.0));
    b.binary(BinOp::Mul, normed, b.broadcast(scale, shape))
}

/// The hash ids of one row's `tokens` (`[t]` I32, every id in range) from its `history`
/// (`[context]` I32, XOR-encoded): `[t, heads]`, and the row's next history `[context]`.
fn row_ids(b: &Builder, tokens: Traced, history: Traced, p: &PleParams) -> (Traced, Traced) {
    let (ctx, heads) = (p.context(), p.heads());
    let t = b.aval(tokens).shape[0];
    let eos = Scalar::I32(p.eos);

    // The decoded history, then the tokens: `H = previous_context ++ tokens`. A zero state decodes
    // as an end-of-sequence prefix.
    let decoded = b.binary_scalar(BinOp::Xor, history, eos);
    let h = b.concat(0, &[decoded, tokens]);
    let s0 = b.slice(h, 0, ctx, ctx + t);
    let s1 = b.slice(h, 0, ctx - 1, ctx - 1 + t);
    // `s2 = (s1 == eos) ? eos : H[t - 2]`: a context that ends a segment hides what precedes it.
    // Equality is `And(GeU(s1, eos), GeU(eos, s1))`, the fill materialized to `s1`'s shape because
    // the planner rejects a literal on the left.
    let s2 = (p.spec.ngram_size >= 3).then(|| {
        let behind = b.slice(h, 0, ctx - 2, ctx - 2 + t);
        let eos_like = b.binary_scalar(
            BinOp::Or,
            b.binary_scalar(BinOp::And, s1, Scalar::I32(0)),
            eos,
        );
        let ge = b.binary_scalar(BinOp::GeU, s1, eos);
        let le = b.binary(BinOp::GeU, eos_like, s1);
        b.select(b.binary(BinOp::And, ge, le), eos_like, behind)
    });

    let m = &p.tables.multipliers;
    let mixed2 = u64_xor(
        b,
        u64_mul_i32_const(b, s0, m[0]),
        u64_mul_i32_const(b, s1, m[1]),
    );
    let mixed3 = s2.map(|s2| u64_xor(b, mixed2, u64_mul_i32_const(b, s2, m[2])));

    let columns: Vec<Traced> = (0..heads)
        .map(|head| {
            let mixed = if head < p.spec.heads_per_ngram {
                mixed2
            } else {
                mixed3.expect("order-3 heads exist only when ngram_size is 3")
            };
            let rem = u64_rem_u32(b, mixed, p.tables.head_vocab[head])
                .expect("head primes are below 2^28: checked by PleParams::new");
            let id = b.binary_scalar(BinOp::Add, rem, Scalar::I32(p.tables.head_offset[head]));
            b.reshape(id, vec![t, 1])
        })
        .collect();

    // The next history is the last `context` entries of `H`, re-encoded.
    let next = b.binary_scalar(BinOp::Xor, b.slice(h, 0, t, t + ctx), eos);
    (b.concat(1, &columns), next)
}

/// The n-gram ids of the step's tokens, `[rows, tokens, heads]` I32; carries each row's history as
/// the step's state `ple.l{layer}.history` and declares the token-range witness.
fn ngram_ids(b: &Builder, step: &Step, layer: usize, p: &PleParams) -> Traced {
    let shape = step.shape();
    let (rows, t) = (shape.rows.get(), shape.tokens.get());
    let (ctx, heads) = (p.context(), p.heads());
    let guard = b.guard_index_bounds(step.tokens(), p.spec.vocab).expect(
        "step tokens are ranked I32 and the vocabulary fits i32: checked by PleParams::new",
    );
    step.validate(
        PleWitness::id(layer),
        format!("ple_tokens_in_vocab.l{layer}"),
        guard.witness,
    );
    let history = step.recurrent_state(b, &format!("ple.l{layer}.history"), DType::I32, &[ctx]);
    let current = history.current();
    let mut ids = Vec::with_capacity(rows);
    let mut next = Vec::with_capacity(rows);
    for row in 0..rows {
        let tokens = b.reshape(b.slice(guard.guarded, 0, row, row + 1), vec![t]);
        let before = b.reshape(b.slice(current, 0, row, row + 1), vec![ctx]);
        let (row_ids, row_next) = row_ids(b, tokens, before, p);
        ids.push(b.reshape(row_ids, vec![1, t, heads]));
        next.push(b.reshape(row_next, vec![1, ctx]));
    }
    step.carry_recurrent(b, history, b.concat(0, &next));
    b.concat(0, &ids)
}

/// The PLE layer's contribution to the multi-stream residual `x` (`[rows, tokens, streams *
/// width]`) at `layer`, the same shape: the caller adds it to the stream. Carries the layer's
/// n-gram history and conv window.
pub fn ple(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &PleWeights,
    p: &PleParams,
) -> Traced {
    let shape = step.shape();
    let (rows, t) = (shape.rows.get(), shape.tokens.get());
    let (streams, width) = (p.spec.streams, p.spec.width);
    let stream = p.stream_width();
    let eps = p.spec.eps;

    let ids = ngram_ids(b, step, layer, p);
    let looked = w.table.lookup(b, ids); // [rows, tokens, heads, head_dim]
    let emb = b.reshape(looked, vec![rows, t, p.spec.embed_dim]);

    let key = grouped_norm(b, linear(b, emb, &w.key, None), &w.key_norm, streams, eps);
    let value = linear(b, emb, &w.value, None); // [rows, tokens, width]
    let query = grouped_norm(b, x, &w.query_norm, streams, eps);

    let grouped = vec![rows, t, streams, width];
    let dot = b.reduce(
        RedOp::Sum,
        b.binary(
            BinOp::Mul,
            b.reshape(key, grouped.clone()),
            b.reshape(query, grouped),
        ),
        3,
        true,
    ); // [rows, tokens, streams, 1]
    let gate = b.binary_scalar(BinOp::Mul, dot, Scalar::F32(1.0 / (width as f32).sqrt()));

    // `sign(gate) * sqrt(max(|gate|, 1e-6))`. The IR has no `Abs` or `Sign`: `|g| = max(g, -g)`, and
    // `sign(g) = [g >= 0] - [-g >= 0]`, which is zero at zero as `torch.sign` is (both terms fire
    // there and cancel). Deriving the sign as `g / max(|g|, 1e-6)` would decay to zero for
    // `0 < |g| < 1e-6` where the reference saturates at one.
    let negated = b.unary(UnOp::Neg, gate);
    let magnitude = b.unary(
        UnOp::Sqrt,
        b.binary_scalar(
            BinOp::Max,
            b.binary(BinOp::Max, gate, negated),
            Scalar::F32(1e-6),
        ),
    );
    let sign = b.binary(
        BinOp::Sub,
        b.binary_scalar(BinOp::Ge, gate, Scalar::F32(0.0)),
        b.binary_scalar(BinOp::Ge, negated, Scalar::F32(0.0)),
    );
    let gate = sigmoid(b, b.binary(BinOp::Mul, magnitude, sign));
    let gated = b.reshape(
        b.binary(BinOp::Mul, gate, b.reshape(value, vec![rows, t, 1, width])),
        vec![rows, t, stream],
    );

    let normed = grouped_norm(b, gated, &w.conv_norm, streams, eps);
    let conv = causal_conv(
        b,
        step,
        &format!("ple.l{layer}.conv"),
        normed,
        &w.conv,
        &p.conv,
    );
    b.binary(BinOp::Add, gated, silu(b, conv))
}

/// Plain-loop references of the PLE layer, in f64: the hash with its own table derivation and the
/// layer's arithmetic as written in the reference model. The family's whole-model reference reuses
/// them.
#[cfg(test)]
pub(crate) mod oracle {
    use super::PleSpec;

    /// The per-layer tables of the reference hash, derived here by a different route than the
    /// component's (`u128` arithmetic, a divisor scan).
    pub(crate) struct Tables {
        pub(crate) head_vocab: Vec<i64>,
        pub(crate) head_offset: Vec<i64>,
        pub(crate) rows: usize,
        pub(crate) multipliers: Vec<i64>,
    }

    fn prime(n: u64) -> bool {
        n >= 2
            && (2..)
                .take_while(|d| d * d <= n)
                .all(|d| !n.is_multiple_of(d))
    }

    pub(crate) fn tables(spec: &PleSpec) -> Tables {
        let heads = (spec.ngram_size - 1) * spec.heads_per_ngram;
        let (mut head_vocab, mut head_offset, mut total) = (Vec::new(), Vec::new(), 0i64);
        let mut candidates = (spec.ngram_vocab_base..).filter(|&n| prime(n));
        // Layer `o` takes the primes after those of layers `0..o`: skip `o * heads` of them.
        for _ in 0..spec.ordinal * heads {
            candidates.next();
        }
        for _ in 0..heads {
            let size = candidates.next().unwrap() as i64;
            head_offset.push(total);
            head_vocab.push(size);
            total += size;
        }
        let divisor = spec.vocab_divisible_by as i64;
        let rows = ((total + divisor - 1) / divisor * divisor) as usize;
        let mix = |z: u128| -> u64 {
            let mask = (1u128 << 64) - 1;
            let mut z = (z + 0x9E37_79B9_7F4A_7C15) & mask;
            z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask;
            z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask;
            (z ^ (z >> 31)) as u64
        };
        let bound = (i64::MAX as u64 / spec.vocab as u64 / 2).max(1);
        let seed = spec.seed as u128 + 10007 * spec.ordinal as u128;
        let multipliers = (0..spec.ngram_size)
            .map(|i| {
                let value = (seed + 0x9E37_79B9_7F4A_7C15u128 * (i as u128 + 1)) & ((1 << 64) - 1);
                (2 * (mix(value) % bound) + 1) as i64
            })
            .collect();
        Tables {
            head_vocab,
            head_offset,
            rows,
            multipliers,
        }
    }

    /// The ids of `tokens` after the `previous` context (raw tokens): `[tokens][heads]`. Each
    /// shifted token is the end-of-sequence id when the shift reaches back past the nearest
    /// end-of-sequence token, found by scanning backward.
    pub(crate) fn ngram_ids(
        spec: &PleSpec,
        tables: &Tables,
        previous: &[i32],
        tokens: &[i32],
    ) -> Vec<Vec<i32>> {
        let eos = i64::from(spec.eos);
        let history: Vec<i64> = previous.iter().chain(tokens).map(|&t| t.into()).collect();
        let segment_start = |t: usize| {
            (0..t)
                .rev()
                .find(|&u| history[u] == eos)
                .map_or(0, |u| u + 1)
        };
        let shifted = |shift: usize, t: usize| -> i64 {
            match t.checked_sub(shift) {
                Some(src) if shift == 0 || src >= segment_start(t) => history[src],
                _ => eos,
            }
        };
        let wrap = |v: i128| (v & ((1i128 << 64) - 1)) as u64 as i64;
        let per = spec.heads_per_ngram;
        let heads = (spec.ngram_size - 1) * per;
        (previous.len()..history.len())
            .map(|t| {
                let mut ids = vec![0i32; heads];
                for order in 2..=spec.ngram_size {
                    let mut mixed =
                        wrap(i128::from(shifted(0, t)) * i128::from(tables.multipliers[0]));
                    for (position, multiplier) in
                        tables.multipliers.iter().enumerate().take(order).skip(1)
                    {
                        mixed ^= wrap(i128::from(shifted(position, t)) * i128::from(*multiplier));
                    }
                    for j in 0..per {
                        let head = (order - 2) * per + j;
                        let size = tables.head_vocab[head];
                        ids[head] =
                            (((mixed % size) + size) % size + tables.head_offset[head]) as i32;
                    }
                }
                ids
            })
            .collect()
    }

    /// A layer's weights in the checkpoint's own `[out, in]` orientation, flattened.
    pub(crate) struct Layer {
        pub(crate) table: Vec<f64>,
        pub(crate) key: Vec<f64>,
        pub(crate) value: Vec<f64>,
        pub(crate) key_norm: Vec<f64>,
        pub(crate) query_norm: Vec<f64>,
        pub(crate) conv_norm: Vec<f64>,
        pub(crate) conv: Vec<f64>,
    }

    /// The layer's contribution per position, and its normed gated value (what the conv reads).
    pub(crate) struct Contribution {
        pub(crate) out: Vec<Vec<f64>>,
        pub(crate) normed: Vec<Vec<f64>>,
    }

    fn matvec(w: &[f64], out: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..out)
            .map(|o| (0..k).map(|j| w[o * k + j] * x[j]).sum())
            .collect()
    }

    fn grouped(x: &[f64], scale: &[f64], groups: usize, eps: f64) -> Vec<f64> {
        let width = x.len() / groups;
        (0..x.len())
            .map(|i| {
                let group = &x[i / width * width..(i / width + 1) * width];
                let ms = group.iter().map(|v| v * v).sum::<f64>() / width as f64;
                x[i] / (ms + eps).sqrt() * (1.0 + scale[i])
            })
            .collect()
    }

    fn sign(v: f64) -> f64 {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    }

    pub(crate) fn contribution(
        spec: &PleSpec,
        w: &Layer,
        x: &[Vec<f64>],
        ids: &[Vec<i32>],
    ) -> Contribution {
        let (width, streams, embed) = (spec.width, spec.streams, spec.embed_dim);
        let stream = width * streams;
        let heads = (spec.ngram_size - 1) * spec.heads_per_ngram;
        let head_dim = embed / heads;
        let eps = f64::from(spec.eps);
        let (mut gated, mut normed) = (Vec::new(), Vec::new());
        for (x, ids) in x.iter().zip(ids) {
            let emb: Vec<f64> = ids
                .iter()
                .flat_map(|&id| {
                    w.table[id as usize * head_dim..(id as usize + 1) * head_dim].to_vec()
                })
                .collect();
            let key = grouped(
                &matvec(&w.key, stream, embed, &emb),
                &w.key_norm,
                streams,
                eps,
            );
            let value = matvec(&w.value, width, embed, &emb);
            let query = grouped(x, &w.query_norm, streams, eps);
            let mut row = vec![0.0; stream];
            for s in 0..streams {
                let dot: f64 = (0..width)
                    .map(|i| key[s * width + i] * query[s * width + i])
                    .sum();
                let gate = dot / (width as f64).sqrt();
                let signed = gate.abs().max(1e-6).sqrt() * sign(gate);
                let sigma = 1.0 / (1.0 + (-signed).exp());
                for i in 0..width {
                    row[s * width + i] = sigma * value[i];
                }
            }
            normed.push(grouped(&row, &w.conv_norm, streams, eps));
            gated.push(row);
        }
        let (kernel, dilation) = (spec.conv_kernel, spec.ngram_size);
        let out = (0..x.len())
            .map(|t| {
                (0..stream)
                    .map(|c| {
                        let mut acc = 0.0;
                        for j in 0..kernel {
                            let back = (kernel - 1 - j) * dilation;
                            if t >= back {
                                acc += w.conv[c * kernel + j] * normed[t - back][c];
                            }
                        }
                        gated[t][c] + acc / (1.0 + (-acc).exp())
                    })
                    .collect()
            })
            .collect();
        Contribution { out, normed }
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Graph, Slot};

    use poot_graph_ir::TensorType;

    use super::*;
    use crate::components::standard::oracle::{self as harness, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const LAYER: usize = 1;

    /// A tiny layer: two streams of width four, four hash heads, an eight-wide embedding, with an
    /// end-of-sequence id inside the vocabulary.
    fn spec() -> PleSpec {
        PleSpec {
            width: 4,
            streams: 2,
            embed_dim: 8,
            conv_kernel: 3,
            ngram_size: 3,
            heads_per_ngram: 2,
            ngram_vocab_base: 17,
            vocab_divisible_by: 8,
            seed: 1234,
            vocab: 11,
            eos: 5,
            eps: 1e-6,
            ordinal: 0,
        }
    }

    fn id(role: PleRole) -> WeightId {
        WeightId::layer(LAYER, WeightRole::Ple(role))
    }

    /// The layer's weights around a table that `table` adds. The table's values come from
    /// [`table_values`], not the fixture's seed, so two fixtures differing only in how the table is
    /// stored share every other weight.
    fn fixture_with(p: &PleParams, table: impl FnOnce(&mut Fx)) -> Fx {
        let s = p.spec();
        let (stream, embed) = (s.streams * s.width, s.embed_dim);
        let mut fx = Fx::new(77);
        table(&mut fx);
        fx.bf16(id(PleRole::Key), &[stream, embed]);
        fx.f32(id(PleRole::Value), &[s.width, embed]);
        for role in [PleRole::KeyNorm, PleRole::QueryNorm, PleRole::ConvNorm] {
            fx.f32(id(role), &[stream]);
        }
        fx.bf16(id(PleRole::Conv), &[stream, 1, s.conv_kernel]);
        fx
    }

    fn table_values(p: &PleParams) -> Vec<f32> {
        harness::random(p.rows() * p.head_dim(), 91)
    }

    fn fixture(p: &PleParams) -> Fx {
        fixture_with(p, |fx| {
            fx.set(
                id(PleRole::Table),
                &[p.rows(), p.head_dim()],
                table_values(p),
            )
        })
    }

    fn layer(fx: &Fx) -> oracle::Layer {
        let f64s = |role| -> Vec<f64> { fx.values(id(role)).into_iter().map(f64::from).collect() };
        oracle::Layer {
            table: f64s(PleRole::Table),
            key: f64s(PleRole::Key),
            value: f64s(PleRole::Value),
            key_norm: f64s(PleRole::KeyNorm),
            query_norm: f64s(PleRole::QueryNorm),
            conv_norm: f64s(PleRole::ConvNorm),
            conv: f64s(PleRole::Conv),
        }
    }

    /// Tokens with an end-of-sequence id in the middle, so the segment rule matters.
    const TOKENS: [i32; 6] = [3, 7, 5, 2, 9, 1];

    /// What the PLE layer reads and returns for `tokens` from `start` over `state` (zeros when
    /// empty): the contribution `[tokens, stream]` and the carried history and conv window.
    fn run(
        fx: &Fx,
        p: &PleParams,
        tokens: &[i32],
        start: i32,
        x: &[f32],
        state: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let w = PleWeights::new(&fx.map(), LAYER, p).unwrap();
        let stream = p.stream_width();
        let b = Builder::new();
        let step = Step::new(&b, step_shape(tokens.len(), CAP, LogitRows::All));
        let xs = b.constant("x", TensorType::f32(vec![1, tokens.len(), stream]));
        let y = ple(&b, &step, LAYER, xs, &w, p);
        let g = step.finish(b, y).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let pos: Vec<i32> = (start..).take(tokens.len()).collect();
        fx.eval(
            &g,
            &[("x", x)],
            &[(Slot::Token, tokens), (Slot::Pos, &pos)],
            &state,
        )
    }

    fn rows(x: &[f32], width: usize) -> Vec<Vec<f64>> {
        x.chunks(width)
            .map(|r| r.iter().map(|&v| v.into()).collect())
            .collect()
    }

    /// The reference's contribution for `tokens` after `previous`, and the state it leaves:
    /// `[history (XOR-encoded), conv window]`.
    fn reference(
        fx: &Fx,
        p: &PleParams,
        previous: &[i32],
        tokens: &[i32],
        x: &[f32],
    ) -> (Vec<f64>, Vec<f64>) {
        let s = p.spec();
        let stream = p.stream_width();
        let ids = oracle::ngram_ids(s, &oracle::tables(s), previous, tokens);
        let c = oracle::contribution(s, &layer(fx), &rows(x, stream), &ids);
        (c.out.concat(), c.normed.concat())
    }

    /// SC-006 (component half): the layer equals the independent plain-loop reference over a
    /// prompt with an end-of-sequence id inside it, a prompt prefilled in chunks and one token at a
    /// time equals the whole at every output and every carried element, and each weight matters.
    /// Mutation: index the table with the previous step's ids (shift the ids by one position); the
    /// reference match goes red at the first output.
    #[test]
    fn the_layer_matches_the_reference_and_chunks_continue_it() {
        let p = PleParams::new(spec()).unwrap();
        let fx = fixture(&p);
        let stream = p.stream_width();
        let eos = [5; 2];
        let x = harness::random(TOKENS.len() * stream, 5);
        let (whole, whole_state) = run(&fx, &p, &TOKENS, 0, &x, &[]);
        let (want, _) = reference(&fx, &p, &eos, &TOKENS, &x);
        harness::assert_matches_f64(&whole, &want, 1e-4);

        // The history decodes to the last two tokens; the window is the last six inputs of the conv.
        let decoded: Vec<i32> = whole_state[0].iter().map(|&v| v as i32 ^ 5).collect();
        assert_eq!(decoded, vec![9, 1]);
        let (_, normed) = reference(&fx, &p, &eos, &TOKENS, &x);
        let window = (p.conv.window()) * stream;
        harness::assert_matches_f64(&whole_state[1], &normed[normed.len() - window..], 1e-4);

        for chunks in [vec![3, 3], vec![1; 6], vec![2, 1, 3]] {
            let (mut state, mut out, mut at) = (Vec::new(), Vec::new(), 0);
            for len in chunks {
                let (y, next) = run(
                    &fx,
                    &p,
                    &TOKENS[at..at + len],
                    at as i32,
                    &x[at * stream..(at + len) * stream],
                    &state,
                );
                out.extend(y);
                state = next;
                at += len;
            }
            harness::assert_matches_f64(
                &out,
                &whole.iter().map(|&v| v.into()).collect::<Vec<f64>>(),
                1e-5,
            );
            for (got, whole) in state.iter().zip(&whole_state) {
                harness::assert_matches_f64(
                    got,
                    &whole.iter().map(|&v| v.into()).collect::<Vec<f64>>(),
                    1e-5,
                );
            }
        }

        for role in [
            PleRole::Table,
            PleRole::Key,
            PleRole::Value,
            PleRole::KeyNorm,
            PleRole::QueryNorm,
            PleRole::ConvNorm,
            PleRole::Conv,
        ] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id(role));
            harness::assert_differs(
                &run(&perturbed, &p, &TOKENS, 0, &x, &[]).0,
                &whole,
                id(role),
            );
        }
    }

    /// The ids of `tokens` the graph computes from `history` (raw, zero-state when `None`), and the
    /// next history decoded.
    fn graph_ids(p: &PleParams, tokens: &[i32], history: Option<&[i32]>) -> (Vec<i32>, Vec<i32>) {
        let b = Builder::new();
        let step = Step::new(&b, step_shape(tokens.len(), CAP, LogitRows::All));
        let ids = ngram_ids(&b, &step, LAYER, p);
        let g = step.finish(b, ids).unwrap();
        let encoded: Vec<f32> = match history {
            Some(raw) => raw.iter().map(|&t| (t ^ p.eos) as f32).collect(),
            None => vec![0.0; p.context()],
        };
        let pos: Vec<i32> = (0..tokens.len() as i32).collect();
        let (ids, state) = Fx::new(1).eval(
            &g,
            &[],
            &[(Slot::Token, tokens), (Slot::Pos, &pos)],
            &[&encoded],
        );
        (
            ids.iter().map(|&v| v as i32).collect(),
            state[0].iter().map(|&v| v as i32 ^ p.eos).collect(),
        )
    }

    /// SC-006: the in-graph ids equal the independent hash for three non-degenerate histories (a
    /// fresh start, a nonzero prior, an end-of-sequence id inside the carried window), the three
    /// differ, a zeroed state decodes as the fresh start, and the carried window is the last
    /// `context` tokens. Mutation: seed the hash with the wrong layer ordinal; every id moves.
    #[test]
    fn the_ids_match_the_hash_over_three_histories() {
        let p = PleParams::new(spec()).unwrap();
        let s = p.spec();
        let tables = oracle::tables(s);
        assert_eq!(tables.rows, p.rows());
        let tokens = [3, 7, 5, 2, 9, 1];
        let histories: [[i32; 2]; 3] = [[5, 5], [3, 8], [4, 5]];
        let mut produced = Vec::new();
        for history in histories {
            let (ids, next) = graph_ids(&p, &tokens, Some(&history));
            let want = oracle::ngram_ids(s, &tables, &history, &tokens);
            assert_eq!(ids, want.concat(), "history {history:?}");
            assert_eq!(next, tokens[tokens.len() - 2..], "history {history:?}");
            produced.push(ids);
        }
        assert_ne!(produced[0], produced[1]);
        assert_ne!(produced[1], produced[2]);
        assert_eq!(graph_ids(&p, &tokens, None).0, produced[0]);
        // The id of every head lies inside that head's slice of the table.
        for (t, ids) in produced[1].chunks(p.heads()).enumerate() {
            for (head, &id) in ids.iter().enumerate() {
                let lo = tables.head_offset[head];
                assert!(
                    (lo..lo + tables.head_vocab[head]).contains(&i64::from(id)),
                    "token {t} head {head}: id {id}"
                );
            }
        }
    }

    /// The tables and the ids against numbers produced by a verbatim Python transcription of the
    /// reference model's hash (`_splitmix64`, `_build_layer_multipliers`, `_find_nth_prime_after`
    /// and `Qwen4ExpTextNGramEmbedding.forward`, int64 wraparound explicit). The one check here
    /// independent of the Rust: a drifted wrapping multiply, remainder sign or segment rule would not
    /// reproduce them. The second block pins the derivation at the real config.
    #[test]
    fn the_hash_matches_the_python_goldens() {
        let p = PleParams::new(spec()).unwrap();
        assert_eq!(p.tables.head_vocab, vec![17, 19, 23, 29]);
        assert_eq!(p.tables.head_offset, vec![0, 17, 36, 59]);
        assert_eq!(p.tables.padded_rows, 88);
        assert_eq!(
            p.tables.multipliers,
            vec![73077407744147129, 749479866669413493, 395950012578265759]
        );
        let ids = graph_ids(&p, &[3, 7, 5, 2, 9, 1], None).0;
        assert_eq!(
            ids,
            vec![
                8, 24, 36, 77, 15, 22, 55, 59, 0, 31, 45, 74, 7, 32, 42, 81, 12, 19, 53, 81, 7, 19,
                51, 86
            ]
        );

        let real = PleParams::new(PleSpec {
            width: 4,
            streams: 1,
            embed_dim: 2560,
            conv_kernel: 4,
            heads_per_ngram: 8,
            ngram_vocab_base: 20_000_000,
            vocab_divisible_by: 128,
            vocab: 248_320,
            eos: 248_044,
            ..spec()
        })
        .unwrap();
        assert_eq!(
            real.tables
                .multipliers
                .iter()
                .map(|&m| m as i64)
                .collect::<Vec<_>>(),
            vec![23703573157769, 20109073645365, 8052911324071]
        );
        assert_eq!(real.heads(), 16);
        assert_eq!(real.head_dim(), 160);
        let real_tables = oracle::tables(real.spec());
        assert_eq!(real_tables.rows, real.rows());
        assert_eq!(
            real_tables.multipliers,
            real.tables
                .multipliers
                .iter()
                .map(|&m| m as i64)
                .collect::<Vec<_>>()
        );
        assert_eq!(real.rows(), 320_001_536);
        assert_eq!(real.rows(), 128 * 2_500_012);
        assert_eq!(real.conv.window(), 9);
    }

    /// `g` with its witnesses dropped and `output` as its result: the oracle refuses a failing
    /// validation, so a test that reads a witness reads it as the result.
    fn unvalidated(g: Graph, output: Option<poot_graph_ir::ValueId>) -> Graph {
        let mut draft = poot_graph_ir::GraphEdit::new();
        *draft.values_mut() = g.values().clone();
        *draft.inputs_mut() = g.inputs().to_vec();
        *draft.consts_mut() = g.consts().to_vec();
        *draft.slots_mut() = g.slots().to_vec();
        *draft.eqns_mut() = g.eqns().to_vec();
        draft.set_result(output.unwrap_or(g.result()));
        for (input, next) in g.state() {
            draft.push_state(input, next);
        }
        draft.commit().expect("the same tables stay well formed")
    }

    /// SC-009 (component half): an id outside the vocabulary leaves the witness non-zero, counts
    /// each offender, names its layer, and reads as id zero in the hash; in-range ids leave it zero.
    /// Mutation: declare nothing for the guard witness; the witness lookup below finds no
    /// validation and the row goes red.
    #[test]
    fn a_token_outside_the_vocabulary_trips_the_witness() {
        let p = PleParams::new(spec()).unwrap();
        let eval = |tokens: &[i32], witness: bool| -> Vec<f32> {
            let b = Builder::new();
            let step = Step::new(&b, step_shape(tokens.len(), CAP, LogitRows::All));
            let ids = ngram_ids(&b, &step, LAYER, &p);
            let g = step.finish(b, ids).unwrap();
            let outputs: Vec<_> = g
                .witnesses()
                .map(|witness| (witness.id, witness.value))
                .collect();
            assert_eq!(outputs.len(), 1, "one witness per PLE layer");
            assert_eq!(PleWitness::decode(outputs[0].0), Some(LAYER));
            let g = unvalidated(g, witness.then_some(outputs[0].1));
            let pos: Vec<i32> = (0..tokens.len() as i32).collect();
            let zeros = vec![0.0; p.context()];
            Fx::new(1)
                .eval(
                    &g,
                    &[],
                    &[(Slot::Token, tokens), (Slot::Pos, &pos)],
                    &[&zeros],
                )
                .0
        };
        let witness = |tokens: &[i32]| eval(tokens, true).iter().sum::<f32>();
        assert_eq!(witness(&[0, 3, 10]), 0.0);
        assert_eq!(witness(&[0, 11, 10]), 1.0);
        assert_eq!(witness(&[-1, 11, 99]), 3.0);
        // Out of range hashes as zero: the first token's ids equal those of a zero token.
        assert_eq!(
            eval(&[11, 3], false)[..p.heads()],
            eval(&[0, 3], false)[..p.heads()]
        );
    }

    #[test]
    fn params_refuse_an_impossible_layer() {
        let refuse = |edit: &dyn Fn(&mut PleSpec)| {
            let mut s = spec();
            edit(&mut s);
            PleParams::new(s).unwrap_err()
        };
        assert_eq!(
            refuse(&|s| s.ngram_size = 4),
            ParamError::new("ngram_size", ConfigReason::Unsupported)
        );
        assert_eq!(
            refuse(&|s| s.embed_dim = 9),
            ParamError::new("ple_embed_dim", ConfigReason::NotDivisible { by: 4 })
        );
        assert_eq!(
            refuse(&|s| s.heads_per_ngram = 0),
            ParamError::new("heads_per_ngram", ConfigReason::Zero)
        );
        assert_eq!(
            refuse(&|s| s.conv_kernel = 1),
            ParamError::new("conv_kernel", ConfigReason::Zero)
        );
    }

    /// Row counts of three shards that put a 1-row shard on a row the hash reads at the middle of
    /// `TOKENS`' ids, so every shard (and a boundary on both sides of that row) is read.
    fn split_around_a_read_row(p: &PleParams) -> Vec<usize> {
        let s = p.spec();
        let mut ids = oracle::ngram_ids(s, &oracle::tables(s), &[5, 5], &TOKENS).concat();
        ids.sort_unstable();
        let at = ids[ids.len() / 2] as usize;
        assert!(
            0 < at && at + 1 < p.rows(),
            "row {at} has rows on both sides"
        );
        assert!(
            ids.iter().any(|&id| (id as usize) < at) && ids.iter().any(|&id| id as usize > at),
            "the hash reads the shards on both sides of row {at}"
        );
        vec![at, 1, p.rows() - at - 1]
    }

    /// The fixture storing the table as shards of `rows` rows each, `table` cut into them, under
    /// `scale` (the shards hold `value / scale`, exact for a power of two).
    fn sharded_f32(p: &PleParams, rows: &[usize], scale: f32) -> Fx {
        fixture_with(p, |fx| {
            let (width, table) = (p.head_dim(), table_values(p));
            let mut start = 0;
            for (shard, &count) in rows.iter().enumerate() {
                let cut = table[start * width..(start + count) * width]
                    .iter()
                    .map(|&v| v / scale)
                    .collect();
                fx.set(id(PleRole::TableShard(shard as u32)), &[count, width], cut);
                start += count;
            }
            fx.set(id(PleRole::TableScale), &[], vec![scale]);
        })
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// SC-001 (F32 half): a table stored as three shards (the middle one a single row, so a
    /// boundary sits on both sides of a row the hash reads), under a scale of one half, gives the
    /// layer's output of the single table bit for bit. Mutation: offset a shard's local row by one
    /// (`shard.start + 1` in `Table::lookup`); the rows on the shard edges read their neighbours and
    /// the equality goes red.
    #[test]
    fn a_sharded_f32_table_equals_the_single_table_bit_for_bit() {
        let p = PleParams::new(spec()).unwrap();
        let x = harness::random(TOKENS.len() * p.stream_width(), 5);
        let (single, _) = run(&fixture(&p), &p, &TOKENS, 0, &x, &[]);
        let rows = split_around_a_read_row(&p);
        let (sharded, _) = run(&sharded_f32(&p, &rows, 0.5), &p, &TOKENS, 0, &x, &[]);
        assert_eq!(bits(&sharded), bits(&single), "shards {rows:?}");
        // The scale is read: the same shards under another scale give another output.
        let (rescaled, _) = run(&sharded_f32(&p, &rows, 0.5), &p, &TOKENS, 0, &x, &[]);
        let mut wrong = sharded_f32(&p, &rows, 0.5);
        wrong.set(id(PleRole::TableScale), &[], vec![1.0]);
        harness::assert_differs(
            &run(&wrong, &p, &TOKENS, 0, &x, &[]).0,
            &rescaled,
            id(PleRole::TableScale),
        );
        // Every shard is read.
        for shard in 0..rows.len() as u32 {
            let mut perturbed = sharded_f32(&p, &rows, 0.5);
            perturbed.perturb(id(PleRole::TableShard(shard)));
            harness::assert_differs(
                &run(&perturbed, &p, &TOKENS, 0, &x, &[]).0,
                &sharded,
                id(PleRole::TableShard(shard)),
            );
        }
    }

    /// Finite E4M3FN codes, one per table element.
    fn table_codes(p: &PleParams) -> Vec<u8> {
        table_values(p)
            .iter()
            .map(|&v| match ((v + 0.5) * 255.0) as u8 {
                code if code & 0x7f == 0x7f => 0x38,
                code => code,
            })
            .collect()
    }

    /// SC-001 (E4M3 half): the checkpoint's storage, E4M3FN shards under a scale of one, gives the
    /// layer's output of the same codes as one E4M3 table bit for bit, and the codes decode (the
    /// output differs from the F32 table of the raw random values).
    #[test]
    fn a_sharded_e4m3_table_equals_the_single_table_bit_for_bit() {
        let p = PleParams::new(spec()).unwrap();
        let (width, codes) = (p.head_dim(), table_codes(&p));
        let x = harness::random(TOKENS.len() * p.stream_width(), 5);
        let single = fixture_with(&p, |fx| {
            fx.e4m3(id(PleRole::Table), &[p.rows(), width], codes.clone());
        });
        let rows = split_around_a_read_row(&p);
        let sharded = fixture_with(&p, |fx| {
            let mut start = 0;
            for (shard, &count) in rows.iter().enumerate() {
                let cut = codes[start * width..(start + count) * width].to_vec();
                fx.e4m3(id(PleRole::TableShard(shard as u32)), &[count, width], cut);
                start += count;
            }
            fx.set(id(PleRole::TableScale), &[], vec![1.0]);
        });
        let (want, _) = run(&single, &p, &TOKENS, 0, &x, &[]);
        let (got, _) = run(&sharded, &p, &TOKENS, 0, &x, &[]);
        assert_eq!(bits(&got), bits(&want), "shards {rows:?}");
        harness::assert_differs(
            &run(&fixture(&p), &p, &TOKENS, 0, &x, &[]).0,
            &want,
            id(PleRole::Table),
        );
    }

    /// SC-002: a row id above `2^24`, which an F32 round trip rounds to a neighbour, reads its own row.
    /// A table of `2^24 + 3` one-byte rows is cut in two so the id (`2^24 + 1`) and its F32 round
    /// trip (`2^24`) lie in different shards, with 2.0 only in the id's row. Mutation: round the
    /// ids through F32 before the lookup (`ids` cast to F32 and back in `Table::lookup`); the id
    /// reads 1.0 and the row goes red.
    #[test]
    fn an_id_above_two_to_the_24_reads_its_exact_row() {
        let (id_row, lossy) = ((1usize << 24) + 1, 1usize << 24);
        assert_ne!(
            id_row as f32 as usize, id_row,
            "the premise: 2^24 + 1 is not an F32"
        );
        assert_eq!(id_row as f32 as usize, lossy);
        let (rows, first) = (id_row + 2, id_row);
        let one = 0x38; // 1.0
        let two = 0x40; // 2.0
        let shard = |range: std::ops::Range<usize>| -> Vec<u8> {
            range
                .map(|row| if row == id_row { two } else { one })
                .collect()
        };
        let mut fx = Fx::new(1);
        fx.e4m3(id(PleRole::TableShard(0)), &[first, 1], shard(0..first));
        fx.e4m3(
            id(PleRole::TableShard(1)),
            &[rows - first, 1],
            shard(first..rows),
        );
        fx.set(id(PleRole::TableScale), &[], vec![1.0]);
        let map = fx.map();
        let table = Table::Sharded {
            shards: vec![
                Shard {
                    weight: Weight::lookup_table(&map, id(PleRole::TableShard(0)), &[first, 1])
                        .unwrap(),
                    start: 0,
                    rows: first as i32,
                },
                Shard {
                    weight: Weight::lookup_table(
                        &map,
                        id(PleRole::TableShard(1)),
                        &[rows - first, 1],
                    )
                    .unwrap(),
                    start: first as i32,
                    rows: (rows - first) as i32,
                },
            ],
            scale: Weight::new(&map, id(PleRole::TableScale), &[]).unwrap(),
        };
        let b = Builder::new();
        let ids = b.slot(Slot::Token, TensorType::new(vec![4], DType::I32));
        let rows = table.lookup(&b, ids);
        let g = b.finish(rows).unwrap();
        let wanted = [id_row, lossy, id_row + 1, 0];
        let tokens: Vec<i32> = wanted.iter().map(|&row| row as i32).collect();
        let (got, _) = fx.eval(&g, &[], &[(Slot::Token, &tokens)], &[]);
        assert_eq!(got, vec![2.0, 1.0, 1.0, 1.0]);
    }

    /// SC-003: the real table (`Qwen/Qwen3.8-Flash-Next-FP8`, `ngram_embedding.shard_0..127`, each
    /// `[2500012, 160]` E4M3) is the derived padded table at the real config, so the 128 equal shards
    /// tile it with no gap or overlap: 320,001,536 rows, 51,200,245,760 bytes.
    #[test]
    fn the_real_table_is_the_checkpoints_128_equal_shards() {
        let real = PleParams::new(PleSpec {
            width: 4,
            streams: 1,
            embed_dim: 2560,
            conv_kernel: 4,
            heads_per_ngram: 8,
            ngram_vocab_base: 20_000_000,
            vocab_divisible_by: 128,
            vocab: 248_320,
            eos: 248_044,
            ..spec()
        })
        .unwrap();
        let (shards, rows_per_shard, width) = (128, 2_500_012, 160);
        assert_eq!(real.head_dim(), width);
        assert_eq!(real.rows(), shards * rows_per_shard);
        assert_eq!(real.rows(), 320_001_536);
        assert_eq!(real.rows() * real.head_dim(), 51_200_245_760);
    }

    /// Shards that do not tile the table are refused with the table's shape, whether they fall
    /// short or run past its end; a scale-less set is unmapped.
    #[test]
    fn shards_that_do_not_tile_the_table_are_refused() {
        let p = PleParams::new(spec()).unwrap();
        let rows = split_around_a_read_row(&p);
        let width = p.head_dim();
        let weights = |fx: &Fx| PleWeights::new(&fx.map(), LAYER, &p).unwrap_err();
        let short = [rows[0], rows[1]];
        assert_eq!(
            weights(&sharded_f32(&p, &short, 1.0)),
            WeightError::Shape {
                id: id(PleRole::Table),
                expected: vec![p.rows(), width],
                found: vec![rows[0] + rows[1], width],
            }
        );
        let long = [rows[0], rows[1], rows[2] + 1];
        let over = fixture_with(&p, |fx| {
            let mut start = 0;
            for (shard, &count) in long.iter().enumerate() {
                let cut = vec![0.0; count * width];
                fx.set(id(PleRole::TableShard(shard as u32)), &[count, width], cut);
                start += count;
            }
            assert_eq!(start, p.rows() + 1);
            fx.set(id(PleRole::TableScale), &[], vec![1.0]);
        });
        assert_eq!(
            weights(&over),
            WeightError::Shape {
                id: id(PleRole::Table),
                expected: vec![p.rows(), width],
                found: vec![p.rows() + 1, width],
            }
        );
        let unscaled = fixture_with(&p, |fx| {
            fx.set(
                id(PleRole::TableShard(0)),
                &[p.rows(), width],
                table_values(&p),
            );
        });
        assert_eq!(
            weights(&unscaled),
            WeightError::Unmapped {
                id: id(PleRole::TableScale)
            }
        );
    }
}
