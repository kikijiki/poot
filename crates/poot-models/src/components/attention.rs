//! Dense GQA self-attention over a fixed-capacity KV cache, one body for every phase.
//!
//! A step of `rows` sequences of `tokens` query positions each (decode is `tokens = 1`) projects
//! Q/K/V (with the optional QKV bias), rotates Q and K at their absolute positions, writes K and V
//! into the carried caches, and attends over the whole cache under the causal-over-cache mask built
//! in the graph from `Slot::Pos` (`ops::causal_mask_from_pos`, Card 550): key `j` of a row is
//! visible to that row's query `i` iff `j <= pos[row, i]`. A step's positions are consecutive per
//! row and below the capacity; the driver chooses them.
//!
//! The cache layout is the step's [`KvLayout`]. Contiguous: a private `[rows, kv_heads, capacity,
//! head_dim]` cache per layer, row `r` written at its own first position, declared
//! [`StateRole::Positional`] on axis 2. Paged: one `[pool_slots, kv_heads, head_dim]` pool per
//! layer shared by every row and addressed through the step's slot maps ([`Step::paged`]),
//! declared `Positional` on axis 0; the leading axis of a pool is the slot axis, not a row axis.
//!
//! A contiguous K or V cache stores [`HeadSpan::padded`](super::kv_stride::HeadSpan::padded) positions per
//! head: the capacity, or one position more when `capacity * head_dim * 4` is a multiple of 2048 B (the head
//! stride would put every head of a position on the same DRAM channel, POOT-1150). Attention reads the stored
//! extent; the extra position is never written and the mask hides it. A paged cache keeps its capacity: its
//! stride is the pool's, which in-place paged reads re-lay out.

use poot_graph_ir::{BinOp, Builder, ComputedConst, Scalar, StateRole, TensorType, Traced, ops};
use poot_quant::weights::{AttnRole, WeightId, WeightMap, WeightRole};

use super::kv_stride::{Exemption, HeadSpan};
use super::linear::{Weight, WeightError, linear};
use super::norm::{NormParams, norm};
use super::rope::{RopeParams, apply_rope, apply_rope_sectioned, rope_rows};
use super::standard::{PagedMaps, ParamError, Step};
use crate::model::TraceError;
use crate::model::WiringError;
use crate::model::{ConfigReason, KvLayout};

/// Where a Q/K RMSNorm normalizes (applied before RoPE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QkNorm {
    /// Over each head's `head_dim` dims, one `[head_dim]` scale shared by the heads.
    PerHead(NormParams),
    /// Over the whole projection (`heads * head_dim` for Q, `kv_heads * head_dim` for K).
    Projection(NormParams),
}

/// Checked attention parameters. The base is RoPE-rotated GQA with a `head_dim^-1/2` score scale;
/// each option below adds one thing a family needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttentionParams {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    qkv_bias: bool,
    rope: Option<RopeParams>,
    scale: f32,
    qk_norm: Option<QkNorm>,
    window: Option<usize>,
    softcap: Option<f32>,
    alibi: bool,
    fused_qkv: bool,
    out_bias: bool,
    sinks: bool,
    value_from_key: bool,
    value_norm: Option<NormParams>,
    output_gate: bool,
}

impl AttentionParams {
    /// `heads` query heads over `kv_heads` shared K/V heads (which must divide them), each
    /// `head_dim` wide, with RoPE no wider than the head; the score scale is `head_dim^-1/2`.
    pub fn new(
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        qkv_bias: bool,
        rope: RopeParams,
    ) -> Result<Self, ParamError> {
        let nope = Self::nope(heads, kv_heads, head_dim, qkv_bias)?;
        if rope.rotary() > head_dim {
            return Err(ParamError::new(
                "rotary_dim",
                ConfigReason::Exceeds { max: head_dim },
            ));
        }
        Ok(Self {
            rope: Some(rope),
            ..nope
        })
    }

    /// [`Self::new`] for an attention with no rotary embedding at all (a NoPE model: Nemotron-H's
    /// attention layers), which needs no RoPE width.
    pub fn nope(
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        qkv_bias: bool,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("heads", heads),
            ("kv_heads", kv_heads),
            ("head_dim", head_dim),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if !heads.is_multiple_of(kv_heads) {
            return Err(ParamError::new(
                "heads",
                ConfigReason::NotDivisible { by: kv_heads },
            ));
        }
        Ok(Self {
            heads,
            kv_heads,
            head_dim,
            qkv_bias,
            rope: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            qk_norm: None,
            window: None,
            softcap: None,
            alibi: false,
            fused_qkv: false,
            out_bias: false,
            sinks: false,
            value_from_key: false,
            value_norm: None,
            output_gate: false,
        })
    }

    /// Attend without rotary embedding (a NoPE layer).
    pub fn without_rope(self) -> Self {
        Self { rope: None, ..self }
    }

    /// Rotate by sectioned positions ([`RopeParams::sectioned`]): the step must carry
    /// `Step::with_mrope`. A NoPE attention has no rotation to section.
    pub fn with_sectioned_rope(self, sections: [usize; 3]) -> Result<Self, ParamError> {
        let rope = self
            .rope
            .ok_or(ParamError::new(
                "rope_scaling.mrope_section",
                ConfigReason::Unsupported,
            ))?
            .sectioned(sections)?;
        Ok(Self {
            rope: Some(rope),
            ..self
        })
    }

    /// Whether the rotation is sectioned, so the step must carry `Step::with_mrope`.
    pub fn is_sectioned(&self) -> bool {
        self.rope.is_some_and(|rope| rope.sections().is_some())
    }

    /// Rotate adjacent even/odd dims instead of the two halves (a GGUF's permuted Q/K rows).
    pub fn with_interleaved_rope(self) -> Self {
        Self {
            rope: self.rope.map(RopeParams::interleaved),
            ..self
        }
    }

    /// The same attention with its RoPE rescaled by llama.cpp's per-frequency `factors`
    /// (`rope_freqs.weight`); a NoPE attention has no RoPE to rescale.
    pub fn with_rope_freq_factors(self, factors: &[f32]) -> Result<Self, ParamError> {
        Ok(Self {
            rope: self
                .rope
                .map(|r| r.with_freq_factors(factors))
                .transpose()?,
            ..self
        })
    }

    /// Replace RoPE with ALiBi: a per-head linear distance bias on the scores (BLOOM, MPT).
    pub fn with_alibi(self) -> Self {
        Self {
            rope: None,
            alibi: true,
            ..self
        }
    }

    /// Normalize Q and K before RoPE.
    pub fn with_qk_norm(self, qk_norm: QkNorm) -> Self {
        Self {
            qk_norm: Some(qk_norm),
            ..self
        }
    }

    /// Each query sees only the `window` most recent keys (itself included).
    pub fn with_window(self, window: usize) -> Result<Self, ParamError> {
        if window == 0 {
            return Err(ParamError::new("window", ConfigReason::Zero));
        }
        Ok(Self {
            window: Some(window),
            ..self
        })
    }

    /// Read Q, K and V from one fused projection whose output rows interleave per head
    /// (`q_h, k_h, v_h` for each head; BLOOM) and split it in the graph. Needs plain multi-head
    /// attention (`kv_heads == heads`). The bias, when `qkv_bias`, is the fused projection's.
    pub fn with_fused_qkv(self) -> Result<Self, ParamError> {
        if self.heads != self.kv_heads {
            return Err(ParamError::new("kv_heads", ConfigReason::Unsupported));
        }
        Ok(Self {
            fused_qkv: true,
            ..self
        })
    }

    /// `softcap * tanh(scores / softcap)` on the scaled scores before the mask.
    pub fn with_softcap(self, softcap: f32) -> Result<Self, ParamError> {
        if self.sinks {
            return Err(ParamError::new("softcap", ConfigReason::Unsupported));
        }
        if !(softcap.is_finite() && softcap > 0.0) {
            return Err(ParamError::new("softcap", ConfigReason::NotFinitePositive));
        }
        Ok(Self {
            softcap: Some(softcap),
            ..self
        })
    }

    /// One learned logit per query head joins every row's softmax (gpt-oss): it counts in the row
    /// maximum and the denominator but adds nothing to the weighted sum of V. Not combinable with
    /// a softcap.
    pub fn with_sinks(self) -> Result<Self, ParamError> {
        if self.softcap.is_some() {
            return Err(ParamError::new("sinks", ConfigReason::Unsupported));
        }
        Ok(Self {
            sinks: true,
            ..self
        })
    }

    /// The output projection has a bias.
    pub fn with_out_bias(self) -> Self {
        Self {
            out_bias: true,
            ..self
        }
    }

    /// The query projection also emits a per-head output gate (Qwen3-Next): each head's `2 * head_dim`
    /// rows are its query then its gate, and the attention output is multiplied by `sigmoid(gate)`
    /// before the output projection. The Q projection (and its bias) is `2 * heads * head_dim` wide.
    /// Needs separate projections.
    pub fn with_output_gate(self) -> Result<Self, ParamError> {
        if self.fused_qkv {
            return Err(ParamError::new(
                "attn_output_gate",
                ConfigReason::Unsupported,
            ));
        }
        Ok(Self {
            output_gate: true,
            ..self
        })
    }

    /// V is the raw K projection (before K's norm and RoPE) instead of its own projection, so the
    /// layer has no V weights (Gemma 4's global layers). Needs separate, unbiased projections.
    pub fn with_value_from_key(self) -> Result<Self, ParamError> {
        if self.fused_qkv || self.qkv_bias {
            return Err(ParamError::new(
                "attention_k_eq_v",
                ConfigReason::Unsupported,
            ));
        }
        Ok(Self {
            value_from_key: true,
            ..self
        })
    }

    /// Normalize V over each head's `head_dim` dims with no learned scale (Gemma 4's V norm).
    pub fn with_value_norm(self, norm: NormParams) -> Self {
        Self {
            value_norm: Some(norm),
            ..self
        }
    }

    /// The score scale in place of `head_dim^-1/2`.
    pub fn with_scale(self, scale: f32) -> Result<Self, ParamError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(ParamError::new("scale", ConfigReason::NotFinitePositive));
        }
        Ok(Self { scale, ..self })
    }

    /// Whether V is the K projection, so the layer has no V weights.
    pub fn value_from_key(&self) -> bool {
        self.value_from_key
    }

    /// The query head count.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// The key/value head count (equal to `heads` for plain multi-head attention, a divisor of it for
    /// grouped-query attention).
    pub fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// The width of one head, shared by Q, K and V.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Three separate Q/K/V projections with their optional biases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeparateQkv {
    /// The query projection `[heads * head_dim, width]` (doubled rows when the layer has an output gate).
    pub q: Weight,
    /// The key projection `[kv_heads * head_dim, width]`.
    pub k: Weight,
    /// `None` when the parameters read V from the K projection.
    pub v: Option<Weight>,
    /// The Q, K and V biases in that order, when the parameters have a QKV bias.
    pub bias: Option<[Weight; 3]>,
}

/// The Q/K/V projections: three separate weights with their optional biases, or one fused
/// per-head-interleaved weight (see [`AttentionParams::with_fused_qkv`]) with its optional bias.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QkvWeights {
    /// Three separate projections.
    Separate(Box<SeparateQkv>),
    /// The fused per-head-interleaved projection.
    Fused {
        /// The fused weight `[3 * heads * head_dim, width]`.
        qkv: Weight,
        /// The fused bias `[3 * heads * head_dim]`, when the parameters have a QKV bias.
        bias: Option<Weight>,
    },
}

/// A layer's projections: Q `[heads * head_dim, width]`, K/V `[kv_heads * head_dim, width]` (or the
/// fused `[3 * heads * head_dim, width]`), `o` `[width, heads * head_dim]` with its bias when the
/// parameters have one, and the Q/K norm scales when the parameters normalize.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionWeights {
    /// The Q/K/V projections.
    pub qkv: QkvWeights,
    /// The output projection `[width, heads * head_dim]`.
    pub o: Weight,
    /// The output projection's bias `[width]`, when the parameters have one.
    pub o_bias: Option<Weight>,
    /// The Q and K norm scales in that order, when the parameters normalize Q and K.
    pub qk_norm: Option<[Weight; 2]>,
    /// The per-head sink logits `[heads]`, when the parameters have sinks.
    pub sinks: Option<Weight>,
}

impl AttentionWeights {
    /// Resolves the layer's attention tensors from `map` and checks each shape against `p` and the model
    /// `width`; a missing or mis-shaped tensor is a typed [`WeightError`].
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &AttentionParams,
    ) -> Result<Self, WeightError> {
        let w = |role, shape: &[usize]| {
            Weight::new(map, WeightId::layer(layer, WeightRole::Attn(role)), shape)
        };
        let (q, kv) = (p.heads * p.head_dim, p.kv_heads * p.head_dim);
        // With an output gate each head projects its query and its gate.
        let q_rows = if p.output_gate { 2 * q } else { q };
        let qkv = if p.fused_qkv {
            QkvWeights::Fused {
                qkv: w(AttnRole::Qkv, &[3 * q, width])?,
                bias: if p.qkv_bias {
                    Some(w(AttnRole::QkvBias, &[3 * q])?)
                } else {
                    None
                },
            }
        } else {
            let bias = if p.qkv_bias {
                Some([
                    w(AttnRole::QBias, &[q_rows])?,
                    w(AttnRole::KBias, &[kv])?,
                    w(AttnRole::VBias, &[kv])?,
                ])
            } else {
                None
            };
            QkvWeights::Separate(Box::new(SeparateQkv {
                q: w(AttnRole::Q, &[q_rows, width])?,
                k: w(AttnRole::K, &[kv, width])?,
                v: if p.value_from_key {
                    None
                } else {
                    Some(w(AttnRole::V, &[kv, width])?)
                },
                bias,
            }))
        };
        let qk_norm = match p.qk_norm {
            None => None,
            Some(QkNorm::PerHead(_)) => Some([
                w(AttnRole::QNorm, &[p.head_dim])?,
                w(AttnRole::KNorm, &[p.head_dim])?,
            ]),
            Some(QkNorm::Projection(_)) => {
                Some([w(AttnRole::QNorm, &[q])?, w(AttnRole::KNorm, &[kv])?])
            }
        };
        Ok(Self {
            qkv,
            o: w(AttnRole::O, &[width, q])?,
            o_bias: if p.out_bias {
                Some(w(AttnRole::OBias, &[width])?)
            } else {
                None
            },
            qk_norm,
            sinks: if p.sinks {
                Some(w(AttnRole::Sinks, &[p.heads])?)
            } else {
                None
            },
        })
    }
}

/// A cache write: the carried state `(input, output)` pair and the `[rows, kv_heads, capacity,
/// head_dim]` cache attention reads, in logical order.
type CacheWrite = ((Traced, Traced), Traced);

/// Each row's first position as the scalar index of its cache write, `[]` I32.
fn first_positions(b: &Builder, step: &Step) -> Result<Vec<Traced>, TraceError> {
    let rows = step.shape().rows.get();
    (0..rows)
        .map(|row| {
            let row_pos = if rows == 1 {
                step.pos()
            } else {
                b.slice(step.pos(), 0, row, row + 1)?
            };
            Ok::<_, TraceError>(b.reshape(b.slice(row_pos, 1, 0, 1)?, vec![])?)
        })
        .collect::<Result<_, TraceError>>()
}

/// What a cache stores per row: `kv_heads` heads of `head_dim` elements at the positions its [`HeadSpan`]
/// stores. A cache is declared from a span, so its head stride is the one the span decided.
#[derive(Clone, Copy, Debug)]
struct CacheDims {
    kv_heads: usize,
    head_dim: usize,
    span: HeadSpan,
}

/// The contiguous layout: row `r` writes its `tokens` new positions at its own first position
/// (`write_at[r]`) into its private `[kv_heads, positions, head_dim]` cache. The mask hides every key
/// past each query's own position, including stale entries beyond the live length.
fn write_contiguous(
    b: &Builder,
    step: &Step,
    write_at: &[Traced],
    name: &str,
    new: Traced,
    dims: CacheDims,
) -> Result<CacheWrite, TraceError> {
    let rows = step.shape().rows.get();
    let CacheDims {
        kv_heads,
        head_dim,
        span,
    } = dims;
    let positions = span.stored();
    let input = b.state_input(
        name,
        TensorType::f32(vec![rows, kv_heads, positions, head_dim]),
        StateRole::Positional { axis: 2 },
    )?;
    let output = if rows == 1 {
        b.dynamic_update_slice_dyn(input, new, write_at[0], 2)?
    } else {
        let updated: Vec<Traced> = (0..rows)
            .map(|row| {
                let cache = b.slice(input, 0, row, row + 1)?;
                let part = b.slice(new, 0, row, row + 1)?;
                Ok::<_, TraceError>(b.dynamic_update_slice_dyn(cache, part, write_at[row], 2)?)
            })
            .collect::<Result<_, TraceError>>()?;
        b.concat(0, &updated)?
    };
    Ok(((input, output), output))
}

/// The paged layout: one `[pool_slots, kv_heads, head_dim]` pool shared by every row. The step's
/// `rows * tokens` new K/V rows are written in place into the pool by the forward write map (token `t`
/// goes to slot `write[t]`; a padding token's -1 writes nothing), so the append costs the tokens, not
/// the pool. Then each row's `capacity` logical positions are viewed by the read map from the
/// post-write pool ([`ops::paged_view`], which attention fuses into reading the pool in place), so the
/// same mask as the contiguous layout applies.
fn write_paged(
    b: &Builder,
    step: &Step,
    maps: PagedMaps,
    name: &str,
    new: Traced,
    dims: CacheDims,
) -> Result<CacheWrite, TraceError> {
    let CacheDims {
        kv_heads: hkv,
        head_dim: d,
        span,
    } = dims;
    debug_assert_eq!(span.pad(), 0, "a paged cache is viewed at its capacity");
    let shape = step.shape();
    let (rows, t) = (shape.rows.get(), shape.tokens.get());
    let KvLayout::Paged { pool_slots, .. } = shape.kv else {
        return Err(WiringError::SlotMapsOnContiguousStep.into());
    };
    let input = b.state_input(
        name,
        TensorType::f32(vec![pool_slots.get(), hkv, d]),
        StateRole::Positional { axis: 0 },
    )?;
    let flat = b.reshape(b.transpose(new, vec![0, 2, 1, 3])?, vec![rows * t, hkv, d])?;
    let output = b.scatter_update(input, flat, maps.write)?;
    Ok(((input, output), ops::paged_view(b, output, maps.read)?))
}

/// Where each row of `step` writes its new positions: the first position of each row under the
/// contiguous layout, nothing under a paged one (it writes through its slot maps).
pub(crate) fn first_writes(b: &Builder, step: &Step) -> Result<Vec<Traced>, TraceError> {
    match step.paged() {
        None => first_positions(b, step),
        Some(_) => Ok(Vec::new()),
    }
}

/// The step's cache `name` of `layer` (`[rows, hkv, capacity, d]` contiguous, or one
/// `[pool_slots, hkv, d]` pool under a paged layout) with the new `[rows, hkv, tokens, d]` values
/// written into it and carried to the next step; returns the `[rows, hkv, capacity, d]` cache
/// attention reads, in logical order. A pool and a private cache are different buffers, so they
/// carry different names: one executable can hold both without a schema clash.
#[allow(clippy::too_many_arguments)]
pub(crate) fn carried_cache(
    b: &Builder,
    step: &Step,
    write_at: &[Traced],
    layer: usize,
    name: &str,
    new: Traced,
    hkv: usize,
    d: usize,
) -> Result<Traced, TraceError> {
    // Its consumers index the cache by logical position (MLA latents, block-pooled index keys).
    let span = HeadSpan::exact(step.shape().capacity.get(), d, Exemption::LogicalExtent);
    let dims = CacheDims {
        kv_heads: hkv,
        head_dim: d,
        span,
    };
    carried(b, step, write_at, layer, name, new, dims)
}

/// One cache of `dims`: the body of [`carried_cache`] and of attention's own key and value caches, which
/// are stored at their [`HeadSpan`] so the head stride is off the DRAM channel period (POOT-1150) and are
/// returned `[rows, kv_heads, positions, head_dim]`. A position past the capacity is never written and the
/// mask hides it.
fn carried(
    b: &Builder,
    step: &Step,
    write_at: &[Traced],
    layer: usize,
    name: &str,
    new: Traced,
    dims: CacheDims,
) -> Result<Traced, TraceError> {
    let ((input, output), read) = match step.paged() {
        None => write_contiguous(b, step, write_at, &format!("kv.l{layer}.{name}"), new, dims)?,
        Some(maps) => {
            let name = format!("kv.pool.l{layer}.{name}");
            write_paged(b, step, maps, &name, new, dims)?
        }
    };
    step.carry(input, output);
    Ok(read)
}

/// Self-attention of `x` (`[rows, tokens, width]`) at layer `layer`; returns `[rows, tokens, width]`.
/// The layer's K/V caches (`kv.l{layer}.k`, `kv.l{layer}.v`) are carried by `step`.
pub fn attention(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &AttentionWeights,
    p: &AttentionParams,
) -> Result<Traced, TraceError> {
    attention_with(b, step, layer, x, w, p, None)
}

/// `mask` (`[rows, heads | 1, tokens, capacity]`, additive) over `span`'s stored positions: the positions
/// past the capacity are appended fully masked.
fn hide_padding(b: &Builder, mask: Traced, span: HeadSpan) -> Result<Traced, TraceError> {
    if span.pad() == 0 {
        return Ok(mask);
    }
    let shape = b.aval(mask)?.shape;
    let last = shape.len() - 1;
    // `-1e9` (the mask's own hidden value) over the padded columns, built from a range so no selector output
    // reaches it: `0 * iota - 1e9` is finite for any mask.
    let hidden = b.binary_scalar(
        BinOp::Sub,
        b.binary_scalar(BinOp::Mul, b.iota(span.pad())?, Scalar::F32(0.0))?,
        Scalar::F32(1.0e9),
    )?;
    let mut hidden_shape = vec![1; shape.len()];
    hidden_shape[last] = span.pad();
    let mut pad_shape = shape;
    pad_shape[last] = span.pad();
    let hidden = b.broadcast(b.reshape(hidden, hidden_shape)?, pad_shape)?;
    Ok(b.concat(last, &[mask, hidden])?)
}

/// Replaces the causal-over-cache mask `[rows, 1, tokens, capacity]` of an attention with the one a
/// selector builds from it (a sparse selection narrows which cached positions a query reads).
pub type MaskSelector<'a> = &'a dyn Fn(&Builder, Traced) -> Result<Traced, TraceError>;

/// [`attention`] under the mask `select` makes of the causal one.
pub fn attention_selecting(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &AttentionWeights,
    p: &AttentionParams,
    select: MaskSelector<'_>,
) -> Result<Traced, TraceError> {
    attention_with(b, step, layer, x, w, p, Some(select))
}

/// The one attention body: `select` is the mask selector of [`attention_selecting`], `None` for plain
/// causal attention.
fn attention_with(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &AttentionWeights,
    p: &AttentionParams,
    select: Option<MaskSelector<'_>>,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let (hq, hkv, d) = (p.heads, p.kv_heads, p.head_dim);
    let heads_first = |y: Traced, heads: usize| {
        Ok::<_, TraceError>(b.transpose(b.reshape(y, vec![rows, t, heads, d])?, vec![0, 2, 1, 3])?)
    };
    let (q, k, v) = match &w.qkv {
        QkvWeights::Separate(sep) => {
            let SeparateQkv { q, k, v, bias } = &**sep;
            let bias = |i: usize| bias.as_ref().map(|biases| &biases[i]);
            let q = linear(b, x, q, bias(0))?;
            let k = linear(b, x, k, bias(1))?;
            let v = match v {
                Some(v) => linear(b, x, v, bias(2))?,
                None => k,
            };
            (q, k, v)
        }
        QkvWeights::Fused { qkv, bias } => {
            // Rows are `[heads, 3, head_dim]`: split the 3-axis, then flatten each part back.
            let fused = b.reshape(linear(b, x, qkv, bias.as_ref())?, vec![rows, t, hq, 3, d])?;
            let part = |i: usize| {
                let y = b.slice(fused, 3, i, i + 1)?;
                Ok::<_, TraceError>(b.reshape(y, vec![rows, t, hq * d])?)
            };
            (part(0)?, part(1)?, part(2)?)
        }
    };
    // Each head's rows are its query then its gate: `[rows, tokens, heads, 2 * head_dim]`.
    let (q, gate) = if p.output_gate {
        let pairs = b.reshape(q, vec![rows, t, hq, 2 * d])?;
        let part = |from: usize| {
            Ok::<_, TraceError>(
                b.reshape(b.slice(pairs, 3, from, from + d)?, vec![rows, t, hq * d])?,
            )
        };
        (part(0)?, Some(part(d)?))
    } else {
        (q, None)
    };
    let (q, k) = match (p.qk_norm, &w.qk_norm) {
        (Some(QkNorm::Projection(np)), Some([qw, kw])) => {
            (norm(b, q, qw, np)?, norm(b, k, kw, np)?)
        }
        _ => (q, k),
    };
    let (q, k) = (heads_first(q, hq)?, heads_first(k, hkv)?);
    let (q, k) = match (p.qk_norm, &w.qk_norm) {
        (Some(QkNorm::PerHead(np)), Some([qw, kw])) => (norm(b, q, qw, np)?, norm(b, k, kw, np)?),
        _ => (q, k),
    };
    let v = heads_first(v, hkv)?;
    let v = match p.value_norm {
        Some(np) => ops::rmsnorm_no_weight(b, v, np.eps())?,
        None => v,
    };

    let (q, k) = match p.rope {
        Some(rope) if rope.sections().is_some() => {
            let mrope = step.mrope().ok_or(WiringError::SectionedRopeWithoutMrope)?;
            (
                apply_rope_sectioned(b, rope, q, mrope, cap)?,
                apply_rope_sectioned(b, rope, k, mrope, cap)?,
            )
        }
        Some(rope) => {
            let rows = rope_rows(b, rope, step.pos(), cap)?;
            (apply_rope(b, q, &rows)?, apply_rope(b, k, &rows)?)
        }
        None => (q, k),
    };

    let write_at = first_writes(b, step)?;
    // A contiguous cache is padded off the channel period; a paged one is read in place through the pool
    // view, whose stride is the pool's row stride, so it keeps its capacity.
    let span = match step.paged() {
        None => HeadSpan::padded(cap, d),
        Some(_) => HeadSpan::exact(cap, d, Exemption::PagedView),
    };
    let dims = CacheDims {
        kv_heads: hkv,
        head_dim: d,
        span,
    };
    let k_cache = carried(b, step, &write_at, layer, "k", k, dims)?;
    let v_cache = carried(b, step, &write_at, layer, "v", v, dims)?;
    // The causal mask over the capacity, narrowed by the selector if any, then the stored positions past the
    // capacity appended hidden: the same on every entry point.
    let mask = if p.alibi {
        let slopes = b.computed(ComputedConst::AlibiSlopes { heads: hq })?;
        ops::alibi_mask_from_pos(b, step.pos(), cap, p.window, slopes, hq)?
    } else {
        ops::causal_mask_from_pos(b, step.pos(), cap, p.window)?
    };
    let mask = match select {
        Some(select) => select(b, mask)?,
        None => mask,
    };
    let mask = hide_padding(b, mask, span)?;
    let out = match &w.sinks {
        Some(sinks) => ops::attention_prefill_with_sink(
            b,
            q,
            k_cache,
            v_cache,
            hq / hkv,
            p.scale,
            mask,
            sinks.declare_f32(b)?,
            hq,
            t,
        )?,
        None => ops::attention_masked_softcap(
            b,
            q,
            k_cache,
            v_cache,
            hq / hkv,
            p.scale,
            mask,
            p.softcap,
        )?,
    };
    let out = b.reshape(b.transpose(out, vec![0, 2, 1, 3])?, vec![rows, t, hq * d])?;
    let out = match gate {
        Some(gate) => b.binary(BinOp::Mul, out, ops::sigmoid(b, gate)?)?,
        None => out,
    };
    linear(b, out, &w.o, w.o_bias.as_ref())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use poot_graph_ir::rope_table::RopeFlavor;
    use poot_graph_ir::{Builder, Slot};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::model::{KvLayout, LogitRows, StepShape};

    const WIDTH: usize = 8;
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 4;
    /// 128 positions of 4 f32: a 2048 B head stride, which camps, so every contiguous case below runs over
    /// a cache that stores one position more than its capacity ([`HeadSpan`]).
    const CAP: usize = 128;
    const THETA: f32 = 10_000.0;
    const EPS: f32 = 1e-5;

    /// How positions enter the scores.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Positions {
        HalfSplit,
        Interleaved,
        None,
        Alibi,
    }

    /// The options one test case turns on.
    #[derive(Clone, Copy, Debug)]
    struct Opts {
        qk_norm: Option<bool>, // Some(true) per head, Some(false) over the projection
        window: Option<usize>,
        softcap: Option<f32>,
        scale: Option<f32>,
        positions: Positions,
        sinks: bool,
        /// Each head's query rows are followed by its output gate (Qwen3-Next).
        gate: bool,
    }

    const BASE: Opts = Opts {
        qk_norm: None,
        window: None,
        softcap: None,
        scale: None,
        positions: Positions::HalfSplit,
        sinks: false,
        gate: false,
    };

    fn params(o: Opts) -> AttentionParams {
        let rope = RopeParams::new(HEAD_DIM, THETA, &RopeFlavor::Plain).unwrap();
        let mut p = AttentionParams::new(HEADS, KV_HEADS, HEAD_DIM, true, rope).unwrap();
        p = match o.positions {
            Positions::HalfSplit => p,
            Positions::Interleaved => p.with_interleaved_rope(),
            Positions::None => p.without_rope(),
            Positions::Alibi => p.with_alibi(),
        };
        let norm = NormParams::new(EPS).unwrap();
        p = match o.qk_norm {
            None => p,
            Some(true) => p.with_qk_norm(QkNorm::PerHead(norm)),
            Some(false) => p.with_qk_norm(QkNorm::Projection(norm)),
        };
        if let Some(w) = o.window {
            p = p.with_window(w).unwrap();
        }
        if let Some(c) = o.softcap {
            p = p.with_softcap(c).unwrap();
        }
        if let Some(s) = o.scale {
            p = p.with_scale(s).unwrap();
        }
        if o.sinks {
            p = p.with_sinks().unwrap();
        }
        if o.gate {
            p = p.with_output_gate().unwrap();
        }
        p
    }

    fn id(role: AttnRole) -> WeightId {
        WeightId::layer(2, WeightRole::Attn(role))
    }

    const ROLES: [AttnRole; 7] = [
        AttnRole::Q,
        AttnRole::K,
        AttnRole::V,
        AttnRole::O,
        AttnRole::QBias,
        AttnRole::KBias,
        AttnRole::VBias,
    ];

    /// The weights beyond [`ROLES`] a case adds: the Q/K norm scales and the sink logits.
    fn norm_roles(o: Opts) -> Vec<AttnRole> {
        let mut roles = Vec::new();
        if o.qk_norm.is_some() {
            roles.extend([AttnRole::QNorm, AttnRole::KNorm]);
        }
        if o.sinks {
            roles.push(AttnRole::Sinks);
        }
        roles
    }

    /// Sink logits large enough, against the scores of the random fixture, to take real softmax mass.
    const SINKS: [f32; HEADS] = [1.5, -0.5, 2.0, 0.3];

    fn fixture(o: Opts) -> Fx {
        let (q, kv) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        let q_rows = if o.gate { 2 * q } else { q };
        let mut fx = Fx::new(31);
        fx.bf16(id(AttnRole::Q), &[q_rows, WIDTH]);
        fx.f32(id(AttnRole::K), &[kv, WIDTH]);
        fx.bf16(id(AttnRole::V), &[kv, WIDTH]);
        fx.f32(id(AttnRole::O), &[WIDTH, q]);
        fx.f32(id(AttnRole::QBias), &[q_rows]);
        fx.bf16(id(AttnRole::KBias), &[kv]);
        fx.f32(id(AttnRole::VBias), &[kv]);
        if let Some(per_head) = o.qk_norm {
            let (qn, kn) = if per_head {
                (HEAD_DIM, HEAD_DIM)
            } else {
                (q, kv)
            };
            fx.bf16(id(AttnRole::QNorm), &[qn]);
            fx.f32(id(AttnRole::KNorm), &[kn]);
        }
        if o.sinks {
            fx.set(id(AttnRole::Sinks), &[HEADS], SINKS.to_vec());
        }
        fx
    }

    /// A `[.., CAP, HEAD_DIM]` cache as the graph stores it: every head gains the padded position, zeroed.
    fn stored_cache(cache: &[f32]) -> Vec<f32> {
        let head = CAP * HEAD_DIM;
        assert_eq!(cache.len() % head, 0);
        let pad = HeadSpan::padded(CAP, HEAD_DIM).pad();
        assert_eq!(
            pad, 1,
            "the fixture capacity must camp, or the padded path is not run"
        );
        cache
            .chunks(head)
            .flat_map(|h| {
                h.iter()
                    .copied()
                    .chain(std::iter::repeat_n(0.0, pad * HEAD_DIM))
            })
            .collect()
    }

    /// The inverse of [`stored_cache`]; the padded position must still be zero: nothing writes it.
    fn logical_cache(stored: &[f32]) -> Vec<f32> {
        let span = HeadSpan::padded(CAP, HEAD_DIM);
        let head = span.stored() * HEAD_DIM;
        assert_eq!(stored.len() % head, 0);
        let mut out = Vec::new();
        for h in stored.chunks(head) {
            let (live, pad) = h.split_at(CAP * HEAD_DIM);
            assert!(
                pad.iter().all(|&v| v == 0.0),
                "a padded position was written: {pad:?}"
            );
            out.extend_from_slice(live);
        }
        out
    }

    /// `fx.eval` over logical `[.., CAP, HEAD_DIM]` caches: they enter and leave in logical form, and the
    /// graph's state carries the stored extent in between.
    fn eval_logical(
        fx: &Fx,
        g: &poot_graph_ir::Graph,
        x: &[f32],
        slots: &[(Slot, &[i32])],
        state: &[Vec<f32>; 2],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let stored = [stored_cache(&state[0]), stored_cache(&state[1])];
        let (y, caches) = fx.eval(g, &[("x", x)], slots, &[&stored[0], &stored[1]]);
        (y, caches.iter().map(|c| logical_cache(c)).collect())
    }

    /// One step of `tokens` at `pos` over caches holding `state`; returns output and caches.
    fn run(
        fx: &Fx,
        o: Opts,
        x: &[f32],
        pos: &[i32],
        state: &[Vec<f32>; 2],
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        run_with(fx, o, x, pos, state, false)
    }

    /// [`run`] through [`attention_selecting`] with a selector that keeps the mask, when `selecting`.
    fn run_with(
        fx: &Fx,
        o: Opts,
        x: &[f32],
        pos: &[i32],
        state: &[Vec<f32>; 2],
        selecting: bool,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let t = pos.len();
        let p = params(o);
        let w = AttentionWeights::new(&fx.map(), 2, WIDTH, &p).unwrap();
        let b = Builder::for_tests();
        let step = Step::new(
            &b,
            StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(t).unwrap(),
                capacity: NonZeroUsize::new(CAP).unwrap(),
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
                hidden: None,
            },
        )
        .unwrap();
        let xs = b.constant("x", TensorType::f32(vec![1, t, WIDTH])).unwrap();
        let y = if selecting {
            attention_selecting(&b, &step, 2, xs, &w, &p, &|_, mask| Ok(mask)).unwrap()
        } else {
            attention(&b, &step, 2, xs, &w, &p).unwrap()
        };
        let g = step.finish(b, y).unwrap();
        let pairs: Vec<_> = g.state_pairs().collect();
        assert_eq!(pairs.len(), 2);
        assert!(
            pairs
                .iter()
                .all(|pair| pair.role == StateRole::Positional { axis: 2 })
        );
        let tokens = vec![0; t];
        eval_logical(
            fx,
            &g,
            x,
            &[(Slot::Token, &tokens), (Slot::Pos, pos)],
            state,
        )
    }

    fn rms(v: &mut [f64], w: &[f32]) {
        let ms = v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64;
        let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
        for (x, &w) in v.iter_mut().zip(w) {
            *x *= inv * f64::from(w);
        }
    }

    /// The f64 reference: projections with bias, the optional Q/K norm, the position scheme (RoPE
    /// over half-split or adjacent pairs, or ALiBi), the new K/V written at `pos[0]..`, softmax
    /// over the visible keys of the GQA-shared head (causal, within the window) of the scaled,
    /// scores.
    fn reference(
        fx: &Fx,
        o: Opts,
        x: &[f32],
        pos: &[i32],
        state: &[Vec<f32>; 2],
    ) -> (Vec<f64>, [Vec<f64>; 2]) {
        let (q_dim, kv_dim) = (HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM);
        let vals = |r| fx.values(id(r));
        let proj = |w: &[f32], bias: &[f32], v: &[f64], out: usize, k: usize| -> Vec<f64> {
            (0..out)
                .map(|o| {
                    (0..k).map(|j| v[j] * f64::from(w[o * k + j])).sum::<f64>()
                        + bias.get(o).map_or(0.0, |&b| f64::from(b))
                })
                .collect()
        };
        let rope = |v: &mut [f64], p: i32| {
            let half = HEAD_DIM / 2;
            for j in 0..half {
                let inv = f64::from(THETA).powf(-((2 * j) as f64) / HEAD_DIM as f64);
                let (s, c) = (f64::from(p) * inv).sin_cos();
                let (lo, hi) = match o.positions {
                    Positions::Interleaved => (2 * j, 2 * j + 1),
                    _ => (j, half + j),
                };
                let (a, b) = (v[lo], v[hi]);
                v[lo] = a * c - b * s;
                v[hi] = b * c + a * s;
            }
        };
        let mut caches: [Vec<f64>; 2] = [
            state[0].iter().map(|&v| v.into()).collect(),
            state[1].iter().map(|&v| v.into()).collect(),
        ];
        let mut queries = Vec::new();
        for (i, &p) in pos.iter().enumerate() {
            let xi: Vec<f64> = x[i * WIDTH..][..WIDTH].iter().map(|&v| v.into()).collect();
            let q_rows = if o.gate { 2 * q_dim } else { q_dim };
            let projected = proj(
                &vals(AttnRole::Q),
                &vals(AttnRole::QBias),
                &xi,
                q_rows,
                WIDTH,
            );
            // Each head's `2 * HEAD_DIM` rows are its query then its gate.
            let (mut q, gate) = if o.gate {
                let part = |from: usize| -> Vec<f64> {
                    (0..HEADS)
                        .flat_map(|h| projected[h * 2 * HEAD_DIM + from..][..HEAD_DIM].to_vec())
                        .collect()
                };
                (part(0), part(HEAD_DIM))
            } else {
                (projected, Vec::new())
            };
            let mut k = proj(
                &vals(AttnRole::K),
                &vals(AttnRole::KBias),
                &xi,
                kv_dim,
                WIDTH,
            );
            let v = proj(
                &vals(AttnRole::V),
                &vals(AttnRole::VBias),
                &xi,
                kv_dim,
                WIDTH,
            );
            match o.qk_norm {
                Some(false) => {
                    rms(&mut q, &vals(AttnRole::QNorm));
                    rms(&mut k, &vals(AttnRole::KNorm));
                }
                Some(true) => {
                    for h in 0..HEADS {
                        rms(&mut q[h * HEAD_DIM..][..HEAD_DIM], &vals(AttnRole::QNorm));
                    }
                    for h in 0..KV_HEADS {
                        rms(&mut k[h * HEAD_DIM..][..HEAD_DIM], &vals(AttnRole::KNorm));
                    }
                }
                None => {}
            }
            let rotates = matches!(o.positions, Positions::HalfSplit | Positions::Interleaved);
            for h in 0..HEADS {
                if rotates {
                    rope(&mut q[h * HEAD_DIM..][..HEAD_DIM], p);
                }
            }
            for h in 0..KV_HEADS {
                if rotates {
                    rope(&mut k[h * HEAD_DIM..][..HEAD_DIM], p);
                }
                let at = (h * CAP + (pos[0] as usize + i)) * HEAD_DIM;
                caches[0][at..at + HEAD_DIM].copy_from_slice(&k[h * HEAD_DIM..][..HEAD_DIM]);
                caches[1][at..at + HEAD_DIM].copy_from_slice(&v[h * HEAD_DIM..][..HEAD_DIM]);
            }
            queries.push((q, gate));
        }
        let scale = o.scale.map_or(1.0 / (HEAD_DIM as f64).sqrt(), f64::from);
        // ALiBi's published slopes for 4 heads: 2^(-8 (h + 1) / 4).
        let slope = |h: usize| 2f64.powf(-8.0 * (h + 1) as f64 / HEADS as f64);
        let mut out = Vec::new();
        for (i, (q, gate)) in queries.iter().enumerate() {
            let mut attn = vec![0.0f64; q_dim];
            for h in 0..HEADS {
                let g = h / (HEADS / KV_HEADS);
                let first = o
                    .window
                    .map_or(0, |w| (pos[i] as usize + 1).saturating_sub(w));
                let keys = first..=pos[i] as usize;
                let scores: Vec<f64> = keys
                    .clone()
                    .map(|j| {
                        let dot: f64 = (0..HEAD_DIM)
                            .map(|e| q[h * HEAD_DIM + e] * caches[0][(g * CAP + j) * HEAD_DIM + e])
                            .sum();
                        let mut s = dot * scale;
                        if let Some(c) = o.softcap {
                            s = f64::from(c) * (s / f64::from(c)).tanh();
                        }
                        if o.positions == Positions::Alibi {
                            s -= slope(h) * (pos[i] as usize - j) as f64;
                        }
                        s
                    })
                    .collect();
                // A sink logit joins the maximum and the denominator, never the weighted sum.
                let sink = o.sinks.then(|| f64::from(vals(AttnRole::Sinks)[h]));
                let max = scores.iter().cloned().chain(sink).fold(f64::MIN, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>() + sink.map_or(0.0, |s| (s - max).exp());
                for (j, e) in keys.zip(&exps) {
                    for c in 0..HEAD_DIM {
                        attn[h * HEAD_DIM + c] += e / sum * caches[1][(g * CAP + j) * HEAD_DIM + c];
                    }
                }
            }
            if o.gate {
                for (a, g) in attn.iter_mut().zip(gate) {
                    *a /= 1.0 + (-g).exp();
                }
            }
            out.extend(proj(&vals(AttnRole::O), &[], &attn, WIDTH, q_dim));
        }
        (out, caches)
    }

    /// A 3-token step continuing at position 2 over caches already holding two (and stale garbage
    /// past them) matches the f64 reference in its output and both carried caches, and every
    /// projection, bias and norm scale changes the output.
    fn assert_matches_reference(o: Opts) {
        let fx = fixture(o);
        let cache_len = KV_HEADS * CAP * HEAD_DIM;
        let state = [oracle::random(cache_len, 41), oracle::random(cache_len, 42)];
        let x = oracle::random(3 * WIDTH, 43);
        let pos = [2, 3, 4];
        let (got, caches) = run(&fx, o, &x, &pos, &state);
        let (want, want_caches) = reference(&fx, o, &x, &pos, &state);
        oracle::assert_matches_f64(&got, &want, 1e-5);
        for (got, want) in caches.iter().zip(&want_caches) {
            oracle::assert_matches_f64(got, want, 1e-5);
        }
        for role in ROLES.into_iter().chain(norm_roles(o)) {
            let mut p = fx.clone();
            p.perturb(id(role));
            oracle::assert_differs(&run(&p, o, &x, &pos, &state).0, &got, id(role));
        }
    }

    /// SC-001: the base attention (half-split RoPE, QKV bias). Mutation (named): drop the QKV bias
    /// add in `attention`; this row goes red.
    #[test]
    fn attention_matches_an_f64_reference_continuing_over_a_filled_cache() {
        assert_matches_reference(BASE);
    }

    /// A mask selector sees the capacity-wide causal mask and the padded position is hidden after it: the
    /// selecting body over a camping capacity gives what plain attention gives, and still matches the
    /// reference. Mutation: have `hide_padding` append a visible (zero) column; the padded zero key takes
    /// softmax mass and the output goes red.
    #[test]
    fn a_selector_over_a_padded_cache_matches_plain_attention() {
        for o in [
            BASE,
            Opts {
                positions: Positions::Alibi,
                ..BASE
            },
        ] {
            let fx = fixture(o);
            let cache_len = KV_HEADS * CAP * HEAD_DIM;
            let state = [oracle::random(cache_len, 41), oracle::random(cache_len, 42)];
            let x = oracle::random(2 * WIDTH, 43);
            let pos = [5, 6];
            let plain = run_with(&fx, o, &x, &pos, &state, false);
            let selected = run_with(&fx, o, &x, &pos, &state, true);
            assert_eq!(plain.0, selected.0, "{o:?}: the selecting body differs");
            assert_eq!(plain.1, selected.1, "{o:?}: the selected caches differ");
        }
    }

    /// A contiguous step stores one head position more than its capacity when that stride camps and attends
    /// over the stored extent; a paged step keeps the capacity (its pool view is exempt).
    /// Mutation: `HeadSpan::padded` returns the capacity; the contiguous stored extent is `CAP` and this
    /// goes red.
    #[test]
    fn a_camping_capacity_stores_one_position_more() {
        let fx = fixture(BASE);
        let p = params(BASE);
        let w = AttentionWeights::new(&fx.map(), 2, WIDTH, &p).unwrap();
        for kv in [
            KvLayout::Contiguous,
            KvLayout::Paged {
                pool_slots: NonZeroUsize::new(POOL).unwrap(),
                seqs: NonZeroUsize::MIN,
            },
        ] {
            let b = Builder::for_tests();
            let step = Step::new(
                &b,
                StepShape {
                    rows: NonZeroUsize::MIN,
                    tokens: NonZeroUsize::MIN,
                    capacity: NonZeroUsize::new(CAP).unwrap(),
                    kv,
                    logits: LogitRows::All,
                    hidden: None,
                },
            )
            .unwrap();
            let xs = b.constant("x", TensorType::f32(vec![1, 1, WIDTH])).unwrap();
            let y = attention(&b, &step, 2, xs, &w, &p).unwrap();
            let g = step.finish(b, y).unwrap();
            let shapes: Vec<Vec<usize>> = g
                .state_pairs()
                .map(|pair| g.meta(pair.input).aval.shape.clone())
                .collect();
            let want = match kv {
                KvLayout::Contiguous => vec![1, KV_HEADS, CAP + 1, HEAD_DIM],
                KvLayout::Paged { .. } => vec![POOL, KV_HEADS, HEAD_DIM],
            };
            assert_eq!(shapes, vec![want.clone(), want], "{kv:?}");
            // Contiguous attention reads (and masks) the stored extent; a paged one never sees it.
            let spans_stored = g
                .eqns()
                .iter()
                .any(|eqn| g.meta(eqn.out).aval.shape.last() == Some(&(CAP + 1)));
            assert_eq!(
                spans_stored,
                kv == KvLayout::Contiguous,
                "{kv:?}: values over the stored positions"
            );
        }
    }

    /// SC-001: Q/K RMSNorm per head (Qwen3) and over the whole projection (OLMo 2), before RoPE.
    /// Mutation: normalize after RoPE; both rows go red.
    #[test]
    fn qk_norm_per_head_and_over_the_projection_match_the_reference() {
        assert_matches_reference(Opts {
            qk_norm: Some(true),
            ..BASE
        });
        assert_matches_reference(Opts {
            qk_norm: Some(false),
            ..BASE
        });
    }

    /// SC-001: a sliding window of 3 keys (the step's queries at positions 2..=4 see fewer keys
    /// than the causal mask allows).
    #[test]
    fn a_sliding_window_matches_the_reference() {
        assert_matches_reference(Opts {
            window: Some(3),
            ..BASE
        });
    }

    /// SC-001: attention-logit softcap and an explicit score scale.
    #[test]
    fn softcap_and_an_explicit_scale_match_the_reference() {
        assert_matches_reference(Opts {
            softcap: Some(0.7),
            ..BASE
        });
        assert_matches_reference(Opts {
            scale: Some(0.31),
            ..BASE
        });
    }

    /// SC-002: per-head sink logits (gpt-oss), alone and under a sliding window, match the f64
    /// reference over a continued cache; the sink weight is in the perturbation list, so the
    /// output depends on it. Mutation: leave the sink out of the softmax denominator; red.
    #[test]
    fn attention_sinks_match_the_reference() {
        assert_matches_reference(Opts {
            sinks: true,
            ..BASE
        });
        assert_matches_reference(Opts {
            sinks: true,
            window: Some(3),
            ..BASE
        });
    }

    /// Sinks are a different attention from none, and they exclude a softcap in either order.
    #[test]
    fn sinks_change_the_output_and_exclude_a_softcap() {
        let cache_len = KV_HEADS * CAP * HEAD_DIM;
        let state = [oracle::random(cache_len, 41), oracle::random(cache_len, 42)];
        let x = oracle::random(3 * WIDTH, 43);
        let pos = [2, 3, 4];
        let with = Opts {
            sinks: true,
            ..BASE
        };
        let base = run(&fixture(BASE), BASE, &x, &pos, &state).0;
        assert_ne!(run(&fixture(with), with, &x, &pos, &state).0, base);

        let rope = RopeParams::new(HEAD_DIM, THETA, &RopeFlavor::Plain).unwrap();
        let plain = AttentionParams::new(HEADS, KV_HEADS, HEAD_DIM, true, rope).unwrap();
        assert_eq!(
            plain.with_softcap(1.0).unwrap().with_sinks(),
            Err(ParamError::new("sinks", ConfigReason::Unsupported))
        );
        assert_eq!(
            plain.with_sinks().unwrap().with_softcap(1.0),
            Err(ParamError::new("softcap", ConfigReason::Unsupported))
        );
    }

    /// The output gate (Qwen3-Next): each head's `2 * head_dim` Q rows are its query then its gate,
    /// the attention output is multiplied by `sigmoid(gate)` before the output projection, alone and
    /// with per-head Q/K norm, and the gate rows matter (the Q weight is in the perturbation list;
    /// a gate that ignored its rows would not change when only they change). Mutation: apply the
    /// gate after the output projection, or read the gate from the query rows; red.
    #[test]
    fn the_output_gate_matches_the_reference() {
        assert_matches_reference(Opts { gate: true, ..BASE });
        assert_matches_reference(Opts {
            gate: true,
            qk_norm: Some(true),
            ..BASE
        });
    }

    /// The gate rows alone change the output: with the same query rows and different gate rows the
    /// attention differs, and the gated attention differs from the ungated one.
    #[test]
    fn only_the_gate_rows_change_a_gated_attention() {
        let cache_len = KV_HEADS * CAP * HEAD_DIM;
        let state = [oracle::random(cache_len, 41), oracle::random(cache_len, 42)];
        let x = oracle::random(3 * WIDTH, 43);
        let pos = [2, 3, 4];
        let gated = Opts { gate: true, ..BASE };
        let fx = fixture(gated);
        let base = run(&fx, gated, &x, &pos, &state).0;
        // Replace the gate rows of every head, keeping its query rows.
        let q_rows = 2 * HEADS * HEAD_DIM;
        let mut values = fx.values(id(AttnRole::Q));
        let fresh = oracle::random(q_rows * WIDTH, 77);
        for head in 0..HEADS {
            let from = (head * 2 * HEAD_DIM + HEAD_DIM) * WIDTH;
            values[from..from + HEAD_DIM * WIDTH]
                .copy_from_slice(&fresh[from..from + HEAD_DIM * WIDTH]);
        }
        let mut other = fx.clone();
        other.set(id(AttnRole::Q), &[q_rows, WIDTH], values);
        assert_ne!(run(&other, gated, &x, &pos, &state).0, base);
        assert_ne!(run(&fixture(BASE), BASE, &x, &pos, &state).0, base);
    }

    /// A fused QKV projection has no per-head query/gate rows to split.
    #[test]
    fn an_output_gate_excludes_the_fused_projection() {
        let rope = RopeParams::new(HEAD_DIM, THETA, &RopeFlavor::Plain).unwrap();
        let fused = AttentionParams::new(HEADS, HEADS, HEAD_DIM, false, rope)
            .unwrap()
            .with_fused_qkv()
            .unwrap();
        assert_eq!(
            fused.with_output_gate(),
            Err(ParamError::new(
                "attn_output_gate",
                ConfigReason::Unsupported
            ))
        );
    }

    /// SC-001: interleaved RoPE (a GGUF's permuted rows), NoPE (no rotation) and ALiBi.
    #[test]
    fn interleaved_rope_nope_and_alibi_match_the_reference() {
        for positions in [Positions::Interleaved, Positions::None, Positions::Alibi] {
            assert_matches_reference(Opts { positions, ..BASE });
        }
    }

    /// Every option is a different attention: each case differs from the base on the same inputs.
    #[test]
    fn each_option_changes_the_output() {
        let cache_len = KV_HEADS * CAP * HEAD_DIM;
        let state = [oracle::random(cache_len, 41), oracle::random(cache_len, 42)];
        let x = oracle::random(3 * WIDTH, 43);
        let pos = [2, 3, 4];
        let base = run(&fixture(BASE), BASE, &x, &pos, &state).0;
        let cases = [
            Opts {
                window: Some(3),
                ..BASE
            },
            Opts {
                softcap: Some(0.7),
                ..BASE
            },
            Opts {
                scale: Some(0.31),
                ..BASE
            },
            Opts {
                positions: Positions::Interleaved,
                ..BASE
            },
            Opts {
                positions: Positions::None,
                ..BASE
            },
            Opts {
                positions: Positions::Alibi,
                ..BASE
            },
        ];
        for o in cases {
            let got = run(&fixture(o), o, &x, &pos, &state).0;
            assert_ne!(got, base, "{o:?} matches the base attention");
        }
    }

    /// SC-001: BLOOM's attention: a fused per-head-interleaved QKV projection (`[heads, 3,
    /// head_dim]` rows) with its bias, an output bias and ALiBi, plain MHA, continuing at position
    /// 2 over a filled cache, against an f64 reference that de-interleaves the fused rows itself.
    /// Every weight matters. Mutation: split the 3-axis as `[3, heads, head_dim]`; red.
    #[test]
    fn fused_interleaved_qkv_with_out_bias_and_alibi_matches_the_reference() {
        const H: usize = 2;
        const D: usize = 4;
        let (q_dim, cap, t) = (H * D, CAP, 3);
        let rope = RopeParams::new(D, THETA, &RopeFlavor::Plain).unwrap();
        let p = AttentionParams::new(H, H, D, true, rope)
            .unwrap()
            .with_alibi()
            .with_fused_qkv()
            .unwrap()
            .with_out_bias();
        let mut fx = Fx::new(77);
        fx.bf16(id(AttnRole::Qkv), &[3 * q_dim, WIDTH]);
        fx.f32(id(AttnRole::QkvBias), &[3 * q_dim]);
        fx.f32(id(AttnRole::O), &[WIDTH, q_dim]);
        fx.bf16(id(AttnRole::OBias), &[WIDTH]);
        let run = |fx: &Fx, x: &[f32], pos: &[i32], state: &[Vec<f32>; 2]| {
            let w = AttentionWeights::new(&fx.map(), 2, WIDTH, &p).unwrap();
            let b = Builder::for_tests();
            let step = Step::new(
                &b,
                StepShape {
                    rows: NonZeroUsize::MIN,
                    tokens: NonZeroUsize::new(t).unwrap(),
                    capacity: NonZeroUsize::new(cap).unwrap(),
                    kv: KvLayout::Contiguous,
                    logits: LogitRows::All,
                    hidden: None,
                },
            )
            .unwrap();
            let xs = b.constant("x", TensorType::f32(vec![1, t, WIDTH])).unwrap();
            let y = attention(&b, &step, 2, xs, &w, &p).unwrap();
            let g = step.finish(b, y).unwrap();
            eval_logical(
                fx,
                &g,
                x,
                &[(Slot::Token, &[0; 3]), (Slot::Pos, pos)],
                state,
            )
        };
        let cache_len = H * cap * D;
        let state = [oracle::random(cache_len, 51), oracle::random(cache_len, 52)];
        let x = oracle::random(t * WIDTH, 53);
        let pos = [2, 3, 4];
        let (got, caches) = run(&fx, &x, &pos, &state);

        // The reference: row `(h * 3 + part) * D + e` of the fused weight is part `part`'s
        // dim `e` of head `h`.
        let (wq, bq) = (
            fx.values(id(AttnRole::Qkv)),
            fx.values(id(AttnRole::QkvBias)),
        );
        let (wo, bo) = (fx.values(id(AttnRole::O)), fx.values(id(AttnRole::OBias)));
        let proj = |xi: &[f64], part: usize| -> Vec<f64> {
            (0..H * D)
                .map(|r| {
                    let row = ((r / D) * 3 + part) * D + r % D;
                    (0..WIDTH)
                        .map(|j| xi[j] * f64::from(wq[row * WIDTH + j]))
                        .sum::<f64>()
                        + f64::from(bq[row])
                })
                .collect()
        };
        let mut kc: Vec<f64> = state[0].iter().map(|&v| v.into()).collect();
        let mut vc: Vec<f64> = state[1].iter().map(|&v| v.into()).collect();
        let mut queries = Vec::new();
        for (i, &ps) in pos.iter().enumerate() {
            let xi: Vec<f64> = x[i * WIDTH..][..WIDTH].iter().map(|&v| v.into()).collect();
            queries.push(proj(&xi, 0));
            let (k, v) = (proj(&xi, 1), proj(&xi, 2));
            for h in 0..H {
                let at = (h * cap + ps as usize) * D;
                kc[at..at + D].copy_from_slice(&k[h * D..][..D]);
                vc[at..at + D].copy_from_slice(&v[h * D..][..D]);
            }
        }
        let mut want = Vec::new();
        for (i, q) in queries.iter().enumerate() {
            let mut attn = vec![0.0f64; q_dim];
            for h in 0..H {
                let slope = 2f64.powf(-8.0 * (h + 1) as f64 / H as f64);
                let keys = 0..=pos[i] as usize;
                let scores: Vec<f64> = keys
                    .clone()
                    .map(|j| {
                        let dot: f64 = (0..D)
                            .map(|e| q[h * D + e] * kc[(h * cap + j) * D + e])
                            .sum();
                        dot / (D as f64).sqrt() - slope * (pos[i] as usize - j) as f64
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for (j, e) in keys.zip(&exps) {
                    for c in 0..D {
                        attn[h * D + c] += e / sum * vc[(h * cap + j) * D + c];
                    }
                }
            }
            want.extend((0..WIDTH).map(|o| {
                (0..q_dim)
                    .map(|j| attn[j] * f64::from(wo[o * q_dim + j]))
                    .sum::<f64>()
                    + f64::from(bo[o])
            }));
        }
        oracle::assert_matches_f64(&got, &want, 1e-5);
        oracle::assert_matches_f64(&caches[0], &kc, 1e-5);
        oracle::assert_matches_f64(&caches[1], &vc, 1e-5);
        for role in [
            AttnRole::Qkv,
            AttnRole::QkvBias,
            AttnRole::O,
            AttnRole::OBias,
        ] {
            let mut pert = fx.clone();
            pert.perturb(id(role));
            oracle::assert_differs(&run(&pert, &x, &pos, &state).0, &got, id(role));
        }
    }

    #[test]
    fn fused_qkv_needs_plain_multi_head_attention() {
        let rope = RopeParams::new(HEAD_DIM, THETA, &RopeFlavor::Plain).unwrap();
        let gqa = AttentionParams::new(HEADS, KV_HEADS, HEAD_DIM, true, rope).unwrap();
        assert_eq!(
            gqa.with_fused_qkv(),
            Err(ParamError::new("kv_heads", ConfigReason::Unsupported))
        );
    }

    /// SC-005 (spec 999): each invalid attention combination is a typed error naming its field.
    #[test]
    fn attention_params_refuse_each_invalid_combination_by_field() {
        let rope = |dim| RopeParams::new(dim, THETA, &RopeFlavor::Plain).unwrap();
        let cases = [
            (
                (0, 2, 4, rope(4)),
                ParamError::new("heads", ConfigReason::Zero),
            ),
            (
                (4, 0, 4, rope(4)),
                ParamError::new("kv_heads", ConfigReason::Zero),
            ),
            (
                (4, 2, 0, rope(4)),
                ParamError::new("head_dim", ConfigReason::Zero),
            ),
            (
                (4, 3, 4, rope(4)),
                ParamError::new("heads", ConfigReason::NotDivisible { by: 3 }),
            ),
            (
                (4, 2, 4, rope(6)),
                ParamError::new("rotary_dim", ConfigReason::Exceeds { max: 4 }),
            ),
        ];
        for ((heads, kv_heads, head_dim, rope), want) in cases {
            assert_eq!(
                AttentionParams::new(heads, kv_heads, head_dim, true, rope),
                Err(want),
                "{heads} {kv_heads} {head_dim}"
            );
        }
        let base = params(BASE);
        assert_eq!(
            base.with_window(0),
            Err(ParamError::new("window", ConfigReason::Zero))
        );
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(
                base.with_softcap(bad),
                Err(ParamError::new("softcap", ConfigReason::NotFinitePositive))
            );
            assert_eq!(
                base.with_scale(bad),
                Err(ParamError::new("scale", ConfigReason::NotFinitePositive))
            );
        }
    }

    /// One step over `rows` rows of `t` tokens each, contiguous or paged, on the oracle. `pos` is
    /// `[rows * t]`. Paged: `state` is the two pools, `maps` the read `[rows * CAP]` and the forward
    /// write `[rows * t]` maps.
    fn run_rows(
        fx: &Fx,
        o: Opts,
        (rows, t): (usize, usize),
        x: &[f32],
        pos: &[i32],
        state: &[Vec<f32>; 2],
        paged: Option<(usize, &[i32], &[i32])>,
    ) -> (Vec<f32>, Vec<Vec<f32>>) {
        let p = params(o);
        let w = AttentionWeights::new(&fx.map(), 2, WIDTH, &p).unwrap();
        let b = Builder::for_tests();
        let kv = match paged {
            None => KvLayout::Contiguous,
            Some((pool_slots, ..)) => KvLayout::Paged {
                pool_slots: NonZeroUsize::new(pool_slots).unwrap(),
                seqs: NonZeroUsize::new(rows).unwrap(),
            },
        };
        let step = Step::new(
            &b,
            StepShape {
                rows: NonZeroUsize::new(rows).unwrap(),
                tokens: NonZeroUsize::new(t).unwrap(),
                capacity: NonZeroUsize::new(CAP).unwrap(),
                kv,
                logits: LogitRows::All,
                hidden: None,
            },
        )
        .unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![rows, t, WIDTH]))
            .unwrap();
        let y = attention(&b, &step, 2, xs, &w, &p).unwrap();
        let g = step.finish(b, y).unwrap();
        let role = match paged {
            None => StateRole::Positional { axis: 2 },
            Some(_) => StateRole::Positional { axis: 0 },
        };
        assert!(g.state_pairs().all(|pair| pair.role == role));
        let tokens = vec![0; rows * t];
        let named: Vec<(&str, &[i32])> = match paged {
            None => Vec::new(),
            Some((_, read, write)) => vec![("slotmap.read", read), ("writemap.write", write)],
        };
        match paged {
            None => {
                let stored = [stored_cache(&state[0]), stored_cache(&state[1])];
                let (y, caches) = fx.eval_named(
                    &g,
                    &[("x", x)],
                    &[(Slot::Token, &tokens), (Slot::Pos, pos)],
                    &named,
                    &[&stored[0], &stored[1]],
                );
                (y, caches.iter().map(|c| logical_cache(c)).collect())
            }
            Some(_) => fx.eval_named(
                &g,
                &[("x", x)],
                &[(Slot::Token, &tokens), (Slot::Pos, pos)],
                &named,
                &[&state[0], &state[1]],
            ),
        }
    }

    /// Two contiguous rows at different positions over different caches produce, for each row, what
    /// that row produces alone, and each row's cache write lands at its own first position.
    /// Mutation: write every row at row 0's first position; row 1 goes red.
    fn assert_contiguous_rows_independent(o: Opts) {
        let fx = fixture(o);
        let cache_len = KV_HEADS * CAP * HEAD_DIM;
        let (t, positions) = (2, [[1, 2], [4, 5]]);
        let caches: Vec<[Vec<f32>; 2]> = (0..2)
            .map(|r| {
                [
                    oracle::random(cache_len, 50 + r),
                    oracle::random(cache_len, 60 + r),
                ]
            })
            .collect();
        let x = oracle::random(2 * t * WIDTH, 70);
        let pos: Vec<i32> = positions.concat();
        let both = [
            [caches[0][0].clone(), caches[1][0].clone()].concat(),
            [caches[0][1].clone(), caches[1][1].clone()].concat(),
        ];
        let (got, got_state) = run_rows(&fx, o, (2, t), &x, &pos, &both, None);
        for r in 0..2 {
            let (alone, alone_state) = run_rows(
                &fx,
                o,
                (1, t),
                &x[r * t * WIDTH..(r + 1) * t * WIDTH],
                &positions[r],
                &caches[r],
                None,
            );
            oracle::assert_matches_f64(
                &got[r * t * WIDTH..(r + 1) * t * WIDTH],
                &alone.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
            for (c, alone_cache) in alone_state.iter().enumerate() {
                oracle::assert_matches_f64(
                    &got_state[c][r * cache_len..(r + 1) * cache_len],
                    &alone_cache
                        .iter()
                        .map(|&v| f64::from(v))
                        .collect::<Vec<_>>(),
                    1e-6,
                );
            }
        }
    }

    #[test]
    fn contiguous_rows_write_and_attend_at_their_own_positions() {
        assert_contiguous_rows_independent(BASE);
    }

    /// The same independence with sink logits (the sink broadcasts over rows).
    #[test]
    fn contiguous_rows_with_sinks_attend_independently() {
        assert_contiguous_rows_independent(Opts {
            sinks: true,
            ..BASE
        });
    }

    const POOL: usize = 2 * CAP + 8;

    /// The pool slot of row `r`'s logical position `j`: the two rows interleave their slots, so
    /// neither row's blocks are contiguous.
    fn slot_of(r: usize, j: usize) -> i32 {
        (2 * j + r + 1) as i32
    }

    /// Paged rows over a shared pool match the same rows run contiguously over the caches the read
    /// map gathers: identical outputs, and the pool afterwards holds each written token's K/V at
    /// its slot with every other slot unchanged. Mutation: build the read map for row 1 from row 0's
    /// slots; row 1's output goes red.
    #[test]
    fn paged_rows_match_contiguous_rows_over_the_gathered_caches() {
        let fx = fixture(BASE);
        let (t, positions) = (2, [[1usize, 2], [4, 5]]);
        let pool_len = POOL * KV_HEADS * HEAD_DIM;
        let pools = [oracle::random(pool_len, 81), oracle::random(pool_len, 82)];
        let read: Vec<i32> = (0..2)
            .flat_map(|r| (0..CAP).map(move |j| slot_of(r, j)))
            .collect();
        // The forward map: flat token `r * t + i` writes the slot of its position.
        let write: Vec<i32> = positions
            .iter()
            .enumerate()
            .flat_map(|(r, row)| row.iter().map(move |&p| slot_of(r, p)))
            .collect();
        let x = oracle::random(2 * t * WIDTH, 83);
        let pos: Vec<i32> = positions.iter().flatten().map(|&p| p as i32).collect();
        let (got, got_pools) = run_rows(
            &fx,
            BASE,
            (2, t),
            &x,
            &pos,
            &pools,
            Some((POOL, &read, &write)),
        );
        // The contiguous cache row `r` sees: the pool gathered at its slots, `[kv_heads, CAP, d]`.
        let gather = |pool: &[f32], r: usize| -> Vec<f32> {
            let mut cache = vec![0.0; KV_HEADS * CAP * HEAD_DIM];
            for h in 0..KV_HEADS {
                for j in 0..CAP {
                    let src = (slot_of(r, j) as usize * KV_HEADS + h) * HEAD_DIM;
                    let dst = (h * CAP + j) * HEAD_DIM;
                    cache[dst..dst + HEAD_DIM].copy_from_slice(&pool[src..src + HEAD_DIM]);
                }
            }
            cache
        };
        let mut want_pools = pools.clone();
        for r in 0..2 {
            let caches = [gather(&pools[0], r), gather(&pools[1], r)];
            let row_pos: Vec<i32> = positions[r].iter().map(|&p| p as i32).collect();
            let (alone, alone_state) = run_rows(
                &fx,
                BASE,
                (1, t),
                &x[r * t * WIDTH..(r + 1) * t * WIDTH],
                &row_pos,
                &caches,
                None,
            );
            oracle::assert_matches_f64(
                &got[r * t * WIDTH..(r + 1) * t * WIDTH],
                &alone.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
            for (c, cache) in alone_state.iter().enumerate() {
                for &p in &positions[r] {
                    for h in 0..KV_HEADS {
                        let src = (h * CAP + p) * HEAD_DIM;
                        let dst = (slot_of(r, p) as usize * KV_HEADS + h) * HEAD_DIM;
                        want_pools[c][dst..dst + HEAD_DIM]
                            .copy_from_slice(&cache[src..src + HEAD_DIM]);
                    }
                }
            }
        }
        for (got, want) in got_pools.iter().zip(&want_pools) {
            oracle::assert_matches_f64(
                got,
                &want.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
        }
    }

    /// A token whose write-map entry is -1 (a padding token) leaves its pool slot untouched while the
    /// other row's tokens still write. Mutation: ignore the write map's -1 (write every token at
    /// slot 0); the untouched-slot assertion goes red.
    #[test]
    fn a_padding_token_writes_nothing() {
        let fx = fixture(BASE);
        let t = 2;
        let pool_len = POOL * KV_HEADS * HEAD_DIM;
        let pools = [oracle::random(pool_len, 91), oracle::random(pool_len, 92)];
        let read: Vec<i32> = (0..2)
            .flat_map(|r| (0..CAP).map(move |j| slot_of(r, j)))
            .collect();
        // Row 1's first token is padding: it writes nowhere; its second token (position 3) does.
        let write = [slot_of(0, 1), slot_of(0, 2), -1, slot_of(1, 3)];
        let x = oracle::random(2 * t * WIDTH, 93);
        let (_, got) = run_rows(
            &fx,
            BASE,
            (2, t),
            &x,
            &[1, 2, 2, 3],
            &pools,
            Some((POOL, &read, &write)),
        );
        let slot = |pool: &[f32], s: i32| {
            let at = s as usize * KV_HEADS * HEAD_DIM;
            pool[at..at + KV_HEADS * HEAD_DIM].to_vec()
        };
        for c in 0..2 {
            assert_eq!(
                slot(&got[c], slot_of(1, 2)),
                slot(&pools[c], slot_of(1, 2)),
                "the padding token's slot kept its contents"
            );
            assert_ne!(
                slot(&got[c], slot_of(1, 3)),
                slot(&pools[c], slot_of(1, 3)),
                "the real token wrote"
            );
        }
    }
}
