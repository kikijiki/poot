//! The Gated DeltaNet (GDN) mixer of a hybrid layer (Qwen3-Next), one body for every phase.
//!
//! A layer projects its input into the mixed query/key/value channels, the output gate `z`, a
//! per-value-head `beta` and a per-value-head decay argument, runs a causal depthwise conv
//! ([`super::conv`]) and SiLU over the mixed channels, L2-normalizes each query and key head and
//! runs the gated delta rule per value head:
//!
//! ```text
//! S      = exp(g) * S                      # forget (g = softplus(alpha + dt_bias) * decay, decay < 0)
//! delta  = beta * (v - S^T k)              # the delta-rule correction
//! S      = S + outer(k, delta)
//! o      = S^T (q / sqrt(head_dim))
//! ```
//!
//! The output is `rmsnorm(o, scale) * silu(z)` and the output projection. The state is a
//! `[head_dim, head_dim]` matrix per value head carried across steps, and the conv carries its input
//! window ([`super::conv::causal_conv`]), so a chunk continues the previous one exactly and one token
//! is a decode step. A single token runs the recurrence ([`ops::gated_delta_net_decode`]); more run
//! the scan composite ([`ScanFamily::DeltaRule`], whose reference definition is the chunked form
//! [`ops::gdn_prefill_chunked`], the UT-transform over `chunk` positions, the same recurrence solved in closed
//! form).
//!
//! A checkpoint spells the layer two ways. GGUF stores one projection per quantity and the decay
//! `ssm_a` already negative, with the value heads reordered so that query and key heads tile
//! ([`GdnHeadOrder::Tiled`]). HF `qwen3_next` stores `in_proj_qkvz` and `in_proj_ba` whose rows
//! group per key head, `A_log` (the decay is `-exp(A_log)`) and value heads grouped by key head
//! ([`GdnHeadOrder::Grouped`]). [`GdnParams`] names which, and the same graph follows.

use poot_graph_ir::{BinOp, Builder, ScanAttrs, ScanFamily, ScanResult, Traced, UnOp, ops};
use poot_quant::weights::{GdnRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::conv::{ConvParams, ConvWeights, causal_conv};
use super::linear::{Weight, WeightError, linear};
use super::norm::NormParams;
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;

/// How a Gated DeltaNet layer pairs its `H_k` query/key heads with its `H_v` value heads.
///
/// With `n_rep = H_v / H_k`, value head `j` reads key head `j % H_k` in tiled order and key head
/// `j / n_rep` in grouped order. The order belongs to the checkpoint source, not to the architecture:
/// HF safetensors store V heads grouped by K head and expand q/k with `repeat_interleave`, while
/// llama.cpp's converter (`_reorder_v_heads` in `conversion/qwen.py`) rewrites every per-V-head tensor
/// to tiled order so GGUF files expand q/k with a plain `ggml_repeat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GdnHeadOrder {
    /// GGUF files: V heads reordered by the converter, q/k tiled `[k0, k1, k0, k1, ...]`.
    Tiled,
    /// HF safetensors: V heads grouped by K head, q/k grouped `[k0, k0, k1, k1, ...]`.
    Grouped,
}

impl GdnHeadOrder {
    /// Expand per-head query or key values `[B, H_k, L, D]` to `[B, H_k * n_rep, L, D]`.
    fn expand(self, b: &Builder, x: Traced, n_rep: usize) -> Traced {
        match self {
            Self::Tiled => ops::repeat_kv_tiled(b, x, n_rep),
            Self::Grouped => ops::repeat_kv(b, x, n_rep),
        }
    }
}

/// The head counts and widths of a layer, checked: `value_heads` is a multiple of `key_heads`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnDims {
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
}

impl GdnDims {
    pub(crate) fn new(
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("linear_num_key_heads", key_heads),
            ("linear_num_value_heads", value_heads),
            ("linear_key_head_dim", head_dim),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if !value_heads.is_multiple_of(key_heads) {
            return Err(ParamError::new(
                "linear_num_value_heads",
                ConfigReason::NotDivisible { by: key_heads },
            ));
        }
        Ok(Self {
            key_heads,
            value_heads,
            head_dim,
        })
    }

    fn key_dim(&self) -> usize {
        self.key_heads * self.head_dim
    }

    fn value_dim(&self) -> usize {
        self.value_heads * self.head_dim
    }

    /// The mixed query, key and value channels the conv runs over.
    pub(crate) fn conv_channels(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    fn n_rep(&self) -> usize {
        self.value_heads / self.key_heads
    }
}

/// The chunk length of the chunked recurrence (a power of two): the UT-transform inverts a
/// `chunk x chunk` triangle per head, so the chunk bounds the work per step (HF's `chunk_size`).
const DEFAULT_CHUNK: usize = 64;

/// Checked GDN parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GdnParams {
    dims: GdnDims,
    conv: ConvParams,
    norm: NormParams,
    order: GdnHeadOrder,
    chunk: usize,
    fused_projections: bool,
    log_decay: bool,
}

impl GdnParams {
    /// `key_heads` query/key heads and `value_heads` value heads of `head_dim` channels each, a
    /// conv of `kernel` taps over the mixed channels and `norm` (its epsilon) for the L2 norm and the
    /// gated RMSNorm, with the checkpoint's value-head `order`. The projections are separate and the
    /// decay stored as the decay itself (a GGUF); see [`Self::with_fused_projections`] and
    /// [`Self::with_log_decay`] for an HF checkpoint.
    pub fn new(
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        kernel: usize,
        norm: NormParams,
        order: GdnHeadOrder,
    ) -> Result<Self, ParamError> {
        let dims = GdnDims::new(key_heads, value_heads, head_dim)?;
        Ok(Self {
            dims,
            conv: ConvParams::new(dims.conv_channels(), kernel, 1)?,
            norm,
            order,
            chunk: DEFAULT_CHUNK,
            fused_projections: false,
            log_decay: false,
        })
    }

    /// The chunk length of the chunked recurrence, in place of the default 64: a power of two (the
    /// UT-transform's triangular inverse doubles). A test shortens it to run several chunks over a
    /// few tokens.
    #[cfg(test)]
    pub(crate) fn with_chunk(self, chunk: usize) -> Result<Self, ParamError> {
        if chunk == 0 {
            return Err(ParamError::new("gdn_chunk", ConfigReason::Zero));
        }
        if !chunk.is_power_of_two() {
            return Err(ParamError::new("gdn_chunk", ConfigReason::Unsupported));
        }
        Ok(Self { chunk, ..self })
    }

    /// The layer stores `in_proj_qkvz` and `in_proj_ba` (HF `qwen3_next`), whose rows group per key
    /// head, instead of one projection per quantity.
    pub fn with_fused_projections(self) -> Self {
        Self {
            fused_projections: true,
            ..self
        }
    }

    /// The stored decay factor is `A_log` and the decay is `-exp(A_log)` (HF), not the decay itself.
    pub fn with_log_decay(self) -> Self {
        Self {
            log_decay: true,
            ..self
        }
    }

    /// The conv's channels: the mixed query, key and value.
    pub fn conv_channels(&self) -> usize {
        self.dims.conv_channels()
    }

    pub fn value_dim(&self) -> usize {
        self.dims.value_dim()
    }

    pub fn value_heads(&self) -> usize {
        self.dims.value_heads
    }

    pub fn key_heads(&self) -> usize {
        self.dims.key_heads
    }

    pub fn head_dim(&self) -> usize {
        self.dims.head_dim
    }
}

/// The input projections: one per quantity, or the two fused ones of an HF checkpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Projections {
    Split {
        qkv: Weight,
        z: Weight,
        beta: Weight,
        alpha: Weight,
    },
    Fused {
        qkvz: Weight,
        ba: Weight,
    },
}

/// A layer's GDN weights: the input projections, the `dt_bias`, the conv kernel, the decay factor,
/// the gated norm scale and the output projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GdnWeights {
    projections: Projections,
    dt_bias: Weight,
    conv: ConvWeights,
    decay: Weight,
    gate_norm: Weight,
    out: Weight,
}

impl GdnWeights {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &GdnParams,
    ) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Gdn(role));
        let w = |role, shape: &[usize]| Weight::new(map, id(role), shape);
        let d = p.dims;
        let (value_dim, heads) = (d.value_dim(), d.value_heads);
        let projections = if p.fused_projections {
            Projections::Fused {
                qkvz: w(GdnRole::Qkvz, &[d.conv_channels() + value_dim, width])?,
                ba: w(GdnRole::Ba, &[2 * heads, width])?,
            }
        } else {
            Projections::Split {
                qkv: w(GdnRole::Qkv, &[d.conv_channels(), width])?,
                z: w(GdnRole::Z, &[value_dim, width])?,
                beta: w(GdnRole::Beta, &[heads, width])?,
                alpha: w(GdnRole::Alpha, &[heads, width])?,
            }
        };
        Ok(Self {
            projections,
            dt_bias: w(GdnRole::DtBias, &[heads])?,
            conv: ConvWeights::new(map, id(GdnRole::Conv), None, &p.conv)?,
            decay: w(GdnRole::A, &[heads])?,
            gate_norm: w(GdnRole::GateNorm, &[d.head_dim])?,
            out: w(GdnRole::Out, &[width, value_dim])?,
        })
    }
}

/// The projected values of a step: the mixed channels, the output gate and the raw beta and alpha.
struct Projected {
    qkv: Traced,
    z: Traced,
    beta: Traced,
    alpha: Traced,
}

/// Project `x` (`[rows, tokens, width]`) through `w`. A fused HF layer's rows group per key head,
/// `[key_heads, 2 * head_dim + 2 * n_rep * head_dim]` for `q|k|v|z` and `[key_heads, 2 * n_rep]` for
/// `b|a`: they are split on the projected activations, and each value part lands in grouped head
/// order.
fn project(b: &Builder, x: Traced, w: &GdnWeights, d: &GdnDims) -> Projected {
    match &w.projections {
        Projections::Split {
            qkv,
            z,
            beta,
            alpha,
        } => Projected {
            qkv: linear(b, x, qkv, None),
            z: linear(b, x, z, None),
            beta: linear(b, x, beta, None),
            alpha: linear(b, x, alpha, None),
        },
        Projections::Fused { qkvz, ba } => {
            let shape = b.aval(x).shape;
            let (rows, t) = (shape[0], shape[1]);
            let (hk, hd, rep) = (d.key_heads, d.head_dim, d.n_rep());
            let grouped = b.reshape(
                linear(b, x, qkvz, None),
                vec![rows, t, hk, 2 * hd + 2 * rep * hd],
            );
            let part = |from: usize, len: usize, width: usize| {
                b.reshape(b.slice(grouped, 3, from, from + len), vec![rows, t, width])
            };
            let q = part(0, hd, hk * hd);
            let k = part(hd, hd, hk * hd);
            let v = part(2 * hd, rep * hd, d.value_dim());
            let z = part(2 * hd + rep * hd, rep * hd, d.value_dim());
            let pairs = b.reshape(linear(b, x, ba, None), vec![rows, t, hk, 2 * rep]);
            let gate = |from: usize| {
                b.reshape(
                    b.slice(pairs, 3, from, from + rep),
                    vec![rows, t, d.value_heads],
                )
            };
            Projected {
                qkv: b.concat(2, &[q, k, v]),
                z,
                beta: gate(0),
                alpha: gate(rep),
            }
        }
    }
}

/// `beta = sigmoid(beta_raw)` and the forget-gate argument `g = softplus(alpha_raw + dt_bias) *
/// decay` (`decay` is negative, so `exp(g)` lies in `(0, 1)`), both `[rows, tokens, value_heads]`.
pub(crate) fn gdn_gates(
    b: &Builder,
    beta_raw: Traced,
    alpha_raw: Traced,
    dt_bias: Traced,
    decay: Traced,
) -> (Traced, Traced) {
    let beta = ops::sigmoid(b, beta_raw);
    let alpha = ops::softplus(b, b.binary(BinOp::Add, alpha_raw, dt_bias));
    (beta, b.binary(BinOp::Mul, alpha, decay))
}

/// `x / sqrt(sum(x^2) + eps)` over the last axis: ggml's `l2_norm`, which differs from an RMSNorm
/// in having no scale and in adding `eps` to the sum, not the mean.
fn l2_norm_last(b: &Builder, x: Traced, eps: f32) -> Traced {
    let last = b.aval(x).shape.len() - 1;
    let squares = b.reduce(
        poot_graph_ir::RedOp::Sum,
        b.binary(BinOp::Mul, x, x),
        last,
        true,
    );
    let norm = b.unary(
        UnOp::Sqrt,
        b.binary_scalar(BinOp::Add, squares, Builder::f32(eps)),
    );
    b.binary(BinOp::Div, x, norm)
}

/// `[rows, tokens, heads * d]` as `[rows, heads, tokens, d]`.
fn heads_first(b: &Builder, y: Traced, heads: usize, d: usize) -> Traced {
    let shape = b.aval(y).shape;
    b.transpose(
        b.reshape(y, vec![shape[0], shape[1], heads, d]),
        vec![0, 2, 1, 3],
    )
}

/// What a [`gdn_scan`] reads besides the carried state.
struct ScanInputs {
    /// The conv's SiLU output `[rows, tokens, conv_channels]`.
    conv_out: Traced,
    /// The output gate `[rows, tokens, value_dim]`.
    z: Traced,
    /// `beta` and the forget-gate argument, `[rows, tokens, value_heads]`.
    beta: Traced,
    g: Traced,
    /// The gated norm's `[head_dim]` scale.
    norm_scale: Traced,
}

/// How a layer's recurrence is shaped: its head counts, value-head order, epsilon and chunk length.
#[derive(Clone, Copy)]
struct Recurrence {
    dims: GdnDims,
    order: GdnHeadOrder,
    eps: f32,
    chunk: usize,
}

/// The recurrence and the gated norm over the conv's SiLU output.
///
/// `conv_out` is `[rows, tokens, conv_channels]` (post-SiLU), `z` `[rows, tokens, value_dim]`,
/// `beta` and `g` `[rows, tokens, value_heads]`, `norm_scale` the `[head_dim]` scale and `state`
/// the carried `[rows, value_heads, head_dim, head_dim]` matrices. Returns the gated output
/// `[rows, tokens, value_dim]` before the output projection and the next state. A single token runs
/// the recurrence step; more run the chunked form with the `rows` folded into the head axis (heads
/// of different rows share the weights but not the state).
fn gdn_scan(
    b: &Builder,
    inputs: ScanInputs,
    state: Traced,
    recurrence: &Recurrence,
) -> (Traced, Traced) {
    let ScanInputs {
        conv_out,
        z,
        beta,
        g,
        norm_scale,
    } = inputs;
    let Recurrence {
        dims: d,
        order,
        eps,
        chunk,
    } = *recurrence;
    let d = &d;
    let shape = b.aval(conv_out).shape;
    let (rows, t) = (shape[0], shape[1]);
    let (hk, hv, hd) = (d.key_heads, d.value_heads, d.head_dim);
    let (key_dim, value_dim) = (d.key_dim(), d.value_dim());

    let q = b.slice(conv_out, 2, 0, key_dim);
    let k = b.slice(conv_out, 2, key_dim, 2 * key_dim);
    let v = b.slice(conv_out, 2, 2 * key_dim, d.conv_channels());
    let q = order.expand(
        b,
        l2_norm_last(b, heads_first(b, q, hk, hd), eps),
        d.n_rep(),
    );
    let k = order.expand(
        b,
        l2_norm_last(b, heads_first(b, k, hk, hd), eps),
        d.n_rep(),
    );
    let v = heads_first(b, v, hv, hd);
    let g = heads_first(b, g, hv, 1);
    let beta = heads_first(b, beta, hv, 1);

    let (o, next) = if t == 1 {
        ops::gated_delta_net_decode(b, q, k, v, g, beta, state)
    } else {
        // The chunk is a power of two (the triangular inverse doubles); a prompt shorter than it
        // runs one chunk of the next power of two and the op zero-pads the tail.
        let c = chunk.min(t.next_power_of_two());
        // One row is already `[1, heads, ..]`: only several rows fold.
        let fold = |y: Traced| {
            if rows == 1 {
                return y;
            }
            let mut shape = b.aval(y).shape;
            shape.splice(0..2, [1, rows * hv]);
            b.reshape(y, shape)
        };
        let ScanResult {
            y: o, s_out: next, ..
        } = b.scan(
            ScanAttrs {
                family: ScanFamily::DeltaRule,
                chunk: c,
            },
            &[fold(q), fold(k), fold(v), fold(g), fold(beta), fold(state)],
        );
        if rows == 1 {
            (o, next)
        } else {
            (
                b.reshape(o, vec![rows, hv, t, hd]),
                b.reshape(next, vec![rows, hv, hd, hd]),
            )
        }
    };

    let normed = ops::rmsnorm(b, o, norm_scale, eps);
    let gate = ops::silu(b, heads_first(b, z, hv, hd));
    let gated = b.binary(BinOp::Mul, normed, gate);
    let flat = b.reshape(
        b.transpose(gated, vec![0, 2, 1, 3]),
        vec![rows, t, value_dim],
    );
    (flat, next)
}

/// The mixer over `x` (`[rows, tokens, width]`) at layer `layer`: `[rows, tokens, width]`. The
/// layer's conv window (`gdn.l{layer}.conv`) and state matrices (`gdn.l{layer}.ssm`) are carried by
/// `step`.
pub fn gdn(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &GdnWeights,
    p: &GdnParams,
) -> Traced {
    let d = &p.dims;
    let projected = project(b, x, w, d);
    let decay = w.decay.declare_f32(b);
    let decay = if p.log_decay {
        b.unary(UnOp::Neg, b.unary(UnOp::Exp, decay)) // decay = -exp(A_log)
    } else {
        decay
    };
    let (beta, g) = gdn_gates(
        b,
        projected.beta,
        projected.alpha,
        w.dt_bias.declare_f32(b),
        decay,
    );
    let conv = causal_conv(
        b,
        step,
        &format!("gdn.l{layer}.conv"),
        projected.qkv,
        &w.conv,
        &p.conv,
    );
    let state = step.recurrent_state(
        b,
        &format!("gdn.l{layer}.ssm"),
        DType::F32,
        &[d.value_heads, d.head_dim, d.head_dim],
    );
    let (mixed, next) = gdn_scan(
        b,
        ScanInputs {
            conv_out: ops::silu(b, conv),
            z: projected.z,
            beta,
            g,
            norm_scale: w.gate_norm.declare_f32(b),
        },
        state.current(),
        &Recurrence {
            dims: *d,
            order: p.order,
            eps: p.norm.eps(),
            chunk: p.chunk,
        },
    );
    step.carry_recurrent(b, state, next);
    linear(b, mixed, &w.out, None)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::Slot;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;
    use poot_graph_ir::TensorType;

    const WIDTH: usize = 6;
    const KEY_HEADS: usize = 2;
    const VALUE_HEADS: usize = 4;
    const HEAD_DIM: usize = 3;
    const KERNEL: usize = 3;
    const EPS: f32 = 1e-6;
    const KEY_DIM: usize = KEY_HEADS * HEAD_DIM;
    const VALUE_DIM: usize = VALUE_HEADS * HEAD_DIM;
    const CONV: usize = 2 * KEY_DIM + VALUE_DIM;
    const REP: usize = VALUE_HEADS / KEY_HEADS;
    /// Six tokens in chunks of four: a full chunk and a ragged tail of two.
    const TOKENS: usize = 6;
    const CHUNK: usize = 4;

    /// One checkpoint spelling of a layer: the head order, whether the projections are fused and
    /// whether the stored decay is `A_log`.
    #[derive(Clone, Copy, Debug)]
    struct Case {
        order: GdnHeadOrder,
        fused: bool,
        log_decay: bool,
    }

    /// A GGUF layer: tiled heads, separate projections, the decay itself.
    const GGUF: Case = Case {
        order: GdnHeadOrder::Tiled,
        fused: false,
        log_decay: false,
    };
    /// An HF `qwen3_next` layer: grouped heads, fused projections, `A_log`.
    const HF: Case = Case {
        order: GdnHeadOrder::Grouped,
        fused: true,
        log_decay: true,
    };
    /// Separate projections over grouped heads (Qwen3.5).
    const HF_SPLIT: Case = Case {
        order: GdnHeadOrder::Grouped,
        fused: false,
        log_decay: true,
    };

    fn id(role: GdnRole) -> WeightId {
        WeightId::layer(0, WeightRole::Gdn(role))
    }

    fn params(c: Case) -> GdnParams {
        let mut p = GdnParams::new(
            KEY_HEADS,
            VALUE_HEADS,
            HEAD_DIM,
            KERNEL,
            NormParams::new(EPS).unwrap(),
            c.order,
        )
        .unwrap()
        .with_chunk(CHUNK)
        .unwrap();
        if c.fused {
            p = p.with_fused_projections();
        }
        if c.log_decay {
            p = p.with_log_decay();
        }
        p
    }

    /// The roles a case stores and their shapes.
    fn roles(c: Case) -> Vec<(GdnRole, Vec<usize>)> {
        let mut r = if c.fused {
            vec![
                (GdnRole::Qkvz, vec![CONV + VALUE_DIM, WIDTH]),
                (GdnRole::Ba, vec![2 * VALUE_HEADS, WIDTH]),
            ]
        } else {
            vec![
                (GdnRole::Qkv, vec![CONV, WIDTH]),
                (GdnRole::Z, vec![VALUE_DIM, WIDTH]),
                (GdnRole::Beta, vec![VALUE_HEADS, WIDTH]),
                (GdnRole::Alpha, vec![VALUE_HEADS, WIDTH]),
            ]
        };
        r.extend([
            (GdnRole::DtBias, vec![VALUE_HEADS]),
            (GdnRole::Conv, vec![CONV, 1, KERNEL]),
            (GdnRole::A, vec![VALUE_HEADS]),
            (GdnRole::GateNorm, vec![HEAD_DIM]),
            (GdnRole::Out, vec![WIDTH, VALUE_DIM]),
        ]);
        r
    }

    fn fixture(c: Case) -> Fx {
        let mut fx = Fx::new(41);
        for (i, (role, shape)) in roles(c).into_iter().enumerate() {
            match role {
                // `A_log` for HF; a decay of a real magnitude (and negative) for a GGUF.
                GdnRole::A => {
                    let values: Vec<f32> = (0..VALUE_HEADS)
                        .map(|h| {
                            if c.log_decay {
                                -0.5 + 0.7 * h as f32
                            } else {
                                -(0.2 + 1.3 * h as f32)
                            }
                        })
                        .collect();
                    fx.set(id(role), &shape, values);
                }
                GdnRole::GateNorm => fx.set(
                    id(role),
                    &shape,
                    (0..HEAD_DIM).map(|i| 0.8 + 0.1 * i as f32).collect(),
                ),
                _ if i % 2 == 0 => fx.f32(id(role), &shape),
                _ => fx.bf16(id(role), &shape),
            }
        }
        fx
    }

    struct Run {
        out: Vec<f32>,
        /// The conv window `[K - 1, CONV]`, then the state `[VALUE_HEADS, D, D]`.
        state: Vec<Vec<f32>>,
    }

    /// `x` through the component from `state` (zeros when empty), positions from `start`.
    fn run(fx: &Fx, c: Case, x: &[f32], start: i32, state: &[Vec<f32>]) -> Run {
        let p = params(c);
        let w = GdnWeights::new(&fx.map(), 0, WIDTH, &p).unwrap();
        let tokens = x.len() / WIDTH;
        let b = Builder::new();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All));
        let xs = b.constant("x", TensorType::f32(vec![1, tokens, WIDTH]));
        let y = gdn(&b, &step, 0, xs, &w, &p);
        let g = step.finish(b, y);
        let zeros: Vec<Vec<f32>> = g
            .state
            .iter()
            .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = if state.is_empty() { &zeros } else { state }
            .iter()
            .map(Vec::as_slice)
            .collect();
        let ids = vec![0; tokens];
        let pos: Vec<i32> = (start..).take(tokens).collect();
        let (out, state) = fx.eval(
            &g,
            &[("x", x)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &state,
        );
        Run { out, state }
    }

    fn silu(v: f64) -> f64 {
        v / (1.0 + (-v).exp())
    }

    fn matvec(w: &[f32], rows: usize, x: &[f64]) -> Vec<f64> {
        let k = x.len();
        (0..rows)
            .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
            .collect()
    }

    /// Split a fused HF projection's output the way HF's `fix_query_key_value_ordering` does:
    /// per key head `[q | k | v (REP heads) | z (REP heads)]` and `[b (REP) | a (REP)]`.
    fn unfuse(qkvz: &[f64], ba: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let block = 2 * HEAD_DIM + 2 * REP * HEAD_DIM;
        let (mut q, mut k, mut v, mut z): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
            (vec![], vec![], vec![], vec![]);
        for g in 0..KEY_HEADS {
            let rows = &qkvz[g * block..(g + 1) * block];
            q.extend(&rows[..HEAD_DIM]);
            k.extend(&rows[HEAD_DIM..2 * HEAD_DIM]);
            v.extend(&rows[2 * HEAD_DIM..2 * HEAD_DIM + REP * HEAD_DIM]);
            z.extend(&rows[2 * HEAD_DIM + REP * HEAD_DIM..]);
        }
        let (mut beta, mut alpha): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        for g in 0..KEY_HEADS {
            let rows = &ba[g * 2 * REP..(g + 1) * 2 * REP];
            beta.extend(&rows[..REP]);
            alpha.extend(&rows[REP..]);
        }
        let mut mixed = q;
        mixed.extend(k);
        mixed.extend(v);
        (mixed, z, beta, alpha)
    }

    /// The mixer over `x` token by token, written from the recurrence in f64: the outputs, the conv
    /// window (the last `K - 1` raw conv inputs) and the state matrices `[heads][i][j]`. Value head
    /// `j` reads key head `j % KEY_HEADS` (tiled) or `j / REP` (grouped).
    fn reference(fx: &Fx, c: Case, x: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let v = |role| fx.values(id(role));
        let (conv_w, dt_bias, a) = (v(GdnRole::Conv), v(GdnRole::DtBias), v(GdnRole::A));
        let norm_w = v(GdnRole::GateNorm);
        let mut raw: Vec<Vec<f64>> = vec![vec![0.0; CONV]; KERNEL - 1];
        let mut state = vec![0.0f64; VALUE_HEADS * HEAD_DIM * HEAD_DIM];
        let mut out = Vec::new();
        for token in x.chunks(WIDTH) {
            let row: Vec<f64> = token.iter().map(|&t| t.into()).collect();
            let (mixed, z, beta, alpha) = if c.fused {
                let qkvz = matvec(&v(GdnRole::Qkvz), CONV + VALUE_DIM, &row);
                let ba = matvec(&v(GdnRole::Ba), 2 * VALUE_HEADS, &row);
                unfuse(&qkvz, &ba)
            } else {
                (
                    matvec(&v(GdnRole::Qkv), CONV, &row),
                    matvec(&v(GdnRole::Z), VALUE_DIM, &row),
                    matvec(&v(GdnRole::Beta), VALUE_HEADS, &row),
                    matvec(&v(GdnRole::Alpha), VALUE_HEADS, &row),
                )
            };
            raw.push(mixed);
            let n = raw.len();
            let conv: Vec<f64> = (0..CONV)
                .map(|ch| {
                    silu(
                        (0..KERNEL)
                            .map(|j| f64::from(conv_w[ch * KERNEL + j]) * raw[n - KERNEL + j][ch])
                            .sum(),
                    )
                })
                .collect();
            let l2 = |head: &[f64]| -> Vec<f64> {
                let norm = (head.iter().map(|x| x * x).sum::<f64>() + f64::from(EPS)).sqrt();
                head.iter().map(|x| x / norm).collect()
            };
            let mut o_flat = vec![0.0f64; VALUE_DIM];
            for j in 0..VALUE_HEADS {
                let kh = match c.order {
                    GdnHeadOrder::Tiled => j % KEY_HEADS,
                    GdnHeadOrder::Grouped => j / REP,
                };
                let q = l2(&conv[kh * HEAD_DIM..(kh + 1) * HEAD_DIM]);
                let k = l2(&conv[KEY_DIM + kh * HEAD_DIM..KEY_DIM + (kh + 1) * HEAD_DIM]);
                let value = &conv[2 * KEY_DIM + j * HEAD_DIM..2 * KEY_DIM + (j + 1) * HEAD_DIM];
                let decay = if c.log_decay {
                    -f64::from(a[j]).exp()
                } else {
                    f64::from(a[j])
                };
                let beta = 1.0 / (1.0 + (-beta[j]).exp());
                let g = (1.0 + (alpha[j] + f64::from(dt_bias[j])).exp()).ln() * decay;
                let s = &mut state[j * HEAD_DIM * HEAD_DIM..(j + 1) * HEAD_DIM * HEAD_DIM];
                s.iter_mut().for_each(|e| *e *= g.exp());
                let delta: Vec<f64> = (0..HEAD_DIM)
                    .map(|jj| {
                        let kv: f64 = (0..HEAD_DIM).map(|i| s[i * HEAD_DIM + jj] * k[i]).sum();
                        beta * (value[jj] - kv)
                    })
                    .collect();
                for i in 0..HEAD_DIM {
                    for jj in 0..HEAD_DIM {
                        s[i * HEAD_DIM + jj] += k[i] * delta[jj];
                    }
                }
                let scale = 1.0 / (HEAD_DIM as f64).sqrt();
                let o: Vec<f64> = (0..HEAD_DIM)
                    .map(|jj| {
                        (0..HEAD_DIM)
                            .map(|i| s[i * HEAD_DIM + jj] * q[i] * scale)
                            .sum()
                    })
                    .collect();
                let inv = 1.0
                    / (o.iter().map(|x| x * x).sum::<f64>() / HEAD_DIM as f64 + f64::from(EPS))
                        .sqrt();
                for e in 0..HEAD_DIM {
                    o_flat[j * HEAD_DIM + e] =
                        o[e] * inv * f64::from(norm_w[e]) * silu(z[j * HEAD_DIM + e]);
                }
            }
            out.extend(matvec(&v(GdnRole::Out), WIDTH, &o_flat));
        }
        let window: Vec<f64> = raw[raw.len() - (KERNEL - 1)..].concat();
        (out, window, state)
    }

    fn inputs() -> Vec<f32> {
        oracle::random(TOKENS * WIDTH, 9)
            .into_iter()
            .map(|v| 2.0 * v)
            .collect()
    }

    /// SC-005: for each checkpoint spelling (GGUF, HF, HF with separate projections) the mixer
    /// equals the f64 recurrence at every output and every carried element, whole (a chunk and a
    /// ragged tail), split in two chunks, and token by token; and each weight changes the output.
    /// Mutation: apply the decay after the state update instead of before (in `gated_delta_net_decode`
    /// and the chunked form's gate), or swap the head order in `GdnHeadOrder::expand`; each goes red.
    #[test]
    fn the_mixer_matches_the_recurrence_and_continues_across_chunks_for_every_spelling() {
        for c in [GGUF, HF, HF_SPLIT] {
            let fx = fixture(c);
            let x = inputs();
            let (want, window, state) = reference(&fx, c, &x);
            let whole = run(&fx, c, &x, 0, &[]);
            oracle::assert_matches_f64(&whole.out, &want, 1e-4);
            oracle::assert_matches_f64(&whole.state[0], &window, 1e-4);
            oracle::assert_matches_f64(&whole.state[1], &state, 1e-4);

            let first = run(&fx, c, &x[..3 * WIDTH], 0, &[]);
            let second = run(&fx, c, &x[3 * WIDTH..], 3, &first.state);
            oracle::assert_matches_f64(&[first.out.clone(), second.out].concat(), &want, 1e-4);
            oracle::assert_matches_f64(&second.state[1], &state, 1e-4);

            let mut steps = first;
            let mut stepped = steps.out.clone();
            for t in 3..TOKENS {
                steps = run(
                    &fx,
                    c,
                    &x[t * WIDTH..(t + 1) * WIDTH],
                    t as i32,
                    &steps.state,
                );
                stepped.extend(&steps.out);
            }
            oracle::assert_matches_f64(&stepped, &want, 1e-4);
            oracle::assert_matches_f64(&steps.state[0], &window, 1e-4);
            oracle::assert_matches_f64(&steps.state[1], &state, 1e-4);

            for (role, _) in roles(c) {
                let mut perturbed = fx.clone();
                perturbed.perturb(id(role));
                oracle::assert_differs(&run(&perturbed, c, &x, 0, &[]).out, &whole.out, id(role));
            }
        }
    }

    /// The head order is a different layer: the same weights give different outputs tiled and
    /// grouped, so a spelling that picked the wrong order could not pass the reference above.
    #[test]
    fn the_head_order_changes_the_output() {
        let tiled = Case {
            order: GdnHeadOrder::Tiled,
            ..HF_SPLIT
        };
        let fx = fixture(HF_SPLIT);
        let x = inputs();
        let grouped = run(&fx, HF_SPLIT, &x, 0, &[]).out;
        let tiled = run(&fx, tiled, &x, 0, &[]).out;
        oracle::assert_differs(&tiled, &grouped, id(GdnRole::Qkv));
    }

    /// Two rows with different prompts give each row's output and state of the same prompt run alone
    /// (the rows fold into the head axis of the chunked op without sharing a state).
    #[test]
    fn two_rows_do_not_share_state() {
        let c = HF;
        let fx = fixture(c);
        let prompts = [inputs(), oracle::random(TOKENS * WIDTH, 21)];
        let single: Vec<Run> = prompts.iter().map(|x| run(&fx, c, x, 0, &[])).collect();

        let p = params(c);
        let w = GdnWeights::new(&fx.map(), 0, WIDTH, &p).unwrap();
        let b = Builder::new();
        let shape = crate::model::StepShape {
            rows: std::num::NonZeroUsize::new(2).unwrap(),
            ..step_shape(TOKENS, CAP, LogitRows::All)
        };
        let step = Step::new(&b, shape);
        let xs = b.constant("x", TensorType::f32(vec![2, TOKENS, WIDTH]));
        let y = gdn(&b, &step, 0, xs, &w, &p);
        let g = step.finish(b, y);
        let zeros: Vec<Vec<f32>> = g
            .state
            .iter()
            .map(|&(si, _)| vec![0.0; g.aval(si).numel()])
            .collect();
        let state: Vec<&[f32]> = zeros.iter().map(Vec::as_slice).collect();
        let x: Vec<f32> = prompts.concat();
        let ids = vec![0; 2 * TOKENS];
        let pos: Vec<i32> = (0..2).flat_map(|_| 0..TOKENS as i32).collect();
        let (out, states) = fx.eval(
            &g,
            &[("x", &x)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &state,
        );
        for (row, want) in single.iter().enumerate() {
            let width = TOKENS * WIDTH;
            oracle::assert_matches_f64(
                &out[row * width..(row + 1) * width],
                &want.out.iter().map(|&v| v.into()).collect::<Vec<f64>>(),
                1e-5,
            );
            for (got, want) in states.iter().zip(&want.state) {
                let per_row = got.len() / 2;
                oracle::assert_matches_f64(
                    &got[row * per_row..(row + 1) * per_row],
                    &want.iter().map(|&v| v.into()).collect::<Vec<f64>>(),
                    1e-5,
                );
            }
        }
    }

    #[test]
    fn params_refuse_heads_that_do_not_divide_and_empty_extents() {
        let norm = NormParams::new(EPS).unwrap();
        let tiled = GdnHeadOrder::Tiled;
        assert_eq!(
            GdnParams::new(2, 3, 4, 3, norm, tiled),
            Err(ParamError::new(
                "linear_num_value_heads",
                ConfigReason::NotDivisible { by: 2 }
            ))
        );
        assert_eq!(
            GdnParams::new(0, 4, 4, 3, norm, tiled),
            Err(ParamError::new("linear_num_key_heads", ConfigReason::Zero))
        );
        assert_eq!(
            GdnParams::new(2, 4, 4, 1, norm, tiled),
            Err(ParamError::new("conv_kernel", ConfigReason::Zero))
        );
        assert_eq!(
            params(GGUF).with_chunk(0),
            Err(ParamError::new("gdn_chunk", ConfigReason::Zero))
        );
        assert_eq!(
            params(GGUF).with_chunk(6),
            Err(ParamError::new("gdn_chunk", ConfigReason::Unsupported))
        );
    }
}
