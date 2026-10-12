//! Heavily compressed attention (HCA) and the window compressor both compressed kinds share.
//!
//! A compressor turns every `rate` consecutive positions of a layer into one entry: per-token key
//! and gate projections (their raw values carried per absolute position, like the key-value cache),
//! the gate plus a learned position bias, a softmax over the window's positions per channel, the
//! weighted sum of the keys, a norm and the layer's rotation at the window's first position. HCA
//! pools disjoint windows and attends every complete one; CSA ([`super::csa`]) pools overlapping
//! windows and keeps the best few.
//!
//! The entries are recomputed from the carried raw history on every step, so a prefill chunk, a
//! decode step and a continuation all read the same function of the positions written so far. A
//! window is visible to a query once the window's last position is: until then it is masked, and
//! the (finite) garbage of an unwritten window never reaches the output.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{CompressRole, WeightId, WeightMap, WeightRole};

use super::config::{DeepseekV4Config, LayerKind};
use super::csa::overlap_pool;
use super::sliding::{Compressed, rotate_tail};
use crate::components::attention::carried_cache;
use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::norm;
use crate::components::rope::{RopeParams, rope_strided_rows};
use crate::components::standard::Step;
use crate::model::TraceError;

/// Additive mask for a window a query may not read: finite, so that adding it to another mask or to
/// a score cannot overflow.
pub(super) const DROPPED: f32 = -1.0e30;

/// One compressor's weights: the key and gate projections `[raw, width]`, the position bias
/// `[rate, raw]` and the norm of the pooled entry `[entry]`. `raw` is the entry width for
/// non-overlapping windows and twice it for overlapping ones.
#[derive(Debug)]
pub(super) struct CompressorWeights {
    kv: Weight,
    gate: Weight,
    bias: Weight,
    norm: Weight,
}

/// Whose compressor: the layer's own, or its lightning indexer's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Owner {
    Layer,
    Indexer,
}

/// How a compressor pools its windows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pooling {
    /// Each window of `rate` positions on its own.
    Disjoint,
    /// Each entry also reads the previous window's first half: raw width is twice the entry's.
    Overlapping,
}

impl CompressorWeights {
    pub(super) fn new(
        map: &WeightMap,
        layer: usize,
        owner: Owner,
        width: usize,
        rate: usize,
        entry: usize,
        pooling: Pooling,
    ) -> Result<Self, WeightError> {
        let raw = match pooling {
            Pooling::Disjoint => entry,
            Pooling::Overlapping => 2 * entry,
        };
        let roles = match owner {
            Owner::Layer => [
                CompressRole::Kv,
                CompressRole::Gate,
                CompressRole::PositionBias,
                CompressRole::Norm,
            ],
            Owner::Indexer => [
                CompressRole::IndexerKv,
                CompressRole::IndexerGate,
                CompressRole::IndexerPositionBias,
                CompressRole::IndexerNorm,
            ],
        };
        let w = |role, shape: &[usize]| {
            Weight::new(
                map,
                WeightId::layer(layer, WeightRole::Compress(role)),
                shape,
            )
        };
        Ok(Self {
            kv: w(roles[0], &[raw, width])?,
            gate: w(roles[1], &[raw, width])?,
            bias: w(roles[2], &[rate, raw])?,
            norm: w(roles[3], &[entry])?,
        })
    }
}

/// How many whole windows of `rate` positions a cache of `capacity` positions holds.
pub(super) fn windows_in(capacity: usize, rate: usize) -> usize {
    capacity / rate
}

/// The pooled, normalized and rotated entries of `w`'s windows: `[rows, 1, windows, entry]`, or
/// `None` when the cache is shorter than one window. This step's raw projections of `normed` are
/// written into the compressor's carried histories first (named `tag`).
#[allow(clippy::too_many_arguments)]
pub(super) fn entries(
    b: &Builder,
    step: &Step,
    layer: usize,
    normed: Traced,
    write_at: &[Traced],
    w: &CompressorWeights,
    tag: &str,
    rate: usize,
    entry: usize,
    pooling: Pooling,
    rope: RopeParams,
    cfg: &DeepseekV4Config,
) -> Result<Option<Traced>, TraceError> {
    let shape = step.shape();
    let (r, t, cap) = (shape.rows.get(), shape.tokens.get(), shape.capacity.get());
    let raw = match pooling {
        Pooling::Disjoint => entry,
        Pooling::Overlapping => 2 * entry,
    };
    let history = |name: &str, weight: &Weight| -> Result<_, TraceError> {
        let new = b.reshape(linear(b, normed, weight, None)?, vec![r, 1, t, raw])?;
        carried_cache(
            b,
            step,
            write_at,
            layer,
            &format!("{tag}.{name}"),
            new,
            1,
            raw,
        )
    };
    let kv = history("kv", &w.kv)?;
    let gate = history("gate", &w.gate)?;

    let n = windows_in(cap, rate);
    if n == 0 {
        return Ok(None);
    }
    let windowed = |x: Traced| {
        Ok::<_, TraceError>(b.reshape(b.slice(x, 2, 0, n * rate)?, vec![r, n, rate, raw])?)
    };
    let bias = b.reshape(w.bias.declare_f32(b)?, vec![1, 1, rate, raw])?;
    let gate = b.binary(BinOp::Add, windowed(gate)?, bias)?;
    let pooled = match pooling {
        Pooling::Disjoint => softmax_pool(b, windowed(kv)?, gate)?,
        Pooling::Overlapping => overlap_pool(b, windowed(kv)?, gate, entry)?,
    };
    let pooled = norm(b, pooled, &w.norm, cfg.norm)?;
    let rows = rope_strided_rows(b, rope, cap, rate, n, r)?;
    Ok(Some(rotate_tail(
        b,
        b.reshape(pooled, vec![r, 1, n, entry])?,
        &rows,
        cfg.attn.rope_dim,
    )?))
}

/// The softmax over the window axis (per channel) of `gate`, the weighted sum of `kv` over it:
/// `[rows, windows, positions, channels]` to `[rows, windows, channels]`. The softmax runs on the
/// trailing axis, so the window axis is moved there and back around it.
pub(super) fn softmax_pool(b: &Builder, kv: Traced, gate: Traced) -> Result<Traced, TraceError> {
    let by_channel = b.transpose(gate, vec![0, 1, 3, 2])?;
    let weights = b.transpose(ops::softmax(b, by_channel)?, vec![0, 1, 3, 2])?;
    Ok(b.reduce(RedOp::Sum, b.binary(BinOp::Mul, kv, weights)?, 2, false)?)
}

/// The additive mask of which of `windows` windows each query token may read:
/// `[rows, 1, tokens, windows]`, `0.0` once the window's last position is at or before the query's
/// own position, [`DROPPED`] before.
pub(super) fn complete_windows(
    b: &Builder,
    step: &Step,
    rate: usize,
    windows: usize,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (r, t) = (shape.rows.get(), shape.tokens.get());
    let pos = b.reshape(
        b.cast(step.pos(), poot_tensor::DType::F32)?,
        vec![r, 1, t, 1],
    )?;
    let last = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, b.iota(windows)?, Scalar::F32(rate as f32))?,
        Scalar::F32((rate - 1) as f32),
    )?;
    let visible = b.binary(BinOp::Ge, pos, b.reshape(last, vec![1, 1, 1, windows])?)?;
    Ok(b.binary_scalar(
        BinOp::Mul,
        b.binary_scalar(BinOp::Sub, visible, Scalar::F32(1.0))?,
        Scalar::F32(-DROPPED),
    )?)
}

/// An HCA layer's compressed attention inputs: its disjoint-window entries and the complete-window
/// mask. `None` when the cache holds no whole window.
pub(super) fn compressed(
    b: &Builder,
    step: &Step,
    layer: usize,
    normed: Traced,
    write_at: &[Traced],
    w: &CompressorWeights,
    cfg: &DeepseekV4Config,
) -> Result<Option<Compressed>, TraceError> {
    let rate = cfg.hca_rate;
    let Some(entries) = entries(
        b,
        step,
        layer,
        normed,
        write_at,
        w,
        "compress",
        rate,
        cfg.attn.head_dim,
        Pooling::Disjoint,
        cfg.rope_of(LayerKind::Hca),
        cfg,
    )?
    else {
        return Ok(None);
    };
    let n = windows_in(step.shape().capacity.get(), rate);
    Ok(Some(Compressed {
        entries,
        mask: complete_windows(b, step, rate, n)?,
    }))
}
