//! Hyper-Connections (`Qwen4ExpTextGatedResidual`): the residual stream is `streams` parallel copies
//! of the hidden state, `[.., streams * width]`, and every block reads a gated mean of them and
//! writes back into each stream with its own weight.
//!
//! ```text
//! normed = grouped_norm(x)                                  // [.., streams * width]
//! gate   = sigmoid(up(silu(down(normed) / streams)))        // low-rank gate, [.., streams * width]
//! mixed  = mean_streams(gate * normed)                      // [.., width]: the block's input
//! inject = 2 * sigmoid(inject_proj(normed) / streams)       // [.., streams]
//! x'     = x + block(mixed)[stream] * inject[stream]        // every stream, from the raw x
//! ```
//!
//! The residual add reads the raw stream, not the normed one. There is no plain pre-norm in the
//! layer: this grouped norm is the block's normalization, and no norm follows the last layer either:
//! the model-level mixer (the same gate without the injection) collapses the streams straight into
//! the head.

use poot_graph_ir::ops::{sigmoid, silu};
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced};
use poot_weights::{HyperRole, WeightId, WeightMap};

use crate::components::linear::{Weight, WeightError, linear};
use crate::components::norm::NormParams;
use crate::components::ple::grouped_norm;
use crate::components::standard::ParamError;
use crate::model::{ConfigReason, ParamBlock, TraceError, WiringError};

/// Checked Hyper-Connections parameters: the stream width, the stream count, the gate's low rank
/// and the grouped norm (its epsilon, and whether the stored scale is an offset from one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HyperParams {
    width: usize,
    streams: usize,
    rank: usize,
    norm: NormParams,
}

impl HyperParams {
    pub(super) fn new(
        width: usize,
        streams: usize,
        rank: usize,
        norm: NormParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("hidden_size", width),
            ("hc_count", streams),
            ("hc_lowrank", rank),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        Ok(Self {
            width,
            streams,
            rank,
            norm,
        })
    }

    pub(super) fn streams(&self) -> usize {
        self.streams
    }

    fn stream_width(&self) -> usize {
        self.streams * self.width
    }

    /// The embedding widened to every stream: `[.., width]` tiled `streams` times along the last
    /// axis.
    pub(super) fn widen(&self, b: &Builder, x: Traced) -> Result<Traced, TraceError> {
        let shape = b.aval(x)?.shape;
        let n = shape.len();
        let mut single = shape.clone();
        single.insert(n - 1, 1);
        let mut wide = shape.clone();
        wide.insert(n - 1, self.streams);
        let mut flat = shape;
        flat[n - 1] = self.stream_width();
        Ok(b.reshape(b.broadcast(b.reshape(x, single)?, wide)?, flat)?)
    }

    /// `[.., streams * width]` as `[.., streams, width]`.
    fn grouped(&self, b: &Builder, x: Traced) -> Result<Traced, TraceError> {
        let mut shape = b.aval(x)?.shape;
        let n = shape.len();
        shape[n - 1] = self.streams;
        shape.push(self.width);
        Ok(b.reshape(x, shape)?)
    }
}

/// One gate's weights: the grouped norm scale `[streams * width]`, the low-rank gate projections
/// `[rank, streams * width]` and `[streams * width, rank]`, and for a layer's gate the injection
/// projection `[streams, streams * width]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct HyperWeights {
    norm: Weight,
    down: Weight,
    up: Weight,
    inject: Option<Weight>,
}

impl HyperWeights {
    pub(super) fn new(
        map: &WeightMap,
        id: impl Fn(HyperRole) -> WeightId,
        p: &HyperParams,
        inject: bool,
    ) -> Result<Self, WeightError> {
        let stream = p.stream_width();
        let w = |role, shape: &[usize]| Weight::new(map, id(role), shape);
        Ok(Self {
            norm: w(HyperRole::Norm, &[stream])?,
            down: w(HyperRole::MixDown, &[p.rank, stream])?,
            up: w(HyperRole::MixUp, &[stream, p.rank])?,
            inject: inject
                .then(|| w(HyperRole::Inject, &[p.streams, stream]))
                .transpose()?,
        })
    }
}

/// The normed stream and the gated mean of it: `([.., streams * width], [.., width])`.
fn mixed(
    b: &Builder,
    x: Traced,
    w: &HyperWeights,
    p: &HyperParams,
) -> Result<(Traced, Traced), TraceError> {
    let normed = grouped_norm(b, x, &w.norm, p.streams, p.norm)?;
    let down = b.binary_scalar(
        BinOp::Mul,
        linear(b, normed, &w.down, None)?,
        Scalar::F32(1.0 / p.streams as f32),
    )?;
    let gate = sigmoid(b, linear(b, silu(b, down)?, &w.up, None)?)?;
    let weighted = b.binary(BinOp::Mul, p.grouped(b, gate)?, p.grouped(b, normed)?)?;
    let axis = b.aval(weighted)?.shape.len() - 2;
    let mean = b.binary_scalar(
        BinOp::Mul,
        b.reduce(RedOp::Sum, weighted, axis, false)?,
        Scalar::F32(1.0 / p.streams as f32),
    )?;
    Ok((normed, mean))
}

/// A layer block's input `[.., width]` from the stream `x`, and the per-stream injection weights
/// `[.., streams]` to hand to [`inject`] with the block's output.
pub(super) fn input(
    b: &Builder,
    x: Traced,
    w: &HyperWeights,
    p: &HyperParams,
) -> Result<(Traced, Traced), TraceError> {
    let (normed, mean) = mixed(b, x, w, p)?;
    let inject = WiringError::require(&w.inject, ParamBlock::Injection)?;
    let logit = b.binary_scalar(
        BinOp::Mul,
        linear(b, normed, inject, None)?,
        Scalar::F32(1.0 / p.streams as f32),
    )?;
    Ok((
        mean,
        b.binary_scalar(BinOp::Mul, sigmoid(b, logit)?, Scalar::F32(2.0))?,
    ))
}

/// The stream `x` plus a block's output `[.., width]` written into every stream by its injection
/// weights.
pub(super) fn inject(
    b: &Builder,
    x: Traced,
    block: Traced,
    weights: Traced,
) -> Result<Traced, TraceError> {
    let mut row = b.aval(block)?.shape;
    row.insert(row.len() - 1, 1); // [.., 1, width]
    let mut column = b.aval(weights)?.shape;
    column.push(1); // [.., streams, 1]
    let injection = b.binary(
        BinOp::Mul,
        b.reshape(block, row)?,
        b.reshape(weights, column)?,
    )?;
    Ok(b.binary(BinOp::Add, x, b.reshape(injection, b.aval(x)?.shape)?)?)
}

/// The model-level mixer: the streams collapsed to one `[.., width]` before the head.
pub(super) fn collapse(
    b: &Builder,
    x: Traced,
    w: &HyperWeights,
    p: &HyperParams,
) -> Result<Traced, TraceError> {
    Ok(mixed(b, x, w, p)?.1)
}
