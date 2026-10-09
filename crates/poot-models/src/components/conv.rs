//! The causal depthwise convolution over the token axis with a carried input window: the short
//! conv in front of a Mamba2 mixer, and any other depthwise causal conv a family needs.
//!
//! The component is the graph op [`ops::causal_conv1d_windowed`] over a window `[rows,
//! (K-1) * dilation, C]` that the step carries as recurrent state (the previous inputs, oldest
//! first), so a chunk of tokens continues the previous chunk exactly and one token is a decode
//! step: one body for every phase, and no zero pad after the first chunk. The kernel is stored as
//! PyTorch stores a depthwise `nn.Conv1d`, `[C, 1, K]`, or as a GGUF stores it, `[C, K]`.

use poot_graph_ir::{BinOp, Builder, Traced, ops};
use poot_quant::weights::{WeightId, WeightMap};
use poot_tensor::DType;

use super::linear::{Weight, WeightError};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;

/// Checked conv parameters: `kernel >= 2` taps `dilation` positions apart over `channels`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvParams {
    channels: usize,
    kernel: usize,
    dilation: usize,
}

impl ConvParams {
    pub fn new(channels: usize, kernel: usize, dilation: usize) -> Result<Self, ParamError> {
        if channels == 0 {
            return Err(ParamError::new("conv_channels", ConfigReason::Zero));
        }
        if dilation == 0 {
            return Err(ParamError::new("conv_dilation", ConfigReason::Zero));
        }
        // A one-tap conv has no past to carry: it is a per-channel scale, not this component.
        if kernel < 2 {
            return Err(ParamError::new("conv_kernel", ConfigReason::Zero));
        }
        Ok(Self {
            channels,
            kernel,
            dilation,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// The inputs the carried window holds: `(K - 1) * dilation`.
    pub fn window(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }
}

/// The conv's kernel `[C, 1, K]` and, when it has one, its per-channel bias `[C]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvWeights {
    kernel: Weight,
    bias: Option<Weight>,
}

impl ConvWeights {
    /// The kernel stored under `kernel` (`[C, 1, K]`, or `[C, K]` as a GGUF stores it) and the bias
    /// under `bias`, shape-checked against `p`.
    pub fn new(
        map: &WeightMap,
        kernel: WeightId,
        bias: Option<WeightId>,
        p: &ConvParams,
    ) -> Result<Self, WeightError> {
        Ok(Self {
            kernel: Weight::new(map, kernel, &[p.channels, 1, p.kernel]).or_else(|first| {
                // Only a shape mismatch has another spelling to try; report the PyTorch one when
                // neither fits.
                Weight::new(map, kernel, &[p.channels, p.kernel]).map_err(|second| match second {
                    WeightError::Shape { .. } => first,
                    other => other,
                })
            })?,
            bias: bias
                .map(|id| Weight::new(map, id, &[p.channels]))
                .transpose()?,
        })
    }
}

/// The conv of `x` (`[rows, tokens, C]`) with the window carried as the step's state `state_name`
/// (`[rows, (K-1) * dilation, C]`, zeros before the first token): `[rows, tokens, C]`, plus the
/// bias when the weights have one. No activation.
pub fn causal_conv(
    b: &Builder,
    step: &Step,
    state_name: &str,
    x: Traced,
    w: &ConvWeights,
    p: &ConvParams,
) -> Traced {
    let window = step.recurrent_state(b, state_name, DType::F32, &[p.window(), p.channels]);
    // `[C, 1, K]` -> the op's `[K, C]` taps.
    let taps = b.transpose(
        b.reshape(w.kernel.declare_f32(b), vec![p.channels, p.kernel]),
        vec![1, 0],
    );
    let (out, next) =
        ops::causal_conv1d_windowed(b, x, taps, window.current(), p.kernel, p.dilation);
    step.carry_recurrent(b, window, next);
    match &w.bias {
        Some(bias) => {
            let shape = b.aval(out).shape;
            b.binary(BinOp::Add, out, b.broadcast(bias.declare_f32(b), shape))
        }
        None => out,
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::Slot;
    use poot_quant::weights::{Mamba2Role, WeightRole};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;
    use poot_graph_ir::TensorType;

    const C: usize = 3;
    const K: usize = 3;
    const T: usize = 6;

    fn kernel_id() -> WeightId {
        WeightId::layer(0, WeightRole::Mamba2(Mamba2Role::Conv))
    }

    fn bias_id() -> WeightId {
        WeightId::layer(0, WeightRole::Mamba2(Mamba2Role::ConvBias))
    }

    fn fixture() -> Fx {
        let mut fx = Fx::new(5);
        fx.bf16(kernel_id(), &[C, 1, K]);
        fx.f32(bias_id(), &[C]);
        fx
    }

    /// One chunk of `tokens` inputs through the component over `state` (zeros when empty): the
    /// output and the carried window.
    fn run(fx: &Fx, dilation: usize, x: &[f32], state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let p = ConvParams::new(C, K, dilation).unwrap();
        let w = ConvWeights::new(&fx.map(), kernel_id(), Some(bias_id()), &p).unwrap();
        let tokens = x.len() / C;
        let b = Builder::new();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All));
        let xs = b.constant("x", TensorType::f32(vec![1, tokens, C]));
        let y = causal_conv(&b, &step, "conv.window", xs, &w, &p);
        let g = step.finish(b, y);
        let zeros = vec![0.0; p.window() * C];
        let state = if state.is_empty() { &zeros[..] } else { state };
        let ids: Vec<i32> = (0..tokens as i32).collect();
        let (out, mut next) = fx.eval(
            &g,
            &[("x", x)],
            &[(Slot::Token, &ids), (Slot::Pos, &ids)],
            &[state],
        );
        (out, next.remove(0))
    }

    /// `out[t, c] = bias[c] + sum_j w[c, j] * x[t - (K-1-j) * dilation, c]` with `x[<0] = 0`, in
    /// f64.
    fn reference(fx: &Fx, dilation: usize, x: &[f32]) -> Vec<f64> {
        let (w, bias) = (fx.values(kernel_id()), fx.values(bias_id()));
        let tokens = x.len() / C;
        let mut out = Vec::new();
        for t in 0..tokens {
            for c in 0..C {
                let mut sum = f64::from(bias[c]);
                for j in 0..K {
                    let back = (K - 1 - j) * dilation;
                    if t >= back {
                        sum += f64::from(w[c * K + j]) * f64::from(x[(t - back) * C + c]);
                    }
                }
                out.push(sum);
            }
        }
        out
    }

    /// SC-002: the conv equals the f64 reference (zero history) for dilation 1 and 2, a chunk
    /// continued from the carried window equals the whole sequence at every output and the final
    /// window, and each weight matters. Mutation: zero the carried window in the second chunk
    /// (today's zero pad); the continued rows go red at the first outputs of the second chunk.
    #[test]
    fn the_conv_matches_an_f64_reference_and_a_continued_chunk_equals_the_whole() {
        let fx = fixture();
        let x = oracle::random(T * C, 9);
        for dilation in [1, 2] {
            let (whole, whole_window) = run(&fx, dilation, &x, &[]);
            oracle::assert_matches_f64(&whole, &reference(&fx, dilation, &x), 1e-5);
            let (first, window) = run(&fx, dilation, &x[..3 * C], &[]);
            let (second, second_window) = run(&fx, dilation, &x[3 * C..], &window);
            let continued = [first, second].concat();
            oracle::assert_matches_f64(
                &continued,
                &whole.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
            oracle::assert_matches_f64(
                &second_window,
                &whole_window
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect::<Vec<_>>(),
                1e-6,
            );
        }
        let base = run(&fx, 1, &x, &[]).0;
        for id in [kernel_id(), bias_id()] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id);
            oracle::assert_differs(&run(&perturbed, 1, &x, &[]).0, &base, id);
        }
    }

    #[test]
    fn params_refuse_a_conv_with_nothing_to_carry() {
        assert_eq!(
            ConvParams::new(C, 1, 1),
            Err(ParamError::new("conv_kernel", ConfigReason::Zero))
        );
        assert_eq!(
            ConvParams::new(C, 3, 0),
            Err(ParamError::new("conv_dilation", ConfigReason::Zero))
        );
        assert_eq!(
            ConvParams::new(0, 3, 1),
            Err(ParamError::new("conv_channels", ConfigReason::Zero))
        );
    }
}
