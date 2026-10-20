//! The Mamba2 (SSD) mixer of a hybrid layer (Nemotron-H), one body for every phase.
//!
//! `in_proj` splits into the output gate `z`, the conv input `xBC` and the timestep `dt`; a causal
//! depthwise conv ([`super::conv`]) and SiLU run over `xBC` (the SSM input `x` and the grouped `B`
//! and `C` together), `Delta = softplus(dt + dt_bias)`, and the SSD recurrence runs per head with
//! the decay `A = -exp(A_log)` (the [`ScanFamily::Ssd`] scan, whose reference definition runs
//! [`ops::mamba2_ssd_prefill`] per chunk, each continuing the previous chunk's state exactly: one token is a
//! decode step). The output is the gated
//! RMSNorm `rmsnorm(y * silu(z), scale)` and the output projection.
//!
//! The step carries two recurrent states per layer: the conv window and the SSM state `[rows,
//! heads, state, head_dim]`. The `rows` sequences of a step fold into the head axis of the SSD
//! op (heads of different rows share the weights but not the state).

use poot_graph_ir::{BinOp, Builder, ScanAttrs, ScanFamily, ScanResult, Traced, UnOp, ops};
use poot_tensor::DType;
use poot_weights::{Mamba2Role, WeightId, WeightMap, WeightRole};

use super::conv::{ConvParams, ConvWeights, causal_conv};
use super::linear::{Weight, WeightError, linear};
use super::norm::{NormParams, norm};
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;
use crate::model::TraceError;

/// The chunk length of the SSD scan's reference decomposition (a power of two): a sequence of at most this
/// many positions is one [`ops::mamba2_ssd_prefill`] call, a longer one is cut into chunks that each continue
/// from the previous one's state. A device schedule is free to differ.
const SSD_CHUNK: usize = 64;

/// Checked Mamba2 parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mamba2Params {
    heads: usize,
    head_dim: usize,
    state: usize,
    groups: usize,
    conv: ConvParams,
    norm: NormParams,
}

impl Mamba2Params {
    /// `heads` heads of `head_dim` channels with an SSM state of `state` per channel, `B` and `C`
    /// shared by the heads of each of `groups` groups, a conv of `kernel` taps and the gated norm
    /// `norm`.
    pub fn new(
        heads: usize,
        head_dim: usize,
        state: usize,
        groups: usize,
        kernel: usize,
        norm: NormParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [
            ("mamba_num_heads", heads),
            ("mamba_head_dim", head_dim),
            ("ssm_state_size", state),
            ("n_groups", groups),
        ] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if !heads.is_multiple_of(groups) {
            return Err(ParamError::new(
                "mamba_num_heads",
                ConfigReason::NotDivisible { by: groups },
            ));
        }
        let conv = ConvParams::new(heads * head_dim + 2 * groups * state, kernel, 1)?;
        Ok(Self {
            heads,
            head_dim,
            state,
            groups,
            conv,
            norm,
        })
    }

    /// The SSM's channel width, `heads * head_dim`.
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    /// The SSM head count.
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// The conv's channels: the SSM input and the grouped `B` and `C`.
    pub fn conv_channels(&self) -> usize {
        self.conv.channels()
    }
}

/// A layer's Mamba2 weights: the three row ranges of `in_proj`, the conv, the per-head scalars, the
/// gated norm scale and the output projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mamba2Weights {
    z: Weight,
    xbc: Weight,
    dt: Weight,
    dt_bias: Weight,
    conv: ConvWeights,
    a_log: Weight,
    d: Weight,
    gate_norm: Weight,
    out: Weight,
}

impl Mamba2Weights {
    /// Resolves layer `layer`'s `in_proj` row ranges, conv, per-head scalars, gate norm and output
    /// tensors from `map` and checks each shape against `p` and the model `width`; a missing or
    /// mis-shaped tensor is a typed [`WeightError`].
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &Mamba2Params,
    ) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Mamba2(role));
        let w = |role, shape: &[usize]| Weight::new(map, id(role), shape);
        let (inner, h) = (p.inner(), p.heads);
        Ok(Self {
            z: w(Mamba2Role::Z, &[inner, width])?,
            xbc: w(Mamba2Role::Xbc, &[p.conv_channels(), width])?,
            dt: w(Mamba2Role::Dt, &[h, width])?,
            dt_bias: w(Mamba2Role::DtBias, &[h])?,
            conv: ConvWeights::new(
                map,
                id(Mamba2Role::Conv),
                Some(id(Mamba2Role::ConvBias)),
                &p.conv,
            )?,
            a_log: w(Mamba2Role::ALog, &[h])?,
            d: w(Mamba2Role::D, &[h])?,
            gate_norm: w(Mamba2Role::GateNorm, &[inner])?,
            out: w(Mamba2Role::Out, &[width, inner])?,
        })
    }
}

/// The mixer over `x` (`[rows, tokens, width]`) at layer `layer`: `[rows, tokens, width]`. The
/// layer's conv window (`mamba2.l{layer}.conv`) and SSM state (`mamba2.l{layer}.ssm`) are carried
/// by `step`.
pub fn mamba2(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &Mamba2Weights,
    p: &Mamba2Params,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, t) = (shape.rows.get(), shape.tokens.get());
    let (h, hd, n, g) = (p.heads, p.head_dim, p.state, p.groups);
    let inner = p.inner();

    let z = linear(b, x, &w.z, None)?;
    let xbc = linear(b, x, &w.xbc, None)?;
    let dt = linear(b, x, &w.dt, Some(&w.dt_bias))?;

    let conv = causal_conv(
        b,
        step,
        &format!("mamba2.l{layer}.conv"),
        xbc,
        &w.conv,
        &p.conv,
    )?;
    let xbc = ops::silu(b, conv)?;
    let x_part = b.slice(xbc, 2, 0, inner)?;
    let b_part = b.slice(xbc, 2, inner, inner + g * n)?;
    let c_part = b.slice(xbc, 2, inner + g * n, inner + 2 * g * n)?;

    // `[rows, tokens, heads * d]` to head-major `[rows, heads, tokens, d]`.
    let head_major = |y: Traced, heads: usize, d: usize| {
        Ok::<_, TraceError>(b.transpose(b.reshape(y, vec![rows, t, heads, d])?, vec![0, 2, 1, 3])?)
    };
    let x_heads = head_major(x_part, h, hd)?;
    // Each group's `B` and `C` serve `heads / groups` consecutive heads (`repeat_interleave`).
    let b_heads = ops::repeat_kv(b, head_major(b_part, g, n)?, h / g)?;
    let c_heads = ops::repeat_kv(b, head_major(c_part, g, n)?, h / g)?;
    let delta = ops::softplus(b, head_major(dt, h, 1)?)?;

    // The rows fold into the head axis: `[1, rows * heads, ..]`, the per-head scalars tiled per row.
    let fold = |y: Traced| {
        let mut shape = b.aval(y)?.shape;
        shape.splice(0..2, [1, rows * h]);
        Ok::<_, TraceError>(b.reshape(y, shape)?)
    };
    let per_head = |weight: &Weight, negate_exp: bool| {
        let v = weight.declare_f32(b)?;
        let v = if negate_exp {
            b.unary(UnOp::Neg, b.unary(UnOp::Exp, v)?)? // A = -exp(A_log)
        } else {
            v
        };
        let tiled = b.broadcast(b.reshape(v, vec![1, h, 1, 1])?, vec![rows, h, 1, 1])?;
        Ok::<_, TraceError>(b.reshape(tiled, vec![1, rows * h, 1, 1])?)
    };
    let state =
        step.recurrent_state(b, &format!("mamba2.l{layer}.ssm"), DType::F32, &[h, n, hd])?;
    let ScanResult {
        y,
        s_out: state_out,
        ..
    } = b.scan(
        ScanAttrs {
            family: ScanFamily::Ssd,
            chunk: SSD_CHUNK,
        },
        &[
            fold(x_heads)?,
            fold(b_heads)?,
            fold(c_heads)?,
            fold(delta)?,
            per_head(&w.a_log, true)?,
            per_head(&w.d, false)?,
            fold(state.current())?,
        ],
    )?;
    step.carry_recurrent(b, state, b.reshape(state_out, vec![rows, h, n, hd])?)?;

    let y = b.reshape(y, vec![rows, h, t, hd])?;
    let y = b.reshape(b.transpose(y, vec![0, 2, 1, 3])?, vec![rows, t, inner])?;
    let gated = b.binary(BinOp::Mul, y, ops::silu(b, z)?)?;
    linear(b, norm(b, gated, &w.gate_norm, p.norm)?, &w.out, None)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::Slot;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const WIDTH: usize = 6;
    const HEADS: usize = 4;
    const HEAD_DIM: usize = 2;
    const STATE: usize = 3;
    const GROUPS: usize = 2;
    const KERNEL: usize = 3;
    const EPS: f32 = 1e-5;
    const INNER: usize = HEADS * HEAD_DIM;
    const CONV: usize = INNER + 2 * GROUPS * STATE;

    fn id(role: Mamba2Role) -> WeightId {
        WeightId::layer(0, WeightRole::Mamba2(role))
    }

    fn params() -> Mamba2Params {
        Mamba2Params::new(
            HEADS,
            HEAD_DIM,
            STATE,
            GROUPS,
            KERNEL,
            NormParams::new(EPS).unwrap(),
        )
        .unwrap()
    }

    fn fixture() -> Fx {
        let mut fx = Fx::new(41);
        fx.f32(id(Mamba2Role::Z), &[INNER, WIDTH]);
        fx.bf16(id(Mamba2Role::Xbc), &[CONV, WIDTH]);
        fx.f32(id(Mamba2Role::Dt), &[HEADS, WIDTH]);
        fx.f32(id(Mamba2Role::DtBias), &[HEADS]);
        fx.f32(id(Mamba2Role::Conv), &[CONV, 1, KERNEL]);
        fx.f32(id(Mamba2Role::ConvBias), &[CONV]);
        fx.f32(id(Mamba2Role::ALog), &[HEADS]);
        fx.f32(id(Mamba2Role::D), &[HEADS]);
        fx.set(
            id(Mamba2Role::GateNorm),
            &[INNER],
            (0..INNER).map(|i| 0.8 + 0.05 * i as f32).collect(),
        );
        fx.f32(id(Mamba2Role::Out), &[WIDTH, INNER]);
        fx
    }

    struct Run {
        out: Vec<f32>,
        state: Vec<Vec<f32>>,
    }

    fn run(fx: &Fx, x: &[f32], start: i32, state: &[Vec<f32>]) -> Run {
        let p = params();
        let w = Mamba2Weights::new(&fx.map(), 0, WIDTH, &p).unwrap();
        let tokens = x.len() / WIDTH;
        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All)).unwrap();
        let xs = b
            .constant("x", poot_graph_ir::TensorType::f32(vec![1, tokens, WIDTH]))
            .unwrap();
        let y = mamba2(&b, &step, 0, xs, &w, &p).unwrap();
        let g = step.finish(b, y).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
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

    /// The mixer over `x` token by token, written from the Mamba2 recurrence in f64: the outputs,
    /// the conv window (the last `K - 1` raw conv inputs) and the SSM state `[heads, state, dim]`.
    fn reference(fx: &Fx, x: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let v = |role| fx.values(id(role));
        let (conv_w, conv_b) = (v(Mamba2Role::Conv), v(Mamba2Role::ConvBias));
        let (dt_bias, a_log, d) = (v(Mamba2Role::DtBias), v(Mamba2Role::ALog), v(Mamba2Role::D));
        let norm_w = v(Mamba2Role::GateNorm);
        let mut raw_inputs: Vec<Vec<f64>> = vec![vec![0.0; CONV]; KERNEL - 1];
        let mut state = vec![0.0f64; HEADS * STATE * HEAD_DIM];
        let mut out = Vec::new();
        for token in x.chunks(WIDTH) {
            let row: Vec<f64> = token.iter().map(|&v| v.into()).collect();
            let z = matvec(&v(Mamba2Role::Z), INNER, &row);
            let xbc = matvec(&v(Mamba2Role::Xbc), CONV, &row);
            let dt = matvec(&v(Mamba2Role::Dt), HEADS, &row);
            raw_inputs.push(xbc);
            let n = raw_inputs.len();
            let conv: Vec<f64> = (0..CONV)
                .map(|c| {
                    let sum: f64 = (0..KERNEL)
                        .map(|j| f64::from(conv_w[c * KERNEL + j]) * raw_inputs[n - KERNEL + j][c])
                        .sum();
                    silu(sum + f64::from(conv_b[c]))
                })
                .collect();
            let (xs, bs, cs) = (
                &conv[..INNER],
                &conv[INNER..INNER + GROUPS * STATE],
                &conv[INNER + GROUPS * STATE..],
            );
            let mut y = [0.0; INNER];
            for h in 0..HEADS {
                let delta = (1.0 + (dt[h] + f64::from(dt_bias[h])).exp()).ln();
                let a = (delta * -f64::from(a_log[h]).exp()).exp();
                let g = h / (HEADS / GROUPS);
                for s in 0..STATE {
                    for p in 0..HEAD_DIM {
                        let cell = &mut state[(h * STATE + s) * HEAD_DIM + p];
                        *cell = a * *cell + delta * bs[g * STATE + s] * xs[h * HEAD_DIM + p];
                        y[h * HEAD_DIM + p] += cs[g * STATE + s] * *cell;
                    }
                }
                for p in 0..HEAD_DIM {
                    y[h * HEAD_DIM + p] += f64::from(d[h]) * xs[h * HEAD_DIM + p];
                }
            }
            let gated: Vec<f64> = y.iter().zip(&z).map(|(y, z)| y * silu(*z)).collect();
            let inv = 1.0
                / (gated.iter().map(|v| v * v).sum::<f64>() / INNER as f64 + f64::from(EPS)).sqrt();
            let normed: Vec<f64> = gated
                .iter()
                .zip(&norm_w)
                .map(|(v, w)| v * inv * f64::from(*w))
                .collect();
            out.extend(matvec(&v(Mamba2Role::Out), WIDTH, &normed));
        }
        let window: Vec<f64> = raw_inputs[raw_inputs.len() - (KERNEL - 1)..].concat();
        (out, window, state)
    }

    fn inputs(tokens: usize) -> Vec<f32> {
        oracle::random(tokens * WIDTH, 9)
            .into_iter()
            .map(|v| 2.0 * v)
            .collect()
    }

    /// SC-002 and the component half of SC-003: the mixer equals the f64 recurrence (outputs, conv
    /// window and SSM state), a prompt split into chunks or continued token by token equals the
    /// whole one at every output and every state element, and each weight matters. Mutation: zero
    /// the carried conv window (`causal_conv`'s `window` input) in a continued chunk; the second
    /// chunk's first output goes red.
    #[test]
    fn the_mixer_matches_the_recurrence_and_continues_across_chunks() {
        let fx = fixture();
        let x = inputs(6);
        let (want, window, state) = reference(&fx, &x);
        let whole = run(&fx, &x, 0, &[]);
        oracle::assert_matches_f64(&whole.out, &want, 1e-4);
        oracle::assert_matches_f64(&whole.state[0], &window, 1e-4);
        oracle::assert_matches_f64(&whole.state[1], &state, 1e-4);

        let first = run(&fx, &x[..3 * WIDTH], 0, &[]);
        let second = run(&fx, &x[3 * WIDTH..], 3, &first.state);
        oracle::assert_matches_f64(&[first.out.clone(), second.out].concat(), &want, 1e-4);
        oracle::assert_matches_f64(&second.state[1], &state, 1e-4);

        let mut steps = first;
        let mut stepped = steps.out.clone();
        for t in 3..6 {
            steps = run(&fx, &x[t * WIDTH..(t + 1) * WIDTH], t as i32, &steps.state);
            stepped.extend(&steps.out);
        }
        oracle::assert_matches_f64(&stepped, &want, 1e-4);
        oracle::assert_matches_f64(&steps.state[0], &window, 1e-4);
        oracle::assert_matches_f64(&steps.state[1], &state, 1e-4);

        for role in [
            Mamba2Role::Z,
            Mamba2Role::Xbc,
            Mamba2Role::Dt,
            Mamba2Role::DtBias,
            Mamba2Role::Conv,
            Mamba2Role::ConvBias,
            Mamba2Role::ALog,
            Mamba2Role::D,
            Mamba2Role::GateNorm,
            Mamba2Role::Out,
        ] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id(role));
            oracle::assert_differs(&run(&perturbed, &x, 0, &[]).out, &whole.out, id(role));
        }
    }

    #[test]
    fn params_refuse_groups_that_do_not_divide_the_heads() {
        let norm = NormParams::new(EPS).unwrap();
        assert_eq!(
            Mamba2Params::new(4, 2, 3, 3, 3, norm),
            Err(ParamError::new(
                "mamba_num_heads",
                ConfigReason::NotDivisible { by: 3 }
            ))
        );
        assert_eq!(
            Mamba2Params::new(4, 2, 0, 2, 3, norm),
            Err(ParamError::new("ssm_state_size", ConfigReason::Zero))
        );
    }
}
