//! The Kimi Delta Attention (KDA) mixer of a hybrid layer (GLM-5.3-Flash), one body for every
//! phase.
//!
//! A layer projects its input into a query, a key and a value per head, runs a causal depthwise
//! conv ([`super::conv`]) and SiLU over each of the three, L2-normalizes each query and key head and
//! runs the gated delta rule per head with a decay per key channel:
//!
//! ```text
//! g      = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))   # one per head and key channel
//! S      = exp(g) * S                       # forget: row i of S decays by exp(g[i])
//! delta  = beta * (v - S^T k)               # the delta-rule correction
//! S      = S + outer(k, delta)
//! o      = S^T (q / sqrt(head_dim))
//! ```
//!
//! The output is `rmsnorm(o, scale) * sigmoid(g_b(g_a(x)))` and the output projection. The state is a
//! `[head_dim, head_dim]` matrix per head carried across steps, and each conv carries its input
//! window ([`super::conv::causal_conv`]), so a chunk continues the previous one exactly and one token
//! is a decode step.
//!
//! The recurrence is traced token by token: a step of `T` tokens emits `T` copies of the update, in
//! primitive operations. The size of that expansion is the caller's bound on a step's tokens (the
//! driver's trace limit) and the common graph budgets; a chunk-parallel schedule is a later
//! component. The decay differs per key channel, so the closed-form chunked solution of
//! [`super::gdn`] does not apply as it stands.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, UnOp, ops};
use poot_quant::weights::{KdaRole, WeightId, WeightMap, WeightRole};
use poot_tensor::DType;

use super::conv::{ConvParams, ConvWeights, causal_conv};
use super::linear::{Weight, WeightError, linear};
use super::norm::NormParams;
use super::standard::{ParamError, Step};
use crate::model::ConfigReason;
use crate::model::TraceError;

/// The epsilon added to the sum of squares of the L2 norm of a query or key head.
const L2_EPS: f32 = 1e-6;

/// Checked KDA parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdaParams {
    heads: usize,
    head_dim: usize,
    conv: ConvParams,
    gate_lower_bound: f32,
    norm: NormParams,
}

impl KdaParams {
    /// `heads` heads of `head_dim` channels each, a conv of `kernel` taps over each of the query,
    /// key and value channels, the log forget gate in `(gate_lower_bound, 0)` (a negative bound) and
    /// `norm` (its epsilon) for the output RMSNorm.
    pub fn new(
        heads: usize,
        head_dim: usize,
        kernel: usize,
        gate_lower_bound: f32,
        norm: NormParams,
    ) -> Result<Self, ParamError> {
        for (field, value) in [("num_heads", heads), ("head_dim", head_dim)] {
            if value == 0 {
                return Err(ParamError::new(field, ConfigReason::Zero));
            }
        }
        if !(gate_lower_bound.is_finite() && gate_lower_bound < 0.0) {
            return Err(ParamError::new(
                "gate_lower_bound",
                ConfigReason::Unsupported,
            ));
        }
        Ok(Self {
            heads,
            head_dim,
            conv: ConvParams::new(heads * head_dim, kernel, 1)?,
            gate_lower_bound,
            norm,
        })
    }

    /// The width of each of the query, key and value projections: `heads * head_dim`.
    pub fn width(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// A layer's KDA weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KdaWeights {
    q: Weight,
    k: Weight,
    v: Weight,
    conv_q: ConvWeights,
    conv_k: ConvWeights,
    conv_v: ConvWeights,
    f_a: Weight,
    f_b: Weight,
    beta: Weight,
    g_a: Weight,
    g_b: Weight,
    a_log: Weight,
    dt_bias: Weight,
    o_norm: Weight,
    out: Weight,
}

impl KdaWeights {
    /// The weights of layer `layer` of a model of `width` hidden units. The two gates are low-rank
    /// with a rank of `head_dim`.
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &KdaParams,
    ) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Kda(role));
        let w = |role, shape: &[usize]| Weight::new(map, id(role), shape);
        let conv = |role| ConvWeights::new(map, id(role), None, &p.conv);
        let (qkv, d, h) = (p.width(), p.head_dim, p.heads);
        Ok(Self {
            q: w(KdaRole::Q, &[qkv, width])?,
            k: w(KdaRole::K, &[qkv, width])?,
            v: w(KdaRole::V, &[qkv, width])?,
            conv_q: conv(KdaRole::ConvQ)?,
            conv_k: conv(KdaRole::ConvK)?,
            conv_v: conv(KdaRole::ConvV)?,
            f_a: w(KdaRole::FA, &[d, width])?,
            f_b: w(KdaRole::FB, &[qkv, d])?,
            beta: w(KdaRole::Beta, &[h, width])?,
            g_a: w(KdaRole::GA, &[d, width])?,
            g_b: w(KdaRole::GB, &[qkv, d])?,
            a_log: w(KdaRole::ALog, &[h])?,
            dt_bias: w(KdaRole::DtBias, &[qkv])?,
            o_norm: w(KdaRole::ONorm, &[d])?,
            out: w(KdaRole::Out, &[width, qkv])?,
        })
    }
}

/// `x / sqrt(sum(x^2) + eps)` over the last axis.
fn l2_norm_last(b: &Builder, x: Traced) -> Result<Traced, TraceError> {
    let last = b.aval(x)?.shape.len() - 1;
    let squares = b.reduce(RedOp::Sum, b.binary(BinOp::Mul, x, x)?, last, true)?;
    let norm = b.unary(
        UnOp::Sqrt,
        b.binary_scalar(BinOp::Add, squares, Scalar::F32(L2_EPS))?,
    )?;
    Ok(b.binary(BinOp::Div, x, norm)?)
}

/// `[rows, tokens, heads * d]` as `[rows, heads, tokens, d]`.
fn heads_first(b: &Builder, y: Traced, heads: usize, d: usize) -> Result<Traced, TraceError> {
    let shape = b.aval(y)?.shape;
    Ok(b.transpose(
        b.reshape(y, vec![shape[0], shape[1], heads, d])?,
        vec![0, 2, 1, 3],
    )?)
}

/// The mixer over `x` (`[rows, tokens, width]`) at layer `layer`: `[rows, tokens, width]`. The
/// layer's three conv windows (`kda.l{layer}.conv_q`, `.conv_k`, `.conv_v`) and state matrices
/// (`kda.l{layer}.state`) are carried by `step`.
pub fn kda(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &KdaWeights,
    p: &KdaParams,
) -> Result<Traced, TraceError> {
    let shape = step.shape();
    let (rows, tokens) = (shape.rows.get(), shape.tokens.get());
    let (h, d) = (p.heads, p.head_dim);

    let mixed = |proj: &Weight, conv: &ConvWeights, name: &str| {
        let conv = causal_conv(
            b,
            step,
            &format!("kda.l{layer}.{name}"),
            linear(b, x, proj, None)?,
            conv,
            &p.conv,
        )?;
        heads_first(b, ops::silu(b, conv)?, h, d)
    };
    let q = l2_norm_last(b, mixed(&w.q, &w.conv_q, "conv_q")?)?;
    let k = l2_norm_last(b, mixed(&w.k, &w.conv_k, "conv_k")?)?;
    let v = mixed(&w.v, &w.conv_v, "conv_v")?;
    let q = b.binary_scalar(BinOp::Mul, q, Scalar::F32(1.0 / (d as f32).sqrt()))?;

    // The log forget gate per head and key channel, and the decay it gives.
    let f = linear(b, linear(b, x, &w.f_a, None)?, &w.f_b, None)?;
    let f = heads_first(b, b.binary(BinOp::Add, f, w.dt_bias.declare_f32(b)?)?, h, d)?;
    let rate = b.reshape(
        b.unary(UnOp::Exp, w.a_log.declare_f32(b)?)?,
        vec![1, h, 1, 1],
    )?;
    let gate = ops::sigmoid(b, b.binary(BinOp::Mul, rate, f)?)?;
    let decay = b.unary(
        UnOp::Exp,
        b.binary_scalar(BinOp::Mul, gate, Scalar::F32(p.gate_lower_bound))?,
    )?; // [rows, heads, tokens, d]
    let beta = heads_first(b, ops::sigmoid(b, linear(b, x, &w.beta, None)?)?, h, 1)?;

    let state = step.recurrent_state(b, &format!("kda.l{layer}.state"), DType::F32, &[h, d, d])?;
    let (mut s, mut outs) = (state.current(), Vec::with_capacity(tokens));
    for t in 0..tokens {
        let at = |y: Traced| Ok::<_, TraceError>(b.slice(y, 2, t, t + 1)?); // [rows, heads, 1, ..]
        let (q, k, v) = (at(q)?, at(k)?, at(v)?);
        // Row `i` of the state, the key channel, decays by `exp(g[i])`.
        let decay = b.transpose(at(decay)?, vec![0, 1, 3, 2])?; // [rows, heads, d, 1]
        s = b.binary(BinOp::Mul, s, decay)?;
        let read = b.matmul(k, s)?; // [rows, heads, 1, d]
        let delta = b.binary(BinOp::Mul, at(beta)?, b.binary(BinOp::Sub, v, read)?)?;
        s = b.binary(
            BinOp::Add,
            s,
            b.matmul(b.transpose(k, vec![0, 1, 3, 2])?, delta)?,
        )?;
        outs.push(b.matmul(q, s)?);
    }
    step.carry_recurrent(b, state, s)?;
    let o = if tokens == 1 {
        outs[0]
    } else {
        b.concat(2, &outs)?
    };

    let o = ops::rmsnorm(b, o, w.o_norm.declare_f32(b)?, p.norm.eps())?;
    let g = linear(b, linear(b, x, &w.g_a, None)?, &w.g_b, None)?;
    let o = b.binary(BinOp::Mul, o, ops::sigmoid(b, heads_first(b, g, h, d)?)?)?;
    let o = b.reshape(b.transpose(o, vec![0, 2, 1, 3])?, vec![rows, tokens, h * d])?;
    linear(b, o, &w.out, None)
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
    const HEADS: usize = 2;
    const DIM: usize = 4;
    const QKV: usize = HEADS * DIM;
    const KERNEL: usize = 3;
    const LOWER_BOUND: f32 = -5.0;
    const EPS: f32 = 1e-5;
    const TOKENS: usize = 6;

    fn id(role: KdaRole) -> WeightId {
        WeightId::layer(0, WeightRole::Kda(role))
    }

    fn params() -> KdaParams {
        KdaParams::new(
            HEADS,
            DIM,
            KERNEL,
            LOWER_BOUND,
            NormParams::new(EPS).unwrap(),
        )
        .unwrap()
    }

    fn fixture() -> Fx {
        let mut fx = Fx::new(31);
        for role in [KdaRole::Q, KdaRole::K, KdaRole::V] {
            fx.bf16(id(role), &[QKV, WIDTH]);
        }
        for role in [KdaRole::ConvQ, KdaRole::ConvK, KdaRole::ConvV] {
            fx.bf16(id(role), &[QKV, 1, KERNEL]);
        }
        for role in [KdaRole::FA, KdaRole::GA] {
            fx.bf16(id(role), &[DIM, WIDTH]);
        }
        for role in [KdaRole::FB, KdaRole::GB] {
            fx.bf16(id(role), &[QKV, DIM]);
        }
        fx.bf16(id(KdaRole::Beta), &[HEADS, WIDTH]);
        fx.set(id(KdaRole::ALog), &[HEADS], vec![0.3, -0.4]);
        fx.f32(id(KdaRole::DtBias), &[QKV]);
        fx.set(id(KdaRole::ONorm), &[DIM], vec![1.1, 0.9, 1.2, 0.8]);
        fx.bf16(id(KdaRole::Out), &[WIDTH, QKV]);
        fx
    }

    struct Run {
        out: Vec<f32>,
        state: Vec<Vec<f32>>,
    }

    /// `x` through the component from `state` (zeros when empty), positions from `start`.
    fn run(fx: &Fx, x: &[f32], start: i32, state: &[Vec<f32>]) -> Run {
        let p = params();
        let w = KdaWeights::new(&fx.map(), 0, WIDTH, &p).unwrap();
        let tokens = x.len() / WIDTH;
        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All)).unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![1, tokens, WIDTH]))
            .unwrap();
        let y = kda(&b, &step, 0, xs, &w, &p).unwrap();
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

    fn sigmoid(v: f64) -> f64 {
        1.0 / (1.0 + (-v).exp())
    }

    fn silu(v: f64) -> f64 {
        v * sigmoid(v)
    }

    fn matvec(w: &[f32], rows: usize, x: &[f64]) -> Vec<f64> {
        let k = x.len();
        (0..rows)
            .map(|r| (0..k).map(|j| f64::from(w[r * k + j]) * x[j]).sum())
            .collect()
    }

    /// The names and layouts of the state the component carries, in carry order: the three conv
    /// windows `[K - 1, QKV]` and the matrices `[head][key channel][value channel]`.
    struct Want {
        out: Vec<f64>,
        windows: [Vec<f64>; 3],
        state: Vec<f64>,
    }

    /// The mixer token by token from the definition, in f64.
    fn reference(fx: &Fx, x: &[f32]) -> Want {
        let v = |role| fx.values(id(role));
        let proj = [v(KdaRole::Q), v(KdaRole::K), v(KdaRole::V)];
        let kernels = [v(KdaRole::ConvQ), v(KdaRole::ConvK), v(KdaRole::ConvV)];
        let mut raw: [Vec<Vec<f64>>; 3] = std::array::from_fn(|_| vec![vec![0.0; QKV]; KERNEL - 1]);
        let mut state = vec![0.0f64; HEADS * DIM * DIM];
        let mut out = Vec::new();
        for token in x.chunks(WIDTH) {
            let row: Vec<f64> = token.iter().map(|&t| t.into()).collect();
            let mut conv: [Vec<f64>; 3] = Default::default();
            for i in 0..3 {
                raw[i].push(matvec(&proj[i], QKV, &row));
                let n = raw[i].len();
                conv[i] = (0..QKV)
                    .map(|ch| {
                        silu(
                            (0..KERNEL)
                                .map(|j| {
                                    f64::from(kernels[i][ch * KERNEL + j])
                                        * raw[i][n - KERNEL + j][ch]
                                })
                                .sum(),
                        )
                    })
                    .collect();
            }
            let f = matvec(&v(KdaRole::FB), QKV, &matvec(&v(KdaRole::FA), DIM, &row));
            let dt = v(KdaRole::DtBias);
            let a = v(KdaRole::ALog);
            let beta = matvec(&v(KdaRole::Beta), HEADS, &row);
            let l2 = |head: &[f64]| -> Vec<f64> {
                let norm = (head.iter().map(|x| x * x).sum::<f64>() + f64::from(L2_EPS)).sqrt();
                head.iter().map(|x| x / norm).collect()
            };
            let mut o_flat = vec![0.0f64; QKV];
            for h in 0..HEADS {
                let range = h * DIM..(h + 1) * DIM;
                let q = l2(&conv[0][range.clone()]);
                let k = l2(&conv[1][range.clone()]);
                let value = &conv[2][range.clone()];
                let s = &mut state[h * DIM * DIM..(h + 1) * DIM * DIM];
                for i in 0..DIM {
                    let ch = h * DIM + i;
                    let g = f64::from(LOWER_BOUND)
                        * sigmoid(f64::from(a[h]).exp() * (f[ch] + f64::from(dt[ch])));
                    s[i * DIM..(i + 1) * DIM]
                        .iter_mut()
                        .for_each(|e| *e *= g.exp());
                }
                let b = sigmoid(beta[h]);
                let delta: Vec<f64> = (0..DIM)
                    .map(|j| {
                        let read: f64 = (0..DIM).map(|i| s[i * DIM + j] * k[i]).sum();
                        b * (value[j] - read)
                    })
                    .collect();
                for i in 0..DIM {
                    for j in 0..DIM {
                        s[i * DIM + j] += k[i] * delta[j];
                    }
                }
                let o: Vec<f64> = (0..DIM)
                    .map(|j| {
                        (0..DIM).map(|i| s[i * DIM + j] * q[i]).sum::<f64>() / (DIM as f64).sqrt()
                    })
                    .collect();
                let inv = 1.0
                    / (o.iter().map(|x| x * x).sum::<f64>() / DIM as f64 + f64::from(EPS)).sqrt();
                let g_gate = matvec(&v(KdaRole::GB), QKV, &matvec(&v(KdaRole::GA), DIM, &row));
                let scale = v(KdaRole::ONorm);
                for e in 0..DIM {
                    o_flat[h * DIM + e] =
                        o[e] * inv * f64::from(scale[e]) * sigmoid(g_gate[h * DIM + e]);
                }
            }
            out.extend(matvec(&v(KdaRole::Out), WIDTH, &o_flat));
        }
        let windows = raw.map(|r| r[r.len() - (KERNEL - 1)..].concat());
        Want {
            out,
            windows,
            state,
        }
    }

    fn inputs() -> Vec<f32> {
        oracle::random(TOKENS * WIDTH, 17)
            .into_iter()
            .map(|v| 2.0 * v)
            .collect()
    }

    fn assert_state(got: &[Vec<f32>], want: &Want) {
        assert_eq!(got.len(), 4, "three conv windows and the matrices");
        for (window, want) in got.iter().zip(&want.windows) {
            oracle::assert_matches_f64(window, want, 1e-4);
        }
        oracle::assert_matches_f64(&got[3], &want.state, 1e-4);
    }

    /// The mixer equals the f64 recurrence at every output and every carried element, whole, in two
    /// chunks and token by token, and each weight changes the output. Mutation: apply the decay after
    /// the state update (move the multiplication below the update); the whole-run rows go red at the
    /// first output.
    #[test]
    fn the_mixer_matches_the_recurrence_and_continues_across_chunks() {
        let fx = fixture();
        let x = inputs();
        let want = reference(&fx, &x);
        let whole = run(&fx, &x, 0, &[]);
        oracle::assert_matches_f64(&whole.out, &want.out, 1e-4);
        assert_state(&whole.state, &want);

        let first = run(&fx, &x[..3 * WIDTH], 0, &[]);
        let second = run(&fx, &x[3 * WIDTH..], 3, &first.state);
        oracle::assert_matches_f64(&[first.out.clone(), second.out].concat(), &want.out, 1e-4);
        assert_state(&second.state, &want);

        let mut steps = run(&fx, &x[..WIDTH], 0, &[]);
        let mut stepped = steps.out.clone();
        for t in 1..TOKENS {
            steps = run(&fx, &x[t * WIDTH..(t + 1) * WIDTH], t as i32, &steps.state);
            stepped.extend(&steps.out);
        }
        oracle::assert_matches_f64(&stepped, &want.out, 1e-4);
        assert_state(&steps.state, &want);

        for role in [
            KdaRole::Q,
            KdaRole::K,
            KdaRole::V,
            KdaRole::ConvQ,
            KdaRole::ConvK,
            KdaRole::ConvV,
            KdaRole::FA,
            KdaRole::FB,
            KdaRole::Beta,
            KdaRole::GA,
            KdaRole::GB,
            KdaRole::ALog,
            KdaRole::DtBias,
            KdaRole::ONorm,
            KdaRole::Out,
        ] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id(role));
            oracle::assert_differs(&run(&perturbed, &x, 0, &[]).out, &whole.out, id(role));
        }
    }

    /// Two rows with different prompts give each row's output and state of the same prompt run
    /// alone.
    #[test]
    fn two_rows_do_not_share_state() {
        let fx = fixture();
        let prompts = [inputs(), oracle::random(TOKENS * WIDTH, 23)];
        let single: Vec<Run> = prompts.iter().map(|x| run(&fx, x, 0, &[])).collect();

        let p = params();
        let w = KdaWeights::new(&fx.map(), 0, WIDTH, &p).unwrap();
        let b = Builder::for_tests();
        let shape = crate::model::StepShape {
            rows: std::num::NonZeroUsize::new(2).unwrap(),
            ..step_shape(TOKENS, CAP, LogitRows::All)
        };
        let step = Step::new(&b, shape).unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![2, TOKENS, WIDTH]))
            .unwrap();
        let y = kda(&b, &step, 0, xs, &w, &p).unwrap();
        let g = step.finish(b, y).unwrap();
        let zeros: Vec<Vec<f32>> = g
            .state()
            .map(|(si, _)| vec![0.0; g.aval(si).numel()])
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
    fn params_refuse_empty_extents_and_a_decay_that_does_not_forget() {
        let norm = NormParams::new(EPS).unwrap();
        assert_eq!(
            KdaParams::new(0, DIM, KERNEL, LOWER_BOUND, norm),
            Err(ParamError::new("num_heads", ConfigReason::Zero))
        );
        assert_eq!(
            KdaParams::new(HEADS, DIM, 1, LOWER_BOUND, norm),
            Err(ParamError::new("conv_kernel", ConfigReason::Zero))
        );
        for bound in [0.0, 1.0, f32::NAN] {
            assert_eq!(
                KdaParams::new(HEADS, DIM, KERNEL, bound, norm),
                Err(ParamError::new(
                    "gate_lower_bound",
                    ConfigReason::Unsupported
                ))
            );
        }
    }
}
