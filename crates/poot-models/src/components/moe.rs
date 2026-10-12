//! The routed mixture-of-experts feed-forward block: route every token ([`super::router`]), run each
//! selected expert's gated FFN over the rows routed to it and sum the weighted outputs back per
//! token. One body for every family (spec 999): families differ in their router, their expert
//! storage and their GLU, all of them parameters here.
//!
//! The `T` tokens of a step and their `top_k` slots flatten to `M = T * top_k` rows, row
//! `t * top_k + r` being token `t` at slot `r`, so every projection is one indexed contraction over
//! all rows and no expert is evaluated for a token that did not select it.
//!
//! Experts bind as stored (ADR-0103). A checkpoint holds one projection of every expert as a stack, mapped
//! as one [`MoeRole::Stack`] weight (a stored `[experts, out, K]` tensor, or the per-expert tensors, dense or
//! packed, viewed as one stack), and every projection is the logical dense
//! [`poot_graph_ir::Builder::indexed_matmul`] over it: a `[experts, out, K]` stack is read through a
//! transposed view that `compile` folds, never a copy. A packed stack is rewritten by the one packed-weight
//! transform (`poot_graph_plan::bind_packed_weights`) into one indexed packed contraction over the stored
//! blocks, so there is no packed arm here.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, ops};
use poot_quant::weights::{ExpertProj, MoeRole, WeightId, WeightMap, WeightRole};

use super::ffn::{Glu, glu};
use super::linear::{Weight, WeightError, linear};
use super::norm::NormParams;
use super::router::{RouterBias, RouterKind, RouterParams, RouterWeights, Routes, route};
use super::standard::{ParamError, Step};
use crate::model::{FamilyKey, ModelError, ShapeReason, StepShape, TraceError};

/// How a dense expert stack is oriented in the checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// `[experts, out, K]`, the layout of a stack of `nn.Linear` weights.
    OutIn,
    /// `[experts, K, out]`, already the contraction's own layout.
    InOut,
}

/// How a checkpoint stores the gate and up projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateUpForm {
    /// Two separate projections.
    Split,
    /// One projection of `2 * inter` outputs: gate then up (granite-MoE).
    FusedHalves,
    /// One projection of `2 * inter` outputs, gate on even outputs and up on odd ones (gpt-oss).
    FusedInterleaved,
}

/// Checked MoE parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeParams {
    router: RouterParams,
    width: usize,
    inter: usize,
    glu: Glu,
    orientation: Orientation,
    gate_up: GateUpForm,
    expert_bias: bool,
    shared: Option<usize>,
    shared_gate: bool,
}

impl MoeParams {
    /// A block of `experts` experts of `inter` hidden units over `width`, routing each token to
    /// `top_k`. Refused, naming the field, for a zero extent or a `top_k` above `experts`.
    pub fn new(
        family: FamilyKey,
        width: usize,
        inter: usize,
        experts: usize,
        top_k: usize,
        kind: RouterKind,
    ) -> Result<Self, ModelError> {
        let as_config = |e: ParamError| e.for_family(family);
        for (field, value) in [("width", width), ("moe_intermediate_size", inter)] {
            if value == 0 {
                return Err(as_config(ParamError::new(
                    field,
                    crate::model::ConfigReason::Zero,
                )));
            }
        }
        Ok(Self {
            router: RouterParams::new(experts, top_k, kind).map_err(as_config)?,
            width,
            inter,
            glu: Glu::Silu,
            orientation: Orientation::OutIn,
            gate_up: GateUpForm::Split,
            expert_bias: false,
            shared: None,
            shared_gate: false,
        })
    }

    /// An always-active shared expert of `inter` hidden units (the shared experts of DeepSeek
    /// combined into one gated FFN), added to every token's routed output with no gate.
    pub fn with_shared(self, inter: usize) -> Result<Self, ParamError> {
        if inter == 0 {
            return Err(ParamError::new(
                "shared_expert_intermediate_size",
                crate::model::ConfigReason::Zero,
            ));
        }
        Ok(Self {
            shared: Some(inter),
            ..self
        })
    }

    /// The shared expert's output is multiplied by `sigmoid(gate . x)`, one scalar per token from the
    /// layer's shared-gate weight (Qwen3-Next). Refused without a shared expert.
    pub fn with_shared_gate(self) -> Result<Self, ParamError> {
        if self.shared.is_none() {
            return Err(ParamError::new(
                "shared_expert_gate",
                crate::model::ConfigReason::Unsupported,
            ));
        }
        Ok(Self {
            shared_gate: true,
            ..self
        })
    }

    /// Limit the router's selection to the experts of the `keep` best of `groups` equal groups.
    pub fn with_group_limit(self, groups: usize, keep: usize) -> Result<Self, ParamError> {
        Ok(Self {
            router: self.router.with_group_limit(groups, keep)?,
            ..self
        })
    }

    /// Multiply every selected expert's weight by `scale` (`routed_scaling_factor`).
    pub fn with_routed_scale(self, scale: f32) -> Result<Self, ParamError> {
        Ok(Self {
            router: self.router.with_scale(scale)?,
            ..self
        })
    }

    /// The expert activation (the default is SwiGLU).
    pub fn with_glu(self, glu: Glu) -> Self {
        Self { glu, ..self }
    }

    /// Dense expert stacks stored `[experts, K, out]`.
    pub fn with_orientation(self, orientation: Orientation) -> Self {
        Self {
            orientation,
            ..self
        }
    }

    /// Gate and up stored as one projection.
    pub fn with_gate_up(self, gate_up: GateUpForm) -> Self {
        Self { gate_up, ..self }
    }

    /// Route on `rmsnorm(x) * scale / sqrt(width)` (see [`RouterParams::with_scaled_input`]).
    pub fn with_scaled_router_input(self, norm: NormParams) -> Self {
        Self {
            router: self.router.with_scaled_input(norm),
            ..self
        }
    }

    /// A learned scale on each expert's gate (see [`RouterParams::with_expert_scale`]); refused
    /// for a router kind that defines no such scale.
    pub fn with_expert_scale(self) -> Result<Self, ParamError> {
        Ok(Self {
            router: self.router.with_expert_scale()?,
            ..self
        })
    }

    /// The router has a bias vector placed as `bias`; refused when the router kind defines no
    /// such placement.
    pub fn with_router_bias(self, bias: RouterBias) -> Result<Self, ParamError> {
        Ok(Self {
            router: self.router.with_bias(bias)?,
            ..self
        })
    }

    /// A bias on every expert projection.
    pub fn with_expert_bias(self) -> Self {
        Self {
            expert_bias: true,
            ..self
        }
    }

    pub fn experts(&self) -> usize {
        self.router.experts()
    }

    pub fn top_k(&self) -> usize {
        self.router.top_k()
    }

    /// Output units of the gate-and-up projection when it is one tensor.
    fn fused_out(&self) -> usize {
        2 * self.inter
    }
}

/// The largest count the routing carries exactly: the router's rank counts and slot comparisons are F32.
const MAX_EXACT_ROWS: usize = 1 << 24;

/// Refuse a step whose `tokens * top_k` routed rows, or whose expert count, exceed what the F32 rank counts
/// carry exactly.
pub fn check_step(
    family: FamilyKey,
    shape: StepShape,
    params: &MoeParams,
) -> Result<(), TraceError> {
    let rows = shape
        .rows
        .get()
        .saturating_mul(shape.tokens.get())
        .saturating_mul(params.top_k());
    if rows > MAX_EXACT_ROWS || params.experts() >= MAX_EXACT_ROWS {
        return Err(TraceError::ShapeUnsupported {
            family,
            shape,
            reason: ShapeReason::RoutedRows {
                max: MAX_EXACT_ROWS,
            },
        });
    }
    Ok(())
}

/// One projection of every expert of a layer as one stack weight `[experts, out, K]` (or `[experts, K, out]`
/// for a dense [`Orientation::InOut`] checkpoint), dense or packed, with its optional bias stack
/// `[experts, out]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertProjection {
    weight: Weight,
    orientation: Orientation,
    bias: Option<Weight>,
    out: usize,
}

impl ExpertProjection {
    /// `proj` of every expert of `layer`: the mapped [`MoeRole::Stack`] weight.
    fn new(
        map: &WeightMap,
        layer: usize,
        proj: ExpertProj,
        p: &MoeParams,
        out: usize,
        k: usize,
    ) -> Result<Self, WeightError> {
        let experts = p.experts();
        let id = |role| WeightId::layer(layer, WeightRole::Moe(role));
        let shape = match p.orientation {
            Orientation::OutIn => vec![experts, out, k],
            Orientation::InOut => vec![experts, k, out],
        };
        let weight = Weight::new(map, id(MoeRole::Stack(proj)), &shape)?;
        let bias = p
            .expert_bias
            .then(|| Weight::new(map, id(MoeRole::StackBias(proj)), &[experts, out]))
            .transpose()?;
        Ok(Self {
            weight,
            orientation: p.orientation,
            bias,
            out,
        })
    }

    /// `x[M, K]` through each row's expert (`ids[M]`): `[M, out]`.
    fn project(&self, b: &Builder, x: Traced, ids: Traced) -> Result<Traced, TraceError> {
        let w = self.weight.declare(b)?;
        let w = match self.orientation {
            Orientation::OutIn => b.transpose(w, vec![0, 2, 1])?,
            Orientation::InOut => w,
        };
        let y = b.indexed_matmul(x, w, ids)?;
        match &self.bias {
            Some(bias) => Ok(b.binary(BinOp::Add, y, expert_rows(b, bias, ids, self.out)?)?),
            None => Ok(y),
        }
    }
}

/// `bias[ids[m]]` for every row: `[M, out]`, read through the same indexed contraction as the
/// weights (a column of ones against each expert's bias as a `[1, out]` matrix).
fn expert_rows(b: &Builder, bias: &Weight, ids: Traced, out: usize) -> Result<Traced, TraceError> {
    let m = b.aval(ids)?.shape[0];
    let experts = bias.shape()[0];
    let one = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, b.iota(1)?, Scalar::F32(0.0))?,
        Scalar::F32(1.0),
    )?;
    let ones = b.broadcast(b.reshape(one, vec![1, 1])?, vec![m, 1])?;
    Ok(b.indexed_matmul(
        ones,
        b.reshape(bias.declare_f32(b)?, vec![experts, 1, out])?,
        ids,
    )?)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GateUp {
    Split {
        gate: ExpertProjection,
        up: ExpertProjection,
    },
    Fused(ExpertProjection),
}

/// The shared expert's gate and up `[inter, width]` and down `[width, inter]` projections, and its
/// scalar gate when the parameters have one.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SharedExpert {
    gate: Weight,
    up: Weight,
    down: Weight,
    /// The gate's one row over the width: `[1, width]` as `nn.Linear(width, 1)` stores it, or
    /// `[width]` as a GGUF stores it.
    scalar: Option<Weight>,
}

/// A layer's router and expert projections, and its shared expert when the parameters have one.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeWeights {
    router: RouterWeights,
    gate_up: GateUp,
    down: ExpertProjection,
    shared: Option<SharedExpert>,
}

impl MoeWeights {
    /// The router's weights, for a family that routes some layers itself.
    pub fn router(&self) -> &RouterWeights {
        &self.router
    }

    pub fn new(map: &WeightMap, layer: usize, p: &MoeParams) -> Result<Self, WeightError> {
        let (width, inter) = (p.width, p.inter);
        let projection = |proj, out, k| ExpertProjection::new(map, layer, proj, p, out, k);
        let gate_up = match p.gate_up {
            GateUpForm::Split => GateUp::Split {
                gate: projection(ExpertProj::Gate, inter, width)?,
                up: projection(ExpertProj::Up, inter, width)?,
            },
            GateUpForm::FusedHalves | GateUpForm::FusedInterleaved => {
                GateUp::Fused(projection(ExpertProj::GateUp, p.fused_out(), width)?)
            }
        };
        let shared = p
            .shared
            .map(|shared_inter| {
                let w = |proj, shape: [usize; 2]| {
                    Weight::new(
                        map,
                        WeightId::layer(layer, WeightRole::Moe(MoeRole::Shared(proj))),
                        &shape,
                    )
                };
                let scalar = p
                    .shared_gate
                    .then(|| {
                        let id = WeightId::layer(layer, WeightRole::Moe(MoeRole::SharedGate));
                        Weight::new(map, id, &[1, width])
                            .or_else(|_| Weight::new(map, id, &[width]))
                    })
                    .transpose()?;
                Ok::<_, WeightError>(SharedExpert {
                    gate: w(ExpertProj::Gate, [shared_inter, width])?,
                    up: w(ExpertProj::Up, [shared_inter, width])?,
                    down: w(ExpertProj::Down, [width, shared_inter])?,
                    scalar,
                })
            })
            .transpose()?;
        Ok(Self {
            router: RouterWeights::new(map, layer, width, &p.router)?,
            gate_up,
            down: projection(ExpertProj::Down, width, inter)?,
            shared,
        })
    }
}

/// The routed block over `x[.., width]` of `layer`: the same shape back. The router reads `route_x`
/// (the same shape as `x`) when given and `x` otherwise; the experts always read `x`. Gemma 4 routes
/// on the residual stream and feeds the experts that stream normalized by its own norm.
pub fn moe(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    route_x: Option<Traced>,
    w: &MoeWeights,
    p: &MoeParams,
) -> Result<Traced, TraceError> {
    let shape = b.aval(x)?.shape;
    if let Some(route_x) = route_x {
        debug_assert_eq!(
            b.aval(route_x)?.shape,
            shape,
            "the router and the experts read one shape"
        );
    }
    let width = p.width;
    let tokens = shape.iter().product::<usize>() / width;

    let flat = b.reshape(x, vec![tokens, width])?;
    let route_x = match route_x {
        Some(r) => b.reshape(r, vec![tokens, width])?,
        None => flat,
    };
    let routes = route(b, step, layer, route_x, &w.router, &p.router)?;
    experts_of(b, flat, shape, routes, w, p)
}

/// The experts of a routed block over `x[.., width]`, given the `routes` of its tokens (`T` rows
/// of `top_k` ids and weights): the same shape back. A family that chooses its experts itself (a
/// hash router) calls this with its own routes; [`moe`] calls it with [`route`]'s.
pub fn experts(
    b: &Builder,
    x: Traced,
    routes: Routes,
    w: &MoeWeights,
    p: &MoeParams,
) -> Result<Traced, TraceError> {
    let shape = b.aval(x)?.shape;
    let tokens = shape.iter().product::<usize>() / p.width;
    experts_of(b, b.reshape(x, vec![tokens, p.width])?, shape, routes, w, p)
}

/// [`experts`] over `x` already flattened to `[tokens, width]`, returning the original `shape`.
fn experts_of(
    b: &Builder,
    x: Traced,
    shape: Vec<usize>,
    routes: Routes,
    w: &MoeWeights,
    p: &MoeParams,
) -> Result<Traced, TraceError> {
    let (width, tokens) = (p.width, b.aval(x)?.shape[0]);
    let k = p.top_k();
    let rows = tokens * k;

    let ids = b.reshape(routes.ids, vec![rows])?;
    // Row `t * k + r` is token `t` at its slot `r`.
    let x_rows = b.reshape(
        b.broadcast(
            b.reshape(x, vec![tokens, 1, width])?,
            vec![tokens, k, width],
        )?,
        vec![rows, width],
    )?;

    let (gate, up) = match &w.gate_up {
        GateUp::Split { gate, up } => (gate.project(b, x_rows, ids)?, up.project(b, x_rows, ids)?),
        GateUp::Fused(both) => {
            let fused = both.project(b, x_rows, ids)?;
            let inter = p.inter;
            match p.gate_up {
                GateUpForm::FusedInterleaved => {
                    let pairs = b.reshape(fused, vec![rows, inter, 2])?;
                    let part = |i| {
                        Ok::<_, TraceError>(
                            b.reshape(b.slice(pairs, 2, i, i + 1)?, vec![rows, inter])?,
                        )
                    };
                    (part(0)?, part(1)?)
                }
                _ => (
                    b.slice(fused, 1, 0, inter)?,
                    b.slice(fused, 1, inter, 2 * inter)?,
                ),
            }
        }
    };
    let hidden = glu(b, gate, up, &p.glu)?;
    let out = w.down.project(b, hidden, ids)?;

    // Weight each row, then sum each token's `k` rows (the executors reduce the last axis only).
    let weighted = b.binary(BinOp::Mul, out, b.reshape(routes.weights, vec![rows, 1])?)?;
    let per_token = b.transpose(b.reshape(weighted, vec![tokens, k, width])?, vec![0, 2, 1])?;
    let y = b.reshape(b.reduce(RedOp::Sum, per_token, 2, false)?, shape.clone())?;
    match &w.shared {
        Some(shared) => {
            let gate = linear(b, x, &shared.gate, None)?;
            let up = linear(b, x, &shared.up, None)?;
            let out = linear(b, glu(b, gate, up, &p.glu)?, &shared.down, None)?;
            let out = match &shared.scalar {
                Some(scalar) => {
                    let row = b.reshape(scalar.declare_f32(b)?, vec![1, width])?;
                    let logit = b.matmul(x, b.transpose(row, vec![1, 0])?)?; // [tokens, 1]
                    b.binary(BinOp::Mul, out, ops::sigmoid(b, logit)?)?
                }
                None => out,
            };
            Ok(b.binary(BinOp::Add, y, b.reshape(out, shape)?)?)
        }
        None => Ok(y),
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Slot, TensorType};
    use poot_quant::format::WeightFormat;

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const FAMILY: FamilyKey = FamilyKey::new("test");
    const TOKENS: usize = 3;
    const WIDTH: usize = 32;
    const INTER: usize = 32;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;

    fn id(role: MoeRole) -> WeightId {
        WeightId::layer(0, WeightRole::Moe(role))
    }

    fn params() -> MoeParams {
        MoeParams::new(
            FAMILY,
            WIDTH,
            INTER,
            EXPERTS,
            TOP_K,
            RouterKind::SoftmaxTopK { renormalize: true },
        )
        .unwrap()
    }

    /// How the fixture stores the expert projections.
    #[derive(Clone, Copy, Debug)]
    enum Storage {
        /// One stored `[experts, out, K]` stack per projection.
        Stack3,
        /// One stored `[experts, K, out]` stack per projection.
        StackInOut,
        /// One packed Q8_0 `[out, K]` weight per expert.
        Packed,
    }

    const PROJECTIONS: [(ExpertProj, usize, usize); 3] = [
        (ExpertProj::Gate, INTER, WIDTH),
        (ExpertProj::Up, INTER, WIDTH),
        (ExpertProj::Down, WIDTH, INTER),
    ];

    fn fixture(storage: Storage) -> Fx {
        let mut fx = Fx::new(21);
        fx.f32(id(MoeRole::Router), &[EXPERTS, WIDTH]);
        for (proj, out, k) in PROJECTIONS {
            match storage {
                Storage::Stack3 => fx.bf16(id(MoeRole::Stack(proj)), &[EXPERTS, out, k]),
                Storage::StackInOut => fx.bf16(id(MoeRole::Stack(proj)), &[EXPERTS, k, out]),
                Storage::Packed => fx.packed_stack(
                    id(MoeRole::Stack(proj)),
                    (0..EXPERTS)
                        .map(|e| {
                            poot_quant::test_support::packed::random_payload(
                                WeightFormat::Q8_0,
                                [out, k],
                                100 * (proj as u64 + 1) + e as u64,
                            )
                        })
                        .collect(),
                ),
            }
        }
        fx
    }

    fn orientation(storage: Storage) -> Orientation {
        match storage {
            Storage::StackInOut => Orientation::InOut,
            _ => Orientation::OutIn,
        }
    }

    /// The block's output for `x[TOKENS, WIDTH]` as `[rows=1, tokens, WIDTH]`.
    fn run(fx: &Fx, storage: Storage, tokens: usize, x: &[f32]) -> Vec<f32> {
        let p = params().with_orientation(orientation(storage));
        let map = fx.map();
        let w = MoeWeights::new(&map, 0, &p).unwrap();
        let b = Builder::for_tests();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All)).unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![1, tokens, WIDTH]))
            .unwrap();
        let y = moe(&b, &step, 0, xs, None, &w, &p).unwrap();
        let g = b.finish(y).unwrap();
        let ids = vec![0; tokens];
        let pos: Vec<i32> = (0..tokens as i32).collect();
        let formats = poot_graph_plan::WeightFormats::from_weight_map(&map);
        let g = poot_graph_plan::bind_packed_weights(&g, &formats).unwrap();
        fx.eval(
            &g,
            &[("x", x)],
            &[(Slot::Token, &ids), (Slot::Pos, &pos)],
            &[],
        )
        .0
    }

    /// The block in f64: each token to its `TOP_K` best experts, each expert `down(silu(gate x) *
    /// up x)`, the outputs weighted by the softmax over the selected logits.
    fn reference(fx: &Fx, storage: Storage, x: &[f32]) -> Vec<f64> {
        let router = fx.values(id(MoeRole::Router));
        // `weight(proj)[e][o][k]` in the logical `[out, K]` orientation.
        let expert = |proj: ExpertProj, out: usize, k: usize| -> Vec<Vec<f32>> {
            (0..EXPERTS)
                .map(|e| match storage {
                    Storage::Packed | Storage::Stack3 => {
                        fx.values(id(MoeRole::Stack(proj)))[e * out * k..(e + 1) * out * k].to_vec()
                    }
                    Storage::StackInOut => {
                        let stack = fx.values(id(MoeRole::Stack(proj)));
                        (0..out * k)
                            .map(|i| stack[e * out * k + (i % k) * out + i / k])
                            .collect()
                    }
                })
                .collect()
        };
        let [gate, up, down] = PROJECTIONS.map(|(proj, out, k)| expert(proj, out, k));
        let matvec = |w: &[f32], v: &[f64], out: usize, k: usize| -> Vec<f64> {
            (0..out)
                .map(|o| (0..k).map(|j| v[j] * f64::from(w[o * k + j])).sum())
                .collect()
        };
        let mut want = Vec::new();
        for token in x.chunks(WIDTH) {
            let v: Vec<f64> = token.iter().map(|&t| t.into()).collect();
            let logits = matvec(&router, &v, EXPERTS, WIDTH);
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let selected = &order[..TOP_K];
            let max = logits[selected[0]];
            let exp: Vec<f64> = selected.iter().map(|&e| (logits[e] - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let mut y = vec![0.0; WIDTH];
            for (&e, weight) in selected.iter().zip(exp.iter().map(|v| v / sum)) {
                let g = matvec(&gate[e], &v, INTER, WIDTH);
                let u = matvec(&up[e], &v, INTER, WIDTH);
                let hidden: Vec<f64> = g
                    .iter()
                    .zip(&u)
                    .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
                    .collect();
                for (acc, o) in y.iter_mut().zip(matvec(&down[e], &hidden, WIDTH, INTER)) {
                    *acc += weight * o;
                }
            }
            want.extend(y);
        }
        want
    }

    fn input() -> Vec<f32> {
        oracle::random(TOKENS * WIDTH, 9)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect()
    }

    /// SC-002: the block over each expert storage (a stored
    /// `[E, out, K]` stack, a stored `[E, K, out]` stack, and a stack of per-expert packed weights) matches an
    /// independent f64 reference, for a prefill-shaped step and a
    /// decode-shaped one.
    #[test]
    fn the_block_matches_an_f64_reference_over_every_expert_storage() {
        for storage in [Storage::Stack3, Storage::StackInOut, Storage::Packed] {
            let fx = fixture(storage);
            let x = input();
            oracle::assert_matches_f64(
                &run(&fx, storage, TOKENS, &x),
                &reference(&fx, storage, &x),
                1e-4,
            );
            for t in 0..TOKENS {
                let one = &x[t * WIDTH..(t + 1) * WIDTH];
                let want = reference(&fx, storage, one);
                oracle::assert_matches_f64(&run(&fx, storage, 1, one), &want, 1e-4);
            }
        }
    }

    /// Each weight matters: perturbing the router or any projection of any expert changes the
    /// output (an expert the fixture never selects would hide a dropped read, so every expert is
    /// perturbed and the union of the effects is required).
    #[test]
    fn every_weight_changes_the_output() {
        for storage in [Storage::Stack3, Storage::Packed] {
            let fx = fixture(storage);
            let x = input();
            let base = run(&fx, storage, TOKENS, &x);
            let mut ids = vec![id(MoeRole::Router)];
            ids.extend(PROJECTIONS.map(|(proj, ..)| id(MoeRole::Stack(proj))));
            for weight in ids {
                if matches!(storage, Storage::Packed) && weight != id(MoeRole::Router) {
                    continue; // a packed weight cannot be perturbed in place; see the packed row below
                }
                let mut perturbed = fx.clone();
                perturbed.perturb(weight);
                oracle::assert_differs(&run(&perturbed, storage, TOKENS, &x), &base, weight);
            }
        }
    }

    /// The packed chain reads each expert's own weight: replacing one expert's payload changes the
    /// output iff some token selects that expert, and the union over experts covers every
    /// projection.
    #[test]
    fn every_packed_expert_is_read() {
        let fx = fixture(Storage::Packed);
        let x = input();
        let base = run(&fx, Storage::Packed, TOKENS, &x);
        for (proj, out, k) in PROJECTIONS {
            let mut any = false;
            for e in 0..EXPERTS {
                let mut changed = fx.clone();
                changed.packed_stack(
                    id(MoeRole::Stack(proj)),
                    (0..EXPERTS)
                        .map(|other| {
                            // Expert `e` replaced, every other expert as it was.
                            let seed = if other == e {
                                9_000 + e as u64
                            } else {
                                100 * (proj as u64 + 1) + other as u64
                            };
                            poot_quant::test_support::packed::random_payload(
                                WeightFormat::Q8_0,
                                [out, k],
                                seed,
                            )
                        })
                        .collect(),
                );
                any |= run(&changed, Storage::Packed, TOKENS, &x)
                    .iter()
                    .zip(&base)
                    .any(|(a, b)| (a - b).abs() > 1e-6);
            }
            assert!(any, "no expert's {proj:?} weight reaches the output");
        }
    }

    /// `MoeParams::new` refuses `top_k > experts` and a zero extent as `ModelError::Config`.
    #[test]
    fn params_refuse_an_impossible_block() {
        let kind = RouterKind::TopKSoftmax;
        let config_error = |result: Result<MoeParams, ModelError>| match result {
            Err(ModelError::Config { field, reason, .. }) => (field, reason),
            other => panic!("expected a config error, got {other:?}"),
        };
        assert_eq!(
            config_error(MoeParams::new(FAMILY, WIDTH, INTER, 4, 5, kind)),
            ("top_k", crate::model::ConfigReason::Exceeds { max: 4 })
        );
        assert_eq!(
            config_error(MoeParams::new(FAMILY, WIDTH, 0, 4, 2, kind)),
            ("moe_intermediate_size", crate::model::ConfigReason::Zero)
        );
        assert!(MoeParams::new(FAMILY, WIDTH, INTER, 4, 4, kind).is_ok());
    }
}

#[cfg(test)]
mod layout_tests {
    //! The routed block against an independent f64 forward pass over every storage and router
    //! option a family uses: split and fused (halves, interleaved) gate and up, `[E, out, K]` and
    //! `[E, K, out]` stacks expert and
    //! router biases, the clamped GLU, and both router arms.

    use std::num::NonZeroUsize;

    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::model::{ConfigReason, KvLayout, LogitRows};

    const FAMILY: FamilyKey = FamilyKey::new("moe-test");
    const WIDTH: usize = 6;
    const INTER: usize = 5;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const TOKENS: usize = 3;
    const LAYER: usize = 1;

    #[derive(Clone, Copy, Debug)]
    struct Case {
        orientation: Orientation,
        form: GateUpForm,
        kind: RouterKind,
        glu: Glu,
        router_bias: bool,
        expert_bias: bool,
        /// Gemma 4's router: it reads its own input, scaled and normalized, and each expert's
        /// gate has a learned scale.
        scaled_router: bool,
        /// An always-active shared expert, with (`shared_gate`) or without a sigmoid scalar gate
        /// (Qwen3-Next; DeepSeek).
        shared: bool,
        shared_gate: bool,
    }

    const SHARED_INTER: usize = 4;

    const BASE: Case = Case {
        orientation: Orientation::OutIn,
        form: GateUpForm::Split,
        kind: RouterKind::SoftmaxTopK { renormalize: true },
        glu: Glu::Silu,
        router_bias: false,
        expert_bias: false,
        scaled_router: false,
        shared: false,
        shared_gate: false,
    };

    fn params(c: Case) -> MoeParams {
        let mut p = MoeParams::new(FAMILY, WIDTH, INTER, EXPERTS, TOP_K, c.kind)
            .unwrap()
            .with_glu(c.glu)
            .with_orientation(c.orientation)
            .with_gate_up(c.form);
        if c.router_bias {
            p = p.with_router_bias(RouterBias::AddedToLogits).unwrap();
        }
        if c.expert_bias {
            p = p.with_expert_bias();
        }
        if c.scaled_router {
            p = p
                .with_scaled_router_input(NormParams::new(1e-6).unwrap())
                .with_expert_scale()
                .unwrap();
        }
        if c.shared {
            p = p.with_shared(SHARED_INTER).unwrap();
        }
        if c.shared_gate {
            p = p.with_shared_gate().unwrap();
        }
        p
    }

    fn id(role: MoeRole) -> WeightId {
        WeightId::layer(LAYER, WeightRole::Moe(role))
    }

    /// The stored weights of a case: `(role, shape)` in the case's layout.
    fn roles(c: Case) -> Vec<(MoeRole, Vec<usize>)> {
        let stack = |out: usize, k: usize| match c.orientation {
            Orientation::OutIn => vec![EXPERTS, out, k],
            Orientation::InOut => vec![EXPERTS, k, out],
        };
        let mut r = vec![(MoeRole::Router, vec![EXPERTS, WIDTH])];
        if c.router_bias {
            r.push((MoeRole::RouterBias, vec![EXPERTS]));
        }
        if c.scaled_router {
            r.push((MoeRole::RouterScale, vec![WIDTH]));
            r.push((MoeRole::ExpertScale, vec![EXPERTS]));
        }
        let projections: Vec<(ExpertProj, usize, usize)> = match c.form {
            GateUpForm::Split => vec![
                (ExpertProj::Gate, INTER, WIDTH),
                (ExpertProj::Up, INTER, WIDTH),
            ],
            _ => vec![(ExpertProj::GateUp, 2 * INTER, WIDTH)],
        }
        .into_iter()
        .chain([(ExpertProj::Down, WIDTH, INTER)])
        .collect();
        for (proj, out, k) in projections {
            r.push((MoeRole::Stack(proj), stack(out, k)));
            if c.expert_bias {
                r.push((MoeRole::StackBias(proj), vec![EXPERTS, out]));
            }
        }
        if c.shared {
            r.push((MoeRole::Shared(ExpertProj::Gate), vec![SHARED_INTER, WIDTH]));
            r.push((MoeRole::Shared(ExpertProj::Up), vec![SHARED_INTER, WIDTH]));
            r.push((MoeRole::Shared(ExpertProj::Down), vec![WIDTH, SHARED_INTER]));
        }
        if c.shared_gate {
            r.push((MoeRole::SharedGate, vec![1, WIDTH]));
        }
        r
    }

    fn fixture(c: Case) -> Fx {
        let mut fx = Fx::new(5);
        for (i, (role, shape)) in roles(c).into_iter().enumerate() {
            if i % 2 == 0 {
                fx.f32(id(role), &shape);
            } else {
                fx.bf16(id(role), &shape);
            }
        }
        fx
    }

    fn inputs() -> Vec<f32> {
        oracle::random(TOKENS * WIDTH, 77)
            .into_iter()
            .map(|v| 6.0 * v)
            .collect()
    }

    /// What the router reads when it reads its own input (a case with a scaled router).
    fn route_inputs() -> Vec<f32> {
        oracle::random(TOKENS * WIDTH, 91)
            .into_iter()
            .map(|v| 4.0 * v)
            .collect()
    }

    fn run(fx: &Fx, c: Case, x: &[f32]) -> Vec<f32> {
        let p = params(c);
        let w = MoeWeights::new(&fx.map(), LAYER, &p).unwrap();
        let b = Builder::for_tests();
        let step = Step::new(
            &b,
            StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(TOKENS).unwrap(),
                capacity: NonZeroUsize::new(8).unwrap(),
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
                hidden: None,
            },
        )
        .unwrap();
        let xs = b
            .constant("x", TensorType::f32(vec![1, TOKENS, WIDTH]))
            .unwrap();
        let routes = route_inputs();
        let (y, named): (_, Vec<(&str, &[f32])>) = if c.scaled_router {
            let rs = b
                .constant("r", TensorType::f32(vec![1, TOKENS, WIDTH]))
                .unwrap();
            (
                moe(&b, &step, LAYER, xs, Some(rs), &w, &p).unwrap(),
                vec![("x", x), ("r", &routes)],
            )
        } else {
            (
                moe(&b, &step, LAYER, xs, None, &w, &p).unwrap(),
                vec![("x", x)],
            )
        };
        let g = step.finish(b, y).unwrap();
        let zeros = vec![0; TOKENS];
        fx.eval(
            &g,
            &named,
            &[(Slot::Token, &zeros), (Slot::Pos, &zeros)],
            &[],
        )
        .0
    }

    /// The f64 reference: per token, the router in f64, the selected experts only, each expert's
    /// projections read through the case's own stored layout.
    fn reference(fx: &Fx, c: Case, x: &[f32]) -> Vec<f64> {
        let vals = |role| -> Vec<f64> {
            fx.values(id(role))
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        };
        let router = vals(MoeRole::Router);
        let router_bias = c.router_bias.then(|| vals(MoeRole::RouterBias));
        // Element `(expert e, output o, input i)` of a stack with `out` outputs and `k` inputs.
        let at = |data: &[f64], e: usize, o: usize, i: usize, out: usize, k: usize| match c
            .orientation
        {
            Orientation::OutIn => data[(e * out + o) * k + i],
            Orientation::InOut => data[(e * k + i) * out + o],
        };
        let project = |proj: ExpertProj, e: usize, v: &[f64], out: usize, k: usize| -> Vec<f64> {
            let data = vals(MoeRole::Stack(proj));
            let bias = c.expert_bias.then(|| vals(MoeRole::StackBias(proj)));
            (0..out)
                .map(|o| {
                    bias.as_ref().map_or(0.0, |b| b[e * out + o])
                        + (0..k)
                            .map(|i| v[i] * at(&data, e, o, i, out, k))
                            .sum::<f64>()
                })
                .collect()
        };
        let gelu_tanh = |g: f64| {
            0.5 * g * (1.0 + (0.797_884_560_802_865_4 * (g + 0.044_715 * g * g * g)).tanh())
        };
        let activation = |g: f64, u: f64| match c.glu {
            Glu::Clamped { alpha, limit } => {
                let (alpha, limit) = (f64::from(alpha), f64::from(limit));
                let g = g.min(limit);
                (u.clamp(-limit, limit) + 1.0) * g / (1.0 + (-alpha * g).exp())
            }
            Glu::ClampedSilu { limit } => {
                let g = g.min(f64::from(limit));
                g / (1.0 + (-g).exp()) * u.clamp(-f64::from(limit), f64::from(limit))
            }
            Glu::Silu => g / (1.0 + (-g).exp()) * u,
            Glu::Gelu => gelu_tanh(g) * u,
        };
        let routes = route_inputs();
        let (router_scale, expert_scale) = if c.scaled_router {
            (vals(MoeRole::RouterScale), vals(MoeRole::ExpertScale))
        } else {
            (Vec::new(), Vec::new())
        };
        let mut out = Vec::new();
        for t in 0..TOKENS {
            let v: Vec<f64> = x[t * WIDTH..(t + 1) * WIDTH]
                .iter()
                .map(|&v| v.into())
                .collect();
            // The router's input: the experts' own, or the scaled and normalized other one.
            let routed: Vec<f64> = if c.scaled_router {
                let r: Vec<f64> = routes[t * WIDTH..(t + 1) * WIDTH]
                    .iter()
                    .map(|&v| v.into())
                    .collect();
                let inv = 1.0 / (r.iter().map(|v| v * v).sum::<f64>() / WIDTH as f64 + 1e-6).sqrt();
                (0..WIDTH)
                    .map(|i| r[i] * inv * router_scale[i] / (WIDTH as f64).sqrt())
                    .collect()
            } else {
                v.clone()
            };
            let logits: Vec<f64> = (0..EXPERTS)
                .map(|e| {
                    router_bias.as_ref().map_or(0.0, |b| b[e])
                        + (0..WIDTH)
                            .map(|i| routed[i] * router[e * WIDTH + i])
                            .sum::<f64>()
                })
                .collect();
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let chosen = &order[..TOP_K];
            let top = logits[chosen[0]];
            let full: f64 = logits.iter().map(|l| (l - top).exp()).sum();
            let selected: f64 = chosen.iter().map(|&e| (logits[e] - top).exp()).sum();
            let denominator = match c.kind {
                RouterKind::SoftmaxTopK { renormalize: false } => full,
                _ => selected,
            };
            let mut y = vec![0.0; WIDTH];
            for &e in chosen {
                let weight = (logits[e] - top).exp() / denominator
                    * if c.scaled_router {
                        expert_scale[e]
                    } else {
                        1.0
                    };
                let (gate, up): (Vec<f64>, Vec<f64>) = match c.form {
                    GateUpForm::Split => (
                        project(ExpertProj::Gate, e, &v, INTER, WIDTH),
                        project(ExpertProj::Up, e, &v, INTER, WIDTH),
                    ),
                    GateUpForm::FusedHalves => {
                        let both = project(ExpertProj::GateUp, e, &v, 2 * INTER, WIDTH);
                        (both[..INTER].to_vec(), both[INTER..].to_vec())
                    }
                    GateUpForm::FusedInterleaved => {
                        let both = project(ExpertProj::GateUp, e, &v, 2 * INTER, WIDTH);
                        (
                            (0..INTER).map(|i| both[2 * i]).collect(),
                            (0..INTER).map(|i| both[2 * i + 1]).collect(),
                        )
                    }
                };
                let hidden: Vec<f64> = gate
                    .iter()
                    .zip(&up)
                    .map(|(&g, &u)| activation(g, u))
                    .collect();
                let down = project(ExpertProj::Down, e, &hidden, WIDTH, INTER);
                for (acc, d) in y.iter_mut().zip(down) {
                    *acc += weight * d;
                }
            }
            if c.shared {
                let shared = |role, rows: usize, k: usize, v: &[f64]| -> Vec<f64> {
                    let w = vals(role);
                    (0..rows)
                        .map(|o| (0..k).map(|i| v[i] * w[o * k + i]).sum::<f64>())
                        .collect()
                };
                let g = shared(MoeRole::Shared(ExpertProj::Gate), SHARED_INTER, WIDTH, &v);
                let u = shared(MoeRole::Shared(ExpertProj::Up), SHARED_INTER, WIDTH, &v);
                let hidden: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| activation(g, u)).collect();
                let down = shared(
                    MoeRole::Shared(ExpertProj::Down),
                    WIDTH,
                    SHARED_INTER,
                    &hidden,
                );
                let scale = if c.shared_gate {
                    let logit: f64 = shared(MoeRole::SharedGate, 1, WIDTH, &v)[0];
                    1.0 / (1.0 + (-logit).exp())
                } else {
                    1.0
                };
                for (acc, d) in y.iter_mut().zip(down) {
                    *acc += scale * d;
                }
            }
            out.extend(y);
        }
        out
    }

    /// The case matches the f64 reference and every stored weight matters.
    fn assert_case(c: Case) {
        let fx = fixture(c);
        let x = inputs();
        let got = run(&fx, c, &x);
        oracle::assert_matches_f64(&got, &reference(&fx, c, &x), 1e-5);
        for (role, _) in roles(c) {
            let mut p = fx.clone();
            p.perturb(id(role));
            oracle::assert_differs(&run(&p, c, &x), &got, id(role));
        }
    }

    /// SC-002: split gate/up over `[E, out, K]` (the HF and GGUF dense forms), with the
    /// renormalizing and the non-renormalizing softmax router. Mutation: renormalize in `softmax_top_k` when `renormalize` is false; the second
    /// row goes red.
    #[test]
    fn split_stacks_and_both_softmax_routers_match_the_reference() {
        assert_case(BASE);
        assert_case(Case {
            kind: RouterKind::SoftmaxTopK { renormalize: false },
            ..BASE
        });
    }

    /// SC-002: the top-k-then-softmax router, with a fused gate-and-up of two halves
    /// (granite-MoE's `input_linear`).
    #[test]
    fn fused_halves_and_the_top_k_softmax_router_match_the_reference() {
        assert_case(Case {
            form: GateUpForm::FusedHalves,
            kind: RouterKind::TopKSoftmax,
            ..BASE
        });
    }

    /// SC-002: gpt-oss's block: an interleaved fused gate-and-up in `[E, K, out]`, biases on the
    /// router and on every expert projection, the clamped GLU (inputs reach past its limit).
    /// Mutation: add the expert bias to the wrong expert's row (`ids` shifted); red.
    #[test]
    fn interleaved_in_out_biased_experts_with_a_clamped_glu_match_the_reference() {
        assert_case(Case {
            orientation: Orientation::InOut,
            form: GateUpForm::FusedInterleaved,
            kind: RouterKind::TopKSoftmax,
            glu: Glu::clamped(1.702, 0.8).unwrap(),
            router_bias: true,
            expert_bias: true,
            scaled_router: false,
            shared: false,
            shared_gate: false,
        });
        // The same biased experts with separate gate and up, `[E, out, K]`.
        assert_case(Case {
            glu: Glu::clamped(1.702, 0.8).unwrap(),
            expert_bias: true,
            ..BASE
        });
    }

    /// Gemma 4's block: the router reads its own input (not the experts'), normalized with no
    /// learned scale then scaled by the router scale and `1 / sqrt(width)`, each expert's gate is
    /// scaled after the gates are normalized, and the experts' activation is GeGLU, over a fused
    /// gate-and-up of two halves. Mutation: take the router's input from the experts' input
    /// (pass `None` for the router input of `moe`); red.
    #[test]
    fn a_routed_input_scaled_gates_and_a_geglu_match_the_reference() {
        assert_case(Case {
            form: GateUpForm::FusedHalves,
            glu: Glu::Gelu,
            scaled_router: true,
            ..BASE
        });
    }

    /// The shared expert (DeepSeek's, ungated) and Qwen3-Next's (a sigmoid scalar gate per token
    /// over the shared output) match the reference; the gate weight is among the perturbed ones.
    /// Mutation: apply the gate to the routed output instead of the shared one; red.
    #[test]
    fn a_shared_expert_with_and_without_a_sigmoid_gate_matches_the_reference() {
        assert_case(Case {
            shared: true,
            ..BASE
        });
        assert_case(Case {
            shared: true,
            shared_gate: true,
            ..BASE
        });
    }

    /// The gate is stored `[1, width]` (an `nn.Linear(width, 1)`) or `[width]` (a GGUF vector), and a
    /// gate needs a shared expert to gate.
    #[test]
    fn the_shared_gate_reads_either_stored_shape_and_needs_a_shared_expert() {
        let c = Case {
            shared: true,
            shared_gate: true,
            ..BASE
        };
        let fx = fixture(c);
        let x = inputs();
        let want = run(&fx, c, &x);
        let mut flat = fx.clone();
        flat.set(
            id(MoeRole::SharedGate),
            &[WIDTH],
            fx.values(id(MoeRole::SharedGate)),
        );
        assert_eq!(run(&flat, c, &x), want);
        let mut wrong = fx.clone();
        wrong.set(id(MoeRole::SharedGate), &[2, WIDTH], vec![0.5; 2 * WIDTH]);
        assert!(MoeWeights::new(&wrong.map(), LAYER, &params(c)).is_err());
        assert_eq!(
            MoeParams::new(FAMILY, WIDTH, INTER, EXPERTS, TOP_K, BASE.kind)
                .unwrap()
                .with_shared_gate(),
            Err(ParamError::new(
                "shared_expert_gate",
                ConfigReason::Unsupported
            ))
        );
    }

    /// `MoeParams::new` refuses each extent no block can use, naming the field.
    #[test]
    fn params_refuse_a_top_k_above_the_experts_and_zero_extents() {
        let kind = RouterKind::TopKSoftmax;
        let cases = [
            (
                MoeParams::new(FAMILY, WIDTH, INTER, 4, 5, kind),
                "top_k",
                ConfigReason::Exceeds { max: 4 },
            ),
            (
                MoeParams::new(FAMILY, WIDTH, INTER, 4, 0, kind),
                "top_k",
                ConfigReason::Zero,
            ),
            (
                MoeParams::new(FAMILY, WIDTH, INTER, 0, 1, kind),
                "experts",
                ConfigReason::Zero,
            ),
            (
                MoeParams::new(FAMILY, 0, INTER, 4, 1, kind),
                "width",
                ConfigReason::Zero,
            ),
            (
                MoeParams::new(FAMILY, WIDTH, 0, 4, 1, kind),
                "moe_intermediate_size",
                ConfigReason::Zero,
            ),
        ];
        for (got, field, reason) in cases {
            match got {
                Err(ModelError::Config {
                    family,
                    field: got_field,
                    reason: got_reason,
                }) => assert_eq!((family, got_field, got_reason), (FAMILY, field, reason)),
                other => panic!("{field}: expected a config error, got {other:?}"),
            }
        }
    }
}
