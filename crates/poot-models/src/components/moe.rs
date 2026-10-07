//! The routed mixture-of-experts feed-forward block: route every token ([`super::router`]), run each
//! selected expert's gated FFN over the rows routed to it and sum the weighted outputs back per
//! token. One body for every family (spec 999): families differ in their router, their expert
//! storage and their GLU, all of them parameters here.
//!
//! The `T` tokens of a step and their `top_k` slots flatten to `M = T * top_k` rows, row
//! `t * top_k + r` being token `t` at slot `r`, so every projection is one indexed contraction over
//! all rows and no expert is evaluated for a token that did not select it.
//!
//! Experts bind as stored (ADR-0103). A dense checkpoint holds one projection of every expert as a
//! stack, read in place by [`poot_graph_ir::Builder::indexed_matmul`] (a `[experts, out, K]` stack is
//! read through a transposed view that `compile` folds, never a copy). A packed checkpoint holds one
//! `[out, K]` weight per expert, and the packed expert chain reads those
//! ([`poot_graph_ir::ops::packed_indexed_linear`] for one row, `packed_grouped_linear` for more).

use poot_graph_ir::ops::{self, PackedLinearGraphRow};
use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced};
use poot_quant::weights::{
    ExpertProj, HandleFormat, MoeRole, WeightHandle, WeightId, WeightMap, WeightRole,
};

use super::ffn::{Glu, glu};
use super::linear::{Weight, WeightError, linear};
use super::norm::NormParams;
use super::router::{RouterBias, RouterKind, RouterParams, RouterWeights, route};
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

/// The largest row count the packed grouped chain orders exactly: its expert ids and row
/// indices are carried in F32.
const MAX_EXACT_ROWS: usize = 1 << 24;

/// Refuse a step whose `tokens * top_k` routed rows exceed what the expert chain indexes exactly.
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

/// A dense expert stack: one weight holding the projection of every expert.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DenseStack {
    weight: Weight,
    orientation: Orientation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Table {
    Dense(DenseStack),
    /// One packed `[out, K]` weight per expert, in expert order.
    Packed(Vec<PackedLinearGraphRow>),
}

/// One projection of every expert of a layer, dense or packed, with its optional bias stack
/// `[experts, out]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertProjection {
    table: Table,
    bias: Option<Weight>,
    out: usize,
}

impl ExpertProjection {
    /// `proj` of every expert of `layer`: the mapped [`MoeRole::Stack`] when the checkpoint holds the
    /// experts dense, else the [`MoeRole::Expert`] weight of each expert (all packed alike).
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
        let stack = id(MoeRole::Stack(proj));
        let table = if map.handle(stack).is_some() {
            // A stack is the stored `[experts, ..]` tensor, or the per-expert tensors stacked into
            // `[experts, out, K]`: always one 3-D weight, read in place.
            let shape = match p.orientation {
                Orientation::OutIn => vec![experts, out, k],
                Orientation::InOut => vec![experts, k, out],
            };
            Table::Dense(DenseStack {
                weight: Weight::new(map, stack, &shape)?,
                orientation: p.orientation,
            })
        } else {
            let mut rows: Vec<PackedLinearGraphRow> = Vec::with_capacity(experts);
            for expert in 0..experts {
                let id = id(MoeRole::Expert(proj, expert as u32));
                Weight::new(map, id, &[out, k]).map_err(|e| match e {
                    // No expert stack and no first expert: the stack is what is missing.
                    WeightError::Unmapped { .. } if expert == 0 => {
                        WeightError::Unmapped { id: stack }
                    }
                    other => other,
                })?;
                let WeightHandle {
                    format: HandleFormat::Packed(descriptor),
                    ..
                } = map.handle(id).expect("checked by Weight::new").clone()
                else {
                    return Err(WeightError::Format {
                        id,
                        format: map.handle(id).expect("checked by Weight::new").format,
                    });
                };
                if rows.first().is_some_and(|r| r.descriptor != descriptor) {
                    return Err(WeightError::Format {
                        id,
                        format: HandleFormat::Packed(descriptor),
                    });
                }
                rows.push(PackedLinearGraphRow {
                    ordinal: expert,
                    linear_id: id.const_name(),
                    descriptor,
                });
            }
            Table::Packed(rows)
        };
        let bias = p
            .expert_bias
            .then(|| Weight::new(map, id(MoeRole::StackBias(proj)), &[experts, out]))
            .transpose()?;
        Ok(Self { table, bias, out })
    }

    /// `x[M, K]` through each row's expert (`ids[M]`): `[M, out]`.
    fn project(&self, b: &Builder, x: Traced, ids: Traced, grouped: bool) -> Traced {
        let y = match &self.table {
            Table::Dense(stack) => {
                let w = stack.weight.declare(b);
                let w = match stack.orientation {
                    Orientation::OutIn => b.transpose(w, vec![0, 2, 1]),
                    Orientation::InOut => w,
                };
                b.indexed_matmul(x, w, ids)
            }
            Table::Packed(rows) => {
                let packed = if grouped {
                    ops::packed_grouped_linear(b, x, ids, rows)
                } else {
                    ops::packed_indexed_linear(b, x, ids, rows)
                };
                packed.expect(
                    "expert rows are checked at construction and the row count by check_step",
                )
            }
        };
        match &self.bias {
            Some(bias) => b.binary(BinOp::Add, y, expert_rows(b, bias, ids, self.out)),
            None => y,
        }
    }
}

/// `bias[ids[m]]` for every row: `[M, out]`, read through the same indexed contraction as the
/// weights (a column of ones against each expert's bias as a `[1, out]` matrix).
fn expert_rows(b: &Builder, bias: &Weight, ids: Traced, out: usize) -> Traced {
    let m = b.aval(ids).shape[0];
    let experts = bias.shape()[0];
    let ones = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, b.reshape(ids, vec![m, 1]), Scalar::F32(0.0)),
        Scalar::F32(1.0),
    );
    b.indexed_matmul(
        ones,
        b.reshape(bias.declare_f32(b), vec![experts, 1, out]),
        ids,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum GateUp {
    Split {
        gate: ExpertProjection,
        up: ExpertProjection,
    },
    Fused(ExpertProjection),
}

/// The shared expert's gate and up `[inter, width]` and down `[width, inter]` projections.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SharedExpert {
    gate: Weight,
    up: Weight,
    down: Weight,
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
                Ok(SharedExpert {
                    gate: w(ExpertProj::Gate, [shared_inter, width])?,
                    up: w(ExpertProj::Up, [shared_inter, width])?,
                    down: w(ExpertProj::Down, [width, shared_inter])?,
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
) -> Traced {
    let shape = b.aval(x).shape;
    debug_assert!(
        route_x.is_none_or(|r| b.aval(r).shape == shape),
        "the router and the experts read one shape"
    );
    let width = p.width;
    let tokens = shape.iter().product::<usize>() / width;
    let k = p.top_k();
    let rows = tokens * k;
    let grouped = tokens > 1;

    let x = b.reshape(x, vec![tokens, width]);
    let route_x = route_x.map_or(x, |r| b.reshape(r, vec![tokens, width]));
    let routes = route(b, step, layer, route_x, &w.router, &p.router);
    let ids = b.reshape(routes.ids, vec![rows]);
    // Row `t * k + r` is token `t` at its slot `r`.
    let x_rows = b.reshape(
        b.broadcast(b.reshape(x, vec![tokens, 1, width]), vec![tokens, k, width]),
        vec![rows, width],
    );

    let (gate, up) = match &w.gate_up {
        GateUp::Split { gate, up } => (
            gate.project(b, x_rows, ids, grouped),
            up.project(b, x_rows, ids, grouped),
        ),
        GateUp::Fused(both) => {
            let fused = both.project(b, x_rows, ids, grouped);
            let inter = p.inter;
            match p.gate_up {
                GateUpForm::FusedInterleaved => {
                    let pairs = b.reshape(fused, vec![rows, inter, 2]);
                    let part = |i| b.reshape(b.slice(pairs, 2, i, i + 1), vec![rows, inter]);
                    (part(0), part(1))
                }
                _ => (
                    b.slice(fused, 1, 0, inter),
                    b.slice(fused, 1, inter, 2 * inter),
                ),
            }
        }
    };
    let hidden = glu(b, gate, up, &p.glu);
    let out = w.down.project(b, hidden, ids, grouped);

    // Weight each row, then sum each token's `k` rows (the executors reduce the last axis only).
    let weighted = b.binary(BinOp::Mul, out, b.reshape(routes.weights, vec![rows, 1]));
    let per_token = b.transpose(b.reshape(weighted, vec![tokens, k, width]), vec![0, 2, 1]);
    let y = b.reshape(b.reduce(RedOp::Sum, per_token, 2, false), shape.clone());
    match &w.shared {
        Some(shared) => {
            let gate = linear(b, x, &shared.gate, None);
            let up = linear(b, x, &shared.up, None);
            let shared = linear(b, glu(b, gate, up, &p.glu), &shared.down, None);
            b.binary(BinOp::Add, y, b.reshape(shared, shape))
        }
        None => y,
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
                Storage::Packed => {
                    for e in 0..EXPERTS {
                        fx.packed(
                            id(MoeRole::Expert(proj, e as u32)),
                            poot_test_util::packed::random_payload(
                                WeightFormat::Q8_0,
                                [out, k],
                                100 * (proj as u64 + 1) + e as u64,
                            ),
                        );
                    }
                }
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
        let b = Builder::new();
        let step = Step::new(&b, step_shape(tokens, CAP, LogitRows::All));
        let xs = b.constant("x", TensorType::f32(vec![1, tokens, WIDTH]));
        let y = moe(&b, &step, 0, xs, None, &w, &p);
        let g = b.finish(y);
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
                    Storage::Packed => fx.values(id(MoeRole::Expert(proj, e as u32))),
                    Storage::Stack3 => {
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
    /// `[E, out, K]` stack, a stored `[E, K, out]` stack, and per-expert packed weights) matches an
    /// independent f64 reference, for a prefill-shaped step (the grouped packed chain) and a
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
            for (proj, ..) in PROJECTIONS {
                match storage {
                    Storage::Packed => {
                        ids.extend((0..EXPERTS).map(|e| id(MoeRole::Expert(proj, e as u32))));
                    }
                    _ => ids.push(id(MoeRole::Stack(proj))),
                }
            }
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
                changed.packed(
                    id(MoeRole::Expert(proj, e as u32)),
                    poot_test_util::packed::random_payload(
                        WeightFormat::Q8_0,
                        [out, k],
                        9_000 + e as u64,
                    ),
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
    }

    const BASE: Case = Case {
        orientation: Orientation::OutIn,
        form: GateUpForm::Split,
        kind: RouterKind::SoftmaxTopK { renormalize: true },
        glu: Glu::Silu,
        router_bias: false,
        expert_bias: false,
        scaled_router: false,
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
        let b = Builder::new();
        let step = Step::new(
            &b,
            StepShape {
                rows: NonZeroUsize::MIN,
                tokens: NonZeroUsize::new(TOKENS).unwrap(),
                capacity: NonZeroUsize::new(8).unwrap(),
                kv: KvLayout::Contiguous,
                logits: LogitRows::All,
            },
        );
        let xs = b.constant("x", TensorType::f32(vec![1, TOKENS, WIDTH]));
        let routes = route_inputs();
        let (y, named): (_, Vec<(&str, &[f32])>) = if c.scaled_router {
            let rs = b.constant("r", TensorType::f32(vec![1, TOKENS, WIDTH]));
            (
                moe(&b, &step, LAYER, xs, Some(rs), &w, &p),
                vec![("x", x), ("r", &routes)],
            )
        } else {
            (moe(&b, &step, LAYER, xs, None, &w, &p), vec![("x", x)])
        };
        let g = step.finish(b, y);
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
