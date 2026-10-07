//! The router of a routed layer: logits over the experts, the `top_k` experts of each token and the
//! weight each contributes. One composition for every family (spec 999): a family picks a
//! [`RouterKind`], never its own copy of the ranking.
//!
//! Ranking is [`poot_graph_ir::ops::stable_descending_rank`]: scores descending, the lower expert
//! index first on a tie, so every row is a permutation of `0..experts` and exactly `top_k` distinct
//! experts are selected. A NaN or infinite logit is canonicalized by the rank (the router never
//! reads it as an ordering accident) and reported by the [`WitnessKind::RouterLogitsFinite`]
//! witness, which stops the step.

use poot_graph_ir::{BinOp, Builder, RedOp, Scalar, Traced, UnOp, ops};
use poot_quant::weights::{MoeRole, WeightId, WeightMap, WeightRole};

use super::linear::{Weight, WeightError, linear};
use super::norm::NormParams;
use super::standard::{ParamError, Step};
use super::witness::{self, WitnessKind};
use crate::model::ConfigReason;

/// How the selected experts' scores become weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterKind {
    /// Softmax over all experts, then the `top_k` probabilities (qwen3-MoE, OLMoE). With
    /// `renormalize` they are divided by their own sum (`norm_topk_prob`); without, they are the
    /// full-softmax probabilities and do not sum to one.
    SoftmaxTopK { renormalize: bool },
    /// The `top_k` logits, then a softmax over just those (Mixtral, granite-MoE, gpt-oss). The
    /// shared normalizer cancels, so this equals `SoftmaxTopK { renormalize: true }`.
    TopKSoftmax,
    /// `sigmoid(logits)` per expert, the `top_k` of `sigmoid + selection bias`, and the gate of
    /// each selected expert taken from the UNBIASED sigmoid score (MiniMax-M2; the DeepSeek-V3
    /// router without its group limit). See [`SigmoidTopK`] for the options.
    SigmoidBiasedTopK(SigmoidTopK),
}

/// The options of [`RouterKind::SigmoidBiasedTopK`]: MiniMax-M2's router is the default, with the
/// selected scores divided by their own sum. A group limit and a routed scaling factor are options of
/// the router as a whole ([`RouterParams::with_group_limit`], [`RouterParams::with_scale`]), because
/// the softmax arm of DeepSeek-V2 has them too and the group code is one copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigmoidTopK {
    renormalize: bool,
}

impl Default for SigmoidTopK {
    fn default() -> Self {
        Self::new()
    }
}

impl SigmoidTopK {
    /// The selected scores divided by their own sum.
    pub const fn new() -> Self {
        Self { renormalize: true }
    }

    /// Whether the selected scores are divided by their own sum (`norm_topk_prob`); without, the
    /// weights are the unbiased sigmoid scores themselves.
    pub const fn with_renormalize(self, renormalize: bool) -> Self {
        Self { renormalize }
    }
}

/// What a router's bias vector does. The two meanings are different computations, so a family
/// states which one it has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterBias {
    /// Added to the logits before everything else (gpt-oss): selection and weights both see it.
    /// For the softmax arms.
    AddedToLogits,
    /// Added to the score for the selection only; the gate weights come from the unbiased scores
    /// (DeepSeek-V3's `e_score_correction_bias`, MiniMax-M2). For [`RouterKind::SigmoidBiasedTopK`].
    SelectionOnly,
}

/// Group-limited selection (DeepSeek): the experts split into `groups` equal groups and only the
/// experts of the `keep` best groups may be selected. How a group scores is a property of the
/// router's kind (see [`group_eligible`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GroupLimit {
    groups: usize,
    keep: usize,
}

/// Checked router parameters: `1 <= top_k <= experts`, and a bias placement the arm defines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouterParams {
    experts: usize,
    top_k: usize,
    kind: RouterKind,
    bias: Option<RouterBias>,
    scaled_input: Option<NormParams>,
    expert_scale: bool,
    group_limit: Option<GroupLimit>,
    scale: Option<f32>,
}

impl RouterParams {
    pub fn new(experts: usize, top_k: usize, kind: RouterKind) -> Result<Self, ParamError> {
        if experts == 0 {
            return Err(ParamError::new("experts", ConfigReason::Zero));
        }
        if top_k == 0 {
            return Err(ParamError::new("top_k", ConfigReason::Zero));
        }
        if top_k > experts {
            return Err(ParamError::new(
                "top_k",
                ConfigReason::Exceeds { max: experts },
            ));
        }
        Ok(Self {
            experts,
            top_k,
            kind,
            bias: None,
            scaled_input: None,
            expert_scale: false,
            group_limit: None,
            scale: None,
        })
    }

    /// Route on `rmsnorm(x) * scale / sqrt(width)` instead of `x` (Gemma 4): a norm with no learned
    /// scale, then the learned `[width]` router scale. It changes only what the logits are computed
    /// from, so it composes with every arm and every bias placement (the bias stays on the logits
    /// or the scores, never on the scaled input).
    pub fn with_scaled_input(self, norm: NormParams) -> Self {
        Self {
            scaled_input: Some(norm),
            ..self
        }
    }

    /// Multiply each expert's gate by its learned `[experts]` scale after the softmax arms have
    /// normalized the gates (Gemma 4's `per_expert_scale`). Refused (`expert_scale`, unsupported)
    /// for the sigmoid arm: its gates are normalized, then scaled by the family's routed factor,
    /// and no family combines the two, so the order has no reference.
    pub fn with_expert_scale(self) -> Result<Self, ParamError> {
        if matches!(self.kind, RouterKind::SigmoidBiasedTopK(_)) {
            return Err(ParamError::new("expert_scale", ConfigReason::Unsupported));
        }
        Ok(Self {
            expert_scale: true,
            ..self
        })
    }

    /// Limit the selection to the experts of the `keep` best of `groups` equal groups. Refused,
    /// naming the field, for a zero count, groups that do not divide the experts, more kept groups
    /// than groups, fewer eligible experts than `top_k`, or (the sigmoid arm scores a group by its two
    /// best experts) a group of fewer than two experts.
    pub fn with_group_limit(self, groups: usize, keep: usize) -> Result<Self, ParamError> {
        if groups == 0 {
            return Err(ParamError::new("n_group", ConfigReason::Zero));
        }
        if !self.experts.is_multiple_of(groups) {
            return Err(ParamError::new(
                "n_routed_experts",
                ConfigReason::NotDivisible { by: groups },
            ));
        }
        if keep == 0 {
            return Err(ParamError::new("topk_group", ConfigReason::Zero));
        }
        if keep > groups {
            return Err(ParamError::new(
                "topk_group",
                ConfigReason::Exceeds { max: groups },
            ));
        }
        let size = self.experts / groups;
        if matches!(self.kind, RouterKind::SigmoidBiasedTopK(_)) && size < 2 {
            return Err(ParamError::new(
                "n_group",
                ConfigReason::Exceeds {
                    max: self.experts / 2,
                },
            ));
        }
        let eligible = keep * size;
        if self.top_k > eligible {
            return Err(ParamError::new(
                "top_k",
                ConfigReason::Exceeds { max: eligible },
            ));
        }
        Ok(Self {
            group_limit: Some(GroupLimit { groups, keep }),
            ..self
        })
    }

    /// Multiply every selected expert's weight by `scale` (`routed_scaling_factor`), after any
    /// renormalization.
    pub fn with_scale(self, scale: f32) -> Result<Self, ParamError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(ParamError::new(
                "routed_scaling_factor",
                ConfigReason::NotFinitePositive,
            ));
        }
        Ok(Self {
            scale: Some(scale),
            ..self
        })
    }

    /// The router has a bias vector placed as `bias`. Refused (`router_bias`, unsupported) when the
    /// arm has no such placement: the softmax arms add the bias to the logits, the sigmoid arm
    /// selects with it.
    pub fn with_bias(self, bias: RouterBias) -> Result<Self, ParamError> {
        let defined = matches!(
            (self.kind, bias),
            (
                RouterKind::SoftmaxTopK { .. } | RouterKind::TopKSoftmax,
                RouterBias::AddedToLogits
            ) | (RouterKind::SigmoidBiasedTopK(_), RouterBias::SelectionOnly)
        );
        if !defined {
            return Err(ParamError::new("router_bias", ConfigReason::Unsupported));
        }
        Ok(Self {
            bias: Some(bias),
            ..self
        })
    }

    /// Whether the layer stores a router bias.
    pub fn biased(&self) -> bool {
        self.bias.is_some()
    }

    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }
}

/// A layer's router projection `[experts, width]` and the optional weights its parameters ask for:
/// a bias `[experts]`, the scale `[width]` on a scaled input and the scale `[experts]` on each
/// gate. [`RouterWeights::new`] loads exactly what the parameters name, so [`route`] cannot meet
/// parameters whose weights are missing.
#[derive(Clone, Debug, PartialEq)]
pub struct RouterWeights {
    weight: Weight,
    bias: Option<Weight>,
    /// The norm and the `[width]` scale of a scaled input (see [`RouterParams::with_scaled_input`]).
    scaled_input: Option<(NormParams, Weight)>,
    /// The `[experts]` scale on each expert's gate.
    expert_scale: Option<Weight>,
}

impl RouterWeights {
    pub fn new(
        map: &WeightMap,
        layer: usize,
        width: usize,
        p: &RouterParams,
    ) -> Result<Self, WeightError> {
        let id = |role| WeightId::layer(layer, WeightRole::Moe(role));
        Ok(Self {
            weight: Weight::new(map, id(MoeRole::Router), &[p.experts, width])?,
            bias: p
                .biased()
                .then(|| Weight::new(map, id(MoeRole::RouterBias), &[p.experts]))
                .transpose()?,
            scaled_input: p
                .scaled_input
                .map(|norm| Ok((norm, Weight::new(map, id(MoeRole::RouterScale), &[width])?)))
                .transpose()?,
            expert_scale: p
                .expert_scale
                .then(|| Weight::new(map, id(MoeRole::ExpertScale), &[p.experts]))
                .transpose()?,
        })
    }
}

/// What the router decided for `T` tokens: the selected experts of each token, best first, and the
/// weight of each. Both `[T, top_k]` F32 (ids are exact small integers).
#[derive(Clone, Copy, Debug)]
pub struct Routes {
    pub ids: Traced,
    pub weights: Traced,
}

/// Route the tokens `x[T, width]` of `layer` through `w`, declaring the layer's three witnesses.
pub fn route(
    b: &Builder,
    step: &Step,
    layer: usize,
    x: Traced,
    w: &RouterWeights,
    p: &RouterParams,
) -> Routes {
    let added = w
        .bias
        .as_ref()
        .filter(|_| p.bias == Some(RouterBias::AddedToLogits));
    let logits = linear(b, scaled_input(b, x, w), &w.weight, added);
    witness::declare(
        step,
        layer,
        WitnessKind::RouterLogitsFinite,
        witness::nonfinite_count(b, logits),
    );

    // The values the selection ranks, and the router's own decision of what a group scores.
    let selection = match p.kind {
        RouterKind::SigmoidBiasedTopK(_) => {
            let scores = ops::sigmoid(b, logits);
            match (&w.bias, p.bias) {
                (Some(bias), Some(RouterBias::SelectionOnly)) => {
                    let shape = b.aval(scores).shape;
                    b.binary(BinOp::Add, scores, b.broadcast(bias.declare_f32(b), shape))
                }
                _ => scores,
            }
        }
        RouterKind::SoftmaxTopK { .. } | RouterKind::TopKSoftmax => logits,
    };
    let rank = match p.group_limit {
        None => ops::stable_descending_rank(b, selection),
        Some(limit) => {
            let eligible = group_eligible(b, selection, p.kind, limit, p.experts);
            ops::stable_descending_rank_masked(b, selection, eligible)
        }
    };
    let ids = b.arg_top_k(rank, p.top_k);
    witness::declare(
        step,
        layer,
        WitnessKind::SelectorInRange,
        witness::out_of_range_count(b, ids, p.experts),
    );

    let gate = match p.kind {
        RouterKind::SoftmaxTopK { renormalize } => {
            softmax_top_k(b, logits, rank, p.top_k, renormalize)
        }
        RouterKind::TopKSoftmax => softmax_top_k(b, logits, rank, p.top_k, true),
        RouterKind::SigmoidBiasedTopK(SigmoidTopK { renormalize }) => {
            sigmoid_top_k(b, logits, rank, p.top_k, renormalize)
        }
    };
    witness::declare(
        step,
        layer,
        WitnessKind::SelectedSumUsable,
        witness::unusable_sum_count(b, gate.denominator),
    );
    let weights = match &w.expert_scale {
        Some(scale) => {
            let shape = b.aval(gate.weights).shape;
            let scale = b.broadcast(b.reshape(scale.declare_f32(b), vec![1, p.experts]), shape);
            b.binary(BinOp::Mul, gate.weights, scale)
        }
        None => gate.weights,
    };
    let weights = by_slot(b, rank, weights, p.top_k);
    Routes {
        ids,
        weights: match p.scale {
            Some(scale) => b.binary_scalar(BinOp::Mul, weights, Scalar::F32(scale)),
            None => weights,
        },
    }
}

/// The router's input: `x`, or with a scaled input `rmsnorm(x) * scale / sqrt(width)`.
fn scaled_input(b: &Builder, x: Traced, w: &RouterWeights) -> Traced {
    let Some((norm, scale)) = &w.scaled_input else {
        return x;
    };
    let shape = b.aval(x).shape;
    let width = shape[shape.len() - 1];
    let normed = ops::rmsnorm_no_weight(b, x, norm.eps());
    let scaled = b.binary(BinOp::Mul, normed, scale.declare_f32(b));
    b.binary_scalar(
        BinOp::Mul,
        scaled,
        Builder::f32(1.0 / (width as f32).sqrt()),
    )
}

/// `1.0` for the experts of the `limit.keep` best groups of each token, `0.0` elsewhere `[T, E]`. A
/// group's score belongs to the kind of router, because the two DeepSeek routers define it
/// differently: the softmax arm (V2) scores a group by its best canonical score, the sigmoid arm
/// (V3) by the sum of its two best BIASED selection scores. Ties go to the lower group.
fn group_eligible(
    b: &Builder,
    selection: Traced,
    kind: RouterKind,
    limit: GroupLimit,
    experts: usize,
) -> Traced {
    let t = b.aval(selection).shape[0];
    let size = experts / limit.groups;
    let scores = ops::canonical_router_scores(b, selection);
    let grouped = b.reshape(scores, vec![t, limit.groups, size]);
    let score = match kind {
        RouterKind::SigmoidBiasedTopK(_) => {
            let rank = ops::stable_descending_rank(b, grouped);
            let two_best = ops::top_k_keep_mask(b, rank, 2);
            b.reduce(
                RedOp::Sum,
                b.binary(BinOp::Mul, grouped, two_best),
                2,
                false,
            )
        }
        RouterKind::SoftmaxTopK { .. } | RouterKind::TopKSoftmax => {
            b.reduce(RedOp::Max, grouped, 2, false)
        }
    }; // [T, groups]
    let rank = ops::stable_descending_rank(b, score);
    let keep = ops::top_k_keep_mask(b, rank, limit.keep); // [T, groups]
    let keep = b.broadcast(
        b.reshape(keep, vec![t, limit.groups, 1]),
        vec![t, limit.groups, size],
    );
    b.reshape(keep, vec![t, experts])
}

/// The gate of every expert (`0` outside the selection) and the softmax denominator it divides by.
struct Gate {
    /// `[T, E]`.
    weights: Traced,
    /// `[T, 1]`.
    denominator: Traced,
}

/// Softmax of the canonicalized `logits`, masked to the `top_k` experts of `rank`. The numerator is
/// always the selected exponentials; the denominator is their sum when `renormalize` and the sum
/// over every expert otherwise.
fn softmax_top_k(
    b: &Builder,
    logits: Traced,
    rank: Traced,
    top_k: usize,
    renormalize: bool,
) -> Gate {
    let last = b.aval(logits).shape.len() - 1;
    let scores = ops::canonical_router_scores(b, logits);
    let keep = ops::top_k_keep_mask(b, rank, top_k);
    // The best expert is selected for every `top_k >= 1`, so the maximum is shared by the
    // selection and the full sum, and every denominator holds at least `exp(0) = 1`.
    let max = b.reduce(RedOp::Max, scores, last, true);
    let exp = b.unary(UnOp::Exp, b.binary(BinOp::Sub, scores, max));
    let selected = b.binary(BinOp::Mul, exp, keep);
    let denominator = if renormalize {
        b.reduce(RedOp::Sum, selected, last, true)
    } else {
        b.reduce(RedOp::Sum, exp, last, true)
    };
    Gate {
        weights: b.binary(BinOp::Div, selected, denominator),
        denominator,
    }
}

/// The unbiased sigmoid scores of `logits`, masked to the `top_k` experts of `rank` (which the
/// selection bias chose) and, with `renormalize`, divided by their sum. The denominator is the
/// selected sum either way: the `SelectedSumUsable` witness reads it.
fn sigmoid_top_k(
    b: &Builder,
    logits: Traced,
    rank: Traced,
    top_k: usize,
    renormalize: bool,
) -> Gate {
    let last = b.aval(logits).shape.len() - 1;
    let scores = ops::canonical_router_scores(b, ops::sigmoid(b, logits));
    let keep = ops::top_k_keep_mask(b, rank, top_k);
    let selected = b.binary(BinOp::Mul, scores, keep);
    let denominator = b.reduce(RedOp::Sum, selected, last, true);
    Gate {
        weights: if renormalize {
            b.binary(BinOp::Div, selected, denominator)
        } else {
            selected
        },
        denominator,
    }
}

/// `gate[T, E]` read at each token's slots: `out[t, r]` is the gate of the expert ranked `r`, a sum
/// over the expert axis of `(rank == r) * gate` (the rank is an exact integer permutation).
fn by_slot(b: &Builder, rank: Traced, gate: Traced, top_k: usize) -> Traced {
    let shape = b.aval(rank).shape;
    let (t, e) = (shape[0], shape[1]);
    let slots = b.slice(b.iota(e), 0, 0, top_k);
    let rank = b.broadcast(b.reshape(rank, vec![t, 1, e]), vec![t, top_k, e]);
    let slots = b.broadcast(b.reshape(slots, vec![1, top_k, 1]), vec![t, top_k, e]);
    let at_least = b.binary(BinOp::Ge, rank, slots);
    let at_most = b.binary(BinOp::Ge, slots, rank);
    let is_slot = b.binary(BinOp::Mul, at_least, at_most);
    let gate = b.broadcast(b.reshape(gate, vec![t, 1, e]), vec![t, top_k, e]);
    b.reduce(RedOp::Sum, b.binary(BinOp::Mul, is_slot, gate), 2, false)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Slot, TensorType};

    use super::*;
    use crate::components::standard::oracle::{self, Fx};
    use crate::components::testing::{CAP, step_shape};
    use crate::model::LogitRows;

    const T: usize = 4;
    const WIDTH: usize = 5;
    const EXPERTS: usize = 6;
    const TOP_K: usize = 3;

    fn router_id() -> WeightId {
        WeightId::layer(0, WeightRole::Moe(MoeRole::Router))
    }

    fn bias_id() -> WeightId {
        WeightId::layer(0, WeightRole::Moe(MoeRole::RouterBias))
    }

    fn fixture(biased: bool) -> Fx {
        let mut fx = Fx::new(11);
        fx.f32(router_id(), &[EXPERTS, WIDTH]);
        if biased {
            fx.f32(bias_id(), &[EXPERTS]);
        }
        fx
    }

    /// Router parameters of `kind`, with the bias placement `kind` defines when `biased`.
    fn params(kind: RouterKind, biased: bool) -> RouterParams {
        let p = RouterParams::new(EXPERTS, TOP_K, kind).unwrap();
        match (biased, kind) {
            (false, _) => p,
            (true, RouterKind::SigmoidBiasedTopK(_)) => {
                p.with_bias(RouterBias::SelectionOnly).unwrap()
            }
            (true, _) => p.with_bias(RouterBias::AddedToLogits).unwrap(),
        }
    }

    /// The ids and the weights the router returns over `x`, each `[T, TOP_K]`.
    fn run(fx: &Fx, kind: RouterKind, biased: bool, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        run_with(fx, &params(kind, biased), x)
    }

    /// [`run`] with the parameters given.
    fn run_with(fx: &Fx, p: &RouterParams, x: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let eval = |pick: fn(&Routes) -> Traced| {
            let map = fx.map();
            let w = RouterWeights::new(&map, 0, WIDTH, p).unwrap();
            let b = Builder::new();
            let step = Step::new(&b, step_shape(T, CAP, LogitRows::All));
            let xs = b.constant("x", TensorType::f32(vec![T, WIDTH]));
            let routes = route(&b, &step, 0, xs, &w, p);
            let g = b.finish(pick(&routes));
            let tokens = vec![0; T];
            let pos: Vec<i32> = (0..T as i32).collect();
            fx.eval(
                &g,
                &[("x", x)],
                &[(Slot::Token, &tokens), (Slot::Pos, &pos)],
                &[],
            )
            .0
        };
        (eval(|r| r.ids), eval(|r| r.weights))
    }

    /// The router in f64, written straight from the definitions: logits `x W^T + b`, the experts
    /// ranked by score (the lower index first on a tie), and the weights of `kind`.
    fn reference(fx: &Fx, kind: RouterKind, biased: bool, x: &[f32]) -> (Vec<f32>, Vec<f64>) {
        let w = fx.values(router_id());
        let bias = biased.then(|| fx.values(bias_id()));
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for token in x.chunks(WIDTH) {
            let logits: Vec<f64> = (0..EXPERTS)
                .map(|e| {
                    let dot: f64 = (0..WIDTH)
                        .map(|j| f64::from(token[j]) * f64::from(w[e * WIDTH + j]))
                        .sum();
                    dot + bias.as_ref().map_or(0.0, |b| f64::from(b[e]))
                })
                .collect();
            if let RouterKind::SigmoidBiasedTopK(_) = kind {
                // Independent of the graph: scores, then selection on score + bias, then the
                // unbiased scores of the selection.
                let plain: Vec<f64> = (0..EXPERTS)
                    .map(|e| {
                        let dot: f64 = (0..WIDTH)
                            .map(|j| f64::from(token[j]) * f64::from(w[e * WIDTH + j]))
                            .sum();
                        1.0 / (1.0 + (-dot).exp())
                    })
                    .collect();
                let chosen_by: Vec<f64> = (0..EXPERTS)
                    .map(|e| plain[e] + bias.as_ref().map_or(0.0, |b| f64::from(b[e])))
                    .collect();
                let mut order: Vec<usize> = (0..EXPERTS).collect();
                order.sort_by(|&a, &b| chosen_by[b].total_cmp(&chosen_by[a]).then(a.cmp(&b)));
                let selected = &order[..TOP_K];
                let sum: f64 = selected.iter().map(|&e| plain[e]).sum();
                for &e in selected {
                    ids.push(e as f32);
                    weights.push(plain[e] / sum);
                }
                continue;
            }
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
            let selected = &order[..TOP_K];
            let max = logits.iter().copied().fold(f64::MIN, f64::max);
            let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
            let selected_sum: f64 = selected.iter().map(|&e| exp[e]).sum();
            let full_sum: f64 = exp.iter().sum();
            let denominator = match kind {
                RouterKind::SoftmaxTopK { renormalize: false } => full_sum,
                _ => selected_sum,
            };
            for &e in selected {
                ids.push(e as f32);
                weights.push(exp[e] / denominator);
            }
        }
        (ids, weights)
    }

    const ARMS: [RouterKind; 3] = [
        RouterKind::SoftmaxTopK { renormalize: true },
        RouterKind::SoftmaxTopK { renormalize: false },
        RouterKind::TopKSoftmax,
    ];

    /// SC-002: every router arm, with and without a bias, returns the experts and weights of an
    /// independent f64 reference, the weights of the un-renormalized arm do not sum to one, and the
    /// renormalizing arms agree with each other.
    #[test]
    fn every_arm_matches_an_f64_reference() {
        let x = oracle::random(T * WIDTH, 3);
        for biased in [false, true] {
            let fx = fixture(biased);
            for kind in ARMS {
                let (ids, weights) = run(&fx, kind, biased, &x);
                let (want_ids, want) = reference(&fx, kind, biased, &x);
                assert_eq!(ids, want_ids, "{kind:?} biased={biased}: selected experts");
                oracle::assert_matches_f64(&weights, &want, 1e-5);
                let sums: Vec<f32> = weights.chunks(TOP_K).map(|r| r.iter().sum()).collect();
                let unit = |s: &f32| (s - 1.0).abs() < 1e-5;
                match kind {
                    RouterKind::SoftmaxTopK { renormalize: false } => {
                        assert!(!sums.iter().all(unit), "{sums:?} should not all be one");
                    }
                    _ => assert!(sums.iter().all(unit), "{kind:?}: {sums:?}"),
                }
            }
            let (_, a) = run(&fx, ARMS[0], biased, &x);
            let (_, b) = run(&fx, ARMS[2], biased, &x);
            oracle::assert_matches_f64(
                &a,
                &b.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                1e-6,
            );
        }
    }

    const SIGMOID_ARMS: [RouterKind; 1] = [RouterKind::SigmoidBiasedTopK(SigmoidTopK::new())];

    /// SC-001 (sigmoid arm): with and without a selection bias, the experts and weights equal an
    /// independent f64 reference, and the weights sum to one. A bias
    /// that dominates the scores selects its experts, yet their weights stay the unbiased sigmoid
    /// scores. Mutation: add the bias to the logits before the sigmoid (or weight by the biased
    /// scores) and the f64 row goes red.
    #[test]
    fn the_sigmoid_arm_selects_with_the_bias_and_weights_without_it() {
        let x = oracle::random(T * WIDTH, 8);
        for biased in [false, true] {
            let fx = fixture(biased);
            for kind in SIGMOID_ARMS {
                let (ids, weights) = run(&fx, kind, biased, &x);
                let (want_ids, want) = reference(&fx, kind, biased, &x);
                assert_eq!(ids, want_ids, "{kind:?} biased={biased}: selected experts");
                oracle::assert_matches_f64(&weights, &want, 1e-5);
                let sums: Vec<f32> = weights.chunks(TOP_K).map(|r| r.iter().sum()).collect();
                assert!(
                    sums.iter().all(|s| (s - 1.0).abs() < 1e-5),
                    "{kind:?}: {sums:?}"
                );
            }
        }
        // A bias of 10 on experts 4 and 5 puts them first whatever the scores are.
        let mut fx = fixture(false);
        fx.set(bias_id(), &[EXPERTS], vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0]);
        for kind in SIGMOID_ARMS {
            let (ids, weights) = run(&fx, kind, true, &x);
            let (want_ids, want) = reference(&fx, kind, true, &x);
            assert_eq!(ids, want_ids, "{kind:?}");
            oracle::assert_matches_f64(&weights, &want, 1e-5);
            for token in ids.chunks(TOP_K) {
                let mut lead = token[..2].to_vec();
                lead.sort_by(f32::total_cmp);
                assert_eq!(lead, [4.0, 5.0], "{kind:?}: the biased experts lead");
            }
        }
    }

    /// What [`grouped_reference`] reproduces: the scoring, the bias, the group limit, the
    /// renormalization and the scale of one router configuration.
    #[derive(Clone, Copy)]
    struct Shape {
        sigmoid: bool,
        biased: bool,
        groups: usize,
        keep: usize,
        renormalize: bool,
        scale: f64,
    }

    /// The unscaled-group reference of both DeepSeek routers in f64, written from `MoEGate.forward`
    /// (V2, softmax) and `DeepseekV3TopkRouter` (V3, sigmoid with a selection bias): the scores, the
    /// group scores (V2: best score; V3: the two best biased scores summed), the `keep` best groups,
    /// the `TOP_K` best choice values among their experts (ties to the lower index), the weights
    /// (renormalized over the selection when asked; V2 un-renormalized weights keep the full-softmax
    /// denominator), times `scale`.
    fn grouped_reference(fx: &Fx, shape: Shape, x: &[f32]) -> (Vec<f32>, Vec<f64>) {
        let Shape {
            sigmoid,
            biased,
            groups,
            keep,
            renormalize,
            scale,
        } = shape;
        let w = fx.values(router_id());
        let bias = biased.then(|| fx.values(bias_id()));
        let size = EXPERTS / groups;
        let (mut ids, mut weights) = (Vec::new(), Vec::new());
        for token in x.chunks(WIDTH) {
            let logits: Vec<f64> = (0..EXPERTS)
                .map(|e| {
                    (0..WIDTH)
                        .map(|j| f64::from(token[j]) * f64::from(w[e * WIDTH + j]))
                        .sum()
                })
                .collect();
            let (score, choice): (Vec<f64>, Vec<f64>) = if sigmoid {
                let score: Vec<f64> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
                let choice = (0..EXPERTS)
                    .map(|e| score[e] + bias.as_ref().map_or(0.0, |b| f64::from(b[e])))
                    .collect();
                (score, choice)
            } else {
                let max = logits.iter().copied().fold(f64::MIN, f64::max);
                let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
                let total: f64 = exp.iter().sum();
                let score: Vec<f64> = exp.iter().map(|e| e / total).collect();
                (score.clone(), score)
            };
            let group_score = |g: usize| {
                let mut members: Vec<f64> = (g * size..(g + 1) * size).map(|e| choice[e]).collect();
                members.sort_by(|a, b| b.total_cmp(a));
                if sigmoid {
                    members[0] + members[1]
                } else {
                    members[0]
                }
            };
            let mut order: Vec<usize> = (0..groups).collect();
            order.sort_by(|&a, &b| group_score(b).total_cmp(&group_score(a)).then(a.cmp(&b)));
            let kept = &order[..keep];
            let mut eligible: Vec<usize> = (0..EXPERTS)
                .filter(|e| kept.contains(&(e / size)))
                .collect();
            eligible.sort_by(|&a, &b| choice[b].total_cmp(&choice[a]).then(a.cmp(&b)));
            let selected = &eligible[..TOP_K];
            let denominator: f64 = if renormalize {
                selected.iter().map(|&e| score[e]).sum()
            } else {
                1.0
            };
            for &e in selected {
                ids.push(e as f32);
                weights.push(score[e] / denominator * scale);
            }
        }
        (ids, weights)
    }

    /// SC-002 (the DeepSeek router arms): the group-limited softmax arm (V2) and the sigmoid arm (V3)
    /// with its options, with and without renormalization, a selection bias and a routed scale,
    /// select the experts and weights of an independent f64 reference, the group limit changes some
    /// selection against the unlimited arm, and each router weight and the bias matter.
    /// Mutations (one each, all red): leak the bias into the sigmoid weights; score a V3 group on the
    /// unbiased scores; drop the scale; keep every group.
    #[test]
    fn the_deepseek_router_arms_match_an_f64_reference() {
        let inputs = |seed| {
            oracle::random(T * WIDTH, seed)
                .into_iter()
                .map(|v| 6.0 * v)
                .collect::<Vec<_>>()
        };
        let x = inputs(21);
        let mut fx = fixture(true);
        // A bias that reorders the unbiased scores, so a bias leak or an unbiased group score moves
        // the selection.
        fx.set(bias_id(), &[EXPERTS], vec![0.9, -0.4, 0.3, 0.7, -0.8, 0.2]);
        let arms = [
            (false, false, false),
            (false, true, false),
            (true, true, true),
            (true, true, false),
            (true, false, true),
        ];
        for (sigmoid, renormalize, biased) in arms {
            for (groups, keep) in [(3, 2), (3, 3), (2, 2), (1, 1)] {
                let kind = if sigmoid {
                    RouterKind::SigmoidBiasedTopK(SigmoidTopK::new().with_renormalize(renormalize))
                } else {
                    RouterKind::SoftmaxTopK { renormalize }
                };
                let mut p = RouterParams::new(EXPERTS, TOP_K, kind)
                    .unwrap()
                    .with_group_limit(groups, keep)
                    .unwrap()
                    .with_scale(2.5)
                    .unwrap();
                if biased {
                    p = p.with_bias(RouterBias::SelectionOnly).unwrap();
                }
                let (ids, weights) = run_with(&fx, &p, &x);
                let (want_ids, want) = grouped_reference(
                    &fx,
                    Shape {
                        sigmoid,
                        biased,
                        groups,
                        keep,
                        renormalize,
                        scale: 2.5,
                    },
                    &x,
                );
                let case = format!(
                    "sigmoid={sigmoid} renormalize={renormalize} biased={biased} groups={groups} keep={keep}"
                );
                assert_eq!(ids, want_ids, "{case}");
                oracle::assert_matches_f64(&weights, &want, 1e-5);
            }
        }
        // The limit is not the identity: over some inputs it selects differently from the unlimited
        // arm, for both scorings.
        for sigmoid in [false, true] {
            let kind = if sigmoid {
                RouterKind::SigmoidBiasedTopK(SigmoidTopK::new())
            } else {
                RouterKind::SoftmaxTopK { renormalize: true }
            };
            let plain = RouterParams::new(EXPERTS, TOP_K, kind).unwrap();
            let limited = plain.with_group_limit(3, 2).unwrap();
            let changed = (21..61).any(|seed| {
                let x = inputs(seed);
                run_with(&fx, &limited, &x).0 != run_with(&fx, &plain, &x).0
            });
            assert!(
                changed,
                "sigmoid={sigmoid}: the group limit never changed a selection"
            );
        }
        // Every router weight matters.
        let p = RouterParams::new(
            EXPERTS,
            TOP_K,
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new()),
        )
        .unwrap()
        .with_bias(RouterBias::SelectionOnly)
        .unwrap()
        .with_group_limit(3, 2)
        .unwrap();
        let base = run_with(&fx, &p, &x);
        for id in [router_id(), bias_id()] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id);
            assert!(
                run_with(&perturbed, &p, &x) != base,
                "no dependence on {id}"
            );
        }
    }

    /// The sigmoid arm with a bias, a group limit and a scale equals the graph-level DeepSeek-V3
    /// router the GLM-5.3 tests also compare against (`deepseek3_router_gate_and_ids`, written
    /// separately from this module): the same experts, the same weights.
    #[test]
    fn the_sigmoid_arm_equals_the_deepseek3_graph_reference() {
        let x = oracle::random(T * WIDTH, 31)
            .into_iter()
            .map(|v| 6.0 * v)
            .collect::<Vec<_>>();
        let mut fx = fixture(true);
        fx.set(bias_id(), &[EXPERTS], vec![0.9, -0.4, 0.3, 0.7, -0.8, 0.2]);
        let p = RouterParams::new(
            EXPERTS,
            TOP_K,
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new()),
        )
        .unwrap()
        .with_bias(RouterBias::SelectionOnly)
        .unwrap()
        .with_group_limit(3, 2)
        .unwrap()
        .with_scale(2.5)
        .unwrap();
        let (ids, weights) = run_with(&fx, &p, &x);

        let w = fx.values(router_id());
        let logits: Vec<f32> = x
            .chunks(WIDTH)
            .flat_map(|token| {
                (0..EXPERTS).map(|e| {
                    (0..WIDTH)
                        .map(|j| f64::from(token[j]) * f64::from(w[e * WIDTH + j]))
                        .sum::<f64>() as f32
                })
            })
            .collect::<Vec<_>>();
        let bias = fx.values(bias_id());
        let reference = |pick: fn((Traced, Traced)) -> Traced| {
            let b = Builder::new();
            let l = b.constant("l", TensorType::f32(vec![T, EXPERTS]));
            let bs = b.constant("bias", TensorType::f32(vec![EXPERTS]));
            let out = pick(crate::deepseek3::deepseek3_router_gate_and_ids(
                &b, l, bs, TOP_K, 3, 2, 2.5,
            ));
            let g = b.finish(out);
            fx.eval(&g, &[("l", &logits), ("bias", &bias)], &[], &[]).0
        };
        assert_eq!(ids, reference(|(_, ids)| ids), "selected experts");
        // The reference's gate is dense over the experts (zero outside the selection).
        let mut dense = [0.0f64; T * EXPERTS];
        for t in 0..T {
            for r in 0..TOP_K {
                dense[t * EXPERTS + ids[t * TOP_K + r] as usize] =
                    f64::from(weights[t * TOP_K + r]);
            }
        }
        let want: Vec<f64> = reference(|(w, _)| w).into_iter().map(f64::from).collect();
        oracle::assert_matches_f64(
            &dense.iter().map(|&v| v as f32).collect::<Vec<_>>(),
            &want,
            1e-5,
        );
    }

    /// An impossible group limit or scale is refused naming the field.
    #[test]
    fn params_refuse_an_impossible_group_limit_or_scale() {
        let softmax = RouterParams::new(EXPERTS, TOP_K, ARMS[0]).unwrap();
        let refused = |r: Result<RouterParams, ParamError>| r.unwrap_err();
        assert_eq!(
            refused(softmax.with_group_limit(0, 1)),
            ParamError::new("n_group", ConfigReason::Zero)
        );
        assert_eq!(
            refused(softmax.with_group_limit(4, 2)),
            ParamError::new("n_routed_experts", ConfigReason::NotDivisible { by: 4 })
        );
        assert_eq!(
            refused(softmax.with_group_limit(3, 0)),
            ParamError::new("topk_group", ConfigReason::Zero)
        );
        assert_eq!(
            refused(softmax.with_group_limit(3, 4)),
            ParamError::new("topk_group", ConfigReason::Exceeds { max: 3 })
        );
        // One group of two eligible experts cannot hold a `top_k` of three.
        assert_eq!(
            refused(softmax.with_group_limit(3, 1)),
            ParamError::new("top_k", ConfigReason::Exceeds { max: 2 })
        );
        // The sigmoid arm scores a group by its two best experts: a group of one has no pair.
        let sigmoid = RouterParams::new(EXPERTS, TOP_K, SIGMOID_ARMS[0]).unwrap();
        assert_eq!(
            refused(sigmoid.with_group_limit(6, 6)),
            ParamError::new("n_group", ConfigReason::Exceeds { max: 3 })
        );
        assert_eq!(
            refused(softmax.with_scale(0.0)),
            ParamError::new("routed_scaling_factor", ConfigReason::NotFinitePositive)
        );
        assert!(
            softmax
                .with_group_limit(3, 2)
                .unwrap()
                .with_scale(2.5)
                .is_ok()
        );
        // The routed scale and Gemma 4's per-expert scale are different options: the sigmoid arm
        // takes the first and refuses the second, the softmax arm composes both with a group limit.
        assert!(sigmoid.with_scale(2.5).is_ok());
        assert_eq!(
            sigmoid.with_expert_scale().map(|_| ()),
            Err(ParamError::new("expert_scale", ConfigReason::Unsupported))
        );
        assert!(
            softmax
                .with_group_limit(3, 2)
                .unwrap()
                .with_scale(2.5)
                .unwrap()
                .with_expert_scale()
                .is_ok()
        );
    }

    /// `RouterParams::with_bias` refuses the placement an arm does not define.
    #[test]
    fn a_bias_placement_the_arm_does_not_define_is_refused() {
        let refused = Err(ParamError::new("router_bias", ConfigReason::Unsupported));
        let sigmoid = RouterParams::new(4, 2, SIGMOID_ARMS[0]).unwrap();
        assert_eq!(
            sigmoid.with_bias(RouterBias::AddedToLogits).map(|_| ()),
            refused
        );
        let softmax = RouterParams::new(4, 2, RouterKind::TopKSoftmax).unwrap();
        assert_eq!(
            softmax.with_bias(RouterBias::SelectionOnly).map(|_| ()),
            refused
        );
        assert!(sigmoid.with_bias(RouterBias::SelectionOnly).is_ok());
        assert!(softmax.with_bias(RouterBias::AddedToLogits).is_ok());
    }

    /// Each router weight matters: perturbing the projection or the bias changes the weights.
    #[test]
    fn the_projection_and_the_bias_change_the_routes() {
        let x = oracle::random(T * WIDTH, 4);
        let fx = fixture(true);
        let (_, base) = run(&fx, ARMS[0], true, &x);
        for id in [router_id(), bias_id()] {
            let mut perturbed = fx.clone();
            perturbed.perturb(id);
            let (ids, weights) = run(&perturbed, ARMS[0], true, &x);
            let (base_ids, _) = run(&fx, ARMS[0], true, &x);
            assert!(
                ids != base_ids || weights != base,
                "the routes do not depend on {id}"
            );
        }
    }

    /// Equal logits rank by expert index: an all-zero router selects experts `0..top_k` in order
    /// and weights them equally.
    #[test]
    fn a_tie_selects_the_lowest_indices_first() {
        let mut fx = Fx::new(1);
        fx.set(router_id(), &[EXPERTS, WIDTH], vec![0.0; EXPERTS * WIDTH]);
        let x = oracle::random(T * WIDTH, 5);
        for kind in ARMS {
            let (ids, weights) = run(&fx, kind, false, &x);
            let want_ids: Vec<f32> = (0..T).flat_map(|_| 0..TOP_K).map(|e| e as f32).collect();
            assert_eq!(ids, want_ids, "{kind:?}");
            let (_, want) = reference(&fx, kind, false, &x);
            oracle::assert_matches_f64(&weights, &want, 1e-6);
        }
    }

    /// `RouterParams::new` refuses a zero extent and a `top_k` above the experts, naming the field.
    #[test]
    fn params_refuse_an_impossible_selection() {
        let kind = RouterKind::TopKSoftmax;
        assert_eq!(
            RouterParams::new(0, 1, kind),
            Err(ParamError::new("experts", ConfigReason::Zero))
        );
        assert_eq!(
            RouterParams::new(4, 0, kind),
            Err(ParamError::new("top_k", ConfigReason::Zero))
        );
        assert_eq!(
            RouterParams::new(4, 5, kind),
            Err(ParamError::new("top_k", ConfigReason::Exceeds { max: 4 }))
        );
        assert!(RouterParams::new(4, 4, kind).is_ok());
    }

    /// Gemma 4's scaled router input composes with every arm and bias placement as an input
    /// transform: routing over `x` with a scaled input equals routing over the transformed input
    /// `rmsnorm(x) * scale / sqrt(width)` without it, so a bias still acts on the logits (softmax
    /// arms) or the scores (sigmoid arm), never on the scaled input. Mutation: route on `x` instead of
    /// `scaled_input(b, x, w)` in `route`; every row goes red.
    #[test]
    fn a_scaled_input_equals_routing_over_the_transformed_input_for_every_arm() {
        let norm = NormParams::new(1e-6).unwrap();
        let scale_id = WeightId::layer(0, WeightRole::Moe(MoeRole::RouterScale));
        let x = oracle::random(T * WIDTH, 31);
        for kind in [
            RouterKind::SoftmaxTopK { renormalize: true },
            RouterKind::SoftmaxTopK { renormalize: false },
            RouterKind::TopKSoftmax,
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new()),
        ] {
            for biased in [false, true] {
                let mut fx = fixture(biased);
                fx.f32(scale_id, &[WIDTH]);
                let scale = fx.values(scale_id);
                let transformed: Vec<f32> = x
                    .chunks(WIDTH)
                    .flat_map(|token| {
                        let inv = 1.0
                            / (token.iter().map(|v| f64::from(v * v)).sum::<f64>() / WIDTH as f64
                                + 1e-6)
                                .sqrt();
                        token
                            .iter()
                            .zip(&scale)
                            .map(|(&v, &s)| {
                                (f64::from(v) * inv * f64::from(s) / (WIDTH as f64).sqrt()) as f32
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let scaled = params(kind, biased).with_scaled_input(norm);
                let (got_ids, got) = run_with(&fx, &scaled, &x);
                let (want_ids, want) = run(&fx, kind, biased, &transformed);
                assert_eq!(got_ids, want_ids, "{kind:?} biased={biased}");
                let want: Vec<f64> = want.into_iter().map(f64::from).collect();
                oracle::assert_matches_f64(&got, &want, 1e-4);
            }
        }
    }

    /// The per-expert gate scale multiplies the normalized gates of the softmax arms (each
    /// selected expert's gate times its scale), and the sigmoid arm refuses it: its gates are
    /// normalized and then scaled by the family's routed factor, and no family combines the two.
    #[test]
    fn an_expert_scale_multiplies_the_softmax_gates_and_the_sigmoid_arm_refuses_it() {
        let scale_id = WeightId::layer(0, WeightRole::Moe(MoeRole::ExpertScale));
        let x = oracle::random(T * WIDTH, 41);
        for kind in [
            RouterKind::SoftmaxTopK { renormalize: true },
            RouterKind::TopKSoftmax,
        ] {
            let mut fx = fixture(false);
            fx.f32(scale_id, &[EXPERTS]);
            let scale = fx.values(scale_id);
            let (ids, plain) = run(&fx, kind, false, &x);
            let scaled = params(kind, false).with_expert_scale().unwrap();
            let (scaled_ids, got) = run_with(&fx, &scaled, &x);
            assert_eq!(ids, scaled_ids);
            let want: Vec<f64> = ids
                .iter()
                .zip(&plain)
                .map(|(&e, &w)| f64::from(w) * f64::from(scale[e as usize]))
                .collect();
            oracle::assert_matches_f64(&got, &want, 1e-5);
        }
        let sigmoid = RouterParams::new(
            EXPERTS,
            TOP_K,
            RouterKind::SigmoidBiasedTopK(SigmoidTopK::new()),
        )
        .unwrap();
        assert_eq!(
            sigmoid.with_expert_scale().unwrap_err(),
            ParamError::new("expert_scale", ConfigReason::Unsupported)
        );
    }
}
