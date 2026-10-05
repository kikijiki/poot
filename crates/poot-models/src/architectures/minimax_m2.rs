//! The MiniMax-M2.5 text graph: token embedding, the configured decoder blocks, final RMSNorm, and the
//! untied LM head, for prefill and fixed-capacity decode.
//!
//! Attention is ordinary GQA with flattened projection-wide Q/K RMSNorm and partial RoPE. The router uses a
//! selection-only correction bias like DeepSeek-V3 but has no group limit, shared expert, or routed scaling
//! factor, so it has its own gate rather than calling `deepseek3`'s (whose group-limiting would be dead in every
//! block).
//!
//! One block composition serves both phases; they differ in the token axis and in which card 369 helper the
//! routed experts call (`packed_indexed_linear` for decode, `packed_grouped_linear` for prefill). Nothing here
//! hand-builds a packed-dequant, expert concat, or grouped sort/scatter/offset.
//!
//! This module composes graph semantics and source names only; it makes no Runner, loader, or backend claim.

#[cfg(test)]
use poot_graph_ir::ops::{
    PackedLinearGraphRow, attention_masked, linear, packed_grouped_linear, packed_indexed_linear,
    packed_linear, rmsnorm, rope, rope_prefill,
};
#[cfg(test)]
use poot_graph_ir::ops::{sigmoid, stable_descending_rank, swiglu, top_k_keep_mask};
#[cfg(test)]
use poot_graph_ir::{
    BinOp, Builder, BuilderAppendError, GraphValidationError, IndexGuardError, RedOp, Traced,
    ValidationId,
};
#[cfg(test)]
use poot_graph_ir::{Graph, Scalar, Slot, StateRole, TensorType, UnOp, ValidationOutputs};
#[cfg(test)]
use poot_load::minimax_m2::{
    MiniMaxM2AttentionProjection, MiniMaxM2DenseRole, MiniMaxM2DenseSource,
    MiniMaxM2ExpertProjection, MiniMaxM2LayerSources, MiniMaxM2PackedSource,
};
#[cfg(test)]
use poot_load::minimax_m2::{MiniMaxM2Config, MiniMaxM2SourceTable, MiniMaxM2SourceTableError};
#[cfg(test)]
use poot_tensor::DType;

/// Graph name of the shared cosine RoPE table.
#[cfg(test)]
pub(crate) const MINIMAX_M2_ROPE_COS: &str = "minimax_m2.rope.cos";
/// Graph name of the shared sine RoPE table.
#[cfg(test)]
pub(crate) const MINIMAX_M2_ROPE_SIN: &str = "minimax_m2.rope.sin";
/// Graph name of the `[0, 1, .., experts]` expert ordinals the grouped prefill route needs.
#[cfg(test)]
pub(crate) const MINIMAX_M2_EXPERT_IOTA: &str = "minimax_m2.expert_iota";

/// The three runtime conditions card 375a's validation packet observes, one family per layer.
///
/// Ids are stable: [`minimax_m2_router_validation_error`] maps a failure id back to a typed model error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum MiniMaxM2Validation {
    /// A router logit is NaN or infinite.
    RouterLogitsFinite,
    /// The selected unbiased scores do not sum to a finite positive number, so the normalization would
    /// publish NaN rather than a convex combination.
    SelectedSumUsable,
    /// A runtime selected expert id fell outside `0..experts`.
    SelectorInRange,
}

#[cfg(test)]
impl MiniMaxM2Validation {
    pub const ALL: [Self; 3] = [
        Self::RouterLogitsFinite,
        Self::SelectedSumUsable,
        Self::SelectorInRange,
    ];

    const fn family(self) -> u32 {
        match self {
            Self::RouterLogitsFinite => 0,
            Self::SelectedSumUsable => 1,
            Self::SelectorInRange => 2,
        }
    }

    #[cfg(test)]
    const fn label(self) -> &'static str {
        match self {
            Self::RouterLogitsFinite => "router_logits_finite",
            Self::SelectedSumUsable => "selected_sum_usable",
            Self::SelectorInRange => "selector_in_range",
        }
    }

    /// The packed validation id of this family on `layer`; [`MiniMaxM2Validation::decode`] inverts it.
    #[cfg(test)]
    fn id(self, layer: usize) -> Result<ValidationId, MiniMaxM2TraceError> {
        let families = Self::ALL.len() as u32;
        let layer =
            u32::try_from(layer).map_err(|_| MiniMaxM2TraceError::ValidationIdOverflow {
                layer,
                family: self.label(),
            })?;
        layer
            .checked_mul(families)
            .and_then(|base| base.checked_add(self.family()))
            .map(ValidationId)
            .ok_or(MiniMaxM2TraceError::ValidationIdOverflow {
                layer: layer as usize,
                family: self.label(),
            })
    }

    /// Recover `(layer, family)` from a validation id this module minted. The token-guard sentinel is rejected
    /// here so the per-layer and graph-wide id spaces cannot be confused.
    pub fn decode(id: ValidationId) -> Option<(usize, Self)> {
        if id == MINIMAX_M2_TOKEN_INDEX_VALIDATION_ID {
            return None;
        }
        let families = Self::ALL.len() as u32;
        let family = Self::ALL
            .into_iter()
            .find(|candidate| candidate.family() == id.0 % families)?;
        Some(((id.0 / families) as usize, family))
    }

    #[cfg(test)]
    fn name(self, layer: usize) -> String {
        format!("minimax_m2.layer{layer}.{}", self.label())
    }
}

/// The token-index guard is one graph-wide witness, so it sits above the per-layer id space.
#[cfg(test)]
const MINIMAX_M2_TOKEN_INDEX_VALIDATION: &str = "minimax_m2.token_index_in_range";
#[cfg(test)]
const MINIMAX_M2_TOKEN_INDEX_VALIDATION_ID: ValidationId = ValidationId(u32::MAX);

/// Why a MiniMax-M2 text graph could not be traced, or why one failed at run time.
#[derive(Debug, thiserror::Error)]
#[cfg(test)]
pub(crate) enum MiniMaxM2TraceError {
    #[error(transparent)]
    SourceTable(#[from] MiniMaxM2SourceTableError),
    #[error("MiniMax-M2 decode requires use_cache")]
    CacheDisabled,
    #[error("MiniMax-M2 {field} must be positive, got 0")]
    ZeroExtent { field: &'static str },
    #[error("MiniMax-M2 prefill of {tokens} tokens does not fit capacity {capacity}")]
    PrefillCapacity { tokens: usize, capacity: usize },
    #[error("MiniMax-M2 top-k {top_k} is outside 1..={experts}")]
    TopK { top_k: usize, experts: usize },
    #[error("MiniMax-M2 rotary_dim {rotary_dim} must be even and at most head_dim {head_dim}")]
    RotaryWidth { rotary_dim: usize, head_dim: usize },
    #[error("MiniMax-M2 validation id for layer {layer} family {family} overflows")]
    ValidationIdOverflow { layer: usize, family: &'static str },
    #[error("MiniMax-M2 token index guard: {0}")]
    TokenIndexGuard(#[from] IndexGuardError),
    #[error("MiniMax-M2 packed graph construction: {0}")]
    PackedAppend(#[from] BuilderAppendError),
    #[error("MiniMax-M2 graph is not well formed: {0}")]
    Invalid(#[from] GraphValidationError),
    #[error("MiniMax-M2 router failed at run time on layer {layer}: {condition}")]
    Router {
        layer: usize,
        condition: &'static str,
    },
    #[error("MiniMax-M2 token index was outside 0..vocab_size at run time")]
    TokenIndexOutOfRange,
}

/// Map a card 375a execution failure back to the typed model error for the condition it observed. The executor
/// reports a neutral `(id, name, lane)`; an id this module did not mint returns `None`.
#[cfg(test)]
pub(crate) fn minimax_m2_router_validation_error(
    failure: &poot_graph_ir::ExecutionValidationFailure,
) -> Option<MiniMaxM2TraceError> {
    if failure.name == MINIMAX_M2_TOKEN_INDEX_VALIDATION {
        return Some(MiniMaxM2TraceError::TokenIndexOutOfRange);
    }
    let (layer, family) = MiniMaxM2Validation::decode(failure.id)?;
    if failure.name != family.name(layer) {
        return None;
    }
    Some(MiniMaxM2TraceError::Router {
        layer,
        condition: family.label(),
    })
}

/// One layer's routing decision, with the intermediates the card 375a witnesses observe.
///
/// Every field is `[T, experts]` except `ids`, which is `[T, top_k]`.
#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct MiniMaxM2Routes {
    /// Stable descending rank of the bias-adjusted scores, lower expert id winning a tie.
    pub rank: Traced,
    /// `[T, 1]` sum of the top-`k` scores, the normalization denominator.
    pub selected_sum: Traced,
    /// Top-`k` scores normalized to sum to one across the selected experts.
    pub weights: Traced,
    /// Rank-ordered selected expert ids.
    pub ids: Traced,
}

/// MiniMax-M2's inference router over `logits[T, experts]`.
///
/// `raw = sigmoid(logits)`; the correction bias is added for selection only; the top `top_k` ranks are
/// kept; the gathered weights come from the unbiased `raw` and are renormalized to one. There is no group
/// limit, shared expert, routed scaling factor, expert clamp, or softmax gate. Stable lower-expert-id tie
/// breaking is poot's deterministic policy for the upstream `topk(sorted=False)` ambiguity.
///
/// Model-local rather than `deepseek3`'s gate, whose group-limiting would be dead weight in all 62 blocks.
#[cfg(test)]
pub(crate) fn minimax_m2_router_routes(
    b: &Builder,
    logits: Traced,
    correction_bias: Traced,
    top_k: usize,
) -> MiniMaxM2Routes {
    let shape = b.aval(logits).shape;
    let [tokens, experts] = match shape.as_slice() {
        &[tokens, experts] => [tokens, experts],
        _ => panic!("MiniMax-M2 router logits must be [tokens, experts], got {shape:?}"),
    };
    assert!(
        (1..=experts).contains(&top_k),
        "invalid MiniMax-M2 top-k {top_k} for {experts} experts"
    );
    assert_eq!(
        b.aval(correction_bias).shape,
        [experts],
        "MiniMax-M2 correction bias must have one value per expert"
    );

    let scores = sigmoid(b, logits);
    let bias = b.broadcast(
        b.reshape(correction_bias, vec![1, experts]),
        vec![tokens, experts],
    );
    let rank = stable_descending_rank(b, b.binary(BinOp::Add, scores, bias));
    let keep = top_k_keep_mask(b, rank, top_k);
    let selected = b.binary(BinOp::Mul, scores, keep);
    let selected_sum = b.reduce(RedOp::Sum, selected, 1, true);
    MiniMaxM2Routes {
        rank,
        selected_sum,
        weights: b.binary(BinOp::Div, selected, selected_sum),
        ids: b.arg_top_k(rank, top_k),
    }
}

/// The source names, roles and packed descriptors one text graph reads, taken from
/// `poot_load::minimax_m2::MiniMaxM2SourceTable` so the graph and the disposition manifest cannot drift.
#[derive(Clone, Debug)]
#[cfg(test)]
pub(crate) struct MiniMaxM2TextSourcePlan {
    config: MiniMaxM2Config,
    table: MiniMaxM2SourceTable,
}

#[cfg(test)]
impl MiniMaxM2TextSourcePlan {
    pub fn new(config: &MiniMaxM2Config) -> Result<Self, MiniMaxM2TraceError> {
        let experts = config.num_local_experts;
        let top_k = config.num_experts_per_tok;
        if !(1..=experts).contains(&top_k) {
            return Err(MiniMaxM2TraceError::TopK { top_k, experts });
        }
        for (field, extent) in [
            ("hidden_size", config.hidden_size),
            ("intermediate_size", config.intermediate_size),
            ("num_hidden_layers", config.num_hidden_layers),
            ("vocab_size", config.vocab_size),
        ] {
            if extent == 0 {
                return Err(MiniMaxM2TraceError::ZeroExtent { field });
            }
        }
        if config.rotary_dim == 0
            || !config.rotary_dim.is_multiple_of(2)
            || config.rotary_dim > config.head_dim
        {
            return Err(MiniMaxM2TraceError::RotaryWidth {
                rotary_dim: config.rotary_dim,
                head_dim: config.head_dim,
            });
        }
        Ok(Self {
            config: config.clone(),
            table: MiniMaxM2SourceTable::new(config)?,
        })
    }

    pub const fn config(&self) -> &MiniMaxM2Config {
        &self.config
    }

    pub const fn table(&self) -> &MiniMaxM2SourceTable {
        &self.table
    }
}

/// Trace the whole-prompt prefill graph for `tokens` positions starting at 0. It publishes the same state names
/// and shapes as [`trace_minimax_m2_text_decode`]; routed experts take card 369's grouped form.
#[cfg(test)]
pub(crate) fn trace_minimax_m2_text_prefill(
    plan: &MiniMaxM2TextSourcePlan,
    tokens: usize,
    capacity: usize,
) -> Result<Graph<ValidationOutputs>, MiniMaxM2TraceError> {
    if tokens == 0 {
        return Err(MiniMaxM2TraceError::ZeroExtent { field: "tokens" });
    }
    if capacity < tokens {
        return Err(MiniMaxM2TraceError::PrefillCapacity { tokens, capacity });
    }
    MiniMaxM2Trace::new(plan, MiniMaxM2Phase::Prefill { tokens }, capacity)?.run()
}

/// Trace one fixed-capacity decode step whose token, position and mask are runtime slots. Routed experts take
/// card 369's indexed form; nothing else differs from prefill.
#[cfg(test)]
pub(crate) fn trace_minimax_m2_text_decode(
    plan: &MiniMaxM2TextSourcePlan,
    capacity: usize,
) -> Result<Graph<ValidationOutputs>, MiniMaxM2TraceError> {
    if !plan.config.use_cache {
        return Err(MiniMaxM2TraceError::CacheDisabled);
    }
    if capacity == 0 {
        return Err(MiniMaxM2TraceError::ZeroExtent { field: "capacity" });
    }
    MiniMaxM2Trace::new(plan, MiniMaxM2Phase::Decode, capacity)?.run()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
enum MiniMaxM2Phase {
    Prefill { tokens: usize },
    Decode,
}

#[cfg(test)]
impl MiniMaxM2Phase {
    #[cfg(test)]
    const fn tokens(self) -> usize {
        match self {
            Self::Prefill { tokens } => tokens,
            Self::Decode => 1,
        }
    }
}

/// How one layer selects the route weight of each `(token, slot)` pair.
#[derive(Clone, Copy, Debug)]
#[cfg(test)]
enum MiniMaxM2RouteSlots {
    /// One token. `Gather` shares its single index set across the one leading row, which is exact.
    Single,
    /// `T > 1`. There is no batched take-along-axis, so the per-slot weight comes from a one-hot over the shared
    /// rank (step 4 of `ops::moe_grouped_prep`) using the expert iota's leading ordinals.
    Batched { expert_iota: Traced },
}

/// One text graph under construction; everything that differs between phases is reachable from `phase` and
/// `route_slots`.
#[cfg(test)]
struct MiniMaxM2Trace<'a> {
    b: Builder,
    plan: &'a MiniMaxM2TextSourcePlan,
    phase: MiniMaxM2Phase,
    capacity: usize,
    rope_tables: (Traced, Traced),
    route_slots: MiniMaxM2RouteSlots,
    /// The runtime decode position; prefill writes its cache rows at a static 0 instead.
    position: Option<Traced>,
    /// `[1, 1, T, capacity]` additive visibility, so both phases use one attention primitive.
    mask: Traced,
    token: Traced,
    state: Vec<(Traced, Traced)>,
    validations: Vec<(ValidationId, String, Traced)>,
}

#[cfg(test)]
impl<'a> MiniMaxM2Trace<'a> {
    #[cfg(test)]
    fn new(
        plan: &'a MiniMaxM2TextSourcePlan,
        phase: MiniMaxM2Phase,
        capacity: usize,
    ) -> Result<Self, MiniMaxM2TraceError> {
        let config = &plan.config;
        let tokens = phase.tokens();
        let b = Builder::new();

        // The token slot is rank 1 in both phases so card 372c's last-axis guard applies unchanged.
        let raw_token = b.slot(Slot::Token, TensorType::new(vec![tokens], DType::I32));
        let guarded = b.guard_index_bounds(raw_token, config.vocab_size)?;

        let position = match phase {
            MiniMaxM2Phase::Prefill { .. } => None,
            // No `Slot::SeqLen`: an unread I32 slot makes the graph unevaluable (the evaluator admits integer
            // inputs only for `Slot::Token` and `Slot::Pos`), and the mask already carries cache visibility.
            MiniMaxM2Phase::Decode => Some(b.slot(Slot::Pos, TensorType::scalar(DType::I32))),
        };
        let mask = b.slot(Slot::Mask, TensorType::f32(vec![tokens, capacity]));
        let mask = b.reshape(mask, vec![1, 1, tokens, capacity]);

        let rope_table = TensorType::f32(vec![config.max_position_embeddings, config.rotary_dim]);
        let rope_tables = (
            b.constant(MINIMAX_M2_ROPE_COS, rope_table.clone()),
            b.constant(MINIMAX_M2_ROPE_SIN, rope_table),
        );
        let route_slots = match phase {
            MiniMaxM2Phase::Decode => MiniMaxM2RouteSlots::Single,
            MiniMaxM2Phase::Prefill { .. } => MiniMaxM2RouteSlots::Batched {
                expert_iota: b.constant(
                    MINIMAX_M2_EXPERT_IOTA,
                    TensorType::f32(vec![config.num_local_experts + 1]),
                ),
            },
        };

        let mut trace = Self {
            b,
            plan,
            phase,
            capacity,
            rope_tables,
            route_slots,
            position,
            mask,
            token: guarded.guarded,
            state: Vec::with_capacity(2 * config.num_hidden_layers),
            validations: Vec::with_capacity(
                1 + MiniMaxM2Validation::ALL.len() * config.num_hidden_layers,
            ),
        };
        trace.validations.push((
            MINIMAX_M2_TOKEN_INDEX_VALIDATION_ID,
            MINIMAX_M2_TOKEN_INDEX_VALIDATION.to_string(),
            guarded.witness,
        ));
        Ok(trace)
    }

    fn run(mut self) -> Result<Graph<ValidationOutputs>, MiniMaxM2TraceError> {
        let layers = self.plan.config.num_hidden_layers;
        let mut x = self.embed();
        for layer in 0..layers {
            let _layer_scope = self.b.layer_scope(layer);
            x = self.block(x, layer)?;
        }
        let logits = self.head(x);
        let Self {
            b,
            state,
            validations,
            ..
        } = self;
        let validations = validations
            .iter()
            .map(|(id, name, value)| (*id, name.as_str(), *value))
            .collect::<Vec<_>>();
        Ok(
            poot_test_util::graph_fixtures::finish_with_state_and_validations(
                b,
                logits,
                &state,
                &validations,
            )?,
        )
    }

    /// `[T, hidden]` F32 activations from the BF16 embedding table. The row gather reads the BF16 source directly
    /// (card 381's `DenseRowGather`), so no F32 copy of `[vocab, hidden]` exists.
    fn embed(&mut self) -> Traced {
        let plan = self.plan;
        let embedding = self.dense(plan.table.embedding());
        let rows = self.b.gather(embedding, 0, self.token);
        self.b.cast(rows, DType::F32)
    }

    /// `x = x + attention(rmsnorm(x))`, then `x = x + routed_moe(rmsnorm(x))`.
    fn block(&mut self, x: Traced, layer: usize) -> Result<Traced, MiniMaxM2TraceError> {
        let plan = self.plan;
        let sources = &plan.table.layers()[layer];
        let eps = plan.config.rms_norm_eps;

        let input_norm = self.dense_f32(sources.input_norm());
        let normed = rmsnorm(&self.b, x, input_norm, eps);
        let attention = self.attention(normed, layer, sources)?;
        let x = self.b.binary(BinOp::Add, x, attention);

        let post_norm = self.dense_f32(sources.post_attention_norm());
        let normed = rmsnorm(&self.b, x, post_norm, eps);
        let routed = self.routed_moe(normed, layer, sources)?;
        Ok(self.b.binary(BinOp::Add, x, routed))
    }

    /// Full-causal GQA with projection-wide Q/K normalization, leading-half RoPE, post-RoPE K, direct V, and
    /// additive capacity masking, on card 355's packed linear.
    fn attention(
        &mut self,
        normed: Traced,
        layer: usize,
        sources: &MiniMaxM2LayerSources,
    ) -> Result<Traced, MiniMaxM2TraceError> {
        let plan = self.plan;
        let config = &plan.config;
        let tokens = self.phase.tokens();
        let (head_dim, query_heads, kv_heads) = (
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        let eps = config.rms_norm_eps;

        let q = self.packed(normed, sources.attention(MiniMaxM2AttentionProjection::Q))?;
        let k = self.packed(normed, sources.attention(MiniMaxM2AttentionProjection::K))?;
        let v = self.packed(normed, sources.attention(MiniMaxM2AttentionProjection::V))?;

        // Projection-wide, before the head reshape; not interchangeable with per-head norm.
        let q_norm = self.dense_f32(sources.q_norm());
        let k_norm = self.dense_f32(sources.k_norm());
        let q = rmsnorm(&self.b, q, q_norm, eps);
        let k = rmsnorm(&self.b, k, k_norm, eps);

        let heads = |b: &Builder, x: Traced, count: usize| {
            b.transpose(
                b.reshape(x, vec![1, tokens, count, head_dim]),
                vec![0, 2, 1, 3],
            )
        };
        let q = heads(&self.b, q, query_heads);
        let k = heads(&self.b, k, kv_heads);
        let v = heads(&self.b, v, kv_heads);
        let (q, k) = self.rope_qk(q, k);

        let cache_type = TensorType::f32(vec![1, kv_heads, self.capacity, head_dim]);
        let k_cache = self.b.state_input(
            &cache_name(layer, "k_cache"),
            cache_type.clone(),
            StateRole::Recurrent,
        );
        let v_cache = self.b.state_input(
            &cache_name(layer, "v_cache"),
            cache_type,
            StateRole::Recurrent,
        );
        let (k_cache_out, v_cache_out) = match self.position {
            // Prefill fills rows `0..T` of the same fixed-capacity buffer the decode step will extend.
            None => (
                self.b.dynamic_update_slice(k_cache, k, 0, 2),
                self.b.dynamic_update_slice(v_cache, v, 0, 2),
            ),
            Some(position) => (
                self.b.dynamic_update_slice_dyn(k_cache, k, position, 2),
                self.b.dynamic_update_slice_dyn(v_cache, v, position, 2),
            ),
        };
        self.state.push((k_cache, k_cache_out));
        self.state.push((v_cache, v_cache_out));

        // `attention_masked` is shape-generic in the query axis, so both phases share it.
        let attention = attention_masked(
            &self.b,
            q,
            k_cache_out,
            v_cache_out,
            config.kv_groups(),
            config.attention_scale(),
            self.mask,
        );
        let attention = self.b.reshape(
            self.b.transpose(attention, vec![0, 2, 1, 3]),
            vec![tokens, config.q_dim()],
        );
        self.packed(
            attention,
            sources.attention(MiniMaxM2AttentionProjection::O),
        )
    }

    /// Leading-`rotary_dim` half-split RoPE on Q and K; the trailing values pass through unchanged.
    fn rope_qk(&self, q: Traced, k: Traced) -> (Traced, Traced) {
        let (cos, sin) = self.rope_tables;
        match (self.phase, self.position) {
            (MiniMaxM2Phase::Prefill { tokens }, _) => (
                rope_prefill(&self.b, q, cos, sin, tokens),
                rope_prefill(&self.b, k, cos, sin, tokens),
            ),
            (MiniMaxM2Phase::Decode, Some(position)) => (
                rope(&self.b, q, cos, sin, position),
                rope(&self.b, k, cos, sin, position),
            ),
            (MiniMaxM2Phase::Decode, None) => unreachable!("decode always binds a position slot"),
        }
    }

    /// The exact router plus routed packed experts: `sum_selected(weight_e * w2(silu(w1 x) * w3 x))`.
    fn routed_moe(
        &mut self,
        x: Traced,
        layer: usize,
        sources: &MiniMaxM2LayerSources,
    ) -> Result<Traced, MiniMaxM2TraceError> {
        let plan = self.plan;
        let config = &plan.config;
        let (hidden, top_k) = (config.hidden_size, config.num_experts_per_tok);
        let tokens = self.phase.tokens();
        let rows = tokens * top_k;

        let (ids, weights) = self.route(x, layer, sources)?;

        // Row `t * top_k + s` is `(token t, slot s)` in all three flattened tensors.
        let repeated = self.b.reshape(
            self.b.broadcast(
                self.b.reshape(x, vec![tokens, 1, hidden]),
                vec![tokens, top_k, hidden],
            ),
            vec![rows, hidden],
        );
        let ids = self.b.reshape(ids, vec![rows]);
        let weights = self.b.reshape(weights, vec![rows]);

        let gate = self.experts(repeated, ids, sources, MiniMaxM2ExpertProjection::W1)?;
        let up = self.experts(repeated, ids, sources, MiniMaxM2ExpertProjection::W3)?;
        let activated = swiglu(&self.b, gate, up);
        let down = self.experts(activated, ids, sources, MiniMaxM2ExpertProjection::W2)?;

        let scaled = self.b.binary(
            BinOp::Mul,
            down,
            self.b
                .broadcast(self.b.reshape(weights, vec![rows, 1]), vec![rows, hidden]),
        );
        let per_slot = self.b.reshape(scaled, vec![tokens, top_k, hidden]);
        Ok(self.b.reduce(RedOp::Sum, per_slot, 1, false))
    }

    /// One packed expert projection through card 369's composition: indexed for decode, grouped for prefill.
    fn experts(
        &self,
        x: Traced,
        ids: Traced,
        sources: &MiniMaxM2LayerSources,
        projection: MiniMaxM2ExpertProjection,
    ) -> Result<Traced, MiniMaxM2TraceError> {
        let rows = sources
            .experts(projection)
            .iter()
            .enumerate()
            .map(|(ordinal, source)| PackedLinearGraphRow {
                ordinal,
                linear_id: source.linear_id().to_string(),
                descriptor: source.descriptor(),
            })
            .collect::<Vec<_>>();
        let output = match self.route_slots {
            MiniMaxM2RouteSlots::Single => packed_indexed_linear(&self.b, x, ids, &rows)?,
            MiniMaxM2RouteSlots::Batched { .. } => packed_grouped_linear(&self.b, x, ids, &rows)?,
        };
        Ok(output)
    }

    /// The MiniMax router for `x[T, hidden]`, plus the three runtime witnesses card 375a observes.
    fn route(
        &mut self,
        x: Traced,
        layer: usize,
        sources: &MiniMaxM2LayerSources,
    ) -> Result<(Traced, Traced), MiniMaxM2TraceError> {
        let plan = self.plan;
        let experts = plan.config.num_local_experts;
        let top_k = plan.config.num_experts_per_tok;

        // `nn.Linear(hidden, experts)` stores `[experts, hidden]`, so the matmul reads the transpose.
        let gate = self.dense_f32(sources.router_gate());
        let logits = linear(&self.b, x, self.b.transpose(gate, vec![1, 0]), None);
        let witness = self.nonfinite_count(logits);
        self.declare(MiniMaxM2Validation::RouterLogitsFinite, layer, witness)?;

        let bias = self.dense_f32(sources.router_bias());
        let routes = minimax_m2_router_routes(&self.b, logits, bias, top_k);

        let witness = self.out_of_range_count(routes.ids, experts);
        self.declare(MiniMaxM2Validation::SelectorInRange, layer, witness)?;
        let witness = self.unusable_sum_count(routes.selected_sum);
        self.declare(MiniMaxM2Validation::SelectedSumUsable, layer, witness)?;

        let weights = self.slot_weights(routes.weights, routes.rank, routes.ids);
        Ok((routes.ids, weights))
    }

    /// The route weight of each `(token, slot)` pair, `[T, top_k]`.
    fn slot_weights(&self, weights: Traced, rank: Traced, ids: Traced) -> Traced {
        route_slot_weights(&self.b, self.route_slots, weights, rank, ids)
    }
    /// Final RMSNorm and the untied LM head, `[T, vocab]`.
    ///
    /// One mixed-dtype `MatMul` of the F32 activation by the BF16 weight, so no F32 copy of `[vocab, hidden]`
    /// exists. Prefill keeps every row so a test can compare prefill against stepwise decode at each position.
    fn head(&mut self, x: Traced) -> Traced {
        let plan = self.plan;
        let norm = self.dense_f32(plan.table.final_norm());
        let normed = rmsnorm(&self.b, x, norm, plan.config.rms_norm_eps);
        let weight = self.dense(plan.table.lm_head());
        linear(&self.b, normed, self.b.transpose(weight, vec![1, 0]), None)
    }

    fn declare(
        &mut self,
        family: MiniMaxM2Validation,
        layer: usize,
        witness: Traced,
    ) -> Result<(), MiniMaxM2TraceError> {
        self.validations
            .push((family.id(layer)?, family.name(layer), witness));
        Ok(())
    }

    /// Count the nonfinite values of `x`, reduced to one lane. `Ge(x, x)` is false exactly for NaN and
    /// `Ge(|x|, inf)` true exactly for infinities, so the sum is zero iff every value is finite. There is no
    /// `UnOp::Abs`; `Max(x, -x)` is the absolute value.
    fn nonfinite_count(&self, x: Traced) -> Traced {
        let b = &self.b;
        let nan = self.logical_not(b.binary(BinOp::Ge, x, x));
        let magnitude = b.binary(BinOp::Max, x, b.unary(UnOp::Neg, x));
        let infinite = b.binary_scalar(BinOp::Ge, magnitude, Scalar::F32(f32::INFINITY));
        self.sum_to_scalar(b.binary(BinOp::Add, nan, infinite))
    }

    /// Count the rows whose selected-score sum is not a finite positive number.
    ///
    /// `sigmoid` is `1 / (1 + exp(-x))`, bounded in `[0, 1]`, so the sum over `top_k` selected scores never
    /// reaches `+inf`. A finite logit below about `-88.7` overflows `exp(-x)` to `+inf` and scores exactly zero,
    /// so a row whose selected logits are all that negative sums to zero and normalizes to `0 / 0 = NaN`. The
    /// logit witness does not cover it (every logit was finite). Same finite-and-positive shape as card 364a's
    /// `deepseek4_unusable_sum_witness`.
    ///
    /// The finiteness half has no reachable MiniMax case; it is kept for symmetry and costs four equations a
    /// layer. `minimax_m2_zero_selected_scores_fail_instead_of_publishing_nan` exercises the positivity half.
    ///
    /// Comparisons keep the value on the left (the planner rejects a `Binary` with a literal left operand) and
    /// are false for `NaN`, so no separate NaN term is needed.
    fn unusable_sum_count(&self, sum: Traced) -> Traced {
        let b = &self.b;
        let negated = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(-1.0));
        // sum <= f32::MAX, false for +inf; sum >= f32::MIN, false for -inf; sum > 0, false at zero.
        let below_max = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(-f32::MAX));
        let above_min = b.binary_scalar(BinOp::Ge, sum, Scalar::F32(f32::MIN));
        let positive = self.logical_not(b.binary_scalar(BinOp::Ge, negated, Scalar::F32(0.0)));
        let usable = b.binary(
            BinOp::Mul,
            b.binary(BinOp::Mul, below_max, above_min),
            positive,
        );
        self.sum_to_scalar(self.logical_not(usable))
    }

    /// Count the selected ids outside `0..experts`. The ids stay F32 out of `ArgTopK`; bounding them with card
    /// 372c's guard would need an F32-to-I32 cast (card 371's surface, which MiniMax must not depend on), so
    /// this is an ordinary F32 witness rather than an in-graph clamp.
    fn out_of_range_count(&self, ids: Traced, experts: usize) -> Traced {
        let b = &self.b;
        let high = b.binary_scalar(BinOp::Ge, ids, Scalar::F32(experts as f32));
        let low = self.logical_not(b.binary_scalar(BinOp::Ge, ids, Scalar::F32(0.0)));
        self.sum_to_scalar(b.binary(BinOp::Add, high, low))
    }

    /// `1 - indicator`, for an indicator that is already 0.0 or 1.0.
    fn logical_not(&self, indicator: Traced) -> Traced {
        let negated = self
            .b
            .binary_scalar(BinOp::Mul, indicator, Scalar::F32(-1.0));
        self.b.binary_scalar(BinOp::Add, negated, Scalar::F32(1.0))
    }

    /// Reduce every axis away, so one witness costs exactly one validation lane.
    fn sum_to_scalar(&self, x: Traced) -> Traced {
        let mut value = x;
        while !self.b.aval(value).shape.is_empty() {
            value = self.b.reduce(RedOp::Sum, value, 0, false);
        }
        value
    }

    /// Stage one dense checkpoint source with its truthful source dtype.
    fn dense(&self, source: &MiniMaxM2DenseSource) -> Traced {
        let dtype = match source.role() {
            MiniMaxM2DenseRole::Bf16 => DType::BF16,
            MiniMaxM2DenseRole::F32 => DType::F32,
        };
        self.b.constant(
            source.name(),
            TensorType::new(source.shape().to_vec(), dtype),
        )
    }

    /// [`Self::dense`] widened to F32 for the arithmetic that has no mixed-dtype form.
    fn dense_f32(&self, source: &MiniMaxM2DenseSource) -> Traced {
        let value = self.dense(source);
        match source.role() {
            MiniMaxM2DenseRole::F32 => value,
            MiniMaxM2DenseRole::Bf16 => self.b.cast(value, DType::F32),
        }
    }

    /// One packed linear through card 355's shared composition. Card 379 owns the two constant names.
    fn packed(
        &self,
        x: Traced,
        source: &MiniMaxM2PackedSource,
    ) -> Result<Traced, MiniMaxM2TraceError> {
        Ok(packed_linear(
            &self.b,
            x,
            source.linear_id(),
            source.descriptor(),
            None,
            None,
        )?)
    }
}

#[cfg(test)]
fn cache_name(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.kv.{suffix}")
}

/// The route weight of each `(token, slot)` pair, `[T, top_k]`, from the dense `[T, experts]` weights. A free
/// function so both modes are testable without building a whole graph.
#[cfg(test)]
fn route_slot_weights(
    b: &Builder,
    slots: MiniMaxM2RouteSlots,
    weights: Traced,
    rank: Traced,
    ids: Traced,
) -> Traced {
    let [tokens, experts] = [b.aval(weights).shape[0], b.aval(weights).shape[1]];
    let top_k = b.aval(ids).shape[1];
    match slots {
        MiniMaxM2RouteSlots::Single => {
            // `Gather` shares one index set across every leading row, exact only for a single token; `Single`
            // is reachable only from the one-token decode phase.
            debug_assert_eq!(tokens, 1, "the Single route-slot form is one-token only");
            let ids = b.reshape(ids, vec![top_k]);
            b.reshape(b.gather(weights, 1, ids), vec![tokens, top_k])
        }
        MiniMaxM2RouteSlots::Batched { expert_iota, .. } => {
            // `slot_weights[t, r] = sum_e eq(rank[t, e], r) * weights[t, e]`, `eq(a, b) = ge(a, b) * ge(b, a)`,
            // exact because rank is integer valued. The `0..top_k` slot ordinals are the leading entries of the
            // expert iota the grouped helper already requires.
            let slots = b.slice(expert_iota, 0, 0, top_k);
            let spread = vec![tokens, top_k, experts];
            let rank_b = b.broadcast(b.reshape(rank, vec![tokens, 1, experts]), spread.clone());
            let slot_b = b.broadcast(b.reshape(slots, vec![1, top_k, 1]), spread.clone());
            let eq = b.binary(
                BinOp::Mul,
                b.binary(BinOp::Ge, rank_b, slot_b),
                b.binary(BinOp::Ge, slot_b, rank_b),
            );
            let weights_b = b.broadcast(b.reshape(weights, vec![tokens, 1, experts]), spread);
            b.reduce(RedOp::Sum, b.binary(BinOp::Mul, eq, weights_b), 2, false)
        }
    }
}

#[cfg(test)]
mod tests;
