#[cfg(test)]
use super::*;

/// Whether a traced V4 graph is one decode step or a whole-sequence prefill. The MoE composition is the
/// same either way; only the Card 369 helper it reaches differs (spec 364 FR-013).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(crate) enum V4MoePhase {
    /// One token. Card 369's `packed_indexed_linear`.
    Decode,
    /// A whole padded sequence. Card 369's `packed_grouped_linear`.
    Prefill,
}

/// Where one layer's expert ids come from. Derived from the layer index and
/// [`DeepseekV4Config::hash_router_layers`]; both rows compute and normalize the same score tensor, so the
/// table chooses ids, never weights.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum V4RouterKind {
    /// `ffn.gate.tid2eid[token_id, :]`, an exact I64-sourced I32 table (Card 371).
    Hash,
    /// Stable top-k of `raw + ffn.gate.bias`. The bias affects selection only.
    Score,
}

/// Per-layer inputs beyond activations, carried as one borrow to keep layer signatures short.
#[cfg(test)]
pub(crate) struct V4MoeContext<'a> {
    /// Card 372c bounds-guarded input token ids, `[T]` I32, shared by every hash-routed layer in the graph.
    pub(crate) tokens: Traced,
    pub(crate) phase: V4MoePhase,
    /// Card 375a declarations this trace has accumulated, in layer order.
    pub(crate) validations: &'a mut Vec<(ValidationId, String, Traced)>,
}

/// Card 375a validation ids this model declares: id 1 is the shared token-index bound, and each layer owns the
/// next two. [`deepseek4_router_validation_error`] is the only reader.
#[cfg(test)]
pub(crate) const V4_TOKEN_INDEX_VALIDATION: ValidationId = ValidationId(1);

/// Both readers assume `DeepseekV4Config::moe_dims` already proved `3 + 2 * layers` fits a `u32`.
#[cfg(test)]
pub(crate) fn v4_logit_validation(layer: usize) -> ValidationId {
    ValidationId(2 + 2 * layer as u32)
}

#[cfg(test)]
pub(crate) fn v4_score_sum_validation(layer: usize) -> ValidationId {
    ValidationId(3 + 2 * layer as u32)
}

/// The typed router failure behind one Card 375a validation packet lane. The nonfinite and nonpositive rows
/// are execution errors, not checkpoint-admission claims, and detection is not claimed to precede expert
/// contraction (spec 364 FR-016).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum DeepseekV4RouterError {
    #[error("DeepSeek-V4 input token id is outside the vocabulary")]
    TokenIndexOutOfRange,
    #[error("DeepSeek-V4 layer {layer} router logits are not all finite")]
    NonfiniteRouterLogits { layer: usize },
    #[error(
        "DeepSeek-V4 layer {layer} selected route scores do not sum to a finite positive value"
    )]
    UnusableSelectedScoreSum { layer: usize },
}

/// Map a Card 375a execution-validation failure back to the typed router error it stands for, or `None`
/// when the id is not one this model declares.
#[cfg(test)]
pub(crate) fn deepseek4_router_validation_error(
    failure: &ExecutionValidationFailure,
) -> Option<DeepseekV4RouterError> {
    let ValidationId(id) = failure.id;
    if id == V4_TOKEN_INDEX_VALIDATION.0 {
        return Some(DeepseekV4RouterError::TokenIndexOutOfRange);
    }
    let layer = usize::try_from(id.checked_sub(2)? / 2).ok()?;
    Some(if id.is_multiple_of(2) {
        DeepseekV4RouterError::NonfiniteRouterLogits { layer }
    } else {
        DeepseekV4RouterError::UnusableSelectedScoreSum { layer }
    })
}

/// Why a V4 MoE graph could not be built. Trace-time only; runtime router failures are
/// [`DeepseekV4RouterError`].
#[derive(Debug, thiserror::Error)]
#[cfg(test)]
pub(crate) enum DeepseekV4MoeError {
    #[error("DeepSeek-V4 MoE field {field} is {actual}, expected {requirement}")]
    Config {
        field: &'static str,
        actual: usize,
        requirement: &'static str,
    },
    #[error("DeepSeek-V4 token index guard over {vocab} ids: {source}")]
    TokenIndexGuard {
        vocab: usize,
        #[source]
        source: IndexGuardError,
    },
    /// Every source-plan failure, including an invalid packed descriptor (card 364b).
    #[error(transparent)]
    Source(#[from] V4SourcePlanError),
    #[error("DeepSeek-V4 layer {layer} graph: {source}")]
    LayerGraph {
        layer: usize,
        #[source]
        source: BuilderAppendError,
    },
    #[error("DeepSeek-V4 validation-bearing graph: {source}")]
    ValidationGraph {
        #[source]
        source: GraphValidationError,
    },
}

/// The validated MoE dims of one config.
#[derive(Clone, Copy, Debug)]
#[cfg(test)]
pub(crate) struct V4MoeDims {
    pub(crate) experts: usize,
    pub(crate) top_k: usize,
    pub(crate) route_scale: f32,
}

#[cfg(test)]
impl DeepseekV4Config {
    /// Check the MoE fields once, at the top of a trace, and hand back the dims the composition needs.
    #[cfg(test)]
    pub(crate) fn moe_dims(&self) -> Result<V4MoeDims, DeepseekV4MoeError> {
        let rows: [(&'static str, usize, &'static str, bool); 5] = [
            (
                "layers",
                self.layers,
                "few enough to index validations without overflow",
                u32::try_from(self.layers)
                    .ok()
                    .and_then(|n| n.checked_mul(2))
                    .and_then(|n| n.checked_add(3))
                    .is_some(),
            ),
            (
                "routed_experts",
                self.routed_experts,
                "at least one",
                self.routed_experts > 0,
            ),
            (
                "experts_per_tok",
                self.experts_per_tok,
                "between 1 and routed_experts",
                (1..=self.routed_experts).contains(&self.experts_per_tok),
            ),
            (
                "moe_intermediate",
                self.moe_intermediate,
                "at least one",
                self.moe_intermediate > 0,
            ),
            (
                "hash_router_layers",
                self.hash_router_layers,
                "at most layers",
                self.hash_router_layers <= self.layers,
            ),
        ];
        for (field, actual, requirement, ok) in rows {
            if !ok {
                return Err(DeepseekV4MoeError::Config {
                    field,
                    actual,
                    requirement,
                });
            }
        }
        Ok(V4MoeDims {
            experts: self.routed_experts,
            top_k: self.experts_per_tok,
            route_scale: self.route_scale,
        })
    }

    /// Which row layer `li` takes. Card 364c checks this partition against the checkpoint manifest.
    #[cfg(test)]
    pub(crate) fn router_kind(&self, li: usize) -> V4RouterKind {
        if li < self.hash_router_layers {
            V4RouterKind::Hash
        } else {
            V4RouterKind::Score
        }
    }
}

/// Bound the runtime token index against the vocabulary through Card 372c's canonical guard, once per graph
/// (spec 364 FR-021).
///
/// `tid2eid` values are proved in `0..routed_experts` at admission, but the index reading the table is a runtime
/// `Slot`, so an id outside `0..vocab` would gather past the table. Only `guard_index_bounds` is admitted here:
/// `poot-graph-plan` re-derives the shape from the graph and rejects a hand-written comparison or `Select`.
#[cfg(test)]
pub(crate) fn deepseek4_guard_token_ids(
    b: &Builder,
    tokens: Traced,
    vocab: usize,
) -> Result<(Traced, Traced), DeepseekV4MoeError> {
    let guarded = b
        .guard_index_bounds(tokens, vocab)
        .map_err(|source| DeepseekV4MoeError::TokenIndexGuard { vocab, source })?;
    Ok((guarded.guarded, guarded.witness))
}

/// One F32 lane counting the nonfinite lanes of `x`.
///
/// Reads [`canonical_router_scores`] backwards: it is the identity on every finite f32 and moves NaN and both
/// infinities, so "canonicalized value differs from the input" means "input was not finite".
#[cfg(test)]
pub(crate) fn deepseek4_nonfinite_witness(b: &Builder, x: Traced) -> Traced {
    let lanes = b.aval(x).numel();
    let canon = canonical_router_scores(b, x);
    let ge_cx = b.binary(BinOp::Ge, canon, x);
    let ge_xc = b.binary(BinOp::Ge, x, canon);
    let same = b.binary(BinOp::Mul, ge_cx, ge_xc);
    let differs = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, same, Scalar::F32(-1.0)),
        Scalar::F32(1.0),
    );
    b.reduce(RedOp::Sum, b.reshape(differs, vec![1, lanes]), 1, true)
}

/// One F32 lane counting the rows whose selected-score sum is not a finite positive number.
///
/// Finiteness is tested as well as positivity because the sum can reach `+inf` from finite inputs: `softplus`
/// is the naive `log(1 + exp(x))`, so `exp` overflows above `x ~ 88.7`, the expert's `raw` is `+inf`, the
/// weights are `inf / inf = NaN`, and a positivity-only witness would publish NaN (FR-016). The
/// nonfinite-logit witness does not cover this, since the logit itself was finite.
///
/// Every comparison has the value on the left, because the planner rejects a `Binary` with a literal left
/// operand; the upper bound is therefore expressed against the negated sum. Each comparison is false for NaN.
#[cfg(test)]
pub(crate) fn deepseek4_unusable_sum_witness(b: &Builder, sum: Traced) -> Traced {
    let lanes = b.aval(sum).numel();
    let negated = b.binary_scalar(BinOp::Mul, sum, Scalar::F32(-1.0));
    // sum <= f32::MAX, false for +inf; sum >= f32::MIN, false for -inf; sum > 0, false at zero.
    let below_max = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(-f32::MAX));
    let above_min = b.binary_scalar(BinOp::Ge, sum, Scalar::F32(f32::MIN));
    let nonpositive = b.binary_scalar(BinOp::Ge, negated, Scalar::F32(0.0));
    let positive = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, nonpositive, Scalar::F32(-1.0)),
        Scalar::F32(1.0),
    );
    let usable = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Mul, below_max, above_min),
        positive,
    );
    let unusable = b.binary_scalar(
        BinOp::Add,
        b.binary_scalar(BinOp::Mul, usable, Scalar::F32(-1.0)),
        Scalar::F32(1.0),
    );
    b.reduce(RedOp::Sum, b.reshape(unusable, vec![1, lanes]), 1, true)
}

/// Stage one plan row's constant at the checkpoint's own shape and dtype (spec 364b FR-005). Every V4 source
/// name reaches the graph through here or [`v4_packed_linear`].
#[cfg(test)]
pub(crate) fn v4_source(b: &Builder, spec: &V4DenseSourceSpec) -> Traced {
    b.constant(&spec.name, spec.tensor_type())
}

/// Stage one plan row and widen it to f32 for f32 arithmetic against a BF16 checkpoint row. An F32 row gets no
/// cast.
#[cfg(test)]
pub(crate) fn v4_source_f32(b: &Builder, spec: &V4DenseSourceSpec) -> Traced {
    let value = v4_source(b, spec);
    if spec.dtype == DType::F32 {
        value
    } else {
        b.cast(value, DType::F32)
    }
}

/// One `[out, in]` checkpoint weight consumed as a matmul RHS.
///
/// The transpose is an explicit equation (FR-005) and the weight keeps its own dtype through it, so a BF16 row
/// is never mirrored in f32 (`Transpose` on BF16 and `MatMul(F32, BF16)` are in Card 376's admitted set, and
/// Card 380 lowers the pair). This matters for `head.weight`, a `[vocab, hidden]` matrix.
#[cfg(test)]
pub(crate) fn v4_source_linear(b: &Builder, x: Traced, spec: &V4DenseSourceSpec) -> Traced {
    linear(b, x, b.transpose(v4_source(b, spec), vec![1, 0]), None)
}

/// Contract `x` against one packed plan row through Card 369's shared helper. Model code names a row and the
/// helper builds the canonical form Card 356 recognizes; it never emits `PackedDequant`, a weight concat, or
/// grouped sort/scatter structure by hand (spec 364 FR-013).
#[cfg(test)]
pub(crate) fn v4_packed_linear(
    b: &Builder,
    plan: &DeepseekV4SourcePlan,
    layer: usize,
    role: V4PackedRole,
    x: Traced,
) -> Result<Traced, DeepseekV4MoeError> {
    let row = plan.packed(layer, role)?;
    packed_linear(b, x, &row.linear_id, row.descriptor, None, None)
        .map_err(|source| DeepseekV4MoeError::LayerGraph { layer, source })
}

/// Contract `x` (`[o_groups, m, in_per_group]`) against `attn.wo_a`'s block-diagonal weight through Card 385's
/// `ops::packed_block_diagonal_linear`. `v4_packed_linear`'s canonical chain cannot reach it: its optional
/// reshape follows the transpose instead of preceding it.
#[cfg(test)]
pub(crate) fn v4_packed_block_diagonal_linear(
    b: &Builder,
    plan: &DeepseekV4SourcePlan,
    layer: usize,
    o_groups: usize,
    x: Traced,
) -> Result<Traced, DeepseekV4MoeError> {
    let row = plan.packed(layer, V4PackedRole::WoA)?;
    packed_block_diagonal_linear(b, x, &row.linear_id, row.descriptor, o_groups, None)
        .map_err(|source| DeepseekV4MoeError::LayerGraph { layer, source })
}

/// The schedule of a single-attention-kind tracer: every layer the same kind. Mixed schedules come from the
/// caller ([`parse_compress_ratios`]).
#[cfg(test)]
pub(crate) fn v4_uniform_schedule(cfg: &DeepseekV4Config, kind: V4LayerKind) -> Vec<V4LayerKind> {
    vec![kind; cfg.layers]
}

/// One trace's preamble: derive the source plan, check the MoE dims, bound the runtime token index once, and
/// open the Card 375a declaration list. Every V4 entry point shares it.
///
/// The guarded ids feed the embedding gather as well as the hash routers, so a V4 graph has one bound on the
/// token index.
#[cfg(test)]
pub(crate) fn deepseek4_moe_preamble(
    b: &Builder,
    cfg: &DeepseekV4Config,
    schedule: &[V4LayerKind],
    tokens: Traced,
    phase: V4MoePhase,
) -> Result<(DeepseekV4SourcePlan, V4MoeDims, Traced, V4MoeTrace), DeepseekV4MoeError> {
    let plan = DeepseekV4SourcePlan::new(cfg, schedule)?;
    let dims = cfg.moe_dims()?;
    let (guarded, witness) = deepseek4_guard_token_ids(b, tokens, cfg.vocab)?;
    let mut validations = Vec::with_capacity(1 + 2 * cfg.layers);
    validations.push((
        V4_TOKEN_INDEX_VALIDATION,
        "deepseek4.token_index".to_string(),
        witness,
    ));
    Ok((
        plan,
        dims,
        guarded,
        V4MoeTrace {
            tokens: guarded,
            phase,
            validations,
        },
    ))
}

/// The MoE state one trace accumulates: shared guarded token ids, the phase, and the declarations so far.
#[cfg(test)]
pub(crate) struct V4MoeTrace {
    pub(crate) tokens: Traced,
    pub(crate) phase: V4MoePhase,
    pub(crate) validations: Vec<(ValidationId, String, Traced)>,
}

#[cfg(test)]
impl V4MoeTrace {
    #[cfg(test)]
    pub(crate) fn context(&mut self) -> V4MoeContext<'_> {
        V4MoeContext {
            tokens: self.tokens,
            phase: self.phase,
            validations: &mut self.validations,
        }
    }

    /// Close the graph with every accumulated declaration, in layer order. The validating finish also checks the
    /// Card 375a packet layout, so a bad witness (not a statically shaped nonzero F32 value) or a declaration
    /// set over the packet byte cap fails here rather than at execution.
    pub(crate) fn finish(
        self,
        b: Builder,
        out: Traced,
        state: &[(Traced, Traced)],
    ) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
        let declared: Vec<(ValidationId, &str, Traced)> = self
            .validations
            .iter()
            .map(|(id, name, value)| (*id, name.as_str(), *value))
            .collect();
        poot_test_util::graph_fixtures::finish_with_state_and_validations(b, out, state, &declared)
            .map_err(|source| DeepseekV4MoeError::ValidationGraph { source })
    }
}

/// Layer `li`'s hash-routed expert ids, `[T, k]` as integer-valued f32: row `tokens[i]` of the exact-I32
/// `ffn.gate.tid2eid` table. The one place a hash id is chosen, shared by [`deepseek4_routed_moe_ffn`] and
/// the [`trace_deepseek4_hash_router_ids`] observation (card 452), so a change here reaches both.
#[cfg(test)]
pub(crate) fn deepseek4_hash_router_ids(
    b: &Builder,
    plan: &DeepseekV4SourcePlan,
    li: usize,
    tokens: Traced,
) -> Result<Traced, DeepseekV4MoeError> {
    let selection = plan.router_selection(li)?;
    let table =
        poot_test_util::graph_fixtures::i32_constant(b, &selection.name, selection.shape.clone())
            .map_err(|source| DeepseekV4MoeError::LayerGraph { layer: li, source })?;
    Ok(b.cast(b.gather(table, 0, tokens), DType::F32))
}

/// Card 452 observation: a graph whose output is the expert ids hash layer `layer` selects for `t` input
/// tokens, `[t, experts_per_tok]` as integer-valued f32.
///
/// It reads no activation, so no embedding or residual signal can reach the output. It goes through the same
/// preamble (guarded token ids) and the same [`deepseek4_hash_router_ids`] as the production MoE block, and
/// names the same `layers.<layer>.ffn.gate.tid2eid` constant. It does not change the model.
#[cfg(test)]
pub(crate) fn trace_deepseek4_hash_router_ids(
    cfg: DeepseekV4Config,
    layer: usize,
    t: usize,
) -> Result<Graph<ValidationOutputs>, DeepseekV4MoeError> {
    if layer >= cfg.hash_router_layers {
        return Err(DeepseekV4MoeError::Config {
            field: "layer",
            actual: layer,
            requirement: "a hash-routed layer (below hash_router_layers)",
        });
    }
    let b = Builder::new();
    let tokens = b.slot(Slot::Token, TensorType::new(vec![t], DType::I32));
    let schedule = v4_uniform_schedule(&cfg, V4LayerKind::Sliding);
    let (plan, _dims, guarded, moe) =
        deepseek4_moe_preamble(&b, &cfg, &schedule, tokens, V4MoePhase::Prefill)?;
    let ids = deepseek4_hash_router_ids(&b, &plan, layer, guarded)?;
    moe.finish(b, ids, &[])
}

/// The routed-plus-shared V4 MoE block: the FFN of every Sliding, CSA, and HCA layer, in both phases (spec 364
/// FR-002, FR-005). `x` is the post-attention-normalized `[1, T, hidden]` stream; the result has the same shape.
///
/// The two layer rows share one composition: `raw = sqrt(softplus(logits))` in f32, `raw` extracted at the
/// selected ids, normalized over exactly those ids, scaled by `route_scale`. Only the id source differs, so a
/// correction bias reaches selection and never a weight.
///
/// Score extraction is a one-hot reduction over the expert axis, not a take-along-axis gather, because
/// `Gather` shares one index set across every leading row. See the comment at the id source for the cost.
#[cfg(test)]
pub(crate) fn deepseek4_routed_moe_ffn(
    b: &Builder,
    cfg: &DeepseekV4Config,
    plan: &DeepseekV4SourcePlan,
    dims: V4MoeDims,
    li: usize,
    x: Traced,
    moe: &mut V4MoeContext<'_>,
) -> Result<Traced, DeepseekV4MoeError> {
    let h = cfg.hidden;
    let V4MoeDims {
        experts: e,
        top_k: k,
        route_scale,
    } = dims;
    let t = b.aval(x).shape[1];
    let m = t * k;
    let xm = b.reshape(x, vec![t, h]);

    // Router scores: the BF16 `ffn.gate.weight` row is `[routed_experts, hidden]`, transposed in the graph; the
    // product is f32. The transform is neither sigmoid nor softmax.
    let logits = v4_source_linear(b, xm, plan.layer_dense(li, V4LayerDense::RouterGate)?);
    let raw = b.unary(UnOp::Sqrt, softplus(b, logits));
    moe.validations.push((
        v4_logit_validation(li),
        format!("deepseek4.layer{li}.router_logits_finite"),
        deepseek4_nonfinite_witness(b, logits),
    ));

    // Expert ids, `[T, k]` as integer-valued f32 either way.
    //
    // The f32 id has two consumers, the packed selector and the score extraction below, and `poot-graph-plan`'s
    // exact-I32 authorization admits only the first, so this graph is CPU-only. Card 383 added the I32 reshape
    // and offset arithmetic a flat-gather form needs: compute `row * E + id` while ids are I32 and cast only the
    // selector (card 364b).
    let selection = plan.router_selection(li)?;
    let ids = match cfg.router_kind(li) {
        V4RouterKind::Hash => deepseek4_hash_router_ids(b, plan, li, moe.tokens)?,
        V4RouterKind::Score => {
            let bias = v4_source_f32(b, selection);
            let biased = b.binary(
                BinOp::Add,
                raw,
                b.broadcast(b.reshape(bias, vec![1, e]), vec![t, e]),
            );
            b.arg_top_k(stable_descending_rank(b, biased), k)
        }
    };

    // Selected-only normalization of the UNBIASED scores, then the fixed route scale. The per-row read is a
    // one-hot over the expert axis using `eq(a,b) = ge(a,b) * ge(b,a)` (as `moe_grouped_prep`), exact because
    // both operands are integer-valued f32.
    // Card 558a: the `0..E` row is computed with `iota`, not a bound named-constant range.
    let iota_e = b.iota(e);
    let ids_grid = b.broadcast(b.reshape(ids, vec![t, k, 1]), vec![t, k, e]);
    let iota_grid = b.broadcast(b.reshape(iota_e, vec![1, 1, e]), vec![t, k, e]);
    let one_hot = b.binary(
        BinOp::Mul,
        b.binary(BinOp::Ge, ids_grid, iota_grid),
        b.binary(BinOp::Ge, iota_grid, ids_grid),
    );
    let raw_grid = b.broadcast(b.reshape(raw, vec![t, 1, e]), vec![t, k, e]);
    let selected = b.reduce(
        RedOp::Sum,
        b.binary(BinOp::Mul, one_hot, raw_grid),
        2,
        false,
    );
    let sum = b.reduce(RedOp::Sum, selected, 1, true);
    moe.validations.push((
        v4_score_sum_validation(li),
        format!("deepseek4.layer{li}.router_score_sum"),
        deepseek4_unusable_sum_witness(b, sum),
    ));
    let weights = b.binary_scalar(
        BinOp::Mul,
        b.binary(BinOp::Div, selected, sum),
        Scalar::F32(route_scale),
    );

    // The routed experts. One `(token, slot)` row per selection, so `M = T * k`.
    let selector = b.reshape(ids, vec![m]);
    let x_rows = b.reshape(
        b.broadcast(b.reshape(xm, vec![t, 1, h]), vec![t, k, h]),
        vec![m, h],
    );
    let routed_linear = |act: Traced, rows: &[PackedLinearGraphRow]| {
        match moe.phase {
            // `packed_grouped_linear` computes its own row/expert iotas with `iota` (card 558a).
            V4MoePhase::Prefill => packed_grouped_linear(b, act, selector, rows),
            V4MoePhase::Decode => packed_indexed_linear(b, act, selector, rows),
        }
        .map_err(|source| DeepseekV4MoeError::LayerGraph { layer: li, source })
    };
    let gate = routed_linear(x_rows, &plan.routed_rows(li, V4ExpertProjection::W1)?)?;
    let up = routed_linear(x_rows, &plan.routed_rows(li, V4ExpertProjection::W3)?)?;
    let activated = deepseek4_clamped_swiglu(b, gate, up, cfg.swiglu_limit);
    let down = routed_linear(activated, &plan.routed_rows(li, V4ExpertProjection::W2)?)?;

    // Weight by the route, then reduce over the k slots (axis 1).
    let weighted = b.binary(BinOp::Mul, down, b.reshape(weights, vec![m, 1]));
    let routed = b.reduce(RedOp::Sum, b.reshape(weighted, vec![t, k, h]), 1, false);

    // The shared expert: the same V4 clamp rules, added once and never route-weighted.
    let shared = |act: Traced, projection: V4ExpertProjection| {
        v4_packed_linear(b, plan, li, V4PackedRole::SharedExpert(projection), act)
    };
    let shared_gate = shared(xm, V4ExpertProjection::W1)?;
    let shared_upv = shared(xm, V4ExpertProjection::W3)?;
    let shared_act = deepseek4_clamped_swiglu(b, shared_gate, shared_upv, cfg.swiglu_limit);
    let shared = shared(shared_act, V4ExpertProjection::W2)?;

    Ok(b.reshape(b.binary(BinOp::Add, routed, shared), vec![1, t, h]))
}

/// First stage of the grouped low-rank output projection (`DeepseekV4GroupedLinear`, FR-005): `o_groups`
/// block-diagonal matmuls (`num_heads*head_dim/o_groups -> o_lora_rank`) through Card 385's
/// `ops::packed_block_diagonal_linear`. `attn` is `[1, l, num_heads, head_dim]` (post attention, post
/// derotation); the result is `[1, l, o_groups * o_lora_rank]`. The second stage is one linear to `hidden`
/// against the E4M3 packed `attn.wo_b`, which the caller contracts through Card 369's helper.
///
/// The grouped `[o_groups, in_per_group, o_lora_rank]` weight view is not a plain 2D transpose of the
/// checkpoint's `[o_groups * o_lora_rank, in_per_group]` storage: it needs a reshape to
/// `[o_groups, o_lora_rank, in_per_group]` and then an axis swap, and reshaping straight into the grouped
/// extents silently transposes two axes (see the grouped output-projection derivation).
/// `ops::packed_block_diagonal_linear` reshapes before its transpose to avoid this (Card 385).
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn deepseek4_grouped_out_a(
    b: &Builder,
    plan: &DeepseekV4SourcePlan,
    layer: usize,
    attn: Traced,
    l: usize,
    num_heads: usize,
    head_dim: usize,
    o_groups: usize,
    o_lora_rank: usize,
) -> Result<Traced, DeepseekV4MoeError> {
    let in_per_group = (num_heads * head_dim) / o_groups;
    let flat_tokens = b.reshape(attn, vec![l, num_heads * head_dim]);
    let grouped = b.reshape(flat_tokens, vec![l, o_groups, in_per_group]);
    let grouped_t = b.transpose(grouped, vec![1, 0, 2]); // [o_groups, l, in_per_group]
    let y = v4_packed_block_diagonal_linear(b, plan, layer, o_groups, grouped_t)?; // [o_groups, l, o_lora_rank]
    let y_t = b.transpose(y, vec![1, 0, 2]); // [l, o_groups, o_lora_rank]
    Ok(b.reshape(y_t, vec![1, l, o_groups * o_lora_rank]))
}
