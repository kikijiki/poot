use super::*;
use poot_graph_ir::op::{ScaleEncoding, WeightFormat};

/// The Qwen3.8-Flash-Next FFN block: `routed_experts_sum + sigmoid(shared_expert_gate(x)) *
/// shared_expert(x)`, the `Qwen4ExpTextSparseMoeBlock.forward` composition (`modeling_qwen4_exp.py`
/// lines 993-1010). The dense synthetic source and Card 362's split packed source both pass through
/// [`qwen38_routed_ffn`]: softmax over all experts, stable top-k, renormalize, per-expert SwiGLU,
/// weighted reduction, and the gated shared expert are one composition. The dense `[E, H, 2*I]`/
/// `[E, I, H]` convention matches the real `Qwen4ExpTextTopKRouter`/`Qwen4ExpTextExperts` forward: softmax
/// over all `num_experts`, top-k, `norm_topk_prob=True` renormalize (the real default,
/// `configuration_qwen4_exp.py:163`; `config.json` leaves it unset), and
/// `gate, up = linear(x, gate_up_proj[e]).chunk(2, dim=-1)` (gate rows first, up rows second, matching
/// `w_in`'s `[E, H, 2*I]`). Shared by [`trace_qwen38_prefill`] and [`trace_qwen38_decode`]; the MoE FFN
/// is stateless, so both call the same composition.
///
/// `x` is the already-normed FFN input, `[1, L, H]`. `p` is the layer's tensor-name prefix
/// (`"layers.{li}"`). Declares these per-checkpoint tensor names, all in-major (pre-transposed from the
/// checkpoint's PyTorch out-major `[out, in]`, a loader concern):
///
/// - `{p}.mlp.gate.weight` -> `router_w [H, E]` (real `Qwen4ExpTextTopKRouter.weight`: `[E, H]`).
/// - `{p}.mlp.experts.gate_up_proj` -> `w_in [E, H, 2*I]` (real `Qwen4ExpTextExperts.gate_up_proj`:
///   `[E, 2*I, H]`, BF16 `[512, 1280, 2560]` per the `model-00002-of-00131.safetensors` header; see spec 282).
/// - `{p}.mlp.experts.down_proj` -> `w_out [E, I, H]` (real `Qwen4ExpTextExperts.down_proj`: `[E, H, I]`,
///   BF16 `[512, 2560, 640]` per `model-00003-of-00131.safetensors`).
/// - `{p}.mlp.shared_expert.{gate,up}_proj.weight` -> `[H, I_shexp]` (real `[I_shexp, H]`, BF16 `[640, 2560]`).
/// - `{p}.mlp.shared_expert.down_proj.weight` -> `[I_shexp, H]` (real `[H, I_shexp]`, BF16 `[2560, 640]`).
/// - `{p}.mlp.shared_expert_gate.weight` -> `[H, 1]` (real `[1, H]`, BF16 `[1, 2560]`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_moe_ffn(
    b: &Builder,
    x: Traced,
    p: &str,
    h: usize,
    n_experts: usize,
    top_k: usize,
    inter: usize,
    shexp_inter: usize,
) -> Traced {
    let router_w = b.constant(
        &format!("{p}.mlp.gate.weight"),
        TensorType::f32(vec![h, n_experts]),
    );
    let w_in = b.constant(
        &format!("{p}.mlp.experts.gate_up_proj"),
        TensorType::f32(vec![n_experts, h, 2 * inter]),
    );
    let w_out = b.constant(
        &format!("{p}.mlp.experts.down_proj"),
        TensorType::f32(vec![n_experts, inter, h]),
    );
    let sg = b.constant(
        &format!("{p}.mlp.shared_expert.gate_proj.weight"),
        TensorType::f32(vec![h, shexp_inter]),
    );
    let su = b.constant(
        &format!("{p}.mlp.shared_expert.up_proj.weight"),
        TensorType::f32(vec![h, shexp_inter]),
    );
    let sd = b.constant(
        &format!("{p}.mlp.shared_expert.down_proj.weight"),
        TensorType::f32(vec![shexp_inter, h]),
    );
    let sgi = b.constant(
        &format!("{p}.mlp.shared_expert_gate.weight"),
        TensorType::f32(vec![h, 1]),
    );
    qwen38_routed_ffn(
        b,
        x,
        router_w,
        sg,
        su,
        sd,
        sgi,
        n_experts,
        top_k,
        inter,
        Qwen4ExpRoutedExpertSource::Dense { w_in, w_out },
    )
    .expect("the dense Qwen4Exp routed source is infallible")
}

/// One projection in the official split routed-expert checkpoint namespace.
///
/// The synthetic whole-model tracer consumes fused three-dimensional tensors; the official FP8
/// repository stores these three projections per expert. One role table keeps name and shape derivation
/// exhaustive without parallel hand-written branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qwen4ExpExpertProjection {
    Gate,
    Up,
    Down,
}

impl Qwen4ExpExpertProjection {
    pub const ALL: [Self; 3] = [Self::Gate, Self::Up, Self::Down];

    pub(crate) fn ordinal(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }

    pub fn checkpoint_component(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    /// Return the official PyTorch `[out, in]` dimensions.
    pub fn out_in(self, hidden: usize, intermediate: usize) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (intermediate, hidden),
            Self::Down => (hidden, intermediate),
        }
    }

    pub fn checkpoint_prefix(self, layer: usize, expert: usize) -> String {
        format!(
            "model.language_model.layers.{layer}.mlp.experts.{expert}.{}",
            self.checkpoint_component()
        )
    }

    #[cfg(test)]
    pub(crate) fn graph_prefix(self, layer: usize, expert: usize) -> String {
        format!(
            "layers.{layer}.mlp.experts.{expert}.{}",
            self.checkpoint_component()
        )
    }
}

/// One role-tagged expert table consumed by the Card 369 packed graph builders. Rows carry only semantic
/// ids and descriptors; packed payload ownership stays outside the graph in Card 359's
/// `LoadedPackedLinear` values, connected later by the source constant names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ExpPackedProjectionTable {
    pub(crate) role: Qwen4ExpExpertProjection,
    pub(crate) rows: Vec<PackedLinearGraphRow>,
}

impl Qwen4ExpPackedProjectionTable {
    pub fn new(role: Qwen4ExpExpertProjection, rows: Vec<PackedLinearGraphRow>) -> Self {
        Self { role, rows }
    }

    pub const fn role(&self) -> Qwen4ExpExpertProjection {
        self.role
    }

    pub fn rows(&self) -> &[PackedLinearGraphRow] {
        &self.rows
    }
}

/// Validated gate/up/down tables for one Qwen4Exp layer. Construction is fallible and validates all
/// three roles before graph mutation, so a role, expert, authenticated linear-id, or descriptor mismatch
/// cannot leave a partially appended routed block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4ExpPackedExpertTables {
    pub(crate) tables: Vec<Qwen4ExpPackedProjectionTable>,
    pub(crate) layer: usize,
    pub(crate) n_experts: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen4ExpPackedExpertTableError {
    #[error("Qwen4Exp packed expert source has zero {field}")]
    ZeroDimension { field: &'static str },
    #[error("Qwen4Exp packed expert source has {actual} projection tables, expected {expected}")]
    ProjectionCount { actual: usize, expected: usize },
    #[error(
        "Qwen4Exp packed projection table {ordinal} has role {actual:?}, expected {expected:?}"
    )]
    ProjectionRole {
        ordinal: usize,
        actual: Qwen4ExpExpertProjection,
        expected: Qwen4ExpExpertProjection,
    },
    #[error("Qwen4Exp packed {role:?} table has {actual} experts, expected {expected}")]
    ExpertCount {
        role: Qwen4ExpExpertProjection,
        actual: usize,
        expected: usize,
    },
    #[error("Qwen4Exp packed {role:?} row {index} has ordinal {actual}, expected {expected}")]
    ExpertOrdinal {
        role: Qwen4ExpExpertProjection,
        index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("Qwen4Exp packed {role:?} row {expert} has an empty linear id")]
    EmptyLinearId {
        role: Qwen4ExpExpertProjection,
        expert: usize,
    },
    #[error(
        "Qwen4Exp packed {role:?} row {expert} has linear id {actual:?}, expected {expected:?}"
    )]
    LinearId {
        role: Qwen4ExpExpertProjection,
        expert: usize,
        actual: String,
        expected: String,
    },
    #[error("Qwen4Exp packed {role:?} row {expert} uses format {actual:?}, expected {expected:?}")]
    Format {
        role: Qwen4ExpExpertProjection,
        expert: usize,
        actual: WeightFormat,
        expected: WeightFormat,
    },
    #[error(
        "Qwen4Exp packed {role:?} row {expert} has logical shape {actual:?}, expected {expected:?}"
    )]
    LogicalShape {
        role: Qwen4ExpExpertProjection,
        expert: usize,
        actual: [usize; 2],
        expected: [usize; 2],
    },
}

impl Qwen4ExpPackedExpertTables {
    pub fn try_new(
        tables: Vec<Qwen4ExpPackedProjectionTable>,
        layer: usize,
        n_experts: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Qwen4ExpPackedExpertTableError> {
        for (field, value) in [
            ("expert count", n_experts),
            ("hidden size", hidden),
            ("intermediate size", intermediate),
        ] {
            if value == 0 {
                return Err(Qwen4ExpPackedExpertTableError::ZeroDimension { field });
            }
        }
        if tables.len() != Qwen4ExpExpertProjection::ALL.len() {
            return Err(Qwen4ExpPackedExpertTableError::ProjectionCount {
                actual: tables.len(),
                expected: Qwen4ExpExpertProjection::ALL.len(),
            });
        }

        for (ordinal, (&expected_role, table)) in Qwen4ExpExpertProjection::ALL
            .iter()
            .zip(tables.iter())
            .enumerate()
        {
            if table.role != expected_role {
                return Err(Qwen4ExpPackedExpertTableError::ProjectionRole {
                    ordinal,
                    actual: table.role,
                    expected: expected_role,
                });
            }
            if table.rows.len() != n_experts {
                return Err(Qwen4ExpPackedExpertTableError::ExpertCount {
                    role: table.role,
                    actual: table.rows.len(),
                    expected: n_experts,
                });
            }
            let (out, k) = table.role.out_in(hidden, intermediate);
            let expected_shape = [out, k];
            for (expert, row) in table.rows.iter().enumerate() {
                if row.ordinal != expert {
                    return Err(Qwen4ExpPackedExpertTableError::ExpertOrdinal {
                        role: table.role,
                        index: expert,
                        actual: row.ordinal,
                        expected: expert,
                    });
                }
                if row.linear_id.is_empty() {
                    return Err(Qwen4ExpPackedExpertTableError::EmptyLinearId {
                        role: table.role,
                        expert,
                    });
                }
                let expected_linear_id = table.role.checkpoint_prefix(layer, expert);
                if row.linear_id != expected_linear_id {
                    return Err(Qwen4ExpPackedExpertTableError::LinearId {
                        role: table.role,
                        expert,
                        actual: row.linear_id.clone(),
                        expected: expected_linear_id,
                    });
                }
                let actual_format = row.descriptor.format();
                let expected_format = WeightFormat::E4m3Block128 {
                    scale: ScaleEncoding::Bf16,
                };
                if actual_format != expected_format {
                    return Err(Qwen4ExpPackedExpertTableError::Format {
                        role: table.role,
                        expert,
                        actual: actual_format,
                        expected: expected_format,
                    });
                }
                let actual_shape = row.descriptor.shape();
                if actual_shape != expected_shape {
                    return Err(Qwen4ExpPackedExpertTableError::LogicalShape {
                        role: table.role,
                        expert,
                        actual: actual_shape,
                        expected: expected_shape,
                    });
                }
            }
        }

        Ok(Self {
            tables,
            layer,
            n_experts,
            hidden,
            intermediate,
        })
    }

    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn n_experts(&self) -> usize {
        self.n_experts
    }

    pub const fn hidden(&self) -> usize {
        self.hidden
    }

    pub const fn intermediate(&self) -> usize {
        self.intermediate
    }

    pub fn projection(&self, role: Qwen4ExpExpertProjection) -> &Qwen4ExpPackedProjectionTable {
        &self.tables[role.ordinal()]
    }
}

pub(crate) enum Qwen4ExpRoutedExpertSource<'a> {
    Dense {
        w_in: Traced,
        w_out: Traced,
    },
    Packed {
        tables: &'a Qwen4ExpPackedExpertTables,
        mode: PackedRoutingMode,
    },
}

/// How a packed routed projection selects its runtime expert.
///
/// Card 369 has two canonical compositions: decode reads one row per activation
/// ([`packed_indexed_linear`]); prefill sorts rows by expert first (`packed_grouped_linear`), which now
/// computes its row-tie-break and expert-offset iotas internally with `iota` (card 558a). This is the
/// decode/prefill split `deepseek4` and `minimax_m2` use for their packed routed experts.
#[derive(Clone, Copy)]
pub(crate) enum PackedRoutingMode {
    Indexed,
    Grouped,
}

pub(crate) fn qwen38_packed_projection(
    b: &Builder,
    mode: PackedRoutingMode,
    x: Traced,
    selector: Traced,
    rows: &[PackedLinearGraphRow],
) -> Result<Traced, BuilderAppendError> {
    match mode {
        PackedRoutingMode::Indexed => packed_indexed_linear(b, x, selector, rows),
        PackedRoutingMode::Grouped => packed_grouped_linear(b, x, selector, rows),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qwen38_routed_ffn(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    shexp_gate_w: Traced,
    shexp_up_w: Traced,
    shexp_down_w: Traced,
    shexp_gin_w: Traced,
    n_experts: usize,
    top_k: usize,
    inter: usize,
    source: Qwen4ExpRoutedExpertSource<'_>,
) -> Result<Traced, BuilderAppendError> {
    let shape = b.aval(x).shape;
    let l = shape[shape.len() - 2];
    let h = shape[shape.len() - 1];
    let xm = b.reshape(x, vec![l, h]);
    let logits = linear(b, xm, router_w, None);
    let (x_flat, expert_ids, expert_weights) = moe_grouped_prep(b, xm, logits, n_experts, top_k);

    let (gate, up, down_source) = match source {
        Qwen4ExpRoutedExpertSource::Dense { w_in, w_out } => {
            let gate_up = b.indexed_matmul(x_flat, w_in, expert_ids);
            (
                b.slice(gate_up, 1, 0, inter),
                b.slice(gate_up, 1, inter, 2 * inter),
                Qwen4ExpDownProjection::Dense(w_out),
            )
        }
        Qwen4ExpRoutedExpertSource::Packed { tables, mode } => {
            let gate = qwen38_packed_projection(
                b,
                mode,
                x_flat,
                expert_ids,
                tables.projection(Qwen4ExpExpertProjection::Gate).rows(),
            )?;
            let up = qwen38_packed_projection(
                b,
                mode,
                x_flat,
                expert_ids,
                tables.projection(Qwen4ExpExpertProjection::Up).rows(),
            )?;
            (gate, up, Qwen4ExpDownProjection::Packed { tables, mode })
        }
    };
    let activated = swiglu(b, gate, up);
    let expert_output = match down_source {
        Qwen4ExpDownProjection::Dense(w_out) => b.indexed_matmul(activated, w_out, expert_ids),
        Qwen4ExpDownProjection::Packed { tables, mode } => qwen38_packed_projection(
            b,
            mode,
            activated,
            expert_ids,
            tables.projection(Qwen4ExpExpertProjection::Down).rows(),
        )?,
    };

    let m = l * top_k;
    let weighted = b.binary(
        BinOp::Mul,
        expert_output,
        b.reshape(expert_weights, vec![m, 1]),
    );
    let weighted = b.reshape(weighted, vec![l, top_k, h]);
    let weighted = b.transpose(weighted, vec![0, 2, 1]);
    let routed = b.reduce(RedOp::Sum, weighted, 2, false);
    let routed = b.reshape(routed, vec![1, l, h]);

    let shared_gate = linear(b, xm, shexp_gate_w, None);
    let shared_up = linear(b, xm, shexp_up_w, None);
    let shared = swiglu(b, shared_gate, shared_up);
    let shared = linear(b, shared, shexp_down_w, None);
    let shared = b.reshape(shared, vec![1, l, h]);
    let shared_weight = sigmoid(b, linear(b, xm, shexp_gin_w, None));
    let shared_weight = b.reshape(shared_weight, vec![1, l, 1]);
    let shared = b.binary(BinOp::Mul, shared_weight, shared);
    Ok(b.binary(BinOp::Add, routed, shared))
}

pub(crate) enum Qwen4ExpDownProjection<'a> {
    Dense(Traced),
    Packed {
        tables: &'a Qwen4ExpPackedExpertTables,
        mode: PackedRoutingMode,
    },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Qwen4ExpPackedRoutedError {
    #[error("Qwen4Exp packed routed input {field} references unknown value v{value}")]
    UnknownInput { field: &'static str, value: usize },
    #[error("Qwen4Exp packed routed input {field} has dtype {actual:?}, expected {expected:?}")]
    InputDType {
        field: &'static str,
        actual: DType,
        expected: DType,
    },
    #[error("Qwen4Exp packed routed input {field} has rank {actual}, expected {expected}")]
    InputRank {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("Qwen4Exp packed routed input {field} axis {axis} is {actual}, expected {expected}")]
    InputDimension {
        field: &'static str,
        axis: usize,
        actual: usize,
        expected: usize,
    },
    #[error("Qwen4Exp packed routed input {field} has a zero dimension at axis {axis}")]
    ZeroInputDimension { field: &'static str, axis: usize },
    #[error("Qwen4Exp packed routed top-k is {actual}, expected 1..={experts}")]
    TopK { actual: usize, experts: usize },
    #[error("Qwen4Exp packed routed {field} exceeds the exact f32 integer limit {max}: {actual}")]
    F32ExactIntegerRange {
        field: &'static str,
        actual: usize,
        max: usize,
    },
    #[error("Qwen4Exp packed routed arithmetic overflowed {field}")]
    ArithmeticOverflow { field: &'static str },
    #[error(transparent)]
    Graph(#[from] BuilderAppendError),
}

pub(crate) fn qwen38_traced_type(
    plan: &BuilderAppendPlan,
    field: &'static str,
    value: Traced,
) -> Result<TensorType, Qwen4ExpPackedRoutedError> {
    plan.type_of(value.id)
        .cloned()
        .ok_or(Qwen4ExpPackedRoutedError::UnknownInput {
            field,
            value: value.id,
        })
}

pub(crate) fn qwen38_require_dtype(
    field: &'static str,
    actual: DType,
) -> Result<(), Qwen4ExpPackedRoutedError> {
    if actual != DType::F32 {
        return Err(Qwen4ExpPackedRoutedError::InputDType {
            field,
            actual,
            expected: DType::F32,
        });
    }
    Ok(())
}

pub(crate) fn qwen38_require_rank(
    field: &'static str,
    shape: &[usize],
    expected: usize,
) -> Result<(), Qwen4ExpPackedRoutedError> {
    if shape.len() != expected {
        return Err(Qwen4ExpPackedRoutedError::InputRank {
            field,
            actual: shape.len(),
            expected,
        });
    }
    Ok(())
}

pub(crate) fn qwen38_require_dimension(
    field: &'static str,
    shape: &[usize],
    axis: usize,
    expected: usize,
) -> Result<(), Qwen4ExpPackedRoutedError> {
    let actual = shape[axis];
    if actual != expected {
        return Err(Qwen4ExpPackedRoutedError::InputDimension {
            field,
            axis,
            actual,
            expected,
        });
    }
    Ok(())
}

pub(crate) fn qwen38_checked_product(
    field: &'static str,
    factors: &[usize],
) -> Result<usize, Qwen4ExpPackedRoutedError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(Qwen4ExpPackedRoutedError::ArithmeticOverflow { field })
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn preflight_qwen38_packed_moe_ffn(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    shexp_gate_w: Traced,
    shexp_up_w: Traced,
    shexp_down_w: Traced,
    shexp_gin_w: Traced,
    top_k: usize,
    tables: &Qwen4ExpPackedExpertTables,
    mode: PackedRoutingMode,
) -> Result<(), Qwen4ExpPackedRoutedError> {
    const F32_EXACT_INT_MAX: usize = 1 << 24;

    let inputs = b.append_plan(0);
    let x_type = qwen38_traced_type(&inputs, "x", x)?;
    qwen38_require_dtype("x", x_type.dtype)?;
    qwen38_require_rank("x", &x_type.shape, 3)?;
    qwen38_require_dimension("x", &x_type.shape, 0, 1)?;
    qwen38_require_dimension("x", &x_type.shape, 2, tables.hidden)?;
    if x_type.shape[1] == 0 {
        return Err(Qwen4ExpPackedRoutedError::ZeroInputDimension {
            field: "x",
            axis: 1,
        });
    }
    let rows = x_type.shape[1];

    let router_type = qwen38_traced_type(&inputs, "router_w", router_w)?;
    qwen38_require_dtype("router_w", router_type.dtype)?;
    qwen38_require_rank("router_w", &router_type.shape, 2)?;
    qwen38_require_dimension("router_w", &router_type.shape, 0, tables.hidden)?;
    qwen38_require_dimension("router_w", &router_type.shape, 1, tables.n_experts)?;

    let shared_gate_type = qwen38_traced_type(&inputs, "shexp_gate_w", shexp_gate_w)?;
    qwen38_require_dtype("shexp_gate_w", shared_gate_type.dtype)?;
    qwen38_require_rank("shexp_gate_w", &shared_gate_type.shape, 2)?;
    qwen38_require_dimension("shexp_gate_w", &shared_gate_type.shape, 0, tables.hidden)?;
    if shared_gate_type.shape[1] == 0 {
        return Err(Qwen4ExpPackedRoutedError::ZeroInputDimension {
            field: "shexp_gate_w",
            axis: 1,
        });
    }
    let shared_intermediate = shared_gate_type.shape[1];

    for (field, traced, expected) in [
        (
            "shexp_up_w",
            shexp_up_w,
            TensorType::f32(vec![tables.hidden, shared_intermediate]),
        ),
        (
            "shexp_down_w",
            shexp_down_w,
            TensorType::f32(vec![shared_intermediate, tables.hidden]),
        ),
        (
            "shexp_gin_w",
            shexp_gin_w,
            TensorType::f32(vec![tables.hidden, 1]),
        ),
    ] {
        let actual = qwen38_traced_type(&inputs, field, traced)?;
        qwen38_require_dtype(field, actual.dtype)?;
        qwen38_require_rank(field, &actual.shape, 2)?;
        for (axis, &dimension) in expected.shape.iter().enumerate() {
            qwen38_require_dimension(field, &actual.shape, axis, dimension)?;
        }
    }

    if top_k == 0 || top_k > tables.n_experts {
        return Err(Qwen4ExpPackedRoutedError::TopK {
            actual: top_k,
            experts: tables.n_experts,
        });
    }
    if tables.n_experts > F32_EXACT_INT_MAX {
        return Err(Qwen4ExpPackedRoutedError::F32ExactIntegerRange {
            field: "expert count",
            actual: tables.n_experts,
            max: F32_EXACT_INT_MAX,
        });
    }
    for (field, factors) in [
        ("router logits", [rows, tables.n_experts, 1]),
        (
            "stable router rank pairs",
            [rows, tables.n_experts, tables.n_experts],
        ),
        ("selected activation rows", [rows, top_k, tables.hidden]),
        (
            "selected expert intermediate",
            [rows, top_k, tables.intermediate],
        ),
        ("shared expert intermediate", [rows, shared_intermediate, 1]),
    ] {
        qwen38_checked_product(field, &factors)?;
    }

    if matches!(mode, PackedRoutingMode::Grouped) {
        // The grouped sort's row-tie-break and expert-offset ranges are computed inside
        // `packed_grouped_linear` with `iota` (card 558a); check their lengths fit the exact-f32 range.
        let m = rows
            .checked_mul(top_k)
            .ok_or(Qwen4ExpPackedRoutedError::ArithmeticOverflow {
                field: "grouped row iota length",
            })?;
        let e1 = tables.n_experts.checked_add(1).ok_or(
            Qwen4ExpPackedRoutedError::ArithmeticOverflow {
                field: "grouped expert offset iota length",
            },
        )?;
        for (field, value) in [("row_iota", m), ("expert_iota", e1)] {
            if value > F32_EXACT_INT_MAX {
                return Err(Qwen4ExpPackedRoutedError::F32ExactIntegerRange {
                    field,
                    actual: value,
                    max: F32_EXACT_INT_MAX,
                });
            }
        }
    }

    let packed_rows = tables
        .n_experts
        .checked_mul(Qwen4ExpExpertProjection::ALL.len())
        .ok_or(Qwen4ExpPackedRoutedError::ArithmeticOverflow {
            field: "packed projection rows",
        })?;
    let component_count =
        packed_rows
            .checked_mul(2)
            .ok_or(Qwen4ExpPackedRoutedError::ArithmeticOverflow {
                field: "packed source components",
            })?;
    let mut namespaces = b.append_plan(packed_rows);
    let mut component = 0usize;
    for role in Qwen4ExpExpertProjection::ALL {
        for row in tables.projection(role).rows() {
            for (name, tensor_type) in packed_source_constants(&row.linear_id, row.descriptor) {
                component = component.checked_add(1).ok_or(
                    Qwen4ExpPackedRoutedError::ArithmeticOverflow {
                        field: "packed source component ordinal",
                    },
                )?;
                if component == component_count {
                    namespaces.input_result(name, tensor_type, Storage::Const)?;
                } else {
                    namespaces.input(name, tensor_type, Storage::Const)?;
                }
            }
        }
    }
    let _prepared = b.preflight_append(namespaces)?;
    Ok(())
}

/// Build Qwen4Exp's routed block with three official split packed projection tables, one row per
/// activation (decode).
///
/// A model-local graph construction seam, not Runner admission. It reuses Card 369's indexed builder for
/// gate, up, and down, while routing, weighting, shared-expert gating, and combining stay in
/// `qwen38_routed_ffn` with the dense synthetic source. Input shapes, size arithmetic, and packed source
/// namespaces are preflighted before that shared definition appends an equation. Prefill's counterpart
/// is [`qwen38_packed_grouped_moe_ffn`].
#[allow(clippy::too_many_arguments)]
pub fn qwen38_packed_indexed_moe_ffn(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    shexp_gate_w: Traced,
    shexp_up_w: Traced,
    shexp_down_w: Traced,
    shexp_gin_w: Traced,
    top_k: usize,
    tables: &Qwen4ExpPackedExpertTables,
) -> Result<Traced, Qwen4ExpPackedRoutedError> {
    let mode = PackedRoutingMode::Indexed;
    preflight_qwen38_packed_moe_ffn(
        b,
        x,
        router_w,
        shexp_gate_w,
        shexp_up_w,
        shexp_down_w,
        shexp_gin_w,
        top_k,
        tables,
        mode,
    )?;
    Ok(qwen38_routed_ffn(
        b,
        x,
        router_w,
        shexp_gate_w,
        shexp_up_w,
        shexp_down_w,
        shexp_gin_w,
        tables.n_experts,
        top_k,
        tables.intermediate,
        Qwen4ExpRoutedExpertSource::Packed { tables, mode },
    )?)
}

/// Build Qwen4Exp's routed block with three official split packed projection tables, rows stable-sorted
/// by expert first (prefill).
///
/// Shares every shape and namespace preflight with [`qwen38_packed_indexed_moe_ffn`]; only the packed
/// projection composition differs (`packed_grouped_linear` instead of `packed_indexed_linear`), via the
/// same [`qwen38_routed_ffn`]. `row_iota` and `expert_iota` are the grouped builder's required operands:
/// `row_iota` breaks the stable sort's ties over the `rows * top_k` activation rows and is load-bearing
/// (it feeds the `Gather(sorted, perm)` that restores row order). `expert_iota` is required and
/// shape-validated (`[n_experts + 1]`) but not currently load-bearing: its only consumer is an
/// `_offsets` reduction (`poot-graph-ir::ops::packed_grouped_linear`, around `ops.rs:1190`) that nothing
/// reads, so any correctly-shaped `expert_iota` gives byte-identical output today. Card 388
/// tracks wiring `_offsets` into a
/// consumer or removing the operand.
#[allow(clippy::too_many_arguments)]
pub fn qwen38_packed_grouped_moe_ffn(
    b: &Builder,
    x: Traced,
    router_w: Traced,
    shexp_gate_w: Traced,
    shexp_up_w: Traced,
    shexp_down_w: Traced,
    shexp_gin_w: Traced,
    top_k: usize,
    tables: &Qwen4ExpPackedExpertTables,
) -> Result<Traced, Qwen4ExpPackedRoutedError> {
    let mode = PackedRoutingMode::Grouped;
    preflight_qwen38_packed_moe_ffn(
        b,
        x,
        router_w,
        shexp_gate_w,
        shexp_up_w,
        shexp_down_w,
        shexp_gin_w,
        top_k,
        tables,
        mode,
    )?;
    Ok(qwen38_routed_ffn(
        b,
        x,
        router_w,
        shexp_gate_w,
        shexp_up_w,
        shexp_down_w,
        shexp_gin_w,
        tables.n_experts,
        top_k,
        tables.intermediate,
        Qwen4ExpRoutedExpertSource::Packed { tables, mode },
    )?)
}
