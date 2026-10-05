//! `op_label` (a static per-`OpKind` label for `EvalError::Unsupported`) and [`classify`], the one
//! admission table this file now keeps: the closed set of Spec 376 equations a BF16-touching value
//! may appear in (moved here from `exact_bf16.rs`, which now holds only the
//! BF16 carrier, so this file is the one admission home).
//!
//! Card 555: this file used to also run a per-equation "admission preflight"
//! (`resolve`/`preflight`) ahead of `walk::eval`'s own evaluation loop. `infer` (`poot-graph-ir`)
//! already restricts every op's dtype combination - `Cast` included, since `infer` now refuses its nine
//! refused pairs into `infer` itself - to exactly what `walk::evaluate_equation` implements for it, so
//! that preflight's match had become a pure `Ok(())` over every `OpKind` variant: a check that cannot
//! fail is not a contract (rule 13 spirit), so it is deleted rather than kept as a vacuous pass. A new
//! `OpKind` variant that `walk::evaluate_equation` does not yet implement still fails loudly, just one
//! level down, the first time an equation of that shape is evaluated (`EvalError::Unsupported`), not
//! before the walk starts.

use poot_graph_ir::{Eqn, Graph, OpKind, Operand, ValidationChannel};
use poot_tensor::DType;

/// A static label for [`EvalError::Unsupported`], one per [`OpKind`] discriminant (never the dynamic,
/// allocating [`OpKind::name`]).
pub(crate) fn op_label(op: &OpKind) -> &'static str {
    match op {
        OpKind::Unary(_) => "unary",
        OpKind::Binary(_) => "binary",
        OpKind::Select => "select",
        OpKind::Reduce { .. } => "reduce",
        OpKind::Broadcast { .. } => "broadcast",
        OpKind::Cast { .. } => "cast",
        OpKind::Reshape { .. } => "reshape",
        OpKind::Transpose { .. } => "transpose",
        OpKind::Slice { .. } => "slice",
        OpKind::Concat { .. } => "concat",
        OpKind::Iota { .. } => "iota",
        OpKind::Gather { .. } => "gather",
        OpKind::Scatter { .. } => "scatter",
        OpKind::ScatterUpdate => "scatter_update",
        OpKind::MatMul => "matmul",
        OpKind::PackedDequant { .. } => "packed_dequant",
        OpKind::PackedContraction { .. } => "packed_contraction",
        OpKind::PackedRowGather { .. } => "packed_row_gather",
        OpKind::DenseContraction { .. } => "dense_contraction",
        OpKind::DenseRowGather { .. } => "dense_row_gather",
        OpKind::MatMulBias => "matmul_bias",
        OpKind::ArgTopK { .. } => "arg_top_k",
        OpKind::IndexedMatMul => "indexed_matmul",
        OpKind::DynamicUpdateSlice { .. } => "dynamic_update_slice",
        OpKind::PackI8 => "pack_i8",
        OpKind::UnpackI8 { .. } => "unpack_i8",
        OpKind::Fused(_) => "fused",
        OpKind::FusedRow(_) => "fused_row",
        OpKind::FlashAttentionDecode { .. } => "flash_attention_decode",
        OpKind::FlashAttentionPrefill { .. } => "flash_attention_prefill",
        OpKind::Rope { .. } => "rope",
        OpKind::AllReduce { .. } => "all_reduce",
        OpKind::AllGather { .. } => "all_gather",
        OpKind::RandomUniform { .. } => "random_uniform",
        OpKind::SampleToken { .. } => "sample_token",
    }
}

/// One Spec 376 BF16 equation. These are exactly the BF16 equations the GLM5Next and Qwen3.8-27B production graphs
/// apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Bf16Equation<'a> {
    /// `Cast(BF16 -> F32)`.
    Widen,
    /// `Gather` on a BF16 table with I32 ids.
    Gather { axis: usize },
    /// `Reshape` of a BF16 value.
    Reshape { shape: &'a [usize] },
    /// `Transpose` of a BF16 value.
    Transpose { perm: &'a [usize] },
    /// `MatMul(F32 activation, BF16 weight) -> F32`, the mixed-dtype linear.
    WeightMatMul,
    /// `DenseContraction(F32 activation, BF16 weight [N, K]) -> F32`, Card 380. The same contraction as
    /// [`Bf16Equation::WeightMatMul`] with the weight in checkpoint order instead of contraction order, so
    /// the recognizer can drop the `Transpose` the LM head would otherwise materialize on device.
    WeightContraction,
    /// `DenseRowGather(BF16 table [V, R], I32 index) -> F32`, Card 381. Exactly
    /// [`Bf16Equation::Gather`] on axis 0 followed by [`Bf16Equation::Widen`], folded into one equation so
    /// the gathered rows never have to exist as a BF16 device value.
    RowGatherWiden,
}

/// Whether `eqn` reads or produces a BF16 value.
pub(crate) fn touches_bf16<V: ValidationChannel>(graph: &Graph<V>, eqn: &Eqn) -> bool {
    graph.aval(eqn.out).dtype == DType::BF16
        || eqn.inputs.iter().any(|operand| match operand {
            Operand::Value(value) => graph.aval(*value).dtype == DType::BF16,
            Operand::Lit(_) => false,
        })
}

/// The Spec 376 equation `eqn` is, if any. Admission, owner preflight, and evaluation all use this one table.
pub(crate) fn classify<'a, V: ValidationChannel>(
    graph: &Graph<V>,
    eqn: &'a Eqn,
) -> Option<Bf16Equation<'a>> {
    if !touches_bf16(graph, eqn) {
        return None;
    }
    let dtype = |position: usize| match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(graph.aval(*value).dtype),
        _ => None,
    };
    let rank = |position: usize| match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(graph.aval(*value).rank()),
        _ => None,
    };
    let (arity, first, second) = (eqn.inputs.len(), dtype(0), dtype(1));
    match &eqn.op {
        OpKind::Cast { to: DType::F32 } if (arity, first) == (1, Some(DType::BF16)) => {
            Some(Bf16Equation::Widen)
        }
        OpKind::Gather { axis }
            if (arity, first, second) == (2, Some(DType::BF16), Some(DType::I32)) =>
        {
            Some(Bf16Equation::Gather { axis: *axis })
        }
        OpKind::Reshape { shape } if (arity, first) == (1, Some(DType::BF16)) => {
            Some(Bf16Equation::Reshape {
                shape: shape.as_slice(),
            })
        }
        OpKind::Transpose { perm } if (arity, first) == (1, Some(DType::BF16)) => {
            Some(Bf16Equation::Transpose {
                perm: perm.as_slice(),
            })
        }
        OpKind::MatMul if (arity, first, second) == (2, Some(DType::F32), Some(DType::BF16)) => {
            Some(Bf16Equation::WeightMatMul)
        }
        // The weight is `[N, K]`: `evaluate_equation` relabels its strides with a rank-2 permutation, so an
        // unadmitted rank must fail here, before anything runs.
        OpKind::DenseContraction {
            weight: DType::BF16,
        } if (arity, first, second) == (2, Some(DType::F32), Some(DType::BF16))
            && rank(1) == Some(2) =>
        {
            Some(Bf16Equation::WeightContraction)
        }
        // The table is `[V, R]`: the row selection reads the gathered axis extent, so an unadmitted rank
        // must fail here, before anything runs.
        OpKind::DenseRowGather {
            source: DType::BF16,
        } if (arity, first, second) == (2, Some(DType::BF16), Some(DType::I32))
            && rank(0) == Some(2) =>
        {
            Some(Bf16Equation::RowGatherWiden)
        }
        _ => None,
    }
}
