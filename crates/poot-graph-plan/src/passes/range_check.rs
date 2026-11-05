//! `check_f16_ranges` (POOT-1475, `receipts/1471-narrow-without-range-guards/design.md` section 6.2): the opt-in range
//! detector behind `--check-ranges`.
//!
//! Under `Narrow` a value past f16's range rounds to infinity and the step's logits are refused, but the refusal
//! cannot say where the range was left (and under an attention score softcap it cannot fire at all, design 4.3).
//! This pass runs after `select_operand_precision` and gives every value an f16-narrowed op rounds a validation
//! witness: 1 when any of its elements would not round to a finite f16 (a NaN counts, as a non-finite value does),
//! else 0 (a count would pass the exact 2^24 bound of a witness head on a long context). The witness is an ordinary
//! graph validation, so it is admitted, computed and checked by the channel the router witnesses use: a failing step
//! is `ExecError::Validation` naming the witness, and the name carries the first reading op, its layer and the
//! operand's role (`f16_range.3.l17.attention.query`).
//!
//! The pass is a compile option, not a kernel branch. When it does not run the graph has no witness and no extra
//! equation: a program compiled without it is the program compiled before this pass existed.
//!
//! What is watched: the operands narrowing rounds, that is the activation of a contraction, both operands of a
//! product of two runtime values, and the query, key and value of an attention. Operands that hold weight bytes are
//! not watched (their range is known at bind and checked there, POOT-1458), nor are operands already stored f16.

use std::collections::HashSet;

use poot_graph_ir::graph::{
    EditError, Eqn, Graph, GraphEdit, LayerIndex, Operand, Storage, ValueMeta,
};
use poot_graph_ir::op::{BinOp, ContractionHead, OpKind, RedOp, UnOp};
use poot_graph_ir::{FragmentDtype, KeyLayout, Scalar, TensorType, ValidationId, ValueId};
use poot_tensor::DType;

/// The largest f32 that rounds to a finite f16: 65520 is the rounding edge (halfway to the next power of two,
/// ties to even, gives infinity), so `|x| < 65520` is `|x| <= 65519.996`.
const LARGEST_IN_RANGE: f32 = 65_519.996;

/// `g` with a range witness on every operand an f16-narrowed op of it rounds. A graph with no such op is returned
/// unchanged.
pub(crate) fn check_f16_ranges(g: &Graph) -> Result<Graph, EditError> {
    let mut edit = g.edit();
    let first_id = g
        .witnesses()
        .map(|witness| witness.id.0 + 1)
        .max()
        .unwrap_or(0);
    let mut added: Vec<ValidationId> = Vec::new();
    let mut watched: HashSet<ValueId> = HashSet::new();
    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns().len());
    for eqn in g.eqns() {
        for (role, position) in rounded_operands(g, eqn) {
            let Operand::Value(value) = eqn.inputs[position] else {
                continue;
            };
            if !watched.insert(value) {
                continue;
            }
            let witness = emit_range_count(&mut edit, &mut eqns, value, eqn.layer);
            let id = ValidationId(first_id + added.len() as u32);
            let layer = eqn
                .layer
                .map_or_else(String::new, |LayerIndex(layer)| format!(".l{layer}"));
            let name = format!(
                "f16_range.{}{layer}.{}.{role}",
                added.len(),
                op_label(&eqn.op)
            );
            edit.push_witness(id, name, witness);
            added.push(id);
        }
        eqns.push(eqn.clone());
    }
    if added.is_empty() {
        return Ok(g.clone());
    }
    *edit.eqns_mut() = eqns;
    edit.commit_adding_witnesses(&added)
}

/// The short name of the op in a witness name.
fn op_label(op: &OpKind) -> &'static str {
    match op {
        OpKind::Attention(_) => "attention",
        OpKind::SelectedAttention(_) => "selected_attention",
        OpKind::MatMul { .. } => "matmul",
        OpKind::DenseContraction { .. } => "dense_contraction",
        OpKind::GroupedDenseContraction { .. } => "grouped_dense_contraction",
        OpKind::PackedContraction { .. } => "packed_contraction",
        OpKind::IndexedMatMul { .. } => "indexed_matmul",
        OpKind::ContractionEpilogue { .. } => "contraction_epilogue",
        _ => "op",
    }
}

/// The operands `eqn` rounds to f16, as `(role, operand position)`: none unless the equation reads its operands
/// through an f16 fragment.
fn rounded_operands(g: &Graph, eqn: &Eqn) -> Vec<(&'static str, usize)> {
    let narrows_f16 = eqn
        .op
        .operand_precision()
        .and_then(|precision| precision.fragment_dtype())
        == Some(FragmentDtype::F16);
    if !narrows_f16 {
        return Vec::new();
    }
    let candidates: Vec<(&'static str, usize)> = match &eqn.op {
        OpKind::Attention(attrs) => {
            let mut operands = vec![("query", 0), ("key", 1), ("value", 2)];
            if matches!(attrs.key, KeyLayout::Split { .. }) {
                operands.push(("key_shared", 4));
            }
            operands
        }
        OpKind::SelectedAttention(_) => vec![("query", 0), ("key", 1), ("value", 2)],
        OpKind::MatMul { .. } | OpKind::IndexedMatMul { .. } => {
            vec![("activation", 0), ("operand", 1)]
        }
        OpKind::ContractionEpilogue { head, .. } => contraction_operands(head),
        OpKind::DenseContraction { .. }
        | OpKind::GroupedDenseContraction { .. }
        | OpKind::PackedContraction { .. } => vec![("activation", 0)],
        _ => Vec::new(),
    };
    candidates
        .into_iter()
        .filter(|&(_, position)| match eqn.inputs.get(position) {
            Some(Operand::Value(value)) => {
                let meta = g.meta(*value);
                !is_already_f16(meta.aval.dtype) && !is_fixed_before_the_run(meta.storage)
            }
            _ => false,
        })
        .collect()
}

/// The operands a fused epilogue head reads, as the bare contraction of the head reads them.
fn contraction_operands(head: &ContractionHead) -> Vec<(&'static str, usize)> {
    match head {
        ContractionHead::MatMul { .. } | ContractionHead::Indexed { .. } => {
            vec![("activation", 0), ("operand", 1)]
        }
        ContractionHead::Dense { .. } | ContractionHead::Packed { .. } => vec![("activation", 0)],
    }
}

/// An f16 value is in f16's range by construction.
fn is_already_f16(dtype: DType) -> bool {
    dtype == DType::F16
}

/// A weight or a table the compiler computes: its range does not depend on the run.
fn is_fixed_before_the_run(storage: Storage) -> bool {
    storage.holds_weight_bytes() || matches!(storage, Storage::Computed(_))
}

/// Push the steps that count the elements of `x` that do not round to a finite f16, reduced to one F32 scalar, and
/// return it. Every step is F32 and the count is exact: a comparison is 0 or 1 for any operand (NaN included),
/// so the chain is a witness head `device_validation` admits. The elements are reduced by their maximum, so a lane
/// holds 0 or 1: a count would pass the exact 2^24 bound of a witness head on a long context.
fn emit_range_count(
    edit: &mut GraphEdit,
    eqns: &mut Vec<Eqn>,
    x: ValueId,
    layer: Option<LayerIndex>,
) -> ValueId {
    let mut step = |edit: &mut GraphEdit, op: OpKind, inputs: Vec<Operand>| {
        let types: Vec<TensorType> = inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => edit.aval(*value).clone(),
                Operand::Lit(scalar) => scalar.ty(),
            })
            .collect();
        let aval = op
            .infer(&types)
            .unwrap_or_else(|error| panic!("check_f16_ranges: {op:?} over {types:?}: {error}"));
        let out = edit
            .values_mut()
            .push(ValueMeta::new(aval, Storage::Device, None));
        eqns.push(Eqn {
            op,
            inputs,
            out,
            layer,
        });
        out
    };
    let value = |v: ValueId| Operand::Value(v);
    let x = if edit.aval(x).dtype == DType::F32 {
        x
    } else {
        step(
            edit,
            OpKind::Cast { to: DType::F32 },
            vec![Operand::Value(x)],
        )
    };
    // `-|x| >= -65519.996` is false for NaN, so the complement counts it.
    let neg = step(edit, OpKind::Unary(UnOp::Neg), vec![value(x)]);
    let magnitude = step(edit, OpKind::Binary(BinOp::Max), vec![value(x), value(neg)]);
    let neg_magnitude = step(edit, OpKind::Unary(UnOp::Neg), vec![value(magnitude)]);
    let in_range = step(
        edit,
        OpKind::Binary(BinOp::Ge),
        vec![
            value(neg_magnitude),
            Operand::Lit(Scalar::F32(-LARGEST_IN_RANGE)),
        ],
    );
    let negated = step(
        edit,
        OpKind::Binary(BinOp::Mul),
        vec![value(in_range), Operand::Lit(Scalar::F32(-1.0))],
    );
    let bad = step(
        edit,
        OpKind::Binary(BinOp::Add),
        vec![value(negated), Operand::Lit(Scalar::F32(1.0))],
    );
    // Two reductions at most, whatever the rank (a reshape is a view): the last axis of every row, then the rows.
    // One flat reduction over a prefill's elements would be one workgroup's serial loop; one reduction per axis
    // pays a launch for each.
    let shape = edit.aval(bad).shape.clone();
    let Some(&last) = shape.last() else {
        return bad;
    };
    let elements: usize = shape.iter().product();
    let rows = elements / last.max(1);
    let reduce_max = |axis: usize| OpKind::Reduce {
        op: RedOp::Max,
        axis,
        keepdim: false,
    };
    if rows <= 1 {
        let flat = if shape.len() > 1 {
            step(
                edit,
                OpKind::Reshape {
                    shape: vec![elements],
                },
                vec![value(bad)],
            )
        } else {
            bad
        };
        return step(edit, reduce_max(0), vec![value(flat)]);
    }
    let matrix = if shape.len() > 2 {
        step(
            edit,
            OpKind::Reshape {
                shape: vec![rows, last],
            },
            vec![value(bad)],
        )
    } else {
        bad
    };
    let per_row = step(edit, reduce_max(1), vec![value(matrix)]);
    step(edit, reduce_max(0), vec![value(per_row)])
}
