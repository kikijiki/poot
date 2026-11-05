//! `select_operand_precision` (POOT-1430): the one pass that writes a non-exact [`OperandPrecision`].
//!
//! The pass reads three things and nothing else: the driver's [`MatmulPrecision`] policy, the float dtypes the
//! target's matrix units take ([`FragmentUnits`]), and the dtypes of the op's operands. It never reads a model, a
//! shape, a benchmark or an environment variable. The policy is a permission: a target with no matrix unit for the
//! op's operands leaves the op `Exact` under any policy, because narrowing there would lose accuracy for no speed.
//! What the planner then makes of a narrowed op (a fragment micro-tile, a scalar micro-tile, a GEMV) is the
//! planner's business; every schedule computes the same declared rounding.
//!
//! A contraction narrowed to f16 reads its activation row-scaled (POOT-1457, [`emit_row_scaled`]), and an attention
//! narrowed to f16 its query (POOT-1455, [`emit_query_scaled`]), so no rounding to f16 can overflow whatever the
//! data (`receipts/1457-f16-range-guard/design.md`, `receipts/1455-attention-f16-range/design.md`).

use std::collections::HashMap;

use poot_graph_ir::graph::{
    EditError, Eqn, Graph, GraphEdit, Operand, Storage, ValueMeta, ValueTable,
};
use poot_graph_ir::op::{BinOp, ContractionHead, OpKind, RedOp, UnOp};
use poot_graph_ir::{
    FragmentDtype, KeyLayout, MatmulPrecision, OperandPrecision, QueryScale, Scalar, TensorType,
    ValueId,
};
use poot_target::{DeviceCaps, FragmentShape, MatrixUnits};

use crate::Target;
use poot_tensor::DType;

/// The matrix units a target reports (`DeviceCaps::matrix_units`), as the pass reads them: it takes a fragment
/// dtype when the device has a 16x16x16 config for it, the one shape the kernels lower. A device that cannot say
/// what it has (`Queried::Unknown`) has no unit the pass will narrow for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct FragmentUnits(MatrixUnits);

impl FragmentUnits {
    /// The units `caps` reports. This is the one place the planner reads the device's matrix hardware for
    /// operand precision.
    pub(crate) fn of(caps: &DeviceCaps) -> Self {
        Self(caps.matrix_units.known().unwrap_or(MatrixUnits::NONE))
    }

    /// These units cut down to the `dtype` config alone, or to none when the device has no such config.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn only(self, dtype: FragmentDtype) -> Self {
        Self(MatrixUnits::of(self.0.config(dtype).cloned()))
    }

    fn takes(self, dtype: FragmentDtype) -> bool {
        self.0
            .config(dtype)
            .is_some_and(|config| config.shape == FragmentShape::M16N16K16)
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says on `target`: the
/// graph `compile`'s `select_operand_precision` pass writes, before the passes that fold and fuse it. It is the
/// graph a CPU oracle evaluates to hold a device that compiled `g` under `policy` to the declared semantics (a
/// program narrowed by the planner is compared with the oracle of the narrowed ops, not of the exact ones). A
/// target with no matrix unit, or the policy `Exact`, returns `g` unchanged.
#[cfg(any(test, feature = "test-support"))]
pub fn with_operand_precision(
    g: &Graph,
    target: &Target,
    policy: MatmulPrecision,
) -> Result<Graph, EditError> {
    match target {
        Target::Gpu { caps, .. } => select_operand_precision(g, policy, FragmentUnits::of(caps)),
        Target::Npu(_) => Ok(g.clone()),
    }
}

/// `g` with every contraction and attention equation reading its operands as `policy` says, where `units` admit
/// it. The policy `Exact` returns `g` unchanged.
///
/// A contraction narrowed to f16 with an f32 activation reads it row-scaled ([`emit_row_scaled`]). The scale
/// steps are part of what this pass writes, not a later pass: the graph the oracle evaluates
/// ([`with_operand_precision`]) and the graph the planner compiles carry the same steps, and they are never
/// reverted (the pass is required).
pub(crate) fn select_operand_precision(
    g: &Graph,
    policy: MatmulPrecision,
    units: FragmentUnits,
) -> Result<Graph, EditError> {
    if policy == MatmulPrecision::Exact {
        return Ok(g.clone());
    }
    let weights = Weights::of(g);
    let mut edit = g.edit();
    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns().len());
    // The scale and the scaled value of each activation already scaled, shared by every reader (q, k and v read
    // one norm output, gate and up another).
    let mut scaled: HashMap<ValueId, (ValueId, ValueId)> = HashMap::new();
    for eqn in g.eqns() {
        let mut eqn = eqn.clone();
        if eqn.op.operand_precision() != Some(OperandPrecision::Exact) {
            eqns.push(eqn);
            continue;
        }
        let guard = f16_guard(g, &eqn, &weights);
        let Some(dtype) = fragment_dtype(g, &eqn, units, guard.is_some()) else {
            eqns.push(eqn);
            continue;
        };
        eqn.op = eqn
            .op
            .with_operand_precision(policy.operand_precision(dtype));
        match guard {
            Some(F16Guard::RowScale) if dtype == FragmentDtype::F16 => {
                if is_attention(&eqn.op) {
                    emit_query_scaled(&mut edit, &mut eqns, &mut scaled, eqn);
                } else {
                    emit_row_scaled(&mut edit, &mut eqns, &mut scaled, eqn);
                }
            }
            _ => eqns.push(eqn),
        }
    }
    *edit.eqns_mut() = eqns;
    edit.commit()
}

/// The largest finite f16, 65504. An f32 at most this large rounds to a finite f16 (the round-to-nearest overflow
/// edge is 65520), so an activation scaled under it cannot overflow when narrowed.
const F16_MAX: f32 = 65504.0;

/// How a contraction narrowed to f16 keeps its activation in f16's range.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum F16Guard {
    /// The activation is stored f16 already: narrowing it rounds nothing.
    Stored,
    /// The f32 activation is read row-scaled ([`emit_row_scaled`]), or an attention's f32 query
    /// ([`emit_query_scaled`]).
    RowScale,
    /// The attention's query arrives row-scaled already: its graph carries the query scale.
    QueryScaled,
}

fn is_attention(op: &OpKind) -> bool {
    matches!(op, OpKind::Attention(_) | OpKind::SelectedAttention(_))
}

/// The values a pass may treat as weights: values fixed before the program runs. That is bound bytes
/// ([`poot_graph_ir::graph::Storage::holds_weight_bytes`]) and constants the compiler materializes, and every value
/// computed from those alone.
///
/// A weight's range is fixed when it is bound, so no runtime scale applies to it. The traced graph reads a
/// projection's weight through ops that `compile` folds into the contraction before this pass runs there: a
/// transpose (`MatMul(x, Transpose(w))`, `fold_dense_contractions`) or a decode of packed bytes
/// (`MatMul(x, Transpose(PackedDequant(w)))`, `recognize_packed_contractions`). Deciding by what a value is computed
/// from, not by the ops between, makes the oracle's graph ([`with_operand_precision`]) and the compiled one take
/// the same decision.
struct Weights(ValueTable<bool>);

impl Weights {
    fn of(g: &Graph) -> Self {
        let mut fixed = ValueTable::filled(g.values().len(), false);
        for (value, meta) in g.values().iter_enumerated() {
            fixed[value] =
                matches!(meta.storage, Storage::Computed(_)) || meta.storage.holds_weight_bytes();
        }
        // Equations are in topological order, so every operand is decided before its readers.
        for eqn in g.eqns() {
            fixed[eqn.out] = eqn.inputs.iter().all(|operand| match operand {
                Operand::Value(value) => fixed[*value],
                Operand::Lit(_) => true,
            });
        }
        Self(fixed)
    }

    fn holds(&self, value: ValueId) -> bool {
        self.0[value]
    }
}

/// How contraction `eqn` keeps its activation in f16's range when it narrows to f16, or `None` when it cannot
/// (so f16 is not a candidate for it).
///
/// - Every operand but the activation (operand 0) must be a weight ([`Weights`]). A contraction of two runtime
///   values (`MatMul(a, b)` outside attention) has no guard and does not narrow to f16.
/// - The activation is f16 already ([`F16Guard::Stored`]), or f32 with the result's leading dims and rows, so
///   the per-row scale `[.., M, 1]` multiplies the f32 result back ([`F16Guard::RowScale`]). A bf16 activation
///   would be rounded in bf16 by the scaling itself, and an activation whose rows are not the result's (a
///   broadcast batch) cannot take the scale back; neither is guarded.
/// - An attention's f32 query is read row-scaled ([`emit_query_scaled`], POOT-1455), or arrives so already
///   ([`F16Guard::QueryScaled`]). Its keys and values are not scaled: [`fragment_dtype`] takes f16 for an
///   attention only where they are stored f16, so they enter its products exactly. The probabilities lie in
///   `[0, 1]`.
fn f16_guard(g: &Graph, eqn: &Eqn, weights: &Weights) -> Option<F16Guard> {
    let query_scale = match &eqn.op {
        OpKind::Attention(attrs) => Some(attrs.query),
        OpKind::SelectedAttention(attrs) => Some(attrs.query),
        _ => None,
    };
    if let Some(query_scale) = query_scale {
        return match (query_scale, value_dtype(g, eqn, 0)?) {
            (QueryScale::PerRow, _) => Some(F16Guard::QueryScaled),
            (QueryScale::Absent, DType::F32) => Some(F16Guard::RowScale),
            (QueryScale::Absent, _) => None,
        };
    }
    if !matches!(
        eqn.op,
        OpKind::MatMul { .. }
            | OpKind::DenseContraction { .. }
            | OpKind::GroupedDenseContraction { .. }
            | OpKind::PackedContraction { .. }
            | OpKind::IndexedMatMul { .. }
    ) {
        return None;
    }
    let all_weights = eqn.inputs[1..].iter().all(|operand| match operand {
        // An index (expert ids) or a packed payload is no float operand of the products.
        Operand::Value(value) => {
            !matches!(g.aval(*value).dtype, DType::F32 | DType::BF16 | DType::F16)
                || weights.holds(*value)
        }
        Operand::Lit(_) => true,
    });
    let Some(Operand::Value(activation)) = eqn.inputs.first() else {
        return None;
    };
    if !all_weights {
        return None;
    }
    let (x, out) = (g.aval(*activation), g.aval(eqn.out));
    match x.dtype {
        DType::F16 => Some(F16Guard::Stored),
        DType::F32 => {
            let rank = x.shape.len();
            let rows_match = rank >= 2
                && out.dtype == DType::F32
                && out.shape.len() == rank
                && x.shape[..rank - 1] == out.shape[..rank - 1];
            rows_match.then_some(F16Guard::RowScale)
        }
        _ => None,
    }
}

/// Push `eqn` reading its activation `x [.., M, K]` row-scaled, and its result multiplied back by the scale:
///
/// ```text
/// s  = max(rowmax(max(x, -x)) / 65504, 1)    [.., M, 1]
/// y  = contraction(x / s, ..) * s            [.., M, N]
/// ```
///
/// Every `|x / s|` is at most 65504, so its rounding to f16 is finite. Where a row is in range, its `s` is exactly
/// 1: `x / 1` and `y * 1` are exact, and the row computes bit for bit what it would without the scale. An
/// activation already scaled reuses its scale and scaled value. This is the dynamic per-token absmax scale engines
/// use to narrow into a limited range (vLLM's `dynamic_per_token_scaled_fp8_quant`, llama.cpp's `quantize_q8_1`;
/// `receipts/1454-narrow-dtype-policy/design.md` section 2), with no threshold but the dtype's own range. The steps
/// are primitives, so the oracle evaluates what the device runs, and `fuse` folds them into the row region that
/// produces `x` (POOT-1456).
fn emit_row_scaled(
    edit: &mut GraphEdit,
    eqns: &mut Vec<Eqn>,
    scaled: &mut HashMap<ValueId, (ValueId, ValueId)>,
    mut eqn: Eqn,
) {
    let (s, x_scaled) = row_scale(edit, eqns, scaled, &eqn);
    eqn.inputs[0] = Operand::Value(x_scaled);
    // The contraction writes a fresh value and the multiply writes the original result, so every reader keeps it.
    let layer = eqn.layer;
    let result = eqn.out;
    let result_type = edit.aval(result).clone();
    let product = edit
        .values_mut()
        .push(ValueMeta::new(result_type, Storage::Device, None));
    eqn.out = product;
    eqns.push(eqn);
    eqns.push(Eqn {
        op: OpKind::Binary(BinOp::Mul),
        inputs: vec![Operand::Value(product), Operand::Value(s)],
        out: result,
        layer,
    });
}

/// Push attention `eqn` reading its query `q [B,Hq,Lq,D]` row-scaled, with the scale as its last operand
/// ([`QueryScale::PerRow`]):
///
/// ```text
/// s     = max(rowmax(max(q, -q)) / 65504, 1)                 [B,Hq,Lq,1]
/// score = ((q / s) . k) * s * scale
/// ```
///
/// The scale is [`emit_row_scaled`]'s, by the same steps; only where it is multiplied back differs. The attention's
/// scores feed its softmax inside the op, so the op multiplies each row's raw products by `s` itself, before its
/// score scale (`ops::attention_over_keys`, and every attention kernel). A row in range has `s` exactly 1 and
/// computes bit for bit what it would without the scale (`receipts/1455-attention-f16-range/design.md`).
fn emit_query_scaled(
    edit: &mut GraphEdit,
    eqns: &mut Vec<Eqn>,
    scaled: &mut HashMap<ValueId, (ValueId, ValueId)>,
    mut eqn: Eqn,
) {
    let (s, q_scaled) = row_scale(edit, eqns, scaled, &eqn);
    eqn.inputs[0] = Operand::Value(q_scaled);
    eqn.inputs.push(Operand::Value(s));
    match &mut eqn.op {
        OpKind::Attention(attrs) => attrs.query = QueryScale::PerRow,
        OpKind::SelectedAttention(attrs) => attrs.query = QueryScale::PerRow,
        op => unreachable!("emit_query_scaled scales an attention's query only, not {op:?}"),
    }
    eqns.push(eqn);
}

/// Push the steps that scale `eqn`'s operand 0, `x [.., M, K]`, by its rows, and return the scale `s [.., M, 1]`
/// and the scaled `x / s`. An operand already scaled reuses its steps.
fn row_scale(
    edit: &mut GraphEdit,
    eqns: &mut Vec<Eqn>,
    scaled: &mut HashMap<ValueId, (ValueId, ValueId)>,
    eqn: &Eqn,
) -> (ValueId, ValueId) {
    let layer = eqn.layer;
    let Operand::Value(x) = eqn.inputs[0] else {
        unreachable!("f16_guard scales a value operand only");
    };
    *scaled.entry(x).or_insert_with(|| {
        let shape = edit.aval(x).shape.clone();
        let mut rows = shape.clone();
        let axis = rows.len() - 1;
        rows[axis] = 1;
        let mut step = |op: OpKind, inputs: Vec<Operand>, shape: &[usize]| {
            let out = edit.values_mut().push(ValueMeta::new(
                TensorType::new(shape.to_vec(), DType::F32),
                Storage::Device,
                None,
            ));
            eqns.push(Eqn {
                op,
                inputs,
                out,
                layer,
            });
            out
        };
        let neg = step(OpKind::Unary(UnOp::Neg), vec![Operand::Value(x)], &shape);
        let magnitude = step(
            OpKind::Binary(BinOp::Max),
            vec![Operand::Value(x), Operand::Value(neg)],
            &shape,
        );
        let largest = step(
            OpKind::Reduce {
                op: RedOp::Max,
                axis,
                keepdim: true,
            },
            vec![Operand::Value(magnitude)],
            &rows,
        );
        let ratio = step(
            OpKind::Binary(BinOp::Div),
            vec![Operand::Value(largest), Operand::Lit(Scalar::F32(F16_MAX))],
            &rows,
        );
        let s = step(
            OpKind::Binary(BinOp::Max),
            vec![Operand::Value(ratio), Operand::Lit(Scalar::F32(1.0))],
            &rows,
        );
        let x_scaled = step(
            OpKind::Binary(BinOp::Div),
            vec![Operand::Value(x), Operand::Value(s)],
            &shape,
        );
        (s, x_scaled)
    })
}

/// The fragment dtype `eqn`'s products run in on `units`, or `None` when no unit admits the equation. `guarded`
/// says whether the contraction keeps its activation in f16's range ([`f16_guard`]); f16 is a candidate for a
/// contraction only then, so no contraction narrows to f16 unguarded.
///
/// - A guarded contraction takes f16 where the target has an f16 fragment, else bf16: f16 has three more mantissa
///   bits (24-41x lower mean KL than bf16, `receipts/1454-narrow-dtype-policy/design.md` section 1), and the row
///   scale keeps its activation in f16's range. It costs at most 3% of TTFT and TPOT over bf16 on ROCm since the
///   generated kernels round and decode f16 with the device's conversion and bf16 weights are rounded to f16 once
///   at bind (`receipts/1458-f16-rounding/design.md` section 6). An operand stored as f16 forces f16 (rounding it
///   to bf16 would lose bits it holds), and an unguarded contraction takes bf16 where there is a unit, else stays
///   exact.
/// - Attention takes f16 when its keys and values are stored f16 (the default KV dtype) and its query is guarded
///   ([`f16_guard`]): K and V enter exactly, Q is row-scaled, and P lies in `[0, 1]`, so no operand can overflow
///   (POOT-1455). A bf16 or f32 cache takes bf16, whose range is f32's: rounding a key or value it holds to f16
///   could overflow, and nothing scales them (`receipts/1455-attention-f16-range/design.md`). So on a target with
///   f16 fragments alone (RADV) such an attention stays exact.
fn fragment_dtype(
    g: &Graph,
    eqn: &Eqn,
    units: FragmentUnits,
    guarded: bool,
) -> Option<FragmentDtype> {
    use FragmentDtype::{Bf16, F16};
    let candidates: &[FragmentDtype] = match &eqn.op {
        OpKind::Attention(_) | OpKind::SelectedAttention(_) => {
            let stored = key_value_dtypes(g, eqn)?;
            let all = |dtype: DType| stored.iter().all(|&d| d == dtype);
            if all(DType::F16) {
                if guarded { &[F16] } else { &[] }
            } else if all(DType::BF16) || all(DType::F32) {
                &[Bf16]
            } else {
                &[]
            }
        }
        _ => {
            let operands = contraction_operands(g, eqn)?;
            match (guarded, operands.contains(&DType::F16)) {
                (true, true) => &[F16],
                (false, true) => &[],
                (true, false) => &[F16, Bf16],
                (false, false) => &[Bf16],
            }
        }
    };
    candidates.iter().copied().find(|&dtype| units.takes(dtype))
}

/// The dtypes of the float operands a contraction multiplies: its activation and, where the weight is a graph
/// operand of a float dtype, its weight. A packed weight is decoded inside the kernel and has no dtype of its
/// own. `None` when an operand is not a float, which no matrix unit takes.
fn contraction_operands(g: &Graph, eqn: &Eqn) -> Option<Vec<DType>> {
    let head = match &eqn.op {
        OpKind::ContractionEpilogue { head, .. } => *head,
        op => head_of(op)?,
    };
    let dtypes: Vec<DType> = match head {
        ContractionHead::MatMul { .. } => vec![value_dtype(g, eqn, 0)?, value_dtype(g, eqn, 1)?],
        ContractionHead::Dense { weight, .. } => vec![value_dtype(g, eqn, 0)?, weight],
        ContractionHead::Packed { .. } => vec![value_dtype(g, eqn, 0)?],
        ContractionHead::Indexed { .. } => vec![value_dtype(g, eqn, 0)?, value_dtype(g, eqn, 1)?],
    };
    dtypes
        .iter()
        .all(|dtype| matches!(dtype, DType::F32 | DType::BF16 | DType::F16))
        .then_some(dtypes)
}

/// The dtypes an attention's keys and values are stored in: the key, the value and a split key's shared part
/// (a paged key's operand 4 is its I32 slot map).
fn key_value_dtypes(g: &Graph, eqn: &Eqn) -> Option<Vec<DType>> {
    let shared =
        matches!(&eqn.op, OpKind::Attention(attrs) if matches!(attrs.key, KeyLayout::Split { .. }));
    let positions: &[usize] = if shared { &[1, 2, 4] } else { &[1, 2] };
    positions
        .iter()
        .map(|&position| value_dtype(g, eqn, position))
        .collect()
}

/// The head a bare contraction equation stands for.
fn head_of(op: &OpKind) -> Option<ContractionHead> {
    let precision = op.operand_precision()?;
    Some(match op {
        OpKind::MatMul { .. } => ContractionHead::MatMul { precision },
        OpKind::DenseContraction { weight, .. }
        | OpKind::GroupedDenseContraction { weight, .. } => ContractionHead::Dense {
            weight: *weight,
            precision,
        },
        OpKind::PackedContraction {
            descriptor, rows, ..
        } => ContractionHead::Packed {
            descriptor: *descriptor,
            rows: *rows,
            precision,
        },
        OpKind::IndexedMatMul { rows, .. } => ContractionHead::Indexed {
            rows: *rows,
            precision,
        },
        _ => return None,
    })
}

fn value_dtype(g: &Graph, eqn: &Eqn, position: usize) -> Option<DType> {
    match eqn.inputs.get(position)? {
        Operand::Value(value) => Some(g.aval(*value).dtype),
        Operand::Lit(_) => None,
    }
}
