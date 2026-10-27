//! Card 235: the mixed-operand-dtype decode `MatMul` planner gap.
//!
//! `plan_eqn`'s `OpKind::MatMul` arm rejects any matmul whose operand dtype does not
//! match the output dtype: every matmul body reads one element type. The loader types every
//! projection weight `DType::BF16` for any natively-bf16 safetensors checkpoint (independent of `to_mixed_bf16`), so a decode matmul's weight is BF16 while the
//! activation/output stay F32, and the planner's guard panics on every backend for such a checkpoint.
//!
//! [`widen_mismatched_matmul_dtypes`] fixes this: a graph transform that inserts an explicit `Cast`
//! widening a mismatched operand back to the output dtype, so `plan_eqn`'s guard fires only on the one
//! mismatch this pass leaves on purpose ([`matmul_bf16_column_gemv`]).
//!
//! This lives in `poot-graph-plan`, not `poot-graph-ir`, because the decode-projection check needs
//! [`poot_target::DeviceCaps`] and `poot-graph-ir` cannot depend on this crate
//! (`poot-graph-plan` already depends on it).
//!
//! Card 1011: a BF16 const is never retyped. Its declared dtype is its stored dtype on every backend, and
//! the lane its bytes upload to (packed `u32` words or native two-byte BF16) is decided by
//! [`bf16_const_is_packed`] and read by the storage plan and the cast planner alike. A reader with no body
//! for that lane refuses at planning; nothing widens the weight on upload.

use crate::Target;
use poot_graph_ir::graph::{Storage, ValueMeta};
use poot_graph_ir::op::{ContractionHead, LocalId, OpKind};
use poot_graph_ir::types::{DTypeClass, DTypeClassExt, TensorType};
use poot_graph_ir::{EditError, Eqn, Graph, GraphEdit, Operand, ValueId, ValueTable};
use poot_kernelgen::Layout;
use poot_target::DeviceCaps;
use poot_tensor::DType;

use std::collections::{HashMap, HashSet};

/// Card 1011: whether the device buffer that holds BF16 const `id` on `target` is the packed `u32` lanes
/// (two little-endian elements per word, [`poot_target::BufferStorage::bf16_packed`]) rather than native
/// two-byte BF16. A BF16 const is never widened: its declared dtype is its stored dtype on every backend,
/// and this decides only the lane its bytes upload to.
///
/// - A device with no native two-byte floats (SPIR-V): always packed. There is no scalar bf16, so a packed
///   reader is the only reader; an operation with no packed reader refuses at planning (the planner's typed
///   `DtypeLowering`).
/// - A device with native two-byte floats (ROCm, PTX): packed exactly when
///   [`bf16_const_feeds_only_packed_readers`] holds; otherwise native, the lane the scalar-bf16 matmul bodies
///   read.
///
/// The one decision the storage plan (`value_storage_kind`) and the cast planner share, so the lane a
/// `Cast` kernel reads and the lane its source uploads to cannot drift apart.
pub(crate) fn bf16_const_is_packed(
    g: &Graph,
    views: &HashMap<ValueId, Layout>,
    id: ValueId,
    target: &Target,
) -> bool {
    match target {
        Target::Gpu { caps, .. } if !caps.native_two_byte_floats => true,
        Target::Gpu { .. } => bf16_const_feeds_only_packed_readers(g, views, id),
        // An AIE core has no BF16 lane: the const stays unpacked and the BF16 consumers refuse; neither does a VPU.
        Target::Npu(_) => false,
    }
}

/// Whether this matmul is a decode projection over a BF16 `[K, N]` weight on a device whose BF16 lane is packed
/// (`caps.widens_bf16_decode_weight` is false): `F32 act x BF16 weight -> F32 out`, `M == 1`, a rank-2 weight.
///
/// Such a projection has no lowering. A one-workgroup-per-column Gemv over `[K, N]` reads the weight at a stride
/// of `N` elements per lane (card 035), and the column-tile body that used to avoid that is retired; the
/// supported orientation is the checkpoint's `[N, K]` read through a `Transpose`, which `compile` folds into the
/// dense Gemv. [`widen_mismatched_matmul_dtypes`] leaves the mixed-dtype matmul in the graph so that `plan_eqn`
/// refuses it with `Capability::DtypeLowering`, rather than materializing a widened F32 copy of the weight.
///
/// A device with `widens_bf16_decode_weight` (NVPTX) is a deliberate asymmetry: it keeps the widening cast (it has two-byte BF16 storage and no packed lane),
/// so a `[K, N]` BF16 decode weight is materialized as an F32 copy there, doubling its bytes. That is a backend
/// capability difference, not a bug, and there is no PTX hardware row for `[K, N]` BF16 decode (the supported
/// `[N, K]` orientation is the dense Gemv on every backend).
pub fn matmul_bf16_column_gemv(
    caps: &DeviceCaps,
    a_dtype: DType,
    b_dtype: DType,
    odt: DType,
    m: usize,
    weight_rank: usize,
) -> bool {
    !caps.widens_bf16_decode_weight
        && a_dtype == DType::F32
        && b_dtype == DType::BF16
        && odt == DType::F32
        && m == 1
        && weight_rank == 2
}

/// Whether `op` reads its operand `position` as packed BF16 lanes on a typed value walk. One row per kernel
/// that decodes the Card 380 lane layout: the `DenseContraction` weight (card 380), the `DenseRowGather` table
/// (card 381), the `Cast` to F32 of a norm scale, bias or other elementwise const (card 1011,
/// `packed_bf16_to_f32`), and the narrow inputs a `FusedRow` or `ContractionEpilogue` region widens in
/// registers with a `FusedOp::Cast` step. A further reader is a row here plus its kernel, never a second
/// hand-written branch.
///
/// Every other operand of those operations is an ordinary F32 or I32 buffer, so the position matters:
/// a contraction activation or a row-gather index read through this table would be a const uploaded as
/// two-byte words for a kernel that reads four-byte ones.
fn packed_bf16_reader(op: &OpKind, position: usize) -> bool {
    let weight = (op.dense_weight().is_some()
        || matches!(
            op,
            OpKind::IndexedMatMul
                | OpKind::ContractionEpilogue {
                    head: ContractionHead::Indexed,
                    ..
                }
        ))
        && position == 1;
    weight
        || match op {
            OpKind::DenseRowGather { .. } | OpKind::Cast { to: DType::F32 } => position == 0,
            OpKind::FusedRow(region) => region.widens_input(LocalId::new(position)),
            // Operands `0..OPERANDS` are the head's; the region's local `n` is operand `OPERANDS + n - 1`.
            OpKind::ContractionEpilogue { head, region } => position
                .checked_sub(head.operands() - 1)
                .is_some_and(|local| local >= 1 && region.widens_input(LocalId::new(local))),
            _ => false,
        }
}

/// Card 380 and card 381: true if every consumer of `id` reads it through
/// `packed_bf16_reader`, and `id` does not escape the graph. This is the condition under
/// which a BF16 const may stay narrow on a typed value walk: those imported kernels are the only ones
/// there that read BF16, and they read it as packed `u32` lanes holding the checkpoint bytes.
///
/// A value with no consumer is not eligible: "every consumer" must not be vacuously true for a const
/// nothing reads, because the binder would then upload two-byte words no kernel can interpret.
///
/// Lives here with [`bf16_const_is_packed`] for one reason: the uploaded lane and the kernel that reads it
/// must be decided by one predicate, or a kernel silently reads a buffer at the wrong element width.
///
/// Holding here is necessary but not sufficient for a narrow const to be safe: it is only safe on a
/// lane that stores packed BF16, which is the typed value walk on each backend that has one - wgpu's,
/// and ROCm's since card 450 ROCm. Every other wgpu entry point rejects a BF16 value through
/// `crate::validate_wgpu_tensor_execution`.
pub fn bf16_const_feeds_only_packed_readers(
    g: &Graph,
    views: &HashMap<ValueId, Layout>,
    id: ValueId,
) -> bool {
    if g.liveness_roots().any(|root| root == id) {
        return false;
    }
    let mut consumed = false;
    for eqn in g.eqns() {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            if !matches!(operand, Operand::Value(value) if *value == id) {
                continue;
            }
            consumed = true;
            // A promoted view of the const is read in place by its consumer (the checkpoint-order `[E, N, K]`
            // expert stack an indexed contraction reads strided), so the const is packed exactly when that
            // view's readers are packed readers.
            let through_view = position == 0
                && views.contains_key(&eqn.out)
                && bf16_const_feeds_only_packed_readers(g, views, eqn.out);
            if !through_view && !packed_bf16_reader(&eqn.op, position) {
                return false;
            }
        }
    }
    consumed
}

/// A rewritten equation's operand map, keyed by the `ValueId` it reads: which equation index
/// produces it. Local to the two folds below (their own copy of `poot-graph-ir`'s private
/// `packed_consumer_map`/`value_input`/`is_graph_escape`, kept small rather than widening
/// `poot-graph-ir`'s public surface for two callers - Card 534a).
fn packed_consumer_map(g: &Graph) -> ValueTable<Vec<usize>> {
    let mut consumers = ValueTable::filled(g.values().len(), Vec::new());
    for (equation, eqn) in g.eqns().iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                consumers[*value].push(equation);
            }
        }
    }
    consumers
}

fn is_graph_escape(g: &Graph, value: ValueId) -> bool {
    g.liveness_roots().any(|root| root == value)
}

fn value_input(eqn: &Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

/// Card 380 (Card 534a: moved out of `poot-graph-ir`, whose `pub mod transform` let every crate that
/// depends on it - `poot-wgpu-device`, `poot-rocm-device`, `poot-ptx-device` included - name and sequence this pass
/// itself, defeating SC-001; Card 534a). Rewrite `MatMul(a, Transpose([1, 0], w))`
/// over a weight of an admitted dtype into one [`poot_graph_ir::OpKind::DenseContraction`] that reads `w` in checkpoint
/// `[N, K]` order, so the binder never materializes a widened F32 copy of it, when:
///
/// - the matmul takes exactly two value operands,
/// - operand 1 is produced by `Transpose { perm: [1, 0] }`,
/// - the transposed value has exactly one consumer and is not a graph escape (a validation witness
///   makes it one, so an observed transpose is left alone),
/// - the pre-transpose weight's dtype is an arithmetic float,
/// - `DenseContraction::infer` agrees with the matmul's own declared output type,
/// - and, for an F16 weight (Card 1007), the weight is a non-escaping const every consumer of which is a
///   transpose this pass folds, so its one packed storage serves every reader.
///
/// The grouped spelling `MatMul(a, Transpose([0, 2, 1], Reshape([G, N, K], w)))` over a rank-2 `[G * N, K]`
/// weight folds the same way into one [`poot_graph_ir::OpKind::GroupedDenseContraction`], the reshape being
/// dropped with the transpose (the reshape's value is single-use and not an escape, like the transpose's).
///
/// Value ids are preserved: the contraction keeps the matmul's own `out` and only the dead transpose
/// equation is dropped, so input, state, and validation bindings stay valid. Idempotent: a folded
/// graph has no `Transpose` left to match.
///
/// `pub(crate)`: reachable only through `compile` (Card 534a SC-001).
///
/// ```compile_fail,E0425
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::fold_dense_contractions(&g);
/// ```
pub(crate) fn fold_dense_contractions(g: &Graph) -> Result<Graph, EditError> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns().iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();
    let mut f16_folds = Vec::new();

    for (matmul_index, matmul) in g.eqns().iter().enumerate() {
        if !matmul.op.is_plain_matmul() || matmul.inputs.len() != 2 {
            continue;
        }
        let (Some(activation), Some(transposed)) = (value_input(matmul, 0), value_input(matmul, 1))
        else {
            continue;
        };
        let Some(&transpose_index) = producer.get(&transposed) else {
            continue;
        };
        let OpKind::Transpose { perm } = &g.eqns()[transpose_index].op else {
            continue;
        };
        if consumers[transposed].len() != 1 || is_graph_escape(g, transposed) {
            continue;
        }
        let Some(source) = value_input(&g.eqns()[transpose_index], 0) else {
            continue;
        };
        // The folded equations: the transpose, and for a grouped weight the reshape that blocks it.
        let (candidate, weight, folded) = match perm.as_slice() {
            [1, 0] => {
                let dtype = g.aval(source).dtype;
                (
                    OpKind::DenseContraction { weight: dtype },
                    source,
                    vec![transpose_index],
                )
            }
            [0, 2, 1] => {
                let Some(&reshape_index) = producer.get(&source) else {
                    continue;
                };
                let OpKind::Reshape { .. } = &g.eqns()[reshape_index].op else {
                    continue;
                };
                if consumers[source].len() != 1 || is_graph_escape(g, source) {
                    continue;
                }
                let Some(weight) = value_input(&g.eqns()[reshape_index], 0) else {
                    continue;
                };
                let (blocks, stored) = (g.aval(source), g.aval(weight));
                let &[groups, ..] = blocks.shape.as_slice() else {
                    continue;
                };
                if groups == 0 {
                    continue;
                }
                // Only a BF16 weight has grouped device rows; any other dtype keeps the traced chain.
                if stored.dtype != DType::BF16 {
                    continue;
                }
                // The reshape must only block the stored `[groups * N, K]` rows into `[groups, N, K]`.
                if stored.rank() != 2
                    || blocks.shape != [groups, stored.shape[0] / groups, stored.shape[1]]
                {
                    continue;
                }
                (
                    OpKind::GroupedDenseContraction {
                        weight: stored.dtype,
                        groups,
                    },
                    weight,
                    vec![transpose_index, reshape_index],
                )
            }
            _ => continue,
        };
        // `infer` below types any arithmetic float weight, and a weight that is not one is no operand of the
        // candidate; every float weight has a dense body or a typed refusal at planning.
        let dtype = g.aval(weight).dtype;
        if dtype.class() != DTypeClass::Arithmetic {
            continue;
        }
        let candidate_inputs = [g.aval(activation).clone(), g.aval(weight).clone()];
        if candidate.infer(&candidate_inputs).ok().as_ref() != Some(g.aval(matmul.out)) {
            continue;
        }
        removed.extend(folded.iter().copied());
        replacements.insert(
            matmul_index,
            Eqn {
                op: candidate,
                inputs: vec![Operand::Value(activation), Operand::Value(weight)],
                out: matmul.out,
                layer: matmul.layer,
            },
        );
        if dtype == DType::F16 {
            f16_folds.push((matmul_index, folded, weight));
        }
    }

    // Card 1007: an F16 weight is stored packed two elements per `u32` word for the contraction to decode, and
    // a buffer has one storage. So an F16 weight folds only when the fold leaves the contraction its sole
    // reader: a const (a computed value is written natively by its producer), not an escape, and every consumer
    // a transpose folded here. Otherwise each of its compositions stays exactly as traced. The decision is per
    // weight, so dropping one weight's folds never changes another's.
    let unpackable: HashSet<ValueId> = f16_folds
        .iter()
        .map(|&(_, _, weight)| weight)
        .filter(|&weight| {
            g.meta(weight).storage != Storage::Const
                || is_graph_escape(g, weight)
                || consumers[weight]
                    .iter()
                    .any(|consumer| !removed.contains(consumer))
        })
        .collect();
    for (matmul_index, folded, weight) in &f16_folds {
        if unpackable.contains(weight) {
            replacements.remove(matmul_index);
            for index in folded {
                removed.remove(index);
            }
        }
    }

    let eqns = g
        .eqns()
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| eqn.clone()))
        })
        .collect();
    crate::passes::with_eqns(g, eqns)
}

/// The table dtypes [`fold_dense_bf16_row_gathers`] rewrites into a [`poot_graph_ir::OpKind::DenseRowGather`]: the
/// ones a target has a packed-lane gather body for (Card 381's BF16 embedding table, Card 405's E4M3FN one). A
/// further narrow float is a row here plus a kernel that decodes it; `infer` already types the op over any
/// float or E4M3FN source.
const DENSE_ROW_GATHER_FOLD_SOURCES: &[DType] = &[DType::BF16, DType::E4M3FN];

/// Card 381 (Card 534a: moved out of `poot-graph-ir`, see [`fold_dense_contractions`]'s doc).
/// Fold `Cast(F32, [Reshape...], Gather { axis: 0 })` over a rank-2 narrow-float table into one
/// [`poot_graph_ir::OpKind::DenseRowGather`] that gathers rows and widens them in a single equation,
/// followed by the same reshapes over F32 values.
///
/// Embedding sibling of [`fold_dense_contractions`]: the only other ways to lower a
/// `[vocab, hidden]` BF16 token-embedding table are to widen it to F32 (multi-GB at real model dims)
/// or leave the gather's output BF16, which on wgpu is representable only as packed `u32` lanes (a
/// kernel producing packed lanes must write two elements per lane). Gathering and widening in one
/// equation avoids both.
///
/// The fold applies only when every condition holds, so it never changes what a graph means:
///
/// - the equation is `Cast { to: F32 }` over a single value operand,
/// - that operand is reached, through zero or more `Reshape` equations, from a `Gather { axis: 0 }`,
/// - every value on that chain has exactly one consumer and is not a graph escape (a validation
///   witness makes it one, so an observed intermediate is left alone),
/// - the table's dtype is in [`DENSE_ROW_GATHER_FOLD_SOURCES`],
/// - and `infer` agrees, at the gather and at every rebuilt reshape, with the type the folded graph
///   declares.
///
/// The `Cast`'s own value id is preserved (the last rebuilt equation writes it), so input, state and
/// validation bindings stay valid. The BF16 values the chain produced keep their `ValueMeta` entries
/// and lose their producers, as the transpose does in the contraction fold. Idempotent: a folded graph
/// holds no `Cast { to: F32 }` over a `Gather`.
///
/// `pub(crate)`: reachable only through `compile` (Card 534a SC-001).
///
/// ```compile_fail,E0425
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::fold_dense_bf16_row_gathers(&g);
/// ```
pub(crate) fn fold_dense_bf16_row_gathers(g: &Graph) -> Result<Graph, EditError> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns().iter().enumerate() {
        producer.insert(eqn.out, index);
    }

    let mut out = g.edit();
    let mut replacements: HashMap<usize, Vec<Eqn>> = HashMap::new();
    let mut removed = HashSet::new();

    for (cast_index, cast) in g.eqns().iter().enumerate() {
        if !matches!(cast.op, OpKind::Cast { to: DType::F32 }) || cast.inputs.len() != 1 {
            continue;
        }
        let Some(mut current) = value_input(cast, 0) else {
            continue;
        };
        // Walk back to the gather through movement that commutes with the widening. `Reshape` is the only
        // admitted link: it is a pure relabel of the same elements in the same order, so reshaping decoded
        // rows and decoding reshaped rows are the same values.
        let mut links = Vec::new();
        let chain = loop {
            let Some(&index) = producer.get(&current) else {
                break None;
            };
            if consumers[current].len() != 1 || is_graph_escape(g, current) {
                break None;
            }
            match g.eqns()[index].op {
                OpKind::Gather { axis: 0 } => break Some(index),
                OpKind::Reshape { .. } => {
                    let Some(source) = value_input(&g.eqns()[index], 0) else {
                        break None;
                    };
                    links.push(index);
                    current = source;
                }
                _ => break None,
            }
        };
        let Some(gather_index) = chain else {
            continue;
        };
        let gather = &g.eqns()[gather_index];
        let (Some(table), Some(index)) = (value_input(gather, 0), value_input(gather, 1)) else {
            continue;
        };
        let dtype = g.aval(table).dtype;
        if !DENSE_ROW_GATHER_FOLD_SOURCES.contains(&dtype) {
            continue;
        }
        let op = OpKind::DenseRowGather { source: dtype };
        let Ok(gathered) = op.infer(&[g.aval(table).clone(), g.aval(index).clone()]) else {
            continue;
        };
        if gathered.shape != g.aval(gather.out).shape {
            continue;
        }

        // Rebuild the chain over F32, innermost first, with the outermost equation writing the cast's own
        // value id. A fresh value is appended for every intermediate the chain still needs.
        let values_before = out.values().len();
        let mut rebuilt = Vec::with_capacity(links.len() + 1);
        let mut inputs = vec![Operand::Value(table), Operand::Value(index)];
        let mut current_op = op;
        let mut current_type = gathered;
        let mut agreed = true;
        for &link in links.iter().rev() {
            let link_op = g.eqns()[link].op.clone();
            let Ok(next_type) = link_op.infer(std::slice::from_ref(&current_type)) else {
                agreed = false;
                break;
            };
            if next_type.shape != g.aval(g.eqns()[link].out).shape {
                agreed = false;
                break;
            }
            let value = out
                .values_mut()
                .push(ValueMeta::new(current_type, Storage::Device, None));
            rebuilt.push(Eqn {
                op: current_op,
                inputs,
                out: value,
                layer: cast.layer,
            });
            inputs = vec![Operand::Value(value)];
            current_op = link_op;
            current_type = next_type;
        }
        if !agreed || &current_type != g.aval(cast.out) {
            // Nothing is committed until every check passes: drop this candidate's appended values, and
            // leave every equation of its chain exactly as it was.
            out.values_mut().truncate(values_before);
            continue;
        }
        rebuilt.push(Eqn {
            op: current_op,
            inputs,
            out: cast.out,
            layer: cast.layer,
        });
        removed.insert(gather_index);
        removed.extend(links);
        replacements.insert(cast_index, rebuilt);
    }

    *out.eqns_mut() = g
        .eqns()
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| vec![eqn.clone()]))
        })
        .flatten()
        .collect();
    out.commit()
}

/// Card 454 D1 (Card 534a: moved out of `poot-graph-ir`, see [`fold_dense_contractions`]'s doc).
/// Rewrite every [`poot_graph_ir::OpKind::Reduce`] whose axis is not the input's last axis into a
/// [`poot_graph_ir::OpKind::Transpose`] that moves that axis to the end, a last-axis reduce, and (when
/// `keepdim` put the size-1 axis in the wrong place) a [`poot_graph_ir::OpKind::Reshape`] back to the
/// original output shape.
///
/// `plan_eqn`'s `Reduce` arm only plans a last-axis reduce (its sole body is `kg::reduce_last_dt`,
/// whose rows are contiguous), refusing any other axis with a typed `Capability::NonLastAxisReduce`.
/// This pass keeps the IR primitive - no new kernel, no hand-fused region - by expressing the same
/// reduction through ops the planner already plans. Model tracers may still emit a non-last reduce
/// naturally (GLM-5.3-Flash's sparse FFN sums its `top_k` rows on axis 0); the rewrite happens at
/// backend preparation, so the pinned model graph is untouched.
///
/// Value preservation: for each output element the last-axis body folds the same element pairs in the
/// same `0..k` order as the original axis reduce (the transpose only reorders which axis those pairs
/// sit on), so the result is bit-identical, not merely close.
///
/// Properties:
///
/// - Only rewrites when every rebuilt shape agrees with what the graph already declares (checked
///   through `OpKind::infer`); a disagreement leaves the equation untouched and the planner's own
///   typed refusal as the failure.
/// - Preserves the reduce's output value id, so input/state/validation bindings and named outputs
///   survive.
/// - Idempotent: a graph with no non-last reduce (or a lowered one) comes back unchanged.
///
/// Card 534a runs it for every backend, through `compile`'s pipeline (and
/// the typed exact-value walk): before that card, only the wgpu copy of this preparation ran it, so a
/// MiniMax-M2 or GLM-5.3-Flash MoE gating reduce planned on wgpu and refused on ROCm and PTX
/// even though ROCm and PTX plan `Reduce` through the same `plan_eqn`, not a
/// separate planner.
///
/// `pub(crate)`: reachable only through `compile` (Card 534a SC-001).
///
/// ```compile_fail,E0425
/// let g = poot_graph_ir::Graph::default();
/// let _ = poot_graph_plan::lower_nonlast_reduces(&g);
/// ```
pub(crate) fn lower_nonlast_reduces(g: &Graph) -> Result<Graph, EditError> {
    let mut out = g.edit();
    let mut replacements: HashMap<usize, Vec<Eqn>> = HashMap::new();

    for (index, eqn) in g.eqns().iter().enumerate() {
        let OpKind::Reduce { op, axis, keepdim } = eqn.op else {
            continue;
        };
        let Some(Operand::Value(input)) = eqn.inputs.first().copied() else {
            continue;
        };
        let input_aval = g.aval(input);
        let rank = input_aval.shape.len();
        if axis + 1 >= rank {
            continue;
        }

        // Move the reduced axis last; the remaining axes keep their relative order, so dropping
        // the moved axis afterwards yields exactly the original output shape.
        let mut perm: Vec<usize> = (0..rank).filter(|&candidate| candidate != axis).collect();
        perm.push(axis);
        let transposed_shape: Vec<usize> = perm.iter().map(|&p| input_aval.shape[p]).collect();
        let transposed_aval = TensorType::new(transposed_shape, input_aval.dtype);
        let Ok(reduced_aval) = OpKind::Reduce {
            op,
            axis: rank - 1,
            keepdim,
        }
        .infer(std::slice::from_ref(&transposed_aval)) else {
            continue;
        };
        let Some(out_aval) = g.values().get(eqn.out).map(|meta| &meta.aval) else {
            continue;
        };

        // Nothing is committed until every check passes: the fresh intermediates are appended to
        // a scratch prefix and dropped again on any disagreement.
        let committed_before = out.values().len();
        let transposed_id =
            out.values_mut()
                .push(ValueMeta::new(transposed_aval, Storage::Device, None));

        let mut rebuilt = Vec::with_capacity(3);
        rebuilt.push(Eqn {
            op: OpKind::Transpose { perm },
            inputs: vec![Operand::Value(input)],
            out: transposed_id,
            layer: eqn.layer,
        });

        let agreed = if reduced_aval == *out_aval {
            // keepdim=false: dropping the moved axis leaves the original shape, so the reduce
            // writes the original output id directly.
            rebuilt.push(Eqn {
                op: OpKind::Reduce {
                    op,
                    axis: rank - 1,
                    keepdim,
                },
                inputs: vec![Operand::Value(transposed_id)],
                out: eqn.out,
                layer: eqn.layer,
            });
            true
        } else if reduced_aval.numel() == out_aval.numel() {
            // keepdim=true: the size-1 axis sits where the reduced axis was, not at the end.
            let reduced_id =
                out.values_mut()
                    .push(ValueMeta::new(reduced_aval, Storage::Device, None));
            rebuilt.push(Eqn {
                op: OpKind::Reduce {
                    op,
                    axis: rank - 1,
                    keepdim,
                },
                inputs: vec![Operand::Value(transposed_id)],
                out: reduced_id,
                layer: eqn.layer,
            });
            rebuilt.push(Eqn {
                op: OpKind::Reshape {
                    shape: out_aval.shape.clone(),
                },
                inputs: vec![Operand::Value(reduced_id)],
                out: eqn.out,
                layer: eqn.layer,
            });
            true
        } else {
            false
        };

        if !agreed {
            out.values_mut().truncate(committed_before);
            continue;
        }
        replacements.insert(index, rebuilt);
    }

    *out.eqns_mut() = g
        .eqns()
        .iter()
        .enumerate()
        .flat_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .unwrap_or_else(|| vec![eqn.clone()])
        })
        .collect();
    out.commit()
}

/// Widen a `MatMul`/`Binary` operand whose dtype does not match the eqn's output dtype
/// back to the output dtype via an explicit `Cast`: every matmul and binary body reads one element type. The
/// one mismatch left in place is the BF16 `[K, N]` decode projection ([`matmul_bf16_column_gemv`]).
///
/// Idempotent and a no-op on any graph with no mismatched-dtype eqn. The pass only inserts `Cast`
/// equations; it never changes a value's declared dtype, and no other pass retypes a const either (Card
/// 1011). A BF16 const stays BF16 on every backend, and the `Cast{BF16 -> F32}` this pass puts in front of
/// it reads the lane the storage plan gave the const ([`bf16_const_is_packed`]).
///
/// Card 235: needed before decode capture on every backend: a mismatched-dtype decode matmul is always
/// widened by this pass (except the BF16-weight decode GEMV, which reads the weight packed).
///
/// `Binary` is covered because a bf16 decode graph's RMSNorm `x * gamma` compiles to a plain
/// `Binary(Mul)` against a `Const` weight whose dtype can differ from the activation's; the operand is
/// widened to the eqn's output dtype by an explicit `Cast`.
///
/// `Gather { axis: 0 }` is also covered (first operand only: the table, never the index). A token-embedding
/// lookup is `gather(embed_const, token)`, a raw `Storage::Const` table operand. `plan_eqn`'s `Gather` arm
/// (unlike `MatMul`'s) has no operand/output dtype-mismatch guard: it picks the kernel's element type from the
/// output dtype (`kg::gather_axis0_dt(_, fty(odt), _)`), so a Const whose dtype differs from the output's,
/// read through that kernel, is silent corruption, not a `plan_eqn` rejection (`bf16 GRAPH MISMATCH (8/8)` in
/// `probe_bf16_graph`).
///
/// `Slice` is covered the same way (evidence: `ptx_prefill_check_bf16`'s per-eqn scan): RoPE's
/// `cos`/`sin` tables are `TensorType::f32(...)` unconditionally at trace (like every norm gamma and
/// bias), and `rope_prefill`'s `cos[0..n]`/`sin[0..n]` is a direct
/// `Slice` of that raw Const. `plan_eqn`'s `Slice` arm has the same no-guard shape as `Gather`'s (a
/// zero-cost `Plan::View` or an imported kernel, both keyed off `fty(odt)` alone), with the same
/// silent corruption (`slice ax=0 0..5 ... maxabs=2.59e38 nonfinite=20/320`). The non-axis-0 general
/// `Gather` is out of scope (no known real trace uses a raw Const as its data operand there).
/// `Unary`/`Cast`/`Reduce`/`Transpose`/`Reshape`/`Concat` etc. are not covered: no known real trace
/// feeds them a raw Const directly (always through a Gather/Slice/compute op first); add them the same
/// way if one does.
pub fn widen_mismatched_matmul_dtypes(g: &Graph, caps: &DeviceCaps) -> Result<Graph, EditError> {
    // Card 273: widen a single-value-operand movement op (`Gather{axis:0}`, `Slice`) whose one operand
    // is directly a mismatched-dtype `Storage::Const`: inserts a Cast on that operand only, so the
    // eqn's output dtype and every downstream consumer are untouched. Both ops can plan to a zero-cost
    // `Plan::View`/imported-kernel path that reads the operand's real device bytes at the eqn's output
    // elem width (`fty(odt)`) with no mismatch guard, so a Const read at an element width its dtype does not
    // have is silent corruption (`probe_bf16_graph` for Gather, `ptx_prefill_check_bf16` for
    // Slice on `rope.cos`/`rope.sin`, both `TensorType::f32(...)` at trace regardless of `proj_dtype`).
    let widen_single_operand = |out: &mut GraphEdit, eqn: &Eqn| -> bool {
        let table_id = match eqn.inputs.first() {
            Some(Operand::Value(v)) => *v,
            _ => return false,
        };
        let odt = g.aval(eqn.out).dtype;
        if g.aval(table_id).dtype == odt {
            return false;
        }
        let aval = TensorType::new(out.aval(table_id).shape.clone(), odt);
        let nv = out
            .values_mut()
            .push(ValueMeta::new(aval, Storage::Device, None));
        out.eqns_mut().push(Eqn {
            op: OpKind::Cast { to: odt },
            inputs: vec![Operand::Value(table_id)],
            out: nv,
            layer: eqn.layer,
        });
        let mut inputs = eqn.inputs.clone();
        inputs[0] = Operand::Value(nv);
        out.eqns_mut().push(Eqn {
            op: eqn.op.clone(),
            inputs,
            out: eqn.out,
            layer: eqn.layer,
        });
        true
    };

    let mut out = g.edit();
    out.eqns_mut().clear();
    for eqn in g.eqns() {
        if matches!(eqn.op, OpKind::Gather { axis: 0 } | OpKind::Slice { .. }) {
            if !widen_single_operand(&mut out, eqn) {
                out.eqns_mut().push(eqn.clone());
            }
            continue;
        }
        let is_matmul = matches!(eqn.op, OpKind::MatMul { .. });
        let widenable = is_matmul || matches!(eqn.op, OpKind::Binary(_));
        let (a_id, b_id) = match (eqn.inputs.first(), eqn.inputs.get(1)) {
            (Some(Operand::Value(a)), Some(Operand::Value(b))) if widenable => (*a, *b),
            _ => {
                out.eqns_mut().push(eqn.clone());
                continue;
            }
        };
        let odt = g.aval(eqn.out).dtype;
        let a_dt = g.aval(a_id).dtype;
        let b_dt = g.aval(b_id).dtype;
        if a_dt == odt && b_dt == odt {
            out.eqns_mut().push(eqn.clone());
            continue;
        }
        if is_matmul {
            let out_shape = &g.aval(eqn.out).shape;
            let r = out_shape.len();
            if r < 2 {
                // MatMul always infers a rank >= 2 output (see OpKind::infer); defensive only.
                out.eqns_mut().push(eqn.clone());
                continue;
            }
            let m = out_shape[r - 2];
            // A BF16 `[K, N]` decode projection has no lowering: a widening Cast would materialize an F32
            // copy of the weight, so it stays mixed and `plan_eqn` refuses it by name.
            if matmul_bf16_column_gemv(caps, a_dt, b_dt, odt, m, g.aval(b_id).shape.len()) {
                out.eqns_mut().push(eqn.clone());
                continue;
            }
        }
        let cast_to = |out: &mut GraphEdit, id: ValueId, dt: DType| -> ValueId {
            if out.aval(id).dtype == dt {
                return id;
            }
            let aval = TensorType::new(out.aval(id).shape.clone(), dt);
            let nv = out
                .values_mut()
                .push(ValueMeta::new(aval, Storage::Device, None));
            out.eqns_mut().push(Eqn {
                op: OpKind::Cast { to: dt },
                inputs: vec![Operand::Value(id)],
                out: nv,
                layer: eqn.layer,
            });
            nv
        };
        let mut inputs = eqn.inputs.clone();
        let na = cast_to(&mut out, a_id, odt);
        inputs[0] = Operand::Value(na);
        let nb = cast_to(&mut out, b_id, odt);
        inputs[1] = Operand::Value(nb);
        // Widened to the output dtype, a matmul's operands already carry its accumulate dtype.
        let op = if is_matmul {
            OpKind::matmul()
        } else {
            eqn.op.clone()
        };
        out.eqns_mut().push(Eqn {
            op,
            inputs,
            out: eqn.out,
            layer: eqn.layer,
        });
    }
    out.commit()
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{Builder, Graph, Storage, TensorType};
    use poot_target::AmdArch;

    use poot_target::Backend;
    use poot_tensor::DType;

    use crate::test_support::compiled_graph;

    /// `a x w^T -> f32` over two consts of the given dtypes, `m x 32` against a checkpoint-order `[32, 32]` weight:
    /// the shape a model traces, which `compile` folds into a dense contraction that reads the weight as stored.
    fn checkpoint_matmul(m: usize, a_dtype: DType, w_dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let a = b
            .constant("a", TensorType::new(vec![m, 32], a_dtype))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![32, 32], w_dtype))
            .unwrap();
        let wt = b.transpose(w, vec![1, 0]).unwrap();
        let mm = if a_dtype == w_dtype {
            b.matmul_f32_accumulate(a, wt).unwrap()
        } else {
            b.matmul(a, wt).unwrap()
        };
        b.finish(mm).unwrap()
    }

    /// Card 1011: no pass retypes a BF16 const, on any backend or AMD arch family, over the checkpoint matmuls
    /// (a prefill matmul no backend has a packed reader for included): the declared dtype of every const is the
    /// dtype it was traced with, so the executor uploads the stored words. A BF16 activation has no lowering on
    /// some targets; there `compile` refuses by name (`Refusal`) instead of retyping anything to plan it.
    ///
    /// Mutation: retype every BF16 const to F32 at the top of `widen_mismatched_matmul_dtypes`; every row
    /// of this test goes red naming the const.
    #[test]
    fn compile_never_retypes_a_bf16_const() {
        let backends = [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::AmdGcn(AmdArch::new("gfx90a", 64)),
            Backend::AmdGcn(AmdArch::new("gfx1030", 32)),
            Backend::AmdGcn(AmdArch::new("gfx1200", 32)),
        ];
        // (label, graph, whether every target must plan it)
        let graphs = [
            (
                "mixed bf16 prefill",
                checkpoint_matmul(32, DType::BF16, DType::BF16),
                false,
            ),
            (
                "mixed bf16 decode",
                checkpoint_matmul(1, DType::BF16, DType::BF16),
                false,
            ),
            (
                "bf16 checkpoint prefill",
                checkpoint_matmul(32, DType::F32, DType::BF16),
                true,
            ),
            (
                "bf16 checkpoint decode",
                checkpoint_matmul(1, DType::F32, DType::BF16),
                true,
            ),
        ];
        for backend in backends {
            for (label, g, must_plan) in &graphs {
                let caps = poot_test_util::device_caps::default_caps_for(backend);
                let prepared = match compiled_graph(g, backend, &caps) {
                    Ok(prepared) => prepared,
                    Err(crate::CompileError::Plan(error))
                        if !must_plan && matches!(*error, crate::PlanError::Refused(_)) =>
                    {
                        continue;
                    }
                    Err(error) => panic!("{label} on {backend:?}: {error}"),
                };
                for id in g
                    .values()
                    .ids()
                    .filter(|&id| g.meta(id).storage == Storage::Const)
                {
                    assert_eq!(
                        prepared.aval(id).dtype,
                        g.aval(id).dtype,
                        "{label} on {backend:?}: const {id} was retyped"
                    );
                }
            }
        }
    }
}

/// Card 534a SC-002: `compile` is the one preparation pipeline feeding every target
/// (R481-003's stated fix), so a MiniMax-M2-shaped MoE gating reduce over axis 0 - which the wgpu-only
/// `lower_nonlast_reduces` copy used to leave unlowered for ROCm and PTX - now legalizes identically on
/// every backend.
#[cfg(test)]
mod compile_preparation_tests {
    use poot_graph_ir::{Builder, OpKind, RedOp, TensorType};
    use poot_target::{AmdArch, Backend};
    use poot_tensor::DType;

    use crate::test_support::compiled_graph;

    /// A tiny MoE-router-shaped graph: `top_k`-style expert weights summed on axis 0 (a non-last-axis
    /// reduce), the same shape MiniMax-M2's gating sum traces.
    fn moe_gate_reduce_graph() -> poot_graph_ir::Graph {
        let b = Builder::for_tests();
        let weights = b
            .constant("expert_weights", TensorType::f32(vec![4, 3]))
            .unwrap();
        let out = b.reduce(RedOp::Sum, weights, 0, false).unwrap();
        b.finish(out).unwrap()
    }

    #[test]
    fn lowers_the_non_last_axis_reduce_identically_on_every_backend() {
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let prepared = compiled_graph(
                &moe_gate_reduce_graph(),
                backend,
                &poot_test_util::device_caps::default_caps_for(backend),
            )
            .unwrap();
            assert!(
                !prepared
                    .eqns()
                    .iter()
                    .any(|eqn| matches!(eqn.op, OpKind::Reduce { axis: 0, .. })),
                "{backend:?}: the axis-0 reduce must be lowered away, not left for the planner to refuse"
            );
            assert!(
                prepared
                    .eqns()
                    .iter()
                    .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. })),
                "{backend:?}: `lower_nonlast_reduces` moves the reduced axis to the end with a transpose"
            );
        }
    }

    /// Card 1011: a prefill matmul (`m = 4`, not the decode GEMV) over a BF16 weight no backend folds into a
    /// packed contraction keeps the weight BF16. Nothing retypes it to F32: the widening is the planned
    /// `Cast{BF16 -> F32}` equation in front of the matmul, which a packed or native cast body reads.
    #[test]
    fn a_bf16_weight_stays_bf16_and_is_cast_in_the_graph() {
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![4, 8], DType::F32))
            .unwrap();
        let weight = b
            .constant("weight", TensorType::new(vec![8, 16], DType::BF16))
            .unwrap();
        let out = b.matmul(x, weight).unwrap();
        let g = b.finish(out).unwrap();

        for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
            let prepared = compiled_graph(
                &g,
                backend,
                &poot_test_util::device_caps::default_caps_for(backend),
            )
            .unwrap();
            assert_eq!(prepared.aval(weight.id()).dtype, DType::BF16, "{backend:?}");
            assert!(
                prepared.eqns().iter().any(|eqn| matches!(
                    eqn.op,
                    OpKind::Cast { to: DType::F32 }
                ) && matches!(eqn.inputs.as_slice(), [poot_graph_ir::Operand::Value(v)] if *v == weight.id())),
                "{backend:?}: the weight reaches the matmul through an explicit Cast"
            );
        }
    }
}

/// Cards 380/381, moved from `poot-graph-ir` in Card 534a: `fold_dense_contractions`
/// and `fold_dense_bf16_row_gathers` are `pub(crate)` here now, so their own unit tests moved with them
/// (`poot-graph-ir` can no longer see them either).
#[cfg(test)]
mod fold_dense_bf16_tests {
    use poot_graph_ir::graph::{Eqn, Operand, Storage, ValueMeta};
    use poot_graph_ir::test_support::graph_fixtures::derive;
    use poot_graph_ir::{Builder, Graph, OpKind, Slot, TensorType, ValueId};
    use poot_tensor::DType;

    use super::{fold_dense_bf16_row_gathers, fold_dense_contractions};

    /// The Qwen3.8-27B LM head shape: a BF16 `[N, K]` checkpoint constant, transposed in the graph,
    /// then used as a matmul weight against an F32 activation.
    fn lm_head_shaped_graph(weight_dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let x = b
            .constant("activation", TensorType::f32(vec![1, 2, 5]))
            .unwrap();
        let weight = b
            .constant("lm_head.weight", TensorType::new(vec![7, 5], weight_dtype))
            .unwrap();
        let logits =
            poot_graph_ir::ops::linear(&b, x, b.transpose(weight, vec![1, 0]).unwrap(), None)
                .unwrap();
        b.finish(logits).unwrap()
    }

    fn const_id(g: &Graph, name: &str) -> ValueId {
        g.values()
            .iter_enumerated()
            .find(|(_, value)| value.name().as_deref() == Some(name))
            .map(|(id, _)| id)
            .expect("named constant")
    }

    /// `Eqn` and `Operand` carry no `PartialEq`, so compare equation lists structurally: the op's name,
    /// its value operands in order, and its output.
    fn eqn_summary(g: &Graph) -> Vec<(String, Vec<ValueId>, ValueId)> {
        g.eqns()
            .iter()
            .map(|eqn| {
                let inputs = eqn
                    .inputs
                    .iter()
                    .filter_map(|operand| match operand {
                        Operand::Value(value) => Some(*value),
                        Operand::Lit(_) => None,
                    })
                    .collect();
                (eqn.op.name(), inputs, eqn.out)
            })
            .collect()
    }

    /// `matmul(x [3, 2, 5], transpose(reshape(w [12, 5], [3, 4, 5]), [0, 2, 1]))`: a grouped projection.
    fn grouped_graph(weight_dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let x = b
            .constant("activation", TensorType::f32(vec![3, 2, 5]))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![12, 5], weight_dtype))
            .unwrap();
        let blocks = b.reshape(w, vec![3, 4, 5]).unwrap();
        let out = b
            .matmul(x, b.transpose(blocks, vec![0, 2, 1]).unwrap())
            .unwrap();
        b.finish(out).unwrap()
    }

    /// POOT-1048. Only a BF16 grouped weight folds (into one `GroupedDenseContraction` keeping the matmul's
    /// output id); an F32 or F16 one has no grouped device row and is left exactly as traced.
    ///
    /// Mutation: drop the BF16 check in the fold's `[0, 2, 1]` arm; the F32 chain folds into an op the planner refuses
    /// and this row goes red.
    #[test]
    fn a_grouped_weight_folds_only_when_it_is_bf16() {
        let graph = grouped_graph(DType::BF16);
        let folded = fold_dense_contractions(&graph).unwrap();
        assert_eq!(folded.eqns().len(), 1, "{:?}", eqn_summary(&folded));
        assert_eq!(
            folded.eqns()[0].op,
            OpKind::GroupedDenseContraction {
                weight: DType::BF16,
                groups: 3
            }
        );
        assert_eq!(folded.eqns()[0].out, graph.result());
        for dtype in [DType::F32, DType::F16] {
            let graph = grouped_graph(dtype);
            assert_eq!(
                eqn_summary(&fold_dense_contractions(&graph).unwrap()),
                eqn_summary(&graph),
                "{dtype:?}"
            );
        }
    }

    /// Card 380 FR-004. The fold replaces the matmul and drops the transpose, keeps the matmul's own output
    /// value id and type, and points the contraction at the constant in its checkpoint order.
    #[test]
    fn fold_matches_only_the_transposed_bf16_weight_shape() {
        let graph = lm_head_shaped_graph(DType::BF16);
        let folded = fold_dense_contractions(&graph).unwrap();

        assert_eq!(folded.result(), graph.result());
        assert_eq!(folded.aval(folded.result()), graph.aval(graph.result()));
        assert!(
            !folded
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. })),
            "the transpose is the copy this fold exists to remove"
        );
        let index = folded
            .eqns()
            .iter()
            .position(|eqn| matches!(eqn.op, OpKind::DenseContraction { .. }))
            .expect("the LM head folds");
        assert_eq!(
            folded.eqns()[index].op,
            OpKind::DenseContraction {
                weight: DType::BF16
            }
        );
        assert_eq!(
            eqn_summary(&folded)[index],
            (
                folded.eqns()[index].op.name(),
                vec![
                    const_id(&graph, "activation"),
                    const_id(&graph, "lm_head.weight")
                ],
                graph.result()
            ),
            "the contraction reads the activation and the constant in checkpoint order"
        );

        // Idempotent: a folded graph has no transpose left to match.
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&folded).unwrap()),
            eqn_summary(&folded),
            "the fold must be idempotent"
        );

        // An F32 weight is in the table (Card 645): it folds the same way, into an F32 contraction.
        let f32_weight = fold_dense_contractions(&lm_head_shaped_graph(DType::F32)).unwrap();
        assert!(
            f32_weight
                .eqns()
                .iter()
                .any(|eqn| eqn.op == OpKind::DenseContraction { weight: DType::F32 }),
            "an F32 checkpoint weight folds"
        );
        assert!(
            !f32_weight
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
        );

        // An F16 const weight is in the table (Card 1007) and the contraction is its sole reader: it folds.
        let f16_weight = fold_dense_contractions(&lm_head_shaped_graph(DType::F16)).unwrap();
        assert!(
            f16_weight
                .eqns()
                .iter()
                .any(|eqn| eqn.op == OpKind::DenseContraction { weight: DType::F16 }),
            "an F16 checkpoint weight folds"
        );
        // An F16 weight that is not a const cannot be stored packed (its producer writes it natively), so the
        // composition is left exactly as traced.
        let computed_f16 = {
            let b = Builder::for_tests();
            let x = b
                .constant("activation", TensorType::f32(vec![1, 2, 5]))
                .unwrap();
            let weight = b
                .slot(Slot::Activation, TensorType::new(vec![7, 5], DType::F16))
                .unwrap();
            let logits =
                poot_graph_ir::ops::linear(&b, x, b.transpose(weight, vec![1, 0]).unwrap(), None)
                    .unwrap();
            b.finish(logits).unwrap()
        };
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&computed_f16).unwrap()),
            eqn_summary(&computed_f16)
        );

        // The transposed value escapes the graph, so something outside reads it and the fold cannot
        // remove the equation that produces it. Its consumer count is still exactly one.
        let transposed = graph
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
            .expect("the traced graph transposes")
            .out;
        let escaping = derive(&graph, |draft| draft.set_result(transposed)).unwrap();
        assert!(escaping.liveness_roots().any(|root| root == transposed));
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&escaping).unwrap()),
            eqn_summary(&escaping),
            "an escaping transpose must not fold"
        );
    }

    /// Card 645: the F32 contraction the fold makes is the matmul it replaced. `eval` of the folded graph
    /// (the oracle's `DenseContraction` arm) equals `eval` of the traced `Transpose` + `MatMul`, bit for
    /// bit, on a weight whose `[N, K]` and `[K, N]` readings differ.
    #[test]
    fn folded_f32_contraction_evaluates_as_the_traced_matmul() {
        use poot_eval::{EvalOptions, Value, eval};
        use poot_tensor::HostTensor;
        use std::collections::HashMap;

        let graph = lm_head_shaped_graph(DType::F32);
        let folded = fold_dense_contractions(&graph).unwrap();
        assert!(
            folded
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::DenseContraction { .. })),
            "the fixture folds"
        );
        let fill = |n: usize, seed: usize| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 7 + seed * 13) % 11) as f32 - 5.0)
                .collect()
        };
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        inputs.insert(
            const_id(&graph, "activation"),
            Value::from(HostTensor::f32(vec![1, 2, 5], fill(10, 1))),
        );
        inputs.insert(
            const_id(&graph, "lm_head.weight"),
            Value::from(HostTensor::f32(vec![7, 5], fill(35, 2))),
        );
        let run = |g: &Graph| {
            eval(g, &inputs, EvalOptions::default())
                .unwrap()
                .output
                .into_host()
                .unwrap()
        };
        let (traced, contracted) = (run(&graph), run(&folded));
        assert_eq!(contracted.shape(), traced.shape());
        let bits = |tensor: &HostTensor| -> Vec<u32> {
            tensor
                .as_f32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        assert_eq!(bits(&contracted), bits(&traced));
    }

    /// Card 380 FR-005. A transposed value something else also reads is a real materialization: folding it
    /// away would leave the other consumer reading a value the graph no longer produces.
    #[test]
    fn fold_declines_a_transpose_with_a_second_consumer() {
        let traced = lm_head_shaped_graph(DType::BF16);
        let transposed = traced
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Transpose { .. }))
            .expect("the traced graph transposes")
            .out;
        let graph = derive(&traced, |draft| {
            let widened = draft.values_mut().push(ValueMeta::new(
                TensorType::f32(vec![5, 7]),
                Storage::Device,
                None,
            ));
            draft.eqns_mut().push(Eqn {
                op: OpKind::Cast { to: DType::F32 },
                inputs: vec![Operand::Value(transposed)],
                out: widened,
                layer: None,
            });
        })
        .unwrap();
        assert_eq!(
            eqn_summary(&fold_dense_contractions(&graph).unwrap()),
            eqn_summary(&graph)
        );
    }

    /// The Qwen3.8-27B token-embedding shape: a BF16 `[V, R]` checkpoint constant gathered along axis 0 by
    /// an I32 token id, reshaped, then widened to F32.
    fn embedding_shaped_graph(table_dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let table = b
            .constant(
                "model.language_model.embed_tokens.weight",
                TensorType::new(vec![7, 5], table_dtype),
            )
            .unwrap();
        let token = b
            .constant("token", TensorType::new(vec![3], DType::I32))
            .unwrap();
        let rows = b.gather(table, 0, token).unwrap();
        let shaped = b.reshape(rows, vec![1, 3, 5]).unwrap();
        let widened = b.cast(shaped, DType::F32).unwrap();
        b.finish(widened).unwrap()
    }

    fn eqn_index(g: &Graph, matches: impl Fn(&Eqn) -> bool) -> usize {
        g.eqns().iter().position(matches).expect("equation")
    }

    /// Card 381 FR-004. The fold replaces the widening cast with one row gather over the checkpoint
    /// constant, drops the gather and the reshape it walked through, and keeps the cast's own output id and
    /// type.
    #[test]
    fn fold_matches_only_a_widened_axis0_gather() {
        let graph = embedding_shaped_graph(DType::BF16);
        let folded = fold_dense_bf16_row_gathers(&graph).unwrap();

        assert_eq!(folded.result(), graph.result());
        assert_eq!(folded.aval(folded.result()), graph.aval(graph.result()));
        assert!(
            !folded
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Gather { .. })),
            "the BF16 gather is what this fold exists to remove"
        );
        assert!(
            !folded
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::Cast { .. })),
            "the widening cast is folded into the row gather"
        );
        let index = eqn_index(&folded, |eqn| {
            matches!(eqn.op, OpKind::DenseRowGather { .. })
        });
        assert_eq!(
            folded.eqns()[index].op,
            OpKind::DenseRowGather {
                source: DType::BF16
            }
        );
        assert_eq!(
            eqn_summary(&folded)[index],
            (
                folded.eqns()[index].op.name(),
                vec![
                    const_id(&graph, "model.language_model.embed_tokens.weight"),
                    const_id(&graph, "token")
                ],
                folded.eqns()[index].out
            ),
            "the row gather reads the checkpoint constant and the token id directly"
        );

        // Idempotent: a folded graph has no widened gather left to match.
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&folded).unwrap()),
            eqn_summary(&folded),
            "the fold must be idempotent"
        );

        // An F32 table is not in the admitted table, so the composition is left exactly as traced.
        let f32_table = embedding_shaped_graph(DType::F32);
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&f32_table).unwrap()),
            eqn_summary(&f32_table)
        );
    }

    /// Card 381 FR-004. The reshape the fold walks through is rebuilt over F32 values, so the cast's own id
    /// keeps its declared `[1, 3, 5]` shape and the row gather writes the `[3, 5]` block underneath it.
    #[test]
    fn fold_rebuilds_the_reshape_chain_over_f32() {
        let graph = embedding_shaped_graph(DType::BF16);
        let folded = fold_dense_bf16_row_gathers(&graph).unwrap();

        let gather = eqn_index(&folded, |eqn| {
            matches!(eqn.op, OpKind::DenseRowGather { .. })
        });
        let reshape = eqn_index(&folded, |eqn| matches!(eqn.op, OpKind::Reshape { .. }));
        assert_eq!(
            folded.aval(folded.eqns()[gather].out),
            &TensorType::f32(vec![3, 5]),
            "the row gather writes decoded F32 rows, not packed BF16"
        );
        assert_eq!(
            eqn_summary(&folded)[reshape],
            (
                folded.eqns()[reshape].op.name(),
                vec![folded.eqns()[gather].out],
                graph.result()
            ),
            "the rebuilt reshape reads the gathered rows and writes the cast's own id"
        );
        assert_eq!(
            folded.eqns()[reshape].op,
            OpKind::Reshape {
                shape: vec![1, 3, 5]
            }
        );
    }

    /// Card 381 FR-005. A gathered value something else also reads is a real materialization: folding it
    /// away would leave the other consumer reading a value the graph no longer produces.
    #[test]
    fn fold_declines_a_gather_with_a_second_consumer() {
        let traced = embedding_shaped_graph(DType::BF16);
        let rows =
            traced.eqns()[eqn_index(&traced, |eqn| matches!(eqn.op, OpKind::Gather { .. }))].out;
        let graph = derive(&traced, |draft| {
            let second = draft.values_mut().push(ValueMeta::new(
                TensorType::f32(vec![3, 5]),
                Storage::Device,
                None,
            ));
            draft.eqns_mut().push(Eqn {
                op: OpKind::Cast { to: DType::F32 },
                inputs: vec![Operand::Value(rows)],
                out: second,
                layer: None,
            });
        })
        .unwrap();
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph).unwrap()),
            eqn_summary(&graph)
        );
    }

    /// Card 381 FR-005. Something outside the graph reads the gathered rows, so the equation that produces
    /// them cannot be removed. Its consumer count is still exactly one.
    #[test]
    fn fold_declines_an_escaping_gather() {
        let traced = embedding_shaped_graph(DType::BF16);
        let rows =
            traced.eqns()[eqn_index(&traced, |eqn| matches!(eqn.op, OpKind::Gather { .. }))].out;
        let graph = derive(&traced, |draft| draft.set_result(rows)).unwrap();
        assert!(graph.liveness_roots().any(|root| root == rows));
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph).unwrap()),
            eqn_summary(&graph),
            "an escaping gather must not fold"
        );
    }

    /// Card 381 FR-005. `Reshape` is the only admitted link. A transpose between the gather and the cast
    /// would be correct to fold through - it relabels the same elements - but no production graph has one,
    /// and the walk admits exactly what production graphs contain.
    #[test]
    fn fold_declines_a_non_reshape_link() {
        let b = Builder::for_tests();
        let table = b
            .constant(
                "model.language_model.embed_tokens.weight",
                TensorType::new(vec![7, 5], DType::BF16),
            )
            .unwrap();
        let token = b
            .constant("token", TensorType::new(vec![3], DType::I32))
            .unwrap();
        let rows = b.gather(table, 0, token).unwrap();
        let moved = b.transpose(rows, vec![1, 0]).unwrap();
        let widened = b.cast(moved, DType::F32).unwrap();
        let graph = b.finish(widened).unwrap();
        assert_eq!(
            eqn_summary(&fold_dense_bf16_row_gathers(&graph).unwrap()),
            eqn_summary(&graph)
        );
    }
}

/// Card 454 D1, moved from `poot-graph-ir` in Card 534a: `lower_nonlast_reduces` is
/// `pub(crate)` here now, so its own unit test moved with it.
#[cfg(test)]
mod lower_nonlast_reduces_tests {
    use poot_graph_ir::graph::Operand;
    use poot_graph_ir::{Builder, OpKind, RedOp, TensorType};

    use super::lower_nonlast_reduces;

    /// Card 454 D1: a reduce whose axis is not last (GLM-5.3-Flash's sparse FFN sums its `top_k`
    /// rows on axis 0 of `[top_k, hidden]`) becomes `Transpose` + a last-axis `Reduce`, keeping the
    /// original output value id and declared shape, and a `keepdim` reduce gets the `Reshape` back
    /// to its declared `[1, hidden]`. A last-axis reduce is left alone, and lowering is idempotent.
    #[test]
    fn lower_nonlast_reduces_moves_a_non_last_reduce_to_the_last_axis() {
        // keepdim = false: [2, 4] --sum axis 0--> [4].
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 4])).unwrap();
        let reduced = b.reduce(RedOp::Sum, x, 0, false).unwrap();
        let graph = b.finish(reduced).unwrap();
        assert_eq!(graph.aval(graph.result()).shape, vec![4]);

        let lowered = lower_nonlast_reduces(&graph).unwrap();
        assert_eq!(lowered.eqns().len(), 2, "transpose + last-axis reduce");
        let OpKind::Transpose { perm } = &lowered.eqns()[0].op else {
            panic!(
                "first lowered equation must be the transpose, got {:?}",
                lowered.eqns()[0].op
            );
        };
        assert_eq!(perm, &vec![1, 0], "axis 0 moves last");
        let OpKind::Reduce { op, axis, keepdim } = &lowered.eqns()[1].op else {
            panic!(
                "second lowered equation must be the reduce, got {:?}",
                lowered.eqns()[1].op
            );
        };
        assert_eq!((*op, *axis, *keepdim), (RedOp::Sum, 1, false));
        assert!(
            matches!(&lowered.eqns()[1].inputs[..], [Operand::Value(id)] if *id == lowered.eqns()[0].out),
            "the reduce reads the transposed value"
        );
        assert_eq!(
            lowered.eqns()[1].out,
            graph.result(),
            "the reduce keeps the original output value id"
        );
        assert_eq!(lowered.aval(lowered.eqns()[1].out).shape, vec![4]);

        // Idempotent: nothing non-last is left to rewrite.
        let again = lower_nonlast_reduces(&lowered).unwrap();
        assert_eq!(again.eqns().len(), lowered.eqns().len());
        assert!(
            again
                .eqns()
                .iter()
                .zip(lowered.eqns())
                .all(|(a, b)| a.op == b.op && a.out == b.out),
            "a lowered graph must come back unchanged"
        );

        // A last-axis reduce is never rewritten.
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 4])).unwrap();
        let reduced = b.reduce(RedOp::Sum, x, 1, false).unwrap();
        let last_axis = b.finish(reduced).unwrap();
        assert_eq!(
            lower_nonlast_reduces(&last_axis).unwrap().eqns().len(),
            1,
            "a last-axis reduce has nothing to lower"
        );

        // keepdim = true: the size-1 axis stays where the reduced axis was, so a Reshape returns it.
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 4])).unwrap();
        let reduced = b.reduce(RedOp::Max, x, 0, true).unwrap();
        let graph = b.finish(reduced).unwrap();
        assert_eq!(graph.aval(graph.result()).shape, vec![1, 4]);

        let lowered = lower_nonlast_reduces(&graph).unwrap();
        assert_eq!(
            lowered.eqns().len(),
            3,
            "transpose + keepdim reduce + reshape"
        );
        let OpKind::Reduce { op, axis, keepdim } = &lowered.eqns()[1].op else {
            panic!(
                "second lowered equation must be the reduce, got {:?}",
                lowered.eqns()[1].op
            );
        };
        assert_eq!((*op, *axis, *keepdim), (RedOp::Max, 1, true));
        let OpKind::Reshape { shape } = lowered.eqns()[2].op.clone() else {
            panic!(
                "third lowered equation must be the reshape, got {:?}",
                lowered.eqns()[2].op
            );
        };
        assert_eq!(shape, vec![1, 4]);
        assert_eq!(lowered.eqns()[2].out, graph.result());
        assert_eq!(lowered.aval(lowered.eqns()[2].out).shape, vec![1, 4]);
    }

    /// Card 454 D1: the CPU oracle evaluates `lower_nonlast_reduces`'s output identically to the
    /// original non-last-axis reduce, bit for bit - moved here from `poot-eval`'s own test crate
    /// (Card 534a: the pass itself can no longer be named from outside
    /// `poot-graph-plan`, so its CPU-oracle bit-identity test moved with it; `poot-graph-plan` already
    /// dev-depends on `poot-eval` for exactly this).
    #[test]
    fn lower_nonlast_reduces_is_bit_identical_to_the_original_reduce() {
        use poot_graph_ir::BinOp;
        use std::collections::HashMap;

        /// One non-trivial `[3, 4]` payload: strictly decreasing, no ties, so Sum and Max both depend
        /// on every element (an all-equal payload could not distinguish an ordered reduction from a
        /// permuted one).
        fn payload() -> Vec<f32> {
            (0..12).map(|index| 12.0 - index as f32 * 0.75).collect()
        }

        fn inputs(x: poot_graph_ir::ValueId) -> HashMap<poot_graph_ir::ValueId, poot_eval::Value> {
            HashMap::from([(
                x,
                poot_eval::Value::from(poot_tensor::HostTensor::f32(vec![3, 4], payload())),
            )])
        }

        /// Every `Reduce` in `g` reduces the last axis of its input.
        fn all_reduces_are_last_axis(g: &poot_graph_ir::Graph) -> bool {
            g.eqns().iter().all(|eqn| {
                let OpKind::Reduce { axis, .. } = eqn.op else {
                    return true;
                };
                let Some(Operand::Value(id)) = eqn.inputs.first() else {
                    return true;
                };
                axis + 1 == g.aval(*id).shape.len()
            })
        }

        for (op, keepdim) in [
            (RedOp::Sum, false),
            (RedOp::Sum, true),
            (RedOp::Max, false),
            (RedOp::Max, true),
        ] {
            // Bare axis-0 reduce: [3, 4] -> [4] (or [1, 4] under keepdim).
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
            let reduced = b.reduce(op, x, 0, keepdim).unwrap();
            let bare = b.finish(reduced).unwrap();

            // With a consumer, so a wrong lowered value would surface downstream too.
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![3, 4])).unwrap();
            let reduced = b.reduce(op, x, 0, keepdim).unwrap();
            let consumed = b.binary(BinOp::Add, reduced, reduced).unwrap();
            let graph = b.finish(consumed).unwrap();

            for original in [bare, graph] {
                let bound = inputs(x.id());
                let base = poot_eval::eval(&original, &bound, poot_eval::EvalOptions::default())
                    .unwrap()
                    .output
                    .into_host()
                    .unwrap();
                let lowered = lower_nonlast_reduces(&original).unwrap();
                assert!(
                    all_reduces_are_last_axis(&lowered),
                    "lowered graph still holds a non-last-axis reduce (op={op:?}, keepdim={keepdim})"
                );
                let out = poot_eval::eval(&lowered, &bound, poot_eval::EvalOptions::default())
                    .unwrap()
                    .output
                    .into_host()
                    .unwrap();

                assert_eq!(out.shape(), base.shape(), "op={op:?}, keepdim={keepdim}");
                let bits = |t: &poot_tensor::HostTensor| {
                    t.as_f32()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&out),
                    bits(&base),
                    "lowering changed values (op={op:?}, keepdim={keepdim})"
                );
            }
        }
    }
}
