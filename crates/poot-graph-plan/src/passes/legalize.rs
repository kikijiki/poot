//! Card 523a: device buffer limits as a legalization stage (target-architecture.md section 2,
//! ADR-0104; closes R474-003).
//!
//! Loaders used to decide whether a graph's embedding table stayed on device from a hard-coded wgpu
//! constant, applied on every backend before a device was even chosen (R488-007: the same decision
//! broke ROCm, whose real buffer limit is nowhere near wgpu's). Tracers hand-rolled the same device
//! math themselves (R474-003: the qwen2 tracer's host embed, `gemma4`'s unconditional host embed,
//! `Qwen35BufferLimits`/`lm_head_chunks`). [`legalize`] is the one place that decision is made now,
//! from the target's own measured [`DeviceCaps::max_buffer_bytes`] (card 522): tracers always emit the
//! plain dense gather and the plain unsplit matmul, and this pass decides, per target, whether a
//! constant stays a device buffer, is hosted, is split, or cannot run at all.
//!
//! `compile` (`poot-graph-plan`) runs this as one of its own passes, between `dce` and `fuse`: it is
//! the one pass whose rewrite can change a graph's declared input set (an oversized embed table
//! becomes a host-computed [`Slot::TokenEmbed`] input instead of a device `Gather`), so a caller that
//! binds a graph's inputs before executing it must bind against what `compile` actually produced
//! (card 523a: a bare call outside `compile` is a second entry point).
//!
//! Three rewrites, tried in order, before anything still over the limit is refused:
//! - [`host_oversized_token_gathers`]: an axis-0 `Gather` of an oversized constant by a `Slot::Token`
//!   index (the tracers' embedding lookup) becomes a host-computed `Slot::TokenEmbed` row.
//! - [`split_oversized_matmul_weights`]: an oversized 2D weight splits into row chunks that each fit, each chunk
//!   computed separately and joined by a `Concat` (Card 453 D1's lm_head precedent, generalized to any matmul -
//!   restoring the "split" capability card 523a's first landing dropped). The weight is read either as the
//!   `[n, k]` operand of a `DenseContraction` (`compile` folds before legalizing, so this is the usual form;
//!   each chunk is a `DenseContraction` over its stored rows) or through a `[1, 0]` transpose into a `MatMul`
//!   (the form the fold declines; each chunk keeps its own transpose).
//! - [`split_oversized_expert_stacks`]: an indexed packed contraction whose stacked expert table is over the single
//!   buffer limit, or over what the kernels' 32-bit bit offsets address, splits into expert ranges that each fit
//!   (see its doc for how the ranges recombine).

use std::collections::{HashMap, HashSet};

use poot_target::DeviceCaps;

use poot_graph_ir::graph::{
    EditError, Eqn, Graph, LayerIndex, Operand, Slot, Storage, ValueId, ValueMeta,
};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{BinOp, ContractionRows, PackedSourceName, Scalar};
use poot_quant::PackedWeight;
use poot_tensor::DType;

use super::GraphBudget;
use crate::{CompileLimits, ExpansionError};
use poot_graph_ir::ValueTable;

/// The stage name an expansion refusal reports.
const STAGE: &str = "legalize";

/// Why [`legalize`] could not fit `g` onto a target's device limits.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum LegalizeError {
    /// A named constant is larger than the target's single-buffer limit, and legalize has no rewrite
    /// for how it is used: today's one rewrite is an axis-0 gather of a constant by a `Slot::Token`
    /// index (the tracers' embedding lookup); anything else over the limit refuses here, before any
    /// device allocation is attempted.
    #[error(
        "constant {name:?} is {bytes} bytes, over the target's {limit}-byte single-buffer limit \
         (DeviceCaps::max_buffer_bytes), and legalize has no rewrite for its use"
    )]
    OversizedConstant {
        name: String,
        bytes: u64,
        limit: u64,
    },
    /// Splitting an oversized weight would grow the graph past the caller's [`CompileLimits`].
    #[error(transparent)]
    Expansion(#[from] ExpansionError),
    /// A rewrite produced a graph the edit check refused: a defect of the pass, never of the input.
    #[error(transparent)]
    Edit(Box<poot_graph_ir::EditError>),
}

impl From<poot_graph_ir::EditError> for LegalizeError {
    fn from(error: poot_graph_ir::EditError) -> Self {
        Self::Edit(Box::new(error))
    }
}

/// Legalize `g` for a target whose measured single-buffer limit is `caps.max_buffer_bytes` (card 522).
///
/// Tracers and loaders never read a device limit: every model traces
/// and loads the same graph regardless of backend, and only this pass, given the caller's real
/// [`DeviceCaps`], decides whether a constant fits as one device buffer. Two rewrites, in order:
/// [`host_oversized_token_gathers`] (an axis-0 `Gather` of a named constant by a `Slot::Token` index,
/// over the limit, becomes a host-computed `Slot::TokenEmbed` row block named after the constant it
/// replaces - the binder reads that name to gather the row(s) host-side instead of uploading the whole
/// table, `poot-llm`'s `gather_token_embed_rows`), then [`split_oversized_matmul_weights`] (an
/// oversized 2D weight read by a `DenseContraction`, or through a transpose into a `MatMul`, splits into row
/// chunks that each fit, joined by a `Concat`). Anything
/// still over the limit after both rewrites is a typed refusal, before any device allocation (SC-002).
#[cfg_attr(
    not(any(test, feature = "test-support")),
    allow(
        unreachable_pub,
        reason = "ADR-0113: `pub` only so the test-support re-export in lib.rs can name it; no production path leaves the crate"
    )
)]
pub fn legalize(
    g: &Graph,
    caps: &DeviceCaps,
    limits: &CompileLimits,
) -> Result<Graph, LegalizeError> {
    let limit = caps.max_buffer_bytes;
    let g = host_oversized_token_gathers(g, limit)?;
    let g = split_oversized_matmul_weights(&g, limit, GraphBudget::of(limits))?;
    let g = split_oversized_expert_stacks(&g, limit, GraphBudget::of(limits))?;
    refuse_oversized_constants(&g, limit)?;
    Ok(g)
}

/// `g` with the split's values and equations and the new chunk constants declared beside the weights
/// they replace, dead-code-eliminated. The replaced weights stay declared until liveness shows nothing
/// reads them; [`without_consts`] then drops the dead ones.
fn staged_split(
    g: &Graph,
    values: ValueTable<ValueMeta>,
    eqns: Vec<Eqn>,
    new_consts: &[ValueId],
) -> Result<Graph, EditError> {
    let mut staged = g.edit();
    *staged.values_mut() = values;
    *staged.eqns_mut() = eqns;
    staged.inputs_mut().extend_from_slice(new_consts);
    staged.consts_mut().extend_from_slice(new_consts);
    super::dce(&staged.commit()?)
}

/// `split` with the constants in `dropped` no longer declared as inputs or constants.
fn without_consts(split: &Graph, dropped: &HashSet<ValueId>) -> Result<Graph, EditError> {
    let mut out = split.edit();
    out.inputs_mut().retain(|id| !dropped.contains(id));
    out.consts_mut().retain(|id| !dropped.contains(id));
    out.commit()
}

/// A named constant's byte footprint as one dense device buffer: element count times dtype width.
fn const_bytes(g: &Graph, id: ValueId) -> u64 {
    let aval = g.aval(id);
    let elems: u64 = aval.shape.iter().map(|&d| d as u64).product();
    elems.saturating_mul(aval.dtype.byte_size() as u64)
}

fn operand_value(o: Operand) -> Option<ValueId> {
    match o {
        Operand::Value(id) => Some(id),
        Operand::Lit(_) => None,
    }
}

/// Host every axis-0 `Gather(Const, Slot::Token)` whose constant exceeds `limit`: the gather's output
/// value becomes a `Slot::TokenEmbed` input named after the constant (so a binder can still find the
/// weight to gather host-side), the gather equation is dropped, and the constant (and the token slot,
/// if nothing else reads it) drop out of the graph's input tables. Every other value keeps its
/// [`ValueId`], so nothing downstream needs rewiring. A constant is hosted only when every reader of it
/// is a hosted gather: a tied head (the same table also read by a `MatMul` over its transpose) keeps
/// the constant, and the later stages split it or refuse it.
fn host_oversized_token_gathers(g: &Graph, limit: u64) -> Result<Graph, EditError> {
    let mut dropped_eqns = std::collections::HashSet::new();
    let mut hosted: Vec<(ValueId, ValueId, ValueId)> = Vec::new(); // (gather output, const, token index)
    let mut candidates: Vec<(usize, ValueId, ValueId, ValueId)> = Vec::new(); // (eqn, out, const, index)

    for (i, eqn) in g.eqns().iter().enumerate() {
        let OpKind::Gather { axis } = eqn.op else {
            continue;
        };
        if axis != 0 || eqn.inputs.len() != 2 {
            continue;
        }
        let (Some(data_id), Some(index_id)) =
            (operand_value(eqn.inputs[0]), operand_value(eqn.inputs[1]))
        else {
            continue;
        };
        if g.meta(data_id).storage != Storage::Const {
            continue;
        }
        if g.meta(index_id).storage != Storage::Slot(Slot::Token) {
            continue;
        }
        if const_bytes(g, data_id) <= limit {
            continue;
        }
        candidates.push((i, eqn.out, data_id, index_id));
    }

    // The constant leaves the graph, so every reader of it must be a gather being hosted: a live
    // `MatMul` over `Transpose(w)` (a tied head) would be left reading a value that no longer exists.
    // Such a constant stays, and `split_oversized_matmul_weights` or `refuse_oversized_constants`
    // answers for it.
    let candidate_eqns: std::collections::HashSet<usize> =
        candidates.iter().map(|&(i, ..)| i).collect();
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    for &(i, out, data_id, index_id) in &candidates {
        let sole_readers = !pinned.contains(&data_id)
            && g.eqns().iter().enumerate().all(|(j, e)| {
                candidate_eqns.contains(&j)
                    || !e.inputs.iter().any(|&o| operand_value(o) == Some(data_id))
            });
        if sole_readers {
            dropped_eqns.insert(i);
            hosted.push((out, data_id, index_id));
        }
    }

    if hosted.is_empty() {
        return Ok(g.clone());
    }

    let eqns: Vec<Eqn> = g
        .eqns()
        .iter()
        .enumerate()
        .filter(|(i, _)| !dropped_eqns.contains(i))
        .map(|(_, e)| e.clone())
        .collect();

    let referenced = |id: ValueId| -> bool {
        eqns.iter()
            .any(|e| e.inputs.iter().any(|&o| operand_value(o) == Some(id)))
            || g.result() == id
            || g.state().any(|(s_in, s_out)| s_in == id || s_out == id)
            || g.witnesses().any(|v| v.value == id)
    };

    let mut values = g.values().clone();
    let mut drop_values = std::collections::HashSet::new();
    let mut new_inputs = Vec::new();
    let mut new_slots = Vec::new();
    for (out, data_id, index_id) in hosted {
        let name = values[data_id]
            .name
            .clone()
            .expect("legalize: a Storage::Const value always carries a checkpoint name");
        values[out] = ValueMeta::new(
            values[out].aval.clone(),
            Storage::Slot(Slot::TokenEmbed),
            Some(name),
        );
        new_inputs.push(out);
        new_slots.push((out, Slot::TokenEmbed));
        drop_values.insert(data_id);
        if !referenced(index_id) {
            drop_values.insert(index_id);
        }
    }

    let inputs: Vec<ValueId> = g
        .inputs()
        .iter()
        .copied()
        .filter(|id| !drop_values.contains(id))
        .chain(new_inputs)
        .collect();
    let consts: Vec<ValueId> = g
        .consts()
        .iter()
        .copied()
        .filter(|id| !drop_values.contains(id))
        .collect();
    let slots: Vec<(ValueId, Slot)> = g
        .slots()
        .iter()
        .copied()
        .filter(|(id, _)| !drop_values.contains(id))
        .chain(new_slots)
        .collect();

    let mut hosted = g.edit();
    *hosted.values_mut() = values;
    *hosted.inputs_mut() = inputs;
    *hosted.consts_mut() = consts;
    *hosted.slots_mut() = slots;
    *hosted.eqns_mut() = eqns;
    hosted.commit()
}

/// Refuse (before any device allocation) every remaining LIVE named constant over `limit`: whatever
/// the rewrites above could not fit has no lowering this pass knows, and the alternative is an
/// executor discovering the same fact against a real device buffer.
///
/// Dead is checked, not merely declared: `compile` runs this pass right after `dce`, and `dce` only
/// shrinks `eqns` (`dce`'s own doc: "the value table is preserved, dead entries are
/// harmless"), so `g.consts` can still list an oversized constant nothing reads any more. Refusing on
/// declaration alone would fail a graph a real device would never even see.
///
/// `legalize` calls this on its own output, and `compile` calls it again on the finished pipeline's
/// graph: `revert_unplannable_rewrites` can undo a split the target cannot plan, so the graph the
/// planner receives may hold a constant `legalize` had already brought under `limit`.
pub(crate) fn refuse_oversized_constants(g: &Graph, limit: u64) -> Result<(), LegalizeError> {
    let live = live_value_ids(g);
    for &id in g.consts() {
        if !live.contains(&id) {
            continue;
        }
        let bytes = const_bytes(g, id);
        if bytes > limit {
            let name = g.meta(id).name.clone().unwrap_or_default();
            return Err(LegalizeError::OversizedConstant { name, bytes, limit });
        }
    }
    Ok(())
}

/// Every [`ValueId`] a real device run cannot avoid computing or binding: the liveness roots
/// (output, validation outputs, state outputs), the state inputs they carry across steps, and
/// everything an eqn reachable from those roots reads - the same backward walk [`super::dce`] runs,
/// so this always agrees with what `dce` just kept.
fn live_value_ids(g: &Graph) -> HashSet<ValueId> {
    let mut live: HashSet<ValueId> = g.pinned_values().collect();
    for eqn in g.eqns().iter().rev() {
        if live.contains(&eqn.out) {
            for o in &eqn.inputs {
                if let Some(id) = operand_value(*o) {
                    live.insert(id);
                }
            }
        }
    }
    live
}

/// `limit` rounded down to a 4096-byte boundary (a driver that aligns `create_buffer` up to 4096 must not push the
/// allocation over `limit`), except for limits below 4096 where no positive count can satisfy that alignment and
/// the plain `limit` is used (tiny fixtures / typed test overrides).
fn usable_bytes(limit: u64) -> u64 {
    if limit >= 4096 {
        (limit / 4096) * 4096
    } else {
        limit
    }
}

/// Column ranges of a `[k, n]` weight, split so each chunk stays within `limit` bytes, or `None` when
/// even one row (`k` elements) is already over the limit and no split can help. Every chunk holds
/// `rows_per_chunk` rows of the `[n, k]` stored weight except a smaller final chunk.
///
/// `rows_per_chunk` is Card 453 D1's `Qwen35BufferLimits::lm_head_chunks` closed form (a chunk of that
/// many output columns is that many rows of the `[n, k]` stored weight), generalized from lm_head to
/// any 2D matmul weight: the usable limit is `limit` rounded down to a 4096-byte boundary (a driver that aligns
/// `create_buffer` up to 4096 must not push the allocation over `limit`), except for limits below 4096
/// where no positive row count can satisfy that alignment and the plain `limit / row_bytes` is used
/// (tiny fixtures / typed test overrides).
fn row_chunks(k: usize, n: usize, elem_bytes: usize, limit: u64) -> Option<Vec<(usize, usize)>> {
    let row_bytes = (k.max(1) as u64).saturating_mul(elem_bytes.max(1) as u64);
    if row_bytes > limit {
        return None; // no split helps: a single row already exceeds the target's real limit.
    }
    let usable = usable_bytes(limit);
    let rows_per_chunk = ((usable / row_bytes.max(1)) as usize).max(1);
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < n {
        let end = (start + rows_per_chunk).min(n);
        ranges.push((start, end));
        start = end;
    }
    Some(ranges)
}

/// A splittable `MatMul`: its activation, the `[n, k]` weight constant it reads through a transpose, and
/// the output-column ranges (rows of the constant) of the chunks.
struct SplitCandidate {
    x: ValueId,
    weight: ValueId,
    ranges: Vec<(usize, usize)>,
    /// Whether the weight is read as a [`OpKind::DenseContraction`] operand (stored `[n, k]`, no transpose)
    /// rather than through a `[1, 0]` transpose into a `MatMul`.
    read: WeightRead,
}

/// How a splittable equation reads its `[n, k]` stored weight.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WeightRead {
    /// `MatMul(x, Transpose([1, 0], w))`.
    Transposed,
    /// `DenseContraction { weight: dtype }(x, w)`.
    Dense,
}

/// Every `[1, 0]` `Transpose` of a 2D named [`Storage::Const`], keyed by the transpose's output value.
fn transposed_consts(g: &Graph) -> HashMap<ValueId, ValueId> {
    g.eqns()
        .iter()
        .filter_map(|eqn| {
            let OpKind::Transpose { perm } = &eqn.op else {
                return None;
            };
            let source = operand_value(*eqn.inputs.first()?)?;
            (perm.as_slice() == [1, 0]
                && g.meta(source).storage == Storage::Const
                && g.aval(source).shape.len() == 2)
                .then_some((eqn.out, source))
        })
        .collect()
}

/// Whether `eqn` is an oversized-weight `MatMul` [`split_oversized_matmul_weights`] can rewrite: a
/// plain `x @ w^T` (not a `ContractionEpilogue`: the epilogue fuse rule runs after this pass) against a 2D
/// named [`Storage::Const`] weight over `limit`, read through a `[1, 0]` transpose (`transposed`, from
/// [`transposed_consts`]: the layout every driver-traced dense weight has), with at least two resulting
/// chunks (a "split" into one chunk is not a rewrite). A matmul that reads a const directly has no
/// split: its chunks would be column slices of the stored weight, which no binder places.
fn splittable_matmul(
    g: &Graph,
    eqn: &Eqn,
    transposed: &HashMap<ValueId, ValueId>,
    limit: u64,
) -> Option<SplitCandidate> {
    if eqn.inputs.len() != 2 {
        return None;
    }
    let x = operand_value(eqn.inputs[0])?;
    let read = operand_value(eqn.inputs[1])?;
    let (weight, how) = match eqn.op {
        OpKind::MatMul => (*transposed.get(&read)?, WeightRead::Transposed),
        OpKind::DenseContraction { .. } if g.meta(read).storage == Storage::Const => {
            (read, WeightRead::Dense)
        }
        _ => return None,
    };
    // Only a dense weight splits into row blocks: a packed source (named as one) has no byte rows a
    // binder cuts, and stays whole for `refuse_oversized_constants` to refuse by name.
    if g.meta(weight)
        .name
        .as_deref()
        .and_then(PackedSourceName::parse)
        .is_some()
    {
        return None;
    }
    let w_aval = g.aval(weight);
    if w_aval.shape.len() != 2 || const_bytes(g, weight) <= limit {
        return None;
    }
    let (n, k) = (w_aval.shape[0], w_aval.shape[1]);
    let ranges = row_chunks(k, n, w_aval.dtype.byte_size(), limit)?;
    if ranges.len() < 2 {
        return None;
    }
    Some(SplitCandidate {
        x,
        weight,
        ranges,
        read: how,
    })
}

/// Split every oversized-weight `MatMul` [`splittable_matmul`] finds into row-chunk matmuls joined by a
/// `Concat` on the output axis, so the outcome still computes at the original output `ValueId`
/// (nothing downstream needs rewiring). Card 453 D1's `Qwen35BufferLimits::lm_head_chunks`, as a generic
/// legalization outcome: any oversized 2D matmul weight splits, not only the lm_head. Chunk `N` is the
/// const `{name}.chunkN` (`N` dense from 0): the `[width, k]` row block of the `[n, k]` stored weight
/// that the chunks before it leave off, read through its own `Transpose`. The executor binder places a
/// chunk const as that row range of `{name}`'s own bytes (`poot-executor`'s `binder::chunks`).
fn split_oversized_matmul_weights(
    g: &Graph,
    limit: u64,
    budget: GraphBudget,
) -> Result<Graph, LegalizeError> {
    let transposed = transposed_consts(g);
    let mut values = g.values().clone();
    let mut new_consts = Vec::new();
    let mut split_weights = HashSet::new();
    let mut eqns = Vec::with_capacity(g.eqns().len());

    for eqn in g.eqns() {
        let Some(SplitCandidate {
            x: x_id,
            weight: w_id,
            ranges,
            read: how,
        }) = splittable_matmul(g, eqn, &transposed, limit)
        else {
            budget.push_eqn(STAGE, &mut eqns, eqn.clone())?;
            continue;
        };
        split_weights.insert(w_id);
        let name = values[w_id]
            .name
            .clone()
            .expect("legalize: a Storage::Const value always carries a checkpoint name");
        let dtype = values[w_id].aval.dtype;
        let k = values[w_id].aval.shape[1];
        let mut prefix = values[eqn.out].aval.shape.clone();
        let out_dtype = values[eqn.out].aval.dtype;
        prefix.pop().expect("a matmul output has at least one axis");
        let concat_axis = prefix.len();

        let mut chunk_outs = Vec::with_capacity(ranges.len());
        for (idx, (start, end)) in ranges.iter().enumerate() {
            let width = end - start;
            let chunk_w = budget.push_value(
                STAGE,
                &mut values,
                ValueMeta::new(
                    TensorType::new(vec![width, k], dtype),
                    Storage::Const,
                    Some(format!("{name}.chunk{idx}")),
                ),
            )?;
            new_consts.push(chunk_w);

            let chunk_read = match how {
                WeightRead::Dense => chunk_w,
                WeightRead::Transposed => {
                    let read = budget.push_value(
                        STAGE,
                        &mut values,
                        ValueMeta::new(
                            TensorType::new(vec![k, width], dtype),
                            Storage::Device,
                            None,
                        ),
                    )?;
                    budget.push_eqn(
                        STAGE,
                        &mut eqns,
                        Eqn {
                            op: OpKind::Transpose { perm: vec![1, 0] },
                            inputs: vec![Operand::Value(chunk_w)],
                            out: read,
                            layer: eqn.layer,
                        },
                    )?;
                    read
                }
            };

            let mut shape = prefix.clone();
            shape.push(width);
            let chunk_out = budget.push_value(
                STAGE,
                &mut values,
                ValueMeta::new(TensorType::new(shape, out_dtype), Storage::Device, None),
            )?;
            budget.push_eqn(
                STAGE,
                &mut eqns,
                Eqn {
                    op: eqn.op.clone(),
                    inputs: vec![Operand::Value(x_id), Operand::Value(chunk_read)],
                    out: chunk_out,
                    layer: eqn.layer,
                },
            )?;
            chunk_outs.push(chunk_out);
        }
        budget.push_eqn(
            STAGE,
            &mut eqns,
            Eqn {
                op: OpKind::Concat { axis: concat_axis },
                inputs: chunk_outs.into_iter().map(Operand::Value).collect(),
                out: eqn.out,
                layer: eqn.layer,
            },
        )?;
    }

    if split_weights.is_empty() {
        return Ok(g.clone());
    }
    // A transposed weight's old `Transpose` read is dead now; `dce` removes it. A split weight some
    // other live eqn still reads stays declared (and is refused if still oversized).
    let split = staged_split(g, values, eqns, &new_consts)?;
    let live = live_value_ids(&split);
    let drop_consts: HashSet<ValueId> = split_weights
        .into_iter()
        .filter(|id| !live.contains(id))
        .collect();
    Ok(without_consts(&split, &drop_consts)?)
}

/// The expert ranges an indexed packed contraction of `experts` experts splits into so that each range's sources
/// are single buffers within `limit` bytes, addressable by the kernels' 32-bit bit offsets
/// ([`PackedWeight::max_stack_experts`]); `None` when the stack fits whole, or when one expert's slab alone is over
/// the buffer limit (no split helps: `refuse_oversized_constants` names the carrier, and an expert past the address
/// range is the planner's typed `DispatchSizeOverflow`).
fn expert_ranges(
    descriptor: PackedWeight,
    experts: usize,
    limit: u64,
) -> Option<Vec<std::ops::Range<usize>>> {
    let usable = usable_bytes(limit);
    let mut per_chunk = descriptor.max_stack_experts();
    for role in descriptor.sources() {
        let slab = descriptor.slab_bytes(role) as u64;
        if slab > limit {
            return None;
        }
        per_chunk = per_chunk.min(((usable / slab.max(1)) as usize).max(1));
    }
    if per_chunk == 0 || per_chunk >= experts {
        return None;
    }
    Some(
        (0..experts)
            .step_by(per_chunk)
            .map(|first| first..(first + per_chunk).min(experts))
            .collect(),
    )
}

/// An indexed packed contraction `legalize` can split: its activation, row ids and packed source carriers (named
/// constants, one per role), and the expert ranges of the pieces.
struct StackSplit {
    descriptor: PackedWeight,
    x: ValueId,
    ids: ValueId,
    carriers: Vec<ValueId>,
    ranges: Vec<std::ops::Range<usize>>,
}

/// The equations and values a rewrite appends, each reserved against the graph budget.
struct Appender<'a> {
    budget: GraphBudget,
    values: &'a mut ValueTable<ValueMeta>,
    eqns: &'a mut Vec<Eqn>,
    layer: Option<LayerIndex>,
}

impl Appender<'_> {
    /// Append one equation `op` over `inputs` writing a fresh device value of type `aval`.
    fn push(
        &mut self,
        op: OpKind,
        inputs: &[Operand],
        aval: TensorType,
    ) -> Result<ValueId, ExpansionError> {
        let out = self.budget.push_value(
            STAGE,
            self.values,
            ValueMeta::new(aval, Storage::Device, None),
        )?;
        self.push_to(op, inputs, out)?;
        Ok(out)
    }

    /// Append one equation `op` over `inputs` writing the existing value `out`.
    fn push_to(
        &mut self,
        op: OpKind,
        inputs: &[Operand],
        out: ValueId,
    ) -> Result<(), ExpansionError> {
        self.budget.push_eqn(
            STAGE,
            self.eqns,
            Eqn {
                op,
                inputs: inputs.to_vec(),
                out,
                layer: self.layer,
            },
        )
    }

    /// `1` for the rows whose id lies in `range` and `0` for any other, a negative id included: `Ge(id, from) -
    /// Ge(id, to)`, an exact I32 indicator.
    fn indicator(
        &mut self,
        ids: ValueId,
        ids_aval: &TensorType,
        range: std::ops::Range<usize>,
    ) -> Result<ValueId, ExpansionError> {
        let ge = |at: usize| [Operand::Value(ids), Operand::Lit(Scalar::I32(at as i32))];
        let from = self.push(
            OpKind::Binary(BinOp::Ge),
            &ge(range.start),
            ids_aval.clone(),
        )?;
        let to = self.push(OpKind::Binary(BinOp::Ge), &ge(range.end), ids_aval.clone())?;
        self.push(
            OpKind::Binary(BinOp::Sub),
            &[Operand::Value(from), Operand::Value(to)],
            ids_aval.clone(),
        )
    }
}

/// Split every indexed packed contraction whose stacked expert table does not fit one buffer
/// ([`expert_ranges`]) into one contraction per expert range, summed at the original output id.
///
/// Piece `c` holds experts `lo..hi`: its carriers are the constants `{carrier}.chunkC` (the rows `lo..hi` of the
/// stack's leading expert axis, which the executor binder places as that run of the parent's experts, like a dense
/// weight's row chunk). Every piece computes every row, so a row the piece does not own is neutralized instead of
/// selected away: the piece reads the activation row times `owned` (an I32 `1` when `lo <= id < hi`, else `0`,
/// widened to F32), which makes its output row exactly `0` for a finite activation row, and the id
/// `max(id - lo, 0) % (hi - lo)`: the id shifted into the piece's table for a row it owns, and some valid
/// row of the table for a row it does not (its activation row is zero). The id is proven in `[0, hi - lo)` by
/// the range rules alone (`Max` and `RemU`), so the split graph passes the same expert-id proof as the unsplit
/// one; an id outside `0..E` never reaches a split (`compile` refused it, and the weight binder checks a
/// declared range). The sum of the pieces is the owning piece's row plus zeros. The split costs one full
/// contraction per piece, and a non-finite activation row turns into NaN rather than whatever the unsplit
/// contraction makes of it.
fn split_oversized_expert_stacks(
    g: &Graph,
    limit: u64,
    budget: GraphBudget,
) -> Result<Graph, LegalizeError> {
    let mut values = g.values().clone();
    let mut new_consts = Vec::new();
    let mut split_carriers = HashSet::new();
    let mut eqns = Vec::with_capacity(g.eqns().len());

    for eqn in g.eqns() {
        let Some(StackSplit {
            descriptor,
            x,
            ids,
            carriers,
            ranges,
        }) = splittable_stack(g, eqn, limit)
        else {
            budget.push_eqn(STAGE, &mut eqns, eqn.clone())?;
            continue;
        };
        split_carriers.extend(carriers.iter().copied());
        let out_aval = values[eqn.out].aval.clone();
        let x_aval = values[x].aval.clone();
        let ids_aval = values[ids].aval.clone();
        let column = TensorType::f32(vec![ids_aval.shape[0], 1]);
        let owned_aval = TensorType::f32(ids_aval.shape.clone());
        let mut emit = Appender {
            budget,
            values: &mut values,
            eqns: &mut eqns,
            layer: eqn.layer,
        };

        let mut sum: Option<ValueId> = None;
        for (index, range) in ranges.iter().enumerate() {
            let owned = emit.indicator(ids, &ids_aval, range.clone())?;
            let local_ids = if range.start == 0 {
                ids
            } else {
                emit.push(
                    OpKind::Binary(BinOp::Sub),
                    &[
                        Operand::Value(ids),
                        Operand::Lit(Scalar::I32(range.start as i32)),
                    ],
                    ids_aval.clone(),
                )?
            };
            let non_negative = emit.push(
                OpKind::Binary(BinOp::Max),
                &[Operand::Value(local_ids), Operand::Lit(Scalar::I32(0))],
                ids_aval.clone(),
            )?;
            let piece_ids = emit.push(
                OpKind::Binary(BinOp::RemU),
                &[
                    Operand::Value(non_negative),
                    Operand::Lit(Scalar::I32(range.len() as i32)),
                ],
                ids_aval.clone(),
            )?;
            let owned_f32 = emit.push(
                OpKind::Cast { to: DType::F32 },
                &[Operand::Value(owned)],
                owned_aval.clone(),
            )?;
            let owned_column = emit.push(
                OpKind::Reshape {
                    shape: column.shape.clone(),
                },
                &[Operand::Value(owned_f32)],
                column.clone(),
            )?;
            let piece_x = emit.push(
                OpKind::Binary(BinOp::Mul),
                &[Operand::Value(x), Operand::Value(owned_column)],
                x_aval.clone(),
            )?;
            let mut piece_inputs = vec![Operand::Value(piece_x), Operand::Value(piece_ids)];
            for &carrier in &carriers {
                let name = emit.values[carrier]
                    .name
                    .clone()
                    .expect("legalize: a Storage::Const value always carries a checkpoint name");
                let mut shape = emit.values[carrier].aval.shape.clone();
                shape[0] = range.len();
                let dtype = emit.values[carrier].aval.dtype;
                let piece = emit.budget.push_value(
                    STAGE,
                    emit.values,
                    ValueMeta::new(
                        TensorType::new(shape, dtype),
                        Storage::Const,
                        Some(format!("{name}.chunk{index}")),
                    ),
                )?;
                new_consts.push(piece);
                piece_inputs.push(Operand::Value(piece));
            }
            let contraction = OpKind::PackedContraction {
                descriptor,
                rows: ContractionRows::Indexed {
                    experts: range.len(),
                },
            };
            let last = index + 1 == ranges.len();
            let piece_out = emit.push(contraction, &piece_inputs, out_aval.clone())?;
            sum = Some(match sum {
                None => piece_out,
                Some(sum) => {
                    let add = [Operand::Value(sum), Operand::Value(piece_out)];
                    if last {
                        emit.push_to(OpKind::Binary(BinOp::Add), &add, eqn.out)?;
                        eqn.out
                    } else {
                        emit.push(OpKind::Binary(BinOp::Add), &add, out_aval.clone())?
                    }
                }
            });
        }
    }

    if split_carriers.is_empty() {
        return Ok(g.clone());
    }
    let split = staged_split(g, values, eqns, &new_consts)?;
    let live = live_value_ids(&split);
    let drop_consts: HashSet<ValueId> = split_carriers
        .into_iter()
        .filter(|id| !live.contains(id))
        .collect();
    Ok(without_consts(&split, &drop_consts)?)
}

/// The split of `eqn` when it is an indexed packed contraction over named constant carriers and I32 ids whose
/// stacked table [`expert_ranges`] splits.
fn splittable_stack(g: &Graph, eqn: &Eqn, limit: u64) -> Option<StackSplit> {
    let OpKind::PackedContraction {
        descriptor,
        rows: ContractionRows::Indexed { experts },
    } = &eqn.op
    else {
        return None;
    };
    let ranges = expert_ranges(*descriptor, *experts, limit)?;
    let operands: Vec<ValueId> = eqn
        .inputs
        .iter()
        .map(|operand| operand_value(*operand))
        .collect::<Option<_>>()?;
    let (&x, &ids) = (operands.first()?, operands.get(1)?);
    let carriers = operands[2..].to_vec();
    (g.aval(ids).dtype == DType::I32
        && carriers.iter().all(|&carrier| {
            g.meta(carrier).storage == Storage::Const && g.meta(carrier).name.is_some()
        }))
    .then_some(StackSplit {
        descriptor: *descriptor,
        x,
        ids,
        carriers,
        ranges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use poot_graph_ir::Builder;
    use poot_graph_ir::types::TensorType;
    use poot_tensor::DType;

    fn caps_with_limit(max_buffer_bytes: u64) -> DeviceCaps {
        DeviceCaps {
            max_buffer_bytes,
            ..DeviceCaps::wgpu_rdna3_igpu()
        }
    }

    /// `[vocab, hidden]` f32 embed table gathered by a scalar `Slot::Token` (the decode shape).
    fn embed_gather_graph(vocab: usize, hidden: usize) -> Graph {
        let b = Builder::for_tests();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
        let embed = b
            .constant(
                "model.embed_tokens.weight",
                TensorType::f32(vec![vocab, hidden]),
            )
            .unwrap();
        let out = b.gather_scalar(embed, 0, token).unwrap();
        b.finish(out).unwrap()
    }

    /// SC-002 (host): an embed table over the limit is rewritten to a host-computed `Slot::TokenEmbed`
    /// input named after the constant it replaces, and the gather and the oversized constant are gone.
    /// Mutation (recorded, not committed): neutering the eqn scan in `host_oversized_token_gathers` so
    /// it never matches a `Gather` left the 2.18 GB embed `Const` in the graph, which
    /// `refuse_oversized_constants` then (correctly) refused; `expect` panicked red with `an embed
    /// table always hosts: OversizedConstant { name: "model.embed_tokens.weight", bytes: 2179989504,
    /// limit: 2147483647 }`. Restoring the scan made the rewrite fire again and the test green.
    #[test]
    fn legalize_hosts_an_embed_gather_over_the_limit() {
        let g = embed_gather_graph(152_064, 3584); // ~2.18 GB f32, qwen2.5-7b's shape
        let caps = caps_with_limit(2_147_483_647); // wgpu's maxBufferSize
        let legalized =
            legalize(&g, &caps, &CompileLimits::STANDARD).expect("an embed table always hosts");

        assert!(
            !legalized
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "the gather is replaced, not merely left decomposed"
        );
        assert!(
            legalized.consts().is_empty(),
            "the oversized const drops out"
        );
        assert_eq!(
            legalized.inputs().len(),
            1,
            "one host-computed input replaces token+const"
        );
        let id = legalized.inputs()[0];
        let meta = legalized.meta(id);
        assert_eq!(meta.storage, Storage::Slot(Slot::TokenEmbed));
        assert_eq!(meta.name.as_deref(), Some("model.embed_tokens.weight"));
        assert_eq!(meta.aval.shape, vec![hidden_of(&g)]);
    }

    /// A tied head: the oversized embed table is also read by a `MatMul` over its transpose. Hosting the
    /// gather would drop a constant the matmul still reads (`InvalidGraph(EquationOperandNotDefined)`
    /// at compile); the rewrite must leave it, so legalize answers with the typed `OversizedConstant`.
    #[test]
    fn legalize_does_not_host_a_gather_whose_table_another_eqn_reads() {
        let b = Builder::for_tests();
        let token = b.slot(Slot::Token, TensorType::scalar(DType::I32)).unwrap();
        let embed = b
            .constant("tied.weight", TensorType::f32(vec![152_064, 3584]))
            .unwrap();
        let row = b.gather_scalar(embed, 0, token).unwrap();
        let x = b.reshape(row, vec![1, 3584]).unwrap();
        let out = b
            .matmul(x, b.transpose(embed, vec![1, 0]).unwrap())
            .unwrap();
        let g = b.finish(out).unwrap();

        let err = legalize(
            &g,
            &caps_with_limit(2_147_483_647),
            &CompileLimits::STANDARD,
        )
        .expect_err("the gather still reads the table, so it cannot be hosted away");
        assert!(
            matches!(err, LegalizeError::OversizedConstant { ref name, .. } if name == "tied.weight"),
            "{err:?}"
        );
    }

    fn hidden_of(g: &Graph) -> usize {
        // The gather's output shape before legalizing (scalar index drops the vocab axis): [hidden].
        g.aval(g.result()).shape[0]
    }

    /// A small model's embed table fits one device buffer: legalize leaves the dense gather alone.
    #[test]
    fn legalize_leaves_a_small_embed_dense() {
        let g = embed_gather_graph(151_936, 896); // qwen2.5-0.5b, ~544 MB
        let caps = caps_with_limit(2_147_483_647);
        let legalized = legalize(&g, &caps, &CompileLimits::STANDARD)
            .expect("under the limit, no rewrite needed");
        assert_eq!(legalized.eqns().len(), g.eqns().len());
        assert!(
            legalized
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "the dense gather stays: this device can hold the table"
        );
        assert_eq!(legalized.consts(), g.consts());
    }

    /// SC-001: the SAME model (qwen2.5-7b's embed shape) on a device whose limit is not wgpu's -
    /// ROCm's real `max_buffer_bytes` is its measured VRAM total, far larger than one embed table -
    /// stays dense. This is the fix: the old loader-side `embed_exceeds_wgpu_buffer` applied wgpu's
    /// constant on every backend, so ROCm quantized decode of a 7B model got a host-embed graph it
    /// never asked for and could not bind. Per-target legalize does not make that mistake.
    #[test]
    fn legalize_keeps_a_7b_embed_dense_on_a_device_with_real_headroom() {
        let g = embed_gather_graph(152_064, 3584);
        let rocm_like_caps = caps_with_limit(64 << 30); // 64 GiB unified memory, not wgpu's ~2 GiB
        let legalized = legalize(&g, &rocm_like_caps, &CompileLimits::STANDARD)
            .expect("ROCm's real limit holds this table");
        assert!(
            legalized
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Gather { .. })),
            "ROCm's real buffer limit holds the whole table; no host rewrite is needed"
        );
    }

    /// SC-002 (split): a plain matmul weight over the limit, with no gather-by-token pattern (so the
    /// host rewrite does not apply), splits into row chunks joined by a `Concat` at the original
    /// output id - Card 453 D1's `Qwen35BufferLimits::lm_head_chunks`, restored as a legalization
    /// outcome generic to any oversized 2D matmul weight, not only a tied lm_head.
    /// Mutation (recorded, not committed): neutering `row_chunks` to always return `None` left the
    /// 3.05 GB `big.weight` const in the graph, which `refuse_oversized_constants` then (correctly)
    /// refused; `expect` panicked red with `an oversized matmul weight always splits: OversizedConstant
    /// { name: "big.weight", bytes: 3276800000, limit: 2147483647 }`. Restoring `row_chunks` made
    /// the split fire again and the test green.
    #[test]
    fn legalize_splits_an_oversized_matmul_weight() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4096])).unwrap();
        let w = b
            .constant("big.weight", TensorType::f32(vec![200_000, 4096]))
            .unwrap(); // ~3.05 GB
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(out).unwrap();

        let caps = caps_with_limit(2_147_483_647);
        let legalized = legalize(&g, &caps, &CompileLimits::STANDARD)
            .expect("an oversized matmul weight always splits");

        assert!(
            legalized
                .consts()
                .iter()
                .all(|&id| legalized.meta(id).name.as_deref() != Some("big.weight")),
            "the unsplit weight is gone"
        );
        let chunk_names: Vec<&str> = legalized
            .consts()
            .iter()
            .filter_map(|&id| legalized.meta(id).name.as_deref())
            .filter(|n| n.starts_with("big.weight.chunk"))
            .collect();
        assert_eq!(
            chunk_names.len(),
            2,
            "3.05 GB over a 2 GiB-ish limit needs 2 chunks"
        );
        assert!(
            legalized
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Concat { .. })),
            "the chunks join by concat"
        );
        assert_eq!(
            legalized.result(),
            g.result(),
            "the concat computes at the original matmul's output id"
        );
        for &id in legalized.consts() {
            assert!(
                const_bytes(&legalized, id) <= 2_147_483_647,
                "every remaining constant, including each chunk, fits the limit"
            );
        }
    }

    /// SC-002 (split, dense): an oversized `[N, K]` weight read by a `DenseContraction` (the fold has run, as in
    /// `compile`) splits into row-chunk `DenseContraction`s over the stored rows - no transpose, no matmul -
    /// joined by a `Concat` at the original output id. Mutation: drop the `DenseContraction` arm in
    /// `splittable_matmul`; the weight stays whole and `legalize` refuses it (`OversizedConstant`).
    #[test]
    fn legalize_splits_an_oversized_dense_contraction_weight() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4096])).unwrap();
        let w = b
            .constant(
                "big.weight",
                TensorType::new(vec![200_000, 4096], DType::BF16),
            )
            .unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = crate::dtype_widen::fold_dense_contractions(&b.finish(out).unwrap()).unwrap();
        assert!(
            g.eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::DenseContraction { .. })),
            "the fixture folds"
        );

        let limit = 1_000_000_000;
        let legalized = legalize(&g, &caps_with_limit(limit), &CompileLimits::STANDARD)
            .expect("an oversized dense weight always splits");
        let chunks = legalized
            .eqns()
            .iter()
            .filter(|e| {
                matches!(
                    e.op,
                    OpKind::DenseContraction {
                        weight: DType::BF16
                    }
                )
            })
            .count();
        assert_eq!(chunks, 2, "1.6 GB over a 1 GB limit needs 2 chunks");
        assert!(
            !legalized
                .eqns()
                .iter()
                .any(|e| matches!(e.op, OpKind::Transpose { .. } | OpKind::MatMul)),
            "the chunks read the stored rows directly"
        );
        assert_eq!(legalized.result(), g.result());
        for &id in legalized.consts() {
            assert!(const_bytes(&legalized, id) <= limit, "every chunk fits");
        }
    }

    /// SC-002 (split, transposed): the tracers' `linear` reads an `[out, in]` const through a `[1, 0]`
    /// transpose. The const splits into row chunks, each read through its own transpose, joined by a
    /// `Concat` at the original output id; the old transpose (and the unsplit const) are gone. A
    /// transpose some other live eqn also reads keeps the const declared, and it is then refused.
    #[test]
    fn legalize_splits_a_transposed_matmul_weight_and_keeps_a_shared_one() {
        let build = |share: bool| {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![1, 4])).unwrap();
            let w = b
                .constant("head.weight", TensorType::f32(vec![10, 4]))
                .unwrap(); // 160 bytes
            let wt = b.transpose(w, vec![1, 0]).unwrap();
            let out = b.matmul(x, wt).unwrap();
            // a second reader of the same transpose that is not a matmul keeps it live.
            let out = if share {
                let s = b
                    .reduce(poot_graph_ir::op::RedOp::Sum, wt, 0, true)
                    .unwrap();
                b.binary(poot_graph_ir::op::BinOp::Add, out, s).unwrap()
            } else {
                out
            };
            b.finish(out).unwrap()
        };
        // 64-byte limit: a row is 16 bytes, 4 rows per chunk -> (0,4) (4,8) (8,10).
        let caps = caps_with_limit(64);

        let g = build(false);
        let legalized = legalize(&g, &caps, &CompileLimits::STANDARD)
            .expect("a transposed oversized weight splits");
        let chunks: Vec<(&str, Vec<usize>)> = legalized
            .consts()
            .iter()
            .filter_map(|&id| {
                let m = legalized.meta(id);
                Some((m.name.as_deref()?, m.aval.shape.clone()))
            })
            .filter(|(n, _)| n.starts_with("head.weight"))
            .collect();
        assert_eq!(
            chunks,
            vec![
                ("head.weight.chunk0", vec![4, 4]),
                ("head.weight.chunk1", vec![4, 4]),
                ("head.weight.chunk2", vec![2, 4]),
            ]
        );
        assert_eq!(legalized.result(), g.result());
        let count = |f: fn(&OpKind) -> bool| legalized.eqns().iter().filter(|e| f(&e.op)).count();
        assert_eq!(count(|op| matches!(op, OpKind::Transpose { .. })), 3);
        assert_eq!(count(|op| matches!(op, OpKind::Concat { .. })), 1);

        let error = legalize(&build(true), &caps, &CompileLimits::STANDARD)
            .expect_err("the shared transpose still reads the whole const");
        assert!(matches!(
            error,
            LegalizeError::OversizedConstant { ref name, .. } if name == "head.weight"
        ));
    }

    /// A weight a matmul reads directly (no transpose) has no split no binder could place - its chunks
    /// would be column slices of the stored weight - so it is refused by name.
    #[test]
    fn legalize_refuses_an_oversized_weight_read_without_a_transpose() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4])).unwrap();
        let w = b
            .constant("direct.weight", TensorType::f32(vec![4, 10]))
            .unwrap();
        let out = b.matmul(x, w).unwrap();
        let g = b.finish(out).unwrap();
        let error = legalize(&g, &caps_with_limit(64), &CompileLimits::STANDARD)
            .expect_err("a direct oversized weight has no split");
        assert!(matches!(
            error,
            LegalizeError::OversizedConstant { ref name, .. } if name == "direct.weight"
        ));
    }

    /// A packed source behind a transpose has no row blocks a binder can cut, so it is not split: it
    /// stays whole and the oversized-constant refusal names it at compile, not at bind.
    ///
    /// Mutation: drop the `PackedSourceName` guard in `splittable_matmul`; the source splits into
    /// `.chunkN` consts and `legalize` returns `Ok`, so the `expect_err` row goes red.
    #[test]
    fn legalize_refuses_an_oversized_packed_source_instead_of_splitting_it() {
        let name = poot_graph_ir::PackedSourceName::weight("w.head").to_string();
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4])).unwrap();
        let w = b.constant(&name, TensorType::f32(vec![10, 4])).unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(out).unwrap();
        let error = legalize(&g, &caps_with_limit(64), &CompileLimits::STANDARD)
            .expect_err("a packed source never splits");
        assert!(matches!(
            error,
            LegalizeError::OversizedConstant { ref name, .. } if name.starts_with("w.head")
        ));
    }

    /// A matmul weight already under the limit stays dense: legalize's split/host rewrites are a
    /// no-op, and consts are untouched.
    #[test]
    fn legalize_leaves_a_small_matmul_weight_dense() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 64])).unwrap();
        let w = b
            .constant("small.weight", TensorType::f32(vec![64, 128]))
            .unwrap(); // 32 KB
        let out = b.matmul(x, w).unwrap();
        let g = b.finish(out).unwrap();

        let caps = caps_with_limit(2_147_483_647);
        let legalized = legalize(&g, &caps, &CompileLimits::STANDARD)
            .expect("under the limit, no rewrite needed");
        assert_eq!(legalized.consts(), g.consts());
        assert_eq!(legalized.eqns().len(), g.eqns().len());
    }

    /// SC-002 (refuse): a matmul weight whose single row already exceeds the limit has no rewrite
    /// legalize knows - split cannot help (even the narrowest possible chunk is still too big) and this
    /// is not a gather-by-token pattern, so it refuses by name before any device allocation, instead of
    /// the executor discovering the same fact against a real buffer. Mutation (recorded, not
    /// committed): short-circuiting the `refuse_oversized_constants` call inside `legalize` returned
    /// `Ok` for this graph, still carrying the oversized `big.weight` const; `expect_err` panicked red,
    /// dumping the whole `Graph` in place of the expected error. Restoring the call made it green.
    #[test]
    fn legalize_refuses_a_matmul_weight_no_split_can_help() {
        let b = Builder::for_tests();
        // k alone (600,000,000 * 4 bytes = 2.4 GB) is over the limit, so even a single-row chunk
        // does not fit: `row_chunks` returns `None` and the split rewrite does not apply. `x` is a
        // broadcast of one real scalar, not a declared 2.4 GB constant: `splittable_matmul` never looks
        // at the activation's storage, only the weight's, so this stays a cheap fixture.
        let x0 = b.constant("x0", TensorType::f32(vec![1, 1])).unwrap();
        let x = b.broadcast(x0, vec![1, 600_000_000]).unwrap();
        let w = b
            .constant("big.weight", TensorType::f32(vec![2, 600_000_000]))
            .unwrap();
        let out = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(out).unwrap();

        let caps = caps_with_limit(2_147_483_647);
        let error = legalize(&g, &caps, &CompileLimits::STANDARD).expect_err(
            "no rewrite hosts a matmul weight whose single row already exceeds the limit",
        );
        assert_eq!(
            error,
            LegalizeError::OversizedConstant {
                name: "big.weight".to_string(),
                bytes: 600_000_000u64 * 2 * 4,
                limit: 2_147_483_647,
            }
        );
    }

    /// `refuse_oversized_constants` checks liveness, not mere declaration: `compile` runs `legalize`
    /// right after `dce`, and `dce` only shrinks `eqns` (dead entries stay in `values`/`consts`), so a
    /// hand-built graph with an oversized const nothing reads any more must NOT refuse - a real device
    /// run never touches it. Mutation (recorded, not committed): reverting `refuse_oversized_constants`
    /// to scan `g.consts` unconditionally (dropping the liveness filter) made this refuse with
    /// `OversizedConstant { name: "dead.weight", bytes: 3276800000, limit: 2147483647 }`; `expect`
    /// panicked red. Restoring the liveness filter made it green.
    #[test]
    fn legalize_ignores_a_dead_oversized_constant() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![1, 4])).unwrap();
        let live = b
            .constant("live.weight", TensorType::f32(vec![4, 8]))
            .unwrap();
        let out = b.matmul(x, live).unwrap();
        let traced = b.finish(out).unwrap();

        // A dead oversized const: declared in `values`/`inputs`/`consts`, referenced by no eqn, no
        // output, no state - exactly what `dce` leaves behind (it never touches these tables).
        let mut edit = traced.edit();
        let dead = edit.values_mut().push(ValueMeta::new(
            TensorType::f32(vec![4096, 200_000]), // ~3.05 GB, unreachable
            Storage::Const,
            Some("dead.weight".to_string()),
        ));
        edit.inputs_mut().push(dead);
        edit.consts_mut().push(dead);
        let g = edit.commit().unwrap();

        let caps = caps_with_limit(2_147_483_647);
        let legalized = legalize(&g, &caps, &CompileLimits::STANDARD)
            .expect("a dead oversized const must not be refused");
        assert_eq!(legalized.result(), g.result());
    }
}
