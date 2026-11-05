//! `store_narrowed_activations` (POOT-1448, `receipts/1448-activation-narrowed-once/design.md`): a value whose
//! every reader narrows it is stored narrow, rounded once by the kernel that computes it.
//!
//! Under a narrowed precision a dense contraction reads its activation rounded to the fragment dtype. When every
//! reader of an f32 value does that, to one dtype, the value's f32 bits are never read: the pass gives the region
//! that computes it a final `Cast` to that dtype and retypes the value, so the producer's kernel rounds each value
//! once, where it computes it, and stores half the bytes; the readers read it as stored. The rounding is the one
//! the readers applied (`Cast` and the operand narrowing are both round to nearest even, non-saturating, NaN
//! kept), and a value already of the fragment dtype enters a narrowed product unrounded, so the program computes
//! what it computed.
//!
//! The pass reads the graph and the planner, and nothing else: no model, shape, kernel name or threshold. A value
//! is narrowed only where the target plans the rewritten producer and every reader (`plans`, the probe every pass
//! revert uses); a target without native two-byte elements, a GEMV reader or a producer whose body has no narrow
//! store refuses by name, and the value stays f32. It never adds an equation, so the plan's dispatches are the
//! same, and under an exact policy no reader narrows, so it changes nothing.

use super::{
    DType, EditError, Eqn, FusedRegion, Graph, HashMap, HashSet, OpKind, Operand, ValueId,
};
use poot_graph_ir::op::{ContractionHead, LocalId};

/// `g` with every value whose readers all narrow it to one dtype stored in that dtype, where `plans` (whether the
/// target plans every one of these equations of a graph) admits the rewritten producer and every reader.
pub(crate) fn store_narrowed_activations(
    g: &Graph,
    plans: impl Fn(&Graph, &[&Eqn]) -> bool,
) -> Result<Graph, EditError> {
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    // Every read of each value: the reading equation's index and the operand position.
    let mut reads: HashMap<ValueId, Vec<(usize, usize)>> = HashMap::new();
    for (e, eqn) in g.eqns().iter().enumerate() {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            if let Operand::Value(value) = operand {
                reads.entry(*value).or_default().push((e, position));
            }
        }
    }
    let mut graph = g.clone();
    for e in 0..g.eqns().len() {
        let eqn = &graph.eqns()[e];
        let value = eqn.out;
        if graph.aval(value).dtype != DType::F32 || pinned.contains(&value) {
            continue;
        }
        let Some(readers) = reads.get(&value) else {
            continue;
        };
        let Some(dtype) = narrowing_dtype(&graph, readers) else {
            continue;
        };
        let Some(op) = narrowing_producer(&eqn.op, dtype) else {
            continue;
        };
        let mut edit = graph.edit();
        edit.eqns_mut()[e].op = op;
        edit.values_mut()[value].aval.dtype = dtype;
        let trial = edit.commit()?;
        let probed: Vec<&Eqn> = std::iter::once(e)
            .chain(readers.iter().map(|&(r, _)| r))
            .map(|i| &trial.eqns()[i])
            .collect();
        if plans(&trial, &probed) {
            graph = trial;
        }
    }
    Ok(graph)
}

/// The one dtype every read in `readers` narrows its value to, if each is the activation operand of a dense
/// contraction whose precision narrows it ([`poot_graph_ir::OperandPrecision::stored_activation`]) and all agree.
fn narrowing_dtype(g: &Graph, readers: &[(usize, usize)]) -> Option<DType> {
    let mut dtypes = readers.iter().map(|&(e, position)| {
        let op = &g.eqns()[e].op;
        let dense = matches!(
            op,
            OpKind::DenseContraction { .. }
                | OpKind::GroupedDenseContraction { .. }
                | OpKind::ContractionEpilogue {
                    head: ContractionHead::Dense { .. },
                    ..
                }
        );
        (dense && position == 0)
            .then(|| op.operand_precision()?.stored_activation())
            .flatten()
    });
    let first = dtypes.next()??;
    dtypes.all(|dtype| dtype == Some(first)).then_some(first)
}

/// The producer `op` with its result narrowed to `dtype` by a final `Cast`: a pointwise, row or epilogue region
/// gains the step; a plain dense contraction becomes the epilogue of that one step. `None` for any other op (a
/// movement, an attention, a gather), whose body has no narrow store, and for a region already narrowing.
fn narrowing_producer(op: &OpKind, dtype: DType) -> Option<OpKind> {
    Some(match op {
        OpKind::Fused(region) if region.narrowed_output().is_none() && region.pack.is_empty() => {
            OpKind::Fused(region.clone().narrowing_output(dtype))
        }
        OpKind::FusedRow(region) if region.narrowed_output().is_none() => {
            OpKind::FusedRow(region.clone().narrowing_output(dtype))
        }
        OpKind::ContractionEpilogue { head, region } if region.narrowed_output().is_none() => {
            OpKind::ContractionEpilogue {
                head: *head,
                region: region.clone().narrowing_output(dtype),
            }
        }
        OpKind::DenseContraction { weight, precision } => OpKind::ContractionEpilogue {
            head: ContractionHead::Dense {
                weight: *weight,
                precision: *precision,
            },
            region: FusedRegion {
                n_inputs: 1,
                steps: Vec::new(),
                output: LocalId::new(0),
                pack: Vec::new(),
            }
            .narrowing_output(dtype),
        },
        _ => return None,
    })
}
