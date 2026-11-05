//! `store_weights_f16` (POOT-1477): the driver's one weight format.
//!
//! Under the `Narrow` policy on a target whose matrix units take f16, a BF16 checkpoint weight that only dense
//! contractions and embedding row gathers read is declared F16 in every program of the driver: the bind converts
//! the checkpoint's bf16 words to f16 once (round to nearest even, a value past f16's range refused), and the
//! graph says so ([`GraphEdit::convert_weight`]), so the oracle evaluates the same f16 values the device reads.
//!
//! The decision reads the policy, the target's matrix units and the *kinds* of a weight's readers, never a
//! reader's own precision or a shape. A weight therefore gets the same format in a prefill program and a decode
//! program, whichever schedule each plans for its readers, and a tied embedding and head, two consts over one
//! stored tensor, get the same lane and share one upload. Every other BF16 weight (a norm scale, a router, an
//! expert stack) keeps the checkpoint's format.

use std::collections::HashSet;

use poot_graph_ir::graph::{EditError, Graph, Operand, Storage, ValueId};
use poot_graph_ir::op::{ContractionHead, OpKind};
use poot_graph_ir::{FragmentDtype, MatmulPrecision};
use poot_tensor::DType;

use super::operand_precision::FragmentUnits;

/// Whether the driver stores its BF16 dense weights as f16: the policy narrows and the units multiply in f16.
pub(crate) fn weights_are_f16(policy: MatmulPrecision, units: FragmentUnits) -> bool {
    policy == MatmulPrecision::Narrow && units.takes(FragmentDtype::F16)
}

/// `g` with every BF16 weight only dense contractions and row gathers read declared F16, and those readers'
/// weight dtype with it. A weight that escapes the graph, or has any other reader, keeps its format.
pub(crate) fn store_weights_f16(g: &Graph) -> Result<Graph, EditError> {
    let escapes: HashSet<ValueId> = g.liveness_roots().collect();
    let mut candidates: HashSet<ValueId> = g
        .values()
        .ids()
        .filter(|&id| {
            let meta = g.meta(id);
            meta.storage == Storage::Const
                && meta.aval.dtype == DType::BF16
                && !escapes.contains(&id)
        })
        .collect();
    // A weight is converted when it has a reader and every reader reads it as a dense weight.
    let mut read = HashSet::new();
    for eqn in g.eqns() {
        for (position, operand) in eqn.inputs.iter().enumerate() {
            let Operand::Value(value) = operand else {
                continue;
            };
            if reads_as_dense_weight(&eqn.op, position) {
                read.insert(*value);
            } else {
                candidates.remove(value);
            }
        }
    }
    candidates.retain(|id| read.contains(id));
    if candidates.is_empty() {
        return Ok(g.clone());
    }
    let mut edit = g.edit();
    for &id in &candidates {
        edit.convert_weight(id)?;
    }
    for eqn in edit.eqns_mut() {
        let reads_converted = matches!(
            eqn.inputs.get(dense_weight_position(&eqn.op)),
            Some(Operand::Value(weight)) if candidates.contains(weight)
        );
        if reads_converted {
            eqn.op = with_f16_weight(&eqn.op);
        }
    }
    edit.commit()
}

/// The operand a dense reader reads its weight or table from.
fn dense_weight_position(op: &OpKind) -> usize {
    match op {
        OpKind::DenseRowGather { .. } => 0,
        _ => 1,
    }
}

/// Whether `op` reads its operand `position` as a BF16 dense weight or embedding table.
fn reads_as_dense_weight(op: &OpKind, position: usize) -> bool {
    match op {
        OpKind::DenseRowGather {
            source: DType::BF16,
        } => position == 0,
        _ => op.dense_weight() == Some(DType::BF16) && position == 1,
    }
}

/// `op` over an F16 weight in place of its BF16 one.
fn with_f16_weight(op: &OpKind) -> OpKind {
    match op {
        OpKind::DenseContraction { precision, .. } => OpKind::DenseContraction {
            weight: DType::F16,
            precision: *precision,
        },
        OpKind::GroupedDenseContraction {
            groups, precision, ..
        } => OpKind::GroupedDenseContraction {
            weight: DType::F16,
            groups: *groups,
            precision: *precision,
        },
        OpKind::ContractionEpilogue {
            head: ContractionHead::Dense { precision, .. },
            region,
        } => OpKind::ContractionEpilogue {
            head: ContractionHead::Dense {
                weight: DType::F16,
                precision: *precision,
            },
            region: region.clone(),
        },
        OpKind::DenseRowGather { .. } => OpKind::DenseRowGather { source: DType::F16 },
        other => unreachable!("{} is no dense weight reader", other.kind_name()),
    }
}
