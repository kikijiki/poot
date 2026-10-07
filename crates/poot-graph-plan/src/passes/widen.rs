use poot_graph_ir::op::{FusedOp, FusedOperand, FusedStep};

use super::*;

/// Fold the widening `Cast` of a narrow weight into the fused region that reads it.
///
/// A norm scale or a bias is stored BF16 and read as F32 by the arithmetic around it, so the traced graph
/// widens it with a `Cast` to F32 before the [`OpKind::FusedRow`] or [`OpKind::ContractionEpilogue`] that
/// uses it. Left alone, that cast is a dispatch per weight per token that materializes an f32 copy of a
/// weight the checkpoint stores in two bytes. This pass makes the cast a [`FusedOp::Cast`] step of the
/// region instead: the region reads the BF16 const through its packed lane and widens it in registers, so
/// the stored weight is the only copy and the dispatch is gone. Widening is exact, so the region computes
/// what it computed over the cast value.
///
/// Folds a cast only when nothing else could read it: the cast's result has the region as its one
/// reader and is not pinned, and the BF16 const it widens has the cast as its one reader, so the const's
/// every reader is a packed reader ([`crate::bf16_const_feeds_only_packed_readers`]). Pointwise
/// [`OpKind::Fused`] regions are left as they are: their kernels read every leaf as f32.
pub fn fuse_widening_casts<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut producer: Vec<Option<usize>> = vec![None; g.values.len()];
    let mut uses: Vec<usize> = vec![0; g.values.len()];
    for (e, eqn) in g.eqns.iter().enumerate() {
        producer[eqn.out] = Some(e);
        for operand in &eqn.inputs {
            if let Operand::Value(v) = operand {
                uses[*v] += 1;
            }
        }
    }
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    // The narrow const a single-use widening `Cast` produces `value` from, and the cast's equation.
    let widened_const = |value: ValueId| -> Option<(usize, ValueId)> {
        let cast = producer[value]?;
        let eqn = &g.eqns[cast];
        let [Operand::Value(source)] = eqn.inputs.as_slice() else {
            return None;
        };
        let foldable = matches!(eqn.op, OpKind::Cast { to: DType::F32 })
            && uses[value] == 1
            && !pinned.contains(&value)
            && g.meta(*source).storage == Storage::Const
            && g.aval(*source).dtype == DType::BF16
            && uses[*source] == 1
            && !pinned.contains(source);
        foldable.then_some((cast, *source))
    };

    let mut folded: HashMap<usize, Eqn> = HashMap::new();
    let mut dropped: HashSet<usize> = HashSet::new();
    for (e, eqn) in g.eqns.iter().enumerate() {
        let mut rewritten = eqn.clone();
        let mut changed = false;
        for position in 0..eqn.inputs.len() {
            let Operand::Value(value) = eqn.inputs[position] else {
                continue;
            };
            // The region's local that operand `position` is: a row region's inputs are its operands, an
            // epilogue's local `n >= 1` is the operand after the head's two, and its local 0 is the head.
            let local = match &eqn.op {
                OpKind::FusedRow(_) => position,
                OpKind::ContractionEpilogue { head, .. } if position >= head.operands() => {
                    position - head.operands() + 1
                }
                _ => continue,
            };
            let Some((cast, source)) = widened_const(value) else {
                continue;
            };
            match &mut rewritten.op {
                OpKind::FusedRow(region) => widen_row_input(region, local, DType::F32),
                OpKind::ContractionEpilogue { region, .. } => {
                    widen_region_input(region, local, DType::F32);
                }
                _ => unreachable!("only region ops reach the rewrite"),
            }
            rewritten.inputs[position] = Operand::Value(source);
            dropped.insert(cast);
            changed = true;
        }
        if changed {
            folded.insert(e, rewritten);
        }
    }
    if folded.is_empty() {
        return g.clone();
    }
    let eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter(|(e, _)| !dropped.contains(e))
        .map(|(e, eqn)| folded.remove(&e).unwrap_or_else(|| eqn.clone()))
        .collect();
    Graph { eqns, ..g.clone() }
}

/// Renumber a region's locals after [`FusedStep`] `Cast` of input `input` is inserted as the first step: the
/// cast result takes local `n_inputs`, every step result shifts up by one, and every read of `input` moves to
/// the cast result.
fn widened_local(local: usize, input: usize, n_inputs: usize) -> usize {
    if local == input {
        n_inputs
    } else if local >= n_inputs {
        local + 1
    } else {
        local
    }
}

fn widened_operand(operand: &FusedOperand, input: usize, n_inputs: usize) -> FusedOperand {
    match operand {
        FusedOperand::Local(local) => FusedOperand::Local(widened_local(*local, input, n_inputs)),
        other => *other,
    }
}

/// The widening step itself: a `Cast` to `to` of region input `input`.
fn widening_step(input: usize, to: DType) -> FusedStep {
    FusedStep {
        op: FusedOp::Cast { to },
        inputs: vec![FusedOperand::Local(input)],
    }
}

/// Insert a `Cast` of input `input` to `to` as `region`'s first step, and read its result wherever the region
/// read the input.
fn widen_region_input(region: &mut FusedRegion, input: usize, to: DType) {
    let n = region.n_inputs;
    for step in &mut region.steps {
        for operand in &mut step.inputs {
            *operand = widened_operand(operand, input, n);
        }
    }
    region.steps.insert(0, widening_step(input, to));
    region.output = widened_local(region.output, input, n);
    for lane in &mut region.pack {
        *lane = widened_local(*lane, input, n);
    }
}

/// [`widen_region_input`] for a row region.
fn widen_row_input(region: &mut RowRegion, input: usize, to: DType) {
    let n = region.n_inputs;
    for step in &mut region.steps {
        match step {
            RowStep::Pointwise { inputs, .. } => {
                for operand in inputs {
                    *operand = widened_operand(operand, input, n);
                }
            }
            RowStep::Reduce { input: reduced, .. } => {
                *reduced = widened_operand(reduced, input, n);
            }
        }
    }
    region.steps.insert(
        0,
        RowStep::Pointwise {
            op: FusedOp::Cast { to },
            inputs: vec![FusedOperand::Local(input)],
        },
    );
    region.output = widened_local(region.output, input, n);
}
