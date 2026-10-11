use poot_graph_ir::op::{FusedOp, FusedOperand, FusedStep};

use super::fuse::{is_i32_pointwise, lone_pointwise_region};
use super::*;
use poot_graph_ir::ValueTable;
use poot_graph_ir::graph::GraphEdit;

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
pub fn fuse_widening_casts(g: &Graph) -> Result<Graph, EditError> {
    let mut producer: ValueTable<Option<usize>> = ValueTable::filled(g.values().len(), None);
    let mut uses: ValueTable<usize> = ValueTable::filled(g.values().len(), 0);
    for (e, eqn) in g.eqns().iter().enumerate() {
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
        let eqn = &g.eqns()[cast];
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
    for (e, eqn) in g.eqns().iter().enumerate() {
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
        return Ok(g.clone());
    }
    let eqns = g
        .eqns()
        .iter()
        .enumerate()
        .filter(|(e, _)| !dropped.contains(e))
        .map(|(e, eqn)| folded.remove(&e).unwrap_or_else(|| eqn.clone()))
        .collect();
    with_eqns(g, eqns)
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

/// Fold the `Cast` of an I32 value to F32 into the pointwise kernel that reads it or the one that writes it.
///
/// A position, a witness flag or a one-hot is an exact I32 word that float arithmetic reads as a number, so
/// the traced graph casts it. Left alone the cast is a dispatch (`I32ToF32`) and a stored f32 copy of a word
/// that is already in memory. Two folds remove it:
///
/// * The **producer** is an exact-I32 pointwise equation (a [`OpKind::Fused`] region or a lone `GeU`, `Sub`,
///   `Select`, ...) whose only reader is the cast. The cast becomes the region's last step
///   ([`FusedOp::Cast`]), so the I32 kernel writes the f32 result itself.
/// * The **consumers** are float pointwise equations ([`OpKind::Fused`] regions or lone pointwise steps),
///   directly or through `Reshape`/`Broadcast` views, which cost no dispatch and carry any dtype. The views
///   become I32 and each consumer region casts the input as its first step, reading the I32 word and
///   converting in a register. A consumer that is not a float pointwise equation (a contraction, a row
///   region, a movement) keeps the cast: no fold is half-applied.
///
/// The conversion is the standalone cast kernel's own (`Cast` to f32 rounds to nearest), so every consumer
/// computes what it computed over the stored value. A producer fold needs the cast to be the producer's one
/// reader and the producer's output unpinned; the cast's own result may be pinned (it reuses the region's
/// output). A consumer fold needs the cast's result and each view's unpinned.
pub fn fuse_index_casts(g: &Graph) -> Result<Graph, EditError> {
    let mut folded = g.edit();
    while let Some(next) = fold_one_index_cast(&folded) {
        folded = next;
    }
    folded.commit()
}

/// The graph with the first foldable I32-to-F32 cast folded, or `None` when none is.
fn fold_one_index_cast(g: &Graph) -> Option<GraphEdit> {
    let mut producer: ValueTable<Option<usize>> = ValueTable::filled(g.values().len(), None);
    let mut readers: ValueTable<Vec<usize>> = ValueTable::filled(g.values().len(), Vec::new());
    for (e, eqn) in g.eqns().iter().enumerate() {
        producer[eqn.out] = Some(e);
        for operand in &eqn.inputs {
            if let Operand::Value(v) = operand {
                readers[*v].push(e);
            }
        }
    }
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    g.eqns().iter().enumerate().find_map(|(cast, eqn)| {
        let (OpKind::Cast { to: DType::F32 }, [Operand::Value(source)]) =
            (&eqn.op, eqn.inputs.as_slice())
        else {
            return None;
        };
        if g.aval(*source).dtype != DType::I32 {
            return None;
        }
        fold_into_producer(g, cast, &producer, &readers, &pinned)
            .or_else(|| fold_into_consumers(g, cast, &readers, &pinned))
    })
}

/// The cast `cast` as the last step of its producer, which then writes the cast's value. When the value's
/// one reader is a float pointwise equation of the same shape, the region joins it: one kernel computes the
/// exact words, converts them and runs the arithmetic that reads them.
fn fold_into_producer(
    g: &Graph,
    cast: usize,
    producer: &ValueTable<Option<usize>>,
    readers: &ValueTable<Vec<usize>>,
    pinned: &HashSet<ValueId>,
) -> Option<GraphEdit> {
    let Operand::Value(source) = g.eqns()[cast].inputs[0] else {
        return None;
    };
    let p = producer[source]?;
    if readers[source].len() != 1 || pinned.contains(&source) {
        return None;
    }
    let produced = &g.eqns()[p];
    if g.aval(produced.out).dtype != DType::I32 {
        return None;
    }
    let (mut region, leaves) = match &produced.op {
        OpKind::Fused(region) if region.pack.is_empty() => {
            (region.clone(), produced.inputs.clone())
        }
        _ if is_i32_pointwise(g, produced) => {
            let (region, leaves) = lone_pointwise_region(g, produced)?;
            (region, leaves.into_iter().map(Operand::Value).collect())
        }
        _ => return None,
    };
    // A packed lane is an exact-I32 region's own store: a float region has no lane read.
    let reads_lane = region.steps.iter().any(|step| {
        step.inputs
            .iter()
            .any(|input| matches!(input, FusedOperand::PackLane { .. }))
    });
    if reads_lane {
        return None;
    }
    region.steps.push(FusedStep {
        op: FusedOp::Cast { to: DType::F32 },
        inputs: vec![FusedOperand::Local(region.output)],
    });
    region.output = region.n_inputs + region.steps.len() - 1;
    let cast_out = g.eqns()[cast].out;
    let mut eqns = g.eqns().to_vec();
    let mut removed = vec![cast];
    let joined = match readers[cast_out].as_slice() {
        [reader] if !pinned.contains(&cast_out) => {
            join_reader(g, *reader, cast_out, &region, &leaves)
        }
        _ => None,
    };
    match joined {
        Some((reader, merged)) => {
            eqns[reader] = merged;
            removed.push(p);
        }
        None => {
            eqns[p] = Eqn {
                op: OpKind::Fused(region),
                inputs: leaves,
                out: cast_out,
                layer: produced.layer,
            };
        }
    }
    removed.sort_unstable();
    for index in removed.into_iter().rev() {
        eqns.remove(index);
    }
    let mut folded = g.edit();
    *folded.eqns_mut() = eqns;
    Some(folded)
}

/// `reader` with the region `producer` (over `producer_leaves`, writing `value`) inlined: the reader's one
/// operand `value` becomes the producer's steps, so the producer's output is never stored. Only an equation
/// of the producer's shape joins it, so the steps keep one output shape; `None` for any other reader.
fn join_reader(
    g: &Graph,
    reader: usize,
    value: ValueId,
    producer: &FusedRegion,
    producer_leaves: &[Operand],
) -> Option<(usize, Eqn)> {
    let eqn = &g.eqns()[reader];
    if !reads_float_pointwise(g, eqn) || g.aval(eqn.out).shape != g.aval(value).shape {
        return None;
    }
    let (consumer, consumer_leaves): (FusedRegion, Vec<Operand>) = match &eqn.op {
        OpKind::Fused(region) => (region.clone(), eqn.inputs.clone()),
        _ => {
            let (region, leaves) = lone_pointwise_region(g, eqn)?;
            (region, leaves.into_iter().map(Operand::Value).collect())
        }
    };
    let slot = consumer_leaves
        .iter()
        .position(|leaf| matches!(leaf, Operand::Value(v) if *v == value))?;
    // The merged leaves: the consumer's but the one joined, then the producer's not already among them.
    let mut leaves: Vec<Operand> = Vec::new();
    let mut consumer_local: Vec<usize> = Vec::with_capacity(consumer_leaves.len());
    for (index, leaf) in consumer_leaves.iter().enumerate() {
        if index == slot {
            consumer_local.push(usize::MAX);
        } else {
            consumer_local.push(leaves.len());
            leaves.push(*leaf);
        }
    }
    let mut producer_local: Vec<usize> = Vec::with_capacity(producer_leaves.len());
    for leaf in producer_leaves {
        let at = leaves
            .iter()
            .position(
                |other| matches!((other, leaf), (Operand::Value(a), Operand::Value(b)) if a == b),
            )
            .unwrap_or_else(|| {
                leaves.push(*leaf);
                leaves.len() - 1
            });
        producer_local.push(at);
    }
    let n_inputs = leaves.len();
    let producer_step = |local: usize| match local < producer.n_inputs {
        true => producer_local[local],
        false => n_inputs + (local - producer.n_inputs),
    };
    let produced = n_inputs + producer.steps.len();
    let consumer_step = |local: usize| {
        if local == slot {
            producer_step(producer.output)
        } else if local < consumer.n_inputs {
            consumer_local[local]
        } else {
            produced + (local - consumer.n_inputs)
        }
    };
    let map = |operand: &FusedOperand, local: &dyn Fn(usize) -> usize| match operand {
        FusedOperand::Local(l) => FusedOperand::Local(local(*l)),
        other => *other,
    };
    let mut steps: Vec<FusedStep> = producer
        .steps
        .iter()
        .map(|step| FusedStep {
            op: step.op,
            inputs: step.inputs.iter().map(|o| map(o, &producer_step)).collect(),
        })
        .collect();
    steps.extend(consumer.steps.iter().map(|step| FusedStep {
        op: step.op,
        inputs: step.inputs.iter().map(|o| map(o, &consumer_step)).collect(),
    }));
    let region = FusedRegion {
        n_inputs,
        steps,
        output: consumer_step(consumer.output),
        pack: Vec::new(),
    };
    let merged = Eqn {
        op: OpKind::Fused(region),
        inputs: leaves,
        out: eqn.out,
        layer: eqn.layer,
    };
    Some((reader, merged))
}

/// The cast `cast` as a first step of every float pointwise equation that reads its value, through views.
fn fold_into_consumers(
    g: &Graph,
    cast: usize,
    readers: &ValueTable<Vec<usize>>,
    pinned: &HashSet<ValueId>,
) -> Option<GraphEdit> {
    let (cast_out, Operand::Value(source)) = (g.eqns()[cast].out, g.eqns()[cast].inputs[0]) else {
        return None;
    };
    // The values the cast result reaches through views, and the equations that read one of them: the views
    // retype to I32, the readers cast.
    let mut views: Vec<usize> = Vec::new();
    let mut terminals: Vec<usize> = Vec::new();
    let mut reached: HashSet<ValueId> = HashSet::from([cast_out]);
    let mut frontier = vec![cast_out];
    while let Some(value) = frontier.pop() {
        if pinned.contains(&value) || readers[value].is_empty() {
            return None;
        }
        for &reader in &readers[value] {
            let eqn = &g.eqns()[reader];
            let is_view = matches!(eqn.op, OpKind::Reshape { .. } | OpKind::Broadcast { .. });
            if is_view && eqn.inputs.len() == 1 {
                if !views.contains(&reader) {
                    views.push(reader);
                    reached.insert(eqn.out);
                    frontier.push(eqn.out);
                }
            } else if reads_float_pointwise(g, eqn) {
                if !terminals.contains(&reader) {
                    terminals.push(reader);
                }
            } else {
                return None;
            }
        }
    }
    // A terminal that already reads the I32 source as itself (a select condition, say) shares one leaf
    // between that read and the cast: casting the leaf would cast the condition too.
    let reads_source = |eqn: &Eqn| {
        eqn.inputs
            .iter()
            .any(|operand| matches!(operand, Operand::Value(v) if *v == source))
    };
    if terminals
        .iter()
        .any(|&terminal| reads_source(&g.eqns()[terminal]))
    {
        return None;
    }
    let mut folded = g.edit();
    for &view in &views {
        folded.values_mut()[g.eqns()[view].out].aval.dtype = DType::I32;
    }
    // The cast result is the I32 source from here on.
    let uncast = |operand: &Operand| match operand {
        Operand::Value(v) if *v == cast_out => Operand::Value(source),
        other => *other,
    };
    let mut rewritten: HashMap<usize, Eqn> = HashMap::new();
    for &view in &views {
        let mut eqn = g.eqns()[view].clone();
        eqn.inputs = eqn.inputs.iter().map(uncast).collect();
        rewritten.insert(view, eqn);
    }
    for &terminal in &terminals {
        let eqn = &g.eqns()[terminal];
        let (mut region, leaves) = match &eqn.op {
            OpKind::Fused(region) => (region.clone(), eqn.inputs.clone()),
            _ => {
                let (region, leaves) = lone_pointwise_region(g, eqn)?;
                (region, leaves.into_iter().map(Operand::Value).collect())
            }
        };
        for (slot, leaf) in leaves.iter().enumerate() {
            if matches!(leaf, Operand::Value(v) if reached.contains(v)) {
                widen_region_input(&mut region, slot, DType::F32);
            }
        }
        rewritten.insert(
            terminal,
            Eqn {
                op: OpKind::Fused(region),
                inputs: leaves.iter().map(uncast).collect(),
                out: eqn.out,
                layer: eqn.layer,
            },
        );
    }
    *folded.eqns_mut() = g
        .eqns()
        .iter()
        .enumerate()
        .filter(|(e, _)| *e != cast)
        .map(|(e, eqn)| rewritten.remove(&e).unwrap_or_else(|| eqn.clone()))
        .collect();
    Some(folded)
}

/// Whether `eqn` is a float pointwise equation a cast of one of its operands can fold into: a fused region
/// or a lone pointwise step, computing in f32.
fn reads_float_pointwise(g: &Graph, eqn: &Eqn) -> bool {
    if g.aval(eqn.out).dtype != DType::F32 {
        return false;
    }
    match &eqn.op {
        OpKind::Fused(region) => region.pack.is_empty(),
        _ => lone_pointwise_region(g, eqn).is_some(),
    }
}
