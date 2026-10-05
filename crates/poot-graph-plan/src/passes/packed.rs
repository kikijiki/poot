use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedDequantErrorContext {
    pub value: ValueId,
    pub format: poot_quant::format::WeightFormat,
    pub logical_shape: [usize; 2],
}

impl PackedDequantErrorContext {
    fn new(value: ValueId, descriptor: poot_quant::PackedWeight) -> Self {
        Self {
            value,
            format: descriptor.format(),
            logical_shape: descriptor.shape(),
        }
    }
}

impl std::fmt::Display for PackedDequantErrorContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "v{} {:?} {:?}",
            self.value, self.format, self.logical_shape
        )
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PackedDequantProductionError {
    #[error(transparent)]
    InvalidGraph(#[from] GraphValidationError),
    #[error("packed contraction {context} was present before Card 355 recognition")]
    PreexistingContraction { context: PackedDequantErrorContext },
    #[error(
        "packed dequant {context} carrier v{carrier} uses {storage:?} storage; Const is required"
    )]
    CarrierStorage {
        context: PackedDequantErrorContext,
        carrier: ValueId,
        storage: Storage,
    },
    #[error(
        "packed dequant {context} {roles:?} carrier v{carrier} has invalid identity {name:?}: two distinct source roles resolved to the same graph value"
    )]
    CarrierIdentity {
        context: PackedDequantErrorContext,
        carrier: ValueId,
        roles: (poot_quant::SourceRole, poot_quant::SourceRole),
        name: Option<String>,
    },
    #[error("packed dequant {context} carrier v{carrier} escapes through {destination}")]
    CarrierEscape {
        context: PackedDequantErrorContext,
        carrier: ValueId,
        destination: &'static str,
    },
    #[error(
        "packed dequant {context} carrier v{carrier} has {consumers} equation consumers; only its claimed candidate is allowed"
    )]
    CarrierConsumerCount {
        context: PackedDequantErrorContext,
        carrier: ValueId,
        consumers: usize,
    },
    #[error(
        "packed dequant {context} carriers are not exactly the descriptor's own source graph values"
    )]
    CarrierOperands { context: PackedDequantErrorContext },
    #[error("packed dequant {context} escapes as the graph output")]
    GraphOutput { context: PackedDequantErrorContext },
    #[error("packed dequant {context} escapes as a validation output")]
    ValidationOutput { context: PackedDequantErrorContext },
    #[error("packed dequant {context} has an internal value that escapes as carried-state output")]
    StateOutput { context: PackedDequantErrorContext },
    #[error(
        "packed dequant {context} internal v{internal} has {consumers} equation consumers; exactly one is required"
    )]
    ConsumerCount {
        context: PackedDequantErrorContext,
        internal: ValueId,
        consumers: usize,
    },
    #[error("packed dequant {context} has transpose permutation {perm:?}; expected [1, 0]")]
    TransposePermutation {
        context: PackedDequantErrorContext,
        perm: Vec<usize>,
    },
    #[error("packed dequant {context} reaches unsupported movement {operation}")]
    Movement {
        context: PackedDequantErrorContext,
        operation: String,
    },
    #[error("reachable packed dequant {context} was not claimed by a contraction")]
    Unmatched { context: PackedDequantErrorContext },
}

fn packed_consumer_map<V: ValidationChannel>(g: &Graph<V>) -> Vec<Vec<usize>> {
    let mut consumers = vec![Vec::new(); g.values.len()];
    for (equation, eqn) in g.eqns.iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                consumers[*value].push(equation);
            }
        }
    }
    consumers
}

fn is_graph_escape<V: ValidationChannel>(g: &Graph<V>, value: ValueId) -> bool {
    g.liveness_roots().any(|root| root == value)
}

#[cfg_attr(not(test), expect(dead_code, reason = "held for POOT-672"))]
pub(crate) fn reject_preexisting_packed_contractions<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PackedDequantProductionError> {
    for eqn in &g.eqns {
        if let OpKind::PackedContraction { descriptor, .. }
        | OpKind::PackedRowGather { descriptor } = eqn.op
        {
            return Err(PackedDequantProductionError::PreexistingContraction {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
    }
    Ok(())
}

fn value_input(eqn: &Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

/// Validate one `PackedDequant`/`PackedContraction` equation's carrier operands and return them in
/// `descriptor.sources()` order.
///
/// The typed contract is the op itself: both `PackedDequant::infer` and `PackedContraction::infer`
/// (which delegates to it over `ins[1..]`) already tie operand `i` to
/// `descriptor.sources()[i]`'s `source_shape`, checked at `graph.validate()` before any recognizer
/// runs - so `carrier_inputs` is already the right length and dtype/shape per role; this function
/// checks the graph-structural facts `infer` cannot see (constant storage, no escape, exactly one
/// consumer, no two roles aliasing the same value). R469-007: identity is checked by graph structure,
/// not by parsing a `PackedSourceName` suffix off `ValueMeta::name` and matching linear-id prefixes -
/// one tracer's naming convention, not the graph's.
fn validate_packed_carriers<V: ValidationChannel>(
    g: &Graph<V>,
    equation: usize,
    descriptor: poot_quant::PackedWeight,
    carrier_inputs: &[Operand],
    consumers: &[Vec<usize>],
) -> Result<Vec<ValueId>, PackedDequantProductionError> {
    let context = PackedDequantErrorContext::new(g.eqns[equation].out, descriptor);
    let roles = descriptor.sources();
    if carrier_inputs.len() != roles.len() {
        return Err(PackedDequantProductionError::CarrierOperands { context });
    }
    let mut carriers = Vec::with_capacity(carrier_inputs.len());
    for input in carrier_inputs {
        let Operand::Value(carrier) = input else {
            return Err(PackedDequantProductionError::CarrierOperands { context });
        };
        carriers.push(*carrier);
    }
    for &carrier in &carriers {
        if g.meta(carrier).storage != Storage::Const {
            return Err(PackedDequantProductionError::CarrierStorage {
                context,
                carrier,
                storage: g.meta(carrier).storage,
            });
        }
        if g.output == carrier {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "graph output",
            });
        }
        if g.validation_outputs()
            .iter()
            .any(|validation| validation.value == carrier)
        {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "validation output",
            });
        }
        if g.state
            .iter()
            .any(|&(state_in, state_out)| state_in == carrier || state_out == carrier)
        {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "carried state",
            });
        }
        if consumers[carrier].as_slice() != [equation] {
            return Err(PackedDequantProductionError::CarrierConsumerCount {
                context,
                carrier,
                consumers: consumers[carrier].len(),
            });
        }
    }
    for i in 0..carriers.len() {
        for j in (i + 1)..carriers.len() {
            if carriers[i] == carriers[j] {
                return Err(PackedDequantProductionError::CarrierIdentity {
                    context,
                    carrier: carriers[i],
                    roles: (roles[i], roles[j]),
                    name: g.meta(carriers[i]).name.clone(),
                });
            }
        }
    }
    Ok(carriers)
}

/// One recognized packed-contraction chain, ready to replace its terminal `MatMul`.
struct PackedContractionCandidate {
    member_indices: Vec<usize>,
    /// The dequant's carrier operands, in `descriptor.sources()` order.
    sources: Vec<ValueId>,
    descriptor: poot_quant::PackedWeight,
    blocks: usize,
}

/// Recognize `ops::packed_linear`'s canonical chain: `PackedDequant -> Transpose([1,0]) -> optional
/// Reshape -> MatMul`. The optional reshape (`packed_linear`'s `alias_weight_shape`) follows the
/// transpose, so this chain can only express an ordinary `[out, k]` dequant - never a view that needs to
/// split the stored rows into blocks before the transpose (Card 385).
fn recognize_canonical_packed_chain<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &HashMap<ValueId, usize>,
    consumers: &[Vec<usize>],
    mut weight: ValueId,
) -> Option<PackedContractionCandidate> {
    let mut member_indices = Vec::new();
    if let Some(&reshape_index) = producer.get(&weight)
        && matches!(g.eqns[reshape_index].op, OpKind::Reshape { .. })
    {
        if consumers[weight].len() != 1 || is_graph_escape(g, weight) {
            return None;
        }
        member_indices.push(reshape_index);
        weight = value_input(&g.eqns[reshape_index], 0)?;
    }
    let &transpose_index = producer.get(&weight)?;
    let OpKind::Transpose { perm } = &g.eqns[transpose_index].op else {
        return None;
    };
    if perm.as_slice() != [1, 0] || consumers[weight].len() != 1 || is_graph_escape(g, weight) {
        return None;
    }
    let dequant_out = value_input(&g.eqns[transpose_index], 0)?;
    let &dequant_index = producer.get(&dequant_out)?;
    let OpKind::PackedDequant { descriptor } = g.eqns[dequant_index].op else {
        return None;
    };
    if consumers[dequant_out].len() != 1 || is_graph_escape(g, dequant_out) {
        return None;
    }
    let sources = validate_packed_carriers(
        g,
        dequant_index,
        descriptor,
        &g.eqns[dequant_index].inputs,
        consumers,
    )
    .ok()?;
    member_indices.extend([transpose_index, dequant_index]);
    Some(PackedContractionCandidate {
        member_indices,
        sources,
        descriptor,
        blocks: 1,
    })
}

/// Recognize `ops::packed_block_diagonal_linear`'s chain: `PackedDequant -> Reshape([blocks, out /
/// blocks, k]) -> Transpose([0,2,1]) -> MatMul`. The reshape runs before the transpose, the mirror of
/// the canonical chain above; it reaches the grouped `[blocks, in_per_group, out / blocks]` view a
/// stored `[out, k]` pair cannot otherwise expose (Card 385). `blocks` is read from the reshape in the
/// graph; a value that disagrees with `descriptor` is caught by the shared infer-and-compare gate in
/// the caller.
fn recognize_blocked_packed_chain<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &HashMap<ValueId, usize>,
    consumers: &[Vec<usize>],
    weight: ValueId,
) -> Option<PackedContractionCandidate> {
    let &transpose_index = producer.get(&weight)?;
    let OpKind::Transpose { perm } = &g.eqns[transpose_index].op else {
        return None;
    };
    if perm.as_slice() != [0, 2, 1] || consumers[weight].len() != 1 || is_graph_escape(g, weight) {
        return None;
    }
    let reshape_out = value_input(&g.eqns[transpose_index], 0)?;
    let &reshape_index = producer.get(&reshape_out)?;
    let OpKind::Reshape { shape } = &g.eqns[reshape_index].op else {
        return None;
    };
    if shape.len() != 3 || consumers[reshape_out].len() != 1 || is_graph_escape(g, reshape_out) {
        return None;
    }
    let blocks = shape[0];
    let dequant_out = value_input(&g.eqns[reshape_index], 0)?;
    let &dequant_index = producer.get(&dequant_out)?;
    let OpKind::PackedDequant { descriptor } = g.eqns[dequant_index].op else {
        return None;
    };
    if consumers[dequant_out].len() != 1 || is_graph_escape(g, dequant_out) {
        return None;
    }
    let sources = validate_packed_carriers(
        g,
        dequant_index,
        descriptor,
        &g.eqns[dequant_index].inputs,
        consumers,
    )
    .ok()?;
    Some(PackedContractionCandidate {
        member_indices: vec![transpose_index, reshape_index, dequant_index],
        sources,
        descriptor,
        blocks,
    })
}

/// Recognize the reversed spelling of `ops::packed_linear`'s canonical chain: `y = x @ W^T`
/// written as `y = (W @ x^T)^T` instead. `ops::packed_linear`'s canonical `PackedDequant` output is
/// already `[out, k]` (the dense sibling below needs a `Transpose` to reach that layout, but a packed
/// dequant starts there), so this spelling needs no transpose on the weight side at all: the dequant
/// (or dequant-then-reshape, matching the canonical chain's optional alias reshape) feeds the *first*
/// `MatMul` operand directly, `x^T` is the second, and the whole product is transposed back. Returns the
/// recognized candidate and the original (untransposed) activation; the caller still runs the shared
/// infer-and-compare gate against the outer `Transpose`'s declared output, so a near-miss (a shape that
/// does not actually correspond to `x @ W^T`) is declined there exactly like the other two chains.
fn recognize_transposed_canonical_packed_chain<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &HashMap<ValueId, usize>,
    consumers: &[Vec<usize>],
    mut weight: ValueId,
    activation_t: ValueId,
) -> Option<(PackedContractionCandidate, ValueId)> {
    let &at_index = producer.get(&activation_t)?;
    let OpKind::Transpose { perm } = &g.eqns[at_index].op else {
        return None;
    };
    if perm.as_slice() != [1, 0]
        || consumers[activation_t].len() != 1
        || is_graph_escape(g, activation_t)
    {
        return None;
    }
    let activation = value_input(&g.eqns[at_index], 0)?;

    let mut member_indices = vec![at_index];
    if let Some(&reshape_index) = producer.get(&weight)
        && matches!(g.eqns[reshape_index].op, OpKind::Reshape { .. })
    {
        if consumers[weight].len() != 1 || is_graph_escape(g, weight) {
            return None;
        }
        member_indices.push(reshape_index);
        weight = value_input(&g.eqns[reshape_index], 0)?;
    }
    let &dequant_index = producer.get(&weight)?;
    let OpKind::PackedDequant { descriptor } = g.eqns[dequant_index].op else {
        return None;
    };
    if consumers[weight].len() != 1 || is_graph_escape(g, weight) {
        return None;
    }
    let sources = validate_packed_carriers(
        g,
        dequant_index,
        descriptor,
        &g.eqns[dequant_index].inputs,
        consumers,
    )
    .ok()?;
    member_indices.push(dequant_index);
    Some((
        PackedContractionCandidate {
            member_indices,
            sources,
            descriptor,
            blocks: 1,
        },
        activation,
    ))
}

/// Replace only the exact single-consumer packed-linear compositions this module's builders emit (the
/// canonical dense chain, Card 385's block-diagonal chain, and the reversed `(W x^T)^T` spelling of the
/// canonical chain). This pass runs before transpose/view analysis and leaves a following
/// ordinary bias Add untouched.
///
/// Generic over the validation channel: a validation value is a liveness root, so a member that a witness
/// observes is a graph escape and its composition is left unrecognized.
pub fn recognize_packed_contractions<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns.iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();

    for (matmul_index, matmul) in g.eqns.iter().enumerate() {
        if !matches!(matmul.op, OpKind::MatMul) || matmul.inputs.len() != 2 {
            continue;
        }
        let (Some(activation), Some(weight)) = (value_input(matmul, 0), value_input(matmul, 1))
        else {
            continue;
        };
        let Some(candidate) = recognize_canonical_packed_chain(g, &producer, &consumers, weight)
            .or_else(|| recognize_blocked_packed_chain(g, &producer, &consumers, weight))
        else {
            continue;
        };
        let candidate_op = OpKind::PackedContraction {
            descriptor: candidate.descriptor,
            blocks: candidate.blocks,
        };
        let candidate_inputs: Vec<poot_graph_ir::types::TensorType> =
            std::iter::once(g.aval(activation).clone())
                .chain(
                    candidate
                        .sources
                        .iter()
                        .map(|&source| g.aval(source).clone()),
                )
                .collect();
        if candidate_op.infer(&candidate_inputs).ok().as_ref() != Some(g.aval(matmul.out)) {
            continue;
        }
        removed.extend(candidate.member_indices.iter().copied());
        replacements.insert(
            matmul_index,
            Eqn {
                op: candidate_op,
                inputs: std::iter::once(Operand::Value(activation))
                    .chain(
                        candidate
                            .sources
                            .iter()
                            .map(|&source| Operand::Value(source)),
                    )
                    .collect(),
                out: matmul.out,
                layer: matmul.layer,
            },
        );
    }

    for (transpose_index, transpose_eqn) in g.eqns.iter().enumerate() {
        if replacements.contains_key(&transpose_index) || removed.contains(&transpose_index) {
            continue;
        }
        let OpKind::Transpose { perm } = &transpose_eqn.op else {
            continue;
        };
        if perm.as_slice() != [1, 0] {
            continue;
        }
        let Some(product) = value_input(transpose_eqn, 0) else {
            continue;
        };
        let Some(&inner_matmul_index) = producer.get(&product) else {
            continue;
        };
        if replacements.contains_key(&inner_matmul_index) || removed.contains(&inner_matmul_index) {
            continue;
        }
        let inner_matmul = &g.eqns[inner_matmul_index];
        if !matches!(inner_matmul.op, OpKind::MatMul) || inner_matmul.inputs.len() != 2 {
            continue;
        }
        if consumers[product].len() != 1 || is_graph_escape(g, product) {
            continue;
        }
        let (Some(weight), Some(activation_t)) =
            (value_input(inner_matmul, 0), value_input(inner_matmul, 1))
        else {
            continue;
        };
        let Some((candidate, activation)) = recognize_transposed_canonical_packed_chain(
            g,
            &producer,
            &consumers,
            weight,
            activation_t,
        ) else {
            continue;
        };
        let candidate_op = OpKind::PackedContraction {
            descriptor: candidate.descriptor,
            blocks: candidate.blocks,
        };
        let candidate_inputs: Vec<poot_graph_ir::types::TensorType> =
            std::iter::once(g.aval(activation).clone())
                .chain(
                    candidate
                        .sources
                        .iter()
                        .map(|&source| g.aval(source).clone()),
                )
                .collect();
        if candidate_op.infer(&candidate_inputs).ok().as_ref() != Some(g.aval(transpose_eqn.out)) {
            continue;
        }
        removed.insert(inner_matmul_index);
        removed.extend(candidate.member_indices.iter().copied());
        replacements.insert(
            transpose_index,
            Eqn {
                op: candidate_op,
                inputs: std::iter::once(Operand::Value(activation))
                    .chain(
                        candidate
                            .sources
                            .iter()
                            .map(|&source| Operand::Value(source)),
                    )
                    .collect(),
                out: transpose_eqn.out,
                layer: transpose_eqn.layer,
            },
        );
    }

    let eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| eqn.clone()))
        })
        .collect();
    Graph { eqns, ..g.clone() }
}

/// Card 545a: claim `Gather { axis: 0 }(PackedDequant(carriers..), ids)` - a quantized
/// token-embedding lookup - as one `PackedRowGather(carriers.., ids)`, so only the gathered rows are
/// decoded and no `[out, K]` table is materialized. The dequant must have the gather as its only
/// consumer and not escape; its carriers must pass the same structural check as every claim.
pub fn recognize_packed_row_gathers<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns.iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();
    for (gather_index, gather) in g.eqns.iter().enumerate() {
        if !matches!(gather.op, OpKind::Gather { axis: 0 }) {
            continue;
        }
        let (Some(table), Some(ids)) = (value_input(gather, 0), value_input(gather, 1)) else {
            continue;
        };
        let Some(&dequant_index) = producer.get(&table) else {
            continue;
        };
        let OpKind::PackedDequant { descriptor } = g.eqns[dequant_index].op else {
            continue;
        };
        if consumers[table].len() != 1 || is_graph_escape(g, table) {
            continue;
        }
        let Ok(sources) = validate_packed_carriers(
            g,
            dequant_index,
            descriptor,
            &g.eqns[dequant_index].inputs,
            &consumers,
        ) else {
            continue;
        };
        let op = OpKind::PackedRowGather { descriptor };
        let inputs: Vec<poot_graph_ir::types::TensorType> = sources
            .iter()
            .chain(std::iter::once(&ids))
            .map(|&value| g.aval(value).clone())
            .collect();
        if op.infer(&inputs).ok().as_ref() != Some(g.aval(gather.out)) {
            continue;
        }
        removed.insert(dequant_index);
        replacements.insert(
            gather_index,
            Eqn {
                op,
                inputs: sources
                    .iter()
                    .chain(std::iter::once(&ids))
                    .map(|&value| Operand::Value(value))
                    .collect(),
                out: gather.out,
                layer: gather.layer,
            },
        );
    }
    let eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter_map(|(index, eqn)| {
            replacements
                .get(&index)
                .cloned()
                .or_else(|| (!removed.contains(&index)).then(|| eqn.clone()))
        })
        .collect();
    Graph { eqns, ..g.clone() }
}

/// Card 642: the one packed MoE shape the escape gate admits until Card 632 replaces it
/// with `RowSelect` contractions - the branch `ops::packed_indexed_linear`/`packed_grouped_linear`
/// stage per expert (`stage_packed_indexed_linear`): `PackedDequant -> Transpose([1, 0]) ->
/// Reshape([1, K, out]) -> Concat(axis 0) -> IndexedMatMul` (the concatenated table is the
/// contraction's weight operand), every `Concat` operand being such a branch of the same `[out, K]`.
/// Each dequant is then planned as `Materialize` and the table runs the dense `IndexedMatMul` kernel,
/// every member single-consumer and none a graph escape. Anything else stays refused.
fn is_canonical_indexed_branch<V: ValidationChannel>(
    g: &Graph<V>,
    dequant: usize,
    consumers: &[Vec<usize>],
    producer: &HashMap<ValueId, usize>,
) -> bool {
    let OpKind::PackedDequant { descriptor } = g.eqns[dequant].op else {
        return false;
    };
    let [out, k] = descriptor.shape();
    let sole_consumer = |value: ValueId| {
        (consumers[value].len() == 1 && !is_graph_escape(g, value))
            .then(|| &g.eqns[consumers[value][0]])
    };
    // The branch rooted at `reshape_out`'s producer chain, walked upward: Reshape <- Transpose <-
    // PackedDequant of this `[out, K]`, each member single-consumer and not an escape.
    let is_branch = |reshape_out: ValueId| -> bool {
        let Some(&reshape) = producer.get(&reshape_out) else {
            return false;
        };
        let reshape = &g.eqns[reshape];
        if !matches!(&reshape.op, OpKind::Reshape { shape } if shape.as_slice() == [1, k, out]) {
            return false;
        }
        let Some(transpose_out) = value_input(reshape, 0) else {
            return false;
        };
        let Some(&transpose) = producer.get(&transpose_out) else {
            return false;
        };
        let transpose = &g.eqns[transpose];
        if !matches!(&transpose.op, OpKind::Transpose { perm } if perm.as_slice() == [1, 0]) {
            return false;
        }
        let Some(dequant_out) = value_input(transpose, 0) else {
            return false;
        };
        let Some(&root) = producer.get(&dequant_out) else {
            return false;
        };
        matches!(g.eqns[root].op, OpKind::PackedDequant { descriptor } if descriptor.shape() == [out, k])
            && [dequant_out, transpose_out, reshape_out]
                .iter()
                .all(|&value| consumers[value].len() == 1 && !is_graph_escape(g, value))
    };
    let Some(transpose) = sole_consumer(g.eqns[dequant].out) else {
        return false;
    };
    let Some(reshape) = sole_consumer(transpose.out) else {
        return false;
    };
    let Some(concat) = sole_consumer(reshape.out) else {
        return false;
    };
    if !matches!(concat.op, OpKind::Concat { axis: 0 }) {
        return false;
    }
    let Some(contraction) = sole_consumer(concat.out) else {
        return false;
    };
    matches!(contraction.op, OpKind::IndexedMatMul)
        && value_input(contraction, 1) == Some(concat.out)
        && value_input(contraction, 0) != Some(concat.out)
        && value_input(contraction, 2) != Some(concat.out)
        && concat.inputs.iter().all(|operand| match operand {
            Operand::Value(value) => is_branch(*value),
            Operand::Lit(_) => false,
        })
}

/// Card 355's final production policy gate. It scans live equations only, after DCE.
pub fn reject_packed_dequant_escapes<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<(), PackedDequantProductionError> {
    let consumers = packed_consumer_map(g);
    let producer: HashMap<ValueId, usize> = g
        .eqns
        .iter()
        .enumerate()
        .map(|(index, eqn)| (eqn.out, index))
        .collect();
    for (equation, eqn) in g.eqns.iter().enumerate() {
        if let OpKind::PackedRowGather { descriptor } = eqn.op {
            let carriers = &eqn.inputs[..eqn.inputs.len().saturating_sub(1)];
            validate_packed_carriers(g, equation, descriptor, carriers, &consumers)?;
        }
        if let OpKind::PackedContraction { descriptor, .. } = eqn.op {
            validate_packed_carriers(g, equation, descriptor, &eqn.inputs[1..], &consumers)?;
            if g.state.iter().any(|&(_, state_out)| state_out == eqn.out) {
                return Err(PackedDequantProductionError::StateOutput {
                    context: PackedDequantErrorContext::new(eqn.out, descriptor),
                });
            }
        }
        let OpKind::PackedDequant { descriptor } = eqn.op else {
            continue;
        };
        validate_packed_carriers(g, equation, descriptor, &eqn.inputs, &consumers)?;
        if g.output == eqn.out {
            return Err(PackedDequantProductionError::GraphOutput {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
        if g.validation_outputs()
            .iter()
            .any(|validation| validation.value == eqn.out)
        {
            return Err(PackedDequantProductionError::ValidationOutput {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
        if g.state.iter().any(|&(_, state_out)| state_out == eqn.out) {
            return Err(PackedDequantProductionError::StateOutput {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
        if consumers[eqn.out].len() != 1 {
            return Err(PackedDequantProductionError::ConsumerCount {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
                internal: eqn.out,
                consumers: consumers[eqn.out].len(),
            });
        }
        if is_canonical_indexed_branch(g, equation, &consumers, &producer) {
            continue;
        }
        let next = &g.eqns[consumers[eqn.out][0]];
        match &next.op {
            OpKind::Transpose { perm } if perm.as_slice() != [1, 0] => {
                return Err(PackedDequantProductionError::TransposePermutation {
                    context: PackedDequantErrorContext::new(eqn.out, descriptor),
                    perm: perm.clone(),
                });
            }
            OpKind::Transpose { .. } => {
                if g.state.iter().any(|&(_, state_out)| state_out == next.out) {
                    return Err(PackedDequantProductionError::StateOutput {
                        context: PackedDequantErrorContext::new(eqn.out, descriptor),
                    });
                }
                if consumers[next.out].len() != 1 {
                    return Err(PackedDequantProductionError::ConsumerCount {
                        context: PackedDequantErrorContext::new(eqn.out, descriptor),
                        internal: next.out,
                        consumers: consumers[next.out].len(),
                    });
                }
            }
            OpKind::Reshape { .. }
            | OpKind::Slice { .. }
            | OpKind::Broadcast { .. }
            | OpKind::Concat { .. } => {
                return Err(PackedDequantProductionError::Movement {
                    context: PackedDequantErrorContext::new(eqn.out, descriptor),
                    operation: next.op.name(),
                });
            }
            _ => {}
        }
        return Err(PackedDequantProductionError::Unmatched {
            context: PackedDequantErrorContext::new(eqn.out, descriptor),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Builder;
    use poot_graph_ir::types::TensorType;
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;

    fn descriptor() -> PackedWeight {
        PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 64]).unwrap()
    }

    /// The canonical `y = x @ W^T` graph `ops::packed_linear` emits: `[0] PackedDequant, [1]
    /// Transpose([1,0]), [2] MatMul(activation, transposed)`.
    fn canonical_graph() -> Graph {
        let b = Builder::new();
        let activation = b.constant("activation", TensorType::f32(vec![2, 64]));
        let out =
            poot_graph_ir::ops::packed_linear(&b, activation, "layer", descriptor(), None, None)
                .unwrap();
        b.finish(out)
    }

    /// R467-009's equivalent spelling: the same linear as `y = (W @ x^T)^T` instead. `canonical`'s
    /// dequant output (already `[out, k]`, eqns[0]) feeds `MatMul(W, x^T)` directly, and the product
    /// transposes back to `[batch, out]` under the graph's own output id.
    fn transposed_spelling_graph() -> Graph {
        let canonical = canonical_graph();
        let dequant = canonical.eqns[0].out;
        let Operand::Value(activation) = canonical.eqns[2].inputs[0] else {
            unreachable!("packed_linear's MatMul reads the activation first")
        };
        let activation_shape = canonical.values[activation].aval.shape.clone();
        let dequant_shape = canonical.values[dequant].aval.shape.clone();

        let mut g = canonical.clone();
        let xt = g.values.len();
        g.values.push(ValueMeta::new(
            TensorType::f32(vec![activation_shape[1], activation_shape[0]]),
            Storage::Device,
            None,
        ));
        let wx = g.values.len();
        g.values.push(ValueMeta::new(
            TensorType::f32(vec![dequant_shape[0], activation_shape[0]]),
            Storage::Device,
            None,
        ));
        let y = g.output;
        g.eqns.truncate(1);
        g.eqns.push(Eqn {
            op: OpKind::Transpose { perm: vec![1, 0] },
            inputs: vec![Operand::Value(activation)],
            out: xt,
            layer: None,
        });
        g.eqns.push(Eqn {
            op: OpKind::MatMul,
            inputs: vec![Operand::Value(dequant), Operand::Value(xt)],
            out: wx,
            layer: None,
        });
        g.eqns.push(Eqn {
            op: OpKind::Transpose { perm: vec![1, 0] },
            inputs: vec![Operand::Value(wx)],
            out: y,
            layer: None,
        });
        g
    }

    fn eqn_kinds(g: &Graph) -> Vec<String> {
        g.eqns.iter().map(|e| e.op.name()).collect()
    }

    /// The graph-level packed-dequant production pipeline (same steps as `transform/tests.rs`'s
    /// in-crate helper of the same name): validate, drop dead oracle-only decoding, recognize, then
    /// reject every remaining escape. Driving this instead of `recognize_packed_contractions` alone
    /// proves the reversed spelling survives the real production entry point, not just an isolated call.
    fn prepare_packed_dequant_production(g: &Graph) -> Result<Graph, PackedDequantProductionError> {
        g.validate()?;
        reject_preexisting_packed_contractions(g)?;
        let graph = crate::passes::cse_dce::dce_with_roots(g, &[]);
        let graph = recognize_packed_contractions(&graph);
        reject_packed_dequant_escapes(&graph)?;
        graph.validate()?;
        Ok(graph)
    }

    /// SC-001: `(W x^T)^T` is recognized as the same packed contraction as the canonical
    /// `x @ W^T` spelling, through the real production pipeline. Mutation: require the outer chain to
    /// start at a `MatMul` root (as the canonical/blocked recognizers do) instead of also trying a
    /// `Transpose` root; this fixture stops recognizing, production rejects the now-unclaimed dequant as
    /// `Unmatched`, and the row goes red.
    #[test]
    fn transposed_spelling_is_recognized_as_one_contraction() {
        let g = transposed_spelling_graph();
        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns.len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                blocks,
            } => {
                assert_eq!(*d, descriptor());
                assert_eq!(*blocks, 1);
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(recognized.aval(recognized.output).shape, vec![2, 3]);
    }

    /// SC-001's near miss: one operand shape changed (the outer transpose's declared output no longer
    /// agrees with what `activation @ W^T` actually produces) must decline, leaving the graph's
    /// equations untouched. Mutation: drop the shared infer-and-compare gate (accept the structural
    /// match alone); this fixture would then also fuse and the row goes red.
    #[test]
    fn transposed_spelling_with_wrong_output_shape_is_declined() {
        let mut g = transposed_spelling_graph();
        let before = eqn_kinds(&g);
        g.values[g.output].aval = TensorType::f32(vec![3, 3]);
        let recognized = recognize_packed_contractions(&g);
        assert!(
            !recognized
                .eqns
                .iter()
                .any(|eqn| matches!(eqn.op, OpKind::PackedContraction { .. })),
            "a shape near-miss must not be recognized as a packed contraction: {recognized:?}"
        );
        assert_eq!(
            eqn_kinds(&recognized),
            before,
            "declining must leave the graph's equations untouched"
        );
    }

    /// R469-007: `validate_packed_carriers` must identify the weight and scale carriers by their
    /// typed operand position - already tied to `descriptor.source_shape(SourceRole::Planar(OperandRole::Codes))` and
    /// `descriptor.source_shape(SourceRole::Planar(OperandRole::Scale))` by `PackedDequant::infer` (and, for `PackedContraction`,
    /// by its own `infer` delegating to `PackedDequant::infer(&ins[1..])`), checked at
    /// `graph.validate()` - not by parsing a `PackedSourceName` suffix off `ValueMeta::name` and
    /// matching linear-id prefixes. Clearing every value's name must not change which contraction is
    /// recognized. Mutation: restore the name/linear-id match; the unnamed fixture goes red with
    /// `CarrierIdentity { roles: (Planar(Codes), Planar(Scale)), name: None }`.
    #[test]
    fn canonical_chain_is_recognized_with_no_packed_source_names() {
        let mut g = canonical_graph();
        assert!(
            g.values.iter().any(|value| value.name.is_some()),
            "fixture has no named value, so clearing names would prove nothing"
        );
        for value in &mut g.values {
            value.name = None;
        }
        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns.len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                blocks,
            } => {
                assert_eq!(*d, descriptor());
                assert_eq!(*blocks, 1);
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(recognized.aval(recognized.output).shape, vec![2, 3]);
    }

    /// Card 541: the recognizer's carrier validation and candidate replacement are role-indexed, not
    /// hardcoded to exactly two `[weight, scale]` operands - a block format's `PackedDequant` has one
    /// carrier (`SourceRole::Blocks`), and this canonical chain (`PackedDequant -> Transpose([1,0]) ->
    /// MatMul`) is recognized into one `PackedContraction` the same way the two-carrier E4M3/E2M1
    /// fixtures above are. Mutation (run 2026-09-29): required `carrier_inputs.len() == 2` in
    /// `validate_packed_carriers` instead of `== roles.len()` -> this test failed at
    /// `prepare_packed_dequant_production(&g).unwrap()` with `CarrierOperands { context: ... Q4_0
    /// [3, 64] }` (the one-carrier `PackedDequant` no longer validates, so `recognize_canonical_packed_chain`
    /// declines and production leaves the dequant unrecognized). Restored.
    #[test]
    fn canonical_chain_is_recognized_for_a_one_source_block_format() {
        let descriptor =
            PackedWeight::try_new(poot_quant::format::WeightFormat::Q4_0, [3, 64]).unwrap();
        assert_eq!(descriptor.sources(), vec![poot_quant::SourceRole::Blocks]);

        let b = Builder::new();
        let activation = b.constant("activation", TensorType::f32(vec![2, 64]));
        let mut plan = b.append_plan(0);
        let blocks = plan
            .input(
                "layer.packed_blocks_source",
                TensorType::new(
                    descriptor
                        .source_shape(poot_quant::SourceRole::Blocks)
                        .to_vec(),
                    DType::I8,
                ),
                Storage::Const,
            )
            .unwrap();
        let weight = plan
            .equation(
                OpKind::PackedDequant { descriptor },
                vec![Operand::Value(blocks.id)],
            )
            .unwrap();
        let weight_t = plan
            .equation(
                OpKind::Transpose { perm: vec![1, 0] },
                vec![Operand::Value(weight.id)],
            )
            .unwrap();
        let output = plan
            .equation(
                OpKind::MatMul,
                vec![Operand::Value(activation.id), Operand::Value(weight_t.id)],
            )
            .unwrap();
        plan.declare_result(output).unwrap();
        let mut prepared = b.preflight_append(plan).unwrap();
        let id = b.commit_append(&mut prepared).unwrap();
        let g = b.finish(poot_graph_ir::Traced { id });

        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns.len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                blocks,
            } => {
                assert_eq!(*d, descriptor);
                assert_eq!(*blocks, 1);
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(
            recognized.eqns[0].inputs.len(),
            2,
            "activation plus the one Blocks carrier"
        );
    }
}
