use super::{
    EditError, Eqn, Graph, HashMap, HashSet, OpKind, Operand, Storage, ValueId, with_eqns,
};
use poot_graph_ir::{ContractionRows, Slot, ValueTable};
/// Names the packed equation a [`PackedDequantProductionError`] is about, for its message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedDequantErrorContext {
    /// The equation's output value.
    pub value: ValueId,
    /// The packed weight format of the equation's descriptor.
    pub format: poot_quant::format::WeightFormat,
    /// The dense `[out, k]` shape the descriptor stands for.
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
            "{} {:?} {:?}",
            self.value, self.format, self.logical_shape
        )
    }
}

/// Why a graph is refused by the final production gate: a packed equation survives that no packed contraction claimed, or its carriers or internal values break the single-consumer, constant-storage and no-escape rules.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PackedDequantProductionError {
    /// A carrier operand is not constant storage.
    #[error(
        "packed dequant {context} carrier {carrier} uses {storage:?} storage; Const or hosted rows is required"
    )]
    CarrierStorage {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The offending carrier value.
        carrier: ValueId,
        /// The carrier's actual storage.
        storage: Storage,
    },
    /// Two source roles resolved to the same graph value.
    #[error(
        "packed dequant {context} {roles:?} carrier {carrier} has invalid identity {name:?}: two distinct source roles resolved to the same graph value"
    )]
    CarrierIdentity {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The value both roles resolved to.
        carrier: ValueId,
        /// The two roles that collided.
        roles: (poot_quant::SourceRole, poot_quant::SourceRole),
        /// The carrier's declared name, if it has one.
        name: Option<String>,
    },
    /// A carrier is also the graph output, a validation output or carried state.
    #[error("packed dequant {context} carrier {carrier} escapes through {destination}")]
    CarrierEscape {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The offending carrier value.
        carrier: ValueId,
        /// Which output the carrier escapes through.
        destination: &'static str,
    },
    /// A carrier has a consumer other than the equation that claims it.
    #[error(
        "packed dequant {context} carrier {carrier} has {consumers} equation consumers; only its claimed candidate is allowed"
    )]
    CarrierConsumerCount {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The offending carrier value.
        carrier: ValueId,
        /// How many equations consume the carrier.
        consumers: usize,
    },
    /// The equation's carrier operands are not exactly the descriptor's source values.
    #[error(
        "packed dequant {context} carriers are not exactly the descriptor's own source graph values"
    )]
    CarrierOperands {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
    },
    /// The dequantized value is also the graph output.
    #[error("packed dequant {context} escapes as the graph output")]
    GraphOutput {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
    },
    /// The dequantized value is also a validation output.
    #[error("packed dequant {context} escapes as a validation output")]
    ValidationOutput {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
    },
    /// The equation's result is also a carried-state output.
    #[error("packed dequant {context} has an internal value that escapes as carried-state output")]
    StateOutput {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
    },
    /// A dequant's result or its transpose has other than one consumer.
    #[error(
        "packed dequant {context} internal {internal} has {consumers} equation consumers; exactly one is required"
    )]
    ConsumerCount {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The value with the wrong consumer count.
        internal: ValueId,
        /// How many equations consume it.
        consumers: usize,
    },
    /// A dequant is followed by a transpose that is not `[1, 0]`.
    #[error("packed dequant {context} has transpose permutation {perm:?}; expected [1, 0]")]
    TransposePermutation {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// The transpose permutation found.
        perm: Vec<usize>,
    },
    /// A dequant feeds a data-movement op the packed chains do not express.
    #[error("packed dequant {context} reaches unsupported movement {operation}")]
    Movement {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
        /// Name of the movement op.
        operation: String,
    },
    /// A dequant survived the contraction pass: no packed contraction claimed it.
    #[error("reachable packed dequant {context} was not claimed by a contraction")]
    Unmatched {
        /// The packed equation the refusal is about.
        context: PackedDequantErrorContext,
    },
}

pub(super) fn packed_consumer_map(g: &Graph) -> ValueTable<Vec<usize>> {
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

pub(super) fn value_input(eqn: &Eqn, position: usize) -> Option<ValueId> {
    match eqn.inputs.get(position) {
        Some(Operand::Value(value)) => Some(*value),
        _ => None,
    }
}

/// The descriptor and carrier operands a packed equation reads, when it is one.
fn claimed_carriers(eqn: &Eqn) -> Option<(poot_quant::PackedWeight, &[Operand])> {
    match &eqn.op {
        OpKind::PackedDequant { descriptor } => Some((*descriptor, &eqn.inputs)),
        OpKind::PackedContraction {
            descriptor, rows, ..
        } => Some((*descriptor, &eqn.inputs[rows.leading_operands()..])),
        OpKind::PackedRowGather { descriptor } => {
            Some((*descriptor, &eqn.inputs[..eqn.inputs.len() - 1]))
        }
        _ => None,
    }
}

/// Validate one `PackedDequant`/`PackedContraction` equation's carrier operands and return them in
/// `descriptor.sources()` order.
///
/// The typed contract is the op itself: both `PackedDequant::infer` and `PackedContraction::infer`
/// (which checks the carriers after its leading operands) already tie operand `i` to
/// `descriptor.sources()[i]`'s `source_shape` (an expert-stacked carrier's `[experts, slab_bytes]`), checked at `graph.validate()` before any recognizer
/// runs - so `carrier_inputs` is already the right length and dtype/shape per role; this function
/// checks the graph-structural facts `infer` cannot see (constant storage, no escape, exactly one
/// consumer, no two roles aliasing the same value). R469-007: identity is checked by graph structure,
/// not by parsing a `PackedSourceName` suffix off `ValueMeta::name` and matching linear-id prefixes -
/// one tracer's naming convention, not the graph's.
fn validate_packed_carriers(
    g: &Graph,
    equation: usize,
    descriptor: poot_quant::PackedWeight,
    carrier_inputs: &[Operand],
    consumers: &ValueTable<Vec<usize>>,
) -> Result<Vec<ValueId>, PackedDequantProductionError> {
    let context = PackedDequantErrorContext::new(g.eqns()[equation].out, descriptor);
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
        // A carrier is a weight constant, or the slot a hosted gather copies its rows into (POOT-1423).
        if g.meta(carrier).storage != Storage::Const
            && g.meta(carrier).storage != Storage::Slot(Slot::Owned(super::HOSTED_ROWS))
        {
            return Err(PackedDequantProductionError::CarrierStorage {
                context,
                carrier,
                storage: g.meta(carrier).storage,
            });
        }
        if g.result() == carrier {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "graph output",
            });
        }
        if g.witnesses().any(|validation| validation.value == carrier) {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "validation output",
            });
        }
        if g.state()
            .any(|(state_in, state_out)| state_in == carrier || state_out == carrier)
        {
            return Err(PackedDequantProductionError::CarrierEscape {
                context,
                carrier,
                destination: "carried state",
            });
        }
        // Only claims over these very carriers may read them: the dequant itself, or one contraction per reader
        // of a dequant that was claimed once per consumer.
        let shared = consumers[carrier].contains(&equation)
            && consumers[carrier].iter().all(|&reader| {
                claimed_carriers(&g.eqns()[reader]) == Some((descriptor, carrier_inputs))
            });
        if !shared {
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
                    name: g.meta(carriers[i]).name().map(std::borrow::Cow::into_owned),
                });
            }
        }
    }
    Ok(carriers)
}

/// The element layout a chain of `Reshape` and `Transpose` leaves over a row-major dequant: one
/// `(size, stride)` per dimension, strides in elements of the dequant's own `[out, k]` buffer. Movement
/// composes onto the view without a copy exactly when [`DenseView::reshaped`] finds strides for the new
/// shape, which is the condition for a contraction to address the dequant through its descriptor.
#[derive(Clone, Debug)]
struct DenseView {
    dims: Vec<(usize, usize)>,
}

impl DenseView {
    /// The view of a row-major `[out, k]` tensor.
    fn row_major(shape: [usize; 2]) -> Self {
        Self {
            dims: vec![(shape[0], shape[1]), (shape[1], 1)],
        }
    }

    /// The view after `Transpose { perm }`, or `None` when `perm` is not a permutation of the dims.
    fn transposed(&self, perm: &[usize]) -> Option<Self> {
        if perm.len() != self.dims.len() {
            return None;
        }
        let dims = perm
            .iter()
            .map(|&axis| self.dims.get(axis).copied())
            .collect::<Option<Vec<_>>>()?;
        Some(Self { dims })
    }

    /// The view after `Reshape { shape }`, or `None` when the new shape does not compose onto these
    /// strides without moving elements (a reshape across a transposed pair, for example).
    fn reshaped(&self, shape: &[usize]) -> Option<Self> {
        if shape.contains(&0) || self.dims.iter().any(|&(size, _)| size == 0) {
            return None;
        }
        let old: Vec<(usize, usize)> = self.dims.iter().copied().filter(|d| d.0 != 1).collect();
        let new_sizes: Vec<usize> = shape.iter().copied().filter(|&size| size != 1).collect();
        let mut new_strides = vec![0; new_sizes.len()];
        let (mut old_at, mut new_at) = (0, 0);
        while old_at < old.len() && new_at < new_sizes.len() {
            let (mut old_end, mut new_end) = (old_at + 1, new_at + 1);
            let (mut old_product, mut new_product) = (old[old_at].0, new_sizes[new_at]);
            while old_product != new_product {
                if old_product < new_product {
                    old_product *= old.get(old_end)?.0;
                    old_end += 1;
                } else {
                    new_product *= new_sizes.get(new_end)?;
                    new_end += 1;
                }
            }
            // The old dims of one group must be contiguous with each other to be re-split or merged.
            if old[old_at..old_end]
                .windows(2)
                .any(|pair| pair[0].1 != pair[1].1 * pair[1].0)
            {
                return None;
            }
            let mut stride = old[old_end - 1].1;
            for at in (new_at..new_end).rev() {
                new_strides[at] = stride;
                stride *= new_sizes[at];
            }
            (old_at, new_at) = (old_end, new_end);
        }
        if old_at != old.len() || new_at != new_sizes.len() {
            return None;
        }
        let mut strides = new_strides.into_iter();
        let dims = shape
            .iter()
            .map(|&size| {
                (
                    size,
                    if size == 1 {
                        1
                    } else {
                        strides.next().unwrap_or(1)
                    },
                )
            })
            .collect();
        Some(Self { dims })
    }

    /// The view with its size-1 dims dropped: their strides address nothing.
    fn significant(&self) -> Vec<(usize, usize)> {
        self.dims.iter().copied().filter(|d| d.0 != 1).collect()
    }

    /// Whether this view addresses the dequant the way `expected` does.
    fn addresses(&self, expected: &[(usize, usize)]) -> bool {
        self.significant() == expected
    }
}

/// The movement between a `PackedDequant` and the value a contraction reads, composed onto one view.
struct WeightChain {
    dequant: usize,
    descriptor: poot_quant::PackedWeight,
    /// The `Reshape` and `Transpose` equations between the dequant and the contraction.
    movement: Vec<usize>,
    view: DenseView,
}

/// Walk back from `weight` through `Reshape` and `Transpose` to a `PackedDequant`, composing the movement onto
/// the dequant's `[out, k]` view. `None` when the producer chain is anything else or the movement does not
/// compose. Consumer counts are not this walk's business: [`claim_packed_contractions`] checks that every
/// reader of the chain is itself part of a claim.
fn weight_chain(
    g: &Graph,
    producer: &HashMap<ValueId, usize>,
    mut weight: ValueId,
) -> Option<WeightChain> {
    let mut movement = Vec::new();
    let (dequant, descriptor) = loop {
        let &index = producer.get(&weight)?;
        match &g.eqns()[index].op {
            OpKind::PackedDequant { descriptor } => break (index, *descriptor),
            OpKind::Reshape { .. } | OpKind::Transpose { .. } => {
                movement.push(index);
                weight = value_input(&g.eqns()[index], 0)?;
            }
            _ => return None,
        }
    };
    movement.reverse();
    let mut view = DenseView::row_major(descriptor.shape());
    for &index in &movement {
        view = match &g.eqns()[index].op {
            OpKind::Reshape { shape } => view.reshaped(shape)?,
            OpKind::Transpose { perm } => view.transposed(perm)?,
            _ => unreachable!("only Reshape and Transpose are pushed"),
        };
    }
    Some(WeightChain {
        dequant,
        descriptor,
        movement,
        view,
    })
}

impl WeightChain {
    /// The row-block count of the contraction whose right operand this view is, when the view addresses the
    /// dequant as `[k, out]` (`blocks == 1`) or as `[blocks, k, out / blocks]` (`blocks > 1`, the block-diagonal
    /// view): both are the transposed product `x @ W^T` over the stored `[out, k]` rows.
    fn right_operand_blocks(&self) -> Option<usize> {
        let [out, k] = self.descriptor.shape();
        let first = self.view.dims.first().map(|&(size, _)| size);
        [Some(1), first]
            .into_iter()
            .flatten()
            .filter(|&blocks| blocks != 0 && out % blocks == 0)
            .find(|&blocks| {
                let block_out = out / blocks;
                let mut expected = Vec::new();
                if blocks != 1 {
                    expected.push((blocks, block_out * k));
                }
                expected.extend([(k, 1), (block_out, k)]);
                expected.retain(|dim| dim.0 != 1);
                self.view.addresses(&expected)
            })
    }

    /// Whether this view is the dequant's own `[out, k]` rows: the left operand of the transposed product
    /// `(W @ x^T)^T`.
    fn is_stored_rows(&self) -> bool {
        let [out, k] = self.descriptor.shape();
        let mut expected = vec![(out, k), (k, 1)];
        expected.retain(|dim| dim.0 != 1);
        self.view.addresses(&expected)
    }
}

/// One contraction claimed over a dequant's carriers: the equation that becomes the `PackedContraction`
/// and the equations the claim consumes, not counting the dequant (several claims can share one).
struct PackedClaim {
    /// The equation whose output the `PackedContraction` now produces.
    root: usize,
    /// The movement (and, for the transposed product, the inner matmul and activation transpose) it consumes.
    consumed: Vec<usize>,
    dequant: usize,
    activation: ValueId,
    blocks: usize,
    sources: Vec<ValueId>,
    descriptor: poot_quant::PackedWeight,
}

/// Claim the contraction `root` reads through `chain` when the typed contraction reproduces `root`'s output.
fn claim_through(
    g: &Graph,
    consumers: &ValueTable<Vec<usize>>,
    chain: &WeightChain,
    activation: ValueId,
    blocks: usize,
    root: usize,
    mut consumed: Vec<usize>,
) -> Option<PackedClaim> {
    let sources = validate_packed_carriers(
        g,
        chain.dequant,
        chain.descriptor,
        &g.eqns()[chain.dequant].inputs,
        consumers,
    )
    .ok()?;
    let op = OpKind::PackedContraction {
        descriptor: chain.descriptor,
        rows: ContractionRows::Blocks(blocks),
        precision: poot_graph_ir::OperandPrecision::Exact,
    };
    let inputs: Vec<poot_graph_ir::types::TensorType> = std::iter::once(g.aval(activation).clone())
        .chain(sources.iter().map(|&source| g.aval(source).clone()))
        .collect();
    if op.infer(&inputs).ok().as_ref() != Some(g.aval(g.eqns()[root].out)) {
        return None;
    }
    consumed.extend(&chain.movement);
    Some(PackedClaim {
        root,
        consumed,
        dequant: chain.dequant,
        activation,
        blocks,
        sources,
        descriptor: chain.descriptor,
    })
}

/// Every contraction that reads a `PackedDequant` through movement that composes onto its weight view: `x @ W^T`
/// (plain or block-diagonal, through any `Reshape`/`Transpose` chain that composes to the `[k, out]` or
/// `[blocks, k, out / blocks]` view) and the transposed product `(W @ x^T)^T`.
fn find_packed_claims(
    g: &Graph,
    producer: &HashMap<ValueId, usize>,
    consumers: &ValueTable<Vec<usize>>,
) -> Vec<PackedClaim> {
    let mut claims: Vec<PackedClaim> = Vec::new();
    for (index, eqn) in g.eqns().iter().enumerate() {
        if !eqn.op.is_plain_matmul() || eqn.inputs.len() != 2 {
            continue;
        }
        let (Some(activation), Some(weight)) = (value_input(eqn, 0), value_input(eqn, 1)) else {
            continue;
        };
        let Some(chain) = weight_chain(g, producer, weight) else {
            continue;
        };
        let Some(blocks) = chain.right_operand_blocks() else {
            continue;
        };
        claims.extend(claim_through(
            g,
            consumers,
            &chain,
            activation,
            blocks,
            index,
            Vec::new(),
        ));
    }
    for (index, outer) in g.eqns().iter().enumerate() {
        let OpKind::Transpose { perm } = &outer.op else {
            continue;
        };
        if perm.as_slice() != [1, 0] {
            continue;
        }
        let Some(product) = value_input(outer, 0) else {
            continue;
        };
        let Some(&inner_index) = producer.get(&product) else {
            continue;
        };
        let inner = &g.eqns()[inner_index];
        if !inner.op.is_plain_matmul()
            || inner.inputs.len() != 2
            || claims.iter().any(|claim| claim.root == inner_index)
        {
            continue;
        }
        let (Some(weight), Some(activation_t)) = (value_input(inner, 0), value_input(inner, 1))
        else {
            continue;
        };
        let Some(&at_index) = producer.get(&activation_t) else {
            continue;
        };
        let at = &g.eqns()[at_index];
        let (OpKind::Transpose { perm: at_perm }, Some(activation)) = (&at.op, value_input(at, 0))
        else {
            continue;
        };
        if at_perm.as_slice() != [1, 0] {
            continue;
        }
        let Some(chain) = weight_chain(g, producer, weight) else {
            continue;
        };
        if !chain.is_stored_rows() {
            continue;
        }
        claims.extend(claim_through(
            g,
            consumers,
            &chain,
            activation,
            1,
            index,
            vec![inner_index, at_index],
        ));
    }
    claims
}

/// Replace every packed dequant whose readers are all contractions that compose onto its weight view by one
/// `PackedContraction` per reader, over the same carriers. This is the planner-facing claim: the weight is
/// never widened and no standalone dequant is lowered (POOT-ADR-0103). The rule is that the movement chain
/// between the dequant and the contraction composes to an `[out, k]` (or blocked) view the descriptor
/// addresses. A dequant with any other reader, or whose chain or product escapes the graph, is left alone for
/// [`reject_packed_dequant_escapes`] to refuse. This pass runs before transpose/view analysis and leaves a
/// following ordinary bias Add untouched.
///
/// Generic over the validation channel: a validation value is a liveness root, so a member that a witness
/// observes is a graph escape and its claim is left unrecognized.
pub fn recognize_packed_contractions(g: &Graph) -> Result<Graph, EditError> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns().iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let claims = find_packed_claims(g, &producer, &consumers);

    let mut dequants: Vec<usize> = claims.iter().map(|claim| claim.dequant).collect();
    dequants.sort_unstable();
    dequants.dedup();
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();
    for dequant in dequants {
        let group: Vec<&PackedClaim> = claims.iter().filter(|c| c.dequant == dequant).collect();
        // Every equation of the group, and every one of them whose output the group reads.
        let inside: HashSet<usize> = group
            .iter()
            .flat_map(|claim| claim.consumed.iter().copied().chain([claim.root]))
            .chain([dequant])
            .collect();
        let consumed: HashSet<usize> = inside
            .iter()
            .copied()
            .filter(|index| !group.iter().any(|claim| claim.root == *index))
            .collect();
        let closed = consumed.iter().all(|&index| {
            let out = g.eqns()[index].out;
            !is_graph_escape(g, out) && consumers[out].iter().all(|reader| inside.contains(reader))
        });
        if !closed {
            continue;
        }
        removed.extend(consumed);
        for claim in group {
            let root = &g.eqns()[claim.root];
            replacements.insert(
                claim.root,
                Eqn {
                    op: OpKind::PackedContraction {
                        descriptor: claim.descriptor,
                        rows: ContractionRows::Blocks(claim.blocks),
                        precision: poot_graph_ir::OperandPrecision::Exact,
                    },
                    inputs: std::iter::once(Operand::Value(claim.activation))
                        .chain(claim.sources.iter().map(|&source| Operand::Value(source)))
                        .collect(),
                    out: root.out,
                    layer: root.layer,
                },
            );
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
    with_eqns(g, eqns)
}

/// Card 545a: claim `Gather { axis: 0 }(PackedDequant(carriers..), ids)` - a quantized
/// token-embedding lookup - as one `PackedRowGather(carriers.., ids)`, so only the gathered rows are
/// decoded and no `[out, K]` table is materialized. The dequant must have the gather as its only
/// consumer and not escape; its carriers must pass the same structural check as every claim.
pub fn recognize_packed_row_gathers(g: &Graph) -> Result<Graph, EditError> {
    let consumers = packed_consumer_map(g);
    let mut producer = HashMap::new();
    for (index, eqn) in g.eqns().iter().enumerate() {
        producer.insert(eqn.out, index);
    }
    let mut replacements: HashMap<usize, Eqn> = HashMap::new();
    let mut removed = HashSet::new();
    for (gather_index, gather) in g.eqns().iter().enumerate() {
        if !matches!(gather.op, OpKind::Gather { axis: 0 }) {
            continue;
        }
        let (Some(table), Some(ids)) = (value_input(gather, 0), value_input(gather, 1)) else {
            continue;
        };
        let Some(&dequant_index) = producer.get(&table) else {
            continue;
        };
        let OpKind::PackedDequant { descriptor } = g.eqns()[dequant_index].op else {
            continue;
        };
        if consumers[table].len() != 1 || is_graph_escape(g, table) {
            continue;
        }
        let Ok(sources) = validate_packed_carriers(
            g,
            dequant_index,
            descriptor,
            &g.eqns()[dequant_index].inputs,
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
    with_eqns(g, eqns)
}

/// Card 355's final production policy gate. It scans live equations only, after DCE.
pub(crate) fn reject_packed_dequant_escapes(g: &Graph) -> Result<(), PackedDequantProductionError> {
    let consumers = packed_consumer_map(g);
    for (equation, eqn) in g.eqns().iter().enumerate() {
        if let OpKind::PackedRowGather { descriptor } = eqn.op {
            let carriers = &eqn.inputs[..eqn.inputs.len().saturating_sub(1)];
            validate_packed_carriers(g, equation, descriptor, carriers, &consumers)?;
        }
        if let OpKind::PackedContraction {
            descriptor, rows, ..
        } = eqn.op
        {
            let carriers = &eqn.inputs[rows.leading_operands()..];
            validate_packed_carriers(g, equation, descriptor, carriers, &consumers)?;
            if g.state().any(|(_, state_out)| state_out == eqn.out) {
                return Err(PackedDequantProductionError::StateOutput {
                    context: PackedDequantErrorContext::new(eqn.out, descriptor),
                });
            }
        }
        let OpKind::PackedDequant { descriptor } = eqn.op else {
            continue;
        };
        validate_packed_carriers(g, equation, descriptor, &eqn.inputs, &consumers)?;
        if g.result() == eqn.out {
            return Err(PackedDequantProductionError::GraphOutput {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
        if g.witnesses().any(|validation| validation.value == eqn.out) {
            return Err(PackedDequantProductionError::ValidationOutput {
                context: PackedDequantErrorContext::new(eqn.out, descriptor),
            });
        }
        if g.state().any(|(_, state_out)| state_out == eqn.out) {
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
        let next = &g.eqns()[consumers[eqn.out][0]];
        match &next.op {
            OpKind::Transpose { perm } if perm.as_slice() != [1, 0] => {
                return Err(PackedDequantProductionError::TransposePermutation {
                    context: PackedDequantErrorContext::new(eqn.out, descriptor),
                    perm: perm.clone(),
                });
            }
            OpKind::Transpose { .. } => {
                if g.state().any(|(_, state_out)| state_out == next.out) {
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
    use crate::passes::DType;
    use crate::passes::ValueMeta;
    use poot_graph_ir::Builder;
    use poot_graph_ir::types::TensorType;
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;

    fn descriptor() -> PackedWeight {
        PackedWeight::try_new(WeightFormat::E2m1Row32, [3, 64]).unwrap()
    }

    /// The canonical `y = x @ W^T` graph `packed_graphs::packed_linear` emits: `[0] PackedDequant, [1]
    /// Transpose([1,0]), [2] MatMul(activation, transposed)`.
    fn canonical_graph() -> Graph {
        let b = Builder::for_tests();
        let activation = b
            .constant("activation", TensorType::f32(vec![2, 64]))
            .unwrap();
        let out = poot_graph_ir::test_support::packed_graphs::packed_linear(
            &b,
            activation,
            "layer",
            descriptor(),
            None,
            None,
        )
        .unwrap();
        b.finish(out).unwrap()
    }

    /// R467-009's equivalent spelling: the same linear as `y = (W @ x^T)^T` instead. `canonical`'s
    /// dequant output (already `[out, k]`, eqns[0]) feeds `MatMul(W, x^T)` directly, and the product
    /// transposes back to `[batch, out]` under the graph's own output id.
    fn transposed_spelling_graph() -> Graph {
        let canonical = canonical_graph();
        let dequant = canonical.eqns()[0].out;
        let Operand::Value(activation) = canonical.eqns()[2].inputs[0] else {
            unreachable!("packed_linear's MatMul reads the activation first")
        };
        let activation_shape = canonical.values()[activation].aval.shape.clone();
        let dequant_shape = canonical.values()[dequant].aval.shape.clone();

        let y = canonical.result();
        poot_graph_ir::test_support::graph_fixtures::derive(&canonical, |g| {
            let xt = g.values_mut().push(ValueMeta::new(
                TensorType::f32(vec![activation_shape[1], activation_shape[0]]),
                Storage::Device,
                None,
            ));
            let wx = g.values_mut().push(ValueMeta::new(
                TensorType::f32(vec![dequant_shape[0], activation_shape[0]]),
                Storage::Device,
                None,
            ));
            g.eqns_mut().truncate(1);
            g.eqns_mut().push(Eqn {
                op: OpKind::Transpose { perm: vec![1, 0] },
                inputs: vec![Operand::Value(activation)],
                out: xt,
                layer: None,
            });
            g.eqns_mut().push(Eqn {
                op: OpKind::matmul(),
                inputs: vec![Operand::Value(dequant), Operand::Value(xt)],
                out: wx,
                layer: None,
            });
            g.eqns_mut().push(Eqn {
                op: OpKind::Transpose { perm: vec![1, 0] },
                inputs: vec![Operand::Value(wx)],
                out: y,
                layer: None,
            });
        })
        .unwrap()
    }

    /// The graph-level packed-dequant production pipeline (same steps as `transform/tests.rs`'s
    /// in-crate helper of the same name): drop dead oracle-only decoding, recognize, then
    /// reject every remaining escape. Driving this instead of `recognize_packed_contractions` alone
    /// proves the reversed spelling survives the real production entry point, not just an isolated call.
    fn prepare_packed_dequant_production(g: &Graph) -> Result<Graph, PackedDequantProductionError> {
        let graph = crate::passes::cse_dce::dce_with_roots(g, &[]).unwrap();
        let graph = recognize_packed_contractions(&graph).unwrap();
        reject_packed_dequant_escapes(&graph)?;
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
            recognized.eqns().len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns()[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                rows,
                ..
            } => {
                assert_eq!(*d, descriptor());
                assert_eq!(*rows, ContractionRows::Blocks(1));
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(recognized.aval(recognized.result()).shape, vec![2, 3]);
    }

    /// R469-007: `validate_packed_carriers` must identify the weight and scale carriers by their
    /// typed operand position - already tied to `descriptor.source_shape(SourceRole::Planar(PlanarRole::Codes))` and
    /// `descriptor.source_shape(SourceRole::Planar(PlanarRole::Scale))` by `PackedDequant::infer` (and, for `PackedContraction`,
    /// by its own `infer` delegating to `PackedDequant::infer(&ins[1..])`), checked at
    /// `graph.validate()` - not by parsing a `PackedSourceName` suffix off `ValueMeta::name` and
    /// matching linear-id prefixes. Clearing every value's name must not change which contraction is
    /// recognized. Mutation: restore the name/linear-id match; the unnamed fixture goes red with
    /// `CarrierIdentity { roles: (Planar(Codes), Planar(Scale)), name: None }`.
    #[test]
    fn canonical_chain_is_recognized_with_no_packed_source_names() {
        let named = canonical_graph();
        assert!(
            named.values().iter().any(|value| value.name().is_some()),
            "fixture has no named value, so clearing names would prove nothing"
        );
        let g = poot_graph_ir::test_support::graph_fixtures::derive(&named, |draft| {
            for value in draft.values_mut().iter_mut() {
                value
                    .clear_name()
                    .expect("the fixture's values are plain named ones");
            }
        })
        .unwrap();
        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns().len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns()[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                rows,
                ..
            } => {
                assert_eq!(*d, descriptor());
                assert_eq!(*rows, ContractionRows::Blocks(1));
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(recognized.aval(recognized.result()).shape, vec![2, 3]);
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

        let b = Builder::for_tests();
        let activation = b
            .constant("activation", TensorType::f32(vec![2, 64]))
            .unwrap();
        let mut plan = b.append_plan(0).unwrap();
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
                vec![Operand::Value(blocks.id())],
            )
            .unwrap();
        let weight_t = plan
            .equation(
                OpKind::Transpose { perm: vec![1, 0] },
                vec![Operand::Value(weight.id())],
            )
            .unwrap();
        let output = plan
            .equation(
                OpKind::matmul(),
                vec![
                    Operand::Value(activation.id()),
                    Operand::Value(weight_t.id()),
                ],
            )
            .unwrap();
        plan.declare_result(output).unwrap();
        let mut prepared = b.preflight_append(plan).unwrap();
        let committed = b.commit_append(&mut prepared).unwrap();
        let g = b.finish(committed).unwrap();

        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns().len(),
            1,
            "expected the whole chain to collapse to one eqn: {recognized:?}"
        );
        match &recognized.eqns()[0].op {
            OpKind::PackedContraction {
                descriptor: d,
                rows,
                ..
            } => {
                assert_eq!(*d, descriptor);
                assert_eq!(*rows, ContractionRows::Blocks(1));
            }
            other => panic!("expected PackedContraction, got {other:?}"),
        }
        assert_eq!(
            recognized.eqns()[0].inputs.len(),
            2,
            "activation plus the one Blocks carrier"
        );
    }

    /// SC-001 (first half): a packed weight with a `Reshape` between the dequant and the matmul. `[out, k]` is
    /// reshaped to `[1, out, k]` and transposed `[0, 2, 1]`: movement that composes onto the weight view although
    /// it is none of the spellings `packed_linear` and `packed_block_diagonal_linear` emit. Mutation: make
    /// `DenseView::reshaped` refuse every reshape; the dequant is left unclaimed and production rejects it as
    /// `Unmatched`.
    #[test]
    fn a_reshape_between_dequant_and_matmul_composes_onto_the_weight_view() {
        let descriptor = PackedWeight::try_new(WeightFormat::Q4_0, [3, 64]).unwrap();
        let b = Builder::for_tests();
        let activation = b
            .constant("activation", TensorType::f32(vec![1, 2, 64]))
            .unwrap();
        let mut plan = b.append_plan(0).unwrap();
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
        let dequant = plan
            .equation(
                OpKind::PackedDequant { descriptor },
                vec![Operand::Value(blocks.id())],
            )
            .unwrap();
        let lifted = plan
            .equation(
                OpKind::Reshape {
                    shape: vec![1, 3, 64],
                },
                vec![Operand::Value(dequant.id())],
            )
            .unwrap();
        let weight = plan
            .equation(
                OpKind::Transpose {
                    perm: vec![0, 2, 1],
                },
                vec![Operand::Value(lifted.id())],
            )
            .unwrap();
        let output = plan
            .equation(
                OpKind::matmul(),
                vec![Operand::Value(activation.id()), Operand::Value(weight.id())],
            )
            .unwrap();
        plan.declare_result(output).unwrap();
        let mut prepared = b.preflight_append(plan).unwrap();
        let committed = b.commit_append(&mut prepared).unwrap();
        let g = b.finish(committed).unwrap();

        let recognized = prepare_packed_dequant_production(&g).unwrap();
        assert_eq!(
            recognized.eqns().len(),
            1,
            "no standalone dequant may survive: {recognized:?}"
        );
        assert!(matches!(
            recognized.eqns()[0].op,
            OpKind::PackedContraction {
                rows: ContractionRows::Blocks(1),
                ..
            }
        ));
        assert_eq!(recognized.aval(recognized.result()).shape, vec![1, 2, 3]);
    }

    /// Movement that does not compose onto the stored rows is not claimed: reshaping the transposed `[k, out]`
    /// view to `[out, k]` would have to move elements. Mutation: skip the contiguity check in
    /// `DenseView::reshaped`; the first assertion goes red.
    #[test]
    fn a_reshape_across_a_transposed_pair_does_not_compose() {
        let stored = DenseView::row_major([3, 64]);
        let transposed = stored.transposed(&[1, 0]).unwrap();
        assert!(transposed.reshaped(&[3, 64]).is_none());
        assert!(transposed.reshaped(&[64, 3]).is_some());
        assert!(stored.reshaped(&[3, 8, 8]).is_some());
    }
}
