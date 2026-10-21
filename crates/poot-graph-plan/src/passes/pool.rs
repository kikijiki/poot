//! The expert-pool transform (POOT-746): a routed MoE layer whose expert tables hold `E` experts is
//! rewritten to read `P < E` resident rows, with the router's global expert ids translated to pool
//! rows by an in-graph `Gather` through a host-written map.
//!
//! One [`PoolSite`] is one selection of expert ids and the tables its indexed contractions read (the
//! up and down projections of one layer share a site). For each site the pass
//!
//! - declares the site's `EXPERT_POOL_MAP` `[E]` (the pool row holding global expert `e`, or `-1`),
//! - inserts `Gather(map, ids)` just before the first contraction that reads the ids and points every
//!   contraction of the site at the translated ids,
//! - rebinds each table const (a dense `[E, ..]` stack, or each source role of a packed stack) as a
//!   resident `EXPERT_POOL` `[P, ..]` of the same name, and a packed contraction's `Indexed`
//!   expert count becomes `P`.
//!
//! The translating `Gather` is where the program is cut: everything before it computes the ids, the
//! host sync reads them and makes those experts resident, and the rest of the layer reads the pool
//! (`staged::pool` owns the cut and the sync point). The tables stay in the stored lane: a refill
//! writes one expert's bytes as stored, never a re-encode (ADR-0103).
//!
//! A selection that cannot be served is a typed [`PoolError`]: a site's step demand above the pool
//! ([`PoolError::CapacityBelowStepDemand`]), a pool that is not smaller than the experts
//! ([`PoolError::CapacityNotBelowExperts`]), or a table the pass cannot page.

use std::collections::{BTreeMap, HashMap, HashSet};

use poot_graph_ir::analysis::{NumericsDelta, NumericsProperty};
use poot_graph_ir::graph::ValueMeta;
use poot_graph_ir::graph::{
    BoundaryId, Eqn, Graph, GraphEdit, Operand, Slot, SlotKey, Storage, ValidationId, ValueId,
};
use poot_graph_ir::op::{ContractionRows, OpKind};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{
    GuardValues, IndexGuardEmitError, IndexGuardError, OwnedSlot, ShapeError, emit_index_guard,
};
use poot_tensor::DType;

use super::GraphBudget;
use super::routed_experts::routed_expert_sites;
use crate::ExpansionError;

const STAGE: &str = "pool_experts";

/// A site's expert-pool assignment table `[E]` (I32, the index dtype `Gather` reads): entry `e` is the pool slot
/// currently holding global expert `e`, or `-1` when not resident in that layer's pool this step. An in-graph
/// `Gather` translates the router's global expert ids into pool-slot ids with it, so the unmodified
/// `IndexedMatMul` op addresses a smaller `[P, ..]` pool weight instead of the full `[E, ..]` stack.
///
/// One graph contains many (one per MoE layer, each with different content in the same step), told apart by the
/// tag `site.N`. A binder resolves them per key, never per role. A refill moves experts between slots, so it is a
/// slot rebound every step, never a `Const` (the const cache keys by name and element count and would serve a
/// stale table).
pub const EXPERT_POOL_MAP: OwnedSlot = OwnedSlot::i32_tensor("expertpoolmap");

/// An expert table paged through a pool: one `[P, ..]` slot per pooled routed table (a dense `[E, ..]` stack or
/// one packed source role of it), declared by the pool pass in place of the table's `Const` and tagged with that
/// const's name. Unlike every other slot it is not written from the step's inputs: it is resident for the
/// executable's life and the expert-residency manager refills one row at a time at a sync point. Several exist in
/// one graph, one per table, so the key is its tag, never the role. Its buffer holds the table's stored bytes.
pub const EXPERT_POOL: OwnedSlot = OwnedSlot::new("expertpool", true, |aval, _| {
    aval.shape.is_empty().then_some("a tensor with a pool axis")
});

/// One pooled table: the const it replaces (its name is how the weight map finds the stored experts), the
/// resident slot that stands in for it, and that slot's type, `[P, ..]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PooledWeight {
    /// The replaced const's name, the key the weight map finds the stored experts by.
    pub name: String,
    /// The resident slot that stands in for the const.
    pub slot: SlotKey,
    /// The slot's tensor type, `[P, ..]`.
    pub pool: TensorType,
}

/// One routed selection and the tables it indexes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolSite {
    /// The decoder layer the site's contractions were traced in, when the tracer tagged one.
    pub layer: Option<usize>,
    /// Experts in each table, `E`.
    pub experts: usize,
    /// Pool rows, `P`.
    pub slots: usize,
    /// The router's expert ids (I32, one per routed row): the value the host reads at the sync.
    pub ids: BoundaryId,
    /// The `[E]` map slot: pool row of each global expert, `-1` when not resident.
    pub map: SlotKey,
    /// The tables, in the order the site's contractions read them.
    pub weights: Vec<PooledWeight>,
}

/// What the pool pass did to a graph: its sites in cut order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolSpec {
    /// The sites, in the order the pass cut them.
    pub sites: Vec<PoolSite>,
}

/// A graph with its expert tables pooled, and where each site cuts it.
pub(crate) struct Pooled {
    pub graph: Graph,
    /// What pooling produced, read off the graph before and after: the slot translation and its index
    /// guard move and compare integer words, and the guard's witness sums their flags in F32, so a pooled
    /// program computes its float values as the resident one and adds one unfused reduction.
    pub numerics: NumericsProperty,
    pub spec: PoolSpec,
    /// `cuts[i]` is the equation of the translating `Gather` of `spec.sites[i]`: the first equation
    /// of the stage that follows the site's sync.
    pub cuts: Vec<usize>,
}

/// Why a graph could not be pooled.
#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    /// One step can route more rows to distinct experts than the pool holds rows: the demand is the
    /// site's id count, `rows * tokens * k`.
    #[error(
        "the routed selection of layer {layer:?} can need {demand} experts in one step, over the pool's {slots} rows"
    )]
    CapacityBelowStepDemand {
        /// The decoder layer of the site, when the tracer tagged one.
        layer: Option<usize>,
        /// Distinct experts one step can need: the site's id count.
        demand: usize,
        /// Rows the pool holds.
        slots: usize,
    },
    /// A pool as large as the expert table pages nothing: serve the table resident.
    #[error("a pool of {slots} rows is not smaller than the {experts} experts it would page")]
    CapacityNotBelowExperts {
        /// Rows the pool holds.
        slots: usize,
        /// Experts in each table.
        experts: usize,
    },
    /// The graph has no routed selection, so there is nothing to page.
    #[error("the graph has no routed expert selection to pool")]
    NoRoutedExperts,
    /// An indexed contraction's table is not a named const reached by leading-axis-preserving views.
    #[error(
        "equation {eqn}: the expert table behind {value} is not a named const the pool can page"
    )]
    TableNotPageable {
        /// Index of the contraction's equation.
        eqn: usize,
        /// The table operand that cannot be paged.
        value: ValueId,
    },
    /// A table is read by something other than its site's contractions, so shrinking it would change
    /// that reader.
    #[error("expert table {name} is also read outside its routed contractions")]
    TableReadOutsideSite {
        /// Name of the table.
        name: String,
    },
    /// One table is indexed by two different selections.
    #[error("expert table {name} is indexed by more than one routed selection")]
    TableSharedAcrossSites {
        /// Name of the table.
        name: String,
    },
    /// The tables of one selection do not agree on how many experts they hold.
    #[error("the tables of one routed selection hold {first} and {second} experts")]
    ExpertCountMismatch {
        /// Expert count of the site's first table.
        first: usize,
        /// Expert count of the table that disagrees.
        second: usize,
    },
    /// The index guard of a site's translated rows cannot be emitted: the pool's row count is past the
    /// `i32` words of a row id, or the translated rows are not an I32 selector.
    #[error(transparent)]
    RowGuard(IndexGuardError),
    /// An equation of a site's row guard does not type-check over the translated rows.
    #[error("an equation of a site's row guard does not type-check: {0}")]
    RowGuardShape(ShapeError),
    /// Growing the graph for the pooled tables and row guard would pass the compile's limits.
    #[error(transparent)]
    Expansion(Box<ExpansionError>),
    /// The graph editor refused a rewrite the pass issued.
    #[error(transparent)]
    Edit(Box<poot_graph_ir::EditError>),
}

impl From<poot_graph_ir::EditError> for PoolError {
    fn from(error: poot_graph_ir::EditError) -> Self {
        Self::Edit(Box::new(error))
    }
}

impl From<ExpansionError> for PoolError {
    fn from(error: ExpansionError) -> Self {
        Self::Expansion(Box::new(error))
    }
}

/// One selection found in the unpooled graph.
struct Found {
    ids: ValueId,
    /// The most experts one step can select: the router's rows (every row of every token) times the
    /// `k` each row picks.
    demand: usize,
    /// The contractions reading `ids`, in equation order.
    eqns: Vec<usize>,
    tables: Vec<Table>,
    experts: usize,
    /// The view equations (transposes) between each table const and its contraction.
    views: Vec<usize>,
}

struct Table {
    value: ValueId,
    name: String,
}

/// The most experts one step of `g` can select at any one routed selection (`rows * tokens * k`), `0`
/// when `g` has none. A driver divides its pool by this per-token demand to bound a step's tokens.
pub fn routed_step_demand(g: &Graph) -> Result<usize, PoolError> {
    Ok(find(g)?.iter().map(|site| site.demand).max().unwrap_or(0))
}

/// Pool every routed selection of `g` to `slots` rows.
pub(crate) fn pool_experts(
    g: &Graph,
    slots: usize,
    budget: &GraphBudget,
) -> Result<Pooled, PoolError> {
    let mut found = find(g)?;
    if found.is_empty() {
        return Err(PoolError::NoRoutedExperts);
    }
    for site in &found {
        if slots >= site.experts {
            return Err(PoolError::CapacityNotBelowExperts {
                slots,
                experts: site.experts,
            });
        }
        if site.demand > slots {
            return Err(PoolError::CapacityBelowStepDemand {
                layer: g.eqns()[site.eqns[0]].layer.map(|layer| layer.0),
                demand: site.demand,
                slots,
            });
        }
    }
    found.sort_by_key(|site| site.eqns[0]);
    budget.reserve(STAGE, g, found.len(), found.len())?;

    let mut pooled = g.edit();
    let mut sites = Vec::with_capacity(found.len());
    // `(original equation the gather goes before, the gather and the guard on its output)`, in ascending
    // position.
    let mut gathers: Vec<(usize, Vec<Eqn>)> = Vec::with_capacity(found.len());
    // The row-guard witnesses this pass declares, one per site.
    let mut added: Vec<ValidationId> = Vec::with_capacity(found.len());
    for (ordinal, site) in found.iter().enumerate() {
        let first = site.eqns[0];
        let layer = g.eqns()[first].layer;
        let tag = format!("site.{ordinal}");
        let map = pooled.declare_slot(
            Slot::Owned(EXPERT_POOL_MAP),
            &tag,
            TensorType::new(vec![site.experts], DType::I32),
        )?;
        let ids_aval = g.aval(site.ids).clone();
        let raw = pooled
            .values_mut()
            .push(ValueMeta::new(ids_aval, Storage::Device, None));
        let mut chain = vec![Eqn {
            op: OpKind::Gather { axis: 0 },
            inputs: vec![Operand::Value(map), Operand::Value(site.ids)],
            out: raw,
            layer,
        }];
        // The map holds `-1` for an expert that is not resident. The host makes every selected expert
        // resident before this stage runs, so such a lane is a sync that did not hold: the canonical index
        // guard (`Builder::guard_index_bounds`) replaces it with row 0 and its witness counts it, declared
        // as a validation output, so the step publishes a failure instead of a wrong expert's product.
        let guard = guard_rows(&mut pooled, &mut chain, raw, slots, layer)?;
        let id = ValidationId(
            pooled
                .witnesses()
                .map(|witness| witness.id.0 + 1)
                .max()
                .unwrap_or(0),
        );
        pooled.push_witness(id, format!("pool.{tag}.resident"), guard.witness);
        added.push(id);
        gathers.push((first, chain));
        let translated = guard.guarded;
        for &e in &site.eqns {
            let eqn = &mut pooled.eqns_mut()[e];
            let position = match eqn.op {
                OpKind::IndexedMatMul => 2,
                OpKind::PackedContraction {
                    descriptor,
                    rows: ContractionRows::Indexed { .. },
                } => {
                    eqn.op = OpKind::PackedContraction {
                        descriptor,
                        rows: ContractionRows::Indexed { experts: slots },
                    };
                    1
                }
                _ => unreachable!("find only collects indexed contractions"),
            };
            eqn.inputs[position] = Operand::Value(translated);
        }
        let mut weights = Vec::with_capacity(site.tables.len());
        for table in &site.tables {
            let mut pooled_aval = pooled.aval(table.value).clone();
            *pooled_aval
                .shape
                .first_mut()
                .ok_or(PoolError::TableNotPageable {
                    eqn: site.eqns[0],
                    value: table.value,
                })? = slots;
            let slot =
                pooled.rebind_const_as_slot(table.value, Slot::Owned(EXPERT_POOL), pooled_aval)?;
            weights.push(PooledWeight {
                name: table.name.clone(),
                slot,
                pool: pooled.aval(table.value).clone(),
            });
        }
        sites.push(PoolSite {
            layer: layer.map(|layer| layer.0),
            experts: site.experts,
            slots,
            ids: BoundaryId(site.ids),
            map: SlotKey::new(Slot::Owned(EXPERT_POOL_MAP), Some(&tag)),
            weights,
        });
    }

    // The views between a table and its contraction now carry `P` rows: re-infer them in order.
    let views: HashSet<usize> = found.iter().flat_map(|site| site.views.clone()).collect();
    for index in 0..pooled.eqns().len() {
        if !views.contains(&index) {
            continue;
        }
        let eqn = &pooled.eqns()[index];
        let inputs: Vec<TensorType> = eqn
            .inputs
            .iter()
            .filter_map(|operand| match operand {
                Operand::Value(value) => Some(pooled.values()[*value].aval.clone()),
                Operand::Lit(_) => None,
            })
            .collect();
        let aval = eqn
            .op
            .infer(&inputs)
            .unwrap_or_else(|error| panic!("pool_experts: view of a pooled table: {error}"));
        let out = eqn.out;
        pooled.values_mut()[out].aval = aval;
    }

    // Rebuild the equation list: each site's gather opens the stage after its sync, so the views
    // between its tables and its contractions move to just after the gather (they read the pool,
    // which is only resident from the sync on). Cuts are the gathers' positions in the new list.
    let original = std::mem::take(pooled.eqns_mut());
    let moved: HashSet<usize> = found.iter().flat_map(|site| site.views.clone()).collect();
    let mut opens: HashMap<usize, usize> = HashMap::new();
    for (ordinal, site) in found.iter().enumerate() {
        opens.insert(site.eqns[0], ordinal);
    }
    let mut gathers: Vec<Option<Vec<Eqn>>> =
        gathers.into_iter().map(|(_, chain)| Some(chain)).collect();
    let mut cuts = Vec::with_capacity(found.len());
    let mut eqns = Vec::with_capacity(original.len() + found.len());
    for (index, eqn) in original.iter().enumerate() {
        if let Some(&ordinal) = opens.get(&index) {
            cuts.push(eqns.len());
            eqns.extend(gathers[ordinal].take().expect("one gather per site"));
            eqns.extend(found[ordinal].views.iter().map(|&v| original[v].clone()));
        }
        if !moved.contains(&index) {
            eqns.push(eqn.clone());
        }
    }
    *pooled.eqns_mut() = eqns;

    let graph = pooled.commit_adding_witnesses(&added)?;
    Ok(Pooled {
        numerics: NumericsDelta::between(g, &graph).produced,
        graph,
        spec: PoolSpec { sites },
        cuts,
    })
}

/// Append the canonical index guard ([`emit_index_guard`]) of `raw` (the translated rows of one site) to
/// `chain`: its equations go after the gather, in the stage that reads the pool. Returns the guarded rows and
/// the witness.
fn guard_rows(
    g: &mut GraphEdit,
    chain: &mut Vec<Eqn>,
    raw: ValueId,
    rows: usize,
    layer: Option<poot_graph_ir::graph::LayerIndex>,
) -> Result<GuardValues, PoolError> {
    let raw_type = g.aval(raw).clone();
    emit_index_guard(raw, &raw_type, rows, |op: OpKind, inputs: Vec<Operand>| {
        let types: Vec<TensorType> = inputs
            .iter()
            .map(|operand| match operand {
                Operand::Value(value) => g.aval(*value).clone(),
                Operand::Lit(scalar) => scalar.ty(),
            })
            .collect();
        let aval = op.infer(&types)?;
        let out = g
            .values_mut()
            .push(ValueMeta::new(aval, Storage::Device, None));
        chain.push(Eqn {
            op,
            inputs,
            out,
            layer,
        });
        Ok::<_, ShapeError>(out)
    })
    .map_err(|error| match error {
        IndexGuardEmitError::Guard(error) => PoolError::RowGuard(error),
        IndexGuardEmitError::Emit(error) => PoolError::RowGuardShape(error),
    })
}

/// Every routed selection of `g` whose contractions read pageable tables.
fn find(g: &Graph) -> Result<Vec<Found>, PoolError> {
    let producer: HashMap<ValueId, usize> = g
        .eqns()
        .iter()
        .enumerate()
        .map(|(i, e)| (e.out, i))
        .collect();
    let mut readers: HashMap<ValueId, Vec<usize>> = HashMap::new();
    for (i, eqn) in g.eqns().iter().enumerate() {
        for operand in &eqn.inputs {
            if let Operand::Value(value) = operand {
                readers.entry(*value).or_default().push(i);
            }
        }
    }

    // Sites grouped by the index value the contractions read: one selection, however many
    // projections consume it.
    let mut by_ids: BTreeMap<ValueId, (usize, Vec<usize>)> = BTreeMap::new();
    for site in routed_expert_sites(g) {
        // Every router row selects `k` experts: that is one step's demand on the pool.
        let OpKind::ArgTopK { k } = g.eqns()[site.select].op else {
            unreachable!("a routed site's selection is an ArgTopK")
        };
        let router_shape = &g.aval(g.eqns()[site.router].out).shape;
        let routed_rows: usize = router_shape[..router_shape.len() - 1].iter().product();
        let demand = routed_rows * k;
        for &e in &site.experts {
            let eqn = &g.eqns()[e];
            let position = match eqn.op {
                OpKind::IndexedMatMul => 2,
                _ => 1,
            };
            let Some(Operand::Value(ids)) = eqn.inputs.get(position) else {
                continue;
            };
            let (_, eqns) = by_ids.entry(*ids).or_insert_with(|| (demand, Vec::new()));
            if !eqns.contains(&e) {
                eqns.push(e);
            }
        }
    }

    let mut owner: HashMap<ValueId, ValueId> = HashMap::new();
    let mut found = Vec::new();
    for (ids, (demand, mut eqns)) in by_ids {
        eqns.sort_unstable();
        let mut tables: Vec<Table> = Vec::new();
        let mut views: Vec<usize> = Vec::new();
        let mut experts: Option<usize> = None;
        let allowed_start = eqns.clone();
        for &e in &eqns {
            let eqn = &g.eqns()[e];
            let operands: Vec<ValueId> = match eqn.op {
                OpKind::IndexedMatMul => vec![operand(eqn, 1)],
                _ => eqn
                    .inputs
                    .iter()
                    .skip(2)
                    .filter_map(|operand| match operand {
                        Operand::Value(value) => Some(*value),
                        Operand::Lit(_) => None,
                    })
                    .collect(),
            };
            for start in operands {
                let (table, chain) = walk_to_const(g, &producer, e, start)?;
                let count = g.aval(table).shape[0];
                match experts {
                    None => experts = Some(count),
                    Some(first) if first != count => {
                        return Err(PoolError::ExpertCountMismatch {
                            first,
                            second: count,
                        });
                    }
                    Some(_) => {}
                }
                let name = g
                    .meta(table)
                    .name()
                    .expect("walk_to_const returns a named const")
                    .into_owned();
                // Every reader of the const and of each view must be this site's own chain.
                let mut allowed: HashSet<usize> = allowed_start.iter().copied().collect();
                allowed.extend(chain.iter().copied());
                let values = std::iter::once(table).chain(chain.iter().map(|&i| g.eqns()[i].out));
                for value in values {
                    let readers = readers.get(&value).map(Vec::as_slice).unwrap_or(&[]);
                    if readers.iter().any(|r| !allowed.contains(r))
                        || g.liveness_roots().any(|root| root == value)
                    {
                        return Err(PoolError::TableReadOutsideSite { name });
                    }
                }
                if let Some(&other) = owner.get(&table)
                    && other != ids
                {
                    return Err(PoolError::TableSharedAcrossSites { name });
                }
                owner.insert(table, ids);
                if !tables.iter().any(|t| t.value == table) {
                    tables.push(Table { value: table, name });
                }
                views.extend(chain);
            }
        }
        views.sort_unstable();
        views.dedup();
        found.push(Found {
            ids,
            demand,
            eqns,
            tables,
            experts: experts.expect("a site has a contraction with a table"),
            views,
        });
    }
    Ok(found)
}

fn operand(eqn: &Eqn, position: usize) -> ValueId {
    match eqn.inputs[position] {
        Operand::Value(value) => value,
        Operand::Lit(_) => unreachable!("an indexed contraction's table is a value"),
    }
}

/// Follow `start` back through transposes that keep the leading axis to a named const, returning
/// the const and the view equations crossed (nearest the contraction first).
fn walk_to_const(
    g: &Graph,
    producer: &HashMap<ValueId, usize>,
    contraction: usize,
    start: ValueId,
) -> Result<(ValueId, Vec<usize>), PoolError> {
    let mut value = start;
    let mut chain = Vec::new();
    loop {
        match producer.get(&value) {
            None => {
                let meta = g.meta(value);
                return if meta.storage == Storage::Const
                    && meta.name().is_some()
                    && !g.aval(value).shape.is_empty()
                {
                    Ok((value, chain))
                } else {
                    Err(PoolError::TableNotPageable {
                        eqn: contraction,
                        value: start,
                    })
                };
            }
            Some(&i) => {
                let eqn = &g.eqns()[i];
                match &eqn.op {
                    OpKind::Transpose { perm } if perm.first() == Some(&0) => {
                        chain.push(i);
                        value = operand(eqn, 0);
                    }
                    _ => {
                        return Err(PoolError::TableNotPageable {
                            eqn: contraction,
                            value: start,
                        });
                    }
                }
            }
        }
    }
}
