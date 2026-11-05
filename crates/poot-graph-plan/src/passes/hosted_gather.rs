//! POOT-1423: hosted gather sites. A weight table that only gathers read never becomes a device buffer.
//!
//! A token embedding, a per-layer n-gram table and a hash router's index table are each read by one
//! axis-0 gather: a step needs `n` rows of the table, never the table. [`host_gather_tables`] finds such a
//! gather by its **table** (a whole declared weight, dense or the `PackedDequant` of one weight's source
//! carriers, float rows, no other reader, nothing pinned) whatever size the table has and whatever backend
//! runs the graph, and takes the table out of the device graph. The table stays in the checkpoint mapping;
//! the executor copies the **stored** bytes of the `n` rows a step gathers into a small [`HOSTED_ROWS`] slot
//! per carrier, and a packed table's decode stays in the device graph (ADR-0103: bytes move as stored).
//!
//! Where the row indices come from decides only when the host copies ([`IndexSource`]):
//!
//! - a host-bound step input (the token slot) is known before the step runs, so the rows are copied at bind
//!   time;
//! - an index the device computes (a hash over carried history) is exported by the stage that produces it and
//!   read at a host sync ([`crate::staged::SyncAction::GatherRows`]), the mechanism the expert pool uses; the
//!   rows are written before the stage that reads them runs. The hash stays in the device graph and its
//!   history stays device state: no host evaluator and no host-owned state exist.
//!
//! The pass runs in [`crate::compile_staged`] before the expert pool and the cuts, on the graph as the tracer
//! emitted it, so a quantized table is still `Gather(PackedDequant(carriers), ids)`: after the packed claims it
//! would be a `PackedRowGather` the planner may refuse (a carrier past a backend's index width) before any later
//! pass could host it.
//!
//! One table has one hosted gather: a table two gathers read is not a sole-reader gather table and stays a
//! device constant, and the planner or `legalize` answers for it (typed refusal when it is oversized).

use std::collections::{HashMap, HashSet};

use poot_graph_ir::graph::{
    BoundaryId, EditError, Eqn, Graph, Operand, Slot, SlotKey, Storage, ValueId, ValueMeta,
};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::types::{DTypeClass, DTypeClassExt, TensorType};
use poot_graph_ir::{ComputedConst, OwnedSlot, WeightRef, packed_source_type};
use poot_quant::PackedWeight;
use poot_tensor::DType;

use super::legalize::{declared_weight, operand_value};

/// The rows a host gathers from a hosted table for one step: one slot per carrier of the table (the dense
/// weight itself, or each source role of a packed weight), tagged with the carrier's const name, which is how
/// the executor finds the stored bytes. Its buffer holds the carrier's stored bytes for the `n` gathered rows,
/// in the lane the readers' storage plan chose, never decoded: a dense table's rows are `[ids.., width]` in
/// the table's dtype, a packed carrier's are `[n, row_bytes]` raw bytes.
pub const HOSTED_ROWS: OwnedSlot = OwnedSlot::new("hostedrows", true, |aval, _| {
    aval.shape.is_empty().then_some("a tensor of gathered rows")
});

/// Where a hosted gather's row indices come from, which decides when the host copies the rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexSource {
    /// A host-bound step input: the rows are copied at bind time, before the step runs.
    Step(SlotKey),
    /// A value the device computes: the stage that produces it exports it, and the rows are copied at the host
    /// sync after that stage.
    Device(BoundaryId),
}

/// One carrier of a hosted table and the slot its gathered rows land in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostedRows {
    /// The [`HOSTED_ROWS`] slot, tagged with `source`.
    pub slot: SlotKey,
    /// The name of the carrier constant the slot replaces: how the executor's weight map finds the stored
    /// bytes the rows are copied from.
    pub source: String,
    /// The slot's type: the rows of one step.
    pub aval: TensorType,
}

/// One hosted gather site: the table's carriers, the rows a step copies from them, and where the ids come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostedGather {
    /// Where the row indices come from.
    pub index: IndexSource,
    /// The ids a step gathers: the element count of the index operand, and so the row count of every slot.
    pub ids: usize,
    /// The stored bytes of the whole table, which stay in the checkpoint mapping instead of the device.
    pub table_bytes: u64,
    /// One entry per carrier of the table, in the order the carriers' roles list them.
    pub rows: Vec<HostedRows>,
}

/// A graph with its gather-only tables hosted.
pub(crate) struct Hosted {
    pub graph: Graph,
    /// The hosted sites with the values their rows land in, in equation order of the gathers they replace.
    pub sites: Vec<HostedSite>,
}

/// One site of [`Hosted`].
pub(crate) struct HostedSite {
    pub gather: HostedGather,
    /// The graph values holding the site's rows: the replaced gather's output (dense), or the rebound
    /// carriers (packed). The equations that read them wait for the site's rows.
    pub values: Vec<ValueId>,
    /// The index value, when the device computes it: the host reads it at a sync, so the program is cut
    /// before the first equation that reads a row.
    pub device_index: Option<ValueId>,
}

/// How a candidate gather's table is held.
enum Table {
    /// A dense whole weight constant, `[rows, width]`.
    Dense(ValueId),
    /// The output of the `PackedDequant` equation `dequant` over one weight's source carriers.
    Packed {
        dequant: usize,
        descriptor: PackedWeight,
        carriers: Vec<ValueId>,
    },
}

/// A gather the pass may host.
struct Candidate {
    eqn: usize,
    out: ValueId,
    index: ValueId,
    weight: WeightRef,
    table: Table,
    source: IndexKind,
}

/// What the index value is, classified before the gather is rewritten.
enum IndexKind {
    Step(SlotKey),
    Device,
}

/// The whole weight whose source carriers are exactly `sources`: every carrier is an unchunked source
/// constant of one weight.
fn packed_table_weight(g: &Graph, sources: &[Operand]) -> Option<WeightRef> {
    let mut whole = None;
    for &operand in sources {
        let id = operand_value(operand)?;
        let weight = g.meta(id).weight().filter(|w| w.chunk().is_none())?;
        weight.source()?;
        if g.meta(id).storage != Storage::Const || *whole.get_or_insert(weight.id()) != weight.id()
        {
            return None;
        }
    }
    whole.map(WeightRef::whole)
}

/// Host every gather-only table of `g`; see the module doc. A graph with none is returned unchanged.
pub(crate) fn host_gather_tables(
    g: &Graph,
    aliases: &poot_weights::WeightAliases,
) -> Result<Hosted, EditError> {
    let producer: HashMap<ValueId, usize> = g
        .eqns()
        .iter()
        .enumerate()
        .map(|(i, eqn)| (eqn.out, i))
        .collect();
    let mut readers: HashMap<ValueId, HashSet<usize>> = HashMap::new();
    for (i, eqn) in g.eqns().iter().enumerate() {
        for value in eqn.inputs.iter().copied().filter_map(operand_value) {
            readers.entry(value).or_default().insert(i);
        }
    }
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    // Only the gather reads `value`, and nothing observes it.
    let read_only_by = |value: ValueId, eqn: usize| {
        !pinned.contains(&value)
            && readers
                .get(&value)
                .is_none_or(|readers| readers.iter().all(|&reader| reader == eqn))
    };

    let mut candidates: Vec<Candidate> = Vec::new();
    for (i, eqn) in g.eqns().iter().enumerate() {
        if eqn.op != (OpKind::Gather { axis: 0 }) || eqn.inputs.len() != 2 {
            continue;
        }
        let (Some(table), Some(index)) =
            (operand_value(eqn.inputs[0]), operand_value(eqn.inputs[1]))
        else {
            continue;
        };
        let source = match (g.meta(index).storage, g.meta(index).slot_key()) {
            (Storage::Slot(Slot::Token), Some(key)) => IndexKind::Step(key.clone()),
            (Storage::Device, _) if producer.contains_key(&index) => IndexKind::Device,
            _ => continue,
        };
        // The rows are floats: an index table (a hash router's I32 words) is no embedding.
        if g.aval(eqn.out).dtype.class() != DTypeClass::Arithmetic {
            continue;
        }
        let (table, weight) = if g.meta(table).storage == Storage::Const {
            let Some(weight) = declared_weight(g, table) else {
                continue;
            };
            if g.aval(table).shape.len() != 2 || !read_only_by(table, i) {
                continue;
            }
            (Table::Dense(table), weight)
        } else if let Some(&d) = producer.get(&table)
            && let OpKind::PackedDequant { descriptor } = g.eqns()[d].op
        {
            let sources = &g.eqns()[d].inputs;
            let Some(weight) = packed_table_weight(g, sources) else {
                continue;
            };
            let carriers: Vec<ValueId> =
                sources.iter().copied().filter_map(operand_value).collect();
            // Every carrier is stored row by row, so `n` rows are `n` stored rows of each.
            let row_major = descriptor
                .sources()
                .into_iter()
                .all(|role| descriptor.source_shape(role)[0] == descriptor.shape()[0]);
            let sole = read_only_by(table, i)
                && carriers.iter().all(|&carrier| read_only_by(carrier, d))
                && !carriers
                    .iter()
                    .any(|&carrier| readers.get(&carrier).is_some_and(|r| r.len() > 1));
            if !row_major || !sole {
                continue;
            }
            (
                Table::Packed {
                    dequant: d,
                    descriptor,
                    carriers,
                },
                weight,
            )
        } else {
            continue;
        };
        candidates.push(Candidate {
            eqn: i,
            out: eqn.out,
            index,
            weight,
            table,
            source,
        });
    }
    // A table whose stored bytes the graph also binds under another weight id (a tied head) is on the device
    // anyway: hosting it would only make the program host-fed.
    let declared: HashSet<poot_weights::WeightId> = g
        .consts()
        .iter()
        .filter_map(|&id| g.meta(id).weight())
        .map(|weight| weight.id())
        .collect();
    candidates.retain(|candidate| {
        !aliases
            .others(candidate.weight.id())
            .any(|other| declared.contains(&other))
    });
    // One gather per table: a second gather of the same weight leaves both on the device.
    let mut per_weight: HashMap<WeightRef, usize> = HashMap::new();
    for candidate in &candidates {
        *per_weight.entry(candidate.weight).or_default() += 1;
    }
    candidates.retain(|candidate| per_weight[&candidate.weight] == 1);
    if candidates.is_empty() {
        return Ok(Hosted {
            graph: g.clone(),
            sites: Vec::new(),
        });
    }

    let mut edit = g.edit();
    let mut sites = Vec::with_capacity(candidates.len());
    let mut dropped_eqns: HashSet<usize> = HashSet::new();
    let mut replaced: HashMap<usize, Vec<Eqn>> = HashMap::new();
    let mut dropped_values: HashSet<ValueId> = HashSet::new();
    for candidate in &candidates {
        let ids = g.aval(candidate.index);
        let n = ids.numel();
        let index = match &candidate.source {
            IndexKind::Step(key) => IndexSource::Step(key.clone()),
            IndexKind::Device => IndexSource::Device(BoundaryId(candidate.index)),
        };
        let device_index = matches!(candidate.source, IndexKind::Device).then_some(candidate.index);
        match &candidate.table {
            Table::Dense(table) => {
                let aval = g.aval(candidate.out).clone();
                let source = candidate.weight.name();
                let key = SlotKey::new(Slot::Owned(HOSTED_ROWS), Some(&source));
                edit.values_mut()[candidate.out] =
                    ValueMeta::slot_input(aval.clone(), key.clone()).map_err(EditError::from)?;
                edit.inputs_mut().push(candidate.out);
                edit.slots_mut()
                    .push((candidate.out, Slot::Owned(HOSTED_ROWS)));
                dropped_eqns.insert(candidate.eqn);
                dropped_values.insert(*table);
                sites.push(HostedSite {
                    gather: HostedGather {
                        index,
                        ids: n,
                        table_bytes: g.aval(*table).numel() as u64
                            * g.aval(*table).dtype.byte_size() as u64,
                        rows: vec![HostedRows {
                            slot: key,
                            source,
                            aval,
                        }],
                    },
                    values: vec![candidate.out],
                    device_index,
                });
            }
            Table::Packed {
                dequant,
                descriptor,
                carriers,
            } => {
                let [_, k] = descriptor.shape();
                let Ok(rows_descriptor) = PackedWeight::try_new(descriptor.format(), [n, k]) else {
                    return Err(EditError::NotRebindable {
                        value: candidate.out,
                        reason: "the gathered rows are not a packed weight of the table's format",
                    });
                };
                // Each carrier becomes the slot holding the gathered stored rows, in role order.
                let mut rows = Vec::with_capacity(carriers.len());
                for (&carrier, role) in carriers.iter().zip(descriptor.sources()) {
                    let aval = packed_source_type(rows_descriptor, role);
                    let source = g
                        .meta(carrier)
                        .name()
                        .map(std::borrow::Cow::into_owned)
                        .ok_or(EditError::NotRebindable {
                            value: carrier,
                            reason: "a packed source carrier carries its name",
                        })?;
                    let slot =
                        edit.rebind_const_as_slot(carrier, Slot::Owned(HOSTED_ROWS), aval.clone())?;
                    rows.push(HostedRows { slot, source, aval });
                }
                // The decode of the gathered rows is a row gather over their own `[n, ..]` block, in order,
                // placed where the table's gather was: after the index, never before it.
                let iota = edit.values_mut().push(ValueMeta::new(
                    TensorType::new(vec![n], DType::I32),
                    Storage::Computed(ComputedConst::Iota {
                        len: n,
                        dtype: DType::I32,
                    }),
                    None,
                ));
                edit.inputs_mut().push(iota);
                edit.consts_mut().push(iota);
                let decoded = edit.values_mut().push(ValueMeta::new(
                    TensorType::new(vec![n, k], DType::F32),
                    Storage::Device,
                    None,
                ));
                let gather = &g.eqns()[candidate.eqn];
                let mut inputs: Vec<Operand> =
                    carriers.iter().map(|&c| Operand::Value(c)).collect();
                inputs.push(Operand::Value(iota));
                replaced.insert(
                    candidate.eqn,
                    vec![
                        Eqn {
                            op: OpKind::PackedRowGather {
                                descriptor: rows_descriptor,
                            },
                            inputs,
                            out: decoded,
                            layer: gather.layer,
                        },
                        Eqn {
                            op: OpKind::Reshape {
                                shape: g.aval(candidate.out).shape.clone(),
                            },
                            inputs: vec![Operand::Value(decoded)],
                            out: candidate.out,
                            layer: gather.layer,
                        },
                    ],
                );
                dropped_eqns.insert(*dequant);
                sites.push(HostedSite {
                    gather: HostedGather {
                        index,
                        ids: n,
                        table_bytes: descriptor.total_source_bytes() as u64,
                        rows,
                    },
                    values: carriers.clone(),
                    device_index,
                });
            }
        }
    }

    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns().len());
    for (i, eqn) in g.eqns().iter().enumerate() {
        if let Some(rewrite) = replaced.remove(&i) {
            eqns.extend(rewrite);
        } else if !dropped_eqns.contains(&i) {
            eqns.push(eqn.clone());
        }
    }
    *edit.eqns_mut() = eqns;
    // The dense table's constant leaves the graph; the token slot too, once nothing reads it.
    edit.inputs_mut().retain(|id| !dropped_values.contains(id));
    edit.consts_mut().retain(|id| !dropped_values.contains(id));
    edit.slots_mut()
        .retain(|(id, _)| !dropped_values.contains(id));
    let graph = edit.commit()?;
    let graph = drop_unread_inputs(&graph, candidates.iter().map(|c| c.index))?;
    Ok(Hosted { graph, sites })
}

/// `g` without the token slots in `candidates` that no equation, output or state pair reads any more: the
/// slot the hosted gather read is bound by the executor from the step's inputs, not declared by the graph.
fn drop_unread_inputs(
    g: &Graph,
    candidates: impl Iterator<Item = ValueId>,
) -> Result<Graph, EditError> {
    let referenced = |id: ValueId| -> bool {
        g.eqns()
            .iter()
            .any(|e| e.inputs.iter().any(|&o| operand_value(o) == Some(id)))
            || g.pinned_values().any(|v| v == id)
    };
    let unread: HashSet<ValueId> = candidates
        .filter(|&id| g.meta(id).storage != Storage::Device && !referenced(id))
        .collect();
    if unread.is_empty() {
        return Ok(g.clone());
    }
    let mut edit = g.edit();
    edit.inputs_mut().retain(|id| !unread.contains(id));
    edit.slots_mut().retain(|(id, _)| !unread.contains(id));
    edit.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    use poot_graph_ir::{BinOp, Builder, Scalar, Traced};
    use poot_quant::format::WeightFormat;
    use poot_weights::{WeightId, WeightRole};

    const EMBED: WeightId = WeightId::model(WeightRole::Embed);

    fn token_ids(b: &Builder, n: usize) -> Traced {
        b.slot(Slot::Token, TensorType::new(vec![n], DType::I32))
            .unwrap()
    }

    /// `[vocab, hidden]` F32 embed table gathered by `n` token ids.
    fn dense_gather_graph(vocab: usize, hidden: usize, n: usize) -> Graph {
        let b = Builder::for_tests();
        let ids = token_ids(&b, n);
        let table = b
            .weight(
                WeightRef::whole(EMBED),
                TensorType::f32(vec![vocab, hidden]),
            )
            .unwrap();
        let rows = b.gather(table, 0, ids).unwrap();
        b.finish(rows).unwrap()
    }

    /// The packed form: a `[vocab, 160]` `Iq4_Nl` weight whose source carriers a tracer declares, dequantized
    /// and gathered by `ids`; `offset` shifts the ids by a literal first, so the index is a value the device
    /// computes (the per-layer table's hash offset has the same shape), and `tied_head` reads the dequantized
    /// table through a transpose too.
    fn packed_gather_graph(vocab: usize, offset: Option<i32>, tied_head: bool) -> Graph {
        let descriptor = PackedWeight::try_new(WeightFormat::Iq4_Nl, [vocab, 160]).unwrap();
        let b = Builder::for_tests();
        let mut ids = token_ids(&b, 2);
        if let Some(offset) = offset {
            ids = b
                .binary_scalar(BinOp::Add, ids, Scalar::I32(offset))
                .unwrap();
        }
        let sources: Vec<Traced> = descriptor
            .sources()
            .into_iter()
            .map(|role| {
                b.weight(
                    WeightRef::whole(EMBED).with_source(role),
                    packed_source_type(descriptor, role),
                )
                .unwrap()
            })
            .collect();
        let table = b.packed_dequant(&sources, descriptor).unwrap();
        let mut out = b.gather(table, 0, ids).unwrap();
        if tied_head {
            let x = b.reshape(out, vec![2, 160]).unwrap();
            out = b
                .matmul(x, b.transpose(table, vec![1, 0]).unwrap())
                .unwrap();
        }
        b.finish(out).unwrap()
    }

    fn hosted_inputs(g: &Graph) -> Vec<ValueId> {
        g.slots()
            .iter()
            .filter(|&&(_, slot)| slot == Slot::Owned(HOSTED_ROWS))
            .map(|&(id, _)| id)
            .collect()
    }

    /// SC-001/SC-004: a dense table only a gather reads is hosted at any size, whatever backend runs it: the
    /// gather and the constant are gone, one `HOSTED_ROWS` slot named after the weight replaces them, and the
    /// token slot drops out because nothing reads it. The index is a step input, so rows are copied at bind
    /// time.
    ///
    /// Mutation: make the reader check (`read_only_by`) return `false` for the table; nothing is hosted and
    /// the first assertion goes red (`the gather is hosted`).
    #[test]
    fn a_sole_reader_dense_table_is_hosted_at_any_size() {
        for (vocab, hidden) in [(1024, 8), (152_064, 3584)] {
            let g = dense_gather_graph(vocab, hidden, 3);
            let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
            assert!(
                !hosted
                    .graph
                    .eqns()
                    .iter()
                    .any(|e| matches!(e.op, OpKind::Gather { .. })),
                "{vocab}x{hidden}: the gather is hosted"
            );
            assert!(hosted.graph.consts().is_empty(), "the table drops out");
            assert_eq!(
                hosted.graph.inputs().len(),
                1,
                "one slot replaces token and table"
            );
            let [id] = hosted_inputs(&hosted.graph)[..] else {
                panic!("one hosted slot")
            };
            assert_eq!(hosted.graph.aval(id).shape, vec![3, hidden]);
            let [site] = hosted.sites.as_slice() else {
                panic!("one site")
            };
            assert_eq!(
                site.gather.index,
                IndexSource::Step(SlotKey::new(Slot::Token, None))
            );
            assert_eq!(site.gather.ids, 3);
            assert_eq!(site.gather.rows[0].source, "w.embed");
            assert_eq!(
                site.gather.table_bytes,
                (vocab * hidden * 4) as u64,
                "the table's bytes stay in the mapping"
            );
            assert_eq!(site.device_index, None);
        }
    }

    /// SC-001: a packed table is hosted at any size: every source carrier becomes a slot of the gathered stored
    /// rows (`[n, row_bytes]` of the table's own bytes), the dequant stays in the graph as a row gather over
    /// them in order, and no packed carrier remains a device constant.
    ///
    /// Mutation: drop the `PackedDequant` arm of the candidate scan (`else if let Some(&d) = producer.get`);
    /// the table is left a device constant and the first assertion goes red.
    #[test]
    fn a_sole_reader_packed_table_is_hosted_at_any_size() {
        for vocab in [1024, 320_001_536] {
            let g = packed_gather_graph(vocab, None, false);
            let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
            assert!(
                hosted
                    .graph
                    .consts()
                    .iter()
                    .all(|&id| matches!(hosted.graph.meta(id).storage, Storage::Computed(_))),
                "{vocab} rows: no packed carrier stays a device constant"
            );
            let [site] = hosted.sites.as_slice() else {
                panic!("one site")
            };
            let [rows] = site.gather.rows.as_slice() else {
                panic!("Iq4_Nl has one source")
            };
            // 160 values of IQ4_NL: five 18-byte blocks, 90 bytes a row, one row per id.
            assert_eq!(rows.aval, TensorType::new(vec![2, 90], DType::I8));
            assert_eq!(rows.source, "w.embed.packed_blocks_source");
            assert!(
                hosted
                    .graph
                    .eqns()
                    .iter()
                    .any(|e| matches!(e.op, OpKind::PackedRowGather { .. })),
                "{vocab} rows: the decode of the gathered rows stays in the graph"
            );
            assert!(
                !hosted
                    .graph
                    .eqns()
                    .iter()
                    .any(|e| matches!(e.op, OpKind::PackedDequant { .. } | OpKind::Gather { .. })),
                "{vocab} rows: no whole-table dequant or gather is left"
            );
            assert_eq!(
                site.gather.table_bytes,
                (vocab * 90) as u64,
                "{vocab} rows: the table's stored bytes"
            );
        }
    }

    /// SC-001: an index the device computes (here the token plus a literal, the shape of the per-layer table's
    /// head offset) is no step input: the site is a device-indexed gather, and the index value is the one the
    /// host reads at a sync.
    ///
    /// Mutation: restore the `Slot::Token` filter on the index; the gather is skipped and nothing is hosted.
    #[test]
    fn a_gather_by_a_device_computed_index_is_a_device_site() {
        let g = packed_gather_graph(4096, Some(7), false);
        let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
        let [site] = hosted.sites.as_slice() else {
            panic!("one site")
        };
        let IndexSource::Device(boundary) = &site.gather.index else {
            panic!(
                "the index is computed on the device: {:?}",
                site.gather.index
            )
        };
        assert_eq!(site.device_index, Some(boundary.0));
        assert!(
            hosted
                .graph
                .eqns()
                .iter()
                .any(|e| e.out == boundary.0 && matches!(e.op, OpKind::Binary(BinOp::Add))),
            "the index producer stays in the graph"
        );
    }

    /// SC-004: a table another equation reads is no sole-reader gather table: a tied head also reads it, so the
    /// graph is unchanged, dense and packed.
    ///
    /// Mutation: skip the reader check (`read_only_by` always `true`); the table is hosted while the matmul
    /// still reads it, and the commit refuses the graph (an operand that is not defined).
    #[test]
    fn a_table_another_equation_reads_stays_on_the_device() {
        let dense = {
            let b = Builder::for_tests();
            let ids = token_ids(&b, 1);
            let embed = b
                .weight(WeightRef::whole(EMBED), TensorType::f32(vec![4096, 64]))
                .unwrap();
            let row = b.gather(embed, 0, ids).unwrap();
            let out = b
                .matmul(row, b.transpose(embed, vec![1, 0]).unwrap())
                .unwrap();
            b.finish(out).unwrap()
        };
        for g in [dense, packed_gather_graph(1024, None, true)] {
            let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
            assert!(hosted.sites.is_empty());
            assert_eq!(hosted.graph.eqns().len(), g.eqns().len());
            assert_eq!(hosted.graph.consts(), g.consts());
        }
    }

    /// SC-004: an index table (I32 words, a hash router's) gathered by the token is no embedding: the rows are
    /// not floats, so the gather stays on the device.
    ///
    /// Mutation: drop the float-rows check; the I32 rows are declared as a hosted slot and the executor's
    /// binder has no lane for them.
    #[test]
    fn an_i32_index_table_is_not_hosted() {
        let b = Builder::for_tests();
        let ids = token_ids(&b, 1);
        let table = b
            .weight(
                WeightRef::whole(EMBED),
                TensorType::new(vec![64, 4], DType::I32),
            )
            .unwrap();
        let rows = b.gather(table, 0, ids).unwrap();
        let g = b.finish(rows).unwrap();
        let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
        assert!(hosted.sites.is_empty());
        assert_eq!(hosted.graph.eqns().len(), g.eqns().len());
    }

    /// Two gathers of one table are not a sole-reader table either: each would need the rows of the other's
    /// ids, so both stay on the device.
    ///
    /// Mutation: drop the one-gather-per-weight filter; the second rebinds a carrier already rebound and the
    /// edit refuses it.
    #[test]
    fn a_table_two_gathers_read_stays_on_the_device() {
        let b = Builder::for_tests();
        let ids = token_ids(&b, 1);
        let other = token_ids(&b, 2);
        let table = b
            .weight(WeightRef::whole(EMBED), TensorType::f32(vec![64, 4]))
            .unwrap();
        let first = b.gather(table, 0, ids).unwrap();
        let second = b.gather(table, 0, other).unwrap();
        let first = b.reshape(first, vec![4]).unwrap();
        let second = b
            .reduce(poot_graph_ir::RedOp::Sum, second, 0, false)
            .unwrap();
        let out = b.binary(BinOp::Add, first, second).unwrap();
        let g = b.finish(out).unwrap();
        let hosted = host_gather_tables(&g, &poot_weights::WeightAliases::none()).unwrap();
        assert!(hosted.sites.is_empty(), "{:?}", hosted.sites.len());
    }
}
