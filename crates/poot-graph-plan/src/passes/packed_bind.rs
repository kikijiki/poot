//! Card 545a: the one place a model's stored weight formats reach a graph.
//!
//! Tracers are storage-agnostic: they declare every weight as a logical dense constant (a projection
//! `[K, out]` read by a matmul, an embedding table `[rows, K]` read by a gather, an expert stack
//! `[E, out, K]` read through `Transpose([0, 2, 1])` by an indexed matmul). A checkpoint that stores some of
//! those weights packed records, per constant name, which packed owner holds it ([`WeightFormats`], from a
//! `Model`'s `WeightMap` handles on the driver path ([`WeightFormats::from_weight_map`])).
//! [`bind_packed_weights`] rewrites each such constant into the decode of its owner - `PackedDequant` over
//! `descriptor.sources()` carrier constants named by [`PackedSourceName`], transposed for a projection -
//! producing the very value id the tracer declared, so every consumer is untouched (a biased projection is
//! already `ops::linear`'s matmul then add, so the claim sees the product). Binding only places storage: the
//! claims (`PackedContraction`, `PackedRowGather`) and the escape gate are `compile`'s (the CPU oracle claims
//! explicitly in `poot-llm`'s `cpu_oracle`).
//!
//! An expert stack has no logical dense value to decode into (a decoded `[E, K, out]` table is exactly what
//! packed serving must never build), so its consumer is rewritten instead: the one `IndexedMatMul` reading it
//! through the transpose becomes one `PackedContraction { rows: Indexed }` over one stacked carrier per source
//! role, and any other use of the stack is a typed refusal. A mismatch between what a tracer declared and what
//! the checkpoint stores is a typed refusal, never a reinterpretation.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::packed::packed_consumer_map;
use super::{EditError, Eqn, Graph, OpKind, Operand, Storage, ValueId, ValueMeta};
use poot_graph_ir::packed_source::{
    PackedSourceName, packed_source_type, packed_stack_source_constants,
};
use poot_graph_ir::types::TensorType;
use poot_graph_ir::{ContractionRows, ValueTable};
use poot_quant::weights::{HandleFormat, WeightMap, WeightView};
use poot_tensor::DType;

/// How a packed owner's logical `[out, K]` decode maps onto the constant a tracer declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedLayout {
    /// The constant is the decode as stored, `[out, K]` (an embedding table read by a gather).
    Rows,
    /// The constant is the decode transposed, `[K, out]` (a projection read by `x @ W`).
    Columns,
    /// The constant is an expert stack `[experts, out, K]` in checkpoint order: `experts` packed weights of
    /// one descriptor, read by an indexed matmul through `Transpose([0, 2, 1])`. One expert is a one-expert
    /// stack, `[1, out, K]`.
    ExpertRows {
        /// Number of experts in the stack.
        experts: usize,
    },
}

/// The packed storage of one dense constant a tracer declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedConst {
    /// The linear id its source carriers are staged under (see [`PackedSourceName`]); for an expert stack
    /// the stack's own const name, one stacked carrier per source role.
    pub linear_id: String,
    /// The owner's descriptor (for a stack, each expert's: one shape and format for the whole stack).
    pub weight: poot_quant::PackedWeight,
    /// How the stored decode is oriented relative to the constant the graph reads.
    pub layout: PackedLayout,
}

impl PackedConst {
    /// The dense shape a tracer must declare for this constant.
    fn declared_shape(&self) -> Vec<usize> {
        let [out, k] = self.weight.shape();
        match self.layout {
            PackedLayout::Rows => vec![out, k],
            PackedLayout::Columns => vec![k, out],
            PackedLayout::ExpertRows { experts } => vec![experts, out, k],
        }
    }
}

/// Which dense constants of a model are stored packed, keyed by the constant name the tracers
/// declare. Built once by the loader from the payloads it loads; empty for a dense model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightFormats {
    consts: BTreeMap<String, PackedConst>,
}

/// A packed-storage record that contradicts another.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WeightFormatsError {
    /// The constant is recorded packed more than once.
    #[error("constant {name} is recorded packed twice")]
    Duplicate {
        /// Name of the constant.
        name: String,
    },
}

impl WeightFormats {
    /// The packed weights of a model's [`WeightMap`] (the driver path reads formats
    /// from the map, never from a tracer). Each mapped weight whose handle is
    /// [`HandleFormat::Packed`] is recorded under its [`WeightId::const_name`], with its carriers
    /// staged under the same name: laid out [`PackedLayout::Rows`] (a `Model` tracer declares every
    /// weight in checkpoint `[out, K]` orientation and spells a projection's transpose itself), or
    /// [`PackedLayout::ExpertRows`] when its view is an [`WeightView::ExpertStack`].
    ///
    /// [`WeightId::const_name`]: poot_quant::weights::WeightId::const_name
    pub fn from_weight_map(map: &WeightMap) -> Self {
        let consts = map
            .iter()
            .filter_map(|(id, view, handle)| match handle.format {
                HandleFormat::Packed(weight) => Some((
                    id.const_name(),
                    PackedConst {
                        linear_id: id.const_name(),
                        weight,
                        layout: match view {
                            WeightView::ExpertStack(parts) => PackedLayout::ExpertRows {
                                experts: parts.len(),
                            },
                            _ => PackedLayout::Rows,
                        },
                    },
                )),
                HandleFormat::Dense(_) => None,
            })
            .collect();
        Self { consts }
    }

    /// Record that constant `name` is stored as `packed`.
    pub fn insert(&mut self, name: String, packed: PackedConst) -> Result<(), WeightFormatsError> {
        if self.consts.contains_key(&name) {
            return Err(WeightFormatsError::Duplicate { name });
        }
        self.consts.insert(name, packed);
        Ok(())
    }

    /// The packed record of constant `name`, if one is recorded.
    pub fn get(&self, name: &str) -> Option<&PackedConst> {
        self.consts.get(name)
    }

    /// Whether no constant is recorded packed.
    pub fn is_empty(&self) -> bool {
        self.consts.is_empty()
    }

    /// Every recorded constant with its packed record, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PackedConst)> {
        self.consts
            .iter()
            .map(|(name, packed)| (name.as_str(), packed))
    }
}

/// Why a traced graph cannot take a model's packed storage.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PackedBindError {
    /// The graph declares the packed constant with a dtype other than F32.
    #[error(
        "constant {name} is stored packed ({format:?}) but the graph declares it {dtype:?}; a packed \
         weight decodes to F32"
    )]
    Dtype {
        /// Name of the constant.
        name: String,
        /// The format the constant is stored in.
        format: poot_quant::format::WeightFormat,
        /// The dtype the graph declares.
        dtype: DType,
    },
    /// The graph declares the packed constant with a shape that is not the stored one.
    #[error(
        "constant {name} is stored packed as {expected:?} ({format:?}) but the graph declares {declared:?}"
    )]
    Shape {
        /// Name of the constant.
        name: String,
        /// The format the constant is stored in.
        format: poot_quant::format::WeightFormat,
        /// The shape the graph declares.
        declared: Vec<usize>,
        /// The shape the stored format decodes to.
        expected: Vec<usize>,
    },
    /// A packed expert stack is read in a way that cannot become one indexed packed contraction.
    #[error("constant {name} is a packed expert stack, but {reason}")]
    ExpertStack {
        /// Name of the constant.
        name: String,
        /// Why the use is refused.
        reason: ExpertStackRefusal,
    },
    /// The rewrite produced a graph the edit check refused.
    #[error("binding packed constant {name} produced an invalid graph: {source}")]
    Graph {
        /// Name of the constant.
        name: String,
        /// The editor's refusal.
        #[source]
        source: Box<EditError>,
    },
}

/// Why a packed expert stack's use cannot become one indexed packed contraction.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum ExpertStackRefusal {
    /// The stack has other than one consumer.
    #[error(
        "it has {consumers} consumers; a stack is read by exactly one transposed indexed matmul"
    )]
    Consumers {
        /// Number of consumers it has.
        consumers: usize,
    },
    /// The stack's consumer is not a `Transpose([0, 2, 1])`.
    #[error("its consumer is `{op}`, not Transpose([0, 2, 1]) into an indexed matmul")]
    NotTransposed {
        /// Name of the consumer's op.
        op: String,
    },
    /// The transposed stack has other than one consumer.
    #[error("its transpose has {consumers} consumers; the table is read by one indexed matmul")]
    TransposeConsumers {
        /// Number of consumers the transpose has.
        consumers: usize,
    },
    /// The transpose does not feed an indexed matmul.
    #[error("its transpose feeds `{op}`, not an indexed matmul")]
    NotIndexed {
        /// Name of the op it feeds.
        op: String,
    },
    /// The stack or its transpose is a graph output, validation output or carried state.
    #[error("it, or its transpose, is a graph output, validation output or carried state")]
    Escapes,
}

/// Rewrite every constant of `g` that `formats` records packed into the decode of its owners (see
/// the module docs). Constants `formats` does not name are untouched, so a dense model's graph is
/// returned unchanged.
pub fn bind_packed_weights(g: &Graph, formats: &WeightFormats) -> Result<Graph, PackedBindError> {
    if formats.is_empty() {
        return Ok(g.clone());
    }
    let consumers = packed_consumer_map(g);
    let mut out = g.edit();
    let mut decode: Vec<Eqn> = Vec::new();
    let mut bound: Vec<ValueId> = Vec::new();
    let mut removed: HashSet<usize> = HashSet::new();
    let mut claimed: HashMap<usize, Eqn> = HashMap::new();
    let mut last_name = None;
    for &id in g.inputs() {
        let meta = g.meta(id);
        if meta.storage != Storage::Const {
            continue;
        }
        let Some(name) = meta.name.as_deref() else {
            continue;
        };
        let Some(packed) = formats.get(name) else {
            continue;
        };
        let format = packed.weight.format();
        if meta.aval.dtype != DType::F32 {
            return Err(PackedBindError::Dtype {
                name: name.to_string(),
                format,
                dtype: meta.aval.dtype,
            });
        }
        let expected = packed.declared_shape();
        if meta.aval.shape != expected {
            return Err(PackedBindError::Shape {
                name: name.to_string(),
                format,
                declared: meta.aval.shape.clone(),
                expected,
            });
        }
        let [rows, k] = packed.weight.shape();
        if let PackedLayout::ExpertRows { experts } = packed.layout {
            let refuse = |reason| PackedBindError::ExpertStack {
                name: name.to_string(),
                reason,
            };
            let (transpose, indexed) = expert_stack_use(g, id, &consumers).map_err(refuse)?;
            let mut inputs = vec![g.eqns()[indexed].inputs[0], g.eqns()[indexed].inputs[2]];
            for (carrier_name, carrier_type) in
                packed_stack_source_constants(&packed.linear_id, packed.weight, experts)
            {
                let carrier = out.values_mut().push(ValueMeta::new(
                    carrier_type,
                    Storage::Const,
                    Some(carrier_name.into()),
                ));
                out.inputs_mut().push(carrier);
                out.consts_mut().push(carrier);
                inputs.push(Operand::Value(carrier));
            }
            removed.insert(transpose);
            claimed.insert(
                indexed,
                Eqn {
                    op: OpKind::PackedContraction {
                        descriptor: packed.weight,
                        rows: ContractionRows::Indexed { experts },
                    },
                    inputs,
                    out: g.eqns()[indexed].out,
                    layer: g.eqns()[indexed].layer,
                },
            );
            bound.push(id);
            last_name = Some(name.to_string());
            continue;
        }
        let mut carriers = Vec::new();
        for role in packed.weight.sources() {
            let carrier = out.values_mut().push(ValueMeta::new(
                packed_source_type(packed.weight, role),
                Storage::Const,
                Some(PackedSourceName::new(&packed.linear_id, role).into()),
            ));
            out.inputs_mut().push(carrier);
            out.consts_mut().push(carrier);
            carriers.push(Operand::Value(carrier));
        }
        // The decoded owner, then (projection) its transpose; the last step writes the declared value id
        // itself.
        let dequant_out = match packed.layout {
            PackedLayout::Rows => id,
            PackedLayout::Columns => out.values_mut().push(ValueMeta::new(
                TensorType::f32(vec![rows, k]),
                Storage::Device,
                None,
            )),
            PackedLayout::ExpertRows { .. } => unreachable!("expert stacks continue above"),
        };
        decode.push(Eqn {
            op: OpKind::PackedDequant {
                descriptor: packed.weight,
            },
            inputs: carriers,
            out: dequant_out,
            layer: None,
        });
        if packed.layout == PackedLayout::Columns {
            decode.push(Eqn {
                op: OpKind::Transpose { perm: vec![1, 0] },
                inputs: vec![Operand::Value(dequant_out)],
                out: id,
                layer: None,
            });
        }
        out.values_mut()[id].storage = Storage::Device;
        out.values_mut()[id].name = None;
        bound.push(id);
        last_name = Some(name.to_string());
    }
    if bound.is_empty() {
        return Ok(g.clone());
    }
    out.inputs_mut().retain(|id| !bound.contains(id));
    out.consts_mut().retain(|id| !bound.contains(id));
    let kept = std::mem::take(out.eqns_mut())
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !removed.contains(index))
        .map(|(index, eqn)| claimed.remove(&index).unwrap_or(eqn));
    decode.extend(kept);
    *out.eqns_mut() = decode;
    out.commit().map_err(|source| PackedBindError::Graph {
        name: last_name.unwrap_or_default(),
        source: Box::new(source),
    })
}

/// The two equations that read expert stack `stack`: the `Transpose([0, 2, 1])` that is its only consumer and
/// the `IndexedMatMul` that is the transpose's only consumer (the typing rule makes the rank-3 table its weight
/// operand). Anything else is the
/// stack's refusal.
fn expert_stack_use(
    g: &Graph,
    stack: ValueId,
    consumers: &ValueTable<Vec<usize>>,
) -> Result<(usize, usize), ExpertStackRefusal> {
    let escapes = |value: ValueId| {
        g.liveness_roots().any(|root| root == value)
            || g.state().any(|(_, state_out)| state_out == value)
    };
    let &[transpose] = consumers[stack].as_slice() else {
        return Err(ExpertStackRefusal::Consumers {
            consumers: consumers[stack].len(),
        });
    };
    let transpose_eqn = &g.eqns()[transpose];
    if !matches!(&transpose_eqn.op, OpKind::Transpose { perm } if perm.as_slice() == [0, 2, 1]) {
        return Err(ExpertStackRefusal::NotTransposed {
            op: transpose_eqn.op.name(),
        });
    }
    if escapes(stack) || escapes(transpose_eqn.out) {
        return Err(ExpertStackRefusal::Escapes);
    }
    let &[indexed] = consumers[transpose_eqn.out].as_slice() else {
        return Err(ExpertStackRefusal::TransposeConsumers {
            consumers: consumers[transpose_eqn.out].len(),
        });
    };
    let indexed_eqn = &g.eqns()[indexed];
    if !matches!(indexed_eqn.op, OpKind::IndexedMatMul) {
        return Err(ExpertStackRefusal::NotIndexed {
            op: indexed_eqn.op.name(),
        });
    }
    Ok((transpose, indexed))
}

#[cfg(test)]
mod tests {
    use crate::passes::BinOp;
    use crate::passes::reject_packed_dequant_escapes;
    use crate::{recognize_packed_contractions, recognize_packed_row_gathers};
    use poot_quant::format::WeightFormat;
    use poot_quant::{PackedWeight, SourceRole};

    use super::*;
    use poot_graph_ir::types::TensorType;
    use poot_graph_ir::{Builder, RedOp, Traced};

    fn q8(out: usize, k: usize) -> PackedWeight {
        PackedWeight::try_new(WeightFormat::Q8_0, [out, k]).unwrap()
    }

    fn formats(entries: &[(&str, PackedConst)]) -> WeightFormats {
        let mut formats = WeightFormats::default();
        for (name, packed) in entries {
            formats.insert(name.to_string(), packed.clone()).unwrap();
        }
        formats
    }

    fn column(out: usize, k: usize) -> PackedConst {
        PackedConst {
            linear_id: "layer".into(),
            weight: q8(out, k),
            layout: PackedLayout::Columns,
        }
    }

    fn ops(g: &Graph) -> Vec<String> {
        g.eqns().iter().map(|eqn| eqn.op.name()).collect()
    }

    /// A projection `x @ W` whose `[K, out]` weight is stored packed binds to `packed_linear`'s
    /// canonical chain over role-named carriers (the claim is `compile`'s, not bind's), and that
    /// chain is exactly what the contraction claim takes; the declared dense constant is no longer
    /// an input. Mutation: bind a `Columns` constant without its transpose; the claim finds nothing.
    #[test]
    fn a_packed_projection_binds_to_the_claimable_contraction_chain() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 64])).unwrap();
        let w = b
            .constant("layer.weight", TensorType::f32(vec![64, 8]))
            .unwrap();
        let y = b.matmul(x, w).unwrap();
        let g = b.finish(y).unwrap();
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        assert_eq!(
            ops(&bound),
            ["packed_dequant q8_0", "transpose [1, 0]", "matmul"]
        );
        let names: Vec<&str> = bound
            .inputs()
            .iter()
            .filter_map(|&id| bound.meta(id).name.as_deref())
            .collect();
        assert_eq!(names, ["x", "layer.packed_blocks_source"]);
        let claimed = recognize_packed_contractions(&bound).unwrap();
        assert!(
            matches!(
                claimed.eqns(),
                [Eqn {
                    op: OpKind::PackedContraction { .. },
                    ..
                }]
            ),
            "{:?}",
            ops(&claimed)
        );
    }

    /// A `Model` graph declares a projection weight in checkpoint `[out, K]` orientation and spells
    /// `matmul(x, transpose(w))`; formats read from the `WeightMap` record the packed handle under
    /// its const name, laid out as stored, so binding yields the same claimable chain and a dense
    /// handle is left alone. Mutation: read every handle as dense; nothing binds.
    #[test]
    fn weight_map_formats_bind_a_checkpoint_oriented_projection_to_the_claimable_chain() {
        use poot_quant::weights::{
            AttnRole, DenseWeight, WeightEntry, WeightId, WeightRole, WeightStore, WeightView,
        };
        let (q, k) = (
            WeightId::layer(0, WeightRole::Attn(AttnRole::Q)),
            WeightId::layer(0, WeightRole::Attn(AttnRole::K)),
        );
        let payload =
            poot_quant::test_support::packed::random_payload(WeightFormat::Q8_0, [8, 64], 3);
        let mut store = WeightStore::builder();
        store
            .insert("q", WeightEntry::Packed(std::sync::Arc::new(payload)))
            .unwrap();
        store
            .insert(
                "k",
                WeightEntry::Dense(
                    DenseWeight::try_new(DType::BF16, vec![8, 64], vec![0u8; 8 * 64 * 2].into())
                        .unwrap(),
                ),
            )
            .unwrap();
        let store = store.build();
        let mut map = WeightMap::builder(&store);
        map.map(q, WeightView::Stored("q".into())).unwrap();
        map.map(k, WeightView::Stored("k".into())).unwrap();
        let formats = WeightFormats::from_weight_map(&map.build());
        assert_eq!(
            formats.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            ["w.l0.attn.q"]
        );

        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 64])).unwrap();
        let w = b
            .constant(&q.const_name(), TensorType::f32(vec![8, 64]))
            .unwrap();
        let y = b.matmul(x, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let bound = bind_packed_weights(&b.finish(y).unwrap(), &formats).unwrap();
        assert_eq!(
            ops(&bound),
            ["packed_dequant q8_0", "transpose [1, 0]", "matmul"]
        );
        assert!(matches!(
            recognize_packed_contractions(&bound).unwrap().eqns(),
            [Eqn {
                op: OpKind::PackedContraction { .. },
                ..
            }]
        ));
    }

    /// A traced biased projection over a packed weight (`ops::linear`: the matmul then an ordinary add
    /// of the bias) keeps its bias outside the claimed contraction, the output keeping its value id.
    #[test]
    fn a_biased_packed_projection_keeps_its_bias_outside_the_contraction() {
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 64])).unwrap();
        let w = b
            .constant("layer.weight", TensorType::f32(vec![64, 8]))
            .unwrap();
        let bias = b.constant("layer.bias", TensorType::f32(vec![8])).unwrap();
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias)).unwrap();
        let g = b.finish(y).unwrap();
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        assert_eq!(bound.result(), g.result());
        let claimed = recognize_packed_contractions(&bound).unwrap();
        assert!(
            matches!(
                claimed.eqns(),
                [
                    Eqn {
                        op: OpKind::PackedContraction { .. },
                        ..
                    },
                    Eqn {
                        op: OpKind::Binary(BinOp::Add),
                        ..
                    }
                ]
            ),
            "{:?}",
            ops(&claimed)
        );
    }

    /// Bind only places storage: a packed weight read by anything but a claimable consumer binds,
    /// and the escape gate (`compile`'s) refuses it rather than materializing the whole decode.
    #[test]
    fn a_packed_weight_read_outside_a_contraction_is_left_for_the_gate() {
        let b = Builder::for_tests();
        let w = b
            .constant("layer.weight", TensorType::f32(vec![64, 8]))
            .unwrap();
        let y = b.unary(poot_graph_ir::op::UnOp::Exp, w).unwrap();
        let g = b.finish(y).unwrap();
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        let claimed =
            recognize_packed_row_gathers(&recognize_packed_contractions(&bound).unwrap()).unwrap();
        assert!(reject_packed_dequant_escapes(&claimed).is_err());
    }

    /// An embedding table stored packed binds as the decode itself (no transpose), so the gather
    /// over it is the row-gather claim's shape.
    #[test]
    fn a_packed_embedding_binds_to_the_claimable_row_gather() {
        let b = Builder::for_tests();
        let ids = b
            .slot(
                poot_graph_ir::Slot::Token,
                TensorType::new(vec![3], poot_tensor::DType::I32),
            )
            .unwrap();
        let table = b
            .constant("embed.weight", TensorType::f32(vec![16, 64]))
            .unwrap();
        let rows = b.gather(table, 0, ids).unwrap();
        let g = b.finish(rows).unwrap();
        let bound = bind_packed_weights(
            &g,
            &formats(&[(
                "embed.weight",
                PackedConst {
                    linear_id: "embed".into(),
                    weight: q8(16, 64),
                    layout: PackedLayout::Rows,
                },
            )]),
        )
        .unwrap();
        assert_eq!(ops(&bound), ["packed_dequant q8_0", "gather ax=0"]);
        assert!(matches!(
            recognize_packed_row_gathers(&bound).unwrap().eqns(),
            [Eqn {
                op: OpKind::PackedRowGather { .. },
                ..
            }]
        ));
    }

    /// An expert stack binds to its one consumer, rewritten: the transpose and the `IndexedMatMul` become one
    /// `PackedContraction { rows: Indexed }` over one stacked carrier per source role, with the activation and
    /// ids operands kept and the output value untouched; no `PackedDequant` (no decoded table) remains. A
    /// one-expert stack keeps its `E` axis. Mutation: bind the stack as per-expert `PackedDequant` branches
    /// (the removed W12 chain); the dequant count and the missing contraction fail this row.
    #[test]
    fn a_packed_expert_stack_binds_to_one_indexed_contraction() {
        for experts in [1usize, 2] {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![3, 64])).unwrap();
            let sel = b
                .constant("sel", TensorType::new(vec![3], DType::I32))
                .unwrap();
            let w = b
                .constant("experts.weight", TensorType::f32(vec![experts, 8, 64]))
                .unwrap();
            let y = b
                .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), sel)
                .unwrap();
            let g = b.finish(y).unwrap();
            let bound = bind_packed_weights(&g, &formats(&[("experts.weight", stack(experts))]))
                .unwrap_or_else(|error| panic!("{experts} experts: {error}"));
            reject_packed_dequant_escapes(&bound)
                .unwrap_or_else(|error| panic!("{experts} experts: {error}"));
            assert_eq!(bound.result(), g.result());
            let [eqn] = bound.eqns() else {
                panic!("{experts} experts: {:?}", ops(&bound));
            };
            assert_eq!(
                eqn.op,
                OpKind::PackedContraction {
                    descriptor: q8(8, 64),
                    rows: ContractionRows::Indexed { experts },
                }
            );
            let names: Vec<&str> = bound
                .inputs()
                .iter()
                .filter_map(|&id| bound.meta(id).name.as_deref())
                .collect();
            assert_eq!(names, ["x", "sel", "experts.weight.packed_blocks_source"]);
            let carrier = bound.inputs()[2];
            assert_eq!(
                bound.aval(carrier),
                &TensorType::new(
                    vec![experts, q8(8, 64).slab_bytes(SourceRole::Blocks)],
                    DType::I8
                )
            );
        }
    }

    fn stack(experts: usize) -> PackedConst {
        PackedConst {
            linear_id: "experts.weight".into(),
            weight: q8(8, 64),
            layout: PackedLayout::ExpertRows { experts },
        }
    }

    /// A square stack (`out == K`), so a transpose with another permutation still types.
    fn square_stack() -> PackedConst {
        PackedConst {
            linear_id: "experts.weight".into(),
            weight: q8(64, 64),
            layout: PackedLayout::ExpertRows { experts: 2 },
        }
    }

    /// SC-003 (bind half): an expert stack that is not read by exactly one `Transpose([0, 2, 1])` into the weight
    /// operand of exactly one `IndexedMatMul` is a typed refusal naming the constant and the offending use,
    /// never a fallback to decoding the table. The rows: a transpose with another permutation, a second
    /// consumer of the stack, a second consumer of the transpose, and a transposed table read by something that
    /// is not an indexed matmul. (The table as the matmul's activation or id operand cannot be built: the
    /// typing rule gives only the weight operand rank 3.) Mutation: accept any transpose permutation in the
    /// rewrite; the `perm [0, 1, 2]` row binds and goes red.
    #[test]
    fn a_packed_expert_stack_with_any_other_use_is_refused_by_name() {
        let refusal = |g: &Graph| match bind_packed_weights(
            g,
            &formats(&[("experts.weight", square_stack())]),
        ) {
            Err(PackedBindError::ExpertStack { name, reason }) => {
                assert_eq!(name, "experts.weight");
                reason
            }
            other => panic!(
                "expected an expert-stack refusal, got {:?}",
                other.map(|g| ops(&g))
            ),
        };
        let build = |f: &dyn Fn(&Builder, Traced, Traced, Traced) -> Traced| {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::f32(vec![3, 64])).unwrap();
            let sel = b
                .constant("sel", TensorType::new(vec![3], DType::I32))
                .unwrap();
            let w = b
                .constant("experts.weight", TensorType::f32(vec![2, 64, 64]))
                .unwrap();
            let y = f(&b, x, sel, w);
            b.finish(y).unwrap()
        };
        // No transpose: the stack `[E, out, K]` read where the matmul wants `[E, K, out]`.
        let g = build(&|b, x, sel, w| {
            b.indexed_matmul(x, b.transpose(w, vec![0, 1, 2]).unwrap(), sel)
                .unwrap()
        });
        assert!(matches!(
            refusal(&g),
            ExpertStackRefusal::NotTransposed { .. }
        ));
        // A second consumer of the stack.
        let g = build(&|b, x, sel, w| {
            let y = b
                .indexed_matmul(x, b.transpose(w, vec![0, 2, 1]).unwrap(), sel)
                .unwrap();
            b.concat(0, &[y, b.reduce(RedOp::Sum, w, 0, false).unwrap()])
                .unwrap()
        });
        assert_eq!(refusal(&g), ExpertStackRefusal::Consumers { consumers: 2 });
        // A second consumer of the transposed table.
        let g = build(&|b, x, sel, w| {
            let wt = b.transpose(w, vec![0, 2, 1]).unwrap();
            let y = b.indexed_matmul(x, wt, sel).unwrap();
            b.concat(0, &[y, b.reduce(RedOp::Sum, wt, 0, false).unwrap()])
                .unwrap()
        });
        assert_eq!(
            refusal(&g),
            ExpertStackRefusal::TransposeConsumers { consumers: 2 }
        );
        // A transposed table read by something that is not an indexed matmul.
        let g = build(&|b, _, _, w| {
            b.reduce(RedOp::Sum, b.transpose(w, vec![0, 2, 1]).unwrap(), 1, false)
                .unwrap()
        });
        assert!(matches!(refusal(&g), ExpertStackRefusal::NotIndexed { .. }));
    }

    /// A declared shape or dtype the stored weight cannot produce is a typed refusal naming the
    /// constant.
    #[test]
    fn a_mismatched_declaration_is_refused_by_name() {
        let packed = PackedConst {
            linear_id: "layer".into(),
            weight: q8(8, 64),
            layout: PackedLayout::Columns,
        };
        let b = Builder::for_tests();
        let x = b.constant("x", TensorType::f32(vec![2, 8])).unwrap();
        let w = b
            .constant("layer.weight", TensorType::f32(vec![8, 64]))
            .unwrap();
        let y = b.matmul(x, w).unwrap();
        let g = b.finish(y).unwrap();
        assert!(matches!(
            bind_packed_weights(&g, &formats(&[("layer.weight", packed.clone())])),
            Err(PackedBindError::Shape { ref name, .. }) if name == "layer.weight"
        ));
        let b = Builder::for_tests();
        let x = b
            .constant("x", TensorType::new(vec![2, 64], DType::BF16))
            .unwrap();
        let w = b
            .constant("layer.weight", TensorType::new(vec![64, 8], DType::BF16))
            .unwrap();
        let y = b.matmul(x, w).unwrap();
        let g = b.finish(y).unwrap();
        assert!(matches!(
            bind_packed_weights(&g, &formats(&[("layer.weight", packed)])),
            Err(PackedBindError::Dtype { .. })
        ));
        let mut dup = WeightFormats::default();
        let one = PackedConst {
            linear_id: "a".into(),
            weight: q8(8, 64),
            layout: PackedLayout::Rows,
        };
        dup.insert("a.weight".into(), one.clone()).unwrap();
        assert!(matches!(
            dup.insert("a.weight".into(), one),
            Err(WeightFormatsError::Duplicate { .. })
        ));
    }
}
