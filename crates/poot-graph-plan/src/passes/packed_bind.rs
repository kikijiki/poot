//! Card 545a: the one place a model's stored weight formats reach a graph.
//!
//! Tracers are storage-agnostic: they declare every weight as a logical dense constant (a projection
//! `[K, out]` read by a matmul, an embedding table `[rows, K]` read by a gather, an expert stack
//! `[E, K, out]` read by an indexed matmul). A checkpoint that stores some of those weights packed
//! records, per constant name, which packed owners hold it ([`WeightFormats`]: from a `Model`'s
//! `WeightMap` handles on the driver path ([`WeightFormats::from_weight_map`]), or
//! derived by the Runner's loader from the payload descriptors it reads until the Runner is
//! deleted). [`bind_packed_weights`] rewrites each such constant into the
//! decode of its owners - `PackedDequant` over `descriptor.sources()` carrier constants named by
//! [`PackedSourceName`], transposed for a projection, stacked for experts - producing the very value id
//! the tracer declared, so every consumer is untouched (a biased projection is already `ops::linear`'s
//! matmul then add, so the claim sees the product). Binding only places storage: the claims
//! (`PackedContraction`, `PackedRowGather`, the W12 MoE chain) and the escape gate are `compile`'s
//! (the CPU oracle claims explicitly in `poot-llm`'s `cpu_oracle`). A mismatch between what
//! a tracer declared and what the checkpoint stores is a typed refusal, never a reinterpretation.

use std::collections::BTreeMap;

use super::*;
use poot_graph_ir::packed_source::{PackedSourceName, packed_source_type};
use poot_graph_ir::types::TensorType;
use poot_quant::weights::{HandleFormat, WeightMap};
use poot_tensor::DType;

/// How a packed owner's logical `[out, K]` decode maps onto the constant a tracer declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PackedLayout {
    /// The constant is the decode as stored, `[out, K]` (an embedding table read by a gather).
    Rows,
    /// The constant is the decode transposed, `[K, out]` (a projection read by `x @ W`).
    Columns,
    /// The constant is an expert stack `[E, K, out]`: one owner per expert, each decode transposed
    /// and stacked along a new leading axis (read by an indexed/grouped matmul). One owner is a
    /// one-expert stack, `[1, K, out]`.
    StackedColumns,
}

/// The packed storage of one dense constant a tracer declares.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedConst {
    /// The packed owners holding it, by linear id (the name their source carriers are staged
    /// under): exactly one for [`PackedLayout::Rows`]/[`PackedLayout::Columns`], one per expert for
    /// [`PackedLayout::StackedColumns`].
    pub linear_ids: Vec<String>,
    /// Every owner's descriptor (one shape and format for the whole stack).
    pub weight: poot_quant::PackedWeight,
    pub layout: PackedLayout,
}

impl PackedConst {
    /// The dense shape a tracer must declare for this constant.
    fn declared_shape(&self) -> Vec<usize> {
        let [out, k] = self.weight.shape();
        match self.layout {
            PackedLayout::Rows => vec![out, k],
            PackedLayout::Columns => vec![k, out],
            PackedLayout::StackedColumns => vec![self.linear_ids.len(), k, out],
        }
    }
}

/// Which dense constants of a model are stored packed, keyed by the constant name the tracers
/// declare. Built once by the loader from the payloads it loads; empty for a dense model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WeightFormats {
    consts: BTreeMap<String, PackedConst>,
}

/// A packed-storage record that contradicts another or itself.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WeightFormatsError {
    #[error("constant {name} is recorded packed twice")]
    Duplicate { name: String },
    #[error("constant {name} names no packed owner")]
    NoOwner { name: String },
    #[error("constant {name} is laid out {layout:?}, which holds one owner, but names {owners}")]
    OwnerCount {
        name: String,
        layout: PackedLayout,
        owners: usize,
    },
}

impl WeightFormats {
    /// The packed weights of a model's [`WeightMap`] (the driver path reads formats
    /// from the map, never from a tracer). Each mapped weight whose handle is
    /// [`HandleFormat::Packed`] is recorded under its [`WeightId::const_name`], with its carriers
    /// staged under the same name, laid out [`PackedLayout::Rows`]: a `Model` tracer declares every
    /// weight in checkpoint `[out, K]` orientation and spells a projection's transpose itself.
    ///
    /// [`WeightId::const_name`]: poot_quant::weights::WeightId::const_name
    pub fn from_weight_map(map: &WeightMap) -> Self {
        let consts = map
            .iter()
            .filter_map(|(id, _, handle)| match handle.format {
                HandleFormat::Packed(weight) => Some((
                    id.const_name(),
                    PackedConst {
                        linear_ids: vec![id.const_name()],
                        weight,
                        layout: PackedLayout::Rows,
                    },
                )),
                HandleFormat::Dense(_) => None,
            })
            .collect();
        Self { consts }
    }

    /// Record that constant `name` is stored as `packed`.
    pub fn insert(&mut self, name: String, packed: PackedConst) -> Result<(), WeightFormatsError> {
        if packed.linear_ids.is_empty() {
            return Err(WeightFormatsError::NoOwner { name });
        }
        if packed.layout != PackedLayout::StackedColumns && packed.linear_ids.len() != 1 {
            return Err(WeightFormatsError::OwnerCount {
                name,
                layout: packed.layout,
                owners: packed.linear_ids.len(),
            });
        }
        if self.consts.contains_key(&name) {
            return Err(WeightFormatsError::Duplicate { name });
        }
        self.consts.insert(name, packed);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&PackedConst> {
        self.consts.get(name)
    }

    pub fn is_empty(&self) -> bool {
        self.consts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &PackedConst)> {
        self.consts
            .iter()
            .map(|(name, packed)| (name.as_str(), packed))
    }
}

/// Why a traced graph cannot take a model's packed storage.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum PackedBindError {
    #[error(
        "constant {name} is stored packed ({format:?}) but the graph declares it {dtype:?}; a packed \
         weight decodes to F32"
    )]
    Dtype {
        name: String,
        format: poot_quant::format::WeightFormat,
        dtype: DType,
    },
    #[error(
        "constant {name} is stored packed as {expected:?} ({format:?}) but the graph declares {declared:?}"
    )]
    Shape {
        name: String,
        format: poot_quant::format::WeightFormat,
        declared: Vec<usize>,
        expected: Vec<usize>,
    },
    #[error("binding packed constant {name} produced an invalid graph: {source}")]
    Graph {
        name: String,
        #[source]
        source: Box<GraphValidationError>,
    },
}

/// Rewrite every constant of `g` that `formats` records packed into the decode of its owners (see
/// the module docs). Constants `formats` does not name are untouched, so a dense model's graph is
/// returned unchanged.
pub fn bind_packed_weights<V: ValidationChannel>(
    g: &Graph<V>,
    formats: &WeightFormats,
) -> Result<Graph<V>, PackedBindError> {
    if formats.is_empty() {
        return Ok(g.clone());
    }
    let mut out = g.clone();
    let mut decode: Vec<Eqn> = Vec::new();
    let mut bound: Vec<ValueId> = Vec::new();
    let mut last_name = None;
    for &id in &g.inputs {
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
        let stacked = packed.layout == PackedLayout::StackedColumns;
        fn push_value<V: ValidationChannel>(out: &mut Graph<V>, aval: TensorType) -> ValueId {
            out.values.push(ValueMeta::new(aval, Storage::Device, None));
            out.values.len() - 1
        }
        let mut branches = Vec::with_capacity(packed.linear_ids.len());
        for linear_id in &packed.linear_ids {
            let mut carriers = Vec::new();
            for role in packed.weight.sources() {
                out.values.push(ValueMeta::new(
                    packed_source_type(packed.weight, role),
                    Storage::Const,
                    Some(PackedSourceName::new(linear_id, role).into()),
                ));
                let carrier = out.values.len() - 1;
                out.inputs.push(carrier);
                out.consts.push(carrier);
                carriers.push(Operand::Value(carrier));
            }
            // The decoded owner, then (projection) its transpose, then (expert stack) a leading
            // unit axis; a single weight's last step writes the declared value id itself.
            let mut steps: Vec<(OpKind, TensorType)> = vec![(
                OpKind::PackedDequant {
                    descriptor: packed.weight,
                },
                TensorType::f32(vec![rows, k]),
            )];
            let slice_shape = match packed.layout {
                PackedLayout::Rows => vec![rows, k],
                PackedLayout::Columns | PackedLayout::StackedColumns => {
                    steps.push((
                        OpKind::Transpose { perm: vec![1, 0] },
                        TensorType::f32(vec![k, rows]),
                    ));
                    vec![k, rows]
                }
            };
            if stacked {
                let mut unit = vec![1];
                unit.extend_from_slice(&slice_shape);
                steps.push((
                    OpKind::Reshape {
                        shape: unit.clone(),
                    },
                    TensorType::f32(unit),
                ));
            }
            let mut inputs = carriers;
            let last = steps.len() - 1;
            for (index, (op, aval)) in steps.into_iter().enumerate() {
                let value = if index == last && !stacked {
                    id
                } else {
                    push_value(&mut out, aval)
                };
                decode.push(Eqn {
                    op,
                    inputs,
                    out: value,
                    layer: None,
                });
                inputs = vec![Operand::Value(value)];
            }
            let Operand::Value(branch) = inputs[0] else {
                unreachable!("every decode step produces a value")
            };
            branches.push(Operand::Value(branch));
        }
        if stacked {
            decode.push(Eqn {
                op: OpKind::Concat { axis: 0 },
                inputs: branches,
                out: id,
                layer: None,
            });
        }
        out.values[id].storage = Storage::Device;
        out.values[id].name = None;
        bound.push(id);
        last_name = Some(name.to_string());
    }
    if bound.is_empty() {
        return Ok(out);
    }
    out.inputs.retain(|id| !bound.contains(id));
    out.consts.retain(|id| !bound.contains(id));
    decode.append(&mut out.eqns);
    out.eqns = decode;
    out.validate().map_err(|source| PackedBindError::Graph {
        name: last_name.unwrap_or_default(),
        source: Box::new(source),
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use poot_quant::PackedWeight;
    use poot_quant::format::WeightFormat;

    use super::*;
    use poot_graph_ir::Builder;
    use poot_graph_ir::types::TensorType;

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
            linear_ids: vec!["layer".into()],
            weight: q8(out, k),
            layout: PackedLayout::Columns,
        }
    }

    fn ops(g: &Graph) -> Vec<String> {
        g.eqns.iter().map(|eqn| eqn.op.name()).collect()
    }

    /// A projection `x @ W` whose `[K, out]` weight is stored packed binds to `packed_linear`'s
    /// canonical chain over role-named carriers (the claim is `compile`'s, not bind's), and that
    /// chain is exactly what the contraction claim takes; the declared dense constant is no longer
    /// an input. Mutation: bind a `Columns` constant without its transpose; the claim finds nothing.
    #[test]
    fn a_packed_projection_binds_to_the_claimable_contraction_chain() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 64]));
        let w = b.constant("layer.weight", TensorType::f32(vec![64, 8]));
        let y = b.matmul(x, w);
        let g = b.finish(y);
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        assert_eq!(
            ops(&bound),
            ["packed_dequant q8_0", "transpose [1, 0]", "matmul"]
        );
        let names: Vec<&str> = bound
            .inputs
            .iter()
            .filter_map(|&id| bound.meta(id).name.as_deref())
            .collect();
        assert_eq!(names, ["x", "layer.packed_blocks_source"]);
        let claimed = recognize_packed_contractions(&bound);
        assert!(
            matches!(
                claimed.eqns.as_slice(),
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
        let payload = poot_test_util::packed::random_payload(WeightFormat::Q8_0, [8, 64], 3);
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

        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 64]));
        let w = b.constant(&q.const_name(), TensorType::f32(vec![8, 64]));
        let y = b.matmul(x, b.transpose(w, vec![1, 0]));
        let bound = bind_packed_weights(&b.finish(y), &formats).unwrap();
        assert_eq!(
            ops(&bound),
            ["packed_dequant q8_0", "transpose [1, 0]", "matmul"]
        );
        assert!(matches!(
            recognize_packed_contractions(&bound).eqns.as_slice(),
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
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 64]));
        let w = b.constant("layer.weight", TensorType::f32(vec![64, 8]));
        let bias = b.constant("layer.bias", TensorType::f32(vec![8]));
        let y = poot_graph_ir::ops::linear(&b, x, w, Some(bias));
        let g = b.finish(y);
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        assert_eq!(bound.output, g.output);
        let claimed = recognize_packed_contractions(&bound);
        assert!(
            matches!(
                claimed.eqns.as_slice(),
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
        let b = Builder::new();
        let w = b.constant("layer.weight", TensorType::f32(vec![64, 8]));
        let y = b.unary(poot_graph_ir::op::UnOp::Exp, w);
        let g = b.finish(y);
        let bound = bind_packed_weights(&g, &formats(&[("layer.weight", column(8, 64))])).unwrap();
        let claimed = recognize_packed_row_gathers(&recognize_packed_contractions(&bound));
        assert!(reject_packed_dequant_escapes(&claimed).is_err());
    }

    /// An embedding table stored packed binds as the decode itself (no transpose), so the gather
    /// over it is the row-gather claim's shape.
    #[test]
    fn a_packed_embedding_binds_to_the_claimable_row_gather() {
        let b = Builder::new();
        let ids = b.slot(poot_graph_ir::Slot::Token, TensorType::f32(vec![3]));
        let table = b.constant("embed.weight", TensorType::f32(vec![16, 64]));
        let rows = b.gather(table, 0, ids);
        let g = b.finish(rows);
        let bound = bind_packed_weights(
            &g,
            &formats(&[(
                "embed.weight",
                PackedConst {
                    linear_ids: vec!["embed".into()],
                    weight: q8(16, 64),
                    layout: PackedLayout::Rows,
                },
            )]),
        )
        .unwrap();
        assert_eq!(ops(&bound), ["packed_dequant q8_0", "gather ax=0"]);
        assert!(matches!(
            recognize_packed_row_gathers(&bound).eqns.as_slice(),
            [Eqn {
                op: OpKind::PackedRowGather { .. },
                ..
            }]
        ));
    }

    /// An expert stack binds as one decode branch per owner, stacked: the W12 canonical chain the
    /// escape gate admits in front of an `IndexedMatMul`. A one-expert stack keeps its `E` axis
    /// (`StackedColumns` states stacked-ness; the owner count does not).
    #[test]
    fn a_packed_expert_stack_binds_to_the_canonical_moe_chain() {
        for experts in [1usize, 2] {
            let b = Builder::new();
            let x = b.constant("x", TensorType::f32(vec![3, 64]));
            let sel = b.constant("sel", TensorType::f32(vec![3]));
            let w = b.constant("experts.weight", TensorType::f32(vec![experts, 64, 8]));
            let y = b.indexed_matmul(x, w, sel);
            let g = b.finish(y);
            let bound = bind_packed_weights(
                &g,
                &formats(&[(
                    "experts.weight",
                    PackedConst {
                        linear_ids: (0..experts).map(|e| format!("experts.{e}")).collect(),
                        weight: q8(8, 64),
                        layout: PackedLayout::StackedColumns,
                    },
                )]),
            )
            .unwrap_or_else(|error| panic!("{experts} experts: {error}"));
            reject_packed_dequant_escapes(&bound)
                .unwrap_or_else(|error| panic!("{experts} experts: {error}"));
            assert_eq!(
                bound
                    .eqns
                    .iter()
                    .filter(|eqn| matches!(eqn.op, OpKind::PackedDequant { .. }))
                    .count(),
                experts
            );
        }
    }

    /// A declared shape or dtype the stored weight cannot produce is a typed refusal naming the
    /// constant.
    #[test]
    fn a_mismatched_declaration_is_refused_by_name() {
        let packed = PackedConst {
            linear_ids: vec!["layer".into()],
            weight: q8(8, 64),
            layout: PackedLayout::Columns,
        };
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![2, 8]));
        let w = b.constant("layer.weight", TensorType::f32(vec![8, 64]));
        let y = b.matmul(x, w);
        let g = b.finish(y);
        assert!(matches!(
            bind_packed_weights(&g, &formats(&[("layer.weight", packed.clone())])),
            Err(PackedBindError::Shape { ref name, .. }) if name == "layer.weight"
        ));
        let b = Builder::new();
        let x = b.constant("x", TensorType::new(vec![2, 64], DType::BF16));
        let w = b.constant("layer.weight", TensorType::new(vec![64, 8], DType::BF16));
        let y = b.matmul(x, w);
        let g = b.finish(y);
        assert!(matches!(
            bind_packed_weights(&g, &formats(&[("layer.weight", packed)])),
            Err(PackedBindError::Dtype { .. })
        ));
        let mut dup = WeightFormats::default();
        let one = PackedConst {
            linear_ids: vec!["a".into()],
            weight: q8(8, 64),
            layout: PackedLayout::Rows,
        };
        dup.insert("a.weight".into(), one.clone()).unwrap();
        assert!(matches!(
            dup.insert("a.weight".into(), one),
            Err(WeightFormatsError::Duplicate { .. })
        ));
        // A single-owner layout naming two owners is refused (stacked-ness is the layout's).
        let two = PackedConst {
            linear_ids: vec!["b.0".into(), "b.1".into()],
            weight: q8(8, 64),
            layout: PackedLayout::Columns,
        };
        assert!(matches!(
            WeightFormats::default().insert("b.weight".into(), two),
            Err(WeightFormatsError::OwnerCount { owners: 2, .. })
        ));
    }
}
