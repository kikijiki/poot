//! Whole-graph storage representation: every value's physical storage, read from one plan per equation.

use crate::predicates::GraphTables;
use crate::refusal::refusal_at;
use crate::types::{GraphStorage, StorageKind, ValueStorage};
use poot_kernelgen::Layout;
use poot_tensor::DType;
use std::collections::{HashMap, HashSet};

use poot_graph_ir::Storage;
use poot_target::TargetId;

use crate::dtype_widen::bf16_const_is_packed;
use crate::passes::{KvExtent, kv_extents};
use crate::{Capability, Plan, PlanError, Source, StorageGap, Target, compute_views, value_ids};
use poot_graph_ir::op::OpKind;
use poot_graph_ir::{Eqn, Graph, Operand, ValueId};

/// Card 526: every value's physical storage (dtype and packing) for one graph on one backend,
/// computed once for the whole graph. Folds in the packed-vs-native distinction of a BF16 const each
/// binder used to re-derive separately:
/// `poot-wgpu-device`, `poot-rocm-device` and `poot-ptx-device`'s binders read the result instead.
///
/// Call once per (graph, backend) after `compile`'s dtype passes have run; a const's storage dtype is its
/// declared dtype (Card 1011: no pass retypes a const).
///
/// A value's declared dtype and storage `Const`/`Device`/`Slot`/`State` residency, both already on
/// `poot_graph_ir::ValueMeta`, are not restated here.
///
/// A value's storage is its declared dtype (POOT-725): an I32 value is stored as I32 words, so there is no
/// per-consumer lane to choose. The plans are read for two checks the declared dtype cannot make alone: a BF16
/// const's readers must have the lane the storage plan gave it, and no kernel may read an I32 value through
/// a `Slice<f32>` param ([`PlanError::I32ReadAsF32`]).
///
/// Every value's storage, read from one plan per equation (in `g.eqns` order; `None` for an equation that
/// did not plan, which constrains no operand). A pure function of the graph and the plans:
/// it plans nothing.
///
/// Also refuses a dense-contraction weight stored in a layout of its own (packed F16, Card 1007) that another
/// reader shares ([`StorageGap::F16PackedLanes`]).
pub(crate) fn value_storage_of_plans<'p>(
    g: &Graph,
    target: &Target,
    plans: impl Iterator<Item = Option<&'p Plan>>,
) -> Result<GraphStorage, PlanError> {
    let plans: Vec<Option<&Plan>> = plans.collect();
    check_i32_reads(g, &plans)?;
    let mut weights = dense_contraction_weight_storages(g)?;
    weights.extend(e4m3_row_gather_table_storages(g)?);
    let views = compute_views(g, target.id());
    let storage = storage_of_kinds(g, target, &views, &weights);
    check_bf16_const_reads(g, target.id(), &storage, &plans)?;
    Ok(storage)
}

/// POOT-1229: no contiguous KV cache of `g` stores a head stride that camps on `target`'s DRAM channels unless the
/// plan records why ([`KvExemption`]). Stored KV extent is the legalizer's decision from the device's channel
/// data; a graph that reaches the planner with a camping cache the legalizer could have padded is a planning
/// bug, refused here by name. A target with no channel data has nothing to camp on.
pub(crate) fn check_kv_head_strides(g: &Graph, target: &Target) -> Result<(), PlanError> {
    let Target::Gpu { caps, .. } = target else {
        return Ok(());
    };
    let Some(channels) = caps.dram_channels else {
        return Ok(());
    };
    for (state, extent) in kv_extents(g, channels) {
        if extent == KvExtent::Camps {
            let shape = &g.aval(state).shape;
            return Err(PlanError::KvHeadStrideCamps {
                state,
                stride_bytes: shape[2] * shape[3] * size_of::<f32>(),
            });
        }
    }
    Ok(())
}

/// Card 1011: a BF16 const is never widened on upload, so the kernel param each reader binds it to must be
/// the lane the storage plan gave it: `u32` words for a packed const ([`bf16_const_is_packed`]), BF16
/// elements for a native one. A reader whose body has no such lane (a plain `Gather` over a packed table on
/// wgpu, a packed-lane contraction sharing a native-lane weight) is refused by name, naming the equation,
/// instead of reading the buffer at the wrong element width or failing later in codegen.
///
/// Each dispatch is read through its own binding: its data param `p` reads the value operand
/// `binding.reads()[p]` names (data params come first with or without a metadata buffer, which rides after
/// them). A read bound to scratch has no graph value and is skipped.
/// `bf16_const_reads_refuse_a_reader_without_the_lane` pins the refusal on a real reader.
fn check_bf16_const_reads(
    g: &Graph,
    target: TargetId,
    storage: &GraphStorage,
    plans: &[Option<&Plan>],
) -> Result<(), PlanError> {
    let mut roots: HashMap<ValueId, ValueId> = HashMap::new();
    for (eqn, plan) in g.eqns().iter().zip(plans) {
        if let Some(Plan::Alias(src) | Plan::View { src, .. }) = plan {
            let root = roots.get(src).copied().unwrap_or(*src);
            roots.insert(eqn.out, root);
            continue;
        }
        let operands = value_ids(eqn);
        for (body, vid, param) in plan.iter().flat_map(|plan| operand_reads(plan, &operands)) {
            let root = roots.get(&vid).copied().unwrap_or(vid);
            if !g.meta(root).storage.holds_weight_bytes() || g.aval(root).dtype != DType::BF16 {
                continue;
            }
            let want = if storage.storage(root).buffer_storage()
                == poot_target::BufferStorage::bf16_packed()
            {
                poot_kernel_ir::Ty::U32
            } else {
                poot_kernel_ir::Ty::BF16
            };
            if slice_elem(body.local_ty(param)).is_some_and(|elem| *elem != want) {
                return Err(refusal_at(
                    GraphTables::from(g),
                    eqn,
                    target,
                    Capability::DtypeLowering,
                ));
            }
        }
    }
    Ok(())
}

fn storage_of_kinds(
    g: &Graph,
    target: &Target,
    views: &HashMap<ValueId, Layout>,
    weights: &HashMap<ValueId, StorageKind>,
) -> GraphStorage {
    let values = g
        .values()
        .ids()
        .map(|id| {
            ValueStorage::new(match weights.get(&id) {
                Some(&kind) => kind,
                None => value_storage_kind(g, views, id, target),
            })
        })
        .collect();
    GraphStorage::new(values)
}

/// The storage a `DenseContraction` weight of `dtype` is read in by the generated `[N, K]` bodies (Card 645,
/// Card 1007): F32 as stored, F16 packed two elements per `u32` word on every backend. The planner names this
/// storage in each request, and the value's planned storage is this same answer
/// ([`dense_contraction_weight_storages`]), so a body and its weight buffer cannot disagree. `None` for BF16,
/// whose imported packed-lane kernels take the BF16 const storage rule ([`value_storage_kind`]).
pub(crate) fn dense_contraction_weight_storage(dtype: DType) -> Option<StorageKind> {
    match dtype {
        DType::F32 => Some(StorageKind::Dense(DType::F32)),
        DType::F16 => Some(StorageKind::F16Packed),
        _ => None,
    }
}

/// Every dense-contraction weight whose planned storage differs from its dtype's dense default (a packed F16
/// weight, Card 1007), with that storage. A buffer has one storage, so such a weight must be a non-escaping const
/// that dense contractions are the only readers of, at their weight operand, all in that same storage: anything
/// else is [`PlanError::UnrepresentableValue`] with [`StorageGap::F16PackedLanes`], naming the value. The fold
/// (`fold_dense_contractions`) only produces F16 contractions that satisfy this, so a refusal here is a graph
/// some other path built.
fn dense_contraction_weight_storages(
    g: &Graph,
) -> Result<HashMap<ValueId, StorageKind>, PlanError> {
    let reads = |eqn: &Eqn| match (eqn.op.dense_weight(), eqn.inputs.get(1)) {
        (Some(weight), Some(Operand::Value(value))) => dense_contraction_weight_storage(weight)
            .filter(|&kind| kind != StorageKind::Dense(weight))
            .map(|kind| (*value, kind)),
        _ => None,
    };
    let weights: HashMap<ValueId, StorageKind> = g.eqns().iter().filter_map(reads).collect();
    for (&value, &kind) in &weights {
        let sole_reader = g.eqns().iter().all(|eqn| {
            eqn.inputs.iter().enumerate().all(|(position, operand)| {
                !matches!(operand, Operand::Value(read) if *read == value)
                    || (position == 1 && reads(eqn) == Some((value, kind)))
            })
        });
        if g.meta(value).storage != Storage::Const
            || g.liveness_roots().any(|root| root == value)
            || !sole_reader
        {
            return Err(PlanError::UnrepresentableValue {
                value,
                dtype: g.aval(value).dtype,
                gap: StorageGap::F16PackedLanes,
            });
        }
    }
    Ok(weights)
}

/// Every E4M3FN table a `DenseRowGather` reads, with the packed lane that gather's body decodes
/// ([`StorageKind::E4m3Packed`], POOT-1060). A buffer has one storage, so such a table must be a non-escaping
/// const whose every reader is a `DenseRowGather` at its table operand: anything else is
/// [`PlanError::UnrepresentableValue`] with [`StorageGap::E4m3PackedLanes`], naming the table. An E4M3FN value
/// no row gather reads keeps its dense record, which no executor binds (the engine refuses it by name).
fn e4m3_row_gather_table_storages(g: &Graph) -> Result<HashMap<ValueId, StorageKind>, PlanError> {
    let gathers = |eqn: &Eqn| match (&eqn.op, eqn.inputs.first()) {
        (
            OpKind::DenseRowGather {
                source: DType::E4M3FN,
            },
            Some(Operand::Value(table)),
        ) => Some(*table),
        _ => None,
    };
    let tables: HashSet<ValueId> = g.eqns().iter().filter_map(gathers).collect();
    for &table in &tables {
        let sole_reader = g.eqns().iter().all(|eqn| {
            eqn.inputs.iter().enumerate().all(|(position, operand)| {
                !matches!(operand, Operand::Value(read) if *read == table)
                    || (position == 0 && gathers(eqn) == Some(table))
            })
        });
        if g.meta(table).storage != Storage::Const
            || g.liveness_roots().any(|root| root == table)
            || !sole_reader
        {
            return Err(PlanError::UnrepresentableValue {
                value: table,
                dtype: DType::E4M3FN,
                gap: StorageGap::E4m3PackedLanes,
            });
        }
    }
    Ok(tables
        .into_iter()
        .map(|table| (table, StorageKind::E4m3Packed))
        .collect())
}

/// The consistency check on I32 reads (POOT-725, the structural fix of POOT-643's class): an I32 value's
/// buffer holds I32 words, so a kernel that reads it through a `Slice<f32>` param would read reinterpreted
/// bits. The planner never produces such a body; a plan that does is refused here, naming the buffer and
/// the equation that reads it.
///
/// The check reads buffers, not values: a [`Plan::Alias`] or [`Plan::View`] output shares its source's
/// buffer, so a read of it is a read of the source's bytes. Each read is keyed by its root buffer (the value
/// an alias/view chain bottoms out at). Data-input params come first, with or without a metadata buffer (it
/// rides after them): each dispatch of a plan is scanned through its own binding.
fn check_i32_reads(g: &Graph, plans: &[Option<&Plan>]) -> Result<(), PlanError> {
    // Each alias/view output's root buffer. Equations are topological, so a source's root is known
    // before any alias of it, and before any reader of that alias.
    let mut roots: HashMap<ValueId, ValueId> = HashMap::new();
    for (eqn, plan) in g.eqns().iter().zip(plans) {
        if let Some(Plan::Alias(src) | Plan::View { src, .. }) = plan {
            let root = roots.get(src).copied().unwrap_or(*src);
            roots.insert(eqn.out, root);
            continue;
        }
        let operands = value_ids(eqn);
        for (body, vid, param) in plan.iter().flat_map(|plan| operand_reads(plan, &operands)) {
            let root = roots.get(&vid).copied().unwrap_or(vid);
            if g.aval(root).dtype == DType::I32
                && matches!(
                    slice_elem(body.local_ty(param)),
                    Some(poot_kernel_ir::Ty::F32)
                )
            {
                return Err(PlanError::I32ReadAsF32 {
                    value: root,
                    reader: eqn.out,
                });
            }
        }
    }
    Ok(())
}

/// Every operand read of `plan`, per dispatch: the dispatch's body, the graph value its binding names (an
/// entry of `operands`, the equation's value ids) and the kernel param that reads it. A read bound to
/// scratch has no graph value and is not listed.
fn operand_reads<'a>(
    plan: &'a Plan,
    operands: &'a [ValueId],
) -> impl Iterator<Item = (&'a poot_kernel_ir::Body, ValueId, poot_kernel_ir::Local)> + 'a {
    plan.dispatches().iter().flat_map(move |dispatch| {
        let params: Vec<_> = dispatch.body().params().collect();
        dispatch
            .binding()
            .reads()
            .iter()
            .enumerate()
            .filter_map(move |(position, source)| match *source {
                Source::Operand(operand) => Some((
                    dispatch.body(),
                    *operands.get(operand)?,
                    *params.get(position)?,
                )),
                Source::Scratch(_) => None,
            })
    })
}

/// A kernel param's slice element type (`Slice<T>` or `&[mut] Slice<T>`).
fn slice_elem(ty: &poot_kernel_ir::Ty) -> Option<&poot_kernel_ir::Ty> {
    use poot_kernel_ir::Ty;
    let inner = match ty {
        Ty::Ref { pointee, .. } => pointee.as_ref(),
        other => other,
    };
    match inner {
        Ty::Slice(elem) => Some(elem.as_ref()),
        _ => None,
    }
}

/// One value's storage kind. A value keeps its declared dtype, densely stored, except a BF16 const: it is never
/// widened (Card 1011), it is BF16 on every backend, and [`bf16_const_is_packed`] decides whether the buffer
/// is packed two-per-`u32` word or native two-byte bf16, the one answer the cast planner reads too.
fn value_storage_kind(
    g: &Graph,
    views: &HashMap<ValueId, Layout>,
    id: ValueId,
    target: &Target,
) -> StorageKind {
    let declared = g.aval(id).dtype;
    if !g.meta(id).storage.holds_weight_bytes() || declared != DType::BF16 {
        return StorageKind::Dense(declared);
    }
    if bf16_const_is_packed(g, views, id, target) {
        StorageKind::Bf16Packed
    } else {
        StorageKind::Dense(DType::BF16)
    }
}

#[cfg(test)]
mod value_storage_tests {
    use crate::Target;
    use crate::staged::{
        DeviceId, DevicePlacement, ExpertPlacement, Partition, TargetSet, compile_staged,
    };
    use poot_graph_ir::{Builder, Graph, OpKind, Storage, TensorType};
    use poot_target::{AmdArch, Backend, DeviceCaps};
    use poot_tensor::DType;

    use super::{GraphStorage, Plan};
    use crate::planner::plan_eqn;

    /// The public `plan_value_storage` entry point was deleted (Card 626; its only callers were the
    /// pre-`compile` executors); `compile`'s live path is [`super::value_storage_of_plans`], which
    /// takes pre-planned equations rather than planning them itself. This test-only wrapper restores
    /// the "plan everything, then derive storage" shape these fixtures want.
    fn fixture_value_storage(g: &Graph, backend: Backend, caps: &DeviceCaps) -> GraphStorage {
        fixture_value_storage_result(g, backend, caps)
            .expect("fixture graph has no I32 reader lane conflict")
    }

    /// [`fixture_value_storage`], keeping the refusal a graph with an unrepresentable value gets.
    fn fixture_value_storage_result(
        g: &Graph,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<GraphStorage, crate::PlanError> {
        let plans: Vec<Option<Plan>> = g
            .eqns()
            .iter()
            .map(|eqn| {
                plan_eqn(
                    g,
                    eqn,
                    backend,
                    caps,
                    &crate::test_support::roomy_body_limits(),
                )
                .ok()
            })
            .collect();
        super::value_storage_of_plans(
            g,
            &Target::Gpu {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
            plans.iter().map(Option::as_ref),
        )
    }

    /// `a x w^T -> f32` over two consts of the given dtypes, `m x 32` against a checkpoint-order `[32, 32]` weight;
    /// mirrors `dtype_widen::tests::checkpoint_matmul`.
    fn checkpoint_matmul(m: usize, a_dtype: DType, w_dtype: DType) -> Graph {
        let b = Builder::for_tests();
        let a = b
            .constant("a", TensorType::new(vec![m, 32], a_dtype))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![32, 32], w_dtype))
            .unwrap();
        let wt = b.transpose(w, vec![1, 0]).unwrap();
        let mm = b.matmul(a, wt).unwrap();
        let g = b.finish(mm).unwrap();
        let out = g
            .eqns()
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .expect("matmul eqn")
            .out;
        poot_graph_ir::test_support::graph_fixtures::derive(&g, |draft| {
            draft.values_mut()[out].aval.dtype = DType::F32;
        })
        .unwrap()
    }

    /// Card 526, Card 1011: the compiled program's per-value storage dtype is the const's declared dtype for every
    /// const of every dense fixture graph, on every backend and AMD arch family: a BF16 const is planned BF16
    /// (packed or native), never F32. This is the general record every binder reads.
    ///
    /// Mutation: have `value_storage_kind` return `StorageKind::Dense(DType::F32)` for a BF16 const; the BF16
    /// rows go red naming the const.
    #[test]
    fn program_storage_dtype_is_the_declared_dtype() {
        let backends = [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::AmdGcn(AmdArch::new("gfx90a", 64)),
            Backend::AmdGcn(AmdArch::new("gfx1030", 32)),
            Backend::AmdGcn(AmdArch::new("gfx1200", 32)),
        ];
        // (label, graph, whether every target must plan it)
        let graphs = [
            (
                "mixed bf16 prefill",
                checkpoint_matmul(32, DType::BF16, DType::BF16),
                false,
            ),
            (
                "mixed bf16 decode",
                checkpoint_matmul(1, DType::BF16, DType::BF16),
                false,
            ),
            (
                "bf16 checkpoint prefill",
                checkpoint_matmul(32, DType::F32, DType::BF16),
                true,
            ),
            (
                "bf16 checkpoint decode",
                checkpoint_matmul(1, DType::F32, DType::BF16),
                true,
            ),
        ];
        for backend in backends {
            for (label, g, must_plan) in &graphs {
                let target = crate::Target::Gpu {
                    backend,
                    caps: poot_test_util::device_caps::default_caps_for(backend),
                };
                let options = crate::CompileOptions::new(crate::CompileLimits::STANDARD);
                let program = match crate::compile(g, &target, &options) {
                    Ok(program) => program,
                    Err(crate::CompileError::Plan(error))
                        if !must_plan && matches!(*error, crate::PlanError::Refused(_)) =>
                    {
                        continue;
                    }
                    Err(error) => panic!("{label} on {backend:?}: {error}"),
                };
                let prepared = program.graph();
                let storage = program.storage();
                for id in prepared
                    .values()
                    .ids()
                    .filter(|&id| prepared.meta(id).storage == Storage::Const)
                {
                    let want = g.aval(id).dtype;
                    let got = storage.storage(id).dtype();
                    assert_eq!(
                        got, want,
                        "{label} on {backend:?} {id}: the program's storage says {got:?}, the const is declared {want:?}"
                    );
                }
            }
        }
    }

    /// One graph holding every shape of I32 consumer that used to choose a lane: a computed `Gather` index
    /// (`Binary(GeU)`), a `Slot::WriteMap` index feeding `ScatterUpdate`'s `slots` operand, and a
    /// `Slot::Token` feeding a plain embedding `Gather` (the shape `poot-wgpu-device`'s
    /// `qwen2_decode_gpu_resident_matches_cpu` exercises). Returns `(graph, gather_index_value_id,
    /// slotmap_value_id, token_value_id)`.
    fn gather_index_slotmap_and_token_graph() -> (
        Graph,
        poot_graph_ir::ValueId,
        poot_graph_ir::ValueId,
        poot_graph_ir::ValueId,
    ) {
        use poot_graph_ir::Slot;
        use poot_graph_ir::op::BinOp;
        let b = Builder::for_tests();

        let (rows, cols) = (8usize, 3usize);
        let idx_a = b
            .constant("idx_a", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let idx_b = b
            .constant("idx_b", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let idx = b.binary(BinOp::GeU, idx_a, idx_b).unwrap();
        let gather_table = b
            .constant(
                "gather_table",
                TensorType::new(vec![rows, cols], DType::F32),
            )
            .unwrap();
        let _gathered = b.gather(gather_table, 0, idx).unwrap();

        const POOL: usize = 4;
        const REST: usize = 2;
        const NSRC: usize = 2;
        let base = b
            .constant("base", TensorType::new(vec![POOL, REST], DType::F32))
            .unwrap();
        let src = b
            .constant("src", TensorType::new(vec![NSRC, REST], DType::F32))
            .unwrap();
        let slot_map = b.write_map("write", NSRC, POOL).unwrap();
        let _scatter = b.scatter_update(base, src, slot_map).unwrap();

        const VOCAB: usize = 8;
        const HIDDEN: usize = 4;
        let embed = b
            .constant("embed", TensorType::new(vec![VOCAB, HIDDEN], DType::F32))
            .unwrap();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![1], DType::I32))
            .unwrap();
        let out = b.gather_scalar(embed, 0, token).unwrap();

        let gather_index_id = idx.id();
        let slot_id = slot_map.id();
        let token_id = token.id();
        (b.finish(out).unwrap(), gather_index_id, slot_id, token_id)
    }

    /// POOT-725: every declared-I32 value is stored as dense I32 words, whatever reads it: a computed `Gather`
    /// index, a `Slot::WriteMap` feeding `ScatterUpdate`'s `slots`, and a `Slot::Token` feeding an embedding
    /// `Gather`, in the same graph, on every backend. There is no F32 mirror to choose.
    ///
    /// Mutation: make `value_storage_kind` return `StorageKind::Dense(DType::F32)` for every I32 value; each
    /// row goes red naming the value.
    #[test]
    fn fixture_value_storage_keeps_every_i32_value_dense_i32() {
        let (g, gather_index_id, slot_id, token_id) = gather_index_slotmap_and_token_graph();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            let storage = fixture_value_storage(&g, backend, &caps);
            for (what, id) in [
                ("a computed Gather index", gather_index_id),
                ("a WriteMap feeding ScatterUpdate's slots", slot_id),
                ("a Token feeding an embedding Gather", token_id),
            ] {
                assert_eq!(
                    storage.storage(id).buffer_storage(),
                    poot_target::BufferStorage::i32(),
                    "{backend:?}: {what} ({id}) must be dense I32 words"
                );
            }
        }
    }

    /// POOT-725 SC-004 (the consistency check, POOT-643's class): a plan that binds an I32 value to a
    /// `Slice<f32>` param is a typed [`crate::PlanError::I32ReadAsF32`] naming the buffer and the reading
    /// equation, through the production storage entry `value_storage_of_plans` that `compile` calls.
    ///
    /// The plan is hand-built from the planner's own body for the F32 twin of the equation: `y = x + 1` over an
    /// F32 `x` plans a `Slice<f32>` body, which is then offered as the plan of `y = x + 1` over an I32 `x`.
    /// The honest I32 plan of the same equation (the planner's own) is accepted.
    ///
    /// Mutation: delete the `check_i32_reads` call in `value_storage_of_plans`; the F32 body is accepted and
    /// this row goes red.
    #[test]
    fn an_i32_value_read_through_a_slice_f32_param_is_a_typed_plan_error() {
        use poot_graph_ir::Scalar;
        use poot_graph_ir::op::BinOp;

        let build = |dtype: DType| {
            let b = Builder::for_tests();
            let x = b.constant("x", TensorType::new(vec![4], dtype)).unwrap();
            let scalar = match dtype {
                DType::I32 => Scalar::I32(1),
                _ => Scalar::F32(1.0),
            };
            let y = b.binary_scalar(BinOp::Add, x, scalar).unwrap();
            (b.finish(y).unwrap(), x.id(), y.id())
        };
        let (f32_graph, ..) = build(DType::F32);
        let (i32_graph, x, y) = build(DType::I32);
        let backend = Backend::SpirvVulkan;
        let caps = poot_test_util::device_caps::default_caps_for(backend);
        let limits = crate::test_support::roomy_body_limits();
        let plan_of = |g: &Graph| plan_eqn(g, &g.eqns()[0], backend, &caps, &limits).unwrap();

        let honest = plan_of(&i32_graph);
        super::value_storage_of_plans(
            &i32_graph,
            &Target::Gpu { backend, caps },
            [Some(&honest)].into_iter(),
        )
        .expect("the planner's own I32 body reads Slice<i32>");

        let forged = plan_of(&f32_graph);
        match super::value_storage_of_plans(
            &i32_graph,
            &Target::Gpu { backend, caps },
            [Some(&forged)].into_iter(),
        ) {
            Err(crate::PlanError::I32ReadAsF32 { value, reader }) => {
                assert_eq!((value, reader), (x, y));
            }
            other => panic!("expected I32ReadAsF32 naming {x} read by {y}, got {other:?}"),
        }
    }

    /// POOT-1115 SC-002: the storage checks read a dispatch's params through its binding, not by position.
    /// The planner's own plan for `gather(table: f32, index: i32)` is rebuilt with its two reads swapped:
    /// the kernel's first param, a `Slice<f32>`, now reads the I32 index. The honest binding is accepted.
    ///
    /// Mutation: in `operand_reads`, pair param `p` with operand `p` instead of the operand
    /// `binding.reads()[p]` names; the swapped plan is accepted and this row goes red.
    #[test]
    fn storage_checks_read_params_through_the_binding() {
        use crate::{Binding, Dispatch};

        let b = Builder::for_tests();
        let table = b
            .constant("table", TensorType::new(vec![8, 3], DType::F32))
            .unwrap();
        let index = b
            .constant("index", TensorType::new(vec![2], DType::I32))
            .unwrap();
        let gathered = b.gather(table, 0, index).unwrap();
        let (g, index, gathered) = (b.finish(gathered).unwrap(), index.id(), gathered.id());
        let backend = Backend::SpirvVulkan;
        let caps = poot_test_util::device_caps::default_caps_for(backend);
        let limits = crate::test_support::roomy_body_limits();
        let gather = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .expect("the gather");
        let honest = plan_eqn(&g, gather, backend, &caps, &limits).unwrap();
        super::value_storage_of_plans(
            &g,
            &Target::Gpu {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
            [Some(&honest)].into_iter(),
        )
        .expect("the planner's own binding reads the index through a Slice<i32>");

        let Plan::Dispatches(plan) = plan_eqn(&g, gather, backend, &caps, &limits).unwrap() else {
            panic!("a gather dispatches");
        };
        let swapped: Vec<_> = plan
            .into_dispatches()
            .into_iter()
            .map(|dispatch| {
                let mut reads = dispatch.binding().reads().to_vec();
                reads.swap(0, 1);
                let binding = Binding::new(reads, dispatch.binding().write());
                let choice = dispatch.choice().clone();
                let Dispatch {
                    body,
                    key,
                    label,
                    launch,
                    ..
                } = dispatch;
                Dispatch::new(body, key, label, launch, binding, choice)
            })
            .collect();
        let swapped = Plan::Dispatches(crate::DispatchPlan::new(swapped, Vec::new()).unwrap());
        match super::value_storage_of_plans(
            &g,
            &Target::Gpu {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            },
            [Some(&swapped)].into_iter(),
        ) {
            Err(crate::PlanError::I32ReadAsF32 { value, reader }) => {
                assert_eq!((value, reader), (index, gathered));
            }
            other => {
                panic!("expected I32ReadAsF32 naming {index} read by {gathered}, got {other:?}")
            }
        }
    }

    /// Card 1011: a reader whose body has no packed lane for a BF16 const is refused by `compile` with the
    /// typed `DtypeLowering` refusal naming the equation, on every backend where the const is packed. A plain
    /// `Gather` of a BF16 table (the shape `fold_dense_bf16_row_gathers` does not match: no widening cast
    /// follows) binds the packed buffer to a BF16-element param.
    ///
    /// Mutation: delete the `check_bf16_const_reads` call in `value_storage_of_plans`; `compile` accepts the
    /// graph and the three refusals below go red.
    #[test]
    fn bf16_const_reads_refuse_a_reader_without_the_lane() {
        use poot_graph_ir::Slot;

        use crate::{Capability, CompileOptions, PlanError, Target};

        let b = Builder::for_tests();
        let table = b
            .constant("table", TensorType::new(vec![16, 8], DType::BF16))
            .unwrap();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![4], DType::I32))
            .unwrap();
        let rows = b.gather(table, 0, token).unwrap();
        let neg = b.unary(poot_graph_ir::UnOp::Neg, rows).unwrap();
        let g = b.finish(neg).unwrap();
        let gather = g
            .eqns()
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .expect("the gather")
            .out;
        // Only wgpu always packs: AMD and PTX keep a native BF16 lane for a const with a non-packed reader,
        // where a BF16-element body is the matching lane.
        let backend = Backend::SpirvVulkan;
        let target = Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        let error = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), target),
            &crate::staged::Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions::new(crate::CompileLimits::STANDARD),
        )
        .expect_err("a gather over a packed BF16 table has no lane to read");
        let crate::staged::StagedCompileError::Compile(crate::CompileError::Plan(plan)) = error
        else {
            panic!("expected a planner refusal, got {error}");
        };
        match *plan {
            PlanError::Refused(refusal) => {
                assert_eq!(refusal.eqn, gather, "the refusal names the gather equation");
                assert_eq!(refusal.missing, Capability::DtypeLowering);
            }
            other => panic!("expected a Refused DtypeLowering, got {other}"),
        }
    }

    /// Card 564: on AmdGcn and Nvptx a BF16 const whose only readers read packed lanes - a checkpoint-
    /// orientation projection weight `compile` folds into a `DenseContraction`, and the embed table it
    /// folds into a `DenseRowGather` - is planned `Bf16Packed`, the lane those bodies read; native
    /// storage there bound a two-byte buffer to a kernel argument declared as `u32` words.
    ///
    /// Mutation: drop the `bf16_const_feeds_only_packed_readers` arm from `value_storage_kind`'s
    /// `AmdGcn` case (or set the `Nvptx` case back to `false`); both consts plan native
    /// `Dense(BF16)` on that backend and this row goes red.
    #[test]
    fn bf16_consts_read_only_by_packed_readers_plan_packed() {
        use poot_graph_ir::Slot;

        use crate::{CompileOptions, Target};

        let b = Builder::for_tests();
        let embed = b
            .constant("embed", TensorType::new(vec![16, 32], DType::BF16))
            .unwrap();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![1, 4], DType::I32))
            .unwrap();
        let rows = b
            .cast(b.gather(embed, 0, token).unwrap(), DType::F32)
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![8, 32], DType::BF16))
            .unwrap();
        let out = b.matmul(rows, b.transpose(w, vec![1, 0]).unwrap()).unwrap();
        let g = b.finish(out).unwrap();
        for backend in [Backend::AmdGcn(AmdArch::gfx1151()), Backend::Nvptx] {
            let target = Target::Gpu {
                backend,
                caps: poot_test_util::device_caps::default_caps_for(backend),
            };
            let staged = compile_staged(
                &g,
                &TargetSet::single(DeviceId(0), target),
                &Partition {
                    experts: ExpertPlacement::AllResident,
                    devices: DevicePlacement::Single(DeviceId(0)),
                },
                &CompileOptions::new(crate::CompileLimits::STANDARD),
            )
            .unwrap();
            let (_, _, program) = staged.stages().next().unwrap();
            let pg = program.graph();
            let readers: Vec<&str> = program
                .planned()
                .map(|(eqn, _)| eqn.op.kind_name())
                .filter(|kind| kind.starts_with("dense_"))
                .collect();
            assert_eq!(readers, ["dense_row_gather", "dense_contraction"]);
            for name in ["embed", "w"] {
                let id = pg
                    .consts()
                    .iter()
                    .copied()
                    .find(|&id| pg.meta(id).name.as_deref() == Some(name))
                    .unwrap();
                assert_eq!(
                    program.storage().storage(id).buffer_storage(),
                    poot_target::BufferStorage::bf16_packed(),
                    "{name}: a BF16 const read only by a packed reader on {backend:?}"
                );
            }
        }
    }

    /// A `Qwen4Exp` PLE shard: an E4M3FN `[rows, width]` const gathered by an I32 index and widened,
    /// which `fold_dense_bf16_row_gathers` makes one `DenseRowGather`.
    fn e4m3_shard_graph(
        rows: usize,
        width: usize,
        second_reader: bool,
    ) -> (Graph, poot_graph_ir::ValueId) {
        use poot_graph_ir::{BinOp, RedOp, Slot};

        let b = Builder::for_tests();
        let table = b
            .constant("shard", TensorType::new(vec![rows, width], DType::E4M3FN))
            .unwrap();
        let token = b
            .slot(Slot::Token, TensorType::new(vec![3], DType::I32))
            .unwrap();
        let gathered = b
            .cast(b.gather(table, 0, token).unwrap(), DType::F32)
            .unwrap();
        let out = if second_reader {
            // The table is also widened whole and summed: a reader that is not a row gather, so the
            // table cannot be the packed lane.
            let widened = b.cast(table, DType::F32).unwrap();
            let sum = |v| {
                b.reduce(
                    RedOp::Sum,
                    b.reduce(RedOp::Sum, v, 1, false).unwrap(),
                    0,
                    false,
                )
                .unwrap()
            };
            b.binary(BinOp::Add, sum(gathered), sum(widened)).unwrap()
        } else {
            gathered
        };
        (
            crate::dtype_widen::fold_dense_bf16_row_gathers(&b.finish(out).unwrap()).unwrap(),
            table.id(),
        )
    }

    /// POOT-1060: an E4M3FN table only `DenseRowGather` reads is planned `e4m3_packed()` on every backend,
    /// at `rows * ceil(width / 4)` words.
    ///
    /// Mutation: drop the `e4m3_row_gather_table_storages` extension in `value_storage_of_plans`; the table
    /// plans as dense E4M3FN raw bytes on every backend and this row goes red.
    #[test]
    fn an_e4m3_table_read_only_by_row_gathers_plans_packed() {
        let (g, table) = e4m3_shard_graph(9, 6, false);
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            let storage = fixture_value_storage(&g, backend, &caps);
            let planned = storage.storage(table).buffer_storage();
            assert_eq!(
                planned,
                poot_target::BufferStorage::e4m3_packed(),
                "{backend:?}"
            );
            assert_eq!(planned.device_elems(&[9, 6]), Some(18), "{backend:?}");
        }
    }

    /// POOT-1060: a buffer has one storage, so an E4M3FN table a second, non-row-gather equation reads is a
    /// typed refusal naming the table, not a packed buffer another kernel would read as raw bytes.
    ///
    /// Mutation: skip the `sole_reader` test in `e4m3_row_gather_table_storages`; the shared table plans
    /// packed and this row goes red.
    #[test]
    fn an_e4m3_table_with_a_second_reader_is_a_typed_refusal() {
        let (g, table) = e4m3_shard_graph(9, 6, true);
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            match fixture_value_storage_result(&g, backend, &caps) {
                Err(crate::PlanError::UnrepresentableValue {
                    value,
                    dtype: DType::E4M3FN,
                    gap: crate::StorageGap::E4m3PackedLanes,
                }) => assert_eq!(value, table, "{backend:?}"),
                Err(other) => panic!("{backend:?}: expected the packed-E4M3 refusal, got {other}"),
                Ok(_) => panic!("{backend:?}: a shared E4M3 table was planned"),
            }
        }
    }
}

/// POOT-1229: the planner-side check over the stored KV extent.
#[cfg(test)]
mod kv_stride_tests {
    use poot_graph_ir::ops::{AttentionKeys, attention_live, causal_mask_from_pos};
    use poot_graph_ir::{
        BinOp, Builder, Graph, IndexRange, Outputs, RedOp, Scalar, Slot, StateRole, TensorType,
    };
    use poot_target::{AmdArch, Backend, DeviceCaps};
    use poot_tensor::DType;

    use crate::compile::{CompileError, CompileOptions, Target};
    use crate::planner::plan_graph;
    use crate::{CompileLimits, PlanError, compile};

    const HEADS: usize = 2;

    /// gfx1151 with the Strix Halo channel data, and a target without any.
    fn with_channels() -> Target {
        let arch = AmdArch::gfx1151();
        Target::Gpu {
            backend: Backend::AmdGcn(arch),
            caps: DeviceCaps {
                dram_channels: poot_target::dram_channels::DramChannels::of_device_name("gfx1151"),
                ..DeviceCaps::rocm_default()
            },
        }
    }

    fn without_channels() -> Target {
        Target::Gpu {
            backend: Backend::Nvptx,
            caps: DeviceCaps::ptx_default(),
        }
    }

    /// How the K cache of the decode graph below is read.
    #[derive(Clone, Copy)]
    enum Reader {
        /// By an attention over dense keys: the legalizer can pad it.
        Attention,
        /// By a reduce over its positions: only the logical extent is meaningful.
        Reduce,
    }

    /// One decode step over a `[1, HEADS, cap, d]` K and V cache pair: the cache write at `Slot::Pos`, then `reader`.
    fn decode(cap: usize, d: usize, reader: Reader) -> Graph {
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::f32(vec![1, HEADS, 1, d]))
            .unwrap();
        let new_k = b
            .constant("k", TensorType::f32(vec![1, HEADS, 1, d]))
            .unwrap();
        let new_v = b
            .constant("v", TensorType::f32(vec![1, HEADS, 1, d]))
            .unwrap();
        let pos = b
            .slot_ranged(
                Slot::Pos,
                TensorType::new(vec![1, 1], DType::I32),
                IndexRange::below(cap).expect("a non-empty range"),
            )
            .unwrap();
        let at = b.reshape(pos, vec![]).unwrap();
        let state = |name: &str| {
            b.state_input(
                name,
                TensorType::f32(vec![1, HEADS, cap, d]),
                StateRole::Positional { axis: 2 },
            )
            .unwrap()
        };
        let (k_in, v_in) = (state("kv.k"), state("kv.v"));
        let k = b.dynamic_update_slice_dyn(k_in, new_k, at, 2).unwrap();
        let v = b.dynamic_update_slice_dyn(v_in, new_v, at, 2).unwrap();
        let out = match reader {
            Reader::Attention => {
                let mask = causal_mask_from_pos(&b, pos, cap, None).unwrap();
                let len = b
                    .binary_scalar(BinOp::Add, b.reshape(pos, vec![1]).unwrap(), Scalar::I32(1))
                    .unwrap();
                attention_live(
                    &b,
                    q,
                    AttentionKeys::Single(k),
                    v,
                    1,
                    0.125,
                    mask,
                    None,
                    len,
                )
                .unwrap()
                .out
            }
            Reader::Reduce => {
                let total = |x| b.reduce(RedOp::Sum, x, 2, true).unwrap();
                b.binary(BinOp::Add, total(k), total(v)).unwrap()
            }
        };
        b.finish(Outputs::new(out).with_state(&[(k_in, k), (v_in, v)]))
            .unwrap()
    }

    /// The graph `compile` hands the legalizer: canonical, deduplicated, its iotas folded and its attention
    /// chain the fused op.
    fn fused(g: &Graph) -> Graph {
        let (canonical, _) = crate::passes::canonicalize_with_declines(g).unwrap();
        let folded = crate::passes::fold_iota(&crate::passes::cse(&canonical).unwrap()).unwrap();
        let fused = crate::passes::flash_attention_capped(&folded, None)
            .unwrap()
            .0;
        assert!(
            fused
                .eqns()
                .iter()
                .any(|eqn| matches!(eqn.op, poot_graph_ir::OpKind::Attention(_))),
            "the decode chain fuses to an attention"
        );
        crate::passes::dce(&fused).unwrap()
    }

    fn stored(program: &crate::Program) -> Vec<usize> {
        program
            .graph()
            .state_pairs()
            .map(|pair| program.graph().aval(pair.input).shape[2])
            .collect()
    }

    fn options() -> CompileOptions {
        CompileOptions::new(CompileLimits::STANDARD)
    }

    /// SC-004: a plan whose stored stride camps on a target with channel data is refused by name when the
    /// graph reaches the planner unpadded; the same graph through `compile` is padded by the legalizer and
    /// plans; a target without channel data plans the exact capacity. Mutation: make
    /// `check_kv_head_strides` return `Ok(())` at its first line; the unpadded graph plans and this goes red.
    #[test]
    fn a_camping_cache_the_legalizer_did_not_pad_is_refused() {
        // 512 positions of a 128-wide head: a 2048 B stride, whole channel periods.
        let g = decode(512, 128, Reader::Attention);
        let limits = CompileLimits::STANDARD.body();
        let Err(refused) = plan_graph(&fused(&g), &with_channels(), &limits) else {
            panic!("a camping cache the legalizer could pad was planned");
        };
        assert!(
            matches!(
                refused,
                CompileError::Plan(ref error)
                    if matches!(**error, PlanError::KvHeadStrideCamps { stride_bytes: 262_144, .. })
            ),
            "{refused:?}"
        );
        assert_eq!(
            stored(&compile(&g, &with_channels(), &options()).expect("the legalizer pads it")),
            [513, 513]
        );
        assert!(plan_graph(&fused(&g), &without_channels(), &limits).is_ok());
        assert_eq!(
            stored(&compile(&g, &without_channels(), &options()).unwrap()),
            [512, 512]
        );
    }

    /// The typed exemptions plan as declared: a head row that is itself whole periods cannot be padded off, a
    /// cache read by anything but an attention keeps its logical extent, and a paged pool's row stride is the
    /// pool's. Mutation: have `kv_caches` classify a whole-period row as `KvExtent::Camps`; the whole-period row goes red.
    #[test]
    fn a_camping_cache_with_a_typed_exemption_plans_as_declared() {
        let limits = CompileLimits::STANDARD.body();
        // 512 f32 per head row is 2048 B: no whole-position pad escapes the period.
        let whole_period = decode(4, 512, Reader::Attention);
        assert!(plan_graph(&fused(&whole_period), &with_channels(), &limits).is_ok());
        assert_eq!(
            stored(&compile(&whole_period, &with_channels(), &options()).unwrap()),
            [4, 4]
        );
        // Read by a reduce: the legalizer cannot add a position the reader would sum.
        let logical = decode(512, 128, Reader::Reduce);
        assert_eq!(
            stored(&compile(&logical, &with_channels(), &options()).unwrap()),
            [512, 512]
        );
        // A pool: `[slots, kv_heads, head_dim]` rows of 8 x 128 f32 are 4096 B.
        let b = Builder::for_tests();
        let pool = b
            .state_input(
                "pool",
                TensorType::f32(vec![64, 8, 128]),
                StateRole::Positional { axis: 0 },
            )
            .unwrap();
        let row = b.constant("row", TensorType::f32(vec![1, 8, 128])).unwrap();
        let at = b.slot(Slot::Pos, TensorType::scalar(DType::I32)).unwrap();
        let next = b.dynamic_update_slice_dyn(pool, row, at, 0).unwrap();
        let g = b
            .finish(Outputs::new(next).with_state(&[(pool, next)]))
            .unwrap();
        assert!(plan_graph(&g, &with_channels(), &limits).is_ok());
    }
}
