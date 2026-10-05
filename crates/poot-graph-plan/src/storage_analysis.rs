//! Whole-graph storage representation analysis, including exact-I32 requirements.

use poot_tensor::DType;
use std::collections::{HashMap, HashSet};

use poot_graph_ir::Storage;
use poot_target::{Backend, DeviceCaps};

use crate::dtype_widen::bf16_const_is_packed;
use crate::*;

#[derive(Debug)]
pub(crate) struct ValueComponents {
    pub(crate) parent: Vec<ValueId>,
    pub(crate) rank: Vec<u8>,
}

impl ValueComponents {
    pub(crate) fn new(values: usize) -> Self {
        Self {
            parent: (0..values).collect(),
            rank: vec![0; values],
        }
    }

    pub(crate) fn find(&mut self, mut value: ValueId) -> ValueId {
        while self.parent[value] != value {
            self.parent[value] = self.parent[self.parent[value]];
            value = self.parent[value];
        }
        value
    }

    pub(crate) fn union(&mut self, left: ValueId, right: ValueId) {
        let left = self.find(left);
        let right = self.find(right);
        if left != right {
            let (root, child) = match self.rank[left].cmp(&self.rank[right]) {
                std::cmp::Ordering::Less => (right, left),
                std::cmp::Ordering::Greater => (left, right),
                std::cmp::Ordering::Equal => {
                    self.rank[left] += 1;
                    (left, right)
                }
            };
            self.parent[child] = root;
        }
    }
}

/// Reusable whole-graph representation selection for exact I32 components.
///
/// Binary, Reshape, Transpose, Slice, Broadcast, and Concat connect I32 values in both directions. A
/// Binary-containing component is exact when it reaches the graph/state output or contains GeU, whose unsigned
/// bit-pattern meaning has no legacy float interpretation. Construction is linear in graph values, equations,
/// and operands. Borrowing the graph prevents mutation, while every public analyzed query also checks graph
/// identity so unrelated graphs with overlapping value IDs cannot reuse the result silently.
///
/// Generic over the validation channel. Observation is seeded from [`Graph::liveness_roots`]. A validation
/// value is F32, so it never joins an I32 component; the canonical helper is used so the root rule cannot
/// drift from the other planner seams.
#[derive(Debug)]
pub struct ExactI32StorageAnalysis<'g, V: ValidationChannel = NoValidations> {
    pub(crate) graph: &'g Graph<V>,
    pub(crate) required: Vec<bool>,
}

/// The requirement table of one [`ExactI32StorageAnalysis`], without its graph type. The per-equation planner
/// body reads only this table, so it is compiled once for every validation channel.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactI32Requirements<'a>(&'a [bool]);

impl ExactI32Requirements<'_> {
    pub(crate) fn required(self, id: ValueId) -> bool {
        self.0.get(id).copied().unwrap_or(false)
    }
}

impl<'g, V: ValidationChannel> ExactI32StorageAnalysis<'g, V> {
    /// Analyze exact-I32 storage requirements for one immutable graph.
    pub fn new(graph: &'g Graph<V>) -> Self {
        let mut components = ValueComponents::new(graph.values.len());
        for eqn in &graph.eqns {
            if graph
                .values
                .get(eqn.out)
                .is_none_or(|value| value.aval.dtype != DType::I32)
                || !matches!(
                    eqn.op,
                    OpKind::Binary(_)
                        | OpKind::Unary(_)
                        | OpKind::Select
                        | OpKind::Fused(_)
                        | OpKind::Reshape { .. }
                        | OpKind::Transpose { .. }
                        | OpKind::Slice { .. }
                        | OpKind::Broadcast { .. }
                        | OpKind::Concat { .. }
                )
            {
                continue;
            }
            for input in &eqn.inputs {
                if let Operand::Value(input) = input
                    && graph
                        .values
                        .get(*input)
                        .is_some_and(|value| value.aval.dtype == DType::I32)
                {
                    components.union(eqn.out, *input);
                }
            }
        }

        let mut has_binary = vec![false; graph.values.len()];
        let mut observed = vec![false; graph.values.len()];
        for eqn in &graph.eqns {
            let Some(op_observed) = exact_i32_pointwise_observation(&eqn.op) else {
                continue;
            };
            let touched = std::iter::once(eqn.out)
                .chain(eqn.inputs.iter().filter_map(|input| match input {
                    Operand::Value(value) => Some(*value),
                    Operand::Lit(_) => None,
                }))
                .filter(|&id| {
                    graph
                        .values
                        .get(id)
                        .is_some_and(|value| value.aval.dtype == DType::I32)
                });
            for id in touched {
                let root = components.find(id);
                has_binary[root] = true;
                if op_observed {
                    observed[root] = true;
                }
            }
        }
        for root in graph.liveness_roots() {
            if root < graph.values.len() {
                let root = components.find(root);
                observed[root] = true;
            }
        }

        let mut required = vec![false; graph.values.len()];
        for (id, value) in graph.values.iter().enumerate() {
            let root = components.find(id);
            required[id] = value.aval.dtype == DType::I32 && has_binary[root] && observed[root];
        }
        Self { graph, required }
    }

    /// Return the graph this analysis describes.
    pub fn graph(&self) -> &'g Graph<V> {
        self.graph
    }

    pub(crate) fn ensure_graph(&self, graph: &Graph<V>) -> Result<(), PlanError> {
        if std::ptr::eq(self.graph, graph) {
            Ok(())
        } else {
            Err(PlanError::AnalysisContextMismatch(
                "analysis and supplied graph have different identities".into(),
            ))
        }
    }

    pub(crate) fn ensure_eqn(&self, eqn: &Eqn) -> Result<(), PlanError> {
        let element_size = std::mem::size_of::<Eqn>();
        let start = self.graph.eqns.as_ptr() as usize;
        let byte_len = element_size
            .checked_mul(self.graph.eqns.len())
            .ok_or_else(|| {
                PlanError::AnalysisContextMismatch(
                    "analyzed graph equation storage length overflowed".into(),
                )
            })?;
        let address = std::ptr::from_ref(eqn) as usize;
        let Some(offset) = address.checked_sub(start) else {
            return Err(PlanError::AnalysisContextMismatch(
                "equation is not a member of the analyzed graph".into(),
            ));
        };
        if element_size != 0 && offset < byte_len && offset.is_multiple_of(element_size) {
            Ok(())
        } else {
            Err(PlanError::AnalysisContextMismatch(
                "equation is not a member of the analyzed graph".into(),
            ))
        }
    }

    /// Test-only convenience: production planning reads [`Self::requirement_table`] directly (card
    /// 546b: `graph_validation::validate_graph_exact_i32`, this method's last production caller, is
    /// gone with the rest of the legacy tensor-only executor gate).
    #[cfg(test)]
    pub(crate) fn required(&self, id: ValueId) -> bool {
        self.requirement_table().required(id)
    }

    pub(crate) fn requirement_table(&self) -> ExactI32Requirements<'_> {
        ExactI32Requirements(&self.required)
    }
}

/// Gather data and output share one kernel element type, so they must agree on exact I32 storage. With the
/// component rule neither is exact for an isolated Gather, which keeps legacy plans unchanged.
pub(crate) fn exact_i32_gather_data(
    analysis: ExactI32Requirements<'_>,
    eqn: &Eqn,
    data: ValueId,
) -> Result<bool, PlanError> {
    let data_exact = analysis.required(data);
    let output_exact = analysis.required(eqn.out);
    if data_exact != output_exact {
        return Err(PlanError::ExactI32GatherStorage {
            gather: eqn.out,
            data,
            data_exact,
            output_exact,
        });
    }
    Ok(data_exact)
}

pub(crate) fn exact_i32_pointwise_observation(op: &OpKind) -> Option<bool> {
    match op {
        OpKind::Binary(
            GBinOp::GeU
            | GBinOp::RemU
            | GBinOp::And
            | GBinOp::Or
            | GBinOp::Xor
            | GBinOp::Shl
            | GBinOp::Shr,
        ) => Some(true),
        OpKind::Binary(_) => Some(false),
        OpKind::Unary(GUnOp::Not | GUnOp::Clz) => Some(true),
        OpKind::Select => Some(true),
        OpKind::Fused(region) => {
            let mut observed = false;
            let mut pointwise = false;
            for step in &region.steps {
                match step.op {
                    FusedOp::Binary(
                        GBinOp::GeU
                        | GBinOp::RemU
                        | GBinOp::And
                        | GBinOp::Or
                        | GBinOp::Xor
                        | GBinOp::Shl
                        | GBinOp::Shr,
                    )
                    | FusedOp::Unary(GUnOp::Not | GUnOp::Clz)
                    | FusedOp::Select => {
                        pointwise = true;
                        observed = true;
                    }
                    FusedOp::Binary(_) | FusedOp::Unary(_) => pointwise = true,
                }
            }
            pointwise.then_some(observed)
        }
        _ => None,
    }
}

/// Card 526: every value's physical storage (dtype and packing) for one graph on one backend,
/// computed once for the whole graph. Folds in the packed-vs-native distinction of a BF16 const each
/// binder used to re-derive separately with its own call to `bf16_const_feeds_decode_bf16_gemv`:
/// `poot-gpu`, `poot-rocm-gpu` and `poot-ptx-gpu`'s binders read the result instead.
///
/// Call once per (graph, backend) after `compile`'s dtype passes have run; a const's storage dtype is its
/// declared dtype (Card 1011: no pass retypes a const).
///
/// A value's declared dtype and storage `Const`/`Device`/`Slot`/`State` residency, both already on
/// `poot_graph_ir::ValueMeta`, are not restated here.
///
/// `caps` is the device's measured [`DeviceCaps`], exactly as [`plan_eqn_analyzed`] takes it (card 522): the I32-slot-vs-F32-mirror decision (card 621) is a real per-eqn planning question -
/// whether a value's consuming kernel param is `Slice<i32>` - that only the planner can answer (a
/// `Gather` index is `Slice<i32>` exactly when [`ExactI32StorageAnalysis`] proves the value's whole
/// component is authoritative I32, `Slice<f32>` otherwise; card 545b: no const
/// is `Slice<i32>` unconditionally any more - `MatMulDequant`, the only op that ever consumed one that
/// way, is deleted). A caller with no measured device yet passes a documented fixture, as for
/// [`plan_eqn_analyzed`].
///
/// Every value's storage, read from one plan per equation (in `g.eqns` order; `None` for an equation that
/// did not plan, which constrains no operand). A pure function of the graph and the plans:
/// it plans nothing.
///
/// Refuses an I32 value whose readers disagree on its lane ([`PlanError::I32ReaderLaneConflict`]): the
/// storage below gives each value one representation, so a disagreeing reader would read reinterpreted
/// bits.
///
/// The plans may carry strided views ([`compute_views`]): a view only ever promotes an F32 movement output,
/// so it never changes which I32 value a kernel reads through a `Slice<i32>` param.
///
/// Also refuses a dense-contraction weight stored in a layout of its own (packed F16, Card 1007) that another
/// reader shares ([`StorageGap::F16PackedLanes`]).
pub(crate) fn value_storage_of_plans<'p, V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    caps: &DeviceCaps,
    plans: impl Iterator<Item = Option<&'p Plan>>,
) -> Result<GraphStorage, PlanError> {
    let plans: Vec<Option<&Plan>> = plans.collect();
    let lanes = I32ReadLanes::of_plans(g, plans.iter().copied());
    lanes.check_agreement()?;
    let weights = dense_contraction_weight_storages(g)?;
    let storage = storage_of_read_lanes(g, backend, &lanes, &weights, caps);
    check_bf16_const_reads(g, backend, &storage, &plans)?;
    Ok(storage)
}

/// Card 1011: a BF16 const is never widened on upload, so the kernel param each reader binds it to must be
/// the lane the storage plan gave it: `u32` words for a packed const ([`bf16_const_is_packed`]), BF16
/// elements for a native one. A reader whose body has no such lane (a plain `Gather` over a packed table on
/// wgpu, a packed-lane contraction sharing a native-lane weight) is refused by name, naming the equation,
/// instead of reading the buffer at the wrong element width or failing later in codegen.
///
/// The invariant this relies on: a plan's value operand `i` is its kernel's data param `i` (data params come
/// first in both `Compute` and `ComputeMeta`, the meta buffer rides after them), and every chunk of a
/// `ComputeChunks` plan shares the first chunk's param types. A reader whose params were not 1:1 with its value
/// operands would be skipped here, so `bf16_const_reads_refuse_a_reader_without_the_lane` pins the refusal on a
/// real reader.
fn check_bf16_const_reads<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    storage: &GraphStorage,
    plans: &[Option<&Plan>],
) -> Result<(), PlanError> {
    let mut roots: HashMap<ValueId, ValueId> = HashMap::new();
    for (eqn, plan) in g.eqns.iter().zip(plans) {
        if let Some(Plan::Alias(src) | Plan::View { src, .. }) = plan {
            let root = roots.get(src).copied().unwrap_or(*src);
            roots.insert(eqn.out, root);
            continue;
        }
        let body = match plan {
            Some(Plan::Compute { body, .. }) | Some(Plan::ComputeMeta { body, .. }) => body,
            Some(Plan::ComputeChunks(chunks)) => match chunks.first() {
                Some(chunk) => &chunk.body,
                None => continue,
            },
            _ => continue,
        };
        let params: Vec<_> = body.params().collect();
        for (i, &vid) in value_ids(eqn).iter().enumerate() {
            let root = roots.get(&vid).copied().unwrap_or(vid);
            if g.meta(root).storage != Storage::Const || g.aval(root).dtype != DType::BF16 {
                continue;
            }
            let Some(&param) = params.get(i) else {
                continue;
            };
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
                    backend,
                    Capability::DtypeLowering,
                ));
            }
        }
    }
    Ok(())
}

fn storage_of_read_lanes<V: ValidationChannel>(
    g: &Graph<V>,
    backend: Backend,
    lanes: &I32ReadLanes,
    weights: &HashMap<ValueId, StorageKind>,
    caps: &DeviceCaps,
) -> GraphStorage {
    let values = (0..g.values.len())
        .map(|id| {
            ValueStorage::new(match weights.get(&id) {
                Some(&kind) => kind,
                None => value_storage_kind(g, id, backend, lanes, caps),
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
fn dense_contraction_weight_storages<V: ValidationChannel>(
    g: &Graph<V>,
) -> Result<HashMap<ValueId, StorageKind>, PlanError> {
    let reads = |eqn: &Eqn| match (&eqn.op, eqn.inputs.get(1)) {
        (OpKind::DenseContraction { weight }, Some(Operand::Value(value))) => {
            dense_contraction_weight_storage(*weight)
                .filter(|&kind| kind != StorageKind::Dense(*weight))
                .map(|kind| (*value, kind))
        }
        _ => None,
    };
    let weights: HashMap<ValueId, StorageKind> = g.eqns.iter().filter_map(reads).collect();
    for (&value, &kind) in &weights {
        let sole_reader = g.eqns.iter().all(|eqn| {
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

/// Card 621 (generalizing the pre-card-621 `i32_param_value_ids` duplicated in `poot-gpu`,
/// `poot-rocm-gpu` and `poot-ptx-gpu`): which kernel param type each value is read through. `exact` holds
/// the values some kernel consumes through a `Slice<i32>` param, i.e. that must be bound as exact i32 bytes
/// rather than the f32 `Tensor` mirror; storage is decided from it per value.
///
/// The agreement check reads buffers, not values: a [`Plan::Alias`] or [`Plan::View`] output shares its
/// source's buffer, so a read of it is a read of the source's bytes. `buffer_exact` and `buffer_mirror` key
/// each read by its root buffer (the value an alias/view chain bottoms out at) and map it to the first such
/// reader's output, to name it in an error; a `Slice<f32>` read counts as a mirror read when the root is
/// declared I32. Without the walk, an exact-I32 value reshaped into the f32-only DynamicUpdateSlice index
/// has two values, each read on one lane, and the conflict hides.
///
/// A value's consumer is well-defined (a `Slot::SlotMap`/`Slot::GdnSlotMap` feeds only its
/// `ScatterUpdate`/`DynamicUpdateSlice` as an f32 mirror; a `Gather` index or an I32->F32 `Cast` source
/// is i32 only when [`ExactI32StorageAnalysis`] proves its whole component authoritative, e.g. a
/// `Slot::Token` feeding embedding lookup alone never is - card 545b: a quant
/// const's `MatMulDequant` consumer, once the third case here, is deleted), so one pass over the eqns
/// suffices: each value's planned kernel body is inspected at most where it is actually consumed.
#[derive(Debug, Default)]
struct I32ReadLanes {
    exact: HashSet<ValueId>,
    buffer_exact: HashMap<ValueId, ValueId>,
    buffer_mirror: HashMap<ValueId, ValueId>,
}

impl I32ReadLanes {
    fn of_plans<'p, V: ValidationChannel>(
        g: &Graph<V>,
        plans: impl Iterator<Item = Option<&'p Plan>>,
    ) -> Self {
        let mut lanes = Self::default();
        // Each alias/view output's root buffer. Equations are topological, so a source's root is known
        // before any alias of it, and before any reader of that alias.
        let mut roots: HashMap<ValueId, ValueId> = HashMap::new();
        for (eqn, plan) in g.eqns.iter().zip(plans) {
            if let Some(Plan::Alias(src) | Plan::View { src, .. }) = plan {
                let root = roots.get(src).copied().unwrap_or(*src);
                roots.insert(eqn.out, root);
                continue;
            }
            // Data-input params come first in both Compute and ComputeMeta (the meta buffer rides after
            // them), so value_ids[i] maps to params[i] either way. ComputeChunks is scanned via chunk 0's
            // body (every chunk shares the same param types).
            let body = match plan {
                Some(Plan::Compute { body, .. }) | Some(Plan::ComputeMeta { body, .. }) => {
                    Some(body)
                }
                Some(Plan::ComputeChunks(chunks)) => chunks.first().map(|c| &c.body),
                _ => None,
            };
            let Some(body) = body else { continue };
            let params: Vec<_> = body.params().collect();
            for (i, &vid) in value_ids(eqn).iter().enumerate() {
                let Some(&param) = params.get(i) else {
                    continue;
                };
                let root = roots.get(&vid).copied().unwrap_or(vid);
                match slice_elem(body.local_ty(param)) {
                    Some(poot_kernel_ir::Ty::I32) => {
                        lanes.exact.insert(vid);
                        lanes.buffer_exact.entry(root).or_insert(eqn.out);
                    }
                    Some(poot_kernel_ir::Ty::F32) if g.aval(root).dtype == DType::I32 => {
                        lanes.buffer_mirror.entry(root).or_insert(eqn.out);
                    }
                    _ => {}
                }
            }
        }
        lanes
    }

    fn is_exact(&self, id: ValueId) -> bool {
        self.exact.contains(&id)
    }

    /// Refuse the lowest-id buffer read on both lanes, if any.
    fn check_agreement(&self) -> Result<(), PlanError> {
        let conflict = self
            .buffer_exact
            .iter()
            .filter_map(|(&value, &i32_reader)| {
                self.buffer_mirror
                    .get(&value)
                    .map(|&f32_reader| (value, i32_reader, f32_reader))
            })
            .min();
        match conflict {
            None => Ok(()),
            Some((value, i32_reader, f32_reader)) => Err(PlanError::I32ReaderLaneConflict {
                value,
                i32_reader,
                f32_reader,
            }),
        }
    }
}

/// A kernel param's slice element type (`Slice<T>` or `&[mut] Slice<T>`). A `Slice<i32>` buffer must be
/// bound as exact i32 bytes rather than the f32 `Tensor` mirror. Generalizes the (now-deleted) copies in
/// `poot-gpu`, `poot-rocm-gpu` and `poot-ptx-gpu`.
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

/// [`fixture_value_storage`]'s per-value decision. A declared-I32 value no kernel reads through `Slice<i32>`
/// ([`I32ReadLanes`]) is the F32-mirror lane (card 621: a `Slot::SlotMap`/`Slot::GdnSlotMap` index, an
/// embedding-lookup `Slot::Token`, or any other I32 value whose planned consumers read `Slice<f32>`). Every other
/// value that is not a BF16 const keeps its declared dtype, densely stored. A BF16 const is never widened
/// (Card 1011): it is BF16 on every backend, and [`bf16_const_is_packed`] decides whether the buffer is
/// packed two-per-`u32` word or native two-byte bf16, the one answer the cast planner reads too.
fn value_storage_kind<V: ValidationChannel>(
    g: &Graph<V>,
    id: ValueId,
    backend: Backend,
    lanes: &I32ReadLanes,
    caps: &DeviceCaps,
) -> StorageKind {
    let declared = g.aval(id).dtype;
    if declared == DType::I32 {
        return if lanes.is_exact(id) {
            StorageKind::Dense(DType::I32)
        } else {
            StorageKind::I32F32Mirror
        };
    }
    if g.meta(id).storage != Storage::Const || declared != DType::BF16 {
        return StorageKind::Dense(declared);
    }
    if bf16_const_is_packed(g, id, backend, caps) {
        StorageKind::Bf16Packed
    } else {
        StorageKind::Dense(DType::BF16)
    }
}

#[cfg(test)]
mod value_storage_tests {
    use poot_graph_ir::{Builder, Graph, OpKind, Storage, TensorType};
    use poot_target::{AmdArch, Backend, DeviceCaps};
    use poot_tensor::DType;

    use super::{ExactI32StorageAnalysis, GraphStorage, Plan};
    use crate::dtype_widen::prepare_target_graph;
    use crate::planner::plan_eqn_analyzed;

    /// The public `plan_value_storage` entry point was deleted (Card 626; its only callers were the
    /// pre-`compile` executors); `compile`'s live path is [`super::value_storage_of_plans`], which
    /// takes pre-planned equations rather than planning them itself. This test-only wrapper restores
    /// the "plan everything, then derive storage" shape these fixtures want.
    fn fixture_value_storage(g: &Graph, backend: Backend, caps: &DeviceCaps) -> GraphStorage {
        let exact_i32 = ExactI32StorageAnalysis::new(g);
        let plans: Vec<Option<Plan>> = g
            .eqns
            .iter()
            .map(|eqn| {
                plan_eqn_analyzed(
                    &exact_i32,
                    g,
                    eqn,
                    backend,
                    caps,
                    &poot_test_util::graph_fixtures::roomy_body_limits(),
                )
                .ok()
            })
            .collect();
        super::value_storage_of_plans(g, backend, caps, plans.iter().map(Option::as_ref))
            .expect("fixture graph has no I32 reader lane conflict")
    }

    /// `a x w -> f32` over two consts of the given dtypes, `m x 32 x 32`; mirrors
    /// `dtype_widen::tests::const_matmul`.
    fn const_matmul(m: usize, a_dtype: DType, w_dtype: DType) -> Graph {
        let b = Builder::new();
        let a = b.constant("a", TensorType::new(vec![m, 32], a_dtype));
        let w = b.constant("w", TensorType::new(vec![32, 32], w_dtype));
        let mm = b.matmul(a, w);
        let mut g = b.finish(mm);
        let out = g
            .eqns
            .iter()
            .find(|e| matches!(e.op, OpKind::MatMul))
            .expect("matmul eqn")
            .out;
        g.values[out].aval.dtype = DType::F32;
        g
    }

    /// Card 526, Card 1011: [`fixture_value_storage`]'s per-value dtype is the const's declared dtype for every
    /// const of every dense fixture graph, on every backend and AMD arch family: a BF16 const is planned BF16
    /// (packed or native), never F32. This is the general record every binder reads.
    ///
    /// Mutation: have `value_storage_kind` return `StorageKind::Dense(DType::F32)` for a BF16 const; the BF16
    /// rows go red naming the const.
    #[test]
    fn fixture_value_storage_dtype_is_the_declared_dtype() {
        let backends = [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
            Backend::AmdGcn(AmdArch::new("gfx90a", 64)),
            Backend::AmdGcn(AmdArch::new("gfx1030", 32)),
            Backend::AmdGcn(AmdArch::new("gfx1200", 32)),
        ];
        let graphs = [
            (
                "mixed bf16 prefill",
                const_matmul(32, DType::BF16, DType::BF16),
            ),
            (
                "mixed bf16 decode",
                const_matmul(1, DType::BF16, DType::BF16),
            ),
            (
                "bf16 checkpoint prefill",
                const_matmul(32, DType::F32, DType::BF16),
            ),
            (
                "bf16 checkpoint decode",
                const_matmul(1, DType::F32, DType::BF16),
            ),
        ];
        for backend in backends {
            for (label, g) in &graphs {
                let caps = poot_test_util::device_caps::default_caps_for(backend);
                let prepared = prepare_target_graph(g, backend, &caps);
                let storage = fixture_value_storage(&prepared, backend, &caps);
                for id in (0..prepared.values.len())
                    .filter(|&id| prepared.meta(id).storage == Storage::Const)
                {
                    let want = g.aval(id).dtype;
                    let got = storage.storage(id).dtype();
                    assert_eq!(
                        got, want,
                        "{label} on {backend:?} v{id}: fixture_value_storage says {got:?}, the const is declared {want:?}"
                    );
                }
            }
        }
    }

    /// Card 621 fixture, rebuilt for card 545b: an authoritative-I32 `Gather` index consumer (`idx`,
    /// `ExactI32StorageAnalysis`-required via `Binary(GeU)`, feeding `Gather` as raw I32), a
    /// `Slot::SlotMap` I32 consumer (feeding `ScatterUpdate`'s `inv` operand as an F32 mirror), and a
    /// `Slot::Token`-shaped I32 consumer feeding a plain embedding `Gather` (also an F32 mirror: it is
    /// not part of any exact-I32 component, the real-regression shape `poot-gpu`'s
    /// `qwen2_decode_gpu_resident_matches_cpu` exercises) - all in the same graph. Returns `(graph,
    /// gather_index_value_id, slotmap_value_id, token_value_id)`.
    fn gather_index_slotmap_and_token_graph() -> (
        Graph,
        poot_graph_ir::ValueId,
        poot_graph_ir::ValueId,
        poot_graph_ir::ValueId,
    ) {
        use poot_graph_ir::Slot;
        use poot_graph_ir::op::BinOp;
        let b = Builder::new();

        // Card 545b: `MatMulDequant`'s packed-quant I32 const (the original raw-I32-lane producer here) is
        // deleted; the surviving I32-lane consumer this pure-planner test can use is a `Gather` index
        // `ExactI32StorageAnalysis` proves authoritative (`storage_analysis.rs`, `exact_i32_gather_data`):
        // `Binary(GeU, idx_a, idx_b)`'s unsigned bit-pattern comparison has no legacy float
        // interpretation, so its whole component - `idx_a`, `idx_b` and the `GeU` output feeding
        // `Gather`'s index - is "required", unconditionally, unlike the isolated Token/SlotMap cases below.
        let (rows, cols) = (8usize, 3usize);
        let idx_a = b.constant("idx_a", TensorType::new(vec![2], DType::I32));
        let idx_b = b.constant("idx_b", TensorType::new(vec![2], DType::I32));
        let idx = b.binary(BinOp::GeU, idx_a, idx_b);
        let gather_table = b.constant(
            "gather_table",
            TensorType::new(vec![rows, cols], DType::F32),
        );
        let _gathered = b.gather(gather_table, 0, idx);

        const POOL: usize = 4;
        const REST: usize = 2;
        const NSRC: usize = 2;
        let base = b.constant("base", TensorType::new(vec![POOL, REST], DType::F32));
        let src = b.constant("src", TensorType::new(vec![NSRC, REST], DType::F32));
        let slot_map = b.slot(Slot::SlotMap, TensorType::new(vec![POOL], DType::I32));
        let _scatter = b.scatter_update(base, src, slot_map);

        const VOCAB: usize = 8;
        const HIDDEN: usize = 4;
        let embed = b.constant("embed", TensorType::new(vec![VOCAB, HIDDEN], DType::F32));
        let token = b.slot(Slot::Token, TensorType::new(vec![1], DType::I32));
        let out = b.gather_scalar(embed, 0, token);

        let gather_index_id = idx.id;
        let slot_id = slot_map.id;
        let token_id = token.id;
        (b.finish(out), gather_index_id, slot_id, token_id)
    }

    /// Card 621 SC-001, rebuilt for card 545b: `fixture_value_storage` keeps an authoritative-I32 `Gather`
    /// index consumer on the raw I32 lane (`Dense(I32)`) but puts a `Slot::SlotMap` I32 consumer of
    /// `ScatterUpdate`'s `inv` operand, and a `Slot::Token` I32 consumer of a plain embedding `Gather`,
    /// on the F32-mirror lane - in the same graph, on every backend. The claim (storage analysis flags
    /// exactly the i32-consumed values, not by construction) survives `MatMulDequant`'s deletion because
    /// `ExactI32StorageAnalysis`'s component rule is a general property of I32-typed pointwise/movement
    /// chains, not specific to the packed-quant carrier that originally demonstrated it.
    ///
    /// Mutation (card 621 review; never left in the tree): swapping `I32ReadLanes::of_plans`'s
    /// `exact` insert for one that also inserts the SlotMap/Token ids turns this test red, naming
    /// the mismatched value in the assertion message below.
    #[test]
    fn fixture_value_storage_distinguishes_i32_lanes() {
        let (g, gather_index_id, slot_id, token_id) = gather_index_slotmap_and_token_graph();
        for backend in [
            Backend::SpirvVulkan,
            Backend::Nvptx,
            Backend::AmdGcn(AmdArch::gfx1151()),
        ] {
            let caps = poot_test_util::device_caps::default_caps_for(backend);
            let storage = fixture_value_storage(&g, backend, &caps);
            let gather_index_storage = storage.storage(gather_index_id);
            assert_eq!(gather_index_storage.dtype(), DType::I32);
            assert!(
                !gather_index_storage.is_f32_mirror(),
                "{backend:?}: authoritative Gather index v{gather_index_id} must stay on the raw I32 lane, not the F32 mirror"
            );
            let slot_storage = storage.storage(slot_id);
            assert_eq!(slot_storage.dtype(), DType::I32);
            assert!(
                slot_storage.is_f32_mirror(),
                "{backend:?}: SlotMap v{slot_id} feeding ScatterUpdate's inv operand must be the F32 mirror, not raw I32"
            );
            let token_storage = storage.storage(token_id);
            assert_eq!(token_storage.dtype(), DType::I32);
            assert!(
                token_storage.is_f32_mirror(),
                "{backend:?}: Token v{token_id} feeding a plain embedding Gather must be the F32 mirror, not raw I32"
            );
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

        use crate::{
            Capability, CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy,
            PlanError, Submission, Target, TargetSet, compile_staged,
        };

        let b = Builder::new();
        let table = b.constant("table", TensorType::new(vec![16, 8], DType::BF16));
        let token = b.slot(Slot::Token, TensorType::new(vec![4], DType::I32));
        let rows = b.gather(table, 0, token);
        let neg = b.unary(poot_graph_ir::UnOp::Neg, rows);
        let g = b.finish(neg).with_validations(Vec::new());
        let gather = g
            .eqns
            .iter()
            .find(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
            .expect("the gather")
            .out;
        // Only wgpu always packs: AMD and PTX keep a native BF16 lane for a const with a non-packed reader,
        // where a BF16-element body is the matching lane.
        let backend = Backend::SpirvVulkan;
        let target = Target {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        let error = compile_staged(
            &g,
            &TargetSet::single(DeviceId(0), target),
            &crate::Partition {
                experts: ExpertPlacement::AllResident,
                devices: DevicePlacement::Single(DeviceId(0)),
            },
            &CompileOptions {
                execution: Submission::Replay,
                fusion: FusionPolicy::Full,
                limits: crate::CompileLimits::STANDARD,
            },
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

        use crate::{
            CompileOptions, DeviceId, DevicePlacement, ExpertPlacement, FusionPolicy, Partition,
            Submission, Target, TargetSet, compile_staged,
        };

        let b = Builder::new();
        let embed = b.constant("embed", TensorType::new(vec![16, 32], DType::BF16));
        let token = b.slot(Slot::Token, TensorType::new(vec![1, 4], DType::I32));
        let rows = b.cast(b.gather(embed, 0, token), DType::F32);
        let w = b.constant("w", TensorType::new(vec![8, 32], DType::BF16));
        let out = b.matmul(rows, b.transpose(w, vec![1, 0]));
        let g = b.finish(out).with_validations(Vec::new());
        for backend in [Backend::AmdGcn(AmdArch::gfx1151()), Backend::Nvptx] {
            let target = Target {
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
                &CompileOptions {
                    execution: Submission::Replay,
                    fusion: FusionPolicy::Full,
                    limits: crate::CompileLimits::STANDARD,
                },
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
                    .consts
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
}
