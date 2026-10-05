//! Types shared by the planner: [`Plan`], [`PlanError`], [`CollectiveKind`], [`ComputeChunk`]. `Backend`
//! moved to `poot-target` (card 522 review): callers import it from there directly.
//! Re-exported at the crate root.

use poot_graph_ir::{GraphValidationError, RedOp, ValidationId, ValueId};
use poot_tensor::DType;

use crate::{DeviceWitnessRejection, Refusal, StorageGap};
use poot_kernel_ir::Body;

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error("invalid graph: {0}")]
    InvalidGraph(#[from] GraphValidationError),
    /// The planner cannot lower one equation for the target (ADR-0104 decision 2). Boxed so a `Result`
    /// carrying a `PlanError` stays small.
    #[error(transparent)]
    Refused(Box<Refusal>),
    /// The graph holds a value whose storage a tensor-bound executor entry point cannot represent. The
    /// check is graph-wide and runs before any equation is planned, so it names a value, not an equation.
    #[error("v{value} ({dtype}) cannot bind to this executor entry point: {gap}")]
    UnrepresentableValue {
        value: ValueId,
        dtype: DType,
        gap: StorageGap,
    },
    /// The op is valid but a static shape it depends on violates its precondition (e.g. an `ArgTopK`
    /// whose rank operand is 0-D). A caller-triggerable shape mismatch is a typed error, not a panic.
    #[error("bad shape for planning: {0}")]
    BadShape(String),
    /// Reusable graph analysis was paired with a different graph, or with an equation not in that graph.
    /// A programmer error, detected so storage is never selected from unrelated value IDs.
    #[error("exact-I32 analysis context mismatch: {0}")]
    AnalysisContextMismatch(String),
    /// A Gather whose data and output disagree on exact I32 storage. One kernel cannot read f32-lane data
    /// and write exact I32 words (or the reverse), so the planner rejects it.
    #[error(
        "Gather v{gather} data v{data} exact I32 storage is {data_exact}, but its output requires {output_exact}"
    )]
    ExactI32GatherStorage {
        gather: ValueId,
        data: ValueId,
        data_exact: bool,
        output_exact: bool,
    },
    /// An I32 value whose planned readers disagree on its lane: one kernel reads it through `Slice<i32>`
    /// (exact words), another through `Slice<f32>` (the f32 mirror). One buffer holds one representation,
    /// so one of the two would read reinterpreted bits (spike-562 F-9). `value` is the buffer's root: a
    /// read through a `Plan::Alias`/`Plan::View` chain counts as a read of it. Readers are named by their
    /// output.
    #[error(
        "I32 v{value} is read as exact i32 by the equation defining v{i32_reader} but as the f32 mirror by v{f32_reader}"
    )]
    I32ReaderLaneConflict {
        value: ValueId,
        i32_reader: ValueId,
        f32_reader: ValueId,
    },
    /// A device executor admits only witness heads whose computation is exact on the device.
    #[error("validation {id:?} v{value} is not a canonical device witness: {reason}")]
    ValidationWitnessNotCanonical {
        id: ValidationId,
        value: ValueId,
        reason: DeviceWitnessRejection,
    },
    /// `state_commit_from_plans` donated `state_input`, but a native transaction's rollback journal needs
    /// the exact bytes the donation overwrote, known without inspecting device data. Only
    /// `DynamicUpdateSlice` provides that; `op` names the producing equation that does not. There is no
    /// whole-buffer snapshot fallback.
    #[error(
        "state v{state_input} is donated by {op}, whose overwrite region a native transaction cannot statically recover for rollback"
    )]
    UnrecoverableDonation { state_input: ValueId, op: String },
}

/// Which cross-rank collective a [`Plan::Collective`] lowers. The first two variants mirror the graph
/// nodes `OpKind::AllReduce{op,axis}` / `OpKind::AllGather{axis}`.
///
/// `PointToPoint` and `AllToAll` are planned by the `multi_device` module (backend-neutral communication
/// rows over a caller-declared device topology, no device dispatch); device-resident dispatch is card 408.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum CollectiveKind {
    /// `AllReduce{op}`: combine every rank's buffer with `op`; each rank ends with the full result in the same shape.
    AllReduce,
    /// `AllGather`: concatenate the per-rank shards along the collective's `axis`; each rank ends
    /// holding the full gathered tensor.
    AllGather,
    /// One explicit ordered transfer from one device to exactly one other device, with no
    /// reduction or concatenation.
    PointToPoint,
    /// An exchange of distinct per-pair payloads among a device group (e.g. MoE routed
    /// expert dispatch/combine), with explicit per-device route counts supplied by the caller.
    AllToAll,
}

/// How to execute one eqn (independent of host vs device buffers, and of the backend).
#[derive(Debug)]
pub enum Plan {
    /// Dispatch `body` reading the eqn's value-operand buffers (in order) and writing the output.
    /// `grid` is the total thread count (`[x, y, z]`); the executor divides it by `body.workgroup_size`
    /// to get the dispatch's workgroup counts. The planner sets it once, alongside the kernel choice
    /// that shapes it (card 525); no executor recomputes it.
    Compute {
        body: Body,
        key: String,
        /// A human-readable name for the kernel (family, form and, for a contraction, its shape):
        /// display only, carried into the per-dispatch timing so device time can be read per shape.
        label: String,
        grid: [u32; 3],
    },
    /// Like [`Plan::Compute`], but binds one extra read-only `u32` metadata buffer between the eqn's data
    /// inputs and the output (param order `[data inputs.., meta, out]`). An imported shape-generic kernel
    /// gets its dims this way instead of baked consts (card 044: batched decode GEMV takes `dims = [B]`).
    /// Used by SPIR-V and NVPTX, and by AMDGCN where the target planner selects the same imported body.
    ComputeMeta {
        body: Body,
        key: String,
        /// The display label, as on [`Plan::Compute`].
        label: String,
        meta: Vec<u32>,
        grid: [u32; 3],
    },
    /// Dispatch several kernels over the same input/output buffers, each a tiled-GEMM variant with a baked
    /// tile offset covering a disjoint sub-2^15 tile range (avoids the RADV 2^15-workgroup miscompile).
    /// Each chunk's `groups` is its workgroup count (< 2^15). Chunks write disjoint output tiles, so
    /// dispatch order is irrelevant.
    ComputeChunks(Vec<ComputeChunk>),
    /// The output aliases the given input value's buffer (reshape).
    Alias(ValueId),
    /// The output is a strided view of `src`'s buffer (spec 132): no dispatch, like [`Plan::Alias`], but
    /// with its own (possibly non-row-major) `strides` and element `offset`. Emitted only when every
    /// consumer can read a strided input (`is_strided_capable`); see `compute_views`. Reshape still lowers
    /// to `Plan::Alias`; this variant is for Transpose/Slice/Broadcast.
    View {
        src: ValueId,
        strides: Vec<usize>,
        offset: usize,
    },
    /// A cross-rank collective run by the multi-rank executor (see
    /// `specs/049-tensor-parallel/multi-gpu-execution.md`). Emitted only at `world_size > 1`; at
    /// `world_size == 1` both collectives lower to [`Plan::Alias`] (byte-identical to the single-rank path,
    /// FR-007). Single-device executors return a "needs the multi-rank executor" error. `op` is meaningful
    /// for `AllReduce` only; `axis` is the shard/concat axis from the graph node.
    Collective {
        kind: CollectiveKind,
        op: RedOp,
        axis: usize,
    },
}

/// Which [`Plan`] variant an equation planned as, without its payload. Executors name it when a walk
/// cannot run the plan it was given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlanKind {
    Compute,
    ComputeMeta,
    ComputeChunks,
    Alias,
    View,
    Collective,
}

impl Plan {
    pub fn kind(&self) -> PlanKind {
        match self {
            Plan::Compute { .. } => PlanKind::Compute,
            Plan::ComputeMeta { .. } => PlanKind::ComputeMeta,
            Plan::ComputeChunks(_) => PlanKind::ComputeChunks,
            Plan::Alias(_) => PlanKind::Alias,
            Plan::View { .. } => PlanKind::View,
            Plan::Collective { .. } => PlanKind::Collective,
        }
    }
}

/// One sub-2^15 dispatch of a chunked tiled GEMM (card 096): a tiled-GEMM `body` with a baked tile offset,
/// launched with `groups` workgroups (one output tile each, in `[offset, offset+groups)`).
#[derive(Clone, Debug)]
pub struct ComputeChunk {
    pub body: Body,
    pub key: String,
    /// The display label, as on [`Plan::Compute`].
    pub label: String,
    pub groups: usize,
    /// Optional u32 metadata buffer (bound like [`Plan::ComputeMeta`]). Empty for a kernelgen chunk (tile
    /// offset baked as a const); an imported chunk carries `[M,K,N,offset]` (card 044).
    pub meta: Vec<u32>,
}

/// How [`ValueStorage`] packs its dtype into a device buffer (card 526).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StorageKind {
    /// One buffer element per value element, at `dtype`'s natural width (f32/i32 four bytes,
    /// bf16/f16 native two bytes).
    Dense(DType),
    /// BF16 checkpoint bytes packed two elements per device `u32` word; the consuming kernel widens
    /// each pair in-register instead of reading two natively-stored two-byte elements (card 380's
    /// `DenseContraction`, card 381's `DenseRowGather`, and the decode GEMV weight lane). Always BF16
    /// (see [`ValueStorage::dtype`]).
    Bf16Packed,
    /// F16 checkpoint bytes packed two elements per device `u32` word (Card 1007): a `DenseContraction`
    /// weight, which every backend's generated body decodes in-register. Always F16.
    F16Packed,
    /// A declared-I32 value bound as its F32 bit-pattern mirror (card 621): a `Slot::SlotMap`/
    /// `Slot::GdnSlotMap` index (or any other I32-declared value) feeding a
    /// `ScatterUpdate`/`DynamicUpdateSlice` index operand. Those imported/kernelgen bodies read that
    /// operand as `Slice<f32>` and cast to i32 in-kernel (spec 045) on every backend, so the device
    /// buffer's own words are F32 while the value's declared dtype stays I32 (see
    /// [`ValueStorage::dtype`]/`ValueStorage::is_f32_mirror`). Raw I32 bytes in that slot reinterpret
    /// as a denormal or NaN, and casting that back to i32 collapses to 0, so a `-1` (keep-base) index is
    /// misread as `0` and corrupts another sequence's pool row.
    I32F32Mirror,
}

/// One planned value's physical storage (card 526): the dtype its device buffer holds and whether it is
/// packed. For a const the dtype is the declared one (Card 1011), so a binder reads this instead of
/// re-deriving storage from graph shape or a local heuristic (R469-004).
///
/// The record's one private field is `poot_target::BufferStorage` (card 527): the same leaf-crate
/// vocabulary every runtime's buffer handle carries, so a binder comparing an allocated buffer's own
/// storage against the plan's record compares like with like, without either crate depending on the
/// other (`poot-target` cannot depend on `poot-graph-ir`, which already depends on it; the conversion
/// from this crate's `DType`-keyed [`StorageKind`] happens once, at `ValueStorage::new`).
///
/// The private field also means an external crate cannot construct a `ValueStorage` the planner never
/// decided (R484-015): see the `compile_fail` example below.
///
/// ```compile_fail,E0451
/// let _ = poot_graph_plan::ValueStorage { storage: poot_target::BufferStorage::f32() };
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ValueStorage {
    storage: poot_target::BufferStorage,
}

impl ValueStorage {
    pub(crate) fn new(kind: StorageKind) -> Self {
        let storage = match kind {
            StorageKind::Dense(dtype) => {
                poot_target::BufferStorage::dense(dtype_to_element(dtype), dtype_to_logical(dtype))
            }
            StorageKind::Bf16Packed => poot_target::BufferStorage::bf16_packed(),
            StorageKind::F16Packed => poot_target::BufferStorage::f16_packed(),
            StorageKind::I32F32Mirror => poot_target::BufferStorage::i32_f32_mirror(),
        };
        Self { storage }
    }

    /// This record in the leaf-crate vocabulary a buffer handle's storage contract shares (card 527): every binder reads this at bind time and compares it against the handle it uploaded
    /// through, returning a typed mismatch when they disagree - the check that actually closes
    /// R484-001/R471-009, as opposed to a check a test can only reach by forging a handle's field.
    pub fn buffer_storage(&self) -> poot_target::BufferStorage {
        self.storage
    }

    /// The dtype a binder uploads: `Dense`'s own dtype, or the packed dtype (`BF16` or `F16`) for a
    /// packed value (still checkpoint bytes, just packed two elements per device word).
    pub fn dtype(&self) -> DType {
        logical_to_dtype(self.storage.dtype())
    }

    /// Card 621: whether this declared-I32 value's device buffer holds the F32 bit-pattern mirror of
    /// each element rather than raw I32 bytes (a `Slot::SlotMap`/`Slot::GdnSlotMap` index feeding a
    /// `ScatterUpdate`/`DynamicUpdateSlice` index operand). A binder reads this to choose the upload
    /// lane instead of re-deriving it from `t.ints`/a local per-eqn walk (R469-004). `pub(crate)`/
    /// test-only: no production binder survives Card 546b's `GpuExecutor` deletion to call this; kept
    /// for the `I32ReadLanes`/`value_storage_kind` coverage in `storage_analysis.rs` and `compile.rs`.
    #[cfg(test)]
    pub(crate) fn is_f32_mirror(&self) -> bool {
        self.storage.element() == poot_target::ElementKind::F32
            && self.storage.dtype() == poot_target::LogicalDType::I32
    }
}

/// [`DType`] -> `poot_target::ElementKind`: the native device-word representation a `Dense`-layout
/// value of this dtype uses. Not a `From` impl (both types are foreign to this crate; see
/// `ValueStorage::new`).
fn dtype_to_element(dtype: DType) -> poot_target::ElementKind {
    match dtype {
        DType::F32 => poot_target::ElementKind::F32,
        DType::BF16 => poot_target::ElementKind::Bf16,
        DType::F16 => poot_target::ElementKind::F16,
        DType::I32 => poot_target::ElementKind::I32,
        DType::I8 | DType::E4M3FN => poot_target::ElementKind::RawBytes,
        // Checkpoint-storage-only dtypes: `poot_tensor::DType` names every stored
        // dtype a checkpoint header can carry, but a traced graph value's declared dtype is always
        // one of the six arms above (`infer` never produces the others).
        DType::Bool
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64
        | DType::E8M0 => unreachable!(
            "{dtype:?} is a checkpoint-storage-only dtype; no traced graph value declares it"
        ),
    }
}

/// [`DType`] -> `poot_target::LogicalDType`: the same dtype, in the leaf-crate vocabulary.
fn dtype_to_logical(dtype: DType) -> poot_target::LogicalDType {
    match dtype {
        DType::F32 => poot_target::LogicalDType::F32,
        DType::BF16 => poot_target::LogicalDType::Bf16,
        DType::F16 => poot_target::LogicalDType::F16,
        DType::I32 => poot_target::LogicalDType::I32,
        DType::I8 => poot_target::LogicalDType::I8,
        DType::E4M3FN => poot_target::LogicalDType::E4M3FN,
        DType::Bool
        | DType::U8
        | DType::I16
        | DType::U16
        | DType::U32
        | DType::I64
        | DType::U64
        | DType::F64
        | DType::E8M0 => unreachable!(
            "{dtype:?} is a checkpoint-storage-only dtype; no traced graph value declares it"
        ),
    }
}

/// The inverse of [`dtype_to_logical`]. `RawBytes` never appears in a [`ValueStorage`] (the plan only
/// ever builds one from a [`StorageKind`], never from an opaque runtime-side word buffer), so it has no
/// `DType` counterpart to return.
fn logical_to_dtype(logical: poot_target::LogicalDType) -> DType {
    match logical {
        poot_target::LogicalDType::F32 => DType::F32,
        poot_target::LogicalDType::Bf16 => DType::BF16,
        poot_target::LogicalDType::F16 => DType::F16,
        poot_target::LogicalDType::I32 => DType::I32,
        poot_target::LogicalDType::I8 => DType::I8,
        poot_target::LogicalDType::E4M3FN => DType::E4M3FN,
        poot_target::LogicalDType::RawBytes => {
            unreachable!(
                "ValueStorage is only ever built from a StorageKind, which never produces RawBytes"
            )
        }
    }
}

/// Every value's [`ValueStorage`] for one graph on one backend (card 526), computed once from the compiled plans
/// (`storage_analysis::value_storage_of_plans`). Replaces each binder's separate re-derivation of the same
/// dtype/layout predicates (R469-004).
#[derive(Debug)]
pub struct GraphStorage {
    values: Vec<ValueStorage>,
}

impl GraphStorage {
    pub(crate) fn new(values: Vec<ValueStorage>) -> Self {
        Self { values }
    }

    /// The storage planned for value `id`. Panics on an out-of-range id, like
    /// [`poot_graph_ir::Graph::aval`]: every graph value has a storage entry, computed for the whole
    /// graph at once.
    pub fn storage(&self, id: ValueId) -> ValueStorage {
        self.values[id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one field of [`ValueStorage`] is private: the planner is the only constructor.
    #[test]
    fn value_storage_dtype_and_packed() {
        let dense = ValueStorage::new(StorageKind::Dense(DType::F32));
        assert_eq!(dense.dtype(), DType::F32);
        assert!(!dense.buffer_storage().is_packed());

        let packed = ValueStorage::new(StorageKind::Bf16Packed);
        assert_eq!(packed.dtype(), DType::BF16);
        assert!(packed.buffer_storage().is_packed());

        let packed_f16 = ValueStorage::new(StorageKind::F16Packed);
        assert_eq!(packed_f16.dtype(), DType::F16);
        assert_eq!(
            packed_f16.buffer_storage(),
            poot_target::BufferStorage::f16_packed()
        );
    }

    /// Card 621: [`StorageKind::I32F32Mirror`] keeps [`ValueStorage::dtype`] at I32 (the value's
    /// declared dtype, not the physical element) while `is_f32_mirror` distinguishes it from a plain
    /// `Dense(I32)` record, and it is never packed.
    #[test]
    fn i32_f32_mirror_dtype_and_predicate() {
        let mirror = ValueStorage::new(StorageKind::I32F32Mirror);
        assert_eq!(mirror.dtype(), DType::I32);
        assert!(mirror.is_f32_mirror());
        assert!(!mirror.buffer_storage().is_packed());

        let dense_i32 = ValueStorage::new(StorageKind::Dense(DType::I32));
        assert_eq!(dense_i32.dtype(), DType::I32);
        assert!(!dense_i32.is_f32_mirror());

        let dense_f32 = ValueStorage::new(StorageKind::Dense(DType::F32));
        assert!(!dense_f32.is_f32_mirror());
    }

    /// Card 527: `ValueStorage`'s private field is `poot_target::BufferStorage` (this crate's own
    /// `DType`-keyed [`StorageKind`] is converted once, in `ValueStorage::new`), the same leaf-crate
    /// vocabulary every runtime's buffer handle carries. `dtype()`/`buffer_storage().is_packed()`
    /// round-trip through it without loss, and a packed record shares its element kind with a
    /// same-width dense record while differing in dtype and layout - the same-element,
    /// different-meaning case R484-001 describes.
    #[test]
    fn value_storage_internal_representation_is_buffer_storage() {
        let dense_f32 = ValueStorage::new(StorageKind::Dense(DType::F32));
        assert_eq!(dense_f32.storage, poot_target::BufferStorage::f32());

        let dense_i8 = ValueStorage::new(StorageKind::Dense(DType::I8));
        assert_eq!(dense_i8.dtype(), DType::I8);
        assert_eq!(
            dense_i8.storage,
            poot_target::BufferStorage::dense(
                poot_target::ElementKind::RawBytes,
                poot_target::LogicalDType::I8
            )
        );

        let packed = ValueStorage::new(StorageKind::Bf16Packed);
        assert_eq!(
            packed.storage.layout(),
            poot_target::StorageLayout::Bf16Packed
        );
        assert_eq!(packed.storage.dtype(), poot_target::LogicalDType::Bf16);
        // Canonical packed element (review F2) is I32, the word every executor's packed-BF16 upload
        // actually uses - same element kind as a dense I32 record, only dtype/layout distinguish them.
        let dense_i32 = ValueStorage::new(StorageKind::Dense(DType::I32));
        assert_eq!(packed.storage.element(), dense_i32.storage.element());
        assert_ne!(packed.storage, dense_i32.storage);
    }
}
