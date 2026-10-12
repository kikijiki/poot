//! Types shared by the planner: [`Plan`], [`PlanError`], [`CollectiveKind`], [`Dispatch`]. `Backend`
//! moved to `poot-target` (card 522 review): callers import it from there directly.
//! Re-exported at the crate root.

use crate::device_validation::DeviceWitnessRejection;
use poot_graph_ir::{ValidationId, ValueId, ValueTable};
use poot_tensor::DType;

use crate::{KernelChoice, Refusal, StorageGap};
use poot_kernel_ir::Body;

/// Why the planner produced no [`Plan`] for an equation or a graph.
#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    /// The planner built dispatches whose bindings are not a valid [`DispatchPlan`].
    #[error("invalid dispatch plan: {0}")]
    InvalidDispatchPlan(#[from] DispatchPlanError),
    /// The planner cannot lower one equation for the target (ADR-0104 decision 2). Boxed so a `Result`
    /// carrying a `PlanError` stays small.
    #[error(transparent)]
    Refused(Box<Refusal>),
    /// The graph holds a value whose storage a tensor-bound executor entry point cannot represent. The
    /// check is graph-wide and runs before any equation is planned, so it names a value, not an equation.
    #[error("{value} ({dtype}) cannot bind to this executor entry point: {gap}")]
    UnrepresentableValue {
        /// The value that cannot be represented.
        value: ValueId,
        /// Its dtype.
        dtype: DType,
        /// What about the dtype the executor entry point cannot carry.
        gap: StorageGap,
    },
    /// The op is valid but a static shape it depends on violates its precondition (e.g. an `ArgTopK`
    /// whose rank operand is 0-D). A caller-triggerable shape mismatch is a typed error, not a panic.
    #[error("bad shape for planning: {0}")]
    BadShape(String),
    /// An I32 value's buffer holds I32 words, but a planned kernel reads it through a `Slice<f32>` param, so
    /// it would read reinterpreted bits (POOT-643's class, spike-562 F-9). `value` is the buffer's root: a
    /// read through a `Plan::Alias`/`Plan::View` chain counts as a read of it. The reader is named by its
    /// output.
    #[error("I32 {value} is read through a Slice<f32> param by the equation defining {reader}")]
    I32ReadAsF32 {
        /// The buffer's root value, whose I32 words are read as `f32`.
        value: ValueId,
        /// The value defined by the reading equation.
        reader: ValueId,
    },
    /// A device executor admits only witness heads whose computation is exact on the device.
    #[error("validation {id:?} {value} is not a canonical device witness: {reason}")]
    ValidationWitnessNotCanonical {
        /// The validation's id.
        id: ValidationId,
        /// The graph value the validation reads.
        value: ValueId,
        /// Why the value is not a canonical device witness.
        reason: DeviceWitnessRejection,
    },
    /// `state_commit_from_plans` donated `state_input`, but a native transaction's rollback journal needs
    /// the exact bytes the donation overwrote, known without inspecting device data. Only
    /// `DynamicUpdateSlice` provides that; `op` names the producing equation that does not. There is no
    /// whole-buffer snapshot fallback.
    #[error(
        "state {state_input} is donated by {op}, whose overwrite region a native transaction cannot statically recover for rollback"
    )]
    UnrecoverableDonation {
        /// The state input that was donated.
        state_input: ValueId,
        /// Name of the op that produces the state's new value.
        op: String,
    },
    /// A `ScatterUpdate` writes only the rows its slot map names, into the buffer of its base: the kernel
    /// never copies the rest. Its output must therefore be donated to the base's own state buffer, which
    /// needs the base to be a carried state whose old value nothing else reads. The graph is refused
    /// instead of leaving its unnamed rows uninitialized.
    #[error(
        "{value} is a ScatterUpdate whose output is not donated to its base's state buffer: the kernel writes only the rows its slot map names, so its base must be a carried state whose old value nothing else reads"
    )]
    ScatterNotInPlace {
        /// The scatter's output value.
        value: ValueId,
    },
}

/// Which cross-device operation a communication row carries (`multi_device`). A collective is a cut
/// point between stage programs, never an equation a stage program plans. The first two variants mirror
/// the graph nodes `OpKind::AllReduce{op,axis}` / `OpKind::AllGather{axis,world}`.
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

/// One kernel launch of a [`Plan::Dispatches`] plan.
///
/// The kernel's params are `[reads.., meta?, write]`: one per [`Binding::reads`] entry in order, then the
/// launch's metadata buffer when [`Launch::meta`] is non-empty, then the one written buffer. `binding` and
/// `choice` are private and the rest is `pub(crate)`: a dispatch is built only by the planner, and a
/// [`DispatchPlan`] validates its bindings against the kernel body and launch before any other crate sees
/// it. Outside this crate the fields read through accessors; the only change after validation is the
/// planner's own launch shaping (`set_workgroup_x`, `append_key`), which cannot change the params.
#[derive(Clone, Debug)]
pub struct Dispatch {
    pub(crate) body: Body,
    pub(crate) key: String,
    /// A human-readable name for the kernel (family, form and, for a contraction, its shape):
    /// display only, carried into the per-dispatch timing so device time can be read per shape.
    pub(crate) label: String,
    pub(crate) launch: Launch,
    binding: Binding,
    choice: KernelChoice,
}

impl Dispatch {
    pub(crate) fn new(
        body: Body,
        key: String,
        label: String,
        launch: Launch,
        binding: Binding,
        choice: KernelChoice,
    ) -> Self {
        Self {
            body,
            key,
            label,
            launch,
            binding,
            choice,
        }
    }

    /// The kernel body.
    pub fn body(&self) -> &Body {
        &self.body
    }

    /// The kernel's cache key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The kernel's display label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The launch geometry and metadata.
    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    /// The planner's launch shaping: bump the workgroup's x extent. Params are unchanged.
    pub(crate) fn set_workgroup_x(&mut self, x: u32) {
        self.body.workgroup_size[0] = x;
    }

    /// The planner's launch shaping: name a changed body in the key.
    pub(crate) fn append_key(&mut self, suffix: &str) {
        self.key.push_str(suffix);
    }

    /// This dispatch under another binding: a planner stage that reads or writes a scratch, and
    /// [`Program::rebind_dispatches`].
    pub(crate) fn with_binding(self, binding: Binding) -> Self {
        Self { binding, ..self }
    }

    /// Which buffers this dispatch reads and writes.
    pub fn binding(&self) -> &Binding {
        &self.binding
    }

    /// The kernel the planner chose for this dispatch.
    pub fn choice(&self) -> &KernelChoice {
        &self.choice
    }
}

/// A plan-local scratch buffer: an intermediate that lives for one equation. The id indexes the scratch
/// buffers its [`DispatchPlan`] declares.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ScratchId(u32);

impl ScratchId {
    /// The scratch with the given index among the plan's declarations.
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    /// The scratch's index among the plan's declarations.
    pub fn index(self) -> u32 {
        self.0
    }
}

/// What a [`DispatchPlan`] declares of one scratch buffer: the storage its elements have and how many
/// device elements it holds. Contents are undefined when the equation starts (arena slots are not zeroed
/// between occupants), so the plan writes a scratch before any read of it ([`DispatchPlan::new`]).
///
/// A scratch buffer is never an export and lives for exactly its equation: the buffer plan colors it with
/// birth and death at that equation's index (POOT-ADR-0115, `BufferKey::Scratch`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScratchDecl {
    storage: poot_target::BufferStorage,
    elems: usize,
}

impl ScratchDecl {
    /// A scratch of `elems` native device elements of `storage`.
    pub fn new(storage: poot_target::BufferStorage, elems: usize) -> Self {
        Self { storage, elems }
    }

    /// The storage lane the scratch lives in.
    pub fn storage(&self) -> poot_target::BufferStorage {
        self.storage
    }

    /// The native device-element count (the unit `Device::allocate` takes), not a logical `numel`.
    pub fn elems(&self) -> usize {
        self.elems
    }

    /// The buffer's size in bytes.
    pub fn bytes(&self) -> u64 {
        self.elems as u64 * self.storage.element().byte_width() as u64
    }
}

/// Where one data param of a dispatch reads from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    /// The equation's value operand `i` (its position among [`value_ids`](crate::value_ids)).
    Operand(usize),
    /// A scratch buffer an earlier dispatch of the same plan wrote.
    Scratch(ScratchId),
}

/// Where a dispatch writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sink {
    /// The equation's output buffer.
    Out,
    /// A scratch buffer of the same plan.
    Scratch(ScratchId),
}

/// The explicit buffer binding of one [`Dispatch`]: its data reads in param order and its one write.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Binding {
    reads: Vec<Source>,
    write: Sink,
}

impl Binding {
    /// A binding that reads `reads` in param order and writes `write`.
    pub fn new(reads: Vec<Source>, write: Sink) -> Self {
        Self { reads, write }
    }

    /// The binding of a dispatch over an equation of `arity` value operands: every operand in order,
    /// writing the equation's output.
    pub fn eqn(arity: usize) -> Self {
        Self::new((0..arity).map(Source::Operand).collect(), Sink::Out)
    }

    /// The data reads, in param order.
    pub fn reads(&self) -> &[Source] {
        &self.reads
    }

    /// The one write.
    pub fn write(&self) -> Sink {
        self.write
    }
}

/// Why a set of dispatches is not a [`DispatchPlan`]. Each variant names the dispatch by its index.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DispatchPlanError {
    /// The plan has no dispatch.
    #[error("a dispatch plan needs at least one dispatch")]
    Empty,
    /// A dispatch binds a different number of params than its kernel has.
    #[error(
        "dispatch {dispatch} binds {bound} params (reads, metadata, write) but its kernel has {params}"
    )]
    ParamCount {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// Params the binding supplies (reads, metadata, write).
        bound: usize,
        /// Params the kernel declares.
        params: usize,
    },
    /// A scratch is declared in a raw-byte lane, which the executors cannot allocate as an intermediate.
    #[error(
        "scratch {scratch:?} is declared in a raw-byte lane, which the executors cannot allocate as an intermediate"
    )]
    ScratchRawBytes {
        /// The scratch.
        scratch: ScratchId,
    },
    /// A scratch declares no elements.
    #[error("scratch {scratch:?} declares no elements")]
    ScratchEmpty {
        /// The scratch.
        scratch: ScratchId,
    },
    /// A dispatch binds a scratch the plan does not declare.
    #[error("dispatch {dispatch} binds scratch {scratch:?}, but the plan declares {declared}")]
    ScratchOutOfRange {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// The scratch it binds.
        scratch: ScratchId,
        /// Number of scratch buffers the plan declares.
        declared: usize,
    },
    /// A dispatch reads a scratch no earlier dispatch wrote.
    #[error("dispatch {dispatch} reads scratch {scratch:?} that no earlier dispatch wrote")]
    ScratchReadBeforeWrite {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// The scratch it reads.
        scratch: ScratchId,
    },
    /// A dispatch reads and writes the same scratch.
    #[error("dispatch {dispatch} both reads and writes scratch {scratch:?}")]
    ScratchReadAndWritten {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// The scratch it reads and writes.
        scratch: ScratchId,
    },
    /// A param's mutability disagrees with whether its binding writes it.
    #[error(
        "dispatch {dispatch} param {param} is {}, but its binding {}",
        if *kernel_mutable { "mutable" } else { "read-only" },
        if *kernel_mutable { "reads it" } else { "writes it" }
    )]
    MutabilityMismatch {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// Index of the param among the dispatch's params.
        param: usize,
        /// Whether the kernel declares the param mutable.
        kernel_mutable: bool,
    },
    /// A dispatch reads an operand after an earlier dispatch wrote the output.
    #[error(
        "dispatch {dispatch} reads operand {operand} after an earlier dispatch wrote the output; a plan with scratch reads every operand before its first output write"
    )]
    OperandReadAfterOutWrite {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// Index of the operand.
        operand: usize,
    },
    /// A dispatch reads an operand the equation does not have.
    #[error("dispatch {dispatch} reads operand {operand} but the equation has {operands}")]
    OperandOutOfRange {
        /// Index of the dispatch within the plan.
        dispatch: usize,
        /// The operand index it reads.
        operand: usize,
        /// Number of value operands the equation has.
        operands: usize,
    },
}

/// The ordered dispatches of one equation, validated against their bindings.
///
/// **Dispatches run in order.** Dispatch `i + 1` observes every write of dispatch `i` (POOT-ADR-0115).
/// The split plans the planner emits today (GEMM tile ranges, scatter and GEMV watchdog splits, the scan
/// row split) write disjoint output ranges and do not depend on the order; a plan with scratch does.
///
/// Fields are private: [`DispatchPlan::new`] is the only constructor, so a plan an executor receives has
/// in-range scratch ids, no read of an unwritten scratch, kernel param mutability matching the bindings,
/// and (with scratch) every operand read before the first output write.
#[derive(Clone, Debug)]
pub struct DispatchPlan {
    dispatches: Vec<Dispatch>,
    scratch: Vec<ScratchDecl>,
}

impl DispatchPlan {
    /// Validate `dispatches`, which bind among the declared `scratch` buffers.
    pub fn new(
        dispatches: Vec<Dispatch>,
        scratch: Vec<ScratchDecl>,
    ) -> Result<Self, DispatchPlanError> {
        if dispatches.is_empty() {
            return Err(DispatchPlanError::Empty);
        }
        for (index, decl) in scratch.iter().enumerate() {
            let id = ScratchId::new(index as u32);
            if decl.elems == 0 {
                return Err(DispatchPlanError::ScratchEmpty { scratch: id });
            }
            if decl.storage.element() == poot_target::ElementKind::RawBytes {
                return Err(DispatchPlanError::ScratchRawBytes { scratch: id });
            }
        }
        let mut written = vec![false; scratch.len()];
        let mut out_written = false;
        for (index, dispatch) in dispatches.iter().enumerate() {
            let Binding { reads, write } = &dispatch.binding;
            let meta = usize::from(!dispatch.launch.meta.is_empty());
            let bound = reads.len() + meta + 1;
            let params: Vec<_> = dispatch.body.params().collect();
            if params.len() != bound {
                return Err(DispatchPlanError::ParamCount {
                    dispatch: index,
                    bound,
                    params: params.len(),
                });
            }
            let mutable = |param: usize| {
                matches!(
                    dispatch.body.local_ty(params[param]),
                    poot_kernel_ir::Ty::Ref { mutable: true, .. }
                )
            };
            let scratch_slot = |id: ScratchId| {
                if id.index() as usize >= scratch.len() {
                    Err(DispatchPlanError::ScratchOutOfRange {
                        dispatch: index,
                        scratch: id,
                        declared: scratch.len(),
                    })
                } else {
                    Ok(id.index() as usize)
                }
            };
            for (param, source) in reads.iter().enumerate() {
                if mutable(param) {
                    return Err(DispatchPlanError::MutabilityMismatch {
                        dispatch: index,
                        param,
                        kernel_mutable: true,
                    });
                }
                match *source {
                    Source::Operand(operand) => {
                        if !scratch.is_empty() && out_written {
                            return Err(DispatchPlanError::OperandReadAfterOutWrite {
                                dispatch: index,
                                operand,
                            });
                        }
                    }
                    Source::Scratch(id) => {
                        if !written[scratch_slot(id)?] {
                            return Err(DispatchPlanError::ScratchReadBeforeWrite {
                                dispatch: index,
                                scratch: id,
                            });
                        }
                    }
                }
            }
            if meta == 1 && mutable(reads.len()) {
                return Err(DispatchPlanError::MutabilityMismatch {
                    dispatch: index,
                    param: reads.len(),
                    kernel_mutable: true,
                });
            }
            if !mutable(bound - 1) {
                return Err(DispatchPlanError::MutabilityMismatch {
                    dispatch: index,
                    param: bound - 1,
                    kernel_mutable: false,
                });
            }
            match *write {
                Sink::Out => out_written = true,
                Sink::Scratch(id) => {
                    let slot = scratch_slot(id)?;
                    if reads.contains(&Source::Scratch(id)) {
                        return Err(DispatchPlanError::ScratchReadAndWritten {
                            dispatch: index,
                            scratch: id,
                        });
                    }
                    written[slot] = true;
                }
            }
        }
        Ok(Self {
            dispatches,
            scratch,
        })
    }

    /// The scratch buffers the plan declares, indexed by [`ScratchId`].
    pub fn scratch(&self) -> &[ScratchDecl] {
        &self.scratch
    }

    /// The one dispatch of a plan that launches exactly one kernel.
    pub fn single(&self) -> Option<&Dispatch> {
        match self.dispatches.as_slice() {
            [dispatch] => Some(dispatch),
            _ => None,
        }
    }

    pub(crate) fn single_mut(&mut self) -> Option<&mut Dispatch> {
        match self.dispatches.as_mut_slice() {
            [dispatch] => Some(dispatch),
            _ => None,
        }
    }
}

/// The shape of one launch: the total thread count and the optional metadata buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    /// The total thread count (`[x, y, z]`); the executor divides it by `body.workgroup_size` to get the
    /// dispatch's workgroup counts. The planner sets it once, alongside the kernel choice that shapes it
    /// (card 525); no executor recomputes it.
    pub grid: [u32; 3],
    /// One read-only `u32` metadata buffer bound between the eqn's data inputs and the output (param
    /// order `[data inputs.., meta, out]`); empty binds none. An imported shape-generic kernel gets its
    /// dims this way instead of baked consts (card 044: batched decode GEMV takes `dims = [B]`).
    pub meta: Vec<u32>,
}

/// Read-only: a validated plan's dispatches are indexed, iterated and counted as a slice.
impl std::ops::Deref for DispatchPlan {
    type Target = [Dispatch];

    fn deref(&self) -> &[Dispatch] {
        &self.dispatches
    }
}

/// How to execute one eqn (independent of host vs device buffers, and of the backend).
#[derive(Debug)]
pub enum Plan {
    /// Dispatch these kernels in order, each over the buffers its own [`Binding`] names. Never empty. A
    /// plan holds several dispatches when one launch cannot cover the output: a tiled GEMM split into
    /// disjoint sub-2^15 tile ranges (the RADV 2^15-workgroup miscompile), a scatter / decode GEMV split
    /// to stay under the device watchdog, or a scan split by rows. Those write disjoint output ranges; the
    /// order of dispatches is nonetheless part of the contract (see [`DispatchPlan`]).
    Dispatches(DispatchPlan),
    /// The output aliases the given input value's buffer (reshape).
    Alias(ValueId),
    /// The output is a strided view of `src`'s buffer (spec 132): no dispatch, like [`Plan::Alias`], but
    /// with its own (possibly non-row-major) `strides` and element `offset`. Emitted only when every
    /// consumer can read a strided input (`is_strided_capable`); see `compute_views`. Reshape still lowers
    /// to `Plan::Alias`; this variant is for Transpose/Slice/Broadcast.
    View {
        /// The value whose buffer the view reads.
        src: ValueId,
        /// Element strides of the view, per output axis.
        strides: Vec<usize>,
        /// Element offset of the view's first element in the source buffer.
        offset: usize,
    },
}

/// Which [`Plan`] variant an equation planned as, without its payload. Executors name it when a walk
/// cannot run the plan it was given.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PlanKind {
    /// The equation plans as one or more kernel dispatches.
    Dispatches,
    /// The output aliases an input's buffer.
    Alias,
    /// The output is a strided view of a source buffer.
    View,
}

impl Plan {
    /// The one dispatch of a plan that launches exactly one kernel; `None` for a split plan and for a
    /// plan with no launch.
    pub fn single(&self) -> Option<&Dispatch> {
        match self {
            Plan::Dispatches(dispatches) => dispatches.single(),
            _ => None,
        }
    }

    /// [`Plan::single`], mutably.
    pub(crate) fn single_mut(&mut self) -> Option<&mut Dispatch> {
        match self {
            Plan::Dispatches(dispatches) => dispatches.single_mut(),
            _ => None,
        }
    }

    /// Every kernel launch the plan makes, in order; empty for a plan with no launch.
    pub fn dispatches(&self) -> &[Dispatch] {
        match self {
            Plan::Dispatches(dispatches) => dispatches,
            _ => &[],
        }
    }

    /// Which variant of [`Plan`] this is, without its payload.
    pub fn kind(&self) -> PlanKind {
        match self {
            Plan::Dispatches(_) => PlanKind::Dispatches,
            Plan::Alias(_) => PlanKind::Alias,
            Plan::View { .. } => PlanKind::View,
        }
    }
}

/// How [`ValueStorage`] packs its dtype into a device buffer (card 526).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StorageKind {
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
    /// E4M3FN checkpoint bytes packed four elements per device `u32` word, each row of the last axis
    /// padded to a whole word (POOT-1060): a `DenseRowGather` table, which every backend's imported
    /// `PackedE4m3RowGather` body decodes in-register. Always E4M3FN.
    E4m3Packed,
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
            StorageKind::E4m3Packed => poot_target::BufferStorage::e4m3_packed(),
        };
        Self { storage }
    }

    /// This record in the leaf-crate vocabulary a buffer handle's storage contract shares (card 527): every binder reads this at bind time and compares it against the handle it uploaded
    /// through, returning a typed mismatch when they disagree - the check that actually closes
    /// R484-001/R471-009, as opposed to a check a test can only reach by forging a handle's field.
    pub fn buffer_storage(&self) -> poot_target::BufferStorage {
        self.storage
    }

    /// The dtype a binder uploads: `Dense`'s own dtype, or the packed dtype (`BF16`, `F16` or `E4M3FN`) for a
    /// packed value (still checkpoint bytes, just packed several elements per device word).
    pub fn dtype(&self) -> DType {
        logical_to_dtype(self.storage.dtype())
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
    values: ValueTable<ValueStorage>,
}

impl GraphStorage {
    pub(crate) fn new(values: ValueTable<ValueStorage>) -> Self {
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
    use poot_kernel_ir::{BasicBlock, LocalDecl, Terminator, Ty};

    use super::*;
    use crate::NoDispatch;

    /// A kernel whose params are the given mutabilities, in order (`true` is `&mut [f32]`).
    fn body(params: &[bool]) -> Body {
        let slice = |mutable| LocalDecl {
            ty: Ty::Ref {
                mutable,
                pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
            },
            mutable: false,
        };
        let locals = std::iter::once(LocalDecl {
            ty: Ty::Unit,
            mutable: false,
        })
        .chain(params.iter().map(|&mutable| slice(mutable)))
        .collect();
        Body::new(
            "k",
            params.len() as u32,
            locals,
            vec![BasicBlock {
                statements: vec![],
                terminator: Terminator::Return,
            }],
        )
    }

    /// A dispatch of a two-read kernel `[a, b, out]` with the given binding.
    fn dispatch(reads: [Source; 2], write: Sink) -> Dispatch {
        dispatch_of(
            body(&[false, false, true]),
            Binding::new(reads.to_vec(), write),
        )
    }

    fn dispatch_of(body: Body, binding: Binding) -> Dispatch {
        Dispatch::new(
            body,
            "k".into(),
            "k".into(),
            Launch {
                grid: [1, 1, 1],
                meta: Vec::new(),
            },
            binding,
            KernelChoice::NoDispatch(NoDispatch::Alias),
        )
    }

    const SCRATCH: ScratchId = ScratchId(0);

    /// `n` one-element f32 scratch declarations.
    fn decls(n: usize) -> Vec<ScratchDecl> {
        vec![ScratchDecl::new(poot_target::BufferStorage::f32(), 1); n]
    }

    fn error_of(dispatches: Vec<Dispatch>, scratch: usize) -> DispatchPlanError {
        DispatchPlan::new(dispatches, decls(scratch)).expect_err("the plan must be refused")
    }

    #[test]
    fn a_plan_with_no_dispatch_is_refused() {
        assert_eq!(error_of(Vec::new(), 0), DispatchPlanError::Empty);
    }

    #[test]
    fn split_plans_and_a_scratch_pipeline_are_accepted() {
        let eqn = || dispatch([Source::Operand(0), Source::Operand(1)], Sink::Out);
        let plan = DispatchPlan::new(vec![eqn(), eqn()], Vec::new())
            .expect("two disjoint-range dispatches");
        assert_eq!(plan.len(), 2);
        // Stage one writes the scratch from the operands; stage two reads it and writes the output.
        let plan = DispatchPlan::new(
            vec![
                dispatch(
                    [Source::Operand(0), Source::Operand(1)],
                    Sink::Scratch(SCRATCH),
                ),
                dispatch([Source::Scratch(SCRATCH), Source::Operand(0)], Sink::Out),
            ],
            decls(1),
        )
        .expect("a scratch written before it is read");
        assert_eq!(plan.scratch(), decls(1).as_slice());
    }

    /// A scratch of no elements or in a raw-byte lane is not a buffer an executor can hand a kernel.
    #[test]
    fn an_empty_or_raw_byte_scratch_is_refused() {
        let writes = || {
            vec![dispatch(
                [Source::Operand(0), Source::Operand(1)],
                Sink::Out,
            )]
        };
        let empty = ScratchDecl::new(poot_target::BufferStorage::f32(), 0);
        assert_eq!(
            DispatchPlan::new(writes(), vec![empty]).expect_err("an empty scratch"),
            DispatchPlanError::ScratchEmpty { scratch: SCRATCH }
        );
        let raw = ScratchDecl::new(poot_target::BufferStorage::raw_bytes(), 4);
        assert_eq!(
            DispatchPlan::new(writes(), vec![raw]).expect_err("a raw-byte scratch"),
            DispatchPlanError::ScratchRawBytes { scratch: SCRATCH }
        );
    }

    #[test]
    fn a_scratch_id_beyond_the_declared_buffers_is_refused() {
        let error = error_of(
            vec![dispatch(
                [Source::Operand(0), Source::Scratch(SCRATCH)],
                Sink::Out,
            )],
            0,
        );
        assert_eq!(
            error,
            DispatchPlanError::ScratchOutOfRange {
                dispatch: 0,
                scratch: SCRATCH,
                declared: 0,
            }
        );
    }

    #[test]
    fn a_scratch_read_before_any_write_is_refused() {
        let error = error_of(
            vec![
                dispatch([Source::Scratch(SCRATCH), Source::Operand(0)], Sink::Out),
                dispatch(
                    [Source::Operand(0), Source::Operand(1)],
                    Sink::Scratch(SCRATCH),
                ),
            ],
            1,
        );
        assert_eq!(
            error,
            DispatchPlanError::ScratchReadBeforeWrite {
                dispatch: 0,
                scratch: SCRATCH,
            }
        );
    }

    #[test]
    fn a_dispatch_that_reads_and_writes_one_scratch_is_refused() {
        let error = error_of(
            vec![
                dispatch(
                    [Source::Operand(0), Source::Operand(1)],
                    Sink::Scratch(SCRATCH),
                ),
                dispatch(
                    [Source::Scratch(SCRATCH), Source::Operand(1)],
                    Sink::Scratch(SCRATCH),
                ),
            ],
            1,
        );
        assert_eq!(
            error,
            DispatchPlanError::ScratchReadAndWritten {
                dispatch: 1,
                scratch: SCRATCH,
            }
        );
    }

    #[test]
    fn kernel_param_mutability_must_match_the_binding() {
        let reads = || vec![Source::Operand(0), Source::Operand(1)];
        // The kernel writes its first param, which the binding reads.
        let writes_a_read =
            dispatch_of(body(&[true, false, true]), Binding::new(reads(), Sink::Out));
        assert_eq!(
            error_of(vec![writes_a_read], 0),
            DispatchPlanError::MutabilityMismatch {
                dispatch: 0,
                param: 0,
                kernel_mutable: true,
            }
        );
        // The kernel's last param is read-only, but the binding writes it.
        let reads_its_write = dispatch_of(
            body(&[false, false, false]),
            Binding::new(reads(), Sink::Out),
        );
        assert_eq!(
            error_of(vec![reads_its_write], 0),
            DispatchPlanError::MutabilityMismatch {
                dispatch: 0,
                param: 2,
                kernel_mutable: false,
            }
        );
    }

    #[test]
    fn a_binding_must_cover_the_kernels_params() {
        let two_reads_for_a_one_read_kernel = dispatch_of(
            body(&[false, true]),
            Binding::new(vec![Source::Operand(0), Source::Operand(1)], Sink::Out),
        );
        assert_eq!(
            error_of(vec![two_reads_for_a_one_read_kernel], 0),
            DispatchPlanError::ParamCount {
                dispatch: 0,
                bound: 3,
                params: 2,
            }
        );
    }

    /// A donated output shares an operand's buffer, so once a plan with scratch has written `Out` an
    /// operand read would see the overwritten bytes. Without scratch the split plans' ranges are disjoint
    /// and the rule does not apply.
    #[test]
    fn an_operand_read_after_the_output_write_is_refused_in_a_plan_with_scratch() {
        let write_out = || dispatch([Source::Operand(0), Source::Operand(1)], Sink::Out);
        let write_scratch = || {
            dispatch(
                [Source::Operand(0), Source::Operand(1)],
                Sink::Scratch(SCRATCH),
            )
        };
        let error = error_of(vec![write_scratch(), write_out(), write_out()], 1);
        assert_eq!(
            error,
            DispatchPlanError::OperandReadAfterOutWrite {
                dispatch: 2,
                operand: 0,
            }
        );
        DispatchPlan::new(vec![write_out(), write_out()], Vec::new())
            .expect("disjoint-range splits read their operands after the first output write");
    }

    /// The one field of [`ValueStorage`] is private: the planner is the only constructor.
    #[test]
    fn value_storage_dtype_and_packed() {
        let dense = ValueStorage::new(StorageKind::Dense(DType::F32));
        assert_eq!(dense.dtype(), DType::F32);
        assert_eq!(
            dense.buffer_storage().layout(),
            poot_target::StorageLayout::Dense
        );

        let packed = ValueStorage::new(StorageKind::Bf16Packed);
        assert_eq!(packed.dtype(), DType::BF16);
        assert_eq!(
            packed.buffer_storage().layout(),
            poot_target::StorageLayout::Bf16Packed
        );

        let packed_f16 = ValueStorage::new(StorageKind::F16Packed);
        assert_eq!(packed_f16.dtype(), DType::F16);
        assert_eq!(
            packed_f16.buffer_storage(),
            poot_target::BufferStorage::f16_packed()
        );
    }

    /// Card 527: `ValueStorage`'s private field is `poot_target::BufferStorage` (this crate's own
    /// `DType`-keyed [`StorageKind`] is converted once, in `ValueStorage::new`), the same leaf-crate
    /// vocabulary every runtime's buffer handle carries. `dtype()`/`buffer_storage().layout()`
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

    impl DispatchPlan {
        /// The dispatches by value, for a test that takes a plan apart.
        pub(crate) fn into_dispatches(self) -> Vec<Dispatch> {
            self.dispatches
        }
    }
}
