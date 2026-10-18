//! The backend seam (dexec 3.1, spike 546 section 4): one backend's mechanism, nothing semantic. No
//! graph, no plan walk, no slot kinds, no masks. Donation is not a `Device` capability:
//! donated state updates in place on every backend, decided by the engine from the state-commit plan,
//! never asked of the device.

use poot_runtime_common::{BufferRole, MemoryCounterSnapshot};
use std::num::NonZeroU32;
use std::ops::Range;
use std::time::Duration;

use poot_graph_plan::{Target, production_plan::Submission};
use poot_runtime_common::CompiledKernel;
use poot_target::{AieTargetId, BufferStorage, VpuTargetId};

/// One dispatch's device duration within a step's recording, in replay order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchDeviceTime {
    /// Order within the step's recording/replay (the same order the entry's `LoadedStep`s were
    /// walked in, including when sampling leaves gaps: a partial sample keeps its real index rather
    /// than shifting to stay contiguous, Card 552 scope).
    pub index: usize,
    /// This dispatch's own `[start, end)` device duration.
    pub duration: Duration,
}

/// At least one of a step's two independent device-time measures (Card 552): never one ambiguous
/// total. Either may be absent (`None`) on a backend/mode that cannot produce it; absence here is
/// still distinct from the whole-step [`DeviceTime::Unknown`], which means neither was even
/// attempted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeasuredDeviceTime {
    /// Sum of each dispatch's own `[start, end)` device duration. Can exceed `device_span` under
    /// device-side concurrency (never collapsed into one number with it, SC-005).
    pub sum_of_dispatch_durations: Option<Duration>,
    /// First dispatch's start to last dispatch's end, this device's own clock domain.
    pub device_span: Option<Duration>,
    /// Per-dispatch durations, present only when the device was asked for detailed collection
    /// (`TimingOptions::Detailed`); empty in counters-only mode (a coverage distinction, never a
    /// zero-length "complete" detail list).
    pub dispatches: Vec<DispatchDeviceTime>,
}

/// Device time of the last synchronized step. Never host wall (Card 552): a device that cannot
/// report timestamps at all returns `Unknown`, never a fabricated zero (SC-004); `Measured` is
/// reported only once the device actually attempted to measure this step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DeviceTime {
    /// The device reported no timestamps for the step: not a zero duration.
    #[default]
    Unknown,
    /// The device attempted to measure the step; each measure inside may still be absent.
    Measured(MeasuredDeviceTime),
}

/// One kernel-dispatch argument: the buffer, and this value's own native device-element count (Card
/// 546a F8, Card 547b; review F5: not a logical tensor `numel` - a packed source
/// publishes its lane-encoded word count, e.g. two BF16 elements per `u32`, exactly what
/// `BufferStorage::device_elems` computed for it). `Dispatch` carries each argument's own element
/// count from the plan, and every runtime reads it instead of the bound buffer's own capacity - an
/// arena slot's buffer (Card 547b) is sized to its largest occupant, so a smaller value sharing it
/// would otherwise let a kernel read or write past its own operand.
pub struct Arg<'a, D: Device + ?Sized> {
    /// The device buffer the kernel reads or writes.
    pub buffer: &'a D::Buffer,
    /// Native device-element count of this value (lane-encoded words for a packed source), not the buffer's capacity.
    pub elems: u32,
}

/// How a dispatch launches: the execution domain of its kernel (POOT-1117). The engine is one engine over
/// every backend, so the domain is a closed enum on the dispatch rather than a second device trait: a
/// device matches it explicitly and refuses the launch kind it does not run, with a typed error, never a
/// silent arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchKind {
    /// A thread grid: `threads` is the plan's total thread grid (Card 525); the device divides it by
    /// `workgroup`.
    Grid {
        /// Total thread grid per axis, in threads (not workgroups).
        threads: [u32; 3],
        /// Threads per workgroup per axis; `threads` is divided by it.
        workgroup: [u32; 3],
    },
    /// One finite run of a loaded AIE kernel's instruction stream over the dispatch's host buffer objects.
    Aie(AieInvocation),
    /// One execution of a loaded Intel NPU graph over the dispatch's host buffers.
    VpuGraph(VpuInvocation),
}

/// One execution of an Intel NPU graph: the graph runs once over the dispatch's host buffers (the `inputs`
/// then the `output`, in the graph's argument order: its inputs, then its results) and ends; there is no
/// grid. It names the target the graph was compiled for and the digest of the artifact the engine loaded
/// (the IR and argument ABI, [`poot_runtime_common::VpuGraphArtifact::digest`]). A device implementation
/// must refuse an invocation that disagrees with the kernel it carries; the other devices read only the
/// variant, to refuse the launch kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VpuInvocation {
    target: VpuTargetId,
    digest: u64,
}

impl VpuInvocation {
    /// An invocation of the graph with `digest` compiled for `target`.
    pub const fn new(target: VpuTargetId, digest: u64) -> Self {
        Self { target, digest }
    }

    /// The target the graph was compiled for.
    pub const fn target(&self) -> VpuTargetId {
        self.target
    }

    /// The digest of the artifact the engine loaded.
    pub const fn digest(&self) -> u64 {
        self.digest
    }
}

/// One run of an AIE kernel: its instruction stream executes once over the dispatch's host buffer objects
/// (the `inputs` then the `output`, in the kernel's shape-ABI binding order) and ends; there is no grid.
/// It names the target the kernel was linked for and the stream's length in 32-bit instruction words (a
/// loaded artifact's stream is validated non-empty and word-aligned). A device implementation (POOT-641)
/// must read `instruction_words` and refuse an invocation that disagrees with the stream of the kernel it
/// loaded; the GPU devices read only `target`, to refuse the launch kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AieInvocation {
    target: AieTargetId,
    instruction_words: NonZeroU32,
}

impl AieInvocation {
    /// An invocation of a stream of `instruction_words` words linked for `target`.
    pub const fn new(target: AieTargetId, instruction_words: NonZeroU32) -> Self {
        Self {
            target,
            instruction_words,
        }
    }

    /// The target the kernel was linked for.
    pub const fn target(&self) -> AieTargetId {
        self.target
    }

    /// The instruction stream's length, in 32-bit words.
    pub const fn instruction_words(&self) -> NonZeroU32 {
        self.instruction_words
    }
}

/// One kernel launch. `inputs` are the kernel's read operands in parameter order (including a metadata
/// buffer), `output` its written operand. `launch` is how the kernel runs (a thread grid, or an AIE
/// instruction stream). `work` is the planner-owned watchdog weight: the device uses it to decide when a
/// long run of dispatches must flush to a new submit before the driver's TDR fires.
pub struct Dispatch<'a, D: Device + ?Sized> {
    /// The loaded kernel to run.
    pub kernel: &'a D::Kernel,
    /// Read operands in the kernel's parameter order, including any metadata buffer.
    pub inputs: &'a [Arg<'a, D>],
    /// The written operand.
    pub output: Arg<'a, D>,
    /// How the kernel launches: a thread grid or an AIE instruction stream.
    pub launch: LaunchKind,
    /// Planner-owned watchdog weight, used to flush a long run of dispatches to a new submit before the driver's TDR fires.
    pub work: u64,
}

/// What a device does with a weight's bytes when the engine uploads it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WeightUpload {
    /// The engine reads the weight from the store, encodes it into the planned lane and writes it.
    #[default]
    Bytes,
    /// The device only sizes the weight: the engine allocates its buffer from the shape (or the stored length
    /// of a packed source) and never reads, encodes or writes the bytes. For a device that records
    /// allocations to plan a footprint and executes nothing. The checks that need the bytes (a non-finite
    /// packed-F16 element) are the executing upload's; the dtype and lane checks run in both modes.
    SizeOnly,
}

/// One backend's mechanism below the executor contract (dexec 3.1). Every method is a scoped
/// transaction's primitive; the engine (`crate::Engine`) owns the transaction itself: it calls
/// [`Device::abort`] on any failure of `begin`/`dispatch`/`copy`/`finish`, exactly once, and never
/// retries or falls back (core review CR30).
pub trait Device {
    /// Owning buffer handle. Dropping it frees the allocation.
    type Buffer: 'static;
    /// A loaded kernel (wgpu pipeline source, HSA code object, CUfunction).
    type Kernel: 'static;
    /// A recorded step: replayable without re-walking the program.
    type Recording: 'static;
    /// The device's failure type; the engine wraps it into [`crate::DeviceError`].
    type Error: std::error::Error + Send + Sync + 'static;

    /// The compile target this device executes.
    fn target(&self) -> Target;

    /// What this device does with weight bytes. Every executing device takes them; only a recording-only
    /// planning device sizes them.
    fn weight_upload(&self) -> WeightUpload {
        WeightUpload::Bytes
    }

    /// Live/peak bytes and allocation count by role, read from this device's own runtime counters
    /// (Card 547a: `Engine::stats().memory` reads this, not an engine-side tally over its own
    /// bookkeeping, so cross-backend equality is not equal by construction).
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)>;

    /// Device memory this process could still allocate right now, in bytes, read from the runtime when
    /// asked (not cached from open). `None` where the runtime cannot say or the query failed: a
    /// consumer treats that as an unknown budget, never as unlimited. The default is `None`, so a
    /// device that cannot observe its memory says so by not overriding it.
    fn free_memory_bytes(&self) -> Option<u64> {
        None
    }

    /// A zero-filled buffer holding `elems` native elements of `storage`.
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error>;

    /// The number of native elements `buffer` holds: its allocated capacity, in units of the element kind it
    /// was allocated with (what [`Arg::elems`] counts). The engine refuses a dispatch whose asserted
    /// length exceeds it before any device sees the dispatch ([`crate::DispatchError::ArgExceedsBuffer`]),
    /// so a kernel may trust the lengths it is handed.
    fn buffer_elems(&self, buffer: &Self::Buffer) -> usize;

    /// Load one compiled kernel. `key` is the plan's content key (kernel caches and labels use it).
    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error>;

    /// Start a step's recording transaction: opens a recording that `finish` closes and `replay`
    /// later submits, regardless of `submission`'s value. `submission` names the contract's
    /// admitted mode (`Engine::add_entry` refuses `Submission::Eager` before `begin` ever runs); no
    /// current `Device` impl gives `Eager` a different meaning here, including a caller driving one
    /// directly, outside the contract (`poot-llm/tests/wgpu_executor_contract.rs`'s SC-015 fault probe
    /// does exactly that).
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error>;
    /// Record one kernel launch into the open recording, ordered after every earlier call of this step.
    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), Self::Error>;
    /// Copy all of `src` into `dst` (same byte length), ordered after every earlier call of this step.
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error>;
    /// End a step, returning the recording `replay` later submits. Every current `Device` impl
    /// returns `Some` unconditionally: nothing has run on the device yet, regardless of
    /// `submission`.
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error>;
    /// Start a step's replays: called once per step, before the first of its (one per stage)
    /// [`Self::replay`] calls. A backend resets whatever it accumulates per step here (device-timing
    /// records), so a step that failed before `device_time` drained them cannot leak into this one
    /// and the replays of a multi-stage step accumulate instead of overwriting each other.
    fn begin_step(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Submit one recording. A step replays one recording per stage, in stage order, between one
    /// [`Self::begin_step`] and one [`Self::synchronize`].
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error>;

    /// Overwrite `len` bytes of `dst` from byte `offset`, ordered before the next submitted step. A
    /// range outside `dst` is a device error. The device hands out host memory it can transfer from:
    /// `fill(range, out)` is called with consecutive, non-overlapping destination ranges covering
    /// `offset..offset + len` in ascending order, with `out.len() == range.len()`, and must write
    /// every byte of `out`. The device bounds its staging memory: it may wait for an earlier chunk's
    /// transfer before the next call, and `fill` has returned for a range before that range's
    /// transfer starts. A device that writes whole words (wgpu) takes a non-zero `offset` in multiples
    /// of four and zero-pads a trailing partial word.
    ///
    /// This is the one way bytes reach a buffer. A caller that holds its bytes already uses [`write_slice`],
    /// the one named helper over this method (a free function, not a second trait method: no device
    /// implements it), which passes a `fill` that copies them. A weight's encoder instead writes each chunk
    /// straight into the device's staging, so no host copy of the weight exists beyond the staging itself
    /// (dmap.md section 6).
    fn write_with(
        &mut self,
        dst: &Self::Buffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(Range<usize>, &mut [u8]),
    ) -> Result<(), Self::Error>;
    /// Read all of `src` into `out` after every submitted step finishes.
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error>;
    /// Wait for the submitted work and raise any staged kernel-assert fault (Card 531c).
    fn synchronize(&mut self) -> Result<(), Self::Error>;
    /// The device time of the last synchronized step; [`DeviceTime::Unknown`] when the device cannot measure.
    fn device_time(&self) -> DeviceTime;

    /// Abort a scoped transaction after an unsuccessful `begin`/`dispatch`/`copy`/`finish`: publish no
    /// partial recording, and leave the attempted recording in no open state. Safely handles a
    /// partially begun or finished recording; if cleanup cannot establish a reusable state, this
    /// implementation poisons the device internally and `Engine` reports it as such (core review
    /// CR30). Called by `Engine` exactly once per failed transaction, never speculatively.
    fn abort(&mut self) -> Result<(), Self::Error>;

    /// `(kernel, code)` when `error` (from [`Self::synchronize`]) is a staged Card 531c kernel-assert
    /// fault, so `Engine::step` can surface it as [`crate::ExecError::Fault`] - a program result, not
    /// a device-level failure - instead of the generic device-error wrap every other
    /// backend error gets. `None` (the default) for a backend with no fault classification yet.
    fn classify_fault(&self, _error: &Self::Error) -> Option<(String, u32)> {
        None
    }
}

/// Write host `bytes` that are already in hand to `dst` from byte `offset`: [`Device::write_with`] with a
/// `fill` that copies them. A slot, a state reset, a computed constant and a test fixture have their bytes
/// before the device asks for them; a weight's encoder writes into the device's staging directly instead.
pub fn write_slice<D: Device>(
    device: &mut D,
    dst: &D::Buffer,
    offset: usize,
    bytes: &[u8],
) -> Result<(), D::Error> {
    device.write_with(dst, offset, bytes.len(), &mut |range, out| {
        out.copy_from_slice(&bytes[range.start - offset..range.end - offset]);
    })
}

/// The native objects a device created and the replays it issued: what a warm replay must leave
/// unchanged. Counted by the device's runtime, never by the engine over its own bookkeeping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeCounters {
    /// Replayable native graphs instantiated from a recording.
    pub graph_instantiations: u64,
    /// Native graph launches.
    pub graph_launches: u64,
    /// Cross-device ordering events created.
    pub events: u64,
    /// Kernel modules loaded.
    pub modules: u64,
    /// Peer copies recorded.
    pub peer_copies: u64,
}

/// A conservative byte bound for each native object a recording session creates beyond the buffers the
/// engine allocates. A driver does not report the size of a graph or an event, so the device states the
/// bound its admission may rely on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SupportBound {
    /// One replayable native graph.
    pub graph_bytes: u64,
    /// One cross-device ordering event.
    pub event_bytes: u64,
}

/// What a device adds to take part in a multi-device program: peer copies recorded into the same
/// recording as its dispatches, and ordering events between devices. Only a backend that can do both
/// implements it; a coordinator over a device without it does not compile.
///
/// There is one recording model: a stage is one `begin`, its dispatches, any peer copy of its exported
/// values, and one `finish`; its recording is replayed with [`Device::replay`] like any other. Ordering
/// between devices is by event, at replay: the coordinator records a stage's event on its device right
/// after replaying the stage, and makes the consumer's device wait for it right before replaying the
/// consumer, so every wait names the record the host enqueued for the same step.
pub trait Collectives: Device {
    /// A cross-device ordering point: recorded by the device that created it, waited for by another.
    type Event: 'static;

    /// The native objects this device created and the replays it issued, counted by its runtime.
    fn native_counters(&self) -> NativeCounters;

    /// The conservative byte bound this device states for each native object a recording session creates.
    fn support_bound(&self) -> SupportBound;

    /// Admit the directed route from this device to `peer`, outside any recording. A route that would
    /// need a host-staged copy is an error: a recording cannot hold it, and no fallback exists.
    fn open_route(&mut self, peer: &Self) -> Result<(), Self::Error>;

    /// A new event owned by this device. Outside any recording.
    fn create_event(&mut self) -> Result<Self::Event, Self::Error>;

    /// Record `event` now, outside any recording: it completes when everything this device has been
    /// handed so far (the recording just replayed) has.
    fn record_event(&mut self, event: &Self::Event) -> Result<(), Self::Error>;

    /// Make this device wait, from this call on, for the record of `event` the host enqueued before
    /// it: the next replay does not start until that record completes. Outside any recording.
    fn wait_event(&mut self, event: &Self::Event) -> Result<(), Self::Error>;

    /// Copy the first `bytes` bytes of `src` (this device's) into `dst` (a peer's, on a route
    /// [`Self::open_route`] admitted), recorded into the open recording after everything it holds so
    /// far.
    fn peer_copy(
        &mut self,
        src: &Self::Buffer,
        dst: &Self::Buffer,
        bytes: usize,
    ) -> Result<(), Self::Error>;
}
