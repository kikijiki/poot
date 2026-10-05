//! The backend seam (dexec 3.1, spike 546 section 4): one backend's mechanism, nothing semantic. No
//! graph, no plan walk, no slot kinds, no masks. Donation is not a `Device` capability:
//! donated state updates in place on every backend, decided by the engine from the state-commit plan,
//! never asked of the device.

use std::time::Duration;

use poot_graph_plan::{Submission, Target};
use poot_runtime_common::CompiledKernel;
pub use poot_runtime_common::{BufferRole, MemoryCounterSnapshot};
use poot_target::BufferStorage;

/// One dispatch's device duration within a step's recording, in replay order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchDeviceTime {
    /// Order within the step's recording/replay (the same order the entry's `LoadedStep`s were
    /// walked in, including when sampling leaves gaps: a partial sample keeps its real index rather
    /// than shifting to stay contiguous, Card 552 scope).
    pub index: usize,
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
    #[default]
    Unknown,
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
    pub buffer: &'a D::Buffer,
    pub elems: u32,
}

/// One kernel launch. `inputs` are the kernel's read operands in parameter order (including a metadata
/// buffer), `output` its written operand. `threads` is the plan's total thread grid (Card 525); the
/// device divides it by `workgroup`. `work` is the planner-owned watchdog weight: the
/// device uses it to decide when a long run of dispatches must flush to a new submit before the
/// driver's TDR fires.
pub struct Dispatch<'a, D: Device + ?Sized> {
    pub kernel: &'a D::Kernel,
    pub inputs: &'a [Arg<'a, D>],
    pub output: Arg<'a, D>,
    pub threads: [u32; 3],
    pub workgroup: [u32; 3],
    pub work: u64,
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
    type Error: std::error::Error + Send + Sync + 'static;

    fn target(&self) -> Target;

    /// Live/peak bytes and allocation count by role, read from this device's own runtime counters
    /// (Card 547a: `Engine::stats().memory` reads this, not an engine-side tally over its own
    /// bookkeeping, so cross-backend equality is not equal by construction).
    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)>;

    /// A zero-filled buffer holding `elems` native elements of `storage`.
    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error>;

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
    /// directly, outside the contract (`poot-gpu/tests/executor_contract.rs`'s SC-015 fault probe
    /// does exactly that).
    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error>;
    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), Self::Error>;
    /// Copy all of `src` into `dst` (same byte length), ordered after every earlier call of this step.
    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error>;
    /// End a step, returning the recording `replay` later submits. Every current `Device` impl
    /// returns `Some` unconditionally: nothing has run on the device yet, regardless of
    /// `submission`.
    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error>;
    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error>;

    /// Overwrite all of `dst` with `bytes`, ordered before the next submitted step.
    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error>;
    /// Read all of `src` into `out` after every submitted step finishes.
    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error>;
    /// Wait for the submitted work and raise any staged kernel-assert fault (Card 531c).
    fn synchronize(&mut self) -> Result<(), Self::Error>;
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
