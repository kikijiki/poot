//! [`XdnaDevice`]: the `poot_executor::Device` of the AMD XDNA2 NPU over XRT.
//!
//! There is one engine over every backend, so this is a thin mechanism below the executor contract: it
//! allocates host-mapped buffer objects, loads a linked artifact as an XRT hardware context, records the
//! engine's dispatches and copies, and replays a recording by running each kernel once and waiting for it.
//! It recognizes no graph and links nothing: the compiler (`poot-codegen`'s `compile_aie`) produced the
//! artifact, the planner admitted the shape, and the engine's AIE arm checked the artifact against
//! [`AieCaps`] before this device was handed it.
//!
//! What this device checks on its own, before any XRT work, are the things only it can see: the kernel is
//! an AIE artifact for this target, the xclbin's declared interface is the one the link driver builds, and
//! each dispatch's launch, invocation and buffers are the kernel's. Each is a typed [`XdnaError`].
//!
//! Completion is synchronous: [`Device::replay`] runs every dispatch of the recording and returns when the
//! last one completed (or failed), so [`Device::synchronize`] has nothing in flight. Any failed run poisons
//! the device (a hardware context that timed out is not known to be idle) and keeps the failed run's
//! buffers alive, charged, until the device itself is dropped.

use std::rc::Rc;
use std::time::Duration;

use poot_executor::{
    AieInvocation, Arg, BufferRole, Device, DeviceTime, Dispatch, LaunchKind, MemoryCounterSnapshot,
};
use poot_graph_plan::{Submission, Target};
use poot_runtime_common::{
    AieLinkedArtifact, AllocGuard, ArgAccess, CompiledKernel, KernelCode, MemoryCounters,
};
use poot_target::{AieCaps, AieTargetId, BufferStorage, Queried, TargetId};

use crate::xrt::{XrtBo, XrtDevice, XrtError, XrtKernel};

/// The data buffer arguments of the kernel the IRON link driver builds: the NPU firmware's dispatch ABI is
/// `opcode, instruction buffer, instruction length, bo0 .. bo4`. Checked against the xclbin's declared
/// arguments on every load, so an xclbin of another interface is refused rather than run.
pub const HOST_BINDINGS: u32 = 5;

/// Arguments before the data buffers: the opcode, the instruction buffer and its length.
const LEADING_ARGS: u32 = 3;

/// How a device is opened. Typed configuration, never an environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XdnaConfig {
    /// How long one kernel run may take before it is a failure. An AIE invocation is a finite stream; a
    /// run that does not complete in this time is a hung context, and the device is poisoned.
    pub run_timeout: Duration,
}

impl Default for XdnaConfig {
    fn default() -> Self {
        Self {
            run_timeout: Duration::from_secs(5),
        }
    }
}

/// Bytes of data memory one AIE2p compute tile has for the FIFO buffers and scratch of its core. See
/// [`XdnaDevice::open`] for how this figure is pinned.
const TILE_LOCAL_BYTES: u32 = 64 * 1024 - 1024;

/// The largest instruction stream the device accepts, in bytes. See [`XdnaDevice::open`].
const INSTRUCTION_BYTES: u64 = 1 << 20;

/// Why a [`XdnaDevice`] call failed. Every refusal that does not need XRT is raised before XRT is touched.
#[derive(Debug, thiserror::Error)]
pub enum XdnaError {
    /// XRT failed: opening the device, allocating, syncing, loading or running.
    #[error("XRT {operation}: {source}")]
    Xrt {
        operation: &'static str,
        source: XrtError,
    },
    /// A kernel was not an AIE artifact: this device runs linked xclbins only.
    #[error(
        "kernel `{key}` is not an AIE artifact (target {target:?}); the XDNA device loads linked xclbins"
    )]
    NotAnAieKernel { key: String, target: TargetId },
    /// The artifact was linked for another AIE target than this device's.
    #[error("kernel `{key}` was linked for {artifact:?}; this device is {device:?}")]
    WrongTarget {
        key: String,
        artifact: AieTargetId,
        device: AieTargetId,
    },
    /// The artifact binds more host buffers than the firmware's dispatch ABI has slots.
    #[error("kernel `{key}` binds {bindings} host buffers; the device has {HOST_BINDINGS} slots")]
    TooManyBindings { key: String, bindings: usize },
    /// The xclbin declares a kernel interface other than the one the link driver builds.
    #[error(
        "kernel `{key}`: the xclbin declares {declared} arguments; the dispatch ABI is {LEADING_ARGS} \
         leading arguments and {HOST_BINDINGS} data buffers"
    )]
    KernelInterface { key: String, declared: u32 },
    /// A grid launch reached a device that runs only finite instruction streams.
    #[error("the XDNA device runs AIE instruction streams, not a {threads:?} thread grid")]
    GridLaunch { threads: [u32; 3] },
    /// The dispatch's invocation names another stream than the kernel it carries.
    #[error(
        "invocation of {invoked_words} words for {invoked_target:?} does not match kernel `{key}` \
         ({kernel_words} words, {kernel_target:?})"
    )]
    InvocationMismatch {
        key: String,
        invoked_target: AieTargetId,
        invoked_words: u32,
        kernel_target: AieTargetId,
        kernel_words: u32,
    },
    /// The dispatch's arguments are not the kernel's shape ABI.
    #[error("kernel `{key}`: {reason}")]
    Bindings { key: String, reason: String },
    /// A buffer or binding whose byte size does not fit the address space.
    #[error("{elems} elements of {width} bytes overflow the byte size")]
    Size { elems: u64, width: usize },
    /// A byte range outside a buffer.
    #[error("range {offset}..{end} is outside a {len}-byte buffer")]
    Range {
        offset: usize,
        end: usize,
        len: usize,
    },
    /// A copy between buffers of different byte lengths.
    #[error("copy of a {src}-byte buffer into a {dst}-byte buffer")]
    CopyLength { src: usize, dst: usize },
    /// `dispatch`, `copy` or `finish` outside a `begin`..`finish` transaction.
    #[error("no recording is open")]
    NotRecording,
    /// `begin` while a recording is already open.
    #[error("a recording is already open")]
    AlreadyRecording,
    /// A previous run failed; the device cannot be trusted with more work.
    #[error("the XDNA device is poisoned by an earlier failed run")]
    Poisoned,
}

fn xrt(operation: &'static str) -> impl FnOnce(XrtError) -> XdnaError {
    move |source| XdnaError::Xrt { operation, source }
}

/// One buffer object plus the charge it holds against the device's memory counters. The charge drops with
/// the buffer.
struct BufferCell {
    bo: XrtBo,
    storage: BufferStorage,
    _charge: AllocGuard,
}

/// A buffer of the XDNA device. Cloning shares the buffer object: a recording holds clones of the buffers
/// its dispatches name, so the buffer outlives the engine's own handle for as long as a replay can run.
#[derive(Clone)]
pub struct XdnaBuffer(Rc<BufferCell>);

impl XdnaBuffer {
    fn bytes(&self) -> usize {
        self.0.bo.len()
    }
}

/// A loaded kernel: the XRT kernel and the artifact facts every dispatch is checked against.
pub struct XdnaKernel(Rc<LoadedKernel>);

struct LoadedKernel {
    key: String,
    xrt: XrtKernel,
    artifact_target: AieTargetId,
    instruction_words: u32,
    /// Per binding, in ABI order: its storage, its element count and whether the kernel writes it.
    bindings: Vec<(BufferStorage, u64, ArgAccess)>,
}

/// One recorded step of a recording.
enum Op {
    Dispatch {
        kernel: Rc<LoadedKernel>,
        buffers: Vec<XdnaBuffer>,
    },
    Copy {
        src: XdnaBuffer,
        dst: XdnaBuffer,
    },
}

/// A closed recording: the steps one replay runs, in order.
pub struct XdnaRecording(Rc<[Op]>);

/// The AMD XDNA2 NPU as an executor [`Device`].
pub struct XdnaDevice {
    device: XrtDevice,
    caps: AieCaps,
    counters: MemoryCounters,
    run_timeout: Duration,
    open: Option<Vec<Op>>,
    poisoned: bool,
    /// Buffers a failed run was using. They stay allocated and charged: the hardware context is not known
    /// to be idle, so nothing it may still write can be freed under it.
    quarantine: Vec<XdnaBuffer>,
}

impl XdnaDevice {
    /// Open the NPU at `/dev/accel/accel0`.
    ///
    /// The capability descriptor a plan is admitted against is this device's own. Which numbers are
    /// measured and which are fixed:
    ///
    /// - `tile_local_bytes` is **measured on this device**: 64 KiB of compute-tile data memory less the
    ///   core's 1 KiB stack. A full-tile add (2688 four-byte elements per binding, three double-buffered
    ///   FIFOs) links and runs; one element more is refused by `aiecc` ("Failed to allocate buffer"). The
    ///   hardware row `the_declared_tile_memory_is_the_size_that_links_and_runs` pins this constant to
    ///   that boundary. The 64 KiB itself is AMD AM020 (AIE-ML architecture manual, retrieved 2026-10-11,
    ///   <https://docs.amd.com/r/en-US/am020-versal-aie-ml>).
    /// - `host_bindings` is **fixed** by the firmware's dispatch ABI ([`HOST_BINDINGS`]: opcode,
    ///   instruction buffer, its length, five data buffers) and verified against the xclbin on every
    ///   kernel load. The planner's elementwise kernels bind three buffers, so no plan reaches the limit
    ///   today: this device's load check is where it is enforced.
    /// - `instruction_bytes` is **fixed, not measured**: the stream is one device-visible buffer and 1 MiB
    ///   is a conservative bound far above any stream the sequential schedule produces.
    /// - `memory_budget_bytes` is **read from the host when the device opens** (the process's locked-memory
    ///   limit, because an NPU's buffer objects are pinned host pages).
    pub fn open() -> Result<Self, XdnaError> {
        Self::open_with(XdnaConfig::default())
    }

    /// [`XdnaDevice::open`] with `config`.
    pub fn open_with(config: XdnaConfig) -> Result<Self, XdnaError> {
        let device = XrtDevice::open(0).map_err(xrt("device open"))?;
        let caps = AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(TILE_LOCAL_BYTES),
            host_bindings: Queried::Known(HOST_BINDINGS),
            instruction_bytes: Queried::Known(INSTRUCTION_BYTES),
            memory_budget_bytes: locked_memory_limit().map_or(Queried::Unknown, Queried::Known),
        };
        Ok(Self {
            device,
            caps,
            counters: MemoryCounters::new(),
            run_timeout: config.run_timeout,
            open: None,
            poisoned: false,
            quarantine: Vec::new(),
        })
    }

    /// The capabilities plans are admitted against.
    pub const fn caps(&self) -> AieCaps {
        self.caps
    }

    fn healthy(&self) -> Result<(), XdnaError> {
        if self.poisoned {
            Err(XdnaError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn recording(&mut self) -> Result<&mut Vec<Op>, XdnaError> {
        self.open.as_mut().ok_or(XdnaError::NotRecording)
    }

    /// Run one dispatch to completion. On failure the device is poisoned and the dispatch's buffers are
    /// quarantined.
    fn run(&mut self, kernel: &LoadedKernel, buffers: &[XdnaBuffer]) -> Result<(), XdnaError> {
        let bos: Vec<&XrtBo> = buffers.iter().map(|b| &b.0.bo).collect();
        let timeout_us = u64::try_from(self.run_timeout.as_micros()).unwrap_or(u64::MAX);
        let ran = kernel.xrt.run(&bos, timeout_us).map_err(xrt("kernel run"));
        let synced = ran.and_then(|()| {
            // The device wrote the written bindings straight into the shared pages; make them visible.
            buffers
                .iter()
                .zip(&kernel.bindings)
                .filter(|(_, (_, _, access))| *access == ArgAccess::Write)
                .try_for_each(|(buffer, _)| {
                    buffer
                        .0
                        .bo
                        .sync_from_device(0, buffer.bytes())
                        .map_err(xrt("bo sync"))
                })
        });
        if let Err(error) = &synced {
            tracing::warn!(kernel = %kernel.key, %error, "XDNA run failed; the device is poisoned");
            self.poisoned = true;
            self.quarantine.extend(buffers.iter().cloned());
        }
        synced
    }
}

/// `RLIMIT_MEMLOCK`'s soft limit in bytes from `/proc/self/limits`; `u64::MAX` for "unlimited".
fn locked_memory_limit() -> Option<u64> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    let line = limits
        .lines()
        .find(|l| l.starts_with("Max locked memory"))?;
    let soft = line
        .strip_prefix("Max locked memory")?
        .split_whitespace()
        .next()?;
    match soft {
        "unlimited" => Some(u64::MAX),
        bytes => bytes.parse().ok(),
    }
}

impl Device for XdnaDevice {
    type Buffer = XdnaBuffer;
    type Kernel = XdnaKernel;
    type Recording = XdnaRecording;
    type Error = XdnaError;

    fn target(&self) -> Target {
        Target::Aie {
            target: self.caps.target,
            caps: self.caps,
        }
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.counters.snapshot_all()
    }

    fn free_memory_bytes(&self) -> Option<u64> {
        match self.caps.memory_budget_bytes {
            Queried::Known(budget) => Some(budget.saturating_sub(self.counters.total_live_bytes())),
            Queried::Unknown => None,
        }
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<XdnaBuffer, XdnaError> {
        self.healthy()?;
        let width = storage.element().byte_width();
        let bytes = elems.checked_mul(width).ok_or(XdnaError::Size {
            elems: elems as u64,
            width,
        })?;
        let bo = self.device.alloc(bytes).map_err(xrt("bo alloc"))?;
        let charge = self.counters.record_alloc(role, bytes as u64);
        Ok(XdnaBuffer(Rc::new(BufferCell {
            bo,
            storage,
            _charge: charge,
        })))
    }

    fn load_kernel(&mut self, key: &str, kernel: CompiledKernel) -> Result<XdnaKernel, XdnaError> {
        self.healthy()?;
        let KernelCode::Aie(artifact) = kernel.code() else {
            return Err(XdnaError::NotAnAieKernel {
                key: key.to_owned(),
                target: kernel.target(),
            });
        };
        self.load_artifact(key, kernel.entry_point(), artifact)
    }

    fn begin(&mut self, _submission: Submission) -> Result<(), XdnaError> {
        self.healthy()?;
        if self.open.is_some() {
            return Err(XdnaError::AlreadyRecording);
        }
        self.open = Some(Vec::new());
        Ok(())
    }

    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), XdnaError> {
        self.healthy()?;
        let invocation = match dispatch.launch {
            LaunchKind::Aie(invocation) => invocation,
            LaunchKind::Grid { threads, .. } => return Err(XdnaError::GridLaunch { threads }),
        };
        let kernel = Rc::clone(&dispatch.kernel.0);
        check_invocation(&kernel, invocation)?;
        let args: Vec<&Arg<'_, Self>> = dispatch
            .inputs
            .iter()
            .chain(std::iter::once(&dispatch.output))
            .collect();
        check_bindings(&kernel, &args)?;
        let buffers = args.iter().map(|arg| arg.buffer.clone()).collect();
        self.recording()?.push(Op::Dispatch { kernel, buffers });
        Ok(())
    }

    fn copy(&mut self, src: &XdnaBuffer, dst: &XdnaBuffer) -> Result<(), XdnaError> {
        self.healthy()?;
        if src.bytes() != dst.bytes() {
            return Err(XdnaError::CopyLength {
                src: src.bytes(),
                dst: dst.bytes(),
            });
        }
        self.recording()?.push(Op::Copy {
            src: src.clone(),
            dst: dst.clone(),
        });
        Ok(())
    }

    fn finish(&mut self) -> Result<Option<XdnaRecording>, XdnaError> {
        let ops = self.open.take().ok_or(XdnaError::NotRecording)?;
        Ok(Some(XdnaRecording(ops.into())))
    }

    fn replay(&mut self, recording: &XdnaRecording) -> Result<(), XdnaError> {
        self.healthy()?;
        for op in recording.0.iter() {
            match op {
                Op::Dispatch { kernel, buffers } => self.run(kernel, buffers)?,
                Op::Copy { src, dst } => {
                    // A failed sync leaves `dst` half updated or the source not visible, which no later
                    // step can tell from a good state: the device is poisoned and keeps both buffers.
                    let copied = src
                        .0
                        .bo
                        .sync_from_device(0, src.bytes())
                        .and_then(|()| {
                            dst.0.bo.copy_from(&src.0.bo);
                            dst.0.bo.sync_to_device(0, dst.bytes())
                        })
                        .map_err(xrt("bo sync"));
                    if let Err(error) = copied {
                        tracing::warn!(%error, "XDNA copy failed; the device is poisoned");
                        self.poisoned = true;
                        self.quarantine.extend([src.clone(), dst.clone()]);
                        return Err(error);
                    }
                }
            }
        }
        Ok(())
    }

    fn write_with(
        &mut self,
        dst: &XdnaBuffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), XdnaError> {
        self.healthy()?;
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= dst.bytes())
            .ok_or(XdnaError::Range {
                offset,
                end: offset.saturating_add(len),
                len: dst.bytes(),
            })?;
        if len == 0 {
            return Ok(());
        }
        // The buffer is the device's own pages: the whole range is one chunk, no staging copy.
        dst.0
            .bo
            .write(offset..end, |bytes| fill(offset..end, bytes));
        dst.0.bo.sync_to_device(offset, len).map_err(xrt("bo sync"))
    }

    fn read(&mut self, src: &XdnaBuffer, out: &mut [u8]) -> Result<(), XdnaError> {
        self.healthy()?;
        if out.len() > src.bytes() {
            return Err(XdnaError::Range {
                offset: 0,
                end: out.len(),
                len: src.bytes(),
            });
        }
        if out.is_empty() {
            return Ok(());
        }
        src.0
            .bo
            .sync_from_device(0, out.len())
            .map_err(xrt("bo sync"))?;
        src.0.bo.read(0, out);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), XdnaError> {
        // Replay is synchronous: every run completed (or failed) before `replay` returned.
        self.healthy()
    }

    fn device_time(&self) -> DeviceTime {
        // XRT's native API exposes no device-side timestamps for an NPU run; a host wall clock is not device
        // time, so none is reported.
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), XdnaError> {
        // A partially built recording is dropped whole; nothing has run on the device for it.
        self.open = None;
        Ok(())
    }
}

impl XdnaDevice {
    fn load_artifact(
        &mut self,
        key: &str,
        entry: &str,
        artifact: &AieLinkedArtifact,
    ) -> Result<XdnaKernel, XdnaError> {
        if artifact.target() != self.caps.target {
            return Err(XdnaError::WrongTarget {
                key: key.to_owned(),
                artifact: artifact.target(),
                device: self.caps.target,
            });
        }
        let bindings = artifact.abi().bindings();
        if bindings.len() > HOST_BINDINGS as usize {
            return Err(XdnaError::TooManyBindings {
                key: key.to_owned(),
                bindings: bindings.len(),
            });
        }
        let loaded = self
            .device
            .load_kernel(artifact.xclbin(), entry, artifact.instructions())
            .map_err(xrt("kernel load"))?;
        let declared = loaded.arg_count();
        if declared != LEADING_ARGS + HOST_BINDINGS {
            return Err(XdnaError::KernelInterface {
                key: key.to_owned(),
                declared,
            });
        }
        let instruction_words = u32::try_from(artifact.instructions().len() / 4)
            .expect("the engine bounds the instruction stream to u32 words before it loads");
        tracing::debug!(
            key,
            instruction_words,
            bindings = bindings.len(),
            "loaded an AIE kernel"
        );
        Ok(XdnaKernel(Rc::new(LoadedKernel {
            key: key.to_owned(),
            xrt: loaded,
            artifact_target: artifact.target(),
            instruction_words,
            bindings: bindings
                .iter()
                .map(|b| (b.storage(), b.elements(), b.access()))
                .collect(),
        })))
    }
}

/// The dispatch's invocation is the stream of the kernel it carries.
fn check_invocation(kernel: &LoadedKernel, invocation: AieInvocation) -> Result<(), XdnaError> {
    if invocation.target() == kernel.artifact_target
        && invocation.instruction_words().get() == kernel.instruction_words
    {
        return Ok(());
    }
    Err(XdnaError::InvocationMismatch {
        key: kernel.key.clone(),
        invoked_target: invocation.target(),
        invoked_words: invocation.instruction_words().get(),
        kernel_target: kernel.artifact_target,
        kernel_words: kernel.instruction_words,
    })
}

/// The dispatch's arguments (the reads, then the written output) are the kernel's shape ABI: the count, and
/// for each binding its storage, its element count, and that exactly the last one is written.
fn check_bindings(kernel: &LoadedKernel, args: &[&Arg<'_, XdnaDevice>]) -> Result<(), XdnaError> {
    let refuse = |reason: String| XdnaError::Bindings {
        key: kernel.key.clone(),
        reason,
    };
    if args.len() != kernel.bindings.len() {
        return Err(refuse(format!(
            "the shape ABI declares {} bindings, the dispatch binds {}",
            kernel.bindings.len(),
            args.len()
        )));
    }
    let last = args.len() - 1;
    for (index, (arg, (storage, elements, access))) in args.iter().zip(&kernel.bindings).enumerate()
    {
        let writes = *access == ArgAccess::Write;
        if writes != (index == last) {
            return Err(refuse(format!(
                "binding {index} is {} in the shape ABI; a dispatch reads its inputs and writes its \
                 one output, last",
                if writes { "written" } else { "read" }
            )));
        }
        if arg.buffer.0.storage != *storage {
            return Err(refuse(format!(
                "binding {index} is {storage:?}, the bound buffer is {:?}",
                arg.buffer.0.storage
            )));
        }
        if u64::from(arg.elems) != *elements {
            return Err(refuse(format!(
                "binding {index} is {elements} elements, the dispatch binds {}",
                arg.elems
            )));
        }
        let width = storage.element().byte_width();
        let needed = usize::try_from(*elements)
            .ok()
            .and_then(|elements| elements.checked_mul(width))
            .ok_or(XdnaError::Size {
                elems: *elements,
                width,
            })?;
        if arg.buffer.bytes() < needed {
            return Err(refuse(format!(
                "binding {index} needs {needed} bytes, the bound buffer holds {}",
                arg.buffer.bytes()
            )));
        }
    }
    Ok(())
}
