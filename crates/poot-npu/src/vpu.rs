//! [`VpuDevice`]: the `poot_executor::Device` of an Intel NPU over Level Zero.
//!
//! There is one engine over every backend, so this is a thin mechanism below the executor contract. An Intel
//! NPU has no kernel ISA: a loaded kernel is a graph the driver's vendor compiler built from the OpenVINO IR
//! `poot-codegen` emitted for the program's region. The device
//!
//! - allocates host memory the NPU addresses (`zeMemAllocHost`) for every buffer,
//! - loads a region by compiling its IR, or by loading the blob an earlier compile left in the blob cache
//!   ([`crate::BlobCache`]), and checks the compiled graph's own argument list against the region's ABI
//!   before anything runs,
//! - records the engine's dispatches and copies into one command list (a graph execution binds its arguments
//!   when it is appended) and closes it: the closed list is the recording, and replay executes it again,
//! - waits for the queue on every replay, so [`Device::synchronize`] has nothing in flight.
//!
//! The device recognizes no graph and emits no IR: the planner admitted every equation, the engine passed
//! the region it assembled, and this crate runs it. What it checks on its own, before the driver is touched,
//! are the things only it can see: the kernel is a graph artifact for this part, the compiled arguments are
//! the region's, and each dispatch's launch, invocation and buffers are the kernel's. Each is a typed
//! [`VpuError`].
//!
//! # Failure
//!
//! A replay that times out or fails poisons the device: the graph may still be running, so nothing it can
//! write is freed. The failed recording and every buffer it names stay allocated and charged until the
//! device itself drops.

use std::cell::{Ref, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use poot_executor::{Arg, Device, DeviceTime, Dispatch, LaunchKind, VpuInvocation};
use poot_graph_plan::Target;
use poot_l0_runtime::{
    GraphDirection, GraphIr, GraphPrecision, L0Context, L0Error, L0Graph, L0HostBuffer,
    L0Recording, L0RecordingBuilder,
};
use poot_runtime_common::{
    AllocGuard, BufferRole, CompiledKernel, KernelCode, MemoryCounterSnapshot, MemoryCounters,
    VpuBinding, VpuGraphArtifact, VpuLayer,
};
use poot_target::{
    BufferStorage, ElementKind, NarrowingContract, Queried, TargetId, VendorCompilerId, VpuCaps,
    VpuTargetId,
};

use crate::blob_cache::{BlobCache, BlobKey};
use crate::range_guard::{Direction, GuardArg, RangeCheck, RangeGuard, RangeViolation};

/// The PCI vendor id of Intel.
const INTEL_VENDOR_ID: u32 = 0x8086;

/// The numerical contract F32 values are narrowed to FP16 under on the Lunar Lake NPU (POOT-1178): the bound the
/// `vpu_narrowing` device rows hold the normwise error of the Qwen2.5-0.5B MLP and attention-core fragments to.
/// Measured 2026-10-18 on the laptop's NPU4 (driver compiler 1.38.0), three steps each against the F32 CPU
/// oracle: worst 4.7e-4 (MLP) and 1.6e-3 (attention core). The recorded 5e-3 is about three times the worst
/// measurement, and covers those region classes only: a graph with longer reductions or a wider dynamic range
/// is outside what was measured. The range precondition is FP16's: the largest finite value, 65504.
const NPU4_F32_NARROWING: NarrowingContract = NarrowingContract {
    max_relative_error_ppm: 5_000,
    max_magnitude: 65_504,
};

/// The graph compiler reads inputs and results in rank at most this many dimensions (the driver reports every
/// argument padded to five).
const MAX_RANK: usize = 5;

/// How a device is opened. Typed configuration, never an environment variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VpuConfig {
    /// How long one replay may take before it is a failure. A graph is a finite program; one that does not
    /// finish in this time is hung, and the device is poisoned.
    pub run_timeout: Duration,
    /// Where compiled blobs are cached between runs. `None` compiles every region every time.
    pub cache_dir: Option<PathBuf>,
}

impl Default for VpuConfig {
    fn default() -> Self {
        Self {
            run_timeout: Duration::from_secs(10),
            cache_dir: None,
        }
    }
}

/// Why a [`VpuDevice`] call failed. Every refusal that does not need the driver is raised before it is touched.
#[derive(Debug, thiserror::Error)]
pub enum VpuError {
    /// Level Zero failed: opening the device, allocating, compiling, loading, recording or running.
    #[error("Level Zero {operation}: {source}")]
    L0 {
        /// What the device was doing.
        operation: &'static str,
        /// What Level Zero reported.
        source: L0Error,
    },
    /// The opened device is not an Intel one.
    #[error("the Level Zero NPU reports vendor id {vendor:#x}, not Intel's 0x8086")]
    NotIntel {
        /// The vendor id the device reported.
        vendor: u32,
    },
    /// A kernel was not a graph artifact: this device runs vendor-compiled graphs only.
    #[error("kernel `{key}` is not an Intel NPU graph (target {target:?})")]
    NotAGraph {
        /// The kernel's key.
        key: String,
        /// The target the kernel was built for.
        target: TargetId,
    },
    /// The artifact was emitted for another NPU part than this device's.
    #[error("kernel `{key}` was emitted for {artifact:?}; this device is {device:?}")]
    WrongTarget {
        /// The kernel's key.
        key: String,
        /// The part the artifact was emitted for.
        artifact: VpuTargetId,
        /// This device's part.
        device: VpuTargetId,
    },
    /// The vendor compiler cannot compile some layers of the region. Each names the poot equation it lowers, so
    /// the refusal points at the graph, not at the IR.
    #[error(
        "kernel `{key}`: the vendor compiler cannot compile the layers lowered from equations {equations:?}"
    )]
    UnsupportedLayers {
        /// The kernel's key.
        key: String,
        /// The IR layers the compiler left out of its supported set.
        layers: Vec<String>,
        /// The indices of the equations' output values those layers lower.
        equations: Vec<u32>,
    },
    /// A binding's element kind is one the NPU boundary does not carry (only F16, F32 and I32 are named).
    #[error(
        "kernel `{key}`: argument `{binding}` is {element}, which the Intel NPU boundary does not carry"
    )]
    Element {
        /// The kernel's key.
        key: String,
        /// The binding's name.
        binding: String,
        /// Its element kind.
        element: ElementKind,
    },
    /// The compiled graph's arguments are not the region's ABI.
    #[error("kernel `{key}`: the compiled graph disagrees with the region: {reason}")]
    Abi {
        /// The kernel's key.
        key: String,
        /// Which argument disagrees and how.
        reason: String,
    },
    /// A grid launch reached a device that runs only graphs.
    #[error("the Intel NPU device runs graphs, not a {threads:?} thread grid")]
    GridLaunch {
        /// The requested thread grid.
        threads: [u32; 3],
    },
    /// An AIE invocation reached the Intel NPU device.
    #[error("the Intel NPU device runs graphs, not an AIE instruction stream")]
    AieLaunch,
    /// The dispatch's invocation names another graph than the kernel it carries.
    #[error(
        "invocation of artifact {invoked_digest:#x} for {invoked_target:?} does not match kernel `{key}` ({kernel_digest:#x}, {kernel_target:?})"
    )]
    InvocationMismatch {
        /// The kernel's key.
        key: String,
        /// The part the invocation was built for.
        invoked_target: VpuTargetId,
        /// The artifact digest the invocation names.
        invoked_digest: u64,
        /// The part the kernel was built for.
        kernel_target: VpuTargetId,
        /// The artifact digest of the kernel.
        kernel_digest: u64,
    },
    /// The dispatch's arguments are not the kernel's ABI.
    #[error("kernel `{key}`: {reason}")]
    Bindings {
        /// The kernel's key.
        key: String,
        /// Which binding the arguments break.
        reason: String,
    },
    /// A buffer or binding whose byte size does not fit the address space.
    #[error("{elems} elements of {width} bytes overflow the byte size")]
    Size {
        /// The element count.
        elems: u64,
        /// The element width, in bytes.
        width: usize,
    },
    /// An F32 argument holds a value outside the range the device's narrowing contract holds for: it would
    /// narrow to infinity (an input, caught before the replay) or did (a result, caught after it).
    #[error(
        "kernel `{key}`: {role} argument {binding} element {index} is {value}, outside the narrowing contract's range of +-{max_magnitude}"
    )]
    NarrowingRange {
        /// The kernel's key.
        key: String,
        /// `"input"` or `"result"`.
        role: &'static str,
        /// The binding's index among the graph's arguments.
        binding: usize,
        /// The first offending element.
        index: usize,
        /// Its value.
        value: f32,
        /// The largest magnitude the contract admits.
        max_magnitude: u32,
    },
    /// A weight uploaded with an F32 element outside the narrowing contract's range.
    #[error(
        "constant upload: element {index} is {value}, outside the narrowing contract's range of +-{max_magnitude}"
    )]
    NarrowingRangeUpload {
        /// The first offending element of the upload.
        index: usize,
        /// Its value.
        value: f32,
        /// The largest magnitude the contract admits.
        max_magnitude: u32,
    },
    /// An upload into an F32 weight that does not cover whole words at a word-aligned offset, so a word of it
    /// could not be range-checked.
    #[error(
        "constant upload of {len} bytes at byte {offset} does not cover whole F32 words, so its range cannot be checked"
    )]
    UnalignedConstantUpload {
        /// The upload's start, in bytes.
        offset: usize,
        /// The upload's length, in bytes.
        len: usize,
    },
    /// A byte range outside a buffer.
    #[error("range {offset}..{end} is outside a {len}-byte buffer")]
    Range {
        /// The start of the range, in bytes.
        offset: usize,
        /// The end of the range, in bytes.
        end: usize,
        /// The buffer's length, in bytes.
        len: usize,
    },
    /// A copy between buffers of different byte lengths.
    #[error("copy of a {src}-byte buffer into a {dst}-byte buffer")]
    CopyLength {
        /// The source buffer's length, in bytes.
        src: usize,
        /// The destination buffer's length, in bytes.
        dst: usize,
    },
    /// The blob cache failed on disk.
    #[error("blob cache: {0}")]
    Cache(#[from] std::io::Error),
    /// `dispatch`, `copy` or `finish` outside a `begin`..`finish` transaction.
    #[error("no recording is open")]
    NotRecording,
    /// `begin` while a recording is already open.
    #[error("a recording is already open")]
    AlreadyRecording,
    /// A previous replay failed; the device cannot be trusted with more work.
    #[error("the Intel NPU device is poisoned by an earlier failed replay")]
    Poisoned,
}

impl From<RangeViolation> for VpuError {
    fn from(violation: RangeViolation) -> Self {
        match violation {
            RangeViolation::Upload {
                index,
                value,
                max_magnitude,
            } => VpuError::NarrowingRangeUpload {
                index,
                value,
                max_magnitude,
            },
            RangeViolation::Misaligned { offset, len } => {
                VpuError::UnalignedConstantUpload { offset, len }
            }
            RangeViolation::Replay {
                key,
                role,
                binding,
                index,
                value,
                max_magnitude,
            } => VpuError::NarrowingRange {
                key,
                role,
                binding,
                index,
                value,
                max_magnitude,
            },
        }
    }
}

fn l0(operation: &'static str) -> impl FnOnce(L0Error) -> VpuError {
    move |source| VpuError::L0 { operation, source }
}

/// One host buffer plus the charge it holds against the device's memory counters. The charge drops with the
/// buffer. Writing the host buffer needs exclusive access; `VpuBuffer` handles are shared, so the cell checks it
/// at runtime.
struct BufferCell {
    host: RefCell<L0HostBuffer>,
    storage: BufferStorage,
    role: BufferRole,
    _charge: AllocGuard,
}

/// A buffer of the Intel NPU device: host memory the NPU addresses. Cloning shares it.
#[derive(Clone)]
pub struct VpuBuffer(Rc<BufferCell>);

impl PartialEq for VpuBuffer {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl VpuBuffer {
    fn bytes(&self) -> usize {
        self.0.host.borrow().len()
    }
}

/// A loaded kernel: the compiled graph and the artifact facts every dispatch is checked against.
pub struct VpuKernel(Rc<LoadedKernel>);

struct LoadedKernel {
    key: String,
    graph: L0Graph,
    target: VpuTargetId,
    digest: u64,
    /// The region's inputs, then its outputs: the graph's arguments in order.
    bindings: Vec<Binding>,
}

/// One argument of a loaded graph.
struct Binding {
    storage: BufferStorage,
    elements: u64,
}

/// A closed recording: the command list one replay executes, and the F32 arguments whose range the device's
/// narrowing contract constrains.
#[derive(Clone)]
pub struct VpuRecording {
    list: L0Recording,
    ranges: Rc<[RangeCheck<VpuBuffer>]>,
}

/// The Intel NPU as an executor [`Device`].
pub struct VpuDevice {
    context: L0Context,
    caps: VpuCaps,
    counters: MemoryCounters,
    config: VpuConfig,
    cache: BlobCache,
    /// The vendor compiler's version, part of every blob's identity.
    compiler: VendorCompilerId,
    open: Option<L0RecordingBuilder>,
    /// The narrowing contract's range guard; `None` when the device records no contract.
    guard: Option<RangeGuard<VpuBuffer>>,
    poisoned: bool,
    /// Recordings a failed replay was running. They stay alive (and with them every buffer they name): the
    /// graph is not known to be idle, so nothing it may still write can be freed under it.
    quarantine: Vec<VpuRecording>,
    /// Replays started on this device: one per [`Device::replay`], however many graphs the recording runs.
    #[cfg(any(test, feature = "test-support"))]
    runs: u64,
    /// Regions the vendor compiler compiled on this device (a blob cache miss).
    #[cfg(any(test, feature = "test-support"))]
    compiles: u64,
    /// Regions loaded from a cached blob (a blob cache hit).
    #[cfg(any(test, feature = "test-support"))]
    cache_hits: u64,
}

impl VpuDevice {
    /// Open the Intel NPU with the default configuration.
    pub fn open() -> Result<Self, VpuError> {
        Self::open_with(VpuConfig::default())
    }

    /// [`VpuDevice::open`] with `config`.
    ///
    /// The capability descriptor a plan is admitted against is this device's own: the part is Lunar Lake's
    /// NPU4 (the only part this crate knows), the compiler version is read from the driver, and the memory
    /// budget is unknown (the driver reports no figure poot trusts).
    pub fn open_with(config: VpuConfig) -> Result<Self, VpuError> {
        let context = L0Context::open().map_err(l0("device open"))?;
        if context.vendor_id() != INTEL_VENDOR_ID {
            return Err(VpuError::NotIntel {
                vendor: context.vendor_id(),
            });
        }
        let properties = context
            .graph_device_properties()
            .map_err(l0("graph properties"))?;
        let compiler = VendorCompilerId {
            major: u32::from(properties.compiler_version.0),
            minor: u32::from(properties.compiler_version.1),
        };
        let caps = VpuCaps {
            target: VpuTargetId::Npu4,
            compiler: Queried::Known(compiler),
            memory_budget_bytes: Queried::Unknown,
            narrowing: Some(NPU4_F32_NARROWING),
        };
        let cache = BlobCache::new(config.cache_dir.clone());
        Ok(Self {
            context,
            caps,
            counters: MemoryCounters::new(),
            config,
            cache,
            compiler,
            open: None,
            guard: caps.narrowing.map(RangeGuard::new),
            poisoned: false,
            quarantine: Vec::new(),
            #[cfg(any(test, feature = "test-support"))]
            runs: 0,
            #[cfg(any(test, feature = "test-support"))]
            compiles: 0,
            #[cfg(any(test, feature = "test-support"))]
            cache_hits: 0,
        })
    }

    /// The capabilities plans are admitted against.
    pub const fn caps(&self) -> VpuCaps {
        self.caps
    }

    /// How many replays this device has started: one per [`Device::replay`], whatever the number of graphs the
    /// recording executes.
    #[cfg(any(test, feature = "test-support"))]
    pub const fn replays(&self) -> u64 {
        self.runs
    }

    /// How many regions the vendor compiler compiled on this device: a blob cache miss.
    #[cfg(any(test, feature = "test-support"))]
    pub const fn compiles(&self) -> u64 {
        self.compiles
    }

    /// How many regions were loaded from a cached blob without the compiler: a blob cache hit.
    #[cfg(any(test, feature = "test-support"))]
    pub const fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// How many constant uploads the range guard checked. A weight is checked once, as it is written.
    #[cfg(any(test, feature = "test-support"))]
    pub fn constant_range_checks(&self) -> u64 {
        self.guard.as_ref().map_or(0, |guard| guard.constant_checks)
    }

    /// How many input and result reads the range guard made on replays: one per checked argument per replay,
    /// and none for a constant.
    #[cfg(any(test, feature = "test-support"))]
    pub fn replay_range_checks(&self) -> u64 {
        self.guard.as_ref().map_or(0, |guard| guard.replay_checks)
    }

    fn healthy(&self) -> Result<(), VpuError> {
        if self.poisoned {
            Err(VpuError::Poisoned)
        } else {
            Ok(())
        }
    }

    fn recording(&mut self) -> Result<&mut L0RecordingBuilder, VpuError> {
        self.open.as_mut().ok_or(VpuError::NotRecording)
    }

    /// The compiled graph of `artifact`: from the blob cache when it holds this exact region for this exact
    /// compiler, else compiled by the vendor compiler (and cached).
    fn graph_of(&mut self, key: &str, artifact: &VpuGraphArtifact) -> Result<L0Graph, VpuError> {
        let options = build_flags(key, artifact)?;
        let blob_key = BlobKey::new(artifact, self.compiler, &options);
        if let Some(blob) = self.cache.get(&blob_key)? {
            // SAFETY: `blob` is what the cache returned for this exact region, compiler version and options; it
            // stores only `native_binary` output and returns it only when its stored digest matches.
            match unsafe { self.context.load_native_graph(&blob) } {
                Ok(graph) => {
                    #[cfg(any(test, feature = "test-support"))]
                    {
                        self.cache_hits += 1;
                    }
                    return Ok(graph);
                }
                // A cached blob the driver no longer accepts (a firmware or driver update that kept the
                // compiler version) is a miss: it is recompiled and overwritten below.
                Err(error) => tracing::warn!(key, %error, "discarding an unloadable cached blob"),
            }
        }
        let ir = GraphIr {
            xml: artifact.xml(),
            weights: artifact.weights(),
        };
        // Ask the compiler which layers it takes before compiling: a rejected layer is a refusal naming the
        // equation it came from, not a build log to read. A query the compiler cannot even parse is not
        // an answer: compiling is the authority, and its failure carries the compiler's own log.
        match self.context.query_supported_layers(&ir, &options) {
            Ok(supported) => {
                let rejected = rejected_layers(artifact.layers(), &supported);
                if !rejected.is_empty() {
                    return Err(VpuError::UnsupportedLayers {
                        key: key.to_owned(),
                        layers: rejected.iter().map(|l| l.name().to_owned()).collect(),
                        equations: rejected.iter().map(|l| l.origin()).collect(),
                    });
                }
            }
            Err(error @ L0Error::GraphBuild { .. }) => {
                tracing::debug!(key, %error, "the layer query failed; compiling is the authority");
            }
            Err(error) => return Err(l0("graph query")(error)),
        }
        let graph = self
            .context
            .compile_graph(&ir, &options)
            .map_err(l0("graph compile"))?;
        #[cfg(any(test, feature = "test-support"))]
        {
            self.compiles += 1;
        }
        let blob = graph.native_binary().map_err(l0("native binary"))?;
        self.cache.put(&blob_key, &blob)?;
        Ok(graph)
    }

    fn poison(&mut self, recording: &VpuRecording, error: &VpuError) {
        tracing::warn!(%error, "Intel NPU replay failed; the device is poisoned");
        self.poisoned = true;
        self.quarantine.push(recording.clone());
    }
}

/// The layers of a region the compiler's `supported` set leaves out.
fn rejected_layers<'a>(layers: &'a [VpuLayer], supported: &[String]) -> Vec<&'a VpuLayer> {
    layers
        .iter()
        .filter(|layer| !supported.iter().any(|name| name == layer.name()))
        .collect()
}

/// The compiler's option string for a region: every argument keeps the precision the region declares, so the
/// compiler adds no narrowing conversion of its own at the boundary.
fn build_flags(key: &str, artifact: &VpuGraphArtifact) -> Result<String, VpuError> {
    let precision = |bindings: &[VpuBinding]| -> Result<String, VpuError> {
        let named = bindings
            .iter()
            .map(|b| {
                let (name, _) = graph_precision(key, b.name(), b.storage().element())?;
                Ok(format!("{}:{name}", b.name()))
            })
            .collect::<Result<Vec<_>, VpuError>>()?;
        Ok(named.join(" "))
    };
    Ok(format!(
        "--inputs_precisions=\"{}\" --outputs_precisions=\"{}\"",
        precision(artifact.inputs())?,
        precision(artifact.outputs())?
    ))
}

/// The compiler's name for an element kind and the precision the compiled graph reports for it. Every kind is
/// named: one the NPU boundary does not carry is a typed refusal, never a silent FP32 (ADR-0103).
fn graph_precision(
    key: &str,
    binding: &str,
    element: ElementKind,
) -> Result<(&'static str, GraphPrecision), VpuError> {
    match element {
        ElementKind::F16 => Ok(("FP16", GraphPrecision::Fp16)),
        ElementKind::F32 => Ok(("FP32", GraphPrecision::Fp32)),
        ElementKind::I32 => Ok(("I32", GraphPrecision::Int32)),
        ElementKind::Bf16 | ElementKind::RawBytes => Err(VpuError::Element {
            key: key.to_owned(),
            binding: binding.to_owned(),
            element,
        }),
    }
}

impl Device for VpuDevice {
    type Buffer = VpuBuffer;
    type Kernel = VpuKernel;
    type Recording = VpuRecording;
    type Error = VpuError;

    fn target(&self) -> Target {
        Target::Vpu {
            target: self.caps.target,
            caps: self.caps,
        }
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.counters.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<VpuBuffer, VpuError> {
        self.healthy()?;
        let width = storage.element().byte_width();
        let bytes = elems.checked_mul(width).ok_or(VpuError::Size {
            elems: elems as u64,
            width,
        })?;
        let host = self.context.alloc_host(bytes).map_err(l0("host alloc"))?;
        let charge = self.counters.record_alloc(role, bytes as u64);
        Ok(VpuBuffer(Rc::new(BufferCell {
            host: RefCell::new(host),
            storage,
            role,
            _charge: charge,
        })))
    }

    fn buffer_elems(&self, buffer: &VpuBuffer) -> usize {
        buffer.bytes() / buffer.0.storage.element().byte_width()
    }

    fn load_kernel(&mut self, key: &str, kernel: CompiledKernel) -> Result<VpuKernel, VpuError> {
        self.healthy()?;
        let KernelCode::Vpu(artifact) = kernel.code() else {
            return Err(VpuError::NotAGraph {
                key: key.to_owned(),
                target: kernel.target(),
            });
        };
        if artifact.target() != self.caps.target {
            return Err(VpuError::WrongTarget {
                key: key.to_owned(),
                artifact: artifact.target(),
                device: self.caps.target,
            });
        }
        if let Some(index) = artifact
            .inputs()
            .iter()
            .chain(artifact.outputs())
            .position(|b| b.shape().len() > MAX_RANK)
        {
            return Err(VpuError::Abi {
                key: key.to_owned(),
                reason: format!("argument {index} has more than {MAX_RANK} dimensions"),
            });
        }
        let graph = self.graph_of(key, artifact)?;
        check_abi(key, artifact, &graph)?;
        let bindings = artifact
            .inputs()
            .iter()
            .chain(artifact.outputs())
            .map(|b| Binding {
                storage: b.storage(),
                elements: b.elements().expect("the artifact validated every shape"),
            })
            .collect();
        tracing::debug!(
            key,
            arguments = artifact.inputs().len() + artifact.outputs().len(),
            "loaded an Intel NPU graph"
        );
        Ok(VpuKernel(Rc::new(LoadedKernel {
            key: key.to_owned(),
            graph,
            target: artifact.target(),
            digest: artifact.digest(),
            bindings,
        })))
    }

    fn begin(&mut self) -> Result<(), VpuError> {
        self.healthy()?;
        if self.open.is_some() {
            return Err(VpuError::AlreadyRecording);
        }
        self.open = Some(
            self.context
                .begin_recording()
                .map_err(l0("begin recording"))?,
        );
        Ok(())
    }

    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), VpuError> {
        self.healthy()?;
        let invocation = match dispatch.launch {
            LaunchKind::VpuGraph(invocation) => invocation,
            LaunchKind::Grid { threads, .. } => return Err(VpuError::GridLaunch { threads }),
            LaunchKind::Aie(_) => return Err(VpuError::AieLaunch),
        };
        let kernel = Rc::clone(&dispatch.kernel.0);
        check_invocation(&kernel, invocation)?;
        let args: Vec<&Arg<'_, Self>> = dispatch
            .inputs
            .iter()
            .chain(std::iter::once(&dispatch.output))
            .collect();
        check_bindings(&kernel, &args)?;
        let hosts: Vec<Ref<'_, L0HostBuffer>> =
            args.iter().map(|arg| arg.buffer.0.host.borrow()).collect();
        let buffers: Vec<&L0HostBuffer> = hosts.iter().map(|host| &**host).collect();
        self.recording()?
            .execute_graph(&kernel.graph, &buffers)
            .map_err(l0("append graph execution"))?;
        if let Some(guard) = &mut self.guard {
            let guard_arg = |arg: &Arg<'_, Self>, bound: &Binding| {
                (
                    GuardArg {
                        buffer: arg.buffer.clone(),
                        role: arg.buffer.0.role,
                        elements: usize::try_from(bound.elements)
                            .expect("check_bindings sized the argument"),
                    },
                    bound.storage.element(),
                )
            };
            let (output, inputs) = args.split_last().expect("a dispatch has an output");
            let inputs: Vec<_> = inputs
                .iter()
                .zip(&kernel.bindings)
                .map(|(arg, bound)| guard_arg(arg, bound))
                .collect();
            let output = guard_arg(output, &kernel.bindings[inputs.len()]);
            guard.executed(&kernel.key, &inputs, output);
        }
        Ok(())
    }

    fn copy(&mut self, src: &VpuBuffer, dst: &VpuBuffer) -> Result<(), VpuError> {
        self.healthy()?;
        if src.bytes() != dst.bytes() {
            return Err(VpuError::CopyLength {
                src: src.bytes(),
                dst: dst.bytes(),
            });
        }
        self.recording()?
            .copy(&dst.0.host.borrow(), &src.0.host.borrow())
            .map_err(l0("append copy"))?;
        if let Some(guard) = &mut self.guard {
            guard.copied(dst, dst.0.storage.element(), dst.bytes() / size_of::<f32>());
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<VpuRecording, VpuError> {
        let mut builder = self.open.take().ok_or(VpuError::NotRecording)?;
        let list = builder.finish().map_err(l0("close recording"))?;
        let ranges = self
            .guard
            .as_mut()
            .map_or_else(|| Rc::from([]), RangeGuard::finished);
        Ok(VpuRecording { list, ranges })
    }

    fn replay(&mut self, recording: &VpuRecording) -> Result<(), VpuError> {
        self.healthy()?;
        #[cfg(any(test, feature = "test-support"))]
        {
            self.runs += 1;
        }
        // An input the contract does not cover is refused before it reaches the device; nothing ran.
        if let Some(guard) = &mut self.guard {
            guard.verify(&recording.ranges, Direction::Input, read_argument)?;
        }
        let ran = self
            .context
            .execute(&recording.list, self.config.run_timeout)
            .map_err(l0("execute"));
        if let Err(error) = &ran {
            self.poison(recording, error);
            return ran;
        }
        // A result that left the range overflowed inside the region; the device is healthy, the result is not.
        if let Some(guard) = &mut self.guard {
            guard.verify(&recording.ranges, Direction::Result, read_argument)?;
        }
        ran
    }

    fn write_with(
        &mut self,
        dst: &VpuBuffer,
        offset: usize,
        len: usize,
        fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), VpuError> {
        self.healthy()?;
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= dst.bytes())
            .ok_or(VpuError::Range {
                offset,
                end: offset.saturating_add(len),
                len: dst.bytes(),
            })?;
        // The buffer is the NPU's own pages: the whole range is one chunk, no staging copy. A constant's range
        // is read from these bytes, once, as they are written.
        let (role, element) = (dst.0.role, dst.0.storage.element());
        let mut admitted = Ok(());
        let guard = &mut self.guard;
        dst.0
            .host
            .borrow_mut()
            .write(offset..end, |bytes| {
                fill(offset..end, bytes);
                if let Some(guard) = guard {
                    admitted = guard
                        .uploaded(role, element, offset, bytes)
                        .map_err(VpuError::from);
                }
            })
            .map_err(l0("host write"))?;
        admitted
    }

    fn read(&mut self, src: &VpuBuffer, out: &mut [u8]) -> Result<(), VpuError> {
        self.healthy()?;
        if out.len() > src.bytes() {
            return Err(VpuError::Range {
                offset: 0,
                end: out.len(),
                len: src.bytes(),
            });
        }
        src.0.host.borrow().read(0, out).map_err(l0("host read"))
    }

    fn synchronize(&mut self) -> Result<(), VpuError> {
        // Replay waits for the queue: every execution completed (or failed) before `replay` returned.
        self.healthy()
    }

    fn device_time(&self) -> DeviceTime {
        // The graph extension reports device timestamps only through a profiling query this device does not
        // issue; a host wall clock is not device time, so none is reported.
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), VpuError> {
        // A partially built recording is dropped whole; nothing has run on the device for it.
        self.open = None;
        if let Some(guard) = &mut self.guard {
            guard.aborted();
        }
        Ok(())
    }
}

/// The contents of the first `bytes` bytes of an argument's buffer.
fn read_argument(buffer: &VpuBuffer, bytes: usize) -> Result<Vec<u8>, VpuError> {
    let mut contents = vec![0u8; bytes];
    buffer
        .0
        .host
        .borrow()
        .read(0, &mut contents)
        .map_err(l0("host read"))?;
    Ok(contents)
}

/// The compiled graph's arguments are the region's ABI, argument for argument: the same names in the same
/// order, the same direction, the same extents and the same precision. A vendor compiler that reordered,
/// retyped or reshaped an argument is a typed refusal, never a wrong answer.
fn check_abi(key: &str, artifact: &VpuGraphArtifact, graph: &L0Graph) -> Result<(), VpuError> {
    let refuse = |reason: String| VpuError::Abi {
        key: key.to_owned(),
        reason,
    };
    let expected = artifact
        .inputs()
        .iter()
        .map(|b| (b, GraphDirection::Input))
        .chain(
            artifact
                .outputs()
                .iter()
                .map(|b| (b, GraphDirection::Output)),
        );
    let arguments = graph.arguments();
    let declared = artifact.inputs().len() + artifact.outputs().len();
    if arguments.len() != declared {
        return Err(refuse(format!(
            "the region has {declared} arguments, the graph {}",
            arguments.len()
        )));
    }
    for (index, ((binding, direction), argument)) in expected.zip(arguments).enumerate() {
        if argument.name != binding.name() {
            return Err(refuse(format!(
                "argument {index} is `{}` in the graph, `{}` in the region",
                argument.name,
                binding.name()
            )));
        }
        if argument.direction != direction {
            return Err(refuse(format!(
                "argument {index} (`{}`) is {:?} in the graph, {direction:?} in the region",
                binding.name(),
                argument.direction
            )));
        }
        let padded = padded_dims(binding.shape());
        if argument.dims != padded {
            return Err(refuse(format!(
                "argument {index} (`{}`) has dimensions {:?} in the graph, {:?} in the region",
                binding.name(),
                argument.dims,
                binding.shape()
            )));
        }
        let (_, want) = graph_precision(key, binding.name(), binding.storage().element())?;
        if argument.precision != want {
            return Err(refuse(format!(
                "argument {index} (`{}`) is {:?} in the compiled graph, {want:?} in the region: the \
                 vendor compiler would convert it at the boundary",
                binding.name(),
                argument.precision
            )));
        }
    }
    Ok(())
}

/// The extents of `shape` as the driver reports them: padded on the right with ones to five.
fn padded_dims(shape: &[usize]) -> Vec<u32> {
    let mut dims: Vec<u32> = shape
        .iter()
        .map(|&d| u32::try_from(d).unwrap_or(u32::MAX))
        .collect();
    dims.resize(MAX_RANK, 1);
    dims
}

/// The dispatch's invocation is the artifact of the kernel it carries.
fn check_invocation(kernel: &LoadedKernel, invocation: VpuInvocation) -> Result<(), VpuError> {
    if invocation.target() == kernel.target && invocation.digest() == kernel.digest {
        return Ok(());
    }
    Err(VpuError::InvocationMismatch {
        key: kernel.key.clone(),
        invoked_target: invocation.target(),
        invoked_digest: invocation.digest(),
        kernel_target: kernel.target,
        kernel_digest: kernel.digest,
    })
}

/// The dispatch's arguments (every tensor but the last, then the `output`) are the kernel's ABI, binding for
/// binding: the count, and for each binding its storage and element count.
fn check_bindings(kernel: &LoadedKernel, args: &[&Arg<'_, VpuDevice>]) -> Result<(), VpuError> {
    let refuse = |reason: String| VpuError::Bindings {
        key: kernel.key.clone(),
        reason,
    };
    if args.len() != kernel.bindings.len() {
        return Err(refuse(format!(
            "the graph has {} arguments, the dispatch binds {}",
            kernel.bindings.len(),
            args.len()
        )));
    }
    for (index, (arg, binding)) in args.iter().zip(&kernel.bindings).enumerate() {
        if arg.buffer.0.storage != binding.storage {
            return Err(refuse(format!(
                "argument {index} is {:?}, the bound buffer is {:?}",
                binding.storage, arg.buffer.0.storage
            )));
        }
        if u64::from(arg.elems) != binding.elements {
            return Err(refuse(format!(
                "argument {index} is {} elements, the dispatch binds {}",
                binding.elements, arg.elems
            )));
        }
        let width = binding.storage.element().byte_width();
        let needed = usize::try_from(binding.elements)
            .ok()
            .and_then(|elements| elements.checked_mul(width))
            .ok_or(VpuError::Size {
                elems: binding.elements,
                width,
            })?;
        if arg.buffer.bytes() < needed {
            return Err(refuse(format!(
                "argument {index} needs {needed} bytes, the bound buffer holds {}",
                arg.buffer.bytes()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region layer the compiler's supported set leaves out is reported with the equation it lowers; the
    /// layers it lists are not.
    ///
    /// Mutation: invert the filter in `rejected_layers`; the supported layer is reported instead.
    #[test]
    fn a_layer_missing_from_the_supported_set_is_reported_with_its_equation() {
        let layers = [VpuLayer::new("v7", 7), VpuLayer::new("v9", 9)];
        let supported = ["a".to_owned(), "v7".to_owned(), "result_0".to_owned()];
        let rejected = rejected_layers(&layers, &supported);
        assert_eq!(rejected.len(), 1);
        assert_eq!((rejected[0].name(), rejected[0].origin()), ("v9", 9));
    }

    /// Every element kind is named or refused: BF16 and raw bytes are typed refusals, never an FP32 the
    /// compiler would be told in their place.
    ///
    /// Mutation: map `Bf16` to `("FP32", GraphPrecision::Fp32)` in `graph_precision`; the refusal is gone.
    #[test]
    fn an_element_kind_the_boundary_does_not_carry_is_refused_not_widened() {
        use poot_target::ElementKind;
        assert!(matches!(
            graph_precision("k", "x", ElementKind::F16),
            Ok(("FP16", GraphPrecision::Fp16))
        ));
        assert!(matches!(
            graph_precision("k", "x", ElementKind::F32),
            Ok(("FP32", GraphPrecision::Fp32))
        ));
        assert!(matches!(
            graph_precision("k", "x", ElementKind::I32),
            Ok(("I32", GraphPrecision::Int32))
        ));
        for element in [ElementKind::Bf16, ElementKind::RawBytes] {
            let Err(VpuError::Element {
                key,
                binding,
                element: refused,
            }) = graph_precision("k", "x", element)
            else {
                panic!("{element:?} must be refused");
            };
            assert_eq!(
                (key.as_str(), binding.as_str(), refused),
                ("k", "x", element)
            );
        }
    }

    /// The driver reports an argument padded on the right to five dimensions; the region's shape compares
    /// against that.
    #[test]
    fn dimensions_are_padded_to_five_on_the_right() {
        assert_eq!(padded_dims(&[2, 3]), [2, 3, 1, 1, 1]);
        assert_eq!(padded_dims(&[]), [1, 1, 1, 1, 1]);
        assert_eq!(padded_dims(&[1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]);
    }

    /// Every cache key input changes the blob's identity: the region's IR, its constants, the compiler version and
    /// the option string.
    ///
    /// Mutation: leave the compiler version out of `BlobKey::new`; the compiler row goes red.
    #[test]
    fn the_blob_key_covers_the_region_the_compiler_and_the_options() {
        let artifact = |xml: &str, weights: &[u8]| {
            VpuGraphArtifact::new(
                VpuTargetId::Npu4,
                xml.as_bytes().to_vec(),
                weights.to_vec(),
                vec![VpuBinding::new("a", BufferStorage::f16(), vec![4])],
                vec![VpuBinding::new("v1", BufferStorage::f16(), vec![4])],
                vec![VpuLayer::new("v1", 1)],
            )
            .unwrap()
        };
        let compiler = VendorCompilerId { major: 8, minor: 3 };
        let base = BlobKey::new(&artifact("<net/>", &[1]), compiler, "--opts");
        assert_eq!(
            base,
            BlobKey::new(&artifact("<net/>", &[1]), compiler, "--opts")
        );
        assert_ne!(
            base,
            BlobKey::new(&artifact("<net />", &[1]), compiler, "--opts")
        );
        assert_ne!(
            base,
            BlobKey::new(&artifact("<net/>", &[2]), compiler, "--opts")
        );
        assert_ne!(
            base,
            BlobKey::new(
                &artifact("<net/>", &[1]),
                VendorCompilerId { major: 8, minor: 4 },
                "--opts"
            )
        );
        assert_ne!(
            base,
            BlobKey::new(&artifact("<net/>", &[1]), compiler, "--other")
        );
    }
}
