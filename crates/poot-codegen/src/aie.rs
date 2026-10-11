//! AIE compilation: a sequential `Body` to a Peano core object, a typed link recipe, and a loadable
//! xclbin/instruction artifact (POOT-664, spec 1002).
//!
//! ```text
//! Body + AieShapeAbi --check_body_abi--> AieLinkRecipe --Peano--> AieObjectPlan --IRON--> AieLinkedArtifact
//! ```
//!
//! The core object is an intermediate, never a runnable module. The linked artifact is the only thing
//! [`crate::kernel_handle`] / [`crate::aie_kernel_handle`] wrap, and the runtime (POOT-641) only loads it:
//! it neither runs IRON nor recognizes graphs.
//!
//! # Identity and publication
//!
//! Two canonical, length-tagged texts name an artifact ([`AieIdentity`]): the core object's (the whole
//! verified Body, target, numerical contract, core ABI revision, codegen source and the Peano build) and the
//! link's (the core digest, every binding's storage, dimensions and access, the recipe, platform, the IRON
//! build, flags, instruction format and finalization policy). A shape, layout, recipe or tool change
//! names a different cache entry. Both tools are resolved and checked before any scratch exists, and the
//! bundle is published as one cache entry after both link outputs were read and validated, so a failure at
//! any stage leaves no entry.
//!
//! # The IRON link driver protocol
//!
//! `$POOT_AIE_IRON` names an executable (the compiler does not embed Python):
//!
//! ```text
//! POOT_AIE_IRON --version     first stdout line: the driver and IRON build identity
//! POOT_AIE_IRON --platforms   the platform names it can link for, one per line
//! POOT_AIE_IRON link --platform <name> --object <core.o> --recipe <recipe.txt> \
//!                    --xclbin <out> --insts <out>
//! ```
//!
//! The recipe file is [`AieLinkRecipe::canonical`]. Success is exit status 0 with both outputs written.
//!
//! The one implementation is `scripts/poot-aie-link.py` (POOT-1118): it builds the one-core IRON program the
//! recipe describes and links it with mlir-aie's `aiecc`. The flake's `npu` devshell (`nix develop .#npu`)
//! exports `POOT_AIE_IRON` as the wrapper of it and `POOT_AIE_LLC` as the pinned Peano `llc`; anywhere else
//! the variables are unset and the compile is the typed [`AieCompileError::ToolchainMissing`]. The
//! model-free tests exercise this orchestration against scripted drivers, which prove the protocol handling
//! and typed failures, not machine code; the ignored
//! `a_real_peano_and_iron_toolchain_links_a_loadable_artifact` row runs the real pair.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};

use poot_kernel_ir::Body;
use poot_runtime_common::{AieArtifactError, AieLinkedArtifact, AieShapeAbi, ArgAccess};
use poot_target::{AieTargetId, ElementKind, StorageLayout};

use crate::cache::{
    ArtifactTooLarge, KernelCache, KernelKey, aie_entry_key, body_fingerprint, codegen_src_hash,
};
use crate::handle::try_arg_schema;
use crate::{CompileError, EmitError, Target, Toolchain, compile_with};

/// The kernel name inside every xclbin the IRON harness builds (`NPUKernel(..., kernel_name="MLIR_AIE")`).
const IRON_KERNEL_NAME: &str = "MLIR_AIE";
/// What the core computes: scalar F32 with the AIE2p unit's own rounding (not IEEE-RNE for add; see the
/// NPU notes), no fused contraction. Part of the core identity.
const NUMERICAL_CONTRACT: &str = "f32-scalar-aie2p-v1";
/// The core object's ABI: interleaved `(ptr, u32 len)` pairs in binding order.
const CORE_ABI_REVISION: u32 = 1;
/// The instruction stream format the driver emits.
const INSTRUCTION_FORMAT: &str = "npu-insts-v1";
/// Revision of what finalization (limits, resource accounting) requires of an artifact.
const FINALIZATION_POLICY_REVISION: u32 = 1;
/// The driver's link flags that change its output.
const LINK_FLAGS: &str = "peano-no-xchesscc";

/// A stage of the AIE pipeline, for typed toolchain causes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AieStage {
    /// Peano's `llc`: the core object.
    Peano,
    /// The IRON link driver: the xclbin and instruction stream.
    Iron,
}

impl std::fmt::Display for AieStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AieStage::Peano => "Peano (core object)",
            AieStage::Iron => "IRON (link)",
        })
    }
}

/// The body and the shape ABI do not describe the same kernel, or the ABI cannot be linked.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AieAbiMismatch {
    #[error("body parameter {index} is not a flat buffer slice of a runtime element type")]
    Parameter { index: usize },
    #[error("the body takes {body} buffers and the shape ABI binds {abi}")]
    Count { body: usize, abi: usize },
    #[error("binding {index}: the body's element is {body}, the shape ABI's is {abi}")]
    Element {
        index: usize,
        body: ElementKind,
        abi: ElementKind,
    },
    #[error("binding {index}: the body writes it = {body_writes}, the shape ABI says {abi_writes}")]
    Access {
        index: usize,
        body_writes: bool,
        abi_writes: bool,
    },
    #[error("the body can trap, and the AIE ABI has no error word")]
    Trap,
    #[error("a binding is not dense row-major storage")]
    Layout { index: usize },
    #[error("the recipe's tile-local byte total overflows")]
    TileLocalOverflow,
}

/// Why an AIE compilation produced no artifact. Every variant leaves nothing published.
#[derive(Debug, thiserror::Error)]
pub enum AieCompileError {
    #[error(transparent)]
    Emit(#[from] EmitError),
    #[error(transparent)]
    Abi(#[from] AieAbiMismatch),
    /// A required tool is not configured (`variable` unset) or not found/runnable at its configured path.
    #[error("the {stage} toolchain is unavailable: ${variable} is unset or names no runnable tool")]
    ToolchainMissing {
        stage: AieStage,
        variable: &'static str,
    },
    /// The tool runs but does not support `target` (no `aie2p` backend, or no such platform).
    #[error("the {stage} toolchain does not support {target:?}")]
    ToolchainUnsupportedTarget {
        stage: AieStage,
        target: AieTargetId,
    },
    #[error("{stage} failed (status {status}):\n{stderr}")]
    Tool {
        stage: AieStage,
        status: String,
        stderr: String,
    },
    #[error("the core object is not an ELF carrying the symbol `{symbol}`")]
    CoreObject { symbol: String },
    #[error("the IRON driver reported success but wrote no {output}")]
    LinkOutput { output: &'static str },
    /// The linked bundle failed the artifact validator.
    #[error(transparent)]
    Artifact(#[from] AieArtifactError),
    #[error(transparent)]
    TooLarge(#[from] ArtifactTooLarge),
    /// A cache entry under this identity describes a different kernel than the request (a stale or
    /// colliding entry): it is refused, never loaded.
    #[error("the cached artifact's {what} differs from the request")]
    StaleEntry { what: &'static str },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Whether a FIFO feeds the core from the host or drains it back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FifoDirection {
    HostToTile,
    TileToHost,
}

/// One object FIFO of a link recipe: the host binding it moves, its direction, depth and element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AieFifo {
    pub binding: usize,
    pub direction: FifoDirection,
    /// Buffers the FIFO holds in flight.
    pub depth: u32,
    /// Elements per FIFO buffer (a whole binding, in the sequential schedule).
    pub elements: u64,
    pub element: ElementKind,
}

/// The typed shape/layout recipe IRON links a core object with: which symbol runs on the one core, what
/// each host binding is, and the FIFO that moves it. Data, not a script.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieLinkRecipe {
    symbol: String,
    target: AieTargetId,
    abi: AieShapeAbi,
    fifos: Vec<AieFifo>,
    /// One finite invocation: the core runs its loop once over the whole tensors.
    invocations: u64,
    /// Tile-local bytes every FIFO buffer reserves at its depth, checked once at construction.
    tile_local_bytes: u64,
}

impl AieLinkRecipe {
    /// IRON's default `ObjectFifo` depth.
    pub const FIFO_DEPTH: u32 = 2;

    /// The one-core sequential schedule: one whole-tensor FIFO per binding, written bindings drained,
    /// read bindings fed, one invocation.
    pub fn sequential(
        symbol: &str,
        target: AieTargetId,
        abi: &AieShapeAbi,
    ) -> Result<Self, AieAbiMismatch> {
        let fifos: Vec<AieFifo> = abi
            .bindings()
            .iter()
            .enumerate()
            .map(|(binding, b)| AieFifo {
                binding,
                direction: match b.access() {
                    ArgAccess::Read => FifoDirection::HostToTile,
                    ArgAccess::Write => FifoDirection::TileToHost,
                },
                depth: Self::FIFO_DEPTH,
                elements: b.elements(),
                element: b.storage().element(),
            })
            .collect();
        let tile_local_bytes = fifos
            .iter()
            .try_fold(0u64, |total, fifo| {
                let one = fifo
                    .elements
                    .checked_mul(fifo.element.byte_width() as u64)?;
                total.checked_add(one.checked_mul(u64::from(fifo.depth))?)
            })
            .ok_or(AieAbiMismatch::TileLocalOverflow)?;
        Ok(Self {
            symbol: symbol.to_owned(),
            target,
            abi: abi.clone(),
            fifos,
            invocations: 1,
            tile_local_bytes,
        })
    }

    /// The target this recipe links for.
    pub fn target(&self) -> AieTargetId {
        self.target
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    pub fn fifos(&self) -> &[AieFifo] {
        &self.fifos
    }

    pub fn abi(&self) -> &AieShapeAbi {
        &self.abi
    }

    pub fn invocations(&self) -> u64 {
        self.invocations
    }

    /// Tile-local bytes every FIFO buffer reserves at its depth (no overflow: construction refused one).
    pub fn tile_local_bytes(&self) -> u64 {
        self.tile_local_bytes
    }

    /// The recipe's canonical text: versioned, tagged and length-prefixed, and the file the driver reads.
    pub fn canonical(&self) -> String {
        let mut out = String::from("poot-aie-recipe-v1\n");
        field(&mut out, "symbol", &self.symbol);
        field(&mut out, "target", &self.target.canonical());
        field(&mut out, "invocations", &self.invocations.to_string());
        for fifo in &self.fifos {
            field(
                &mut out,
                "fifo",
                &format!(
                    "binding={};dir={:?};depth={};elems={};elem={:?}",
                    fifo.binding, fifo.direction, fifo.depth, fifo.elements, fifo.element
                ),
            );
        }
        out.push_str(&abi_canonical(&self.abi));
        out
    }
}

fn field(out: &mut String, tag: &str, value: &str) {
    let _ = writeln!(out, "{tag}[{}]={value}", value.len());
}

/// The shape ABI in canonical text: ordered, with every dimension (equal element counts under different
/// dimensions are different ABIs), the full storage record, access and proved access range.
fn abi_canonical(abi: &AieShapeAbi) -> String {
    let mut out = String::new();
    for (index, binding) in abi.bindings().iter().enumerate() {
        let storage = binding.storage();
        field(
            &mut out,
            "binding",
            &format!(
                "index={index};element={:?};dtype={:?};layout={:?};access={:?};dims={:?};accessed={}",
                storage.element(),
                storage.dtype(),
                storage.layout(),
                binding.access(),
                binding.dims(),
                binding.accessed_elements()
            ),
        );
    }
    out
}

/// Check that `body`'s buffer parameters and `abi`'s bindings describe one kernel: count, element kind and
/// which buffers are written, in order. A body that can trap is refused (the ABI binds no error word).
pub(crate) fn check_body_abi(body: &Body, abi: &AieShapeAbi) -> Result<(), AieAbiMismatch> {
    if body.has_trap() {
        return Err(AieAbiMismatch::Trap);
    }
    let schema = try_arg_schema(body).map_err(|index| AieAbiMismatch::Parameter { index })?;
    if schema.len() != abi.bindings().len() {
        return Err(AieAbiMismatch::Count {
            body: schema.len(),
            abi: abi.bindings().len(),
        });
    }
    for (index, (arg, binding)) in schema.iter().zip(abi.bindings()).enumerate() {
        if binding.storage().layout() != StorageLayout::Dense {
            return Err(AieAbiMismatch::Layout { index });
        }
        if arg.element() != binding.storage().element() {
            return Err(AieAbiMismatch::Element {
                index,
                body: arg.element(),
                abi: binding.storage().element(),
            });
        }
        if arg.access() != binding.access() {
            return Err(AieAbiMismatch::Access {
                index,
                body_writes: arg.access() == ArgAccess::Write,
                abi_writes: binding.access() == ArgAccess::Write,
            });
        }
    }
    Ok(())
}

/// A resolved tool: where it runs from, the first line of its `--version` and a hash of its bytes. The
/// path and version line alone cannot see a binary replaced under the same name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieToolIdentity {
    pub path: String,
    pub version: String,
    pub content: u64,
}

/// The two tools an AIE compilation needs, resolved and checked against the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieTools {
    pub peano: AieToolIdentity,
    pub iron: AieToolIdentity,
}

impl AieTools {
    /// The identity text of the pair, as an artifact records it.
    fn canonical(&self) -> String {
        format!(
            "peano={}@{:016x};iron={}@{:016x}",
            self.peano.version, self.peano.content, self.iron.version, self.iron.content
        )
    }
}

/// The path a configured tool name runs from, or `None` when no such file exists.
fn locate(name: &str) -> Option<PathBuf> {
    if name.contains('/') {
        return Path::new(name).is_file().then(|| PathBuf::from(name));
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|candidate| candidate.is_file())
    })
}

/// FNV-1a of a tool's bytes, memoized per path for the process (a toolchain is not swapped under a
/// running process).
fn content_hash(path: &Path) -> std::io::Result<u64> {
    static HASHES: LazyLock<Mutex<HashMap<PathBuf, u64>>> = LazyLock::new(Mutex::default);
    let mut hashes = HASHES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&hash) = hashes.get(path) {
        return Ok(hash);
    }
    let hash = poot_runtime_common::fnv1a_bytes(&std::fs::read(path)?);
    hashes.insert(path.to_path_buf(), hash);
    Ok(hash)
}

/// Run `tool args..` and return its stdout, mapping "cannot run" to the stage's missing-toolchain cause.
fn query(
    stage: AieStage,
    variable: &'static str,
    path: &Path,
    args: &[&str],
) -> Result<String, AieCompileError> {
    let output = match Command::new(path).args(args).output() {
        Ok(output) => output,
        Err(error)
            if matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            return Err(AieCompileError::ToolchainMissing { stage, variable });
        }
        Err(error) => return Err(error.into()),
    };
    if !output.status.success() {
        return Err(AieCompileError::Tool {
            stage,
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn resolve_stage(
    stage: AieStage,
    variable: &'static str,
    target: AieTargetId,
    configured: Option<&str>,
    supports: impl Fn(&Path) -> Result<bool, AieCompileError>,
) -> Result<AieToolIdentity, AieCompileError> {
    let missing = || AieCompileError::ToolchainMissing { stage, variable };
    let path = configured.and_then(locate).ok_or_else(missing)?;
    let version = query(stage, variable, &path, &["--version"])?
        .lines()
        .next()
        .map(str::to_owned)
        .unwrap_or_default();
    if !supports(&path)? {
        return Err(AieCompileError::ToolchainUnsupportedTarget { stage, target });
    }
    let content = content_hash(&path).map_err(|_| missing())?;
    Ok(AieToolIdentity {
        path: path.to_string_lossy().into_owned(),
        version,
        content,
    })
}

/// Resolve and check both tools `target` needs, before any scratch is written or entry published.
/// Peano must list `target`'s ISA among its backends; the IRON driver must list its platform.
pub(crate) fn resolve_aie_tools(
    toolchain: &Toolchain,
    target: AieTargetId,
) -> Result<AieTools, AieCompileError> {
    let peano = resolve_stage(
        AieStage::Peano,
        "POOT_AIE_LLC",
        target,
        toolchain.aie_llc.as_deref(),
        |path| {
            Ok(
                query(AieStage::Peano, "POOT_AIE_LLC", path, &["--version"])?
                    .contains(target.isa().march()),
            )
        },
    )?;
    let iron = resolve_stage(
        AieStage::Iron,
        "POOT_AIE_IRON",
        target,
        toolchain.aie_iron.as_deref(),
        |path| {
            Ok(
                query(AieStage::Iron, "POOT_AIE_IRON", path, &["--platforms"])?
                    .lines()
                    .any(|line| line.trim() == target.platform().name()),
            )
        },
    )?;
    Ok(AieTools { peano, iron })
}

/// The content identity of one linked artifact (see the module doc): the text of the core object's and
/// the link's identities and their digests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieIdentity {
    core: String,
    link: String,
}

impl AieIdentity {
    /// The identity of linking `body` with `recipe` using `tools`. Pure over its inputs: it reads no
    /// environment and starts no process, so a test names exactly the dimension it varies.
    pub fn new(body: &Body, recipe: &AieLinkRecipe, tools: &AieTools) -> Self {
        let target = recipe.target;
        let mut core = String::from("poot-aie-core-v1\n");
        field(
            &mut core,
            "body",
            &format!("{:016x}", body_fingerprint(body)),
        );
        field(&mut core, "symbol", &body.name);
        field(&mut core, "isa", target.isa().march());
        field(&mut core, "numerics", NUMERICAL_CONTRACT);
        field(&mut core, "core-abi", &CORE_ABI_REVISION.to_string());
        field(&mut core, "codegen", codegen_src_hash());
        field(&mut core, "flags", &Target::AieCore.llc_args().join(" "));
        field(&mut core, "peano", &tools.peano.canonical());
        let mut link = String::from("poot-aie-link-v1\n");
        field(
            &mut link,
            "core",
            &format!("{:016x}", poot_runtime_common::fnv1a_bytes(core.as_bytes())),
        );
        link.push_str(&recipe.canonical());
        field(&mut link, "platform", target.platform().name());
        field(&mut link, "abi", &target.abi_revision().to_string());
        field(&mut link, "iron", &tools.iron.canonical());
        field(&mut link, "flags", LINK_FLAGS);
        field(&mut link, "insts", INSTRUCTION_FORMAT);
        field(
            &mut link,
            "finalization",
            &FINALIZATION_POLICY_REVISION.to_string(),
        );
        Self { core, link }
    }

    pub fn core_digest(&self) -> u64 {
        poot_runtime_common::fnv1a_bytes(self.core.as_bytes())
    }

    pub fn link_digest(&self) -> u64 {
        poot_runtime_common::fnv1a_bytes(self.link.as_bytes())
    }

    /// The cache entry this identity names.
    pub fn cache_key(&self) -> KernelKey {
        aie_entry_key(self.core_digest(), self.link_digest())
    }
}

impl AieToolIdentity {
    fn canonical(&self) -> String {
        format!("{}@{:016x}", self.version, self.content)
    }
}

/// A compiled core object with the typed recipe it will be linked with. Not runnable.
#[derive(Clone, Debug)]
pub(crate) struct AieObjectPlan {
    pub symbol: String,
    pub object: Vec<u8>,
    pub recipe: AieLinkRecipe,
}

/// Compile `body` to an AIE core object and pair it with its recipe. `scratch` is a private directory the
/// caller owns and removes.
pub(crate) fn build_object_plan(
    body: &Body,
    recipe: &AieLinkRecipe,
    toolchain: &Toolchain,
    scratch: &Path,
) -> Result<AieObjectPlan, AieCompileError> {
    check_body_abi(body, recipe.abi())?;
    let out = scratch.join(format!("{}.o", body.name));
    compile_with(body, Target::AieCore, &out, toolchain).map_err(|error| match error {
        CompileError::Emit(error) => AieCompileError::Emit(error),
        CompileError::Io(error) => AieCompileError::Io(error),
        CompileError::ToolchainMissing { variable, .. } => AieCompileError::ToolchainMissing {
            stage: AieStage::Peano,
            variable,
        },
        CompileError::Llc { status, stderr, .. } => AieCompileError::Tool {
            stage: AieStage::Peano,
            status,
            stderr,
        },
        CompileError::NoContraction(_) | CompileError::ArtifactTooLarge(_) => {
            unreachable!("only SPIR-V post-processing and bounded cache reads raise this")
        }
    })?;
    let object = std::fs::read(&out)?;
    // A smoke check, not a symbol-table lookup: an ELF whose bytes name the core symbol. An empty name has
    // no symbol to find, so it is the same typed refusal.
    if !object.starts_with(b"\x7fELF")
        || body.name.is_empty()
        || !object
            .windows(body.name.len())
            .any(|window| window == body.name.as_bytes())
    {
        return Err(AieCompileError::CoreObject {
            symbol: body.name.clone(),
        });
    }
    Ok(AieObjectPlan {
        symbol: body.name.clone(),
        object,
        recipe: recipe.clone(),
    })
}

/// Link `plan` into a loadable artifact with the IRON driver. Both outputs are read and the bundle
/// validated before it is returned; a failure at any point returns an error and nothing else.
pub(crate) fn link_object(
    plan: &AieObjectPlan,
    tools: &AieTools,
    target: AieTargetId,
    scratch: &Path,
) -> Result<AieLinkedArtifact, AieCompileError> {
    let object = scratch.join(format!("{}.link.o", plan.symbol));
    let recipe = scratch.join("recipe.txt");
    let xclbin = scratch.join("final.xclbin");
    let insts = scratch.join("insts.bin");
    std::fs::write(&object, &plan.object)?;
    std::fs::write(&recipe, plan.recipe.canonical())?;
    let output = Command::new(&tools.iron.path)
        .args(["link", "--platform", target.platform().name()])
        .arg("--object")
        .arg(&object)
        .arg("--recipe")
        .arg(&recipe)
        .arg("--xclbin")
        .arg(&xclbin)
        .arg("--insts")
        .arg(&insts)
        .output()?;
    if !output.status.success() {
        return Err(AieCompileError::Tool {
            stage: AieStage::Iron,
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    let read = |path: &Path, output: &'static str| {
        std::fs::read(path).map_err(|error| match error.kind() {
            std::io::ErrorKind::NotFound => AieCompileError::LinkOutput { output },
            _ => AieCompileError::Io(error),
        })
    };
    let xclbin = read(&xclbin, "xclbin")?;
    let insts = read(&insts, "instruction stream")?;
    let tile_local = plan.recipe.tile_local_bytes();
    Ok(AieLinkedArtifact::new(
        xclbin,
        insts,
        IRON_KERNEL_NAME,
        plan.recipe.abi().clone(),
        target,
        &tools.canonical(),
        tile_local,
    )?)
}

/// A removed-on-drop private scratch directory beside the cache entries.
struct Scratch(PathBuf);

impl Scratch {
    fn create(cache: &KernelCache, key: &crate::KernelKey) -> std::io::Result<Self> {
        let path = cache.scratch_path(key, "work");
        std::fs::create_dir_all(&path)?;
        Ok(Self(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A linked AIE artifact and whether this call built it.
#[derive(Debug)]
pub struct AieCompiled {
    pub artifact: AieLinkedArtifact,
    pub built: bool,
    pub identity: AieIdentity,
}

/// The whole pipeline: resolve the tools, then serve the artifact for `(body, recipe)` from `cache`
/// or build it (core object, recipe, IRON link), publishing the validated bundle as one entry. Tool
/// resolution comes first, so a missing or unsupported toolchain fails before any scratch exists or any
/// entry is written.
pub fn compile_aie(
    body: &Body,
    recipe: &AieLinkRecipe,
    toolchain: &Toolchain,
    cache: &KernelCache,
    max_artifact_bytes: NonZeroU64,
) -> Result<AieCompiled, AieCompileError> {
    let (abi, target) = (recipe.abi(), recipe.target());
    let tools = resolve_aie_tools(toolchain, target)?;
    check_body_abi(body, abi)?;
    let identity = AieIdentity::new(body, recipe, &tools);
    let key = identity.cache_key();
    let built = cache.load_or_build(&key, max_artifact_bytes, || {
        let scratch = Scratch::create(cache, &key)?;
        let plan = build_object_plan(body, recipe, toolchain, &scratch.0)?;
        Ok::<_, AieCompileError>(link_object(&plan, &tools, target, &scratch.0)?.to_bytes())
    })?;
    let artifact = AieLinkedArtifact::from_bytes(&built.bytes, target)?;
    // The identity is a 64-bit digest pair: an entry that does not describe this request (a stale file or a
    // collision) is refused here, whether it was just built or served.
    if artifact.abi() != abi {
        return Err(AieCompileError::StaleEntry { what: "shape ABI" });
    }
    if artifact.toolchain() != tools.canonical() {
        return Err(AieCompileError::StaleEntry { what: "toolchain" });
    }
    Ok(AieCompiled {
        artifact,
        built: built.built,
        identity,
    })
}
