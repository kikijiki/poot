//! AIE compilation: the sequential `Body`s of one region to Peano core objects, a typed link recipe that
//! places them on compute tiles, and one loadable xclbin/instruction artifact (POOT-664, spec 1002,
//! POOT-1160).
//!
//! ```text
//! Bodies + AieLinkRecipe --check_tile--> Peano (one object per distinct Body) --IRON--> AieLinkedArtifact
//! ```
//!
//! A region is a set of planned equations that run in one hardware context. Each equation's `Body` is a
//! tile program: one compute tile runs it once. Every value the region touches is one object FIFO: a value
//! the region reads from the host streams in, a value the region returns streams out, and a value that only
//! other equations of the region read moves tile to tile and never reaches host memory. The recipe says
//! which tile reads and writes which FIFO; it is data, not a script. The host side of the artifact is the
//! recipe's [`AieShapeAbi`]: a handful of host arenas, each region value at a fixed offset inside one.
//!
//! The core objects are intermediates, never runnable modules. The linked artifact is the only thing
//! [`crate::aie_kernel_handle`] wraps, and the runtime (POOT-641) only loads it: it neither runs IRON nor
//! recognizes graphs.
//!
//! # Identity and publication
//!
//! Two canonical, length-tagged texts name an artifact ([`AieIdentity`]): the core objects' (every tile's
//! whole verified Body, target, numerical contract, core ABI revision, codegen source and the Peano build)
//! and the link's (the core digest of every tile in tile order, the recipe with its wiring, every binding's
//! storage, dimensions, access and arena offset, the platform, the IRON build, flags, instruction format
//! and finalization policy). A shape, layout, wiring, placement order or tool change names a different
//! cache entry. Both tools are resolved and checked before any scratch exists, and the bundle is published
//! as one cache entry after both link outputs were read and validated, so a failure at any stage leaves no
//! entry.
//!
//! # The IRON link driver protocol
//!
//! `$POOT_AIE_IRON` names an executable (the compiler does not embed Python):
//!
//! ```text
//! POOT_AIE_IRON --version     first stdout line: the driver and IRON build identity
//! POOT_AIE_IRON --platforms   the platform names it can link for, one per line
//! POOT_AIE_IRON link --platform <name> --objects <dir> --recipe <recipe.txt> \
//!                    --xclbin <out> --insts <out>
//! ```
//!
//! The recipe file is [`AieLinkRecipe::canonical`] followed by one `object` line per tile naming that
//! tile's core object file inside `<dir>` and its entry symbol. Success is exit status 0 with both outputs
//! written.
//!
//! The one implementation is `scripts/poot-aie-link.py` (POOT-1118, POOT-1160): it builds the IRON program
//! the recipe describes, one worker per tile and one object FIFO per value, and links it with mlir-aie's
//! `aiecc`. The flake's `npu` devshell (`nix develop .#npu`) exports `POOT_AIE_IRON` as the wrapper of it
//! and `POOT_AIE_LLC` as the pinned Peano `llc`; anywhere else the variables are unset and the compile is
//! the typed [`AieCompileError::ToolchainMissing`]. The model-free tests exercise this orchestration
//! against scripted drivers, which prove the protocol handling and typed failures, not machine code; the
//! ignored `a_real_peano_and_iron_toolchain_links_a_loadable_artifact` row runs the real pair.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{LazyLock, Mutex};

use crate::aie_bundle::{decode_bundle, encode_bundle};
use crate::aie_recipe::{AieLinkRecipe, canonical_field};
use poot_codegen::{
    ArtifactTooLarge, CODEGEN_SRC_HASH, CompileError, EmitError, KernelCache, KernelId, KernelKey,
    Target, Toolchain, compile_with, try_arg_schema,
};
use poot_kernel_ir::Body;
use poot_npu_artifacts::{AieAbiMismatch, AieArtifactError, AieLinkedArtifact};
use poot_target::AieTargetId;

/// The 64-bit content fingerprint of a kernel `Body` (a hash of its `Debug` form): what names a body in an AIE
/// identity text.
fn body_fingerprint(body: &Body) -> u64 {
    poot_runtime_common::fnv1a_bytes(format!("{body:?}").as_bytes())
}

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

/// A tile's body and the recipe do not describe the same region, or the recipe cannot be linked.
/// Why an AIE compilation produced no artifact. Every variant leaves nothing published.
#[derive(Debug, thiserror::Error)]
pub enum AieCompileError {
    /// The kernel body could not be emitted for the Peano stage.
    #[error(transparent)]
    Emit(#[from] EmitError),
    /// The body and shape ABI do not describe one kernel, or the ABI cannot be linked.
    #[error(transparent)]
    Abi(#[from] AieAbiMismatch),
    /// A required tool is not configured (`variable` unset) or not found/runnable at its configured path.
    #[error("the {stage} toolchain is unavailable: ${variable} is unset or names no runnable tool")]
    ToolchainMissing {
        /// Which stage's tool is missing.
        stage: AieStage,
        /// Name of the environment variable that should name the tool.
        variable: &'static str,
    },
    /// The tool runs but does not support `target` (no `aie2p` backend, or no such platform).
    #[error("the {stage} toolchain does not support {target:?}")]
    ToolchainUnsupportedTarget {
        /// Which stage's tool lacks support.
        stage: AieStage,
        /// The AIE target the tool was asked to build for.
        target: AieTargetId,
    },
    /// A toolchain stage ran and exited unsuccessfully.
    #[error("{stage} failed (status {status}):\n{stderr}")]
    Tool {
        /// Which stage failed.
        stage: AieStage,
        /// The tool's exit status as the process library renders it (an exit code or the terminating signal).
        status: String,
        /// The tool's captured standard error.
        stderr: String,
    },
    /// The Peano output is not an ELF object defining the kernel symbol.
    #[error("the core object is not an ELF carrying the symbol `{symbol}`")]
    CoreObject {
        /// The symbol the recipe names and the object was expected to define.
        symbol: String,
    },
    /// The IRON driver exited 0 but left one of its two output files unwritten.
    #[error("the IRON driver reported success but wrote no {output}")]
    LinkOutput {
        /// Which output is missing (the xclbin or the instruction stream).
        output: &'static str,
    },
    /// The linked bundle failed the artifact validator.
    #[error(transparent)]
    Artifact(#[from] AieArtifactError),
    /// The linked bundle exceeds the cache's artifact size limit.
    #[error(transparent)]
    TooLarge(#[from] ArtifactTooLarge),
    /// A cache entry under this identity describes a different kernel than the request (a stale or
    /// colliding entry): it is refused, never loaded.
    #[error("the cached artifact's {what} differs from the request")]
    StaleEntry {
        /// Which recorded property of the entry differs from the request.
        what: &'static str,
    },
    /// Reading, writing or publishing a file in the cache or scratch directory failed.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// A resolved tool: where it runs from, the first line of its `--version` and a hash of its bytes. The
/// path and version line alone cannot see a binary replaced under the same name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieToolIdentity {
    /// Filesystem path the tool was resolved to.
    pub path: String,
    /// First line of the tool's `--version` output.
    pub version: String,
    /// FNV-1a hash of the tool binary's bytes.
    pub content: u64,
}

/// The two tools an AIE compilation needs, resolved and checked against the target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieTools {
    /// The Peano `llc` that builds the core object.
    pub peano: AieToolIdentity,
    /// The IRON link driver that produces the xclbin and instruction stream.
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
        toolchain.aie_llc(),
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
        toolchain.aie_iron(),
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
    /// The identity of linking `bodies` (one per tile, in tile order) with `recipe` using `tools`. Pure
    /// over its inputs: it reads no environment and starts no process, so a test names exactly the
    /// dimension it varies. The core text names the distinct core objects and how they are built; the
    /// link text places them: which body runs on which tile and how the tiles are wired.
    pub fn new(bodies: &[&Body], recipe: &AieLinkRecipe, tools: &AieTools) -> Self {
        let target = recipe.target();
        let mut distinct: Vec<(u64, String)> = bodies
            .iter()
            .enumerate()
            .map(|(tile, body)| (body_fingerprint(body), core_symbol(recipe, tile, body)))
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut core = String::from("poot-aie-core-v2\n");
        for (fingerprint, symbol) in distinct {
            canonical_field(&mut core, "body", &format!("{fingerprint:016x}"));
            canonical_field(&mut core, "symbol", &symbol);
        }
        canonical_field(&mut core, "isa", target.isa().march());
        canonical_field(&mut core, "numerics", NUMERICAL_CONTRACT);
        canonical_field(&mut core, "core-abi", &CORE_ABI_REVISION.to_string());
        canonical_field(&mut core, "codegen", CODEGEN_SRC_HASH);
        canonical_field(&mut core, "flags", &Target::AieCore.llc_args().join(" "));
        canonical_field(&mut core, "peano", &tools.peano.canonical());
        let mut link = String::from("poot-aie-link-v2\n");
        canonical_field(
            &mut link,
            "core",
            &format!("{:016x}", poot_runtime_common::fnv1a_bytes(core.as_bytes())),
        );
        for (tile, body) in bodies.iter().enumerate() {
            canonical_field(
                &mut link,
                "placed",
                &format!("tile={tile};body={:016x}", body_fingerprint(body)),
            );
        }
        link.push_str(&recipe.canonical());
        canonical_field(&mut link, "platform", target.platform().name());
        canonical_field(&mut link, "abi", &target.abi_revision().to_string());
        canonical_field(&mut link, "iron", &tools.iron.canonical());
        canonical_field(&mut link, "flags", LINK_FLAGS);
        canonical_field(&mut link, "insts", INSTRUCTION_FORMAT);
        canonical_field(
            &mut link,
            "finalization",
            &FINALIZATION_POLICY_REVISION.to_string(),
        );
        Self { core, link }
    }

    /// FNV-1a digest of the core object's canonical identity text.
    pub fn core_digest(&self) -> u64 {
        poot_runtime_common::fnv1a_bytes(self.core.as_bytes())
    }

    /// FNV-1a digest of the link's canonical identity text.
    pub fn link_digest(&self) -> u64 {
        poot_runtime_common::fnv1a_bytes(self.link.as_bytes())
    }

    /// The cache entry this identity names.
    pub fn cache_key(&self) -> KernelKey {
        let (core, link) = (self.core_digest(), self.link_digest());
        let mut id = [0; 16];
        id[..8].copy_from_slice(&core.to_be_bytes());
        id[8..].copy_from_slice(&link.to_be_bytes());
        KernelKey::artifact(format!("a{core:016x}{link:016x}.aie"), KernelId::from(id))
    }
}

impl AieToolIdentity {
    fn canonical(&self) -> String {
        format!("{}@{:016x}", self.version, self.content)
    }
}

/// The compiled core objects of a region with the typed recipe they will be linked with. Not runnable.
#[derive(Clone, Debug)]
pub(crate) struct AieObjectPlan {
    /// The distinct core objects: the file name the driver finds each under, its entry symbol and bytes.
    pub objects: Vec<AieObject>,
    /// For each tile of the recipe, the index of its object in `objects`.
    pub tile_objects: Vec<usize>,
    pub recipe: AieLinkRecipe,
}

/// One compiled core object.
#[derive(Clone, Debug)]
pub(crate) struct AieObject {
    pub file: String,
    pub symbol: String,
    pub bytes: Vec<u8>,
}

/// Compile every distinct body of `bodies` (one per tile of `recipe`) to an AIE core object and pair them
/// with the recipe. `scratch` is a private directory the caller owns and removes.
pub(crate) fn build_object_plan(
    bodies: &[&Body],
    recipe: &AieLinkRecipe,
    toolchain: &Toolchain,
    scratch: &Path,
) -> Result<AieObjectPlan, AieCompileError> {
    check_bodies(bodies, recipe)?;
    let mut objects: Vec<AieObject> = Vec::new();
    let mut symbols: Vec<String> = Vec::new();
    let mut tile_objects = Vec::with_capacity(bodies.len());
    for (tile, body) in bodies.iter().enumerate() {
        let symbol = core_symbol(recipe, tile, body);
        if let Some(index) = symbols.iter().position(|known| *known == symbol) {
            tile_objects.push(index);
            continue;
        }
        // A body with no name has no symbol for a core object to define.
        if body.name.is_empty() {
            return Err(AieCompileError::CoreObject {
                symbol: String::new(),
            });
        }
        let file = format!("{symbol}.o");
        let out = scratch.join(&file);
        let mut core = (*body).clone();
        core.name = symbol.clone();
        compile_with(&core, Target::AieCore, &out, toolchain).map_err(|error| match error {
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
            CompileError::NoContraction(_)
            | CompileError::ArtifactTooLarge(_)
            | CompileError::NotAByteTarget(_) => {
                unreachable!("only SPIR-V post-processing and the byte-kernel cache raise this")
            }
        })?;
        let bytes = std::fs::read(&out)?;
        // A smoke check, not a symbol-table lookup: an ELF whose bytes name the core symbol.
        if !bytes.starts_with(b"\x7fELF")
            || !bytes
                .windows(core.name.len())
                .any(|window| window == core.name.as_bytes())
        {
            return Err(AieCompileError::CoreObject {
                symbol: core.name.clone(),
            });
        }
        tile_objects.push(objects.len());
        symbols.push(symbol);
        objects.push(AieObject {
            file,
            symbol: core.name.clone(),
            bytes,
        });
    }
    Ok(AieObjectPlan {
        objects,
        tile_objects,
        recipe: recipe.clone(),
    })
}

/// The entry symbol of the core object tile `tile` runs. Every core object of a region is linked into one
/// IRON module, where a kernel symbol is global and carries the argument types it was declared with. Two
/// distinct bodies of one generator share a name, and one body runs on tiles of different shapes (its lengths
/// are arguments), so the symbol names the body's fingerprint and the element counts of the tile's ports:
/// two tiles share an object exactly when they run the same body over the same shapes.
fn core_symbol(recipe: &AieLinkRecipe, tile: usize, body: &Body) -> String {
    let sizes: Vec<String> = recipe.tiles()[tile]
        .ports
        .iter()
        .map(|&fifo| recipe.fifos()[fifo].elements.to_string())
        .collect();
    format!(
        "{}_{:016x}_{}",
        body.name,
        body_fingerprint(body),
        sizes.join("_")
    )
}

/// One body per tile, each the core its tile's ports describe.
fn check_bodies(bodies: &[&Body], recipe: &AieLinkRecipe) -> Result<(), AieAbiMismatch> {
    if bodies.len() != recipe.tiles().len() {
        return Err(AieAbiMismatch::BodyCount {
            tiles: recipe.tiles().len(),
            bodies: bodies.len(),
        });
    }
    for (tile, body) in bodies.iter().enumerate() {
        let args = try_arg_schema(body)
            .map_err(|index| AieAbiMismatch::Parameter { tile, index })?
            .iter()
            .map(|arg| (arg.element(), arg.access()))
            .collect::<Vec<_>>();
        recipe.check_tile(tile, body.has_trap(), &args)?;
    }
    Ok(())
}

/// Link `plan` into a loadable artifact with the IRON driver. Both outputs are read and the bundle
/// validated before it is returned; a failure at any point returns an error and nothing else.
pub(crate) fn link_object(
    plan: &AieObjectPlan,
    tools: &AieTools,
    target: AieTargetId,
    scratch: &Path,
) -> Result<AieLinkedArtifact, AieCompileError> {
    let objects = scratch.join("objects");
    let recipe = scratch.join("recipe.txt");
    let xclbin = scratch.join("final.xclbin");
    let insts = scratch.join("insts.bin");
    std::fs::create_dir_all(&objects)?;
    for object in &plan.objects {
        std::fs::write(objects.join(&object.file), &object.bytes)?;
    }
    let mut text = plan.recipe.canonical();
    for (tile, &object) in plan.tile_objects.iter().enumerate() {
        let object = &plan.objects[object];
        canonical_field(
            &mut text,
            "object",
            &format!("tile={tile};file={};symbol={}", object.file, object.symbol),
        );
    }
    std::fs::write(&recipe, text)?;
    let output = Command::new(&tools.iron.path)
        .args(["link", "--platform", target.platform().name()])
        .arg("--objects")
        .arg(&objects)
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

/// A linked AIE artifact and whether this call built it.
#[derive(Debug)]
pub struct AieCompiled {
    /// The validated xclbin and instruction stream.
    pub artifact: AieLinkedArtifact,
    /// Whether this call ran the toolchain (`true`) or served the artifact from the cache (`false`).
    pub built: bool,
    /// The canonical core and link identities that name the cache entry.
    pub identity: AieIdentity,
}

/// The whole pipeline: resolve the tools, then serve the artifact for `(bodies, recipe)` from `cache`
/// or build it (core objects, recipe, IRON link), publishing the validated bundle as one entry. `bodies` are
/// one per tile of `recipe`, in tile order. Tool resolution comes first, so a missing or unsupported
/// toolchain fails before any scratch exists or any entry is written.
pub fn compile_aie(
    bodies: &[&Body],
    recipe: &AieLinkRecipe,
    toolchain: &Toolchain,
    cache: &KernelCache,
    max_artifact_bytes: NonZeroU64,
) -> Result<AieCompiled, AieCompileError> {
    let (abi, target) = (recipe.abi(), recipe.target());
    let tools = resolve_aie_tools(toolchain, target)?;
    check_bodies(bodies, recipe)?;
    let identity = AieIdentity::new(bodies, recipe, &tools);
    let key = identity.cache_key();
    let built = cache.load_or_build(&key, max_artifact_bytes, || {
        let scratch = cache.work_dir(&key)?;
        let plan = build_object_plan(bodies, recipe, toolchain, scratch.path())?;
        Ok::<_, AieCompileError>(encode_bundle(&link_object(
            &plan,
            &tools,
            target,
            scratch.path(),
        )?))
    })?;
    let artifact = decode_bundle(&built.bytes, target)?;
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
