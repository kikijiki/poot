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

/// A tile's body and the recipe do not describe the same region, or the recipe cannot be linked.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AieAbiMismatch {
    /// Body parameter `index` of a tile is not a flat buffer slice of a runtime element type.
    #[error(
        "tile {tile}: body parameter {index} is not a flat buffer slice of a runtime element type"
    )]
    Parameter {
        /// The tile whose body is at fault.
        tile: usize,
        /// Position of the offending parameter in the body's signature.
        index: usize,
    },
    /// A tile's body and the recipe disagree on how many buffers the tile takes.
    #[error("tile {tile}: the body takes {body} buffers and the recipe wires {recipe}")]
    Count {
        /// The tile whose body is at fault.
        tile: usize,
        /// Buffer parameters the body declares.
        body: usize,
        /// Ports the recipe wires for the tile.
        recipe: usize,
    },
    /// A tile port has one element kind in the body and another in its FIFO.
    #[error("tile {tile} port {port}: the body's element is {body}, the FIFO's is {fifo}")]
    Element {
        /// The tile whose body is at fault.
        tile: usize,
        /// Port position, shared by the body parameter list and the tile's ports.
        port: usize,
        /// Element kind the body's parameter declares.
        body: ElementKind,
        /// Element kind the FIFO carries.
        fifo: ElementKind,
    },
    /// A tile port is written by one side and read-only on the other.
    #[error(
        "tile {tile} port {port}: the body writes it = {body_writes}, the recipe says {fifo_writes}"
    )]
    Access {
        /// The tile whose body is at fault.
        tile: usize,
        /// Port position, shared by the body parameter list and the tile's ports.
        port: usize,
        /// Whether the body writes the buffer.
        body_writes: bool,
        /// Whether the recipe has the tile produce the FIFO.
        fifo_writes: bool,
    },
    /// A tile's body contains a trap; the AIE ABI binds no error word to report it through.
    #[error("tile {tile}: the body can trap, and the AIE ABI has no error word")]
    Trap {
        /// The tile whose body is at fault.
        tile: usize,
    },
    /// A binding's storage is not dense row-major, the only layout the sequential schedule moves.
    #[error("a binding is not dense row-major storage")]
    Layout {
        /// Position of the offending binding.
        index: usize,
    },
    /// The summed tile-local FIFO bytes of a tile overflow `u64`.
    #[error("the recipe's tile-local byte total overflows")]
    TileLocalOverflow,
    /// The recipe's wiring is not a region: see [`RecipeFault`].
    #[error("the link recipe is malformed: {0}")]
    Recipe(#[from] RecipeFault),
    /// The tile bodies handed to the compiler are not one per recipe tile.
    #[error("the recipe has {tiles} tiles and {bodies} bodies were given")]
    BodyCount {
        /// Tiles the recipe places.
        tiles: usize,
        /// Bodies the caller supplied.
        bodies: usize,
    },
}

/// Why a region's wiring is not a dataflow a link driver can build. Every fault is raised at recipe
/// construction, before any tool runs.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RecipeFault {
    /// The recipe places no tile.
    #[error("the recipe places no tile")]
    NoTiles,
    /// A FIFO has no consumer: its values would be produced and never read.
    #[error("FIFO {fifo} has no consumer")]
    NoConsumer {
        /// The FIFO.
        fifo: usize,
    },
    /// A FIFO is moved with a zero depth or zero elements.
    #[error("FIFO {fifo} has a zero depth or zero elements")]
    Empty {
        /// The FIFO.
        fifo: usize,
    },
    /// A FIFO names a host binding the shape ABI does not have.
    #[error("FIFO {fifo} names host binding {binding}; the shape ABI has {bindings}")]
    UnknownBinding {
        /// The FIFO.
        fifo: usize,
        /// The binding it names.
        binding: usize,
        /// Bindings the shape ABI declares.
        bindings: usize,
    },
    /// A FIFO's host end contradicts the binding's access: the host fills a read binding and drains a
    /// written one.
    #[error(
        "FIFO {fifo}: host binding {binding} is {access:?} but the FIFO's host end is the other side"
    )]
    HostDirection {
        /// The FIFO.
        fifo: usize,
        /// The binding it names.
        binding: usize,
        /// The binding's access.
        access: ArgAccess,
    },
    /// A FIFO carries other elements than its host binding holds.
    #[error(
        "FIFO {fifo} moves {elements} {element:?} elements; host binding {binding} holds others"
    )]
    HostElements {
        /// The FIFO.
        fifo: usize,
        /// The binding it names.
        binding: usize,
        /// Elements the FIFO moves.
        elements: u64,
        /// The element kind the FIFO moves.
        element: ElementKind,
    },
    /// A host binding is not the host end of exactly one FIFO.
    #[error("host binding {binding} is the host end of {fifos} FIFOs, not one")]
    BindingStreams {
        /// The binding.
        binding: usize,
        /// FIFOs that name it.
        fifos: usize,
    },
    /// A FIFO end names a tile or port the recipe does not have.
    #[error("FIFO {fifo} names tile {tile} port {port}, which the recipe does not have")]
    UnknownPort {
        /// The FIFO.
        fifo: usize,
        /// The tile it names.
        tile: usize,
        /// The port it names.
        port: usize,
    },
    /// A tile port is not the end of exactly the FIFO the tile lists for it.
    #[error("tile {tile} port {port} is wired by {ends} FIFO ends matching its FIFO, not one")]
    PortWiring {
        /// The tile.
        tile: usize,
        /// The port.
        port: usize,
        /// FIFO ends that name this port as the FIFO the tile lists.
        ends: usize,
    },
    /// A tile lists a FIFO the recipe does not have.
    #[error("tile {tile} port {port} lists FIFO {fifo}, which the recipe does not have")]
    UnknownFifo {
        /// The tile.
        tile: usize,
        /// The port.
        port: usize,
        /// The FIFO it lists.
        fifo: usize,
    },
    /// A FIFO's producer and consumer are both the host: it would move host data to host memory.
    #[error("FIFO {fifo} runs from the host to the host")]
    HostToHost {
        /// The FIFO.
        fifo: usize,
    },
}

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

/// One end of an object FIFO: the host, or one port of one tile's core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FifoEnd {
    /// The host side. The shape ABI's binding `binding` is filled into the FIFO (a read binding) or drained
    /// out of it (a written one).
    Host {
        /// Index of the host binding in the shape ABI.
        binding: usize,
    },
    /// Port `port` of tile `tile`: body parameter `port` of that tile's core.
    Tile {
        /// Index of the tile in the recipe.
        tile: usize,
        /// Index of the port in the tile's parameter list.
        port: usize,
    },
}

/// One object FIFO of a link recipe: one value of the region, moved from its producer to its consumers.
/// A value the region reads has the host as its producer; a value the region returns has the host among its
/// consumers; a value only the region's own tiles read has no host end and never reaches host memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieFifo {
    /// The one end that writes the value.
    pub producer: FifoEnd,
    /// The ends that read it: every consumer receives every element.
    pub consumers: Vec<FifoEnd>,
    /// Buffers the FIFO holds in flight at each tile it touches.
    pub depth: u32,
    /// Elements per FIFO buffer (a whole tensor, in the sequential schedule).
    pub elements: u64,
    /// Element kind of the buffers the FIFO carries.
    pub element: ElementKind,
}

/// One tile of a link recipe: the core placed on it runs one body once per invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieTile {
    /// For each parameter of the tile's body, in parameter order, the FIFO that parameter is wired to.
    pub ports: Vec<usize>,
}

/// The typed link recipe IRON links a region with: which body runs on each compute tile, the FIFO each of
/// its parameters is wired to, and the host binding of every FIFO that reaches host memory. Data, not a
/// script. Tile order is placement order and part of the artifact's identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AieLinkRecipe {
    target: AieTargetId,
    abi: AieShapeAbi,
    tiles: Vec<AieTile>,
    fifos: Vec<AieFifo>,
    /// One finite invocation: every core runs its loop once over the whole tensors.
    invocations: u64,
    /// Tile-local bytes the busiest tile reserves for its FIFO buffers at their depth, checked once at
    /// construction.
    tile_local_bytes: u64,
}

impl AieLinkRecipe {
    /// IRON's default `ObjectFifo` depth.
    pub const FIFO_DEPTH: u32 = 2;

    /// The recipe placing `tiles` and wiring them with `fifos`, with `abi` the host side. Every FIFO must
    /// have a consumer, every host binding must be the host end of exactly one FIFO of its own direction
    /// and size, and every tile port must be the end of exactly the FIFO the tile lists for it.
    pub fn new(
        target: AieTargetId,
        abi: AieShapeAbi,
        tiles: Vec<AieTile>,
        fifos: Vec<AieFifo>,
    ) -> Result<Self, AieAbiMismatch> {
        if tiles.is_empty() {
            return Err(RecipeFault::NoTiles.into());
        }
        check_host_layout(&abi)?;
        let mut host_ends = vec![0usize; abi.bindings().len()];
        let mut claimed: Vec<Vec<usize>> =
            tiles.iter().map(|tile| vec![0; tile.ports.len()]).collect();
        for (index, tile) in tiles.iter().enumerate() {
            for (port, &fifo) in tile.ports.iter().enumerate() {
                if fifo >= fifos.len() {
                    return Err(RecipeFault::UnknownFifo {
                        tile: index,
                        port,
                        fifo,
                    }
                    .into());
                }
            }
        }
        for (fifo_index, fifo) in fifos.iter().enumerate() {
            if fifo.depth == 0 || fifo.elements == 0 {
                return Err(RecipeFault::Empty { fifo: fifo_index }.into());
            }
            if fifo.consumers.is_empty() {
                return Err(RecipeFault::NoConsumer { fifo: fifo_index }.into());
            }
            let ends = std::iter::once((fifo.producer, true))
                .chain(fifo.consumers.iter().map(|&end| (end, false)));
            for (end, produces) in ends {
                match end {
                    FifoEnd::Host { binding } => {
                        let Some(declared) = abi.bindings().get(binding) else {
                            return Err(RecipeFault::UnknownBinding {
                                fifo: fifo_index,
                                binding,
                                bindings: abi.bindings().len(),
                            }
                            .into());
                        };
                        // The host fills what the region reads and drains what it writes.
                        let host_fills = produces;
                        if host_fills != (declared.access() == ArgAccess::Read) {
                            return Err(RecipeFault::HostDirection {
                                fifo: fifo_index,
                                binding,
                                access: declared.access(),
                            }
                            .into());
                        }
                        if declared.elements() != fifo.elements
                            || declared.storage().element() != fifo.element
                        {
                            return Err(RecipeFault::HostElements {
                                fifo: fifo_index,
                                binding,
                                elements: fifo.elements,
                                element: fifo.element,
                            }
                            .into());
                        }
                        host_ends[binding] += 1;
                    }
                    FifoEnd::Tile { tile, port } => {
                        let listed = tiles.get(tile).and_then(|t| t.ports.get(port));
                        if listed.is_none() {
                            return Err(RecipeFault::UnknownPort {
                                fifo: fifo_index,
                                tile,
                                port,
                            }
                            .into());
                        }
                        if listed == Some(&fifo_index) {
                            claimed[tile][port] += 1;
                        }
                    }
                }
            }
            if matches!(fifo.producer, FifoEnd::Host { .. })
                && fifo
                    .consumers
                    .iter()
                    .any(|end| matches!(end, FifoEnd::Host { .. }))
            {
                return Err(RecipeFault::HostToHost { fifo: fifo_index }.into());
            }
        }
        for (binding, &fifos) in host_ends.iter().enumerate() {
            if fifos != 1 {
                return Err(RecipeFault::BindingStreams { binding, fifos }.into());
            }
        }
        for (tile, ports) in claimed.iter().enumerate() {
            for (port, &ends) in ports.iter().enumerate() {
                if ends != 1 {
                    return Err(RecipeFault::PortWiring { tile, port, ends }.into());
                }
            }
        }
        let mut tile_local_bytes = 0u64;
        for tile in &tiles {
            let reserved = tile.ports.iter().try_fold(0u64, |total, &fifo| {
                let fifo = &fifos[fifo];
                let one = fifo
                    .elements
                    .checked_mul(fifo.element.byte_width() as u64)?;
                total.checked_add(one.checked_mul(u64::from(fifo.depth))?)
            });
            tile_local_bytes =
                tile_local_bytes.max(reserved.ok_or(AieAbiMismatch::TileLocalOverflow)?);
        }
        Ok(Self {
            target,
            abi,
            tiles,
            fifos,
            invocations: 1,
            tile_local_bytes,
        })
    }

    /// The target this recipe links for.
    pub fn target(&self) -> AieTargetId {
        self.target
    }

    /// The tiles, in placement order.
    pub fn tiles(&self) -> &[AieTile] {
        &self.tiles
    }

    /// The FIFOs of the region, one per value.
    pub fn fifos(&self) -> &[AieFifo] {
        &self.fifos
    }

    /// The shape ABI the recipe was built for: the region's host side.
    pub fn abi(&self) -> &AieShapeAbi {
        &self.abi
    }

    /// How many times each core runs its loop (one, in the sequential schedule).
    pub fn invocations(&self) -> u64 {
        self.invocations
    }

    /// Tile-local bytes the busiest tile reserves for its FIFO buffers at their depth (no overflow:
    /// construction refused one).
    pub fn tile_local_bytes(&self) -> u64 {
        self.tile_local_bytes
    }

    /// How many streams tile `tile` reads and writes: the DMA channels its core uses.
    pub fn tile_streams(&self, tile: usize) -> (u32, u32) {
        let mut reads = 0;
        let mut writes = 0;
        for (port, &fifo) in self.tiles[tile].ports.iter().enumerate() {
            let end = FifoEnd::Tile { tile, port };
            if self.fifos[fifo].producer == end {
                writes += 1;
            } else {
                reads += 1;
            }
        }
        (reads, writes)
    }

    /// How many streams the host opens into the array and out of it: the shim channels the region uses.
    pub fn host_streams(&self) -> (u32, u32) {
        let reads = self
            .fifos
            .iter()
            .filter(|fifo| matches!(fifo.producer, FifoEnd::Host { .. }))
            .count();
        let writes = self
            .fifos
            .iter()
            .filter(|fifo| {
                fifo.consumers
                    .iter()
                    .any(|end| matches!(end, FifoEnd::Host { .. }))
            })
            .count();
        (
            u32::try_from(reads).unwrap_or(u32::MAX),
            u32::try_from(writes).unwrap_or(u32::MAX),
        )
    }

    /// The recipe's canonical text: versioned, tagged and length-prefixed, and the file the driver reads
    /// (with the driver's `object` lines appended).
    pub fn canonical(&self) -> String {
        let mut out = String::from("poot-aie-recipe-v2\n");
        field(&mut out, "target", &self.target.canonical());
        field(&mut out, "invocations", &self.invocations.to_string());
        for (index, tile) in self.tiles.iter().enumerate() {
            field(
                &mut out,
                "tile",
                &format!("index={index};ports={}", join(tile.ports.iter())),
            );
        }
        for (index, fifo) in self.fifos.iter().enumerate() {
            field(
                &mut out,
                "fifo",
                &format!(
                    "index={index};producer={};consumers={};depth={};elems={};elem={:?}",
                    end_canonical(fifo.producer),
                    join(fifo.consumers.iter().map(|&end| end_canonical(end))),
                    fifo.depth,
                    fifo.elements,
                    fifo.element
                ),
            );
        }
        out.push_str(&abi_canonical(&self.abi));
        out
    }
}

/// `items` joined by commas.
fn join<T: std::fmt::Display>(items: impl Iterator<Item = T>) -> String {
    items
        .map(|item| item.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn end_canonical(end: FifoEnd) -> String {
    match end {
        FifoEnd::Host { binding } => format!("host.{binding}"),
        FifoEnd::Tile { tile, port } => format!("tile.{tile}.{port}"),
    }
}

fn field(out: &mut String, tag: &str, value: &str) {
    let _ = writeln!(out, "{tag}[{}]={value}", value.len());
}

/// The shape ABI in canonical text: ordered, with every dimension (equal element counts under different
/// dimensions are different ABIs), the full storage record, access, proved access range and the arena
/// placement the host and the link driver both address.
fn abi_canonical(abi: &AieShapeAbi) -> String {
    let mut out = String::new();
    for (index, arena) in abi.arenas().iter().enumerate() {
        field(
            &mut out,
            "arena",
            &format!(
                "index={index};access={:?};elem={:?};words={}",
                arena.access(),
                arena.element(),
                arena.words()
            ),
        );
    }
    for (index, binding) in abi.bindings().iter().enumerate() {
        let storage = binding.storage();
        field(
            &mut out,
            "binding",
            &format!(
                "index={index};element={:?};dtype={:?};layout={:?};access={:?};dims={:?};accessed={};arena={};offset={}",
                storage.element(),
                storage.dtype(),
                storage.layout(),
                binding.access(),
                binding.dims(),
                binding.accessed_elements(),
                binding.arena(),
                binding.offset()
            ),
        );
    }
    out
}

/// Check that tile `tile`'s `body` and the recipe describe one core: its buffer parameters are the tile's
/// ports in count, element kind and which ones the tile writes. A body that can trap is refused (the ABI
/// binds no error word), and so is a host binding that is not dense row-major.
pub(crate) fn check_tile_body(
    recipe: &AieLinkRecipe,
    tile: usize,
    body: &Body,
) -> Result<(), AieAbiMismatch> {
    if body.has_trap() {
        return Err(AieAbiMismatch::Trap { tile });
    }
    let schema = try_arg_schema(body).map_err(|index| AieAbiMismatch::Parameter { tile, index })?;
    let ports = &recipe.tiles[tile].ports;
    if schema.len() != ports.len() {
        return Err(AieAbiMismatch::Count {
            tile,
            body: schema.len(),
            recipe: ports.len(),
        });
    }
    for (port, (arg, &fifo)) in schema.iter().zip(ports).enumerate() {
        let fifo = &recipe.fifos[fifo];
        if arg.element() != fifo.element {
            return Err(AieAbiMismatch::Element {
                tile,
                port,
                body: arg.element(),
                fifo: fifo.element,
            });
        }
        let writes = fifo.producer == FifoEnd::Tile { tile, port };
        if (arg.access() == ArgAccess::Write) != writes {
            return Err(AieAbiMismatch::Access {
                tile,
                port,
                body_writes: arg.access() == ArgAccess::Write,
                fifo_writes: writes,
            });
        }
    }
    Ok(())
}

/// Check the recipe's host side is dense row-major storage, the only layout the sequential schedule moves.
fn check_host_layout(abi: &AieShapeAbi) -> Result<(), AieAbiMismatch> {
    for (index, binding) in abi.bindings().iter().enumerate() {
        if binding.storage().layout() != StorageLayout::Dense {
            return Err(AieAbiMismatch::Layout { index });
        }
    }
    Ok(())
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
    /// The identity of linking `bodies` (one per tile, in tile order) with `recipe` using `tools`. Pure
    /// over its inputs: it reads no environment and starts no process, so a test names exactly the
    /// dimension it varies. The core text names the distinct core objects and how they are built; the
    /// link text places them: which body runs on which tile and how the tiles are wired.
    pub fn new(bodies: &[&Body], recipe: &AieLinkRecipe, tools: &AieTools) -> Self {
        let target = recipe.target;
        let mut distinct: Vec<(u64, String)> = bodies
            .iter()
            .enumerate()
            .map(|(tile, body)| (body_fingerprint(body), core_symbol(recipe, tile, body)))
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        let mut core = String::from("poot-aie-core-v2\n");
        for (fingerprint, symbol) in distinct {
            field(&mut core, "body", &format!("{fingerprint:016x}"));
            field(&mut core, "symbol", &symbol);
        }
        field(&mut core, "isa", target.isa().march());
        field(&mut core, "numerics", NUMERICAL_CONTRACT);
        field(&mut core, "core-abi", &CORE_ABI_REVISION.to_string());
        field(&mut core, "codegen", codegen_src_hash());
        field(&mut core, "flags", &Target::AieCore.llc_args().join(" "));
        field(&mut core, "peano", &tools.peano.canonical());
        let mut link = String::from("poot-aie-link-v2\n");
        field(
            &mut link,
            "core",
            &format!("{:016x}", poot_runtime_common::fnv1a_bytes(core.as_bytes())),
        );
        for (tile, body) in bodies.iter().enumerate() {
            field(
                &mut link,
                "placed",
                &format!("tile={tile};body={:016x}", body_fingerprint(body)),
            );
        }
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
        aie_entry_key(self.core_digest(), self.link_digest())
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
        check_tile_body(recipe, tile, body)?;
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
        field(
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
        Ok::<_, AieCompileError>(link_object(&plan, &tools, target, scratch.path())?.to_bytes())
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
