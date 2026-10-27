//! The link recipe of an AIE region: which compute tile runs which core, which object FIFO carries which
//! value, and the host side of the region as a shape ABI (POOT-664, POOT-1160).
//!
//! A recipe is data, not a script. It is validated at construction, before any tool runs: every FIFO has a
//! consumer, every host binding is the host end of exactly one FIFO of its own direction and size, and every
//! tile port is the end of exactly the FIFO the tile lists for it. The canonical text of a recipe is the part
//! of an artifact's identity that names its wiring.

use std::fmt::Write as _;

use poot_npu_artifacts::{AieAbiMismatch, AieShapeAbi, RecipeFault};
use poot_target::{AieTargetId, ArgAccess, ElementKind, StorageLayout};

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
        canonical_field(&mut out, "target", &self.target.canonical());
        canonical_field(&mut out, "invocations", &self.invocations.to_string());
        for (index, tile) in self.tiles.iter().enumerate() {
            canonical_field(
                &mut out,
                "tile",
                &format!("index={index};ports={}", join(tile.ports.iter())),
            );
        }
        for (index, fifo) in self.fifos.iter().enumerate() {
            canonical_field(
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

/// Append one canonical `tag[len]=value` record to `out`: the length-tagged form every identity text is
/// built from, so two different field sequences cannot render the same text.
pub(crate) fn canonical_field(out: &mut String, tag: &str, value: &str) {
    let _ = writeln!(out, "{tag}[{}]={value}", value.len());
}

/// The shape ABI in canonical text: ordered, with every dimension (equal element counts under different
/// dimensions are different ABIs), the full storage record, access, proved access range and the arena
/// placement the host and the link driver both address.
fn abi_canonical(abi: &AieShapeAbi) -> String {
    let mut out = String::new();
    for (index, arena) in abi.arenas().iter().enumerate() {
        canonical_field(
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
        canonical_field(
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

/// Check the recipe's host side is dense row-major storage, the only layout the sequential schedule moves.
fn check_host_layout(abi: &AieShapeAbi) -> Result<(), AieAbiMismatch> {
    for (index, binding) in abi.bindings().iter().enumerate() {
        if binding.storage().layout() != StorageLayout::Dense {
            return Err(AieAbiMismatch::Layout { index });
        }
    }
    Ok(())
}

impl AieLinkRecipe {
    /// Check that tile `tile`'s core and the recipe describe one core: its buffer parameters `args` are the
    /// tile's ports in count, element kind and which ones the tile writes. A core that can trap is refused (the
    /// ABI binds no error word).
    ///
    /// # Errors
    ///
    /// [`AieAbiMismatch`] naming the first disagreement.
    pub fn check_tile(
        &self,
        tile: usize,
        has_trap: bool,
        args: &[(ElementKind, ArgAccess)],
    ) -> Result<(), AieAbiMismatch> {
        if has_trap {
            return Err(AieAbiMismatch::Trap { tile });
        }
        let ports = &self.tiles[tile].ports;
        if args.len() != ports.len() {
            return Err(AieAbiMismatch::Count {
                tile,
                body: args.len(),
                recipe: ports.len(),
            });
        }
        for (port, (&(element, access), &fifo)) in args.iter().zip(ports).enumerate() {
            let fifo = &self.fifos[fifo];
            if element != fifo.element {
                return Err(AieAbiMismatch::Element {
                    tile,
                    port,
                    body: element,
                    fifo: fifo.element,
                });
            }
            let writes = fifo.producer == FifoEnd::Tile { tile, port };
            if (access == ArgAccess::Write) != writes {
                return Err(AieAbiMismatch::Access {
                    tile,
                    port,
                    body_writes: access == ArgAccess::Write,
                    fifo_writes: writes,
                });
            }
        }
        Ok(())
    }
}
