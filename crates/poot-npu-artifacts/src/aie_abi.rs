//! Why an AIE region cannot be linked: the typed mismatches between a region's tile bodies and its link recipe,
//! and the faults in the recipe's wiring (POOT-664, POOT-1160). `poot-graph-plan` carries them as the typed cause
//! of a refused region; the recipe that raises them is `poot-npu`'s.

use poot_target::{AieTargetId, ArgAccess, ElementKind};

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

/// Why an AIE limit check refused: the limit that is unknown, or the need and the offer.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AieLimitRefusal {
    /// The artifact was linked for another target than the limits describe.
    #[error("the artifact was linked for {artifact:?}, the limits describe {caps:?}")]
    Target {
        /// The target the artifact was linked for.
        artifact: AieTargetId,
        /// The target the limits describe.
        caps: AieTargetId,
    },
    /// The target's limit is not measured, so the artifact's need of it cannot be admitted.
    #[error("the target's {limit} is not measured")]
    Unknown {
        /// Name of the unmeasured limit.
        limit: &'static str,
    },
    /// The artifact needs more of a measured resource than the target offers.
    #[error("{resource}: needed {needed}, the target offers {available}")]
    Exceeded {
        /// Name of the resource that is over its limit.
        resource: &'static str,
        /// What the artifact needs, in the resource's own unit.
        needed: u64,
        /// What the target offers, in the same unit.
        available: u64,
    },
}
