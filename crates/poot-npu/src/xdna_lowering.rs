//! The AMD XDNA side of lowering: the shape ABI a serialized sequential schedule links against, the link
//! recipe of a region of planned equations, and every limit check on them (POOT-664, POOT-1160).
//!
//! The planner chose the schedule (the ordinary request builders under its one-core mode); what is here is
//! what only the NPU knows: how a plan's bindings become host arenas, how a run of equations becomes tiles
//! and object FIFOs, and whether either fits the part's measured limits. An unmeasured limit refuses.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::aie_recipe::{AieFifo, AieLinkRecipe, AieTile, FifoEnd};
use poot_graph_ir::{Graph, ValueId};
use poot_graph_plan::{
    AieAbiRefusal, Dispatch, KernelChoice, NpuTarget, Plan, Program, Region, Sink, Source, Target,
    value_ids,
};
use poot_kernel_ir::Body;
use poot_kernelgen as kg;
use poot_npu_artifacts::{AieBinding, AieLimitRefusal, AieLinkedArtifact, AieShapeAbi};
use poot_target::{AieCaps, AieTargetId, ArgAccess, BufferStorage, ElementKind, Queried};

fn within<T: Into<u64> + Copy>(
    limit: Queried<T>,
    resource: &'static str,
    needed: u64,
) -> Result<(), AieLimitRefusal> {
    match limit {
        Queried::Unknown => Err(AieLimitRefusal::Unknown { limit: resource }),
        Queried::Known(available) if needed > available.into() => Err(AieLimitRefusal::Exceeded {
            resource,
            needed,
            available: available.into(),
        }),
        Queried::Known(_) => Ok(()),
    }
}

/// The part of finalization a plan answers on its own: the host buffers its shape ABI needs against the
/// target's. The firmware binds a region's host side as arenas, so what is counted is the arenas the
/// bindings are laid out in, not the bindings. An unmeasured limit refuses.
pub(crate) fn check_shape_abi_limits(
    abi: &AieShapeAbi,
    caps: &AieCaps,
) -> Result<(), AieLimitRefusal> {
    within(
        caps.host_bindings,
        "host buffers",
        abi.arenas().len() as u64,
    )
}

/// A region's link recipe against everything the target bounds about one region, before anything is
/// linked: the compute tiles it places, the DMA channels each tile and the shim open, the tile-local bytes
/// of its busiest tile and the host buffers of its arenas. This is the one definition of "a region fits":
/// `package` closes a region where it stops being true and the loader refuses an artifact where it never
/// was. An unmeasured limit refuses.
pub(crate) fn validate_region_recipe(
    recipe: &AieLinkRecipe,
    caps: &AieCaps,
) -> Result<(), AieLimitRefusal> {
    let limits = &caps.region;
    within(
        limits.compute_tiles,
        "compute tiles",
        recipe.tiles().len() as u64,
    )?;
    for tile in 0..recipe.tiles().len() {
        let (reads, writes) = recipe.tile_streams(tile);
        within(
            limits.tile_input_channels,
            "tile input channels",
            u64::from(reads),
        )?;
        within(
            limits.tile_output_channels,
            "tile output channels",
            u64::from(writes),
        )?;
    }
    let (reads, writes) = recipe.host_streams();
    within(
        limits.host_input_streams,
        "host input streams",
        u64::from(reads),
    )?;
    within(
        limits.host_output_streams,
        "host output streams",
        u64::from(writes),
    )?;
    validate_recipe_tile_local(recipe.tile_local_bytes(), caps)?;
    check_shape_abi_limits(recipe.abi(), caps)
}

/// The tile-local bytes a link recipe reserves, against the target's limit, before anything is linked: the
/// recipe's FIFO buffers are fixed by the shape ABI, so an over-limit plan is refused here as a typed value
/// instead of as a toolchain failure. An unmeasured limit refuses.
fn validate_recipe_tile_local(reserved_bytes: u64, caps: &AieCaps) -> Result<(), AieLimitRefusal> {
    within(
        caps.tile_local_bytes,
        "link recipe tile-local bytes",
        reserved_bytes,
    )
}

/// The finalization only a linked artifact can answer, run by the loader before it wraps the artifact: the
/// artifact was linked for `caps`' target, and its tile-local reservation, host bindings and instruction
/// stream fit the target's limits. An unmeasured limit refuses.
pub(crate) fn validate_linked_artifact(
    artifact: &AieLinkedArtifact,
    caps: &AieCaps,
) -> Result<(), AieLimitRefusal> {
    if artifact.target() != caps.target {
        return Err(AieLimitRefusal::Target {
            artifact: artifact.target(),
            caps: caps.target,
        });
    }
    within(
        caps.tile_local_bytes,
        "tile-local bytes",
        artifact.tile_local_bytes(),
    )?;
    check_shape_abi_limits(artifact.abi(), caps)?;
    within(
        caps.instruction_bytes,
        "instruction bytes",
        artifact.instructions().len() as u64,
    )
}

/// The shape ABI of a planned sequential schedule: one binding per body parameter, in binding order,
/// each with the dimensions and access range the generator proved when it produced the body. The ABI is
/// read from the plan's own choice, so the body and the ABI come from the one generation the planner ran.
pub(crate) fn aie_shape_abi(choice: &KernelChoice) -> Result<AieShapeAbi, AieAbiRefusal> {
    let KernelChoice::Serialized { bindings, .. } = choice else {
        return Err(AieAbiRefusal::NotSerialized);
    };
    let bindings = bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            AieBinding::new(
                index,
                word_storage(binding.element),
                if binding.writes {
                    ArgAccess::Write
                } else {
                    ArgAccess::Read
                },
                &binding.dims,
                binding.accessed,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AieShapeAbi::new(bindings)?)
}

/// The canonical word buffer of an element kind an AIE binding may have (F32, else I32). The shape ABI
/// refuses any other storage when it builds the binding.
fn word_storage(element: ElementKind) -> BufferStorage {
    match element {
        ElementKind::F32 => BufferStorage::f32(),
        _ => BufferStorage::i32(),
    }
}

/// What the region's equations declare of one value: its dimensions, element kind and proved access.
type Declared<'a> = (&'a [u64], ElementKind, u64);

/// Record the dimensions, element kind and proved access `binding` declares for `value`. A second
/// declaration must have the same dimensions and element kind; the larger proved access wins.
fn note_dims<'a>(
    seen: &mut HashMap<ValueId, Declared<'a>>,
    value: ValueId,
    binding: &'a kg::SerializedBinding,
) -> Result<(), AieAbiRefusal> {
    match seen.entry(value) {
        Entry::Vacant(slot) => {
            slot.insert((&binding.dims, binding.element, binding.accessed));
        }
        Entry::Occupied(mut slot) => {
            if *slot.get().0 != *binding.dims || slot.get().1 != binding.element {
                return Err(AieAbiRefusal::ReaderDims { value });
            }
            slot.get_mut().2 = slot.get().2.max(binding.accessed);
        }
    }
    Ok(())
}

/// The link recipe of `region`: one tile per equation, in program order, and one FIFO per value. A value the
/// region reads from outside streams in from the host, a value it returns streams out to the host, and a
/// value only its own equations read moves from the tile that produces it to the tiles that read it. The
/// host side is the shape ABI of the values read and then the values returned, so the last binding is always
/// a written one and a dispatch binds them in that order.
///
/// Each tile's ports are its body's parameters: the equation's operands, then its output. The bodies and
/// the proved access ranges come from the same planner run as `plans`, so the recipe cannot disagree with
/// the bodies it is linked against; `poot-codegen` still checks every body against it.
pub(crate) fn aie_region_recipe(
    g: &Graph,
    plans: &[Plan],
    region: &Region,
    target: AieTargetId,
) -> Result<AieLinkRecipe, AieAbiRefusal> {
    /// One member's dispatch seen as a tile program.
    struct Member<'a> {
        out: ValueId,
        reads: Vec<ValueId>,
        bindings: &'a [kg::SerializedBinding],
    }
    let members = region
        .members()
        .map(|index| {
            let eqn = &g.eqns()[index];
            let not_one = AieAbiRefusal::NotOneDispatch { eqn: eqn.out };
            let plan = &plans[index];
            let dispatch = plan.single().ok_or(not_one.clone())?;
            if !matches!(plan, Plan::Dispatches(d) if d.scratch().is_empty())
                || dispatch.binding().write() != Sink::Out
            {
                return Err(not_one);
            }
            let operands = value_ids(eqn);
            let reads = dispatch
                .binding()
                .reads()
                .iter()
                .map(|source| match *source {
                    Source::Operand(operand) => operands.get(operand).copied(),
                    Source::Scratch(_) => None,
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(AieAbiRefusal::NotOneDispatch { eqn: eqn.out })?;
            let KernelChoice::Serialized { bindings, .. } = dispatch.choice() else {
                return Err(AieAbiRefusal::NotSerialized);
            };
            let ordered = bindings.len() == reads.len() + 1
                && bindings.iter().rev().skip(1).all(|b| !b.writes)
                && bindings.last().is_some_and(|b| b.writes);
            if !ordered {
                return Err(AieAbiRefusal::Bindings { eqn: eqn.out });
            }
            Ok(Member {
                out: eqn.out,
                reads,
                bindings,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The dimensions and proved access of every value the host sees: what the first equation that touches
    // it declares. Two readers must agree on the dimensions; the larger proved access wins.
    let mut seen: HashMap<ValueId, Declared<'_>> = HashMap::new();
    for member in &members {
        for (&value, binding) in member.reads.iter().zip(member.bindings) {
            note_dims(&mut seen, value, binding)?;
        }
        let write = member.bindings.last().expect("checked non-empty above");
        note_dims(&mut seen, member.out, write)?;
    }

    let host_binding = |index: usize, value: ValueId, access: ArgAccess| {
        let (dims, element, accessed) = seen[&value];
        AieBinding::new(index, word_storage(element), access, dims, accessed)
    };
    let mut host = Vec::new();
    for (index, &value) in region.reads().iter().enumerate() {
        host.push(host_binding(index, value, ArgAccess::Read)?);
    }
    for (index, &value) in region.writes().iter().enumerate() {
        host.push(host_binding(
            region.reads().len() + index,
            value,
            ArgAccess::Write,
        )?);
    }
    let abi = AieShapeAbi::new(host)?;

    // One FIFO per value: the region's reads first, then each equation's output.
    let mut fifo_of: HashMap<ValueId, usize> = HashMap::new();
    let mut fifos: Vec<AieFifo> = Vec::new();
    let mut open = |value: ValueId, producer: FifoEnd| {
        let (dims, element, _) = seen[&value];
        fifo_of.insert(value, fifos.len());
        fifos.push(AieFifo {
            producer,
            consumers: Vec::new(),
            depth: AieLinkRecipe::FIFO_DEPTH,
            elements: dims.iter().product(),
            element,
        });
    };
    for (binding, &value) in region.reads().iter().enumerate() {
        open(value, FifoEnd::Host { binding });
    }
    for (tile, member) in members.iter().enumerate() {
        open(
            member.out,
            FifoEnd::Tile {
                tile,
                port: member.reads.len(),
            },
        );
    }
    let mut tiles = Vec::with_capacity(members.len());
    for (tile, member) in members.iter().enumerate() {
        let mut ports = Vec::with_capacity(member.reads.len() + 1);
        for (port, value) in member.reads.iter().enumerate() {
            let fifo = *fifo_of
                .get(value)
                .ok_or(AieAbiRefusal::NotOneDispatch { eqn: member.out })?;
            fifos[fifo].consumers.push(FifoEnd::Tile { tile, port });
            ports.push(fifo);
        }
        ports.push(fifo_of[&member.out]);
        tiles.push(AieTile { ports });
    }
    for (index, &value) in region.writes().iter().enumerate() {
        let fifo = *fifo_of
            .get(&value)
            .ok_or(AieAbiRefusal::NotOneDispatch { eqn: value })?;
        fifos[fifo].consumers.push(FifoEnd::Host {
            binding: region.reads().len() + index,
        });
    }
    Ok(AieLinkRecipe::new(target, abi, tiles, fifos)?)
}

/// What `region` of `program` is linked from: its link recipe and, in tile order, the body each tile runs.
/// The recipe is derived from the same plans the bodies are, so the two cannot disagree.
pub fn region_link<'p>(
    program: &'p Program,
    region: &Region,
) -> Result<(AieLinkRecipe, Vec<&'p Body>), AieAbiRefusal> {
    let Target::Npu(NpuTarget::Aie { target, .. }) = *program.target() else {
        return Err(AieAbiRefusal::NotAnAieProgram);
    };
    let plans = program.plans();
    let recipe = aie_region_recipe(program.graph(), plans, region, target)?;
    let bodies = region
        .members()
        .map(|index| {
            plans[index]
                .single()
                .map(Dispatch::body)
                .ok_or_else(|| AieAbiRefusal::NotOneDispatch {
                    eqn: program.graph().eqns()[index].out,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((recipe, bodies))
}

#[cfg(test)]
mod tests;
