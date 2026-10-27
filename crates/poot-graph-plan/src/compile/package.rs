//! `package`: the compile step that groups the planned equations of a program into regions, for a target
//! whose artifacts are regions rather than kernels (POOT-1093 design section 4.2, POOT-1160).
//!
//! An XDNA2 NPU holds 16 hardware contexts and one loaded artifact is one context, so a program that needs
//! more artifacts than that cannot run if every equation is its own. A region puts many planned equations in
//! one artifact: each runs on its own compute tile, and a value only the region's own equations read moves
//! tile to tile and never reaches host memory.
//!
//! `package` makes no support decision and does no semantic interpretation: every equation it groups was
//! already planned, and was admitted or refused per equation by the planner. It walks the planned equations
//! in program order and closes a region when the next equation would break one of the target's declared
//! bounds ([`poot_target::aie::RegionLimits`] and the tile and host-buffer limits of
//! [`poot_target::AieCaps`]). The bounds are checked on the link recipe the region would be linked with,
//! the recipe the engine links, so there is one definition of "fits" and the engine never meets a region
//! the target cannot hold. An equation that does not fit a region of its own is refused with a typed
//! capability naming it; it is never silently split into a fallback.
//!
//! A region is a contiguous run of dispatching equations. Contiguity keeps the execution order of the
//! program: a region runs where its last equation sat, so every value it returns is available to every
//! reader that came after the equation that produced it.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use poot_graph_ir::{Graph, ValueId};
use poot_target::aie::RegionLimits;

use crate::npu::{NpuLowering, NpuTarget};
use crate::predicates::GraphTables;
use crate::refusal::refusal_at;
use crate::{CompileError, Plan, PlanError, Target, value_ids};

/// How big a target's compiled artifacts are, and so whether `package` groups equations for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactGranularity {
    /// One artifact per planned kernel (every GPU target).
    Kernel,
    /// One artifact per region of planned equations (an XDNA NPU), bounded by `limits`.
    Region {
        /// The bounds of one region.
        limits: RegionLimits,
    },
    /// One artifact for the whole graph: the vendor graph compiler of an Intel NPU compiles every planned
    /// equation together, so `package` groups nothing for it.
    Graph,
}

/// A contiguous run of planned equations that are linked into one artifact and run in one invocation.
///
/// `reads` are the values the region takes from outside it (in the order its first reader meets them) and
/// `writes` the values it returns: the ones a later equation outside the region reads, or that the program
/// publishes (its result, a boundary export, a carried-state update or a validation witness). Every other
/// value an equation of the region produces is read only inside it and has no buffer at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    members: Range<usize>,
    reads: Vec<ValueId>,
    writes: Vec<ValueId>,
}

impl Region {
    /// The equations of the region, as a range of indices into the program graph's equations.
    pub fn members(&self) -> Range<usize> {
        self.members.clone()
    }

    /// The values the region reads from outside it.
    pub fn reads(&self) -> &[ValueId] {
        &self.reads
    }

    /// The values the region returns.
    pub fn writes(&self) -> &[ValueId] {
        &self.writes
    }
}

/// What `package` needs to know about how each value is used, computed once per program.
struct Uses {
    /// The equation that defines each equation-produced value.
    defined_at: HashMap<ValueId, usize>,
    /// The equations that read each value, by index.
    readers: HashMap<ValueId, Vec<usize>>,
    /// Values the program publishes: they must exist in memory whatever reads them.
    published: HashSet<ValueId>,
}

impl Uses {
    fn of(g: &Graph) -> Self {
        let mut defined_at = HashMap::new();
        let mut readers: HashMap<ValueId, Vec<usize>> = HashMap::new();
        for (index, eqn) in g.eqns().iter().enumerate() {
            defined_at.insert(eqn.out, index);
            for value in value_ids(eqn) {
                let readers = readers.entry(value).or_default();
                if readers.last() != Some(&index) {
                    readers.push(index);
                }
            }
        }
        let mut published: HashSet<ValueId> = HashSet::new();
        published.insert(g.result());
        published.extend(g.exports().map(|export| export.value));
        published.extend(g.state().map(|(_, next)| next));
        published.extend(g.witnesses().map(|witness| witness.value));
        Self {
            defined_at,
            readers,
            published,
        }
    }

    /// The region `members` would make: what it reads from outside and what it returns.
    fn region(&self, g: &Graph, members: Range<usize>) -> Region {
        let eqns = &g.eqns()[members.clone()];
        let mut reads = Vec::new();
        for eqn in eqns {
            for value in value_ids(eqn) {
                let inside = self
                    .defined_at
                    .get(&value)
                    .is_some_and(|defined| members.contains(defined));
                if !inside && !reads.contains(&value) {
                    reads.push(value);
                }
            }
        }
        // A value is returned when anything outside the region needs it. One nothing reads at all is
        // returned too: it is computed either way, and a FIFO with no reader is not a dataflow.
        let writes = eqns
            .iter()
            .map(|eqn| eqn.out)
            .filter(|value| {
                let readers = self.readers.get(value).map_or(&[][..], Vec::as_slice);
                self.published.contains(value)
                    || readers.is_empty()
                    || readers.iter().any(|reader| *reader >= members.end)
            })
            .collect();
        Region {
            members,
            reads,
            writes,
        }
    }
}

impl Region {
    /// The one region of a graph artifact: every equation of `g`, the values it reads from outside and the
    /// values it returns.
    pub fn whole_graph(g: &Graph) -> Region {
        Uses::of(g).region(g, 0..g.eqns().len())
    }
}

/// Group the planned equations of `g` into regions for `target`, or return no regions when the target's
/// artifacts are kernels. Errors name the first equation of a region that cannot be linked on its own.
pub(crate) fn package(
    g: &Graph,
    plans: &[Plan],
    target: &Target,
    npu: Option<&dyn NpuLowering>,
) -> Result<Vec<Region>, CompileError> {
    let (ArtifactGranularity::Region { .. }, Target::Npu(NpuTarget::Aie { target: aie, caps })) =
        (target.granularity(), target)
    else {
        // A graph artifact is the whole program: every equation planned as a node of it, in one region.
        return Ok(match target.granularity() {
            ArtifactGranularity::Graph if !plans.is_empty() => vec![Region::whole_graph(g)],
            _ => Vec::new(),
        });
    };
    let npu = npu.ok_or(PlanError::NpuLoweringMissing {
        target: (*aie).into(),
    })?;
    let uses = Uses::of(g);
    let tables = GraphTables::from(g);
    let dispatches = |index: usize| matches!(plans[index], Plan::Dispatches(_));
    let fits = |region: &Region| npu.region_fits(g, plans, region, *aie, caps);
    let mut regions = Vec::new();
    let mut start = 0;
    while start < plans.len() {
        if !dispatches(start) {
            start += 1;
            continue;
        }
        let mut region = uses.region(g, start..start + 1);
        if let Err(missing) = fits(&region) {
            return Err(refusal_at(tables, &g.eqns()[start], *aie, missing).into());
        }
        while region.members.end < plans.len() && dispatches(region.members.end) {
            let grown = uses.region(g, start..region.members.end + 1);
            if fits(&grown).is_err() {
                break;
            }
            region = grown;
        }
        start = region.members.end;
        regions.push(region);
    }
    Ok(regions)
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::{BinOp, Builder, Slot, TensorType};
    use poot_target::Backend;

    use super::*;
    use crate::{CompileLimits, CompileOptions, compile};

    /// A GPU target's artifacts are kernels: nothing is packaged and every dispatch stays its own.
    #[test]
    fn a_gpu_target_has_kernel_granularity_and_no_regions() {
        let b = Builder::for_tests();
        let shape = vec![2, 3];
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
            .unwrap();
        let c = b.constant("c", TensorType::f32(shape)).unwrap();
        let mut acc = x;
        for i in 0..3 {
            let op = if i % 2 == 0 { BinOp::Add } else { BinOp::Mul };
            acc = b.binary(op, acc, c).unwrap();
        }
        let g = b.finish(acc).unwrap();
        let backend = Backend::SpirvVulkan;
        let target = Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        assert_eq!(target.granularity(), ArtifactGranularity::Kernel);
        let program = compile(&g, &target, &CompileOptions::new(CompileLimits::STANDARD)).unwrap();
        assert!(program.regions().is_empty());
    }
}
