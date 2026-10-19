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

use crate::kernel_mapping::{aie_region_recipe, validate_region_recipe};
use crate::predicates::GraphTables;
use crate::refusal::refusal_at;
use crate::{Capability, CompileError, Plan, Target, value_ids};

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

/// The one region of a graph artifact: every equation of `g`, the values it reads from outside and the values
/// it returns.
pub(crate) fn whole_graph_region(g: &Graph) -> Region {
    Uses::of(g).region(g, 0..g.eqns().len())
}

/// Group the planned equations of `g` into regions for `target`, or return no regions when the target's
/// artifacts are kernels. Errors name the first equation of a region that cannot be linked on its own.
pub(crate) fn package(
    g: &Graph,
    plans: &[Plan],
    target: &Target,
) -> Result<Vec<Region>, CompileError> {
    let (ArtifactGranularity::Region { .. }, Target::Aie { target: aie, caps }) =
        (target.granularity(), target)
    else {
        // A graph artifact is the whole program: every equation planned as a node of it, in one region.
        return Ok(match target.granularity() {
            ArtifactGranularity::Graph if !plans.is_empty() => vec![whole_graph_region(g)],
            _ => Vec::new(),
        });
    };
    let uses = Uses::of(g);
    let tables = GraphTables::from(g);
    let dispatches = |index: usize| matches!(plans[index], Plan::Dispatches(_));
    let fits = |region: &Region| {
        aie_region_recipe(g, plans, region, *aie)
            .map_err(Capability::AieAbi)
            .and_then(|recipe| validate_region_recipe(&recipe, caps).map_err(Capability::AieLimit))
    };
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
    use poot_graph_ir::{BinOp, Builder, Graph, Outputs, Slot, StateRole, TensorType};
    use poot_target::aie::RegionLimits;
    use poot_target::{AieCaps, AieTargetId, Backend, Queried};

    use super::*;
    use crate::{AieLimitRefusal, CompileLimits, CompileOptions, PlanError, Program, compile};

    /// SYNTHETIC limits, roomy unless a row narrows one: not a device measurement (the NPU rows pin the
    /// measured ones).
    fn caps() -> AieCaps {
        AieCaps {
            target: AieTargetId::Xdna2,
            tile_local_bytes: Queried::Known(1 << 16),
            host_bindings: Queried::Known(8),
            instruction_bytes: Queried::Known(1 << 20),
            memory_budget_bytes: Queried::Known(1 << 30),
            region: RegionLimits {
                compute_tiles: Queried::Known(32),
                tile_input_channels: Queried::Known(2),
                tile_output_channels: Queried::Known(2),
                host_input_streams: Queried::Known(16),
                host_output_streams: Queried::Known(16),
            },
        }
    }

    fn compiled(g: &Graph, caps: AieCaps) -> Result<Program, crate::CompileError> {
        let target = Target::Aie {
            target: AieTargetId::Xdna2,
            caps,
        };
        compile(g, &target, &CompileOptions::new(CompileLimits::STANDARD))
    }

    /// `x` through `len` chained equations alternating add and mul by a constant `c`.
    fn chain(len: usize) -> Graph {
        let b = Builder::for_tests();
        let shape = vec![2, 3];
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
            .unwrap();
        let c = b.constant("c", TensorType::f32(shape)).unwrap();
        let mut acc = x;
        for i in 0..len {
            let op = if i % 2 == 0 { BinOp::Add } else { BinOp::Mul };
            acc = b.binary(op, acc, c).unwrap();
        }
        b.finish(acc).unwrap()
    }

    /// `n` independent updates `next_i = state_i + x_i` of distinct shapes, each state carried to the next
    /// step: nothing reads one equation's output but the program's own outputs, so every output is a write.
    fn independent(n: usize) -> Graph {
        let b = Builder::for_tests();
        let mut states = Vec::new();
        let mut result = None;
        for i in 0..n {
            let ty = || TensorType::f32(vec![i + 1, 2]);
            let state = b
                .state_input(&format!("s{i}"), ty(), StateRole::Recurrent)
                .unwrap();
            let x = b
                .slot_named(Slot::Activation, &format!("x{i}"), ty())
                .unwrap();
            let next = b.binary(BinOp::Add, state, x).unwrap();
            result.get_or_insert(next);
            states.push((state, next));
        }
        b.finish(Outputs::new(result.unwrap()).with_state(&states))
            .unwrap()
    }

    fn refused(error: crate::CompileError) -> crate::Refusal {
        match error {
            crate::CompileError::Plan(plan) => match *plan {
                PlanError::Refused(refusal) => *refusal,
                other => panic!("expected a typed refusal, got {other:?}"),
            },
            other => panic!("expected a planner refusal, got {other:?}"),
        }
    }

    fn members(program: &Program) -> Vec<Range<usize>> {
        program.regions().iter().map(Region::members).collect()
    }

    /// A chain of equations is one region: it reads the slot and the constant once, returns the last
    /// value, and the five values in between exist only inside the artifact, so the buffer plan has exactly
    /// one slot (the returned value) and not one per equation.
    ///
    /// Mutation: in `plan_buffers` drop the `inside_region` skip. Every intermediate gets a slot and the
    /// slot count is no longer one.
    #[test]
    fn a_chain_is_one_region_and_its_intermediates_have_no_buffer() {
        let program = compiled(&chain(6), caps()).unwrap();
        assert_eq!(members(&program), vec![0..6]);
        let region = &program.regions()[0];
        let g = program.graph();
        assert_eq!(region.reads().len(), 2, "the slot and the constant");
        assert_eq!(
            region.writes(),
            &[g.eqns()[5].out],
            "only the last value leaves"
        );
        assert_eq!(
            program.buffer_plan().slot_count(),
            1,
            "the five intermediates have no arena slot"
        );
    }

    /// A region closes where the next equation would break a bound: here the partition has 4 compute
    /// tiles, so ten equations are three regions, and each region reads what the one before returned.
    ///
    /// Mutation: drop the compute-tile bound in `validate_region_recipe`. The ten equations are one region.
    #[test]
    fn a_region_closes_where_the_next_equation_would_break_the_tile_bound() {
        let narrow = AieCaps {
            region: RegionLimits {
                compute_tiles: Queried::Known(4),
                ..caps().region
            },
            ..caps()
        };
        let program = compiled(&chain(10), narrow).unwrap();
        assert_eq!(members(&program), vec![0..4, 4..8, 8..10]);
        let g = program.graph();
        for pair in program.regions().windows(2) {
            let handed_on = pair[0].writes();
            assert_eq!(handed_on.len(), 1);
            assert!(
                pair[1].reads().contains(&handed_on[0]),
                "the next region reads what this one returned"
            );
            assert!(g.eqns()[pair[0].members().end - 1].out == handed_on[0]);
        }
    }

    /// The host streams the shim opens bound a region as well: every external read is a stream in and every
    /// returned value a stream out. Five independent updates each read two values and return one.
    ///
    /// Mutation: drop either host-stream bound in `validate_region_recipe`; the matching expectation below
    /// finds one region of five equations.
    #[test]
    fn a_region_closes_where_the_next_equation_would_break_a_host_stream_bound() {
        let reads = AieCaps {
            region: RegionLimits {
                host_input_streams: Queried::Known(4),
                ..caps().region
            },
            ..caps()
        };
        let program = compiled(&independent(5), reads).unwrap();
        assert_eq!(members(&program), vec![0..2, 2..4, 4..5]);

        let writes = AieCaps {
            region: RegionLimits {
                host_output_streams: Queried::Known(1),
                ..caps().region
            },
            ..caps()
        };
        let program = compiled(&independent(5), writes).unwrap();
        assert_eq!(members(&program), vec![0..1, 1..2, 2..3, 3..4, 4..5]);
    }

    /// An equation that does not fit a region of its own is refused with a typed capability naming it and
    /// the limit, never split into a fallback; a limit nobody measured refuses the same way.
    ///
    /// Mutation: have `package` skip the single-equation check (push the region unchecked). The compile
    /// succeeds and both expectations below fail.
    #[test]
    fn an_equation_that_fits_no_region_is_a_typed_refusal_naming_it() {
        let g = independent(2);
        let one_stream = AieCaps {
            region: RegionLimits {
                host_input_streams: Queried::Known(1),
                ..caps().region
            },
            ..caps()
        };
        let refusal = refused(compiled(&g, one_stream).unwrap_err());
        assert_eq!(refusal.eqn, g.eqns()[0].out);
        assert_eq!(
            refusal.missing,
            Capability::AieLimit(AieLimitRefusal::Exceeded {
                resource: "host input streams",
                needed: 2,
                available: 1
            })
        );

        let unmeasured = AieCaps {
            region: RegionLimits {
                compute_tiles: Queried::Unknown,
                ..caps().region
            },
            ..caps()
        };
        let refusal = refused(compiled(&g, unmeasured).unwrap_err());
        assert_eq!(
            refusal.missing,
            Capability::AieLimit(AieLimitRefusal::Unknown {
                limit: "compute tiles"
            })
        );
    }

    /// The tile-local memory bounds an equation as it bounds a region: a core's FIFO buffers must fit its
    /// tile, so an add whose three double-buffered tensors take 144 bytes is refused with a typed
    /// capability naming it when the tile holds 143, and compiles when it holds 144. It is the planner's
    /// refusal at compile, not a toolchain failure at load.
    ///
    /// Mutation: drop the `validate_recipe_tile_local` call in `validate_region_recipe`. The compile
    /// succeeds and the refusal expectation fails.
    #[test]
    fn an_equation_past_the_tile_memory_is_refused_at_compile() {
        let g = chain(1);
        let roomy = AieCaps {
            tile_local_bytes: Queried::Known(144),
            ..caps()
        };
        assert!(
            compiled(&g, roomy).is_ok(),
            "control: 144 bytes hold the add"
        );
        let tight = AieCaps {
            tile_local_bytes: Queried::Known(143),
            ..caps()
        };
        let refusal = refused(compiled(&g, tight).unwrap_err());
        assert_eq!(
            refusal.missing,
            Capability::AieLimit(AieLimitRefusal::Exceeded {
                resource: "link recipe tile-local bytes",
                needed: 144,
                available: 143
            })
        );
    }

    /// A value the program publishes is returned even when a later equation of the region reads it: here
    /// the first equation's output is carried state and feeds the second. The region links it tile to tile
    /// and to the host.
    #[test]
    fn a_published_value_is_returned_and_still_feeds_the_next_equation_on_chip() {
        let b = Builder::for_tests();
        let shape = vec![2, 3];
        let x = b
            .slot_named(Slot::Activation, "x", TensorType::f32(shape.clone()))
            .unwrap();
        let state = b
            .state_input("s", TensorType::f32(shape), StateRole::Recurrent)
            .unwrap();
        let first = b.binary(BinOp::Add, x, state).unwrap();
        let second = b.binary(BinOp::Mul, first, x).unwrap();
        let g = b
            .finish(Outputs::new(second).with_state(&[(state, first)]))
            .unwrap();
        let program = compiled(&g, caps()).unwrap();
        assert_eq!(members(&program), vec![0..2]);
        let region = &program.regions()[0];
        assert_eq!(region.writes().len(), 2, "the carried state and the result");
        let (recipe, bodies) = program.region_link(region).unwrap();
        assert_eq!(bodies.len(), 2);
        let first_fifo = recipe.tiles()[0].ports[2];
        assert_eq!(
            recipe.fifos()[first_fifo].consumers.len(),
            2,
            "the second tile and the host both read it"
        );
    }

    /// A GPU target's artifacts are kernels: nothing is packaged and every dispatch stays its own.
    #[test]
    fn a_gpu_target_has_kernel_granularity_and_no_regions() {
        let backend = Backend::SpirvVulkan;
        let target = Target::Gpu {
            backend,
            caps: poot_test_util::device_caps::default_caps_for(backend),
        };
        assert_eq!(target.granularity(), ArtifactGranularity::Kernel);
        let program = compile(
            &chain(3),
            &target,
            &CompileOptions::new(CompileLimits::STANDARD),
        )
        .unwrap();
        assert!(program.regions().is_empty());
    }

    /// An Intel NPU's artifact is the whole graph: one region over every equation, reading the slot and the
    /// constant once and returning only the last value; the values in between have no buffer, and the NPU
    /// graph's arguments are exactly the region's reads and writes.
    ///
    /// Mutation: return no region for `ArtifactGranularity::Graph` in `package`; the region list is empty.
    #[test]
    fn an_intel_npu_target_packages_the_whole_graph_as_one_region() {
        use poot_target::{VendorCompilerId, VpuCaps, VpuTargetId};
        use poot_tensor::DType;
        let b = Builder::for_tests();
        let shape = vec![2, 3];
        let x = b
            .slot_named(
                Slot::Activation,
                "x",
                TensorType::new(shape.clone(), DType::F16),
            )
            .unwrap();
        let c = b.constant("c", TensorType::new(shape, DType::F16)).unwrap();
        let mut acc = x;
        for i in 0..6 {
            let op = if i % 2 == 0 { BinOp::Add } else { BinOp::Mul };
            acc = b.binary(op, acc, c).unwrap();
        }
        let g = b.finish(acc).unwrap();
        let target = Target::Vpu {
            target: VpuTargetId::Npu4,
            caps: VpuCaps {
                target: VpuTargetId::Npu4,
                compiler: Queried::Known(VendorCompilerId { major: 8, minor: 3 }),
                memory_budget_bytes: Queried::Unknown,
                narrowing: None,
            },
        };
        let program = compile(&g, &target, &CompileOptions::new(CompileLimits::STANDARD)).unwrap();
        assert_eq!(members(&program), vec![0..6]);
        let region = &program.regions()[0];
        let graph = program.graph();
        assert_eq!(region.reads().len(), 2, "the slot and the constant");
        assert_eq!(region.writes(), &[graph.eqns()[5].out]);
        assert_eq!(program.buffer_plan().slot_count(), 1);
        let vpu = program.vpu_region().expect("the program's NPU graph");
        assert_eq!(vpu.inputs.len(), region.reads().len());
        assert_eq!(vpu.outputs.len(), region.writes().len());
        assert_eq!(vpu.nodes.len(), 6);
    }
}
