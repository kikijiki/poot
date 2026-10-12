//! Card 559: the one authoring route and one asset manifest for kernels authored in ordinary Rust
//! (`pootc/kernels/<family>/*.rs`), imported by `pootc` to a [`Body`], shipped as a committed
//! `assets/*.kir.json` and parsed once per process via `OnceLock`. One file per family
//! ([`contraction`], [`packed`], [`sampling`], [`probe`]) so a later card
//! that retires a whole family (560a/560b/561a) deletes one file plus its lines here, never a
//! scattered edit. [`MANIFEST`] is the single source of truth `pootc/tests/import_run.rs`'s
//! `committed_assets_match_their_kernel_sources` iterates: every committed asset has exactly one entry,
//! and every entry's source regenerates its committed bytes exactly (`just regen-kernel-assets`, card
//! 559 R-559-1..2).
//!
//! [`ImportedKernel`] is the one production accessor: `ImportedKernel::GemvCoalesced.body()`. Each
//! family file defines its own slice of `ImportedKernel` variants (via `kernel_assets!`) plus the
//! matching [`AssetEntry`] table; [`ImportedKernel::body`] tries each family's generated `body_*` arm in
//! turn. The macro also emits each family's own equivalence tests (name, param count, distinct
//! `OnceLock` cells), so no family file hand-duplicates that test module.

use std::sync::OnceLock;

use poot_kernel_ir::Body;

/// Parse `json` into a [`Body`] on first access and cache it in `cell` for the process (every embedded
/// asset's `.body()` accessor shares this one parse-once helper).
pub(crate) fn parse_once(cell: &'static OnceLock<Body>, json: &str, name: &str) -> &'static Body {
    cell.get_or_init(|| {
        serde_json::from_str(json)
            .unwrap_or_else(|e| panic!("embedded {name} Body must deserialize: {e}"))
    })
}

/// Where a manifest entry's committed `.kir.json` lives: `poot-graph-plan`'s own `assets/`, or another
/// crate's asset directory (a path from the workspace's `crates/` directory, e.g.
/// `"poot-rocm-device/assets"` or `"poot-wgpu-device/src/tests/assets"` - poot-wgpu-device's probe assets live under its
/// own `src/tests/`, not a crate-root `assets/`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(any(test, feature = "test-support"))]
pub enum AssetDest {
    GraphPlan,
    Crate(&'static str),
}

#[cfg(any(test, feature = "test-support"))]
impl AssetDest {
    /// The destination directory, relative to the workspace's `crates/` directory.
    pub fn dir(self) -> &'static str {
        match self {
            AssetDest::GraphPlan => "poot-graph-plan/assets",
            AssetDest::Crate(dir) => dir,
        }
    }
}

/// One manifest row: a committed asset, the kernel source it is regenerated from, the kernel's own
/// entry-point name (its `body.name` once imported) and parameter count, and which crate ships it.
#[derive(Clone, Copy, Debug)]
#[cfg(any(test, feature = "test-support"))]
pub struct AssetEntry {
    /// The asset's file stem (`<name>.kir.json`) and, for every entry but the coarsened tiled-GEMM
    /// family, the kernel source's file stem too.
    pub name: &'static str,
    /// The kernel source's file stem under `pootc/kernels/<family>/` (differs from `name` only for the
    /// coarsened tiled-GEMM bodies).
    pub source: &'static str,
    /// The imported `Body`'s own entry-point name (`__poot_kernel_<source>`), checked against the
    /// parsed body so an entry can never silently point at the wrong asset.
    pub entry: &'static str,
    /// The kernel's expected parameter count (buffers + metadata), checked against the parsed body.
    pub params: u32,
    pub dest: AssetDest,
    /// The kernel this row ships, so a consumer walks the variants through the one manifest.
    pub kernel: ImportedKernel,
}

/// `AssetDest::GraphPlan` or `AssetDest::Crate(dir)` from the same `GraphPlan`/`Crate(..)` token shape
/// [`kernel_assets!`] matches, so there is exactly one place a manifest entry's destination is written.
#[cfg(any(test, feature = "test-support"))]
macro_rules! kernel_asset_dest {
    (GraphPlan) => {
        $crate::imported::AssetDest::GraphPlan
    };
    (Crate($dir:literal)) => {
        $crate::imported::AssetDest::Crate($dir)
    };
}
#[cfg(any(test, feature = "test-support"))]
pub(crate) use kernel_asset_dest;

/// The committed asset's `include_str!` path, built from the same `dest`/`name` tokens the manifest
/// entry is built from (card 559): the embedded bytes and the path the staleness test checks
/// against `AssetDest::dir()` can never disagree, because both come from this one macro. Relative to
/// this file's directory (`src/imported/`), matching every family file's own position.
macro_rules! kernel_asset_json {
    (GraphPlan, $name:literal) => {
        include_str!(concat!("../../assets/", $name, ".kir.json"))
    };
    (Crate($dir:literal), $name:literal) => {
        include_str!(concat!("../../../", $dir, "/", $name, ".kir.json"))
    };
}
pub(crate) use kernel_asset_json;

/// Define one family's [`ImportedKernel`] variants, their [`AssetEntry`] table, and the family's
/// `ImportedKernel::body_<family>` arm (the entry/param-count, parse-once and pairwise-distinct body
/// checks over every family's entries live in `needs_tests`). `family` names the generated method
/// (`body_contraction`, `body_probe`, ...), dispatched from [`ImportedKernel::body`] below. `dest` is
/// `GraphPlan` or `Crate("<dir>")` (bare tokens, not `AssetDest::..`): [`kernel_asset_dest`] and
/// [`kernel_asset_json`] both build from it, so the entry's `dest` and its embedded JSON's path can
/// never name two different files (review F7).
macro_rules! kernel_assets {
    (
        family = $family:ident;
        $(
            $variant:ident {
                name: $name:literal,
                source: $source:literal,
                entry: $entry:literal,
                params: $params:expr,
                dest: $dest_kind:ident $(( $dir:literal ))? $(,)?
            }
        ),+ $(,)?
    ) => {
        #[cfg(any(test, feature = "test-support"))]
        pub(crate) const ENTRIES: &[$crate::imported::AssetEntry] = &[
            $(
                $crate::imported::AssetEntry {
                    name: $name,
                    source: $source,
                    entry: $entry,
                    params: $params,
                    dest: $crate::imported::kernel_asset_dest!($dest_kind $(($dir))?),
                    kernel: $crate::imported::ImportedKernel::$variant,
                }
            ),+
        ];

        impl $crate::imported::ImportedKernel {
            pub(crate) fn $family(self) -> ::std::option::Option<&'static ::poot_kernel_ir::Body> {
                match self {
                    $(
                        $crate::imported::ImportedKernel::$variant => {
                            static BODY: ::std::sync::OnceLock<::poot_kernel_ir::Body> =
                                ::std::sync::OnceLock::new();
                            ::std::option::Option::Some($crate::imported::parse_once(
                                &BODY,
                                $crate::imported::kernel_asset_json!($dest_kind $(($dir))?, $name),
                                $name,
                            ))
                        }
                    )+
                    _ => ::std::option::Option::None,
                }
            }
        }
    };
}

mod contraction;
mod packed;
mod probe;
mod sampling;

/// The one kernel-authoring accessor: `ImportedKernel::GemvCoalesced.body()`. One variant per shipped
/// kernel, contributed by its family file's `kernel_assets!` invocation. `pub`, not `pub(crate)`: the
/// `probe` family's kernels are each consumed by a different crate (`poot-rocm-device`, `poot-wgpu-device`), never
/// by this crate's own planner, so this is the one route every crate reaches a shipped kernel through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedKernel {
    // contraction.rs
    GemvCoalesced,
    GemvBatchedCoalesced,
    TiledGemm,
    TiledGemmBatched,
    // packed.rs
    PackI8,
    UnpackI8,
    PackedBf16RowGather,
    PackedBf16ToF32,
    PackedE4m3RowGather,
    // sampling.rs
    ArgmaxChunk,
    ArgmaxCombine,
    SampleGumbelArgmaxBatched,
    SampleTruncatedGumbelArgmaxBatched,
    SampleToppGumbelArgmaxBatched,
    RandomUniform,
    // probe.rs
    AssertTrap,
    AtomicAddSumSmoke,
    CopyWords,
    SpinBusy,
    RowsumFor,
    RmsnormDyn,
}

/// What a shipped kernel needs from the device that its body does not state. The resources the body does
/// state (static LDS, workgroup shape, fragments) are measured from it by the finalizer; a body that uses
/// a matrix fragment without declaring `subgroup_lanes` is refused there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ImportedNeeds {
    /// The subgroup lane count the body's collectives are written for.
    pub(crate) subgroup_lanes: Option<u32>,
    /// The dependent steps each thread runs, as a rule over the equation's shapes.
    pub(crate) serial: SerialLoop,
}

/// How a shipped kernel's per-thread serial loop depth follows from its equation (`kernel_mapping`
/// evaluates the rule into a `SerialWork`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SerialLoop {
    /// No per-thread loop whose bound is a data dimension (loops over rank, over a constant, or none).
    None,
    /// One thread per output element loops over the contraction depth K, the first operand's last
    /// dimension.
    ContractionDepth,
    /// One workgroup of [`SAMPLER_LANES`] lanes per row of a `[rows, vocab]` logits operand; each lane
    /// strides the row's vocabulary `passes` times (a reduction pass, a bisection iteration, the final
    /// perturbed argmax), so a lane runs `passes * ceil(vocab / SAMPLER_LANES)` dependent steps. The
    /// output is `[rows, 2]`: `rows` is half its element count. The formula omits lane 0's cross-lane
    /// reduce (`SAMPLER_LANES` steps per pass), negligible against `ceil(vocab / SAMPLER_LANES)`.
    RowVocab { passes: u64 },
}

/// The lanes of the one workgroup each sampler body runs per row (`W` in `pootc/kernels/sampling/*.rs`).
pub(crate) const SAMPLER_LANES: u64 = 64;

/// Vocabulary passes of each stochastic `SampleToken` body: Gumbel scans for the finite
/// range, then for the perturbed argmax; the truncated rule adds a 30-step top-k bisection, each step a
/// scan; top-p adds a sum scan and a second 30-step bisection. `a_sampling_declaration_matches_the_loops_of_its_body`
/// derives the count from each body's control flow and fails when a declaration drifts from it.
const BISECTION_STEPS: u64 = 30;
const GUMBEL_PASSES: u64 = 2;
const TRUNCATED_GUMBEL_PASSES: u64 = GUMBEL_PASSES + BISECTION_STEPS;
const TOPP_GUMBEL_PASSES: u64 = TRUNCATED_GUMBEL_PASSES + 1 + BISECTION_STEPS;

impl ImportedNeeds {
    const NONE: Self = Self::serial(SerialLoop::None);

    const fn serial(serial: SerialLoop) -> Self {
        Self {
            subgroup_lanes: None,
            serial,
        }
    }
}

impl ImportedKernel {
    /// The declaration for this kernel. Exhaustive: a new kernel does not compile until it declares.
    pub(crate) fn needs(self) -> ImportedNeeds {
        match self {
            // Every contraction kernel loops over K per thread (the GEMVs per output column, the tiled
            // GEMMs per output tile element).
            ImportedKernel::GemvCoalesced
            | ImportedKernel::GemvBatchedCoalesced
            | ImportedKernel::TiledGemm
            | ImportedKernel::TiledGemmBatched => {
                ImportedNeeds::serial(SerialLoop::ContractionDepth)
            }
            ImportedKernel::SampleGumbelArgmaxBatched => {
                ImportedNeeds::serial(SerialLoop::RowVocab {
                    passes: GUMBEL_PASSES,
                })
            }
            ImportedKernel::SampleTruncatedGumbelArgmaxBatched => {
                ImportedNeeds::serial(SerialLoop::RowVocab {
                    passes: TRUNCATED_GUMBEL_PASSES,
                })
            }
            ImportedKernel::SampleToppGumbelArgmaxBatched => {
                ImportedNeeds::serial(SerialLoop::RowVocab {
                    passes: TOPP_GUMBEL_PASSES,
                })
            }
            // Packing loops are bounded by rank or a constant; `RandomUniform` is one thread per
            // element. The two greedy stages are not bounded by `max_dispatch_work`: `ArgmaxChunk`
            // loops over its `ARGMAX_CHUNK`-element slice of a row (16 steps per lane) and
            // `ArgmaxCombine` over a row's chunk count, so each lane's work is bounded by vocab / 1024
            // (a handful of steps at any real vocabulary), not by the vocabulary itself.
            // The probes are test kernels no planner route selects, so none is bounded here
            // (`RmsnormDyn` loops over its row width; `RowsumFor` over a constant). None uses a
            // subgroup collective or a matrix fragment
            // (`a_declared_subgroup_matches_the_fragments_the_body_uses` pins that against the bodies).
            ImportedKernel::PackI8
            | ImportedKernel::UnpackI8
            | ImportedKernel::PackedBf16RowGather
            | ImportedKernel::PackedBf16ToF32
            | ImportedKernel::PackedE4m3RowGather
            | ImportedKernel::ArgmaxChunk
            | ImportedKernel::ArgmaxCombine
            | ImportedKernel::RandomUniform
            | ImportedKernel::AssertTrap
            | ImportedKernel::AtomicAddSumSmoke
            | ImportedKernel::CopyWords
            | ImportedKernel::SpinBusy
            | ImportedKernel::RowsumFor
            | ImportedKernel::RmsnormDyn => ImportedNeeds::NONE,
        }
    }

    /// Parse (once per process) and return this kernel's imported `Body`. Tries each family's
    /// generated arm in turn; exactly one ever matches (`kernel_assets!`'s per-family match is
    /// exhaustive over that family's own variants).
    pub fn body(self) -> &'static Body {
        self.body_contraction()
            .or_else(|| self.body_packed())
            .or_else(|| self.body_sampling())
            .or_else(|| self.body_probe())
            .unwrap_or_else(|| unreachable!("{self:?} has no family body() arm"))
    }
}

/// The one kernel-asset manifest: every committed `.kir.json` this workspace ships, grouped by family.
/// `pootc/tests/import_run.rs::committed_assets_match_their_kernel_sources` iterates it to prove every
/// destination `assets/` directory holds exactly these files, byte-identical to regenerating each entry
/// from its `pootc/kernels/<family>/<source>.rs`.
#[cfg(any(test, feature = "test-support"))]
pub const MANIFEST: &[&[AssetEntry]] = &[
    contraction::ENTRIES,
    packed::ENTRIES,
    sampling::ENTRIES,
    probe::ENTRIES,
];

#[cfg(test)]
mod needs_tests {

    use std::collections::HashMap;

    use poot_graph_ir::op::{OpKind, SampleRule};
    use poot_graph_ir::{Builder, Graph, TensorType};
    use poot_kernelgen::BodyUse;
    use poot_target::{Backend, DeviceCaps, Queried};
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::device_validation::ResourceRefusal;
    use crate::{Capability, KernelChoice, PlanError, Target};

    /// Every shipped body states its fragment use, and a body with none declares no subgroup: the
    /// declaration table and the bodies cannot drift apart. Mutation: declare `subgroup_lanes: Some(32)`
    /// for one kernel (or ship a body with a `WmmaLoad`) and this fails naming the kernel.
    #[test]
    fn a_declared_subgroup_matches_the_fragments_the_body_uses() {
        for kernel in all_kernels() {
            let used = BodyUse::measure(kernel.body()).expect("measurable LDS");
            assert_eq!(
                kernel.needs().subgroup_lanes.is_some(),
                !used.fragments.is_empty(),
                "{kernel:?}: declared subgroup vs fragments used"
            );
        }
    }

    /// The vocabulary passes a sampler body runs, read from its control flow: every natural loop whose
    /// bound is not a constant scans the row (`while j < vocab`), and counts once per trip of the
    /// constant-bound loops around it (a bisection's `iter < 30`). Loops bounded by a constant that
    /// enclose no scan (the cross-lane reduce over the 64 lanes) add nothing. A scan inside a loop whose
    /// bound is not a constant has no static count and panics.
    fn vocabulary_passes(body: &Body) -> u64 {
        use poot_kernel_ir::{BinOp, Constant, Operand, Rvalue, Statement, Terminator};
        use std::collections::{BTreeMap, BTreeSet};

        let successors = |block: usize| -> Vec<usize> {
            match &body.blocks[block].terminator {
                Terminator::Goto { target }
                | Terminator::Barrier { target }
                | Terminator::ThreadIndexCall { target, .. } => vec![target.index as usize],
                Terminator::SwitchInt { targets, .. } => targets
                    .branches
                    .iter()
                    .map(|(_, target)| target.index as usize)
                    .chain([targets.otherwise.index as usize])
                    .collect(),
                Terminator::Return | Terminator::Trap { .. } => Vec::new(),
            }
        };
        let mut predecessors = vec![Vec::new(); body.blocks.len()];
        for block in 0..body.blocks.len() {
            for next in successors(block) {
                predecessors[next].push(block);
            }
        }
        // Back edges of a depth-first walk from the entry; each header's loop is every block that reaches
        // the edge's tail without passing the header.
        let mut state = vec![0u8; body.blocks.len()];
        let mut back_edges = Vec::new();
        let mut stack = vec![(0usize, successors(0))];
        state[0] = 1;
        while let Some((block, pending)) = stack.last_mut() {
            let block = *block;
            match pending.pop() {
                Some(next) if state[next] == 1 => back_edges.push((block, next)),
                Some(next) if state[next] == 0 => {
                    state[next] = 1;
                    stack.push((next, successors(next)));
                }
                Some(_) => {}
                None => {
                    state[block] = 2;
                    stack.pop();
                }
            }
        }
        let mut loops: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for (tail, header) in back_edges {
            let members = loops.entry(header).or_default();
            members.extend([header, tail]);
            let mut work = vec![tail];
            while let Some(block) = work.pop() {
                if block == header {
                    continue;
                }
                for &pred in &predecessors[block] {
                    if members.insert(pred) {
                        work.push(pred);
                    }
                }
            }
        }
        // `Some(n)`: the header tests `i < n` against a constant; `None`: against anything else.
        let constant_trip = |header: usize| -> Option<u64> {
            let block = &body.blocks[header];
            let Terminator::SwitchInt {
                discr: Operand::Move(cond) | Operand::Copy(cond),
                ..
            } = &block.terminator
            else {
                return None;
            };
            block
                .statements
                .iter()
                .find_map(|statement| match statement {
                    Statement::Assign(
                        dst,
                        Rvalue::BinaryOp(BinOp::Lt, _, Operand::Const(bound)),
                    ) if dst == cond => match bound {
                        Constant::Usize(n) => Some(*n),
                        Constant::U32(n) => Some(u64::from(*n)),
                        Constant::I32(n) => u64::try_from(*n).ok(),
                        _ => None,
                    },
                    _ => None,
                })
        };
        let mut passes = 0;
        for (&header, members) in &loops {
            if constant_trip(header).is_some() {
                continue;
            }
            let mut trips = 1u64;
            for (&outer, outer_members) in &loops {
                if outer != header
                    && outer_members.contains(&header)
                    && outer_members.is_superset(members)
                {
                    trips *= constant_trip(outer).unwrap_or_else(|| {
                        panic!(
                            "{}: a scan loop sits in the non-constant loop at block {outer}",
                            body.name
                        )
                    });
                }
            }
            passes += trips;
        }
        passes
    }

    /// A sampler's declared pass count is the one its body's control flow has. Mutation: change a
    /// bisection's trip count (`while iter < 30`) or add or remove a `while j < vocab` scan in a
    /// `pootc/kernels/sampling/*.rs` body and regenerate its asset; the declaration for that kernel no
    /// longer matches and this fails naming it.
    #[test]
    fn a_sampling_declaration_matches_the_loops_of_its_body() {
        for kernel in sampling::ENTRIES.iter().map(|entry| entry.kernel) {
            let SerialLoop::RowVocab { passes } = kernel.needs().serial else {
                continue;
            };
            assert_eq!(
                passes,
                vocabulary_passes(kernel.body()),
                "{kernel:?}: declared vocabulary passes vs the body's loops"
            );
        }
    }

    fn all_kernels() -> impl Iterator<Item = ImportedKernel> {
        MANIFEST
            .iter()
            .flat_map(|family| family.iter().map(|entry| entry.kernel))
    }

    /// Every manifest row's body is the entry point and arity the row declares, with blocks.
    #[test]
    fn each_body_matches_its_manifest_entry() {
        for entry in MANIFEST.iter().flat_map(|family| family.iter()) {
            let body = entry.kernel.body();
            assert_eq!(body.name, entry.entry, "asset {}", entry.name);
            assert_eq!(
                body.params().count() as u32,
                entry.params,
                "asset {}",
                entry.name
            );
            assert!(!body.blocks.is_empty(), "{} has no blocks", entry.name);
        }
    }

    #[test]
    fn each_body_is_parsed_once_and_static() {
        for kernel in all_kernels() {
            assert!(
                std::ptr::eq(kernel.body(), kernel.body()),
                "{kernel:?} reparsed"
            );
        }
    }

    #[test]
    fn bodies_do_not_share_a_cell_or_a_kernel_name() {
        let kernels: Vec<_> = all_kernels().collect();
        for (i, a) in kernels.iter().enumerate() {
            for b in &kernels[i + 1..] {
                assert!(
                    !std::ptr::eq(a.body(), b.body()),
                    "{a:?} and {b:?} share a Body"
                );
                assert_ne!(
                    a.body().name,
                    b.body().name,
                    "{a:?} and {b:?} share a kernel name"
                );
            }
        }
    }

    /// A shipped kernel and a graph equation the planner lowers to it.
    struct Fixture {
        kernel: ImportedKernel,
        backend: Backend,
        graph: Graph,
        is_eqn: fn(&OpKind) -> bool,
    }

    fn matmul_graph(batch: usize, m: usize, k: usize, n: usize, weight: DType) -> Graph {
        let b = Builder::for_tests();
        let a = b
            .constant("a", TensorType::new(vec![batch, m, k], DType::F32))
            .unwrap();
        let w = b
            .constant("w", TensorType::new(vec![k, n], weight))
            .unwrap();
        let out = poot_graph_ir::ops::linear(&b, a, w, None).unwrap();
        b.finish(out).unwrap()
    }

    fn is_matmul(op: &OpKind) -> bool {
        matches!(op, OpKind::MatMul)
    }

    fn is_sample_token(op: &OpKind) -> bool {
        matches!(op, OpKind::SampleToken { .. })
    }

    const SAMPLE_ROWS: usize = 2;
    const SAMPLE_VOCAB: usize = 1000;

    /// A `[SAMPLE_ROWS, SAMPLE_VOCAB]` sampling equation of `rule`, with the operands the rule takes.
    fn sample_graph(rule: SampleRule) -> Graph {
        let b = Builder::for_tests();
        let logits = b
            .constant(
                "logits",
                TensorType::new(vec![SAMPLE_ROWS, SAMPLE_VOCAB], DType::F32),
            )
            .unwrap();
        let out = if rule == SampleRule::Greedy {
            b.sample_token(rule, logits, None, None, None).unwrap()
        } else {
            let noise = b
                .constant(
                    "noise",
                    TensorType::new(vec![SAMPLE_ROWS, SAMPLE_VOCAB], DType::F32),
                )
                .unwrap();
            let param_cols = if rule == SampleRule::GumbelTopKTopP {
                4
            } else {
                3
            };
            let params = b
                .constant(
                    "params",
                    TensorType::new(vec![SAMPLE_ROWS, param_cols], DType::F32),
                )
                .unwrap();
            let top_k = (rule != SampleRule::Gumbel).then(|| {
                b.constant("top_k", TensorType::new(vec![SAMPLE_ROWS], DType::I32))
                    .unwrap()
            });
            b.sample_token(rule, logits, Some(noise), Some(params), top_k)
                .unwrap()
        };
        b.finish(out).unwrap()
    }

    /// Each stochastic sampler's declared vocabulary work is `rows * lanes` threads of `passes * ceil(vocab / lanes)`
    /// steps: a budget of exactly that admits the dispatch, one step less refuses it with `DispatchWork`
    /// naming it. Splitting is not available (a row's reduction cannot cross launches), so the bound is a
    /// typed refusal, not a split.
    ///
    /// Mutation: make `SerialLoop::RowVocab` evaluate to no work (`None`) in `choice_requirements`, or change
    /// a rule's pass count in `ImportedKernel::needs`; the exact-budget or one-step-less row fails.
    #[test]
    fn a_sampling_dispatch_is_bounded_by_its_vocabulary_passes() {
        use ImportedKernel::*;
        let backend = Backend::SpirvVulkan;
        let lane_steps = (SAMPLE_VOCAB as u64).div_ceil(SAMPLER_LANES);
        for (rule, kernel) in [
            (SampleRule::Gumbel, SampleGumbelArgmaxBatched),
            (SampleRule::GumbelTopK, SampleTruncatedGumbelArgmaxBatched),
            (SampleRule::GumbelTopKTopP, SampleToppGumbelArgmaxBatched),
        ] {
            // The pass count is the kernel's declaration; the body is pinned to it by
            // `a_sampling_declaration_matches_the_loops_of_its_body`.
            let SerialLoop::RowVocab { passes } = kernel.needs().serial else {
                panic!("{kernel:?} declares no vocabulary loop");
            };
            let graph = sample_graph(rule);
            let eqn = graph
                .eqns()
                .iter()
                .find(|e| is_sample_token(&e.op))
                .expect("fixture has a SampleToken equation");
            let plan = |budget: u64| {
                let mut caps = default_caps_for(backend);
                caps.max_dispatch_work = Queried::Known(budget);
                crate::planner::plan_eqn_compiled(
                    &graph,
                    eqn,
                    &Target::Gpu { backend, caps },
                    &HashMap::new(),
                    &crate::test_support::roomy_body_limits(),
                )
            };
            let work = SAMPLE_ROWS as u64 * SAMPLER_LANES * passes * lane_steps;
            let admitted = plan(work).unwrap_or_else(|e| panic!("{rule:?}: {e}"));
            assert!(
                matches!(admitted.choice, KernelChoice::Imported { kernel: k, .. } if k == kernel),
                "{rule:?}: lowers to {:?}",
                admitted.choice
            );
            match plan(work - 1) {
                Err(PlanError::Refused(refusal)) => assert!(
                    matches!(
                        refusal.missing,
                        Capability::KernelResources(ResourceRefusal::DispatchWork { work: w, budget })
                            if w == work && budget == work - 1
                    ),
                    "{rule:?}: wrong refusal {:?}",
                    refusal.missing
                ),
                other => panic!(
                    "{rule:?}: expected DispatchWork {{ work: {work} }}, got {:?}",
                    other.map(|p| p.choice)
                ),
            }
        }
    }

    fn fixtures() -> Vec<Fixture> {
        use ImportedKernel::*;
        let wgpu = Backend::SpirvVulkan;
        let f32w = DType::F32;
        let mut all = vec![
            // Decode GEMVs: single sequence and batched, F32 weights.
            (GemvCoalesced, matmul_graph(1, 1, 256, 512, f32w)),
            (GemvBatchedCoalesced, matmul_graph(4, 1, 256, 512, f32w)),
        ]
        .into_iter()
        .map(|(kernel, graph)| Fixture {
            kernel,
            backend: wgpu,
            graph,
            is_eqn: is_matmul,
        })
        .collect::<Vec<_>>();
        all.extend(
            [
                (SampleGumbelArgmaxBatched, SampleRule::Gumbel),
                (SampleTruncatedGumbelArgmaxBatched, SampleRule::GumbelTopK),
                (SampleToppGumbelArgmaxBatched, SampleRule::GumbelTopKTopP),
            ]
            .map(|(kernel, rule)| Fixture {
                kernel,
                backend: wgpu,
                graph: sample_graph(rule),
                is_eqn: is_sample_token,
            }),
        );
        let attention_scores = {
            let b = Builder::for_tests();
            let q = b
                .constant("q", TensorType::new(vec![1, 14, 64, 64], DType::F32))
                .unwrap();
            let kt = b
                .constant("kt", TensorType::new(vec![1, 14, 64, 64], DType::F32))
                .unwrap();
            let out = b.matmul(q, kt).unwrap();
            b.finish(out).unwrap()
        };
        // A batched decode projection past the NVPTX GEMV's width gate: not a decode GEMV, and its
        // `[B, 1, N]` output has no tileable M, so the shipped tiled GEMM plans it.
        all.push(Fixture {
            kernel: TiledGemm,
            backend: Backend::Nvptx,
            graph: matmul_graph(4, 1, 256, 512, f32w),
            is_eqn: is_matmul,
        });
        // The per-batch attention matmul.
        all.push(Fixture {
            kernel: TiledGemmBatched,
            backend: wgpu,
            graph: attention_scores,
            is_eqn: is_matmul,
        });
        all
    }

    /// Every kernel whose declaration is a serial loop bounds a dispatch the planner really builds: the
    /// equation lowers to it on a backend whose gate selects it, and a device whose dispatch budget is one
    /// step refuses it with `DispatchWork`. The table covers every declared kernel.
    ///
    /// Mutation, per group: move one variant (a bias, batched, bf16 or tiled GEMM, or the dense BF16 GEMV)
    /// to `ImportedNeeds::NONE` in `ImportedKernel::needs`; that fixture plans under the
    /// one-step budget and its row fails ("expected a DispatchWork refusal").
    #[test]
    fn every_serial_declaration_bounds_a_planned_dispatch() {
        let fixtures = fixtures();
        for kernel in all_kernels() {
            if kernel.needs().serial != SerialLoop::None {
                assert!(
                    fixtures.iter().any(|f| f.kernel == kernel),
                    "{kernel:?} declares a serial loop but has no planner fixture"
                );
            }
        }
        for Fixture {
            kernel,
            backend,
            graph,
            is_eqn,
        } in fixtures
        {
            let eqn = graph
                .eqns()
                .iter()
                .find(|e| is_eqn(&e.op))
                .unwrap_or_else(|| panic!("{kernel:?}: fixture has no matching equation"));
            let plan = |caps: DeviceCaps| {
                crate::planner::plan_eqn_compiled(
                    &graph,
                    eqn,
                    &Target::Gpu { backend, caps },
                    &HashMap::new(),
                    &crate::test_support::roomy_body_limits(),
                )
            };
            let roomy = default_caps_for(backend);
            let planned = plan(roomy).unwrap_or_else(|e| panic!("{kernel:?}: {e}"));
            let chosen = match &planned.choice {
                KernelChoice::Imported { kernel, .. } => Some(*kernel),
                _ => None,
            };
            assert_eq!(
                chosen,
                Some(kernel),
                "{kernel:?}: the fixture lowers to {:?}",
                planned.choice
            );

            let mut tight = roomy;
            tight.max_dispatch_work = Queried::Known(1);
            match plan(tight) {
                Err(PlanError::Refused(refusal))
                    if matches!(
                        refusal.missing,
                        Capability::KernelResources(ResourceRefusal::DispatchWork { .. })
                    ) => {}
                other => panic!(
                    "{kernel:?}: expected a DispatchWork refusal, got {:?}",
                    other.map(|p| p.choice)
                ),
            }
        }
    }
}
