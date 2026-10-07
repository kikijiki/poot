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
/// `"poot-rocm-gpu/assets"` or `"poot-gpu/src/tests/assets"` - poot-gpu's probe assets live under its
/// own `src/tests/`, not a crate-root `assets/`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssetDest {
    GraphPlan,
    Crate(&'static str),
}

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
}

/// `AssetDest::GraphPlan` or `AssetDest::Crate(dir)` from the same `GraphPlan`/`Crate(..)` token shape
/// [`kernel_assets!`] matches, so there is exactly one place a manifest entry's destination is written.
macro_rules! kernel_asset_dest {
    (GraphPlan) => {
        $crate::imported::AssetDest::GraphPlan
    };
    (Crate($dir:literal)) => {
        $crate::imported::AssetDest::Crate($dir)
    };
}
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
/// `ImportedKernel::body_<family>` arm, plus the family's own equivalence tests (entry/param-count
/// match, parse-once caching, pairwise-distinct bodies). `family` names the generated method
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
        pub(crate) const ENTRIES: &[$crate::imported::AssetEntry] = &[
            $(
                $crate::imported::AssetEntry {
                    name: $name,
                    source: $source,
                    entry: $entry,
                    params: $params,
                    dest: $crate::imported::kernel_asset_dest!($dest_kind $(($dir))?),
                }
            ),+
        ];

        /// The variants this family owns, in the same order as [`ENTRIES`] (card 559's own tests zip
        /// the two tables; a macro-generated pair can never drift out of step by construction). Test-only:
        /// nothing outside this family's own equivalence tests needs the variant list as data.
        #[cfg(test)]
        pub(crate) const VARIANTS: &[$crate::imported::ImportedKernel] = &[
            $( $crate::imported::ImportedKernel::$variant ),+
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

        #[cfg(test)]
        mod tests {
            use super::{ENTRIES, VARIANTS};

            #[test]
            fn each_body_matches_its_manifest_entry() {
                for (entry, variant) in ENTRIES.iter().zip(VARIANTS.iter()) {
                    let body = variant.body();
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
                for variant in VARIANTS {
                    assert!(
                        ::std::ptr::eq(variant.body(), variant.body()),
                        "{variant:?} reparsed"
                    );
                }
            }

            #[test]
            fn bodies_do_not_share_a_cell_or_a_kernel_name() {
                for (i, a) in VARIANTS.iter().enumerate() {
                    for b in &VARIANTS[i + 1..] {
                        assert!(
                            !::std::ptr::eq(a.body(), b.body()),
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
        }
    };
}

mod contraction;
mod packed;
mod probe;
mod sampling;

/// The one kernel-authoring accessor: `ImportedKernel::GemvCoalesced.body()`. One variant per shipped
/// kernel, contributed by its family file's `kernel_assets!` invocation. `pub`, not `pub(crate)`: the
/// `probe` family's kernels are each consumed by a different crate (`poot-rocm-gpu`, `poot-gpu`), never
/// by this crate's own planner, so this is the one route every crate reaches a shipped kernel through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportedKernel {
    // contraction.rs
    GemvCoalesced,
    GemvBatchedCoalesced,
    GemvCoalescedBf16,
    GemvBatchedCoalescedBf16,
    TiledGemm,
    TiledGemmBatched,
    // packed.rs
    PackI8,
    UnpackI8,
    PackedBf16RowGather,
    PackedBf16ToF32,
    PackedE4m3RowGather,
    // sampling.rs
    ArgmaxBatched,
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
}

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
            | ImportedKernel::GemvCoalescedBf16
            | ImportedKernel::GemvBatchedCoalescedBf16
            | ImportedKernel::TiledGemm
            | ImportedKernel::TiledGemmBatched => {
                ImportedNeeds::serial(SerialLoop::ContractionDepth)
            }
            // Packing loops are bounded by rank or a constant; sampling's per-row
            // vocabulary loops are not modeled yet (POOT-634). The probes are test kernels no planner
            // route selects, so none is bounded here (`RmsnormDyn` loops over its row width; `RowsumFor`
            // over a constant). None uses a subgroup collective or a matrix fragment
            // (`a_declared_subgroup_matches_the_fragments_the_body_uses` pins that against the bodies).
            ImportedKernel::PackI8
            | ImportedKernel::UnpackI8
            | ImportedKernel::PackedBf16RowGather
            | ImportedKernel::PackedBf16ToF32
            | ImportedKernel::PackedE4m3RowGather
            | ImportedKernel::ArgmaxBatched
            | ImportedKernel::SampleGumbelArgmaxBatched
            | ImportedKernel::SampleTruncatedGumbelArgmaxBatched
            | ImportedKernel::SampleToppGumbelArgmaxBatched
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
pub const MANIFEST: &[&[AssetEntry]] = &[
    contraction::ENTRIES,
    packed::ENTRIES,
    sampling::ENTRIES,
    probe::ENTRIES,
];

#[cfg(test)]
mod needs_tests {
    use std::collections::HashMap;

    use poot_graph_ir::op::OpKind;
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

    fn all_kernels() -> impl Iterator<Item = ImportedKernel> {
        [
            contraction::VARIANTS,
            packed::VARIANTS,
            sampling::VARIANTS,
            probe::VARIANTS,
        ]
        .into_iter()
        .flatten()
        .copied()
    }

    /// A shipped kernel and a graph equation the planner lowers to it.
    struct Fixture {
        kernel: ImportedKernel,
        backend: Backend,
        graph: Graph,
        is_eqn: fn(&OpKind) -> bool,
        /// Whether the compile is guarded against the routed-expert hang, which keeps a matmul off the
        /// generated tiled GEMM and so on the shipped one.
        guarded: bool,
    }

    fn matmul_graph(batch: usize, m: usize, k: usize, n: usize, weight: DType) -> Graph {
        let b = Builder::new();
        let a = b.constant("a", TensorType::new(vec![batch, m, k], DType::F32));
        let w = b.constant("w", TensorType::new(vec![k, n], weight));
        let out = poot_graph_ir::ops::linear(&b, a, w, None);
        b.finish(out)
    }

    fn is_matmul(op: &OpKind) -> bool {
        matches!(op, OpKind::MatMul)
    }

    fn fixtures() -> Vec<Fixture> {
        use ImportedKernel::*;
        let wgpu = Backend::SpirvVulkan;
        let f32w = DType::F32;
        let bf16w = DType::BF16;
        let mut all = vec![
            // Decode GEMVs: single sequence and batched, F32 and BF16 weights.
            (GemvCoalesced, matmul_graph(1, 1, 256, 512, f32w)),
            (GemvBatchedCoalesced, matmul_graph(4, 1, 256, 512, f32w)),
            (GemvCoalescedBf16, matmul_graph(1, 1, 256, 512, bf16w)),
            (
                GemvBatchedCoalescedBf16,
                matmul_graph(4, 1, 256, 512, bf16w),
            ),
            // Prefill tiled GEMM: shared weight, and the per-batch attention matmul below.
            (TiledGemm, matmul_graph(1, 64, 256, 512, f32w)),
        ]
        .into_iter()
        .map(|(kernel, graph)| Fixture {
            kernel,
            backend: wgpu,
            graph,
            is_eqn: is_matmul,
            guarded: matches!(kernel, TiledGemm),
        })
        .collect::<Vec<_>>();
        let attention_scores = {
            let b = Builder::new();
            let q = b.constant("q", TensorType::new(vec![1, 14, 64, 64], DType::F32));
            let kt = b.constant("kt", TensorType::new(vec![1, 14, 64, 64], DType::F32));
            let out = b.matmul(q, kt);
            b.finish(out)
        };
        all.push(Fixture {
            kernel: TiledGemmBatched,
            backend: wgpu,
            graph: attention_scores,
            is_eqn: is_matmul,
            guarded: false,
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
            guarded,
        } in fixtures
        {
            let eqn = graph
                .eqns
                .iter()
                .find(|e| is_eqn(&e.op))
                .unwrap_or_else(|| panic!("{kernel:?}: fixture has no matching equation"));
            let plan = |caps: DeviceCaps| {
                crate::plan_eqn_compiled(
                    &crate::ExactI32StorageAnalysis::new(&graph),
                    &graph,
                    eqn,
                    &Target { backend, caps },
                    &HashMap::new(),
                    guarded,
                    &poot_test_util::graph_fixtures::roomy_body_limits(),
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
