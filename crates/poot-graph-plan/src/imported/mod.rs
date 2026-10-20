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
    /// The asset lives in this crate's own `assets/` directory.
    GraphPlan,
    /// The asset lives in another crate's asset directory, a path from the workspace's `crates/` directory.
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
    /// Which crate's asset directory holds the committed `.kir.json`.
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
    /// Decode GEMV over an f32 `[K, N]` weight: a workgroup owns a tile of output columns and adjacent lanes read adjacent columns.
    GemvCoalesced,
    /// The coalesced decode GEMV over B rows that share one weight.
    GemvBatchedCoalesced,
    /// Thread-coarsened workgroup-tiled GEMM; shapes ride in `dims = [M, K, N]`.
    TiledGemm,
    /// Tiled GEMM with a weight per batch entry (per-expert MoE GEMM, per-head attention matmuls); shapes ride in `dims = [E, M, K, N]`.
    TiledGemmBatched,
    // packed.rs
    /// Axis-0 embedding gather over a dense BF16 table read as packed `u32` lanes, widened to F32 in the kernel.
    PackedBf16RowGather,
    /// The `Cast(BF16 -> F32)` of a BF16 const read from its packed `u32` lanes.
    PackedBf16ToF32,
    /// Axis-0 embedding gather over a dense E4M3FN table read as packed `u32` lanes, widened to F32 in the kernel.
    PackedE4m3RowGather,
    // sampling.rs
    /// Greedy sampling stage 1: one workgroup per (row, chunk) reduces a chunk of logits to a partial `(value, index, bad index)`.
    ArgmaxChunk,
    /// Greedy sampling stage 2: reduces a row's chunk partials to the row's `(token, non_finite_index)`.
    ArgmaxCombine,
    /// Temperature and min-p sampling by the Gumbel-max trick, one workgroup per row (`SampleRule::Gumbel`).
    SampleGumbelArgmaxBatched,
    /// Top-k, temperature and min-p sampling: the Gumbel-max pass behind a top-k threshold bisection (`SampleRule::GumbelTopK`).
    SampleTruncatedGumbelArgmaxBatched,
    /// Top-p, top-k, temperature and min-p sampling in one dispatch (`SampleRule::GumbelTopKTopP`).
    SampleToppGumbelArgmaxBatched,
    /// Exact uniform noise for `OpKind::RandomUniform`, one thread per output element.
    RandomUniform,
    // probe.rs
    /// Probe: a reachable bounds-check `Assert`, to prove it lowers to a real device trap.
    AssertTrap,
    /// Probe: every lane atomically adds 1 into one `u32` counter, an exact order-independent reduction.
    AtomicAddSumSmoke,
    /// Probe: a raw word-for-word copy, the engine's state-copy kernel for a device with no native buffer
    /// copy (`CopyMechanism::Kernel`).
    CopyWords,
    /// Probe: a finite busy-spin of a caller-chosen length, used to force a real completion-wait timeout.
    SpinBusy,
    /// Probe: row sums over a constant column count, a loop with a compile-time bound.
    RowsumFor,
    /// Probe: RMSNorm whose hidden width is a runtime value from `dims`.
    RmsnormDyn,
}

impl ImportedKernel {
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
mod manifest_tests {
    use super::*;

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
}
