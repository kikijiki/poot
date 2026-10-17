//! POOT-664: the AIE artifact, shape ABI, cache identity and toolchain boundary, model-free.
//!
//! The Peano and IRON tools here are small shell scripts run through the real orchestration (resolution,
//! scratch, process, validation, publication). They prove the compiler's protocol handling and typed
//! failures. They are not machine code and not hardware. The one test that claims a Peano/IRON positive
//! compile, `a_real_peano_and_iron_toolchain_links_a_loadable_artifact`, is `#[ignore]`d and runs the pinned
//! Peano `llc` and the real `scripts/poot-aie-link.py` driver under `nix develop .#npu` (POOT-1118); it is
//! compile-only, no hardware executes. The "fake loader" is a host-side stand-in for the XDNA runtime's
//! pre-XRT checks.

use std::num::NonZeroU64;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use poot_codegen::{
    AieAbiMismatch, AieCompileError, AieIdentity, AieLinkRecipe, AieStage, AieToolIdentity,
    AieTools, KernelCache, Target, Toolchain, aie_kernel_handle, compile_aie,
    test_support::single_tile_recipe,
};
use poot_runtime_common::{
    AieArtifactError, AieBindError, AieBinding, AieBound, AieLinkedArtifact, AieShapeAbi,
    ArgAccess, KernelCode, check_aie_bindings, check_kernel_args,
};
use poot_target::{AieTargetId, BufferStorage, ElementKind, LogicalDType, TargetId};

const LIMIT: NonZeroU64 = NonZeroU64::new(1 << 20).unwrap();

fn limits() -> poot_kernelgen::BodyLimits {
    poot_kernelgen::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(4096).unwrap(),
        max_locals: std::num::NonZeroUsize::new(256).unwrap(),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("poot-npu-artifact-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn script(dir: &Path, name: &str, body: &str) -> String {
    // One file per distinct tool: a test that builds several tools must not overwrite an earlier one.
    static NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = dir.join(format!("{name}-{nonce}"));
    std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path.to_string_lossy().into_owned()
}

/// A Peano stand-in: lists `aie2p` in `--version`, and "compiles" by writing an ELF header and the
/// emitted IR (which names the kernel's symbol) to the `-o` path.
fn peano(dir: &Path, targets: &str) -> String {
    script(
        dir,
        "peano-llc",
        &format!(
            r#"case "$1" in --version) echo "fake-peano 1 ({targets})"; exit 0;; esac
for arg; do case "$prev" in -o) out="$arg";; esac; case "$arg" in *.ll) ll="$arg";; esac; prev="$arg"; done
printf '\177ELF' > "$out"
cat "$ll" >> "$out""#
        ),
    )
}

/// How an IRON stand-in behaves when asked to link.
enum Iron {
    Links,
    /// Writes the xclbin, then fails before the instruction stream: the half-built bundle case.
    FailsAfterFirstOutput,
    /// Exits 0 without writing the instruction stream.
    WritesNoInstructions,
    /// Writes an xclbin without the `xclbin2` magic.
    WritesGarbage,
}

fn iron(dir: &Path, platforms: &str, mode: Iron) -> String {
    let link = match mode {
        Iron::Links => r#"printf 'xclbin2\0payload' > "$x"; printf 'IIII' > "$i""#,
        Iron::FailsAfterFirstOutput => r#"printf 'xclbin2\0payload' > "$x"; echo boom >&2; exit 3"#,
        Iron::WritesNoInstructions => r#"printf 'xclbin2\0payload' > "$x""#,
        Iron::WritesGarbage => r#"printf 'not an xclbin' > "$x"; printf 'IIII' > "$i""#,
    };
    script(
        dir,
        "iron-link",
        &format!(
            r#"case "$1" in --version) echo "fake-iron 7"; exit 0;; --platforms) echo "{platforms}"; exit 0;; esac
while [ $# -gt 0 ]; do case "$1" in --xclbin) x="$2";; --insts) i="$2";; esac; shift; done
{link}"#
        ),
    )
}

fn toolchain(peano: Option<String>, iron: Option<String>) -> Toolchain {
    Toolchain::from_env_with(move |name| match name {
        "POOT_AIE_LLC" => peano.clone(),
        "POOT_AIE_IRON" => iron.clone(),
        _ => None,
    })
}

/// The sequential schedule of the ordinary elementwise-binary request over `dims` (the request the GPU
/// planner builds; only the target's schedule differs).
fn schedule(op: poot_kernel_ir::BinOp, dims: &[u64]) -> poot_kernelgen::SerializedGenerated {
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let operand = || poot_kernelgen::ViewOperand {
        shape: shape.clone(),
        layout: poot_kernelgen::Layout::contiguous(&shape),
    };
    let request =
        poot_kernelgen::KernelRequest::Elementwise(poot_kernelgen::ElementwiseSpec::Binary {
            op: poot_kernelgen::ValueBinary::Basic(op, poot_kernel_ir::Ty::F32),
            out_shape: shape.clone(),
            a: operand(),
            b: operand(),
            fold: poot_kernelgen::Fold {
                numel: shape.iter().product(),
                two_d: false,
                width: 256,
            },
        });
    poot_kernelgen::serialized_schedule(&request, &limits()).unwrap()
}

/// The real sequential body the ordinary request generates over `dims`, and its shape ABI (the bindings
/// and proved access ranges come from the generator).
fn generated(n: u64, dims: &[u64]) -> (poot_kernel_ir::Body, AieShapeAbi) {
    assert_eq!(n, dims.iter().product::<u64>());
    let generated = schedule(poot_kernel_ir::BinOp::Add, dims);
    let bindings = generated
        .bindings
        .iter()
        .enumerate()
        .map(|(index, b)| {
            AieBinding::new(
                index,
                BufferStorage::f32(),
                if b.writes {
                    ArgAccess::Write
                } else {
                    ArgAccess::Read
                },
                &b.dims,
                b.accessed,
            )
            .unwrap()
        })
        .collect();
    (generated.body, AieShapeAbi::new(bindings).unwrap())
}

fn entries(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// A host-side stand-in for the XDNA runtime's load boundary: the checks that must refuse before XRT is
/// reached, and a counter proving whether XRT would have been.
#[derive(Default)]
struct FakeLoader {
    xrt_loads: usize,
}

#[derive(Debug, PartialEq)]
enum LoadRefusal {
    Schema(poot_runtime_common::KernelArgError),
    Bind(AieBindError),
    NotAie,
}

impl FakeLoader {
    fn load(
        &mut self,
        kernel: &poot_runtime_common::CompiledKernel,
        bound: &[AieBound],
    ) -> Result<(), LoadRefusal> {
        let KernelCode::Aie(artifact) = kernel.code() else {
            return Err(LoadRefusal::NotAie);
        };
        let elements: Vec<ElementKind> = bound.iter().map(|b| b.storage.element()).collect();
        check_kernel_args(kernel, &elements).map_err(LoadRefusal::Schema)?;
        check_aie_bindings(artifact, bound).map_err(LoadRefusal::Bind)?;
        self.xrt_loads += 1;
        Ok(())
    }
}

fn bound<'a>(dims: &'a [u64], writable: bool) -> AieBound<'a> {
    AieBound {
        storage: BufferStorage::f32(),
        dims,
        writable,
    }
}

fn linked(dir: &Path, n: u64, dims: &[u64]) -> (poot_kernel_ir::Body, AieLinkedArtifact) {
    let (body, abi) = generated(n, dims);
    let tc = toolchain(
        Some(peano(dir, "aie2p")),
        Some(iron(dir, "npu2", Iron::Links)),
    );
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let compiled = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap();
    (body, compiled.artifact)
}

// ---- SC-001: ABI argument shape/order/storage through the real validator and a fake loader -----------

/// The ordinary handle carries the compiler-derived ABI: three ordered F32 buffers, the last written, the
/// core lengths the loop bound, entry and target from the artifact. Mutation: restore the `AieCore` panic in
/// `aie_kernel_handle` (return the old "no runtime CompiledKernel for AieCore" panic) and this fails.
#[test]
fn the_ordinary_handle_carries_the_compiler_derived_abi() {
    let dir = scratch("handle");
    let (_, artifact) = linked(&dir, 6, &[2, 3]);
    assert_eq!(artifact.abi().core_lengths(), vec![6, 6, 6]);

    let handle = aie_kernel_handle(artifact.clone());
    assert_eq!(handle.target(), TargetId::Aie(AieTargetId::Xdna2));
    assert_eq!(handle.entry_point(), "MLIR_AIE");
    assert!(!handle.has_trap());
    let args: Vec<_> = handle
        .args()
        .iter()
        .map(|a| (a.element(), a.access()))
        .collect();
    assert_eq!(
        args,
        vec![
            (ElementKind::F32, ArgAccess::Read),
            (ElementKind::F32, ArgAccess::Read),
            (ElementKind::F32, ArgAccess::Write)
        ]
    );
    let KernelCode::Aie(carried) = handle.code() else {
        panic!(
            "an AIE handle must carry the linked artifact, got {:?}",
            handle.code()
        );
    };
    assert_eq!(*carried, artifact);
}

/// The fake loader reaches "XRT" only for the exact bound shape; each malformed binding is refused first.
/// Mutation: drop a check in `check_aie_bindings` (the storage, writability or dims comparison); the
/// matching row below admits its malformed fixture and `xrt_loads` becomes non-zero.
#[test]
fn malformed_bindings_are_refused_before_the_loader_reaches_xrt() {
    let dir = scratch("loader");
    let (_, artifact) = linked(&dir, 6, &[2, 3]);
    let handle = aie_kernel_handle(artifact);
    let dims = [2u64, 3];
    let mut loader = FakeLoader::default();

    let good = [bound(&dims, false), bound(&dims, false), bound(&dims, true)];
    assert_eq!(loader.load(&handle, &good), Ok(()));
    assert_eq!(loader.xrt_loads, 1);

    let mut refused = FakeLoader::default();
    // Missing binding.
    assert_eq!(
        refused.load(&handle, &good[..2]),
        Err(LoadRefusal::Schema(
            poot_runtime_common::KernelArgError::Count {
                expected: 3,
                actual: 2
            }
        ))
    );
    // Reordered: a read-only buffer where the kernel writes.
    let reordered = [bound(&dims, true), bound(&dims, false), bound(&dims, false)];
    assert_eq!(
        refused.load(&handle, &reordered),
        Err(LoadRefusal::Bind(AieBindError::ReadOnlyWritten {
            index: 2
        }))
    );
    // [3,2] in place of [2,3]: the same six words, a different shape.
    let transposed = [3u64, 2];
    let wrong_shape = [
        bound(&dims, false),
        bound(&transposed, false),
        bound(&dims, true),
    ];
    assert_eq!(
        refused.load(&handle, &wrong_shape),
        Err(LoadRefusal::Bind(AieBindError::Dims {
            index: 1,
            expected: dims.into(),
            actual: transposed.into(),
        }))
    );
    // F32 mirror: an I32 buffer holds the identical physical words and is still not an F32 binding.
    let mut mirror = good;
    mirror[0].storage = BufferStorage::i32();
    let result = refused.load(&handle, &mirror);
    assert!(
        matches!(
            result,
            Err(LoadRefusal::Schema(_))
                | Err(LoadRefusal::Bind(AieBindError::Storage { index: 0, .. }))
        ),
        "an I32 buffer must not bind to an F32 binding: {result:?}"
    );
    // Packed BF16: a different element width, refused by the schema.
    let mut packed = good;
    packed[1].storage = BufferStorage::bf16_packed();
    assert!(refused.load(&handle, &packed).is_err());
    assert_eq!(refused.xrt_loads, 0, "no malformed binding may reach XRT");
}

/// The artifact validator refuses what cannot be a loadable, self-consistent kernel. Mutation: delete one
/// check in `AieBinding::new` / `AieShapeAbi::new` / `AieLinkedArtifact::new`; its row below stops erroring.
#[test]
fn the_artifact_validator_refuses_inconsistent_abis_and_bundles() {
    let f32w = |dims: &[u64], accessed| {
        AieBinding::new(0, BufferStorage::f32(), ArgAccess::Write, dims, accessed)
    };
    assert_eq!(
        AieBinding::new(0, BufferStorage::bf16_packed(), ArgAccess::Read, &[4], 4),
        Err(AieArtifactError::Storage {
            index: 0,
            storage: BufferStorage::bf16_packed()
        })
    );
    assert_eq!(
        f32w(&[2, 0], 0),
        Err(AieArtifactError::ZeroDimension { index: 0 })
    );
    assert_eq!(
        f32w(&[u64::MAX, 2], 1),
        Err(AieArtifactError::ExtentOverflow { index: 0 })
    );
    assert_eq!(
        f32w(&[4], 5),
        Err(AieArtifactError::AccessOutOfRange {
            index: 0,
            accessed: 5,
            elements: 4
        })
    );
    let read = AieBinding::new(0, BufferStorage::f32(), ArgAccess::Read, &[4], 4).unwrap();
    assert_eq!(
        AieShapeAbi::new(vec![read.clone()]),
        Err(AieArtifactError::NoBindings),
        "a kernel that writes nothing has no output"
    );
    let abi = AieShapeAbi::new(vec![read, f32w(&[4], 4).unwrap()]).unwrap();
    let new = |xclbin: &[u8], insts: &[u8], entry: &str| {
        AieLinkedArtifact::new(
            xclbin.to_vec(),
            insts.to_vec(),
            entry,
            abi.clone(),
            AieTargetId::Xdna2,
            "tc",
            0,
        )
    };
    assert_eq!(new(b"xclbin2\0..", b"IIII", "k").map(|_| ()), Ok(()));
    assert_eq!(
        new(b"nope", b"IIII", "k"),
        Err(AieArtifactError::XclbinMagic)
    );
    assert_eq!(
        new(b"xclbin2\0..", b"III", "k"),
        Err(AieArtifactError::InstructionStream { bytes: 3 })
    );
    assert_eq!(
        new(b"xclbin2\0..", b"", "k"),
        Err(AieArtifactError::InstructionStream { bytes: 0 })
    );
    assert_eq!(
        new(b"xclbin2\0..", b"IIII", ""),
        Err(AieArtifactError::NoEntry)
    );
}

/// A serialized artifact round-trips exactly and re-validates on read: a truncated blob, trailing bytes and
/// a blob built for another target are refused, never loaded. Mutation: skip the target comparison or the
/// trailing-byte check in `from_bytes`.
#[test]
fn a_serialized_artifact_round_trips_and_rejects_corruption() {
    let dir = scratch("bytes");
    let (_, artifact) = linked(&dir, 6, &[2, 3]);
    let bytes = artifact.to_bytes();
    assert_eq!(
        AieLinkedArtifact::from_bytes(&bytes, AieTargetId::Xdna2).unwrap(),
        artifact
    );
    assert!(AieLinkedArtifact::from_bytes(&bytes[..bytes.len() - 1], AieTargetId::Xdna2).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(AieLinkedArtifact::from_bytes(&trailing, AieTargetId::Xdna2).is_err());
    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(AieLinkedArtifact::from_bytes(&bad_magic, AieTargetId::Xdna2).is_err());
}

/// A body and a recipe that do not describe one core never become an artifact: wrong port count, an I32 port
/// over an F32 body, a read port the body writes. The check runs before any tool does.
///
/// Mutation: delete the element or access comparison in `check_tile_body`; its row compiles instead.
#[test]
fn a_body_and_recipe_that_disagree_are_refused_before_any_tool_runs() {
    let dir = scratch("abi");
    let (body, _) = generated(4, &[4]);
    let read = |i, storage| AieBinding::new(i, storage, ArgAccess::Read, &[4], 4).unwrap();
    let write = |i, storage| AieBinding::new(i, storage, ArgAccess::Write, &[4], 4).unwrap();
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let tc = toolchain(
        Some(peano(&dir, "aie2p")),
        Some(iron(&dir, "npu2", Iron::Links)),
    );
    let refusal = |bindings| {
        let abi = AieShapeAbi::new(bindings).unwrap();
        match compile_aie(
            &[&body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &tc,
            &cache,
            LIMIT,
        ) {
            Err(AieCompileError::Abi(mismatch)) => mismatch,
            other => panic!(
                "expected a body/recipe mismatch, got {:?}",
                other.map(|_| ())
            ),
        }
    };
    assert_eq!(
        refusal(vec![
            read(0, BufferStorage::f32()),
            write(1, BufferStorage::f32())
        ]),
        AieAbiMismatch::Count {
            tile: 0,
            body: 3,
            recipe: 2
        }
    );
    assert_eq!(
        refusal(vec![
            read(0, BufferStorage::f32()),
            read(1, BufferStorage::i32()),
            write(2, BufferStorage::f32()),
        ]),
        AieAbiMismatch::Element {
            tile: 0,
            port: 1,
            body: ElementKind::F32,
            fifo: ElementKind::I32
        }
    );
    // A written port the body only reads: the recipe is valid on its own, the body disagrees at port 0.
    assert_eq!(
        refusal(vec![
            write(0, BufferStorage::f32()),
            read(1, BufferStorage::f32()),
            write(2, BufferStorage::f32()),
        ]),
        AieAbiMismatch::Access {
            tile: 0,
            port: 0,
            body_writes: false,
            fifo_writes: true
        }
    );
    assert_nothing_published(&cache);
}

// ---- SC-001: cache identity across shapes and toolchains ------------------------------------------------

fn tool(version: &str, content: u64) -> AieToolIdentity {
    AieToolIdentity {
        path: "/opt/tool".into(),
        version: version.into(),
        content,
    }
}

fn tools(peano: AieToolIdentity, iron: AieToolIdentity) -> AieTools {
    AieTools { peano, iron }
}

fn identity(
    body: &poot_kernel_ir::Body,
    dims: &[u64],
    target: AieTargetId,
    t: &AieTools,
) -> AieIdentity {
    let n = dims.iter().product();
    let (_, abi) = generated(n, dims);
    let recipe = single_tile_recipe(target, &abi);
    AieIdentity::new(&[body], &recipe, t)
}

/// Same Body and the same element count under different dimensions, a different Peano or IRON build (version
/// line or bytes) and a different link ABI revision are all different artifacts; an unchanged schema reuses
/// its identity. Mutation: leave the dimensions out of `abi_canonical` (equal-numel regression), or the IRON
/// identity out of the link text; the matching assertion fails.
#[test]
fn the_cache_identity_sees_shape_toolchain_and_abi() {
    let (body, _) = generated(6, &[2, 3]);
    let base_tools = tools(tool("peano 1", 1), tool("iron 1", 2));
    let t = AieTargetId::Xdna2;
    let base = identity(&body, &[2, 3], t, &base_tools);

    // Unchanged schema: same identity (and independent of any data or address, which it never sees).
    assert_eq!(base, identity(&body, &[2, 3], t, &base_tools));

    let differs = |other: AieIdentity, what: &str| {
        assert_ne!(
            base.link_digest(),
            other.link_digest(),
            "{what} must change the link identity"
        );
    };
    differs(
        identity(&body, &[3, 2], t, &base_tools),
        "dimensions at equal numel",
    );
    differs(identity(&body, &[6], t, &base_tools), "rank at equal numel");
    differs(
        identity(
            &body,
            &[2, 3],
            t,
            &tools(tool("peano 2", 1), tool("iron 1", 2)),
        ),
        "the Peano version line",
    );
    differs(
        identity(
            &body,
            &[2, 3],
            t,
            &tools(tool("peano 1", 9), tool("iron 1", 2)),
        ),
        "the Peano binary's bytes under the same name",
    );
    differs(
        identity(
            &body,
            &[2, 3],
            t,
            &tools(tool("peano 1", 1), tool("iron 2", 2)),
        ),
        "the IRON version line",
    );
    differs(
        identity(
            &body,
            &[2, 3],
            t,
            &tools(tool("peano 1", 1), tool("iron 1", 8)),
        ),
        "the IRON driver's bytes",
    );

    // The Peano build is a core-object input; the IRON build is not.
    let iron_only = identity(
        &body,
        &[2, 3],
        t,
        &tools(tool("peano 1", 1), tool("iron 2", 2)),
    );
    assert_eq!(base.core_digest(), iron_only.core_digest());
    let peano_changed = identity(
        &body,
        &[2, 3],
        t,
        &tools(tool("peano 2", 1), tool("iron 1", 2)),
    );
    assert_ne!(base.core_digest(), peano_changed.core_digest());
}

/// A different Body (the op) is a different core object even at the same shape.
#[test]
fn a_different_body_is_a_different_core_identity() {
    let (add, _) = generated(6, &[2, 3]);
    let mul = schedule(poot_kernel_ir::BinOp::Mul, &[2, 3]).body;
    let t = tools(tool("peano 1", 1), tool("iron 1", 2));
    let a = identity(&add, &[2, 3], AieTargetId::Xdna2, &t);
    let m = identity(&mul, &[2, 3], AieTargetId::Xdna2, &t);
    assert_ne!(a.core_digest(), m.core_digest());
}

/// The shape of a two-tile region `out = (a op0 b) op1 c`: tile 0 reads host tensors `a` and `b` and feeds
/// tile 1 over an on-chip link, tile 1 reads the link and host tensor `c` and writes `out` to the host.
struct Chain2 {
    ops: [poot_kernel_ir::BinOp; 2],
    /// Which host binding each of `a`, `b`, `c` is: the order decides every arena offset.
    binding_of: [usize; 3],
    /// Whether tile 1 reads `(c, link)` rather than `(link, c)`.
    swap_second: bool,
    /// Whether tile 0 and tile 1 swap places in the recipe (the wiring follows them).
    swap_tiles: bool,
}

impl Chain2 {
    fn new(op0: poot_kernel_ir::BinOp, op1: poot_kernel_ir::BinOp) -> Self {
        Self {
            ops: [op0, op1],
            binding_of: [0, 1, 2],
            swap_second: false,
            swap_tiles: false,
        }
    }

    /// The bodies in tile order, and the recipe.
    fn build(&self, dims: &[u64]) -> (Vec<poot_kernel_ir::Body>, AieLinkRecipe) {
        use poot_codegen::{AieFifo, AieTile, FifoEnd};
        let elements: u64 = dims.iter().product();
        let binding = |i: usize, access| {
            AieBinding::new(i, BufferStorage::f32(), access, dims, elements).unwrap()
        };
        let abi = AieShapeAbi::new(vec![
            binding(0, ArgAccess::Read),
            binding(1, ArgAccess::Read),
            binding(2, ArgAccess::Read),
            binding(3, ArgAccess::Write),
        ])
        .unwrap();
        let [a, b, c] = self.binding_of;
        let (t0, t1) = if self.swap_tiles { (1, 0) } else { (0, 1) };
        let fifo = |producer, consumers| AieFifo {
            producer,
            consumers,
            depth: AieLinkRecipe::FIFO_DEPTH,
            elements,
            element: ElementKind::F32,
        };
        let (second_link, second_c) = if self.swap_second { (1, 0) } else { (0, 1) };
        let fifos = vec![
            fifo(
                FifoEnd::Host { binding: a },
                vec![FifoEnd::Tile { tile: t0, port: 0 }],
            ),
            fifo(
                FifoEnd::Host { binding: b },
                vec![FifoEnd::Tile { tile: t0, port: 1 }],
            ),
            fifo(
                FifoEnd::Host { binding: c },
                vec![FifoEnd::Tile {
                    tile: t1,
                    port: second_c,
                }],
            ),
            fifo(
                FifoEnd::Tile { tile: t0, port: 2 },
                vec![FifoEnd::Tile {
                    tile: t1,
                    port: second_link,
                }],
            ),
            fifo(
                FifoEnd::Tile { tile: t1, port: 2 },
                vec![FifoEnd::Host { binding: 3 }],
            ),
        ];
        let first = AieTile {
            ports: vec![0, 1, 3],
        };
        let second = AieTile {
            ports: if self.swap_second {
                vec![2, 3, 4]
            } else {
                vec![3, 2, 4]
            },
        };
        let tiles = if self.swap_tiles {
            vec![second, first]
        } else {
            vec![first, second]
        };
        let bodies = [
            schedule(self.ops[0], dims).body,
            schedule(self.ops[1], dims).body,
        ];
        let bodies = if self.swap_tiles {
            vec![bodies[1].clone(), bodies[0].clone()]
        } else {
            bodies.to_vec()
        };
        (
            bodies,
            AieLinkRecipe::new(AieTargetId::Xdna2, abi, tiles, fifos).unwrap(),
        )
    }

    fn identity(&self, dims: &[u64], t: &AieTools) -> AieIdentity {
        let (bodies, recipe) = self.build(dims);
        let bodies: Vec<&poot_kernel_ir::Body> = bodies.iter().collect();
        AieIdentity::new(&bodies, &recipe, t)
    }
}

/// A region's cache identity sees every equation's body, which tile each runs on, how the tiles are wired
/// and where each tensor sits in the host arenas: a region that differs in any of them is a different
/// artifact, so a reordered chain can never be served another chain's xclbin. The core objects, which are
/// the same set of bodies, keep one core identity under a different placement.
///
/// Mutation: leave the `placed` lines out of the link text in `AieIdentity::new` and the tile and fifo
/// lines out of `AieLinkRecipe::canonical`; the reordered, rewired and re-laid-out rows share the base
/// digest.
#[test]
fn the_region_identity_sees_every_body_the_placement_the_wiring_and_the_offsets() {
    use poot_kernel_ir::BinOp::{Add, Mul, Sub};
    let t = tools(tool("peano 1", 1), tool("iron 1", 2));
    let dims = [2u64, 3];
    let base = Chain2::new(Add, Sub).identity(&dims, &t);
    assert_eq!(base, Chain2::new(Add, Sub).identity(&dims, &t), "stable");

    let differs = |other: AieIdentity, what: &str| {
        assert_ne!(
            base.link_digest(),
            other.link_digest(),
            "{what} must change the link identity"
        );
    };
    differs(
        Chain2::new(Add, Mul).identity(&dims, &t),
        "another body on the second tile",
    );
    differs(
        Chain2::new(Mul, Sub).identity(&dims, &t),
        "another body on the first tile",
    );
    let reordered = Chain2 {
        swap_tiles: true,
        ..Chain2::new(Add, Sub)
    };
    differs(
        reordered.identity(&dims, &t),
        "the same two bodies on swapped tiles",
    );
    assert_eq!(
        base.core_digest(),
        reordered.identity(&dims, &t).core_digest(),
        "the core objects are the same set of bodies wherever they sit"
    );
    differs(
        Chain2 {
            swap_second: true,
            ..Chain2::new(Add, Sub)
        }
        .identity(&dims, &t),
        "the second tile reading its operands the other way round (sub is not commutative)",
    );
    differs(
        Chain2 {
            binding_of: [2, 1, 0],
            ..Chain2::new(Add, Sub)
        }
        .identity(&dims, &t),
        "another host binding order (and so other arena offsets and FIFO ends) for the same tensors",
    );
    differs(
        Chain2::new(Add, Sub).identity(&[3, 2], &t),
        "the dimensions at equal element counts",
    );
}

/// A wiring that is not a region is a typed [`RecipeFault`] at construction, before any tool runs: a FIFO no
/// one reads, a host end that contradicts its binding, a binding streamed twice or never, a tile port wired
/// to something other than the FIFO its tile lists. Mutation: delete one check in `AieLinkRecipe::new`;
/// its row below builds a recipe.
#[test]
fn a_wiring_that_is_not_a_region_is_a_typed_fault_at_construction() {
    use poot_codegen::{AieFifo, AieTile, FifoEnd, RecipeFault};
    let dims = [2u64, 3];
    let binding = |i, access| AieBinding::new(i, BufferStorage::f32(), access, &dims, 6).unwrap();
    let abi = || {
        AieShapeAbi::new(vec![
            binding(0, ArgAccess::Read),
            binding(1, ArgAccess::Write),
        ])
        .unwrap()
    };
    let fifo = |producer, consumers| AieFifo {
        producer,
        consumers,
        depth: 2,
        elements: 6,
        element: ElementKind::F32,
    };
    let host = |binding| FifoEnd::Host { binding };
    let tile = |port| FifoEnd::Tile { tile: 0, port };
    let good_fifos = || vec![fifo(host(0), vec![tile(0)]), fifo(tile(1), vec![host(1)])];
    let one_tile = || vec![AieTile { ports: vec![0, 1] }];
    let build = |tiles: Vec<AieTile>, fifos: Vec<AieFifo>| {
        AieLinkRecipe::new(AieTargetId::Xdna2, abi(), tiles, fifos).map(|_| ())
    };
    let fault = |result: Result<(), AieAbiMismatch>| match result {
        Err(AieAbiMismatch::Recipe(fault)) => fault,
        other => panic!("expected a recipe fault, got {other:?}"),
    };
    assert_eq!(build(one_tile(), good_fifos()), Ok(()), "control");
    assert_eq!(fault(build(vec![], good_fifos())), RecipeFault::NoTiles);
    let mut empty = good_fifos();
    empty[0].depth = 0;
    assert_eq!(
        fault(build(one_tile(), empty)),
        RecipeFault::Empty { fifo: 0 }
    );
    let mut unread = good_fifos();
    unread[1].consumers.clear();
    assert_eq!(
        fault(build(one_tile(), unread)),
        RecipeFault::NoConsumer { fifo: 1 }
    );
    let mut unknown = good_fifos();
    unknown[0].producer = host(7);
    assert_eq!(
        fault(build(one_tile(), unknown)),
        RecipeFault::UnknownBinding {
            fifo: 0,
            binding: 7,
            bindings: 2
        }
    );
    // The host fills a written binding.
    let mut backwards = good_fifos();
    backwards[0].producer = host(1);
    backwards[1].consumers = vec![host(0)];
    assert_eq!(
        fault(build(one_tile(), backwards)),
        RecipeFault::HostDirection {
            fifo: 0,
            binding: 1,
            access: ArgAccess::Write
        }
    );
    let mut sized = good_fifos();
    sized[0].elements = 5;
    assert_eq!(
        fault(build(one_tile(), sized)),
        RecipeFault::HostElements {
            fifo: 0,
            binding: 0,
            elements: 5,
            element: ElementKind::F32
        }
    );
    let mut twice = good_fifos();
    twice.push(fifo(host(0), vec![tile(1)]));
    assert_eq!(
        fault(build(one_tile(), twice)),
        RecipeFault::BindingStreams {
            binding: 0,
            fifos: 2
        }
    );
    let mut never = good_fifos();
    never[1].consumers = vec![FifoEnd::Tile { tile: 0, port: 0 }];
    assert_eq!(
        fault(build(one_tile(), never)),
        RecipeFault::BindingStreams {
            binding: 1,
            fifos: 0
        }
    );
    let mut missing_port = good_fifos();
    missing_port[0].consumers = vec![FifoEnd::Tile { tile: 0, port: 9 }];
    assert_eq!(
        fault(build(one_tile(), missing_port)),
        RecipeFault::UnknownPort {
            fifo: 0,
            tile: 0,
            port: 9
        }
    );
    // The tile lists FIFO 1 for port 0 but FIFO 0 names that port: port 0 is claimed by nobody.
    assert_eq!(
        fault(build(vec![AieTile { ports: vec![1, 0] }], good_fifos())),
        RecipeFault::PortWiring {
            tile: 0,
            port: 0,
            ends: 0
        }
    );
    assert_eq!(
        fault(build(vec![AieTile { ports: vec![0, 5] }], good_fifos())),
        RecipeFault::UnknownFifo {
            tile: 0,
            port: 1,
            fifo: 5
        }
    );
    let mut loopback = good_fifos();
    loopback[0].consumers.push(host(1));
    assert_eq!(
        fault(build(one_tile(), loopback)),
        RecipeFault::HostToHost { fifo: 0 }
    );
}

// ---- SC-003: unavailable toolchain, typed, nothing published ---------------------------------------------

fn assert_nothing_published(cache: &KernelCache) {
    assert_eq!(
        entries(cache.dir()),
        Vec::<String>::new(),
        "a failed AIE compile must leave no entry and no scratch in the cache"
    );
}

/// Each unavailable-toolchain case returns its named typed cause, at its stage, with an empty cache.
/// Mutation: replace a `ToolchainMissing`/`ToolchainUnsupportedTarget` return with a panic, or have the
/// resolver return a fabricated artifact; the `matches!` or the empty-cache assertion fails.
#[test]
fn an_unavailable_toolchain_is_a_typed_stage_cause_and_publishes_nothing() {
    let dir = scratch("toolchain");
    let (body, abi) = generated(6, &[2, 3]);
    let good_peano = peano(&dir, "aie2p");
    let good_iron = iron(&dir, "npu2", Iron::Links);
    let run = |tc: Toolchain, name: &str| {
        let cache = KernelCache::at(dir.join(format!("cache-{name}")), Target::AieCore);
        let result = compile_aie(
            &[&body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &tc,
            &cache,
            LIMIT,
        );
        assert_nothing_published(&cache);
        result.map(|_| ()).unwrap_err()
    };

    // Absent Peano (variable unset), absent IRON, and a configured path that names no file.
    assert!(matches!(
        run(toolchain(None, Some(good_iron.clone())), "no-peano"),
        AieCompileError::ToolchainMissing {
            stage: AieStage::Peano,
            variable: "POOT_AIE_LLC"
        }
    ));
    assert!(matches!(
        run(toolchain(Some(good_peano.clone()), None), "no-iron"),
        AieCompileError::ToolchainMissing {
            stage: AieStage::Iron,
            variable: "POOT_AIE_IRON"
        }
    ));
    assert!(matches!(
        run(
            toolchain(
                Some("/nonexistent/peano-llc".into()),
                Some(good_iron.clone())
            ),
            "bad-path"
        ),
        AieCompileError::ToolchainMissing {
            stage: AieStage::Peano,
            ..
        }
    ));
    // A Peano that lacks the `aie2p` backend (a system llc), and an IRON without the platform.
    assert!(matches!(
        run(
            toolchain(Some(peano(&dir, "x86 nvptx")), Some(good_iron.clone())),
            "no-aie2p"
        ),
        AieCompileError::ToolchainUnsupportedTarget {
            stage: AieStage::Peano,
            ..
        }
    ));
    assert!(matches!(
        run(
            toolchain(
                Some(good_peano.clone()),
                Some(iron(&dir, "npu1", Iron::Links))
            ),
            "no-platform"
        ),
        AieCompileError::ToolchainUnsupportedTarget {
            stage: AieStage::Iron,
            ..
        }
    ));
}

/// A link that fails after its first output (or writes garbage, or no second output) leaves no cache entry,
/// no scratch and no handle: the bundle is validated whole before one atomic publish.
#[test]
fn a_failed_link_leaves_no_half_published_bundle() {
    let dir = scratch("halflink");
    let (body, abi) = generated(6, &[2, 3]);
    let peano = peano(&dir, "aie2p");
    for (name, mode) in [
        ("fails", Iron::FailsAfterFirstOutput),
        ("no-insts", Iron::WritesNoInstructions),
        ("garbage", Iron::WritesGarbage),
    ] {
        let tc = toolchain(Some(peano.clone()), Some(iron(&dir, "npu2", mode)));
        let cache = KernelCache::at(dir.join(format!("cache-{name}")), Target::AieCore);
        let error = compile_aie(
            &[&body],
            &single_tile_recipe(AieTargetId::Xdna2, &abi),
            &tc,
            &cache,
            LIMIT,
        )
        .map(|_| ())
        .unwrap_err();
        match (name, &error) {
            (
                "fails",
                AieCompileError::Tool {
                    stage: AieStage::Iron,
                    stderr,
                    ..
                },
            ) => {
                assert!(
                    stderr.contains("boom"),
                    "the driver's stderr is the cause: {stderr:?}"
                );
            }
            (
                "no-insts",
                AieCompileError::LinkOutput {
                    output: "instruction stream",
                },
            ) => {}
            ("garbage", AieCompileError::Artifact(AieArtifactError::XclbinMagic)) => {}
            other => panic!("unexpected cause for {name}: {other:?}"),
        }
        assert_nothing_published(&cache);
    }
}

/// With both tools present the pipeline publishes one validated entry; a second request for the same
/// schema is a verified hit (`built == false`), and a different shape is a different entry.
#[test]
fn a_linked_artifact_is_published_once_and_served_by_identity() {
    let dir = scratch("publish");
    let tc = toolchain(
        Some(peano(&dir, "aie2p")),
        Some(iron(&dir, "npu2", Iron::Links)),
    );
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let (body, abi) = generated(6, &[2, 3]);
    let first = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap();
    assert!(first.built);
    assert_eq!(entries(cache.dir()).len(), 1);
    let second = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap();
    assert!(!second.built, "an unchanged schema is a cache hit");
    assert_eq!(second.artifact, first.artifact);

    let (body_t, abi_t) = generated(6, &[3, 2]);
    let other = compile_aie(
        &[&body_t],
        &single_tile_recipe(AieTargetId::Xdna2, &abi_t),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap();
    assert!(
        other.built,
        "equal element count, different dimensions: a different artifact"
    );
    assert_eq!(entries(cache.dir()).len(), 2);
    assert_eq!(other.artifact.abi().bindings()[0].dims(), &[3, 2]);
}

/// The one row only a real toolchain can answer (SC-002): Peano compiles the ordinary sequential body and
/// the IRON driver links it, and the ordinary handle carries the artifact. Ignored, not skipped: a box
/// without `POOT_AIE_LLC` and `POOT_AIE_IRON` reports it as ignored and the row stays unverified. Run it with
/// `nix develop .#npu`, by exact name and `--run-ignored ignored-only`.
#[test]
#[ignore = "needs a real Peano llc (POOT_AIE_LLC) and the IRON link driver (POOT_AIE_IRON); unverified without them"]
fn a_real_peano_and_iron_toolchain_links_a_loadable_artifact() {
    let tc = Toolchain::from_env();
    let dir = std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap()
        .join(".cache/poot-codegen-test/aie-npu-artifact");
    let cache = KernelCache::at(dir, Target::AieCore);
    let (body, abi) = generated(6, &[2, 3]);
    let compiled = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap_or_else(|e| panic!("real AIE compile failed: {e}"));
    let handle = aie_kernel_handle(compiled.artifact);
    assert_eq!(handle.args().len(), 3);
}

/// A binding whose storage is not the canonical F32/I32 word buffer cannot enter an ABI: the serialized form
/// records only the element kind, so a non-canonical logical dtype (here I8 carried in I32 words) would
/// round-trip as a different ABI. Mutation: relax the exact-storage comparison in `AieBinding::new`; the
/// I8-in-I32 binding is admitted and `Storage` is not returned.
#[test]
fn a_non_canonical_logical_dtype_is_refused_not_silently_rewritten() {
    let i8_words = BufferStorage::dense(ElementKind::I32, LogicalDType::I8);
    assert_eq!(
        AieBinding::new(0, i8_words, ArgAccess::Read, &[4], 4),
        Err(AieArtifactError::Storage {
            index: 0,
            storage: i8_words
        })
    );
    // The canonical I32 binding round-trips with its storage intact.
    let dir = scratch("dtype");
    let (_, artifact) = linked(&dir, 4, &[4]);
    let abi = AieShapeAbi::new(vec![
        AieBinding::new(0, BufferStorage::i32(), ArgAccess::Read, &[4], 4).unwrap(),
        AieBinding::new(1, BufferStorage::f32(), ArgAccess::Write, &[4], 4).unwrap(),
    ])
    .unwrap();
    let with_i32 = AieLinkedArtifact::new(
        artifact.xclbin().to_vec(),
        artifact.instructions().to_vec(),
        artifact.entry(),
        abi,
        AieTargetId::Xdna2,
        artifact.toolchain(),
        0,
    )
    .unwrap();
    let round = AieLinkedArtifact::from_bytes(&with_i32.to_bytes(), AieTargetId::Xdna2).unwrap();
    assert_eq!(round, with_i32);
    assert_eq!(round.abi().bindings()[0].storage(), BufferStorage::i32());
}

/// A cache entry under this request's identity that describes another shape (a stale file, or a 64-bit
/// digest collision) is refused with a typed cause, never returned. Mutation: delete the post-load ABI
/// comparison in `compile_aie`; the stale entry is served and the `StaleEntry` assertion fails.
#[test]
fn a_stale_cache_entry_for_another_shape_is_refused() {
    let dir = scratch("stale");
    let tc = toolchain(
        Some(peano(&dir, "aie2p")),
        Some(iron(&dir, "npu2", Iron::Links)),
    );
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let (body, abi) = generated(6, &[2, 3]);
    let first = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .unwrap();

    // The artifact of a different shape, planted under the [2,3] request's key.
    let (_, other_abi) = generated(6, &[3, 2]);
    let other = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &other_abi),
        &tc,
        &KernelCache::at(dir.join("other"), Target::AieCore),
        LIMIT,
    )
    .unwrap();
    cache
        .store(&first.identity.cache_key(), &other.artifact.to_bytes())
        .unwrap();

    let error = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .map(|_| ())
    .unwrap_err();
    assert!(
        matches!(error, AieCompileError::StaleEntry { what: "shape ABI" }),
        "got {error:?}"
    );
}

/// A core object that does not name the body's symbol, or a body with no symbol to name, is a typed
/// `CoreObject` refusal and publishes nothing. Mutation: drop the `is_empty()` guard in
/// `build_object_plan`; the body compiles under the symbol its fingerprint alone names and the object is
/// accepted.
#[test]
fn a_body_with_no_symbol_is_a_typed_core_object_error() {
    let dir = scratch("emptysym");
    let tc = toolchain(
        Some(peano(&dir, "aie2p")),
        Some(iron(&dir, "npu2", Iron::Links)),
    );
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let (mut body, abi) = generated(6, &[2, 3]);
    body.name.clear();
    let error = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .map(|_| ())
    .unwrap_err();
    assert!(
        matches!(&error, AieCompileError::CoreObject { symbol } if symbol.is_empty()),
        "got {error:?}"
    );
    assert_nothing_published(&cache);
}

/// A body that can trap is refused with the typed `Trap` cause, and nothing is published: the AIE ABI has no
/// error word, so a failed bounds check could not be reported (POOT-1120). This is what happened to the
/// MIR-imported sequential kernels (`pootc`'s former `*_lowers_to_aie2p` rows): rustc's slice indexing
/// leaves an `Assert`, which imports as a `Trap`, and the typed schedule takes only trap-free bodies such
/// as the generator's. Mutation: delete the `has_trap()` guard in `check_tile_body`; the compile links and
/// the `Trap` assertion fails.
#[test]
fn a_body_that_can_trap_is_refused_before_any_tool_runs_a_link() {
    let dir = scratch("trap");
    let tc = toolchain(
        Some(peano(&dir, "aie2p")),
        Some(iron(&dir, "npu2", Iron::Links)),
    );
    let cache = KernelCache::at(dir.join("cache"), Target::AieCore);
    let (mut body, abi) = generated(6, &[2, 3]);
    let last = body.blocks.last_mut().unwrap();
    last.terminator = poot_kernel_ir::Terminator::Trap { code: 1 };
    assert!(body.has_trap());
    let error = compile_aie(
        &[&body],
        &single_tile_recipe(AieTargetId::Xdna2, &abi),
        &tc,
        &cache,
        LIMIT,
    )
    .map(|_| ())
    .unwrap_err();
    assert!(
        matches!(
            error,
            AieCompileError::Abi(AieAbiMismatch::Trap { tile: 0 })
        ),
        "got {error:?}"
    );
    assert_nothing_published(&cache);
}
