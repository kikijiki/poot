//! poot-codegen: lower a [`poot_kernel_ir::Body`] to target code. The body is emitted as target-shaped
//! textual LLVM IR, then the selected toolchain produces SPIR-V for wgpu/raw Vulkan, PTX for NVIDIA,
//! HSACO for AMDGPU/ROCm, or an AIE core object for AMD XDNA2. [`Target`] names all four paths; individual
//! IR operations can have narrower target support.
//!
//! The rustc-bearing MIR import (Stable MIR -> Body) is the separate `pootc` crate; this crate is the
//! rustc-free emit + llc half, host-testable with hand-built fixtures.
//!
//! Card 537 (ADR-0104 decision 5): every `env::var`/`env::var_os` read in this crate (`Toolchain`'s
//! `POOT_LLC`/`POOT_AIE_LLC`/`POOT_AMD_CLANG`/`POOT_ROCM_DEVICE_LIBS`, and `cache.rs`'s
//! `POOT_KERNEL_CACHE`/`POOT_KERNEL_CACHE_DIR`) resolves a toolchain binary's path or the kernel-cache's
//! location/kill-switch - never what a kernel computes. `compile`'s output for a given `Body` and
//! [`Target`] does not depend on any of them (a different `POOT_LLC` selects a different `llc` binary, not
//! a different compiled artifact for the same input and toolchain version; `epoch` folds the resolved tool
//! and its version into the cache key precisely so a toolchain change cannot silently serve a stale
//! artifact). Kept per the Scope exception for a read that is a toolchain path or a cache kill-switch.

mod cache;
mod debranch;
mod emit;
mod handle;
mod spirv_postprocess;
mod structurize;

use std::path::Path;
use std::process::Command;

pub use cache::{
    Artifact, ArtifactTooLarge, KernelCache, KernelKey, body_fingerprint,
    epoch as kernel_cache_epoch, kernel_cache_root, kernel_entry_key,
};
pub use emit::emit_llvm_ir_marked;
pub use handle::kernel_handle;
use poot_target::AmdArch;

/// A codegen backend target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    /// SPIR-V for Vulkan/wgpu (triple `spirv-unknown-vulkan1.3-compute`, carried in the IR text).
    SpirvVulkan,
    /// PTX for NVIDIA/cudarc (triple `nvptx64-nvidia-cuda`).
    Nvptx,
    /// AIE core machine code for the AMD XDNA2 NPU, via Peano (`llc --march=aie2p`). One AIE core runs a
    /// scalar loop kernel (no SPMD thread-index, no workgroup barrier, no LDS); the IRON harness tiles the
    /// data and does host<->tile movement. The IR is plain scalar LLVM IR with flat `ptr` args (card 089 /
    /// spec 057). The triple/datalayout come from Peano via `--march`, not the IR text.
    AieCore,
    /// HSACO for an AMD GPU (e.g. the Radeon 8060S, gfx1151) via the ROCm-flavored clang.
    /// `llc -filetype=obj` produces a relocatable without the `PT_AMDGPU_HSA_LOAD_*` program headers the
    /// HSA runtime's alt loader needs; `clang -target amdgcn-amd-amdhsa -mcpu=<gfx>` produces a loadable
    /// HSACO directly. No HIP/clr; the runtime talks to the device through HSA.
    ///
    /// Carries an [`AmdArch`] descriptor (gfx code + wavefront + tensor-core family, defined in
    /// `poot-target`, card 522 review) so codegen follows the detected device (spec 120). The payload is
    /// `Copy`, so `Target` stays `Copy`.
    AmdGcn(AmdArch),
}

/// What this target's codegen can lower today, a query the planner reads instead of restating a
/// per-backend match by hand (card 522, R468-014, R468-015, R469-022).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Capability {
    /// A strided (non-row-major) value operand can be read directly by a kernel body, with no
    /// materialize copy first (spec 132's `compute_views`). The AIE core has no view-capable kernel
    /// bodies today (R469-022): strided reads there still materialize. NVPTX's kernel bodies for every
    /// `is_strided_capable` op kind (`poot-graph-plan/src/predicates.rs`) are the same backend-neutral
    /// `poot-kernelgen` bodies wgpu/ROCm already read strided operands through, compiled for
    /// `Target::Nvptx` like any other kernel; `false` here just meant PTX paid an extra materialize copy
    /// for no codegen reason (R-546-9, card 549).
    pub strided_views: bool,
    /// The imported tiled-GEMM bodies (`MatMul`/`MatMulBias`, card 044) compile for this target. Their
    /// Rust-compiled LDS arrays carry a zeroinitializer in addrspace(3), which AMDGPU LLVM rejects
    /// (R468-015); AmdGcn takes the synthesized tiled-GEMM body (`kg::tiled_region`) instead, which
    /// uses an uninitialized `WorkgroupLocalDecl`.
    pub imported_lds_array_bodies: bool,
}

/// The capability query for `target`. See [`Capability`]'s fields for what each one means and why it
/// differs by target.
pub fn capability(target: Target) -> Capability {
    match target {
        Target::SpirvVulkan => Capability {
            strided_views: true,
            imported_lds_array_bodies: true,
        },
        Target::AmdGcn(_) => Capability {
            strided_views: true,
            imported_lds_array_bodies: false,
        },
        Target::Nvptx => Capability {
            strided_views: true,
            imported_lds_array_bodies: true,
        },
        Target::AieCore => Capability {
            strided_views: false,
            imported_lds_array_bodies: false,
        },
    }
}

impl Target {
    /// The `llc` arguments for this target (the triple for SPIR-V is carried in the IR `target triple`).
    fn llc_args(&self) -> Vec<String> {
        match self {
            // sm_80 (Ampere): bf16 tensor cores (wmma m16n16k16 bf16) need sm_80+ (spec 025). The driver
            // JITs sm_80 PTX for every pod GPU poot targets (RTX 3090/A5000 = sm_86, 4090 = sm_89).
            Target::Nvptx => {
                vec!["-mtriple=nvptx64-nvidia-cuda".into(), "-mcpu=sm_80".into()]
            }
            Target::SpirvVulkan => vec![
                "-O0".into(),
                "-filetype=obj".into(),
                // `--spirv-ext` does not accumulate across repeated occurrences (a later flag replaces the
                // earlier one; seen as workgroup_local_atomic_compiles_to_both_backends failing "requires
                // SPV_EXT_shader_atomic_float_add"), so all extensions ride one comma-separated flag. They
                // are inert unless the IR uses them: float atomics (min_max covers emit_rvalue's
                // AtomicOp::Min/Max lowering to atomicrmw fmin/fmax), and
                // SPV_KHR_cooperative_matrix (card 154) for `OpTypeCooperativeMatrixKHR` /
                // `OpCooperativeMatrix{Load,Store,MulAdd}KHR` in the SPIR-V WMMA path (RADV coopmat
                // config 13: F16xF16->F32).
                "--spirv-ext=+SPV_EXT_shader_atomic_float_add,+SPV_EXT_shader_atomic_float_min_max,\
                 +SPV_KHR_cooperative_matrix"
                    .into(),
            ],
            // Peano (llvm-aie) lowers scalar LLVM IR to the AIE2p core ISA. The IRON harness links the `.o`
            // by C symbol via `external_func(..., link_with=...)`. aie2p Peano is reported -O0-stable-only
            // (llvm-aie #315), but the scalar kernel lowers cleanly at -O1 (spec 057 open question). `llc`
            // here is the Peano binary (`$POOT_AIE_LLC` / `$POOT_LLC`), not the system llc.
            Target::AieCore => vec![
                "--march=aie2p".into(),
                "-O1".into(),
                "--filetype=obj".into(),
            ],
            // AMDGPU: the AMD-flavored clang (`rocmPackages.rocm-toolchain`) with `-target
            // amdgcn-amd-amdhsa` and `-mcpu=<gfx>`; `-x ir` takes the textual IR from `emit_llvm_ir`. `-shared`
            // runs lld in the same invocation, which resolves R_AMDGPU_REL64 (wiring the kernel descriptor's
            // `kernel_code_entry_byte_offset` to the .text load address) and adds the `PT_AMDGPU_HSA_LOAD_*`
            // program headers the HSA alt loader needs. Bare `llc -filetype=obj` or `clang -c` yields a
            // relocatable that fails to load with HSA_STATUS_ERROR_INVALID_CODE_OBJECT. This is the flow
            // `hipcc` / `amdclang` use. Device libs (ockl, oclc_*) are auto-linked via
            // `--rocm-device-lib-path=<amdgcn/bitcode>` (see run_llc).
            Target::AmdGcn(arch) => vec![
                // -O2 is the standard opt level for HSACO.
                "-target".into(),
                "amdgcn-amd-amdhsa".into(),
                format!("-mcpu={}", arch.mcpu()),
                "-O2".into(),
                "-x".into(),
                "ir".into(),
                "-shared".into(),
            ],
        }
    }
    fn output_ext(&self) -> &'static str {
        match self {
            Target::Nvptx => "ptx",
            Target::SpirvVulkan => "spv",
            Target::AieCore => "o",
            // ELF (`e_machine = EM_AMDGPU`); `.hsaco` distinguishes it from the AIE `.o`.
            Target::AmdGcn(_) => "hsaco",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// The body failed [`poot_kernel_ir::Body::verify`]: it is malformed, not merely unsupported on this target.
    #[error(transparent)]
    InvalidBody(#[from] poot_kernel_ir::VerifyError),
    #[error("unsupported in kernel codegen: {0}")]
    Unsupported(String),
    /// Control-flow structurization failed: an irreducible CFG, or a fixpoint that blew its block budget.
    /// Its own class rather than `Unsupported`, so a body the pass cannot structure reads as a structurizer
    /// failure and not as an unimplemented feature (card 461).
    #[error("control-flow structurization: {0}")]
    Structurize(String),
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error(transparent)]
    Emit(#[from] EmitError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The toolchain that produces a loadable object for `target` is not configured. AMDGCN needs the AMD clang:
    /// a bare `llc` object has no HSA load headers, so the loader rejects it after `compile` would have reported
    /// success (R468-019).
    #[error(
        "{target:?} codegen needs ${variable} (the AMD-flavored clang; the devshell sets it); a bare llc \
         object is rejected by the HSA loader"
    )]
    ToolchainMissing {
        target: Target,
        variable: &'static str,
    },
    #[error("llc failed for {target:?} (status {status}):\n{stderr}")]
    Llc {
        target: Target,
        status: String,
        stderr: String,
    },
    /// `spirv_postprocess::fix_no_contraction` could not place every `Rvalue::BinaryOpNoContract` marker
    /// (card 628): an emitter/postprocess ordinal drift, not a malformed body (that fails at `body.verify()`
    /// earlier) - loud rather than a silently-unprotected compiled kernel.
    #[error("SPIR-V no-contraction decoration: {0}")]
    NoContraction(#[from] spirv_postprocess::NoContractionError),
    /// The compiled artifact is longer than the caller's `max_artifact_bytes`; nothing was loaded or
    /// published.
    #[error(transparent)]
    ArtifactTooLarge(#[from] cache::ArtifactTooLarge),
}

/// Emit + `llc` a body to a target artifact written at `out`. Returns the emitted LLVM IR (for debugging
/// and golden tests). The tools come from `POOT_LLC` (SPIR-V, PTX), `POOT_AIE_LLC` (AIE core), and
/// `POOT_AMD_CLANG` with `POOT_ROCM_DEVICE_LIBS` (AMDGCN, which has no fallback).
pub fn compile(
    body: &poot_kernel_ir::Body,
    target: Target,
    out: &Path,
) -> Result<String, CompileError> {
    compile_with(body, target, out, &Toolchain::from_env())
}

/// The external tools [`compile`] shells out to, resolved from the environment once per compile.
struct Toolchain {
    /// `$POOT_LLC` or `llc` on PATH: lowers SPIR-V and PTX.
    llc: String,
    /// `$POOT_AIE_LLC` (Peano, a separate llc fork from the system LLVM), else `llc`.
    aie_llc: String,
    /// `$POOT_AMD_CLANG`: the AMD-flavored clang that links the HSACO and adds the `PT_AMDGPU_HSA_LOAD_*` program
    /// headers the HSA loader needs. There is no fallback: a bare llc object is rejected at load with
    /// `HSA_STATUS_ERROR_INVALID_CODE_OBJECT` (card 106b).
    amd_clang: Option<String>,
    /// `$POOT_ROCM_DEVICE_LIBS`: the `amdgcn/bitcode` directory holding ockl and the oclc libraries.
    rocm_device_libs: Option<String>,
}

impl Toolchain {
    fn from_env() -> Self {
        Self::from_env_with(|name| std::env::var(name).ok())
    }

    /// The toolchain `lookup` describes: the environment lookup is injected so tests can present an environment
    /// without mutating the process's.
    fn from_env_with(lookup: impl Fn(&str) -> Option<String>) -> Self {
        let llc = lookup("POOT_LLC").unwrap_or_else(|| "llc".to_string());
        Toolchain {
            aie_llc: lookup("POOT_AIE_LLC").unwrap_or_else(|| llc.clone()),
            llc,
            amd_clang: lookup("POOT_AMD_CLANG"),
            rocm_device_libs: lookup("POOT_ROCM_DEVICE_LIBS"),
        }
    }

    /// The one binary that lowers `target`, or [`CompileError::ToolchainMissing`] when it is not configured.
    fn binary(&self, target: Target) -> Result<&str, CompileError> {
        match target {
            // AIE core uses Peano (llvm-aie), a separate llc fork from the system LLVM that lowers SPIR-V/PTX
            // (a manylinux binary run via an FHS wrapper on NixOS).
            Target::AieCore => Ok(&self.aie_llc),
            Target::AmdGcn(_) => self
                .amd_clang
                .as_deref()
                .ok_or(CompileError::ToolchainMissing {
                    target,
                    variable: "POOT_AMD_CLANG",
                }),
            Target::Nvptx | Target::SpirvVulkan => Ok(&self.llc),
        }
    }
}

fn compile_with(
    body: &poot_kernel_ir::Body,
    target: Target,
    out: &Path,
    toolchain: &Toolchain,
) -> Result<String, CompileError> {
    let (ir, no_contract_ordinals) = emit::emit_llvm_ir_marked(body, target)?;

    // The `.ll` scratch and the object llc writes must be unique across all concurrent compiles in this
    // process. `out` is content-addressed (same `body`+arch -> same `out`), so two threads compiling the
    // same body would otherwise race on a half-written intermediate/object and publish a corrupt artifact
    // (CUDA_ERROR_INVALID_IMAGE / HSA_STATUS_ERROR_INVALID_CODE_OBJECT). Tensor-parallel serving hits this,
    // with per-rank threads compiling the same kernels at once. So write unique per-compile scratch, then
    // atomically rename onto `out`. Identical bodies give identical bytes, so the last writer winning is
    // safe.
    static COMPILE_NONCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let nonce = COMPILE_NONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let (ll_path, tmp_out) = compile_scratch_paths(out, std::process::id(), nonce);

    // Resolve the tool before writing anything, so a missing toolchain leaves no scratch behind.
    let tool = tool_command(target, toolchain, &ll_path, &tmp_out)?;
    std::fs::write(&ll_path, &ir)?;
    let llc_res = run_tool(tool, target);
    // Always drop the .ll scratch so it does not accumulate.
    let _ = std::fs::remove_file(&ll_path);
    if let Err(e) = llc_res {
        let _ = std::fs::remove_file(&tmp_out);
        return Err(e);
    }
    if target == Target::SpirvVulkan
        && let Err(e) = fixup_spirv_barriers(&tmp_out, &no_contract_ordinals)
    {
        let _ = std::fs::remove_file(&tmp_out);
        return Err(e);
    }
    // Atomic publish onto the content-addressed path (same directory, so the rename is atomic).
    std::fs::rename(&tmp_out, out)?;
    Ok(ir)
}

/// Derive unique per-compile scratch paths (the `.ll` and the temp object llc writes) next to the
/// content-addressed `out`. `out`'s file name is kept as a prefix; the `{pid}.{nonce}` suffix makes them
/// unique across concurrent compiles (see the nonce in [`compile`]). Both live in `out`'s parent so the
/// final rename is same-filesystem and atomic. Pure, so uniqueness is unit-testable without a toolchain.
fn compile_scratch_paths(
    out: &Path,
    pid: u32,
    nonce: u64,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let stem = out
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "kernel".to_string());
    let uniq = format!("{stem}.{pid}.{nonce}");
    (
        parent.join(format!("{uniq}.ll")),
        parent.join(format!("{uniq}.tmp")),
    )
}

/// Read the emitted `.spv`, apply the [`spirv_postprocess`] fixes, and write it back: workgroup-barrier
/// memory semantics (SequentiallyConsistent -> AcquireRelease | WorkgroupMemory, as Vulkan requires),
/// `Statement::Fence` `OpMemoryBarrier` semantics (the storage-class bit LLVM's generic `fence` lowering
/// omits; spec 138 Phase 2), the coopmat builtin-call rewrite (card 154: LLVM's Vulkan-mode
/// `__spirv_CooperativeMatrix{Load,Store,MulAdd}KHR`/`__spirv_CompositeConstruct` stub calls become real
/// instructions), and the `NoContraction` decoration (card 628: `no_contract_ordinals`, from
/// `emit::emit_llvm_ir_marked`, names which compiled float binops `Rvalue::BinaryOpNoContract` marked - see
/// `spirv_postprocess::fix_no_contraction`). No-op for a module with none of these.
fn fixup_spirv_barriers(
    spv_path: &Path,
    no_contract_ordinals: &[usize],
) -> Result<(), CompileError> {
    let bytes = std::fs::read(spv_path)?;
    // SPIR-V is a stream of little-endian 32-bit words.
    if bytes.len() % 4 != 0 {
        return Ok(()); // not a word-aligned SPIR-V blob; leave it
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let fixed = spirv_postprocess::fix_barrier_semantics(words);
    let fixed = spirv_postprocess::fix_fence_semantics(fixed);
    let fixed = spirv_postprocess::fix_coopmat_calls(fixed);
    let fixed = spirv_postprocess::fix_no_contraction(fixed, no_contract_ordinals)?;
    let mut out = Vec::with_capacity(fixed.len() * 4);
    for w in fixed {
        out.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::write(spv_path, out)?;
    Ok(())
}

/// The command that lowers `ll` to `out` for `target`. AMDGCN has exactly one toolchain: the AMD-flavored clang
/// (`$POOT_AMD_CLANG`), which calls llc and adds the `PT_AMDGPU_HSA_LOAD_*` program headers. A bare llc object
/// fails to load with HSA_STATUS_ERROR_INVALID_CODE_OBJECT (card 106b), so a missing clang is an error and not a
/// silent fallback to an object that cannot run.
fn tool_command(
    target: Target,
    toolchain: &Toolchain,
    ll: &Path,
    out: &Path,
) -> Result<Command, CompileError> {
    let mut cmd = Command::new(toolchain.binary(target)?);
    cmd.args(target.llc_args());
    // AMDGCN: auto-link device libs via `--rocm-device-lib-path=<amdgcn/bitcode>` (POOT_ROCM_DEVICE_LIBS, set by
    // the flake's shellHook). Not `--rocm-path`, which expects the parent of `amdgcn/`.
    if let (Target::AmdGcn(_), Some(libs)) = (target, &toolchain.rocm_device_libs) {
        cmd.arg(format!("--rocm-device-lib-path={libs}"));
    }
    cmd.arg(ll).arg("-o").arg(out);
    Ok(cmd)
}

fn run_tool(mut tool: Command, target: Target) -> Result<(), CompileError> {
    let output = tool.output()?;
    if !output.status.success() {
        return Err(CompileError::Llc {
            target,
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        });
    }
    Ok(())
}

/// Convenience: a default artifact path next to `dir` named `<kernel>.<ext>`.
pub fn artifact_path(dir: &Path, kernel: &str, target: Target) -> std::path::PathBuf {
    dir.join(format!("{kernel}.{}", target.output_ext()))
}

#[cfg(test)]
mod arch_tests {
    use super::*;

    // `parse_gfx`/tensor-core classification now live in `poot-target` (card 522 review), with their own
    // tests there; this module keeps only what's codegen's own behavior.

    #[test]
    fn arch_threads_mcpu_into_llc_args() {
        // FR-001: the `-mcpu` string follows the threaded arch, not a constant.
        let arch = AmdArch::from_isa_name("amdgcn-amd-amdhsa--gfx1100", 32).unwrap();
        assert_eq!(arch.mcpu(), "gfx1100");
        assert_eq!(arch.wave, 32);
        let args = Target::AmdGcn(arch).llc_args();
        assert!(
            args.iter().any(|a| a == "-mcpu=gfx1100"),
            "llc args must carry -mcpu=gfx1100, got {args:?}"
        );
        assert!(
            args.iter().all(|a| a != "-mcpu=gfx1151"),
            "no hardcoded gfx1151 must leak: {args:?}"
        );
    }

    /// R468-019: without the AMD-flavored clang, AMDGCN `compile` ran a bare llc and reported success for an
    /// object the HSA loader rejects. It must be a typed error, with nothing written.
    #[test]
    fn amdgcn_compile_without_the_amd_clang_is_a_typed_toolchain_error() {
        let dir = std::env::temp_dir().join(format!("poot-amd-toolchain-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let target = Target::AmdGcn(AmdArch::gfx1151());
        let out = artifact_path(&dir, "add", target);
        // An environment with no variables set at all: the seam that maps the environment into the toolchain
        // decides that AMDGCN has no tool.
        let toolchain = Toolchain::from_env_with(|_| None);
        let result = compile_with(
            &poot_kernel_ir::fixtures::add_kernel(),
            target,
            &out,
            &toolchain,
        );
        let Err(err) = result else {
            panic!("compile succeeded without POOT_AMD_CLANG")
        };
        let CompileError::ToolchainMissing {
            target: t,
            variable,
        } = &err
        else {
            panic!("expected ToolchainMissing, got {err:?}")
        };
        assert_eq!((*t, *variable), (target, "POOT_AMD_CLANG"));
        let message = err.to_string();
        assert!(
            !message.contains('\n') && !message.contains('\\'),
            "the message must read as one plain sentence: {message:?}"
        );
        assert!(message.contains("$POOT_AMD_CLANG"), "{message}");
        assert!(
            !out.exists(),
            "an object was published without the toolchain"
        );
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            0,
            "scratch left behind"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// The environment seam maps each variable to its tool: `POOT_AMD_CLANG` selects the AMDGCN binary (and only
    /// that), and SPIR-V/PTX/AIE keep their llc defaults.
    #[test]
    fn toolchain_maps_the_environment_to_one_binary_per_target() {
        let env = |name: &str| match name {
            "POOT_AMD_CLANG" => Some("/opt/amd/clang".to_string()),
            "POOT_AIE_LLC" => Some("/opt/peano/llc".to_string()),
            _ => None,
        };
        let toolchain = Toolchain::from_env_with(env);
        let binary = |target| toolchain.binary(target).unwrap().to_string();
        assert_eq!(binary(Target::AmdGcn(AmdArch::gfx1151())), "/opt/amd/clang");
        assert_eq!(binary(Target::AieCore), "/opt/peano/llc");
        assert_eq!(binary(Target::SpirvVulkan), "llc");
        assert_eq!(binary(Target::Nvptx), "llc");
    }

    #[test]
    fn from_isa_name_rejects_empty() {
        assert!(AmdArch::from_isa_name("", 32).is_err());
    }

    #[test]
    fn scratch_paths_are_unique_per_nonce_and_distinct_from_out() {
        // Card 227: two compiles of the same content-addressed `out` must get different scratch, or they
        // race on a shared `.ll`/temp.
        let out = Path::new("/tmp/poot/k0123456789abcdef.hsaco");
        let (ll_a, tmp_a) = compile_scratch_paths(out, 4242, 0);
        let (ll_b, tmp_b) = compile_scratch_paths(out, 4242, 1);
        // Same nonce -> deterministic (a retry reuses its own scratch).
        assert_eq!(
            compile_scratch_paths(out, 4242, 0),
            (ll_a.clone(), tmp_a.clone())
        );
        // Different nonce -> every path differs, so concurrent compiles never collide.
        assert_ne!(ll_a, ll_b);
        assert_ne!(tmp_a, tmp_b);
        // The `.ll` and temp object of a single compile also differ from each other.
        assert_ne!(ll_a, tmp_a);
        // Neither scratch path equals the shared `out` (the atomic-rename target).
        for p in [&ll_a, &tmp_a, &ll_b, &tmp_b] {
            assert_ne!(
                *p,
                out.to_path_buf(),
                "scratch path aliased the shared output"
            );
        }
        // Scratch lives next to `out` so the publish rename is same-filesystem (atomic).
        assert_eq!(ll_a.parent(), out.parent());
        assert_eq!(tmp_a.parent(), out.parent());
        // `out`'s file name is kept as a prefix.
        assert!(
            ll_a.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("k0123456789abcdef.hsaco"),
            "scratch should keep the output name as a prefix: {ll_a:?}"
        );
    }

    #[test]
    fn scratch_paths_handle_an_out_with_no_parent() {
        // A bare filename (no parent dir) yields scratch in the current dir.
        let (ll, tmp) = compile_scratch_paths(Path::new("kernel.ptx"), 7, 3);
        assert_eq!(ll, Path::new("kernel.ptx.7.3.ll"));
        assert_eq!(tmp, Path::new("kernel.ptx.7.3.tmp"));
    }
}
