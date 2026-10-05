//! [`DeviceCaps`]: the device-capability descriptor the runtime measures once and the planner and
//! codegen read (target-architecture.md section 2, card 522). A leaf crate: nothing here depends on a
//! compiler or runtime crate; every other poot crate may depend on it, never the reverse
//! ([[sound-architecture-over-convenience]]).
//!
//! [`Backend`] is the one lowering-backend enum every crate reads (moved here from `poot-graph-plan`,
//! card 522 review): most of the planner is backend-neutral; the NVPTX-only kernels (tensor-core gemm,
//! spec 025) and the AMD `AmdArch`-carrying arm are gated on it. [`AmdArch`] is the AMD GPU codegen
//! target (gfx code, wavefront, tensor-core family), derived at runtime from the HSA-reported ISA name.
//! What the planner needs to make a device rule - the buffer limit, LDS size, grid caps, watchdog
//! budget, tensor-core support and known miscompiles - is a [`DeviceCaps`] value, filled once per real
//! device by that device's runtime crate (`Context::device_caps()`) and threaded by the caller to every
//! planner entry point (card 522); a caller with no measured device yet (a test, or the
//! plan-summary corpus) uses a documented fixture default instead.
//!
//! [`storage`]: a device buffer's storage contract (element kind, logical dtype, layout), shared by
//! every runtime's buffer handle and by the planner's per-value storage record (card 527).

pub mod storage;
pub use storage::{BufferStorage, ElementKind, LogicalDType, StorageLayout};

/// The lowering backend. Most of `plan_eqn` is backend-neutral; the NVPTX-only kernels (tensor-core
/// gemm, spec 025) are gated on this so the SPIR-V/wgpu path keeps the portable kernel.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    Nvptx,
    SpirvVulkan,
    /// ROCm/AMDGPU executor. Carries the per-device arch (gfx code, wavefront, tensor-core family) so
    /// `plan_eqn` can gate the WMMA path on `TensorCoreSupport::Wmma16x16x16Rdna3` (spec 130).
    AmdGcn(AmdArch),
}

/// Matrix (tensor-core) hardware a device exposes, as measured from a real capability query - never
/// parsed from a marketing device-name string (card 522 review: a descriptor that says "no tensor
/// cores" on a WMMA-capable part because the name string carried a codename instead of a gfx/config
/// token is wrong data, not a conservative default).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum TensorCoreSupport {
    /// Queried and confirmed absent: no matrix hardware the planner's WMMA/MFMA path targets.
    #[default]
    None,
    /// RDNA3 / RDNA3.5 (gfx11xx / a matching 16x16x16 f16-in f32-out cooperative-matrix config):
    /// WMMA `16x16x16`, wave32.
    Wmma16x16x16Rdna3,
    /// RDNA4 (gfx12xx): WMMA with a different fragment packing. Layout not implemented.
    Rdna4,
    /// CDNA (gfx9xx MI-series): MFMA matrix cores. Not implemented.
    CdnaMfma,
    /// NVIDIA compute capability 8.0 or newer: the `wmma` `16x16x16` bf16/f16-in f32-out fragments the
    /// NVPTX tensor-core kernels use, 32-lane warps. Classified from the device's own compute
    /// capability ([`TensorCoreSupport::from_cuda_compute_capability`]).
    NvidiaWmma16x16x16Sm80,
    /// This API gives no way to query matrix hardware on this backend (distinct from `None`, which is a
    /// confirmed absence): a wgpu backend other than Vulkan, or any future caller that has not wired a
    /// real query yet. Never used to mean "not present" - a caller that cannot tell must say so, not
    /// guess.
    UnknownNotExposedByApi,
}

impl TensorCoreSupport {
    /// Classify an AMD gfx code (e.g. `"gfx1151"`), from an HSA ISA name query
    /// ([`AmdArch::from_isa_name`]). Unknown codes map to `None` so an unrecognized device does not
    /// silently take a matrix-core path.
    pub fn from_amd_gfx_code(gfx: &str) -> Self {
        let n = gfx.strip_prefix("gfx").unwrap_or(gfx);
        if n.starts_with("11") {
            TensorCoreSupport::Wmma16x16x16Rdna3
        } else if n.starts_with("12") {
            TensorCoreSupport::Rdna4
        } else if matches!(n, "908" | "90a" | "940" | "941" | "942") {
            TensorCoreSupport::CdnaMfma
        } else {
            TensorCoreSupport::None
        }
    }

    /// Classify a CUDA device by its compute capability major version (`CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`).
    /// poot's NVPTX tensor-core kernels use bf16 `wmma` fragments, which exist from sm_80; an older
    /// device is a confirmed `None` for those kernels.
    pub fn from_cuda_compute_capability(major: u32) -> Self {
        if major >= 8 {
            TensorCoreSupport::NvidiaWmma16x16x16Sm80
        } else {
            TensorCoreSupport::None
        }
    }

    /// Classify one reported Vulkan `VK_KHR_cooperative_matrix` configuration (or wgpu's
    /// `CooperativeMatrixProperties`, the same data): `m`/`n`/`k` are the matrix-multiply dimensions,
    /// `ab_f16` is whether the A/B operand type is 16-bit float, and `result_f32` is whether the
    /// accumulator/result type is 32-bit float. Matches RDNA3's one emitted WMMA shape (16x16x16,
    /// f16-in f32-out, card 110); a device may report several configs; fold with
    /// [`TensorCoreSupport::most_specific`] to combine them. Every other shape/type combination the
    /// extension can report (RDNA4's, CDNA's, NVIDIA's, Apple's) is not yet classified here, so it
    /// returns `UnknownNotExposedByApi` for this one config (card 522): the extension reported
    /// a real config the device exposes, it is only this classifier that cannot name its family, which
    /// is a different fact from a device whose reported-config list is empty (a genuinely confirmed
    /// absence, which `most_specific`'s fold over an empty list leaves at `None`). Read the fold of every
    /// reported config, not one in isolation.
    pub fn from_cooperative_matrix_config(
        m: u32,
        n: u32,
        k: u32,
        ab_f16: bool,
        result_f32: bool,
    ) -> Self {
        if m == 16 && n == 16 && k == 16 && ab_f16 && result_f32 {
            TensorCoreSupport::Wmma16x16x16Rdna3
        } else {
            TensorCoreSupport::UnknownNotExposedByApi
        }
    }

    /// Fold two classifications of the same device (e.g. one per reported cooperative-matrix config)
    /// into one: a confirmed family wins over `None`, and `UnknownNotExposedByApi` only survives if
    /// nothing else ever did.
    pub fn most_specific(self, other: Self) -> Self {
        match (self, other) {
            (TensorCoreSupport::None | TensorCoreSupport::UnknownNotExposedByApi, other) => other,
            (this, _) => this,
        }
    }

    /// Find and classify a `gfx<code>` token anywhere in a device identity string, case-insensitively:
    /// an HSA ISA name (`"amdgcn-amd-amdhsa--gfx1151"`). `None` if the string carries no such token.
    /// Kept for identity strings that genuinely carry the gfx code (unlike Mesa RADV's wgpu/Vulkan
    /// device name, which reports a marketing codename instead - see `from_cooperative_matrix_config`
    /// for the measured query that backend needs).
    pub fn from_device_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        let Some(start) = lower.find("gfx") else {
            return TensorCoreSupport::None;
        };
        let rest = &lower[start..];
        let end = rest[3..]
            .find(|c: char| !c.is_ascii_alphanumeric())
            .map(|i| i + 3)
            .unwrap_or(rest.len());
        TensorCoreSupport::from_amd_gfx_code(&rest[..end])
    }
}

/// A short, `Copy` ASCII holder for an AMD gfx code (e.g. `"gfx1151"`); the inline buffer keeps
/// [`AmdArch`] `Copy`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GfxCode {
    buf: [u8; 16],
    len: u8,
}

impl GfxCode {
    /// Panics if `s` exceeds the 16-byte inline buffer (no real gfx code is that long).
    pub fn new(s: &str) -> Self {
        let bytes = s.as_bytes();
        assert!(bytes.len() <= 16, "gfx code too long: {s:?}");
        let mut buf = [0u8; 16];
        buf[..bytes.len()].copy_from_slice(bytes);
        GfxCode {
            buf,
            len: bytes.len() as u8,
        }
    }
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len as usize]).unwrap_or("")
    }
}

impl std::fmt::Debug for GfxCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// An AMD GPU codegen target: the `-mcpu` gfx code plus wavefront size and tensor-core family. Derived
/// at runtime from the HSA-reported ISA name and wavefront size ([`AmdArch::from_isa_name`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AmdArch {
    gfx: GfxCode,
    /// Wavefront size (32 or 64), from the runtime-detected value.
    pub wave: u32,
    /// Matrix-core family, classified from the gfx code.
    pub tensor_core: TensorCoreSupport,
}

/// Error classifying a runtime ISA name into an [`AmdArch`].
#[derive(Debug, thiserror::Error)]
pub enum ArchError {
    #[error("could not find a gfx code in AMD ISA name {0:?}")]
    NoGfxCode(String),
}

impl AmdArch {
    /// The gfx1151 (Strix Halo, Radeon 8060S) descriptor: RDNA3.5, wave32.
    pub fn gfx1151() -> Self {
        AmdArch {
            gfx: GfxCode::new("gfx1151"),
            wave: 32,
            tensor_core: TensorCoreSupport::Wmma16x16x16Rdna3,
        }
    }

    /// The `-mcpu` string, e.g. `"gfx1151"`.
    pub fn mcpu(&self) -> &str {
        self.gfx.as_str()
    }

    /// Build a descriptor from an explicit gfx code and wavefront size; the tensor-core family is
    /// classified from the gfx code.
    pub fn new(gfx: &str, wave: u32) -> Self {
        AmdArch {
            gfx: GfxCode::new(gfx),
            wave,
            tensor_core: TensorCoreSupport::from_amd_gfx_code(gfx),
        }
    }

    /// Derive a descriptor from an HSA-reported ISA name (`"amdgcn-amd-amdhsa--gfx1151"` or a bare
    /// `"gfx1151"`) plus the wavefront size. Returns [`ArchError::NoGfxCode`] if no `gfx<...>` token is
    /// present.
    pub fn from_isa_name(isa: &str, wave: u32) -> Result<Self, ArchError> {
        let gfx = parse_gfx(isa).ok_or_else(|| ArchError::NoGfxCode(isa.to_string()))?;
        Ok(AmdArch::new(gfx, wave))
    }
}

/// Extract the `gfx<digits><suffix>` token from an ISA name (`amdgcn-amd-amdhsa--gfx1151` ->
/// `gfx1151`). The suffix can be a letter (`gfx90a`) or feature flags after a `:` (stripped).
fn parse_gfx(isa: &str) -> Option<&str> {
    let start = isa.find("gfx")?;
    let rest = &isa[start..];
    // `gfx` + alphanumerics; stop at the first non-alphanumeric (`:`, `-`, end).
    let end = rest[3..]
        .find(|c: char| !c.is_ascii_alphanumeric())
        .map(|i| i + 3)
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

/// A device property the runtime either measured (or took from a documented API guarantee) or whose API
/// does not expose it. `Unknown` is never a guess: a validator that needs the property refuses a plan
/// it cannot prove rather than assume a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Queried<T> {
    Known(T),
    /// This API gives no way to read the property on this device.
    Unknown,
}

/// Whether a device runs subgroup (warp/wavefront) collectives, and the lane counts one can have.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SubgroupSupport {
    /// Queried and confirmed absent.
    Absent,
    /// A subgroup has between `min_size` and `max_size` lanes (equal where the size is fixed).
    Present { min_size: u32, max_size: u32 },
}

/// The iGPU display watchdog can kill a dispatch that runs too long (Strix Halo RADV, card 163); no
/// other backend in this workspace has one. Each field is the per-dispatch work budget (in the unit the
/// op family already chunks on) that family reads to decide whether to split into several dispatches.
/// `None` on [`DeviceCaps::watchdog_budget`] means this device has no such budget.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WatchdogBudget {
    /// Output elements for an unchunked decode-GEMV dispatch (card 258).
    pub decode_gemv_out_elems: u64,
    /// `out_elems * K` for an unchunked serial K-quant dequant matmul (card 163).
    pub serial_dequant_work: u64,
    /// Output elements for an unchunked scatter-update / dynamic-update-slice dispatch (card 159).
    pub scatter_update_work: u64,
    /// `out_elems * K` for an unchunked indexed (MoE) dequant matmul.
    pub indexed_dequant_work: u64,
}

/// Hardware/driver defects the planner must route around by construction. `None` means "not known to be
/// present on this device"; a runtime records only a defect it has confirmed for its device family, and
/// a device that has never been checked keeps `None` rather than guessing.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct KnownMiscompiles {
    /// RADV: the LDS-cooperative tiled GEMM produces wrong, non-deterministic output at or above this
    /// many workgroups in one dispatch (card 095). `None` on a device with no confirmed ceiling.
    pub tiled_gemm_max_workgroups: Option<u32>,
}

/// A device-capability descriptor: measured once by the runtime, read by the planner and codegen for
/// every device rule that used to be a hard-coded per-backend constant (card 522, R469-006, R468-014).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DeviceCaps {
    /// The largest single buffer allocation this device and its driver are safe to use, in bytes. Not
    /// always the raw device maximum: a driver can wedge below its own advertised limit (RADV; see
    /// `poot-runtime`'s `AMD_RADV_SAFE_MAX_BUFFER_BYTES`).
    pub max_buffer_bytes: u64,
    /// Workgroup-local (LDS/shared-memory) budget, in bytes, one dispatch may use.
    pub lds_bytes: u32,
    /// The largest workgroup extent per dimension (`[x, y, z]`, in invocations).
    pub max_workgroup_size: [u32; 3],
    /// The largest workgroup, in invocations (`x * y * z`), which can be below the product of the
    /// per-dimension limits.
    pub max_workgroup_invocations: u32,
    /// Subgroup (warp/wavefront) support, as the device reports it.
    pub subgroup: Queried<SubgroupSupport>,
    /// The largest conservative work (threads times serial steps per thread) one dispatch may carry
    /// before it risks a watchdog reset; `Unknown` where no bound is calibrated for the device. A single
    /// dispatch is bounded by this, separately from how dispatches batch into submits
    /// (`Dispatch::work`).
    pub max_dispatch_work: Queried<u64>,
    /// The per-dimension dispatch grid ceiling (`gridDim.{x,y,z}` or the HSA/CUDA equivalent).
    pub max_grid: [u32; 3],
    /// The iGPU display-watchdog chunk budgets, or `None` on a device with no such watchdog.
    pub watchdog_budget: Option<WatchdogBudget>,
    /// Matrix (tensor-core) hardware this device exposes to the planner.
    pub tensor_core: TensorCoreSupport,
    /// Confirmed hardware/driver defects this device is known to have.
    pub known_miscompiles: KnownMiscompiles,
    /// Compute units (AMD CUs, NVIDIA SMs): how many workgroups the device runs side by side. Probed
    /// where the backend exposes it; the planner's Gemv launch policy sizes its grid
    /// against it (at least two workgroups per compute unit).
    pub compute_units: u32,
}

/// Max head dim the imported flash-decode and flash-prefill kernels support: both size their LDS output
/// scratch `o[D]` to this many f32 lanes. One owner, read by `poot-graph-plan`'s attention kernel
/// choice (Card 557) so no second copy can desync (card 522; previously a literal mirrored by hand in
/// each crate). A larger head dim stays on the imported / NVPTX kernel.
pub const FLASH_LDS_CAP: usize = 256;

/// Max head dim the synthesized flash-prefill kernel supports. Its LDS `o[D]` scratch sizes dynamically
/// to `d` (no fixed array), unlike decode's D>256 fallback (NVPTX-only), so it can go higher; verified
/// correct at D=512 on wgpu/RADV (card 185, max_abs 5.96e-8). Independent of [`FLASH_LDS_CAP`]: raising
/// this must not raise decode's cap.
pub const FLASH_PREFILL_LDS_CAP: usize = 512;

/// Card 258: output-element count (one `gemv_lds` workgroup per element) above which a wgpu decode-GEMV
/// splits into several smaller dispatches on a device with this box's measured watchdog budget (see
/// [`DeviceCaps::wgpu_rdna3_igpu`]). One definition (card 522 review): `poot-graph-plan`'s
/// `decode_gemv_plan` reads it through `caps.watchdog_budget`, and `dtype_widen`'s bf16 decode-GEMV
/// eligibility gate imports this constant directly for its own chunk-trigger check (card 546b threaded
/// `DeviceCaps.max_grid` into that gate's separate grid-cap check, but this trigger stays a plain
/// constant both sides read), so the two can never independently drift.
///
/// A single dispatch at BLOOM's scale (`N=250880, K=1024`) completes in ~0.6s alone, but the same
/// dispatch right after another large one (a materializing `transpose` of the same weight, ~3.7s) trips
/// the Strix Halo iGPU's display-watchdog TDR within ~1.5s, even though the wgpu executor polls after
/// every submit (the pre-contract `GpuExecutor::run` did this; the executor contract's `WgpuDevice`
/// does the same). The trigger is the total busy stretch across a rapid sequence of large dispatches,
/// not one dispatch's duration. Splitting gives display work scheduling gaps (see
/// `docs/tasks/done/258-decode-gemv-watchdog-chunking-gap.md` for the calibration table).
///
/// `N` is the dominant hazard axis: BLOOM `N=250880` hangs; gpt-oss `N=201088` (tied, `K=32`) and
/// SmolLM3 `N=128256` (tied, `K=2048`, higher `out_numel*K` than BLOOM's `2.57e8`) do not. So the trigger
/// is element-count-only (unlike the K-scaled `serial_dequant_work`), set at 220_000 between the
/// proven-safe 201088 and the proven-hanging 250880.
pub const DECODE_GEMV_CHUNK_TRIGGER: u64 = 220_000;

/// Strix Halo's (gfx1151, Radeon 8060S) compute units, as HSA reports them: the RDNA3 default for
/// [`DeviceCaps::wgpu_rdna3_igpu`] (wgpu has no compute-unit query) and [`DeviceCaps::rocm_default`]
/// (a live ROCm context reports its own agent's count).
pub const STRIX_HALO_COMPUTE_UNITS: u32 = 40;

/// The AMDGPU backend's flat workgroup ceiling (LLVM `amdgpu-flat-work-group-size` maximum, every gfx9+
/// part): the documented per-dimension and total workgroup limit [`DeviceCaps::rocm_default`] and a live
/// ROCm context report, since HSA's workgroup-size query returns the same value on these parts.
pub const AMDGPU_MAX_WORKGROUP_INVOCATIONS: u32 = 1024;

/// An A100's (sm_80, the NVPTX codegen target) streaming multiprocessors: [`DeviceCaps::ptx_default`]'s
/// value (a live PTX context reports its own device's count).
pub const SM80_A100_COMPUTE_UNITS: u32 = 108;

impl DeviceCaps {
    /// The values measured on this box's wgpu/RADV device (Strix Halo, gfx1151) as of card 522: the
    /// display-watchdog budgets cards 163/258/159 calibrated, and the card-095 tiled-GEMM RADV
    /// miscompile ceiling. Used as the default for [`Backend::SpirvVulkan`] until a real per-call
    /// `DeviceCaps` is threaded through every entry point (Card 532), and as a synthetic fixture in
    /// tests and the plan-summary corpus (card 599).
    pub fn wgpu_rdna3_igpu() -> Self {
        DeviceCaps {
            max_buffer_bytes: 1_342_177_280, // poot-runtime::AMD_RADV_SAFE_MAX_BUFFER_BYTES (1.25 GiB)
            lds_bytes: 16_352,               // wgpu downlevel_defaults() workgroup-storage limit
            // The wgpu runtime raises x and the invocation count to the adapter maximum (RADV: 1024);
            // y and z stay at `downlevel_defaults()`.
            max_workgroup_size: [1024, 256, 64],
            max_workgroup_invocations: 1024,
            // RDNA3 runs 32- or 64-lane waves; wgpu reports the adapter's range.
            subgroup: Queried::Known(SubgroupSupport::Present {
                min_size: 32,
                max_size: 64,
            }),
            max_dispatch_work: Queried::Unknown,
            max_grid: [65_535, 65_535, 65_535],
            watchdog_budget: Some(WatchdogBudget {
                decode_gemv_out_elems: DECODE_GEMV_CHUNK_TRIGGER,
                serial_dequant_work: 400_000_000,
                scatter_update_work: 1_000_000,
                indexed_dequant_work: 800_000_000,
            }),
            tensor_core: TensorCoreSupport::Wmma16x16x16Rdna3,
            known_miscompiles: KnownMiscompiles {
                tiled_gemm_max_workgroups: Some(1 << 15),
            },
            compute_units: STRIX_HALO_COMPUTE_UNITS,
        }
    }

    /// A ROCm/HSA device with no confirmed display watchdog or tiled-GEMM ceiling: HSA queues
    /// dispatches on hardware with no host display-refresh TDR, and the RADV miscompile is a wgpu/RADV
    /// codegen-path defect this backend does not share. `tensor_core` and `lds_bytes` should be filled
    /// per device from the HSA agent query where one exists; this is the fallback for what it does not
    /// expose.
    pub fn rocm_default() -> Self {
        DeviceCaps {
            max_buffer_bytes: u64::MAX,
            lds_bytes: 65_536, // 64 KiB LDS per CU, every RDNA/CDNA generation to date
            max_workgroup_size: [AMDGPU_MAX_WORKGROUP_INVOCATIONS; 3],
            max_workgroup_invocations: AMDGPU_MAX_WORKGROUP_INVOCATIONS,
            // The wavefront size is per device (32 or 64): only a live agent query knows it.
            subgroup: Queried::Unknown,
            max_dispatch_work: Queried::Unknown,
            max_grid: [u32::MAX, u32::MAX, u32::MAX],
            watchdog_budget: None,
            tensor_core: TensorCoreSupport::None,
            known_miscompiles: KnownMiscompiles::default(),
            compute_units: STRIX_HALO_COMPUTE_UNITS,
        }
    }

    /// A PTX/CUDA device: no display watchdog, no confirmed miscompile ceiling.
    pub fn ptx_default() -> Self {
        DeviceCaps {
            max_buffer_bytes: u64::MAX,
            lds_bytes: 49_152, // sm_80's default (non-opt-in) shared-memory-per-block ceiling
            // CUDA's documented compute capability 8.0 limits.
            max_workgroup_size: [1024, 1024, 64],
            max_workgroup_invocations: 1024,
            subgroup: Queried::Known(SubgroupSupport::Present {
                min_size: 32,
                max_size: 32,
            }),
            max_dispatch_work: Queried::Unknown,
            max_grid: [u32::MAX, 65_535, 65_535],
            watchdog_budget: None,
            tensor_core: TensorCoreSupport::NvidiaWmma16x16x16Sm80,
            known_miscompiles: KnownMiscompiles::default(),
            compute_units: SM80_A100_COMPUTE_UNITS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_name_classification_finds_the_gfx_token_in_an_hsa_isa_string() {
        assert_eq!(
            TensorCoreSupport::from_device_name("amdgcn-amd-amdhsa--gfx1151"),
            TensorCoreSupport::Wmma16x16x16Rdna3
        );
        assert_eq!(
            TensorCoreSupport::from_device_name("NVIDIA GeForce RTX 4090"),
            TensorCoreSupport::None
        );
    }

    #[test]
    fn amd_gfx_code_classification_matches_known_families() {
        assert_eq!(
            TensorCoreSupport::from_amd_gfx_code("gfx1151"),
            TensorCoreSupport::Wmma16x16x16Rdna3
        );
        assert_eq!(
            TensorCoreSupport::from_amd_gfx_code("gfx1201"),
            TensorCoreSupport::Rdna4
        );
        assert_eq!(
            TensorCoreSupport::from_amd_gfx_code("gfx90a"),
            TensorCoreSupport::CdnaMfma
        );
        assert_eq!(
            TensorCoreSupport::from_amd_gfx_code("gfx900"),
            TensorCoreSupport::None
        );
    }

    #[test]
    fn cooperative_matrix_config_classifies_only_the_rdna3_wmma_shape() {
        assert_eq!(
            TensorCoreSupport::from_cooperative_matrix_config(16, 16, 16, true, true),
            TensorCoreSupport::Wmma16x16x16Rdna3
        );
        // An unclassified but present config reports "we don't know", never "confirmed absent"
        // (card 522): a device reporting only these configs still has real matrix
        // hardware, just not one this classifier names.
        assert_eq!(
            TensorCoreSupport::from_cooperative_matrix_config(8, 8, 8, false, true),
            TensorCoreSupport::UnknownNotExposedByApi
        );
        assert_eq!(
            TensorCoreSupport::from_cooperative_matrix_config(16, 16, 16, false, true),
            TensorCoreSupport::UnknownNotExposedByApi
        );
    }

    #[test]
    fn most_specific_prefers_a_confirmed_family_over_none_or_unknown() {
        assert_eq!(
            TensorCoreSupport::None.most_specific(TensorCoreSupport::Wmma16x16x16Rdna3),
            TensorCoreSupport::Wmma16x16x16Rdna3
        );
        assert_eq!(
            TensorCoreSupport::UnknownNotExposedByApi.most_specific(TensorCoreSupport::Rdna4),
            TensorCoreSupport::Rdna4
        );
        assert_eq!(
            TensorCoreSupport::Wmma16x16x16Rdna3.most_specific(TensorCoreSupport::None),
            TensorCoreSupport::Wmma16x16x16Rdna3
        );
    }

    #[test]
    fn wgpu_default_has_a_watchdog_budget_and_rocm_default_does_not() {
        assert!(DeviceCaps::wgpu_rdna3_igpu().watchdog_budget.is_some());
        assert!(DeviceCaps::rocm_default().watchdog_budget.is_none());
        assert!(DeviceCaps::ptx_default().watchdog_budget.is_none());
    }
}
