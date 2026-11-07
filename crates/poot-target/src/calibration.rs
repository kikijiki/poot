//! Calibration: the figures no driver API reports, measured once on a named device and keyed by that device's
//! identity. A device nobody measured is [`Queried::Unknown`], never given a neighbour's figures: the
//! display-watchdog budgets and the tiled-GEMM ceiling below were observed on one part (Strix Halo behind
//! RADV), and copying them to every AMD GPU chunked work a discrete card runs unsplit and fenced it with a
//! ceiling nobody observed on it. A performance figure ([`PerfCalibration`]) only ranks schedules that are all
//! correct, so a device nobody measured plans with one named [`PerfCalibration::PORTABLE`] entry and the plan
//! records that it did ([`Calibrated`]).

use crate::{
    AMD_RADV_SAFE_MAX_BUFFER_BYTES, ATTENTION_CHUNK_WORK, DeviceCaps, KnownMiscompiles,
    PACKED_TILED_CHUNK_WORK, Queried, RegisterBudget, SubgroupSupport, WatchdogBudget,
};

/// The lane-grid edge of the narrow register-tiled schedule: 64 lanes, one wave64, the widest workgroup every
/// device is known to run.
pub const REGISTER_LANE_EDGE: u32 = 8;

/// The lane-grid edge of the wide register-tiled schedule: 256 lanes, offered to the cost model on a device whose
/// workgroup and LDS hold it.
pub const REGISTER_LANE_EDGE_WIDE: u32 = 16;

/// The packed decode Gemv's longest lane run on a device nobody measured: the run every device launched before
/// POOT-1398. [`PerfCalibration::PORTABLE`]'s figure.
pub const PORTABLE_PACKED_GEMV_RUN: u32 = 8;

/// The packed decode Gemv's longest lane run on Strix Halo (gfx1151, through HSA or RADV), measured (POOT-1390):
/// a 32-element run (one Q4_K/Q5_K sub-block) amortises the block addressing and the factor decode over
/// four times the weights of the portable run and keeps the body within the register file on both
/// compilers.
pub const STRIX_HALO_PACKED_GEMV_RUN: u32 = 32;

/// The packed decode Gemv's longest lane run on the RTX PRO 6000 Blackwell Server (sm_120), the one NVIDIA part it
/// was measured on (POOT-1405). With the activation read in 16-byte groups and 32-bit index math, a 16-element run
/// amortises a K-quant's block decode over twice the weights of the 8-element one, while a 32-element run spreads
/// each warp-wide activation load over 16 cache lines and loses (Q4_K 980 at run 16 against 665 at 32 and 721 at
/// 8 GB/s). Another NVIDIA part is unmeasured and plans with [`PerfCalibration::PORTABLE`] until a pod sweep says
/// otherwise.
pub const SM_120_PACKED_GEMV_RUN: u32 = 16;

/// The lanes of one wave64, the widest wave poot targets: the grain of a tiled attention's workgroup width, which
/// is whole waves of the widest wave so it runs on every device.
pub(crate) const WAVE_LANES: u32 = 64;

/// The Gemv columns of a dense `[N, K]` weight on Strix Halo (card 727): `cols` 4 while that still launches
/// [`FILL_WORKGROUPS_PER_UNIT`] workgroups per compute unit, else 2. Set from the card's sweep on gfx1151 over the
/// qwen2.5-0.5b BF16 decode shapes (wgpu Gemv device time per token: `cols` 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9,
/// 4 4.7; ROCm decode 78 to 81 tok/s from `cols` 16 to 2). Nearly flat between 4 and 8, so not derivable.
pub const STRIX_HALO_DENSE_GEMV_COLS: GemvCols = GemvCols { wide: 4, narrow: 2 };

/// The Gemv columns of a packed weight on Strix Halo (card 653): `{16, 8}` with 8-element runs was 3.7 ms of Gemv
/// device time per Q4_K_M token against 4.2 ms for the dense table and 9.7 ms for the one-column-per-workgroup
/// body (gfx1151, qwen2.5-0.5b GGUF projection shapes).
pub const STRIX_HALO_PACKED_GEMV_COLS: GemvCols = GemvCols {
    wide: 16,
    narrow: 8,
};

/// The width of the LDS-tree Gemv on Strix Halo behind RADV (Qwen2.5-0.5B decode, fused ms per token: w=32 50.9,
/// 64 53.7, 128 39.8, 256 45.6). Non-monotone, so a measurement, not a derivation.
pub const STRIX_HALO_LDS_GEMV_WIDTH: u32 = 128;

/// The cost model's ratios of a device nobody identified them on, and of Strix Halo behind RADV: the POOT-1403 fit
/// of the LDS read and the staged value (every tiling of a qwen3-0.6b prefill at 512 and 2048 tokens, ROCm), which
/// repeats within about 6 % on ROCm but not behind RADV (POOT-1494: the staged ratio moved 38 % between two runs and
/// 1.5-2x with the corpus), and the Gemv's direct value, which the POOT-1489 microbench measured behind RADV: the one
/// value at which the model's Gemv-versus-grouped choices are never slower than the retired 32-row switch, 7.6.
pub const PORTABLE_COST_RATIOS: CostRatios = CostRatios {
    lds_read: 1.0,
    staged_value: 8.0,
    direct_value: 7.6,
};

/// The cost model's ratios on Strix Halo through HSA (ROCm), identified by `calibrate-launch` over every candidate
/// tiling of 8 shapes (POOT-1494, two runs of 80 rows: LDS read 10.2 and 9.6, staged value 8.6 and 8.2, repeating
/// within about 6 %), refitted over the 160 rows together: LDS read 9.88, staged value 8.43, R^2 0.93. The direct
/// value is the POOT-1489 microbench's range in which the model's Gemv-versus-grouped choices are never slower than
/// the retired 32-row switch (11.9 to 23.6), cut at the top by the dense decode Gemv: above 15.2 the model puts a
/// one-row contraction of a vocabulary-sized output onto a matrix unit, which is slower (POOT-1503). The centre of
/// 11.9 to 15.2.
pub const STRIX_HALO_HSA_COST_RATIOS: CostRatios = CostRatios {
    lds_read: 9.88,
    staged_value: 8.43,
    direct_value: 13.4,
};

/// The Gemv's direct value on the RTX PRO 6000 Blackwell Server (sm_120), identified by `calibrate-launch indexed` on
/// that part (POOT-1490): the centre of the range in which the model's Gemv-versus-grouped choices are never slower
/// than the retired 32-row switch, 11.9 to 16.8. It is not [`PORTABLE_COST_RATIOS`]'s 7.6, which lies below the range.
pub const SM_120_DIRECT_VALUE_COST: f64 = 14.15;

/// The keys per staged block of the tiled attention schedule on Strix Halo, widest first. A 64-key block was
/// measured 20-30% slower than a 32-key one on ROCm and wgpu and no faster on PTX (POOT-1402 bank sweep).
pub const STRIX_HALO_ATTENTION_KEY_BLOCKS: [usize; 2] = [32, 16];

/// The resident lanes per compute unit the register-tiled schedule's occupancy model treats as saturating it:
/// eight 32-lane waves, four per SIMD of an RDNA compute unit (two SIMD32s) or two per partition of an NVIDIA SM
/// (four), the residency at which a unit hides its LDS and memory latency behind other waves' FMAs. Structural,
/// not measured per device.
pub const SATURATING_LANES: u32 = 256;

/// The workgroups per compute unit a launch aims at: a unit holds two resident workgroups, so one workgroup's tail
/// overlaps the next one's start. Structural, the one definition of the Gemv column rule, the split-KV count and
/// the selected-head grouping (the factor vLLM's `wvSplitK` grid and Marlin's `tiles * 4 <= sms` use).
pub const FILL_WORKGROUPS_PER_UNIT: u32 = 2;

/// The bytes of the widest load every backend emits (`dwordx4`, `ld.v4`): the contiguous activation bytes a Gemv
/// lane reads per trip. Structural.
pub const VECTOR_LOAD_BYTES: u32 = 16;

/// The output columns a Gemv workgroup owns: `wide` while the outputs still fill the device
/// ([`FILL_WORKGROUPS_PER_UNIT`] workgroups per compute unit), `narrow` otherwise. A weight row is contiguous
/// along K, so fewer columns give each row more lanes and a longer contiguous run per trip. Measured per weight
/// family: the best pair depends on the body's trip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemvCols {
    /// Columns while the outputs fill the device.
    pub wide: u32,
    /// Columns for a small output count, so the workgroups do not starve the device.
    pub narrow: u32,
}

/// The ratios of the contraction cost model, in FMA issues (the unit): what one lane's LDS read, one staged value
/// (a global load, its LDS write and its share of the round's two barriers) and one value a Gemv lane reads straight
/// from global memory cost. Memory-hierarchy ratios, which only a measurement of the device gives
/// ([`PerfCalibration`]'s fit tool).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CostRatios {
    /// One LDS read of one operand value.
    pub lds_read: f64,
    /// One staged value of a tile.
    pub staged_value: f64,
    /// One value a Gemv lane reads straight from global memory: no LDS stage and no barrier, but also no row to
    /// amortize the load over, so a Gemv lane pays it per FMA where a tile pays a staged value per tile row.
    pub direct_value: f64,
}

/// The performance figures one device's launches were measured with. A figure here only ranks schedules that are
/// all correct, so a device nobody measured plans with [`PerfCalibration::PORTABLE`] and the plan says so
/// ([`Calibrated::Portable`]); a safety figure (a watchdog budget, a miscompile ceiling) is [`Calibration`]'s and
/// stays absent for such a device.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PerfCalibration {
    /// The Gemv columns of a dense `[N, K]` weight.
    pub dense_gemv_cols: GemvCols,
    /// The Gemv columns of a packed weight.
    pub packed_gemv_cols: GemvCols,
    /// The workgroup width of the LDS-tree attention Gemv (`attn_scores_v_gemv_lds`: one output per workgroup,
    /// the decode scores-by-V matmul), where the width is not the saturating residency.
    pub lds_gemv_width: u32,
    /// The register-tiled contraction cost model's ratios.
    pub cost: CostRatios,
    /// The keys per staged block of the tiled attention schedule, widest first.
    pub attention_key_blocks: &'static [usize],
    /// The longest run of contiguous K elements a packed decode Gemv lane decodes per trip: the register and issue
    /// budget of the device's compiler. The planner runs the format's factor run up to this cap.
    pub packed_gemv_max_run: u32,
}

impl PerfCalibration {
    /// The prior an unmeasured device plans with. Every field is named for the device it was measured on or the
    /// launch every device ran before it was measured, never a value nobody observed.
    pub const PORTABLE: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv_cols: STRIX_HALO_PACKED_GEMV_COLS,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: PORTABLE_COST_RATIOS,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        packed_gemv_max_run: PORTABLE_PACKED_GEMV_RUN,
    };

    /// Strix Halo (gfx1151) through HSA (ROCm): the device the Gemv figures were measured on, with the cost ratios
    /// `calibrate-launch` identified on ROCm.
    pub const STRIX_HALO_HSA: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv_cols: STRIX_HALO_PACKED_GEMV_COLS,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: STRIX_HALO_HSA_COST_RATIOS,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        packed_gemv_max_run: STRIX_HALO_PACKED_GEMV_RUN,
    };

    /// Strix Halo (gfx1151) behind RADV (wgpu, raw Vulkan): the same silicon and Gemv figures, but the cost ratios
    /// `calibrate-launch` could not identify behind RADV (POOT-1494) are spelled as [`Self::PORTABLE`]'s.
    pub const STRIX_HALO_VULKAN: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv_cols: STRIX_HALO_PACKED_GEMV_COLS,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: Self::PORTABLE.cost,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        packed_gemv_max_run: STRIX_HALO_PACKED_GEMV_RUN,
    };

    /// The RTX PRO 6000 Blackwell Server (compute capability 12.0). Measured for the packed Gemv run (POOT-1405) and
    /// the Gemv's direct value (POOT-1490); a field it did not identify is spelled out as [`Self::PORTABLE`]'s, so a
    /// reader sees which figures are borrowed: the LDS read and the staged value (the tiling fit is underdetermined on
    /// PTX: the 80 candidate rows are one tile family), both Gemv columns tables (the dense sweep ties at every
    /// output count but one, the packed table has no bank producer to sweep), the LDS-tree Gemv width and the
    /// attention key blocks.
    pub const SM_120: Self = Self {
        dense_gemv_cols: Self::PORTABLE.dense_gemv_cols,
        packed_gemv_cols: Self::PORTABLE.packed_gemv_cols,
        lds_gemv_width: Self::PORTABLE.lds_gemv_width,
        cost: CostRatios {
            lds_read: Self::PORTABLE.cost.lds_read,
            staged_value: Self::PORTABLE.cost.staged_value,
            direct_value: SM_120_DIRECT_VALUE_COST,
        },
        attention_key_blocks: Self::PORTABLE.attention_key_blocks,
        packed_gemv_max_run: SM_120_PACKED_GEMV_RUN,
    };
}

/// A device whose performance figures were measured: the identity [`Calibrated::Measured`] names, one per entry of
/// the table.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeasuredDevice {
    /// Strix Halo (gfx1151) through HSA (ROCm).
    StrixHaloHsa,
    /// Strix Halo (gfx1151) behind RADV (wgpu, raw Vulkan).
    StrixHaloVulkan,
    /// The RTX PRO 6000 Blackwell Server (compute capability 12.0).
    Sm120,
}

impl MeasuredDevice {
    /// What was measured on this device.
    pub const fn figures(self) -> PerfCalibration {
        match self {
            Self::StrixHaloHsa => PerfCalibration::STRIX_HALO_HSA,
            Self::StrixHaloVulkan => PerfCalibration::STRIX_HALO_VULKAN,
            Self::Sm120 => PerfCalibration::SM_120,
        }
    }
}

/// Whether a device's performance figures were measured on it or are the portable prior. It is part of
/// [`GemmLaunch`], so of every plan made for the device: two devices that differ only in this plan differently
/// and the plan names the one it used.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Calibrated {
    /// Measured on this device.
    Measured(MeasuredDevice),
    /// Nobody measured this device: it plans with [`PerfCalibration::PORTABLE`].
    Portable,
}

impl Calibrated {
    /// Strix Halo through HSA's measured entry.
    pub const STRIX_HALO_HSA: Self = Self::Measured(MeasuredDevice::StrixHaloHsa);

    /// Strix Halo behind RADV's measured entry.
    pub const STRIX_HALO_VULKAN: Self = Self::Measured(MeasuredDevice::StrixHaloVulkan);

    /// The RTX PRO 6000 Blackwell's measured entry.
    pub const SM_120: Self = Self::Measured(MeasuredDevice::Sm120);

    /// The figures the planner reads.
    pub const fn figures(self) -> PerfCalibration {
        match self {
            Self::Measured(device) => device.figures(),
            Self::Portable => PerfCalibration::PORTABLE,
        }
    }

    /// The entry of an AMD device the HSA runtime drives, by its gfx code.
    pub fn of_hsa_arch(arch: crate::AmdArch) -> Self {
        if is_strix_halo(arch.mcpu()) {
            Self::STRIX_HALO_HSA
        } else {
            Self::Portable
        }
    }

    /// The entry of the Vulkan device a `vendor_id` and device name identify. Keyed by identity, not by the
    /// runtime family: another RADV part, or an Intel or NVIDIA device behind Vulkan, is [`Self::Portable`].
    pub fn of_vulkan_device(vendor_id: u32, device_name: &str) -> Self {
        if vendor_id == VENDOR_AMD && is_strix_halo(device_name) {
            Self::STRIX_HALO_VULKAN
        } else {
            Self::Portable
        }
    }

    /// The entry of the CUDA device of this compute capability.
    pub fn of_cuda_compute_capability(major: u32, minor: u32) -> Self {
        if (major, minor) == (12, 0) {
            Self::SM_120
        } else {
            Self::Portable
        }
    }
}

/// The launch figures of the contraction schedules on one device: the narrow and wide lane grids of the
/// register-tiled one and the residency its occupancy model saturates a compute unit at, the Gemv geometry the
/// device's limits allow, and the measured or portable [`PerfCalibration`]. Resolved once from the device
/// ([`DeviceCaps::gemm_launch`]); the planner reads the fields and holds no tile, width or run literal of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GemmLaunch {
    /// The narrow register-tiled lane-grid edge.
    pub register_lane_edge: u32,
    /// The wide register-tiled lane-grid edge.
    pub register_lane_edge_wide: u32,
    /// Resident lanes that saturate one compute unit, in the register-tiled schedule's occupancy model.
    pub saturating_lanes: u32,
    /// What one lane of a workgroup holds in registers across a loop, from the device's register file. A tiled
    /// contraction and a tiled attention size their register tiles against it.
    pub register_budget: RegisterBudget,
    /// The widest workgroup a tiled attention runs: the saturating residency, bounded by the device's workgroup
    /// limits. [`Self::tiled_lanes`] lists the widths it considers.
    pub tiled_lanes_widest: u32,
    /// Workgroups per compute unit a launch aims at ([`FILL_WORKGROUPS_PER_UNIT`]).
    pub fill_workgroups_per_unit: u32,
    /// Lanes of a Gemv workgroup: the saturating residency, within the device's workgroup limits, whole waves.
    pub gemv_width: u32,
    /// Contiguous `K` elements a dense Gemv lane reads per trip: one [`VECTOR_LOAD_BYTES`] load of the f32
    /// activation.
    pub gemv_unroll: u32,
    /// The performance figures, and whether the device measured them.
    pub calibrated: Calibrated,
}

impl GemmLaunch {
    /// The launch of `caps`. No device has a confirmed defect that narrows the register-tiled lane grids (the RADV
    /// wave64 LDS defect of update 0169 does not reproduce, see `REGISTER_LANE_EDGE_WIDE`'s provenance).
    ///
    /// The tiled attention's widest workgroup is the saturating lanes bounded by the workgroup limits. The Gemv
    /// width is the same bound rounded down to whole waves (the largest subgroup the device reports, else
    /// [`WAVE_LANES`]): a device that runs 128 invocations is planned at 128, not handed an unlaunchable 256.
    pub fn of(caps: &DeviceCaps) -> Self {
        let wave = match caps.subgroup {
            Queried::Known(SubgroupSupport::Present { max_size, .. }) if max_size > 0 => max_size,
            _ => WAVE_LANES,
        };
        let limit = SATURATING_LANES
            .min(caps.max_workgroup_invocations)
            .min(caps.max_workgroup_size[0]);
        let whole_waves = limit / wave * wave;
        Self {
            register_lane_edge: REGISTER_LANE_EDGE,
            register_lane_edge_wide: REGISTER_LANE_EDGE_WIDE,
            saturating_lanes: SATURATING_LANES,
            register_budget: RegisterBudget {
                file: caps.register_file,
            },
            tiled_lanes_widest: limit,
            fill_workgroups_per_unit: FILL_WORKGROUPS_PER_UNIT,
            gemv_width: if whole_waves == 0 { limit } else { whole_waves },
            gemv_unroll: VECTOR_LOAD_BYTES / size_of::<f32>() as u32,
            calibrated: caps.perf_calibration,
        }
    }

    /// The workgroup widths a tiled attention considers, widest first: the powers of two of whole wave64s up to
    /// [`Self::tiled_lanes_widest`] (256, 128 and 64 where the device runs 256 lanes). None where the device runs
    /// fewer lanes than one wave64.
    pub fn tiled_lanes(&self) -> impl Iterator<Item = usize> {
        let widest = (self.tiled_lanes_widest / WAVE_LANES)
            .checked_ilog2()
            .map(|waves| (WAVE_LANES << waves) as usize);
        std::iter::successors(widest, |&lanes| Some(lanes / 2))
            .take_while(|&lanes| lanes >= WAVE_LANES as usize)
    }
}

/// The matrix unit's FMA rate on the Vulkan device a `vendor_id` and device name identify: measured on Strix Halo
/// (4x, [`crate::FragmentConfig::fma_rate`]), `Unknown` for a device nobody measured (the cooperative-matrix
/// extension reports no rate).
pub fn vulkan_matrix_fma_rate(vendor_id: u32, device_name: &str) -> Queried<u8> {
    if vendor_id == VENDOR_AMD && is_strix_halo(device_name) {
        Queried::Known(GFX11_MATRIX_FMA_RATE)
    } else {
        Queried::Unknown
    }
}

/// How many times faster than the lane FMA rate the gfx11 (RDNA3, 3.5) WMMA unit runs: the measured gfx1151 peaks,
/// 40-45 against 10.8 TFLOP/s (`receipts/1429-narrow-operands/design.md` section 1.1).
pub const GFX11_MATRIX_FMA_RATE: u8 = 4;

/// The A100's (sm_80) matrix-unit rate: NVIDIA's A100 datasheet, BF16 and FP16 Tensor Core 312 TFLOPS dense
/// against FP32 19.5 TFLOPS, 16x.
pub const SM80_MATRIX_FMA_RATE: u8 = 16;

/// The H100 SXM's (sm_90) matrix-unit rate: NVIDIA's H100 datasheet, BF16 and FP16 Tensor Core 989 TFLOPS dense
/// against FP32 67 TFLOPS, 14.8x, rounded to 15.
pub const SM90_MATRIX_FMA_RATE: u8 = 15;

/// The matrix-unit rate every sm_86, sm_89 and sm_120 part reaches at least (POOT-1496). A capability does not fix the
/// rate: the GeForce parts run the Tensor Cores at half rate when they accumulate in f32 (RTX 3090: 71 TFLOPS dense
/// against FP32 35.6, 2.0x; RTX 4090: 165.2 against 82.6, 2.0x; RTX 5090: 209.5 against 104.8, 2.0x), while the
/// datacenter and workstation parts of the same capability run it at full rate (A40: 149.7 against 37.4, 4.0x; L40S
/// 362 against 91.6, 4.0x). The cost model takes the floor, so it never credits a unit with speed it may not have.
pub const GEFORCE_MATRIX_FMA_RATE: u8 = 2;

/// PCI vendor id of AMD, as Vulkan reports it.
pub const VENDOR_AMD: u32 = 0x1002;

/// Whether a device identity string names the Strix Halo part (gfx1151): an HSA ISA name
/// (`"amdgcn-amd-amdhsa--gfx1151"`) or a Mesa RADV device name (`"AMD Radeon 8060S Graphics (RADV STRIX_HALO)"`).
pub(crate) fn is_strix_halo(device_name: &str) -> bool {
    let name = device_name.to_ascii_lowercase();
    name.contains("gfx1151") || name.contains("strix_halo")
}

/// What was measured on one Vulkan device, for the planner's chunking and routing around observed faults.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Calibration {
    /// The per-dispatch work budgets that keep the display watchdog from killing a dispatch; `None` on a
    /// device measured to need none.
    pub watchdog_budget: Option<WatchdogBudget>,
    /// The defects observed on this device.
    pub known_miscompiles: KnownMiscompiles,
    /// A single-buffer ceiling below the device's advertised limit, where a larger buffer wedged the ring.
    pub safe_max_buffer_bytes: Option<u64>,
}

impl Calibration {
    /// Strix Halo behind RADV, as cards 095, 163, 258 and 159 measured it and the 2026-09-25 lm_head wedge
    /// bounded the buffer size.
    const STRIX_HALO_MISCOMPILES: KnownMiscompiles = KnownMiscompiles {
        tiled_gemm_max_workgroups: Some(1 << 15),
    };

    /// Strix Halo (gfx1151) behind RADV, the one device these figures were measured on.
    pub const STRIX_HALO_RADV: Self = Self {
        watchdog_budget: Some(WatchdogBudget {
            serial_dequant_work: 400_000_000,
            scatter_update_work: 1_000_000,
            indexed_dequant_work: 800_000_000,
            packed_tiled_work: PACKED_TILED_CHUNK_WORK,
            // Measured on Strix Halo behind RADV only (POOT-1313); another RADV or display-attached device has
            // no calibration, so no `watchdog_budget` and an unsplit 8192-token prefill.
            attention_work: ATTENTION_CHUNK_WORK,
        }),
        known_miscompiles: Self::STRIX_HALO_MISCOMPILES,
        safe_max_buffer_bytes: Some(AMD_RADV_SAFE_MAX_BUFFER_BYTES),
    };

    /// The calibration of the Vulkan device a `vendor_id` and device name identify, or `Unknown` for a
    /// device nobody measured.
    pub fn of_vulkan_device(vendor_id: u32, device_name: &str) -> Queried<Self> {
        if vendor_id == VENDOR_AMD && is_strix_halo(device_name) {
            Queried::Known(Self::STRIX_HALO_RADV)
        } else {
            Queried::Unknown
        }
    }

    /// The calibration a device has when nobody measured it: no budget, no known defect, no buffer ceiling.
    /// It claims nothing; a consumer that needs a figure sees it absent.
    pub const UNMEASURED: Self = Self {
        watchdog_budget: None,
        known_miscompiles: KnownMiscompiles::NONE,
        safe_max_buffer_bytes: None,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the part that was measured gets its figures: the Strix Halo RADV device is calibrated and another
    /// AMD Vulkan device is not. Mutation: key on the vendor id alone; the discrete card reads `Known`.
    #[test]
    fn calibration_is_keyed_by_device_identity_not_vendor() {
        let strix = Calibration::of_vulkan_device(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)");
        assert!(matches!(
            strix,
            Queried::Known(c) if c.watchdog_budget.is_some()
        ));
        assert_eq!(
            Calibration::of_vulkan_device(VENDOR_AMD, "AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            Queried::Unknown
        );
        assert_eq!(
            Calibration::of_vulkan_device(0x10de, "NVIDIA GeForce RTX 4090"),
            Queried::Unknown
        );
    }

    /// Where a launch figure came from: what it is called in the source, its value, and the evidence for it. A
    /// literal that shapes a launch has an entry in a table of these, so a number is never free-floating.
    struct Provenance {
        /// The constant's name in the source.
        name: &'static str,
        /// Its value.
        value: f64,
        /// The measurement, card or structural reason it rests on; never empty.
        source: &'static str,
    }

    /// The provenance of each figure a [`GemmLaunch`] and a [`PerfCalibration`] entry carry.
    const LAUNCH_PROVENANCE: &[Provenance] = &[
        Provenance {
            name: "REGISTER_LANE_EDGE",
            value: 8.0,
            source: "card 542b: the 8x8 lane grid, one wave64",
        },
        Provenance {
            name: "REGISTER_LANE_EDGE_WIDE",
            value: 16.0,
            source: "POOT-1193: the 16x16 lane grid, 256 lanes. POOT-1403: runs on RADV too; the wave64 LDS defect (update 0169: a 256-lane LDS-tiled GEMM read stale upper-wave LDS) does not reproduce on Strix Halo RADV Mesa 26.1.3, on the register-tiled body at 16x16 (16.5 KiB LDS; poot-kernelgen the_register_tiled_body_runs_256_lane_workgroups_on_radv, mutation-proven) nor on the deleted one-output-per-lane Tiled body at 16x16 (bit-exact); master already ran the packed register-tiled 16x16 there",
        },
        Provenance {
            name: "SATURATING_LANES",
            value: 256.0,
            source: "structural (POOT-1403): eight wave32s per compute unit, four per RDNA SIMD32 or two per NVIDIA SM partition; the register-tiled cost model's residency, not a per-device measurement",
        },
        Provenance {
            name: "ACCUMULATOR_SHARE",
            value: 4.0,
            source: "structural (POOT-1479): a lane holds at most a quarter of its resident registers as accumulators across a loop, the rest carrying the step's operands, addresses and loop state; gfx1151 spill rows (poot-kernelgen register_budget): 64 accumulators of 256 compile to 149 VGPRs unspilled, 128 spill. The rounding is up: ceil(255 / 4) = 64 keeps sm_80 at 8 x 8, offline ptxas -v row",
        },
        Provenance {
            name: "HELD_SHARE",
            value: 2.0,
            source: "structural (POOT-1447): accumulators and held operands together fill at most half of a lane's resident registers; below it RDNA3 keeps 12 waves per SIMD and an NVIDIA SM 16 warps (the held-Q spill rows)",
        },
        Provenance {
            name: "PORTABLE_ACCUMULATORS",
            value: 64.0,
            source: "a quarter of the 256 registers a lane addresses on gfx11 and (255) NVIDIA, the devices register-tiled launches were measured on; the budget of a device whose API reports no register file, recorded in the plan",
        },
        Provenance {
            name: "WAVE_LANES",
            value: 64.0,
            source: "structural: the widest wave poot targets (wave64 on GCN/CDNA and RADV's wave64 mode), the grain of a tiled attention's workgroup width",
        },
        Provenance {
            name: "FILL_WORKGROUPS_PER_UNIT",
            value: 2.0,
            source: "structural (POOT-1479): a unit holds two resident workgroups so one's tail overlaps the next; the factor of vLLM's wvSplitK grid (CuCount * 4, one tile per SIMD) and Marlin (tiles * 4 <= sms), stated once for the Gemv columns, split-KV and selected-head grouping",
        },
        Provenance {
            name: "VECTOR_LOAD_BYTES",
            value: 16.0,
            source: "structural: the widest load every backend emits (dwordx4, ld.v4); POOT-1405 counts the Gemv activation read in 16-byte groups, so an f32 activation unrolls 4",
        },
        Provenance {
            name: "STRIX_HALO_DENSE_GEMV_COLS_WIDE",
            value: 4.0,
            source: "card 727, gfx1151 sweep over the qwen2.5-0.5b BF16 decode shapes: wgpu Gemv device time per token cols 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9, 4 4.7; ROCm decode 78-81 tok/s from cols 16 to 2",
        },
        Provenance {
            name: "STRIX_HALO_DENSE_GEMV_COLS_NARROW",
            value: 2.0,
            source: "card 727: the narrower fallback keeps a small N from starving the device; same sweep as the wide columns",
        },
        Provenance {
            name: "STRIX_HALO_PACKED_GEMV_COLS_WIDE",
            value: 16.0,
            source: "card 653, gfx1151 per-kernel sweep over the qwen2.5-0.5b GGUF projection shapes: {16, 8} 3.7 ms Gemv device time per Q4_K_M token against 4.2 ms for the dense table and 9.7 ms for one column per workgroup",
        },
        Provenance {
            name: "STRIX_HALO_PACKED_GEMV_COLS_NARROW",
            value: 8.0,
            source: "card 653: same sweep as the wide columns; the plan of a device whose compute units are unknown",
        },
        Provenance {
            name: "STRIX_HALO_LDS_GEMV_WIDTH",
            value: 128.0,
            source: "RADV STRIX_HALO sweep, Qwen2.5-0.5B decode, fused ms per token: w=32 50.9, 64 53.7, 128 39.8, 256 45.6; non-monotone, measured on RADV only and borrowed elsewhere",
        },
        Provenance {
            name: "PORTABLE_LDS_READ_COST",
            value: 1.0,
            source: "POOT-1403 fit, gfx1151 ROCm, every tiling of a qwen3-0.6b prefill at 512 and 2048 tokens: contraction device time follows the ratios within a few percent. POOT-1494 refit over every candidate tiling: ROCm identifies it (9.6-10.2), RADV does not (staged value 37-52 between two runs), so RADV and an unmeasured device keep this fit",
        },
        Provenance {
            name: "PORTABLE_STAGED_VALUE_COST",
            value: 8.0,
            source: "POOT-1403 fit (same run as the LDS read cost): a global load, its LDS write and its share of the round's two barriers; not identified on RADV by POOT-1494, so the prior of every device but ROCm gfx1151",
        },
        Provenance {
            name: "STRIX_HALO_HSA_LDS_READ_COST",
            value: 9.88,
            source: "POOT-1494 calibrate-launch, gfx1151 ROCm, every candidate tiling of 8 shapes (M 16-1024, weights 1024x1024 and 3072x1024), two runs of 80 rows: 10.18 and 9.60, R^2 0.935 and 0.927, spread 5.9 %; one fit over the 160 rows together (POOT-1489): 9.88, R^2 0.931",
        },
        Provenance {
            name: "STRIX_HALO_HSA_STAGED_VALUE_COST",
            value: 8.43,
            source: "POOT-1494 calibrate-launch (same runs as the LDS read cost): 8.65 and 8.23, spread 5.0 %; the joint fit over the 160 rows: 8.43",
        },
        Provenance {
            name: "PORTABLE_DIRECT_VALUE_COST",
            value: 7.6,
            source: "POOT-1489 calibrate-launch indexed, gfx1151 behind RADV (wgpu), 88 cells of 8-128 experts x 4-4096 routed rows x two weight shapes, the Gemv and the grouped tiling of each timed, two runs: the only direct value at which the model's Gemv-versus-grouped choices are never slower than the retired 32-row switch on any cell is 7.6 (7.7 in the first run), mean regret against the faster schedule 1.10 against the switch's 1.61. It replaces POOT-1476's provisional 4.0, which was 15 % slower than the switch in one near-tie cell. The prior of a device nobody measured",
        },
        Provenance {
            name: "STRIX_HALO_HSA_DIRECT_VALUE_COST",
            value: 13.4,
            source: "POOT-1489 calibrate-launch indexed, gfx1151 ROCm, the same 88 cells: the direct values at which the model's choices are never slower than the retired 32-row switch (within 5 %) and within 5 % of the best mean regret are 11.9 to 23.6. POOT-1503 cut the range at 15.2, the crossover at which the model's cost of a one-row dense Gemv of four columns per workgroup passes a fragment tiling's once both fill the device (the 16.8 centre moved the 151936-column head of qwen3-0.6b and qwen3-4b onto a matrix unit: decode 8 % and 5 % slower on ROCm); the geometric centre of 11.9 to 15.2. Two runs; the dense side is the model's own crossover, confirmed by the A/B of that head on ROCm",
        },
        Provenance {
            name: "SM_120_DIRECT_VALUE_COST",
            value: 14.15,
            source: "POOT-1490 calibrate-launch indexed on one RTX PRO 6000 Blackwell Server (sm_120, secure pod, 88 cells, 176 rows): the direct values at which the model's choices are never slower than the retired 32-row switch (within 5 %) and within 5 % of the best mean regret are 11.9 to 16.8; the geometric centre. Mean regret against the faster schedule 1.056 against the retired switch's 1.177. An RTX 4090 (sm_89) gave 7.6 to 11.9, so the figure moves between parts and this one is that part's; receipts/1490-cuda-sweep/",
        },
        Provenance {
            name: "STRIX_HALO_ATTENTION_KEY_BLOCK_WIDEST",
            value: 32.0,
            source: "POOT-1402 bank sweep: a 64-key block was 20-30% slower than a 32-key one on ROCm and wgpu (Strix Halo) and no faster on PTX",
        },
        Provenance {
            name: "STRIX_HALO_ATTENTION_KEY_BLOCK_NARROWEST",
            value: 16.0,
            source: "POOT-1402: the narrower block the staged tile falls back to when LDS holds no 32-key one",
        },
        Provenance {
            name: "GFX11_MATRIX_FMA_RATE",
            value: 4.0,
            source: "measured gfx1151 peaks, 40-45 TFLOP/s matrix against 10.8 TFLOP/s FMA (receipts/1429-narrow-operands/design.md section 1.1)",
        },
        Provenance {
            name: "SM80_MATRIX_FMA_RATE",
            value: 16.0,
            source: "NVIDIA A100 datasheet: BF16 and FP16 Tensor Core 312 TFLOPS dense over FP32 19.5 TFLOPS",
        },
        Provenance {
            name: "SM90_MATRIX_FMA_RATE",
            value: 15.0,
            source: "NVIDIA H100 SXM datasheet: BF16 and FP16 Tensor Core 989 TFLOPS dense over FP32 67 TFLOPS (14.8, rounded). Other compute capabilities: no sourced peak, rate Unknown (no credit)",
        },
        Provenance {
            name: "GEFORCE_MATRIX_FMA_RATE",
            value: 2.0,
            source: "NVIDIA datasheets, FP16 Tensor Core with FP32 accumulate, dense, over non-Tensor FP32: RTX 3090 (GA102 whitepaper) 71 over 35.6; RTX 4090 (Ada whitepaper) 165.2 over 82.6; RTX 5090 209.5 over 104.8. The floor over the parts of sm_86, sm_89 and sm_120: A40 (149.7 over 37.4) and L40S (362 over 91.6) run 4x",
        },
        Provenance {
            name: "PORTABLE_PACKED_GEMV_RUN",
            value: 8.0,
            source: "card 653: the 8-element run every class launched before POOT-1398; the run of a decode that addresses each element on its own (gemv_run's Elementwise arm)",
        },
        Provenance {
            name: "STRIX_HALO_PACKED_GEMV_RUN",
            value: 32.0,
            source: "measured: gfx1151 ROCm and RADV at 17408 x 5120, run 32 vs 8: Q4_K 120 -> 195 GB/s, Q5_K 119 -> 190 (ROCm), Q4_K 143 -> 171-184 (Vulkan); POOT-1390 section 5 and the POOT-1398 receipt",
        },
        Provenance {
            name: "SM_120_PACKED_GEMV_RUN",
            value: 16.0,
            source: "measured on one part only, RTX PRO 6000 Blackwell Server (sm_120), at 17408 x 5120 with the vector activation read and 32-bit index math, run 8 / 16 / 32 GB/s: Q4_K 721/980/665, Q5_K 747/955/741, Q6_K 861/1254/1253 (factor run 16), Q3_K 567/710/711, Q2_K 539/695/695, Q4_0 1026/1094/751, Q8_0 1383/1402/1260, IQ4_XS 894/856/601, IQ4_NL 998/911/641; POOT-1405 receipts. Other NVIDIA parts unmeasured",
        },
    ];

    /// The provenance entry named `name`, whose value is `value` and whose source is stated.
    fn assert_provenance(name: &str, value: impl Into<f64>) {
        let value = value.into();
        let entry = LAUNCH_PROVENANCE
            .iter()
            .find(|entry| entry.name == name)
            .unwrap_or_else(|| panic!("{name} has no provenance entry"));
        assert_eq!(entry.value, value, "{name}");
        assert!(!entry.source.is_empty(), "{name} states no source");
    }

    /// Every provenance entry is claimed by a figure below, so a row for a deleted figure does not linger.
    fn claimed(names: &[&str]) {
        let table: Vec<&str> = LAUNCH_PROVENANCE.iter().map(|entry| entry.name).collect();
        let mut sorted = names.to_vec();
        sorted.sort_unstable();
        let mut table_sorted = table.clone();
        table_sorted.sort_unstable();
        assert_eq!(table_sorted, sorted, "provenance rows and figures differ");
    }

    /// Every figure a `GemmLaunch` and every `PerfCalibration` entry carries has a provenance row whose value is
    /// the figure's and whose source is stated. Both structs are destructured without `..`, so a new field fails
    /// to compile here until it is listed. Each row pins the value as a literal, so a figure changed without its row fails. Mutation: change
    /// `STRIX_HALO_PACKED_GEMV_RUN` without its row; the value check fails.
    #[test]
    fn every_launch_figure_has_provenance() {
        let GemmLaunch {
            register_lane_edge,
            register_lane_edge_wide,
            saturating_lanes,
            register_budget: _,
            tiled_lanes_widest,
            fill_workgroups_per_unit,
            // Derived from the caps, not a literal: `the_gemv_geometry_follows_the_device_limits` covers them.
            gemv_width: _,
            gemv_unroll: _,
            calibrated: _,
        } = DeviceCaps::rocm_default().gemm_launch();
        assert_eq!(
            tiled_lanes_widest, saturating_lanes,
            "the widest tiled attention is the saturating residency"
        );
        assert_provenance("ACCUMULATOR_SHARE", crate::register_file::ACCUMULATOR_SHARE);
        assert_provenance("HELD_SHARE", crate::register_file::HELD_SHARE);
        assert_provenance("WAVE_LANES", WAVE_LANES);
        assert_provenance(
            "PORTABLE_ACCUMULATORS",
            RegisterBudget {
                file: Queried::Unknown,
            }
            .accumulators(256),
        );
        assert_provenance("REGISTER_LANE_EDGE", register_lane_edge);
        assert_provenance("REGISTER_LANE_EDGE_WIDE", register_lane_edge_wide);
        assert_provenance("SATURATING_LANES", saturating_lanes);
        assert_provenance("FILL_WORKGROUPS_PER_UNIT", fill_workgroups_per_unit);
        assert_provenance("VECTOR_LOAD_BYTES", VECTOR_LOAD_BYTES);
        for (entry, row) in [
            (PerfCalibration::PORTABLE, "PORTABLE_PACKED_GEMV_RUN"),
            (
                PerfCalibration::STRIX_HALO_HSA,
                "STRIX_HALO_PACKED_GEMV_RUN",
            ),
            (
                PerfCalibration::STRIX_HALO_VULKAN,
                "STRIX_HALO_PACKED_GEMV_RUN",
            ),
            (PerfCalibration::SM_120, "SM_120_PACKED_GEMV_RUN"),
        ] {
            let PerfCalibration {
                dense_gemv_cols,
                packed_gemv_cols,
                lds_gemv_width,
                cost: _,
                attention_key_blocks,
                packed_gemv_max_run,
            } = entry;
            assert_provenance(row, packed_gemv_max_run);
            // Every measured entry spells the Strix Halo figures (SM_120 borrows them from `PORTABLE`, which
            // are the same values): one row each, checked against whichever entry carries it.
            assert_provenance("STRIX_HALO_DENSE_GEMV_COLS_WIDE", dense_gemv_cols.wide);
            assert_provenance("STRIX_HALO_DENSE_GEMV_COLS_NARROW", dense_gemv_cols.narrow);
            assert_provenance("STRIX_HALO_PACKED_GEMV_COLS_WIDE", packed_gemv_cols.wide);
            assert_provenance(
                "STRIX_HALO_PACKED_GEMV_COLS_NARROW",
                packed_gemv_cols.narrow,
            );
            assert_provenance("STRIX_HALO_LDS_GEMV_WIDTH", lds_gemv_width);
            let [widest, narrowest] = attention_key_blocks else {
                panic!("two key blocks, widest first: {attention_key_blocks:?}");
            };
            assert_provenance("STRIX_HALO_ATTENTION_KEY_BLOCK_WIDEST", *widest as u32);
            assert_provenance(
                "STRIX_HALO_ATTENTION_KEY_BLOCK_NARROWEST",
                *narrowest as u32,
            );
        }
        // The cost ratios: the portable prior is the unidentified devices' (RADV borrows it, sm_120 borrows all but the direct value), ROCm
        // gfx1151 has its own.
        let portable = PerfCalibration::PORTABLE.cost;
        assert_provenance("PORTABLE_LDS_READ_COST", portable.lds_read);
        assert_provenance("PORTABLE_STAGED_VALUE_COST", portable.staged_value);
        assert_provenance("PORTABLE_DIRECT_VALUE_COST", portable.direct_value);
        let hsa = PerfCalibration::STRIX_HALO_HSA.cost;
        assert_provenance("STRIX_HALO_HSA_LDS_READ_COST", hsa.lds_read);
        assert_provenance("STRIX_HALO_HSA_STAGED_VALUE_COST", hsa.staged_value);
        assert_provenance("STRIX_HALO_HSA_DIRECT_VALUE_COST", hsa.direct_value);
        assert_eq!(
            PerfCalibration::STRIX_HALO_VULKAN.cost,
            portable,
            "RADV borrows the prior"
        );
        // sm_120 identified the direct value only: the other two ratios are the prior's.
        let sm_120 = PerfCalibration::SM_120.cost;
        assert_provenance("SM_120_DIRECT_VALUE_COST", sm_120.direct_value);
        assert_eq!(
            (sm_120.lds_read, sm_120.staged_value),
            (portable.lds_read, portable.staged_value),
            "sm_120 borrows the prior's LDS read and staged value"
        );
        assert_provenance("GFX11_MATRIX_FMA_RATE", GFX11_MATRIX_FMA_RATE);
        assert_provenance("SM80_MATRIX_FMA_RATE", SM80_MATRIX_FMA_RATE);
        assert_provenance("SM90_MATRIX_FMA_RATE", SM90_MATRIX_FMA_RATE);
        assert_provenance("GEFORCE_MATRIX_FMA_RATE", GEFORCE_MATRIX_FMA_RATE);
        claimed(&[
            "REGISTER_LANE_EDGE",
            "REGISTER_LANE_EDGE_WIDE",
            "SATURATING_LANES",
            "ACCUMULATOR_SHARE",
            "HELD_SHARE",
            "WAVE_LANES",
            "PORTABLE_ACCUMULATORS",
            "FILL_WORKGROUPS_PER_UNIT",
            "VECTOR_LOAD_BYTES",
            "STRIX_HALO_DENSE_GEMV_COLS_WIDE",
            "STRIX_HALO_DENSE_GEMV_COLS_NARROW",
            "STRIX_HALO_PACKED_GEMV_COLS_WIDE",
            "STRIX_HALO_PACKED_GEMV_COLS_NARROW",
            "STRIX_HALO_LDS_GEMV_WIDTH",
            "PORTABLE_LDS_READ_COST",
            "PORTABLE_STAGED_VALUE_COST",
            "PORTABLE_DIRECT_VALUE_COST",
            "STRIX_HALO_HSA_LDS_READ_COST",
            "STRIX_HALO_HSA_STAGED_VALUE_COST",
            "STRIX_HALO_HSA_DIRECT_VALUE_COST",
            "SM_120_DIRECT_VALUE_COST",
            "STRIX_HALO_ATTENTION_KEY_BLOCK_WIDEST",
            "STRIX_HALO_ATTENTION_KEY_BLOCK_NARROWEST",
            "GFX11_MATRIX_FMA_RATE",
            "SM80_MATRIX_FMA_RATE",
            "SM90_MATRIX_FMA_RATE",
            "GEFORCE_MATRIX_FMA_RATE",
            "PORTABLE_PACKED_GEMV_RUN",
            "STRIX_HALO_PACKED_GEMV_RUN",
            "SM_120_PACKED_GEMV_RUN",
        ]);
    }

    /// A device's figures follow its identity, not its runtime family: Strix Halo is measured through HSA and
    /// through Vulkan, another AMD, an Intel or an NVIDIA Vulkan device is portable, and CUDA is measured on
    /// sm_120 only. Mutation: key the Vulkan arm on the vendor alone; the discrete AMD card reads `Measured`.
    #[test]
    fn performance_figures_follow_device_identity() {
        assert_eq!(
            Calibrated::of_hsa_arch(crate::AmdArch::gfx1151()),
            Calibrated::STRIX_HALO_HSA
        );
        assert_eq!(
            Calibrated::of_vulkan_device(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)"),
            Calibrated::STRIX_HALO_VULKAN
        );
        for (vendor, name) in [
            (VENDOR_AMD, "AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            (0x8086, "Intel(R) Arc(tm) A770 Graphics (DG2)"),
            (0x10de, "NVIDIA GeForce RTX 4090"),
        ] {
            assert_eq!(
                Calibrated::of_vulkan_device(vendor, name),
                Calibrated::Portable,
                "{name}"
            );
        }
        assert_eq!(
            Calibrated::of_cuda_compute_capability(12, 0),
            Calibrated::SM_120
        );
        for (major, minor) in [(8, 0), (9, 0), (10, 0), (12, 1)] {
            assert_eq!(
                Calibrated::of_cuda_compute_capability(major, minor),
                Calibrated::Portable,
                "sm_{major}{minor}"
            );
        }
    }

    /// The cost ratios follow the backend they were identified on, not the silicon (POOT-1489): gfx1151 through HSA has
    /// the ratios `calibrate-launch` identified on ROCm, gfx1151 behind RADV keeps the portable prior (its ratios
    /// moved 38 % between two runs, POOT-1494), and every other figure of the two entries is the same device's.
    /// Mutation: resolve `STRIX_HALO_HSA` for the Vulkan identity; the prior row goes red.
    #[test]
    fn the_cost_ratios_follow_the_backend_that_measured_them() {
        let hsa = Calibrated::of_hsa_arch(crate::AmdArch::gfx1151()).figures();
        let vulkan = Calibrated::of_vulkan_device(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)")
            .figures();
        assert_ne!(hsa.cost, vulkan.cost, "the two backends share one entry");
        assert_eq!(
            vulkan.cost,
            PerfCalibration::PORTABLE.cost,
            "RADV keeps the prior"
        );
        assert_eq!(
            PerfCalibration {
                cost: vulkan.cost,
                ..hsa
            },
            vulkan,
            "only the cost ratios differ between the two entries of one silicon"
        );
    }

    /// The packed Gemv run an unmeasured device plans with is the portable 8, whichever runtime drives it, and a
    /// measured one carries its own: the launch a plan is made with names which. Mutation: make `figures` return
    /// the measured run for `Portable`.
    #[test]
    fn a_launch_carries_the_figures_of_its_calibration() {
        let launch = |calibrated| {
            DeviceCaps {
                perf_calibration: calibrated,
                ..DeviceCaps::rocm_default()
            }
            .gemm_launch()
        };
        let portable = launch(Calibrated::Portable);
        let strix = launch(Calibrated::STRIX_HALO_HSA);
        let sm_120 = launch(Calibrated::SM_120);
        assert_eq!(portable.calibrated.figures().packed_gemv_max_run, 8);
        assert_eq!(strix.calibrated.figures().packed_gemv_max_run, 32);
        assert_eq!(sm_120.calibrated.figures().packed_gemv_max_run, 16);
        assert_ne!(
            portable, strix,
            "the launch must record which calibration it used"
        );
    }

    /// A tiled attention's workgroup widths are the powers of two of whole wave64s up to the saturating residency
    /// and the device's workgroup limit, widest first, and a device that runs fewer lanes than a wave64 has none.
    /// Mutation: skip the `min` with the limit in `GemmLaunch::of`; the 128-lane device lists 256.
    #[test]
    fn tiled_attention_widths_follow_the_workgroup_limit() {
        let widths = |limit| {
            DeviceCaps {
                max_workgroup_invocations: limit,
                max_workgroup_size: [limit, 1024, 1024],
                ..DeviceCaps::rocm_default()
            }
            .gemm_launch()
            .tiled_lanes()
            .collect::<Vec<_>>()
        };
        assert_eq!(widths(1024), [256, 128, 64]);
        assert_eq!(widths(256), [256, 128, 64]);
        assert_eq!(widths(192), [128, 64]);
        assert_eq!(widths(128), [128, 64]);
        assert_eq!(widths(64), [64]);
        assert!(widths(32).is_empty());
    }

    /// The Gemv width is the saturating residency within the device's workgroup limits, in whole waves of the
    /// largest subgroup it reports (64 where none): a device that runs fewer invocations is planned at fewer, not
    /// handed an unlaunchable 256. Mutation: drop the `max_workgroup_invocations` bound from `GemmLaunch::of`; the
    /// 128-invocation row plans 256.
    #[test]
    fn the_gemv_geometry_follows_the_device_limits() {
        let width = |invocations: u32, size_x: u32, wave: Option<u32>| {
            DeviceCaps {
                max_workgroup_invocations: invocations,
                max_workgroup_size: [size_x, 1024, 1024],
                subgroup: wave.map_or(Queried::Unknown, |size| {
                    Queried::Known(SubgroupSupport::Present {
                        min_size: size,
                        max_size: size,
                    })
                }),
                ..DeviceCaps::rocm_default()
            }
            .gemm_launch()
            .gemv_width
        };
        assert_eq!(width(1024, 1024, Some(32)), 256, "the reference caps");
        assert_eq!(width(1024, 1024, None), 256, "an unknown wave is 64");
        assert_eq!(width(128, 1024, Some(32)), 128, "invocations bound it");
        assert_eq!(width(1024, 192, Some(64)), 192, "the X extent bounds it");
        assert_eq!(width(1024, 200, Some(64)), 192, "whole waves");
        assert_eq!(
            width(48, 1024, Some(64)),
            48,
            "less than a wave is the limit itself"
        );
        assert_eq!(DeviceCaps::rocm_default().gemm_launch().gemv_unroll, 4);
    }

    /// The matrix-unit rate is sourced per compute capability: the A100 and H100 carry their datasheet ratios, and
    /// sm_86, sm_89 and sm_120 their GeForce floor, every other capability, and a Vulkan device nobody measured, is `Unknown` (no speed credit). Mutation: key
    /// `vulkan_matrix_fma_rate` on the vendor alone; the discrete card reads `Known`.
    #[test]
    fn the_matrix_rate_is_known_only_where_it_is_sourced() {
        let rate = |cc| {
            crate::MatrixUnits::nvidia(cc)
                .iter()
                .map(|config| config.fma_rate)
                .collect::<Vec<_>>()
        };
        assert_eq!(rate((8, 0)), vec![Queried::Known(16); 2]);
        assert_eq!(rate((9, 0)), vec![Queried::Known(15); 2]);
        for cc in [(8, 6), (8, 9), (12, 0)] {
            assert_eq!(rate(cc), vec![Queried::Known(2); 2], "sm_{}{}", cc.0, cc.1);
        }
        assert_eq!(rate((10, 0)), vec![Queried::Unknown; 2], "sm_100");
        assert_eq!(
            vulkan_matrix_fma_rate(VENDOR_AMD, "AMD Radeon 8060S (RADV STRIX_HALO)"),
            Queried::Known(4)
        );
        assert_eq!(
            vulkan_matrix_fma_rate(VENDOR_AMD, "AMD Radeon RX 7900 XTX (RADV NAVI31)"),
            Queried::Unknown
        );
    }
}
