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

/// The fewest steps of a clustered attention walk a split-KV share takes, on a device nobody measured: a share of
/// fewer steps spends more on the second launch's combine than the extra workgroup hides. The prior of the POOT-1549
/// design's roofline model (`receipts/1549-design/model.txt`); a pod sweep of {2, 4, 8} replaces it per device.
/// [`PerfCalibration::PORTABLE`]'s figure, and every measured entry's until its device is swept.
pub const PORTABLE_ATTENTION_SPLIT_MIN_STEPS: u32 = 4;

/// The lanes of one wave64, the widest wave poot targets: the grain of a tiled attention's workgroup width, which
/// is whole waves of the widest wave so it runs on every device.
pub(crate) const WAVE_LANES: u32 = 64;

/// The Gemv columns of a dense `[N, K]` weight on Strix Halo (card 727): `cols` 4 while that still launches
/// [`FILL_WORKGROUPS_PER_UNIT`] workgroups per compute unit, else 2. Set from the card's sweep on gfx1151 over the
/// qwen2.5-0.5b BF16 decode shapes (wgpu Gemv device time per token: `cols` 64 7.1 ms, 32 6.0, 16 5.3, 8 4.9,
/// 4 4.7; ROCm decode 78 to 81 tok/s from `cols` 16 to 2). Nearly flat between 4 and 8, so not derivable.
pub const STRIX_HALO_DENSE_GEMV_COLS: GemvCols = GemvCols { wide: 4, narrow: 2 };

/// The width of the LDS-tree Gemv on Strix Halo behind RADV (Qwen2.5-0.5B decode, fused ms per token: w=32 50.9,
/// 64 53.7, 128 39.8, 256 45.6). Non-monotone, so a measurement, not a derivation.
pub const STRIX_HALO_LDS_GEMV_WIDTH: u32 = 128;

/// The packed Gemv figures of a device nobody measured, and of Strix Halo until `calibrate-launch` measures it
/// (POOT-1550 D4): the prior of the design's cost model in cycles of a ~2 GHz unit clock, over the structural facts of an
/// RDNA3.5 compute unit (two SIMD32, 16 resident waves each). Nobody observed the three rates on this part: a plan on
/// it says so ([`Calibrated::Portable`]).
pub const PORTABLE_GEMV_DEVICE: GemvDevice = GemvDevice {
    dram_latency_cycles: 1600.0,
    l1_lines_per_cycle: 1.0,
    dispatch_cycles: 8000.0,
    dependent_issue_cycles: 4.0,
    issue_per_cycle: 2.0,
    resident_lanes: 1024,
};

/// The packed Gemv figures of the RTX PRO 6000 Blackwell Server (sm_120) at the part's 2.43 GHz SM clock, over its
/// four warp schedulers and 1536 resident lanes per SM. The latency, L1 rate and dependent issue interval are
/// identified by `benchmarks/calibration/fit_packed_gemv.py` over the 2440-row corpus of POOT-1608, POOT-1610 and POOT-1611 (the
/// `lines` of its body-facts file are the generator's, every column group's activation read paid); the dispatch is the
/// measured empty launch.
pub const SM_120_GEMV_DEVICE: GemvDevice = GemvDevice {
    dram_latency_cycles: 2070.0,
    l1_lines_per_cycle: 0.77,
    dispatch_cycles: 4450.0,
    dependent_issue_cycles: 2.9,
    issue_per_cycle: 4.0,
    resident_lanes: 1536,
};

/// Warp instructions one SM issues per cycle at compute capabilities 8.6 and 8.9: four schedulers.
const SM_86_89_ISSUE_PER_CYCLE: f64 = 4.0;

/// Lanes one SM holds resident at compute capabilities 8.6 and 8.9: 48 warps of 32.
const SM_86_89_RESIDENT_LANES: u32 = 1536;

/// The packed Gemv figures of the RTX 4090 (sm_89, 2.52 GHz boost): the SM's structural facts (four warp schedulers,
/// 1536 resident lanes), the L1 rate the POOT-1614 sweep identified (852 rows over five cells on one RTX 4090,
/// `benchmarks/calibration/ptx-sm89-packed-gemv.tsv`), and priors for what that sweep did not identify (the latency
/// and the dependent issue interval) or did not measure (the dispatch, borrowed from sm_120's empty launch). Their
/// provenance rows say which is which.
pub const SM_89_GEMV_DEVICE: GemvDevice = GemvDevice {
    dram_latency_cycles: 2016.0,
    l1_lines_per_cycle: 0.824,
    dispatch_cycles: 4610.0,
    dependent_issue_cycles: 4.0,
    issue_per_cycle: SM_86_89_ISSUE_PER_CYCLE,
    resident_lanes: SM_86_89_RESIDENT_LANES,
};

/// The packed Gemv figures of the RTX 3090 and A5000 (sm_86): the same SM as sm_89's in the two structural facts, and
/// every rate the portable prior's: no packed Gemv sweep exists on an sm_86 part (POOT-1605 ran only POOT-1602's
/// expert-projection row there), so nothing of sm_89's identified L1 rate is carried over.
pub const SM_86_GEMV_DEVICE: GemvDevice = GemvDevice {
    dram_latency_cycles: PORTABLE_GEMV_DEVICE.dram_latency_cycles,
    l1_lines_per_cycle: PORTABLE_GEMV_DEVICE.l1_lines_per_cycle,
    dispatch_cycles: PORTABLE_GEMV_DEVICE.dispatch_cycles,
    dependent_issue_cycles: PORTABLE_GEMV_DEVICE.dependent_issue_cycles,
    issue_per_cycle: SM_86_89_ISSUE_PER_CYCLE,
    resident_lanes: SM_86_89_RESIDENT_LANES,
};

/// The cost model's ratios of a device nobody identified them on, and of Strix Halo behind RADV: the POOT-1403 fit
/// of the LDS read and the staged value (every tiling of a qwen3-0.6b prefill at 512 and 2048 tokens, ROCm), which
/// repeats within about 6 % on ROCm but not behind RADV (POOT-1494: the staged ratio moved 38 % between two runs and
/// 1.5-2x with the corpus), and the Gemv's direct value, which the POOT-1489 microbench measured behind RADV: the one
/// value at which the model's Gemv-versus-grouped choices are never slower than the retired 32-row switch, 7.6.
pub const PORTABLE_COST_RATIOS: CostRatios = CostRatios {
    lds_read: 1.0,
    staged_value: 8.0,
    direct_value: 7.6,
    fragment_round: None,
    group_scan: None,
    gemv_dot: None,
};

/// The cost model's ratios on Strix Halo through HSA (ROCm), identified by `calibrate-launch` over every candidate
/// tiling of 8 shapes (POOT-1494, two runs of 80 rows: LDS read 10.2 and 9.6, staged value 8.6 and 8.2, repeating
/// within about 6 %), refitted over the 160 rows together: LDS read 9.88, staged value 8.43, R^2 0.93. The direct
/// value is the POOT-1489 microbench's range in which the model's Gemv-versus-grouped choices are never slower than
/// the retired 32-row switch (11.9 to 23.6), cut at the top by the dense decode Gemv: above 15.2 the model puts a
/// one-row contraction of a vocabulary-sized output onto a matrix unit, which is slower (POOT-1503). The centre of
/// 11.9 to 15.2.
///
/// The grouped fragment tilings and the grouping of rows by expert were identified on the corpus of POOT-1547
/// (`scripts/calibrate-launch rocm fragment`, rows in `benchmarks/calibration/rocm-gfx1151.tsv`): the packed grouped
/// projections IQ2_XS up and IQ4_NL down of a 512-expert mixture at 128, 512 and 2048 tokens, four dense prefill
/// projections and two vocabulary-head decodes, every tiling the device runs exact and narrow plus the Gemv, 611 rows
/// with the dispatches timed one by one. A fragment tiling over rows grouped by expert costs its multiplies at the
/// matrix unit's rate plus [`CostRatios::fragment_round`] FMA issues per `K` round and nothing for staging (no LDS
/// read or staged value is identified on one: the fit puts both at zero; the dense fragment cells keep the scalar
/// model, which serves them); a grouping scan step costs [`CostRatios::group_scan`] (4.6 direct reads). The round is
/// the interior best-fit point of the grid cells that minimise the worst per-cell regret and then the mean. Under these
/// figures and a tie rule that takes the deeper round (the grouped fragment tilings tie in the model) the packed cells
/// choose within 1.02 of the fastest measured schedule (the shipped model: 1.49), and the IQ2_XS up projection at 128
/// tokens, which the shipped model lost 2.5 % to the Gemv on, narrows to a tiling 14 % faster than the Gemv.
/// `benchmarks/calibration/fit_fragment_cost.py` recomputes both figures from the rows. A packed format the fit never
/// saw (Q4_K up and down, 264 rows in `rocm-gfx1151-heldout-q4k.tsv`, POOT-1565) chooses within 1.26 of the fastest
/// schedule, mean 1.180, where the scalar model of before the fit scores 1.218 and 1.125: the round and the grouping
/// win the fitted formats and not the unfitted one on the mean. Refitting over both corpora puts the round at 673 inside
/// the same optimal range (141 to 800) and changes no choice.
pub const STRIX_HALO_HSA_COST_RATIOS: CostRatios = CostRatios {
    lds_read: 9.88,
    staged_value: 8.43,
    direct_value: 13.4,
    fragment_round: Some(800.0),
    group_scan: Some(61.4),
    gemv_dot: None,
};

/// The Gemv's direct value on the RTX PRO 6000 Blackwell Server (sm_120), identified by `calibrate-launch indexed` on
/// that part (POOT-1490): the centre of the range in which the model's Gemv-versus-grouped choices are never slower
/// than the retired 32-row switch, 11.9 to 16.8. It is not [`PORTABLE_COST_RATIOS`]'s 7.6, which lies below the range.
pub const SM_120_DIRECT_VALUE_COST: f64 = 14.15;

/// The Gemv's direct value and per-dot-product cost on the RTX 4090 (sm_89), identified together over the two
/// `calibrate-launch` corpora of that part (POOT-1602 fit): the fragment corpus of POOT-1599
/// (`benchmarks/calibration/ptx-sm89.tsv`, 551 rows) and the Gemv-versus-grouped corpus of POOT-1490
/// (`ptx-sm89-indexed.tsv`, 88 cells). With no dot term no direct value lets both packed cells at 128 tokens of the
/// fragment corpus (the IQ2_XS up projection, K 2560, where the Gemv is fastest, and the IQ4_NL down projection, K 640,
/// where a narrowed tile is 1.46x faster than the Gemv) choose within 1.10 of their fastest schedule. With it, the grid
/// of one octave in four over both figures, scored by the worst cell regret over both corpora and then the mean, is
/// optimal on a ridge of six points (direct values 3.4 to 4.8, a lower one taking a dearer dot): worst 1.455 (a dense
/// cell neither figure touches), mean 1.013 against 1.035 at the portable prior; these are the point nearest the
/// ridge's centre. On the fragment corpus alone they score mean 1.083 against 1.099 at the prior, the up cell choosing
/// the Gemv and the down cell the narrowed tile (1.09). `fit_fragment_cost.py --device sm89` recomputes them.
pub const SM_89_COST_RATIOS: CostRatios = CostRatios {
    lds_read: PORTABLE_COST_RATIOS.lds_read,
    staged_value: PORTABLE_COST_RATIOS.staged_value,
    direct_value: 4.0,
    fragment_round: None,
    group_scan: None,
    gemv_dot: Some(2378.0),
};

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

/// The device figures the packed Gemv's launch score weighs a body's cost with (`poot-kernelgen`'s `GemvBodyCost`),
/// in cycles of one compute unit's clock, so the clock itself never enters a score. The first three are rates only a
/// measurement gives; the last two are structural facts of the unit.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GemvDevice {
    /// Cycles a load waits for DRAM under load: what a lane's loads of one trip cost before its decode may use them.
    pub dram_latency_cycles: f64,
    /// Cache lines per cycle one unit's L1 serves.
    pub l1_lines_per_cycle: f64,
    /// Cycles one Gemv dispatch costs besides its trips: launch, ramp, the column tree and the store.
    pub dispatch_cycles: f64,
    /// Cycles between two dependent instructions of one warp: the chain a warp runs after its loads land, which
    /// the latency arm of a trip adds to the DRAM latency per instruction of the trip.
    pub dependent_issue_cycles: f64,
    /// Warp instructions one unit issues per cycle (its schedulers).
    pub issue_per_cycle: f64,
    /// Lanes one unit holds resident.
    pub resident_lanes: u32,
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
    /// On a device that identified it, the FMA issues of one lane a `K` round of a fragment tiling over rows grouped
    /// by expert costs (the barriers and the waits on its staged loads); such a tiling is then priced by its
    /// multiplies and its rounds alone. `None`, and every other tiling, prices a tile like a scalar one (an LDS read
    /// and a staged value per operand value).
    pub fragment_round: Option<f64>,
    /// On a device that identified it, one scan step of the expert-grouping kernel (`ExpertGroup`: one workgroup whose
    /// lanes each scan the row ids for their buckets), in FMA issues of one lane: a grouped contraction then pays
    /// the grouping before its first tile. `None` leaves the grouping unpriced.
    pub group_scan: Option<f64>,
    /// On a device that identified it, what one Gemv dot product costs whatever its `K`, in FMA issues of one lane:
    /// its weight row's stream start, its row's base and its fold and store (identified, not derived: the measured
    /// Gemv costs about 1.3x its per-`K` price at a `K` of 512 to 640 against 2560, on two vendors and for one format
    /// at both, POOT-1602). It is what makes a short-`K` Gemv lose to a tile its long-`K` twin beats. `None` leaves it
    /// unpriced.
    pub gemv_dot: Option<f64>,
}

/// The performance figures one device's launches were measured with. A figure here only ranks schedules that are
/// all correct, so a device nobody measured plans with [`PerfCalibration::PORTABLE`] and the plan says so
/// ([`Calibrated::Portable`]); a safety figure (a watchdog budget, a miscompile ceiling) is [`Calibration`]'s and
/// stays absent for such a device.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PerfCalibration {
    /// The Gemv columns of a dense `[N, K]` weight.
    pub dense_gemv_cols: GemvCols,
    /// The figures the packed Gemv's launch score weighs a body's cost with.
    pub packed_gemv: GemvDevice,
    /// The workgroup width of the LDS-tree attention Gemv (`attn_scores_v_gemv_lds`: one output per workgroup,
    /// the decode scores-by-V matmul), where the width is not the saturating residency.
    pub lds_gemv_width: u32,
    /// The register-tiled contraction cost model's ratios.
    pub cost: CostRatios,
    /// The keys per staged block of the tiled attention schedule, widest first.
    pub attention_key_blocks: &'static [usize],
    /// The fewest steps of a clustered attention walk a split-KV share takes: the floor of a share is `clusters`
    /// times this many keys. A ranking figure: a smaller floor splits a short cache into more workgroups.
    pub attention_split_min_steps: u32,
}

impl PerfCalibration {
    /// The prior an unmeasured device plans with. Every field is named for the device it was measured on or the
    /// launch every device ran before it was measured, never a value nobody observed.
    pub const PORTABLE: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv: PORTABLE_GEMV_DEVICE,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: PORTABLE_COST_RATIOS,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
    };

    /// Strix Halo (gfx1151) through HSA (ROCm): the device the Gemv figures were measured on, with the cost ratios
    /// `calibrate-launch` identified on ROCm.
    pub const STRIX_HALO_HSA: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv: PORTABLE_GEMV_DEVICE,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: STRIX_HALO_HSA_COST_RATIOS,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
    };

    /// Strix Halo (gfx1151) behind RADV (wgpu, raw Vulkan): the same silicon and Gemv figures, but the cost ratios
    /// `calibrate-launch` could not identify behind RADV (POOT-1494) are spelled as [`Self::PORTABLE`]'s.
    pub const STRIX_HALO_VULKAN: Self = Self {
        dense_gemv_cols: STRIX_HALO_DENSE_GEMV_COLS,
        packed_gemv: PORTABLE_GEMV_DEVICE,
        lds_gemv_width: STRIX_HALO_LDS_GEMV_WIDTH,
        cost: Self::PORTABLE.cost,
        attention_key_blocks: &STRIX_HALO_ATTENTION_KEY_BLOCKS,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
    };

    /// The RTX PRO 6000 Blackwell Server (compute capability 12.0). Measured for the packed Gemv run (POOT-1405), the
    /// Gemv's direct value (POOT-1490) and the packed Gemv columns (POOT-1319); a field it did not identify is spelled
    /// out as [`Self::PORTABLE`]'s, so a reader sees which figures are borrowed: the LDS read and the staged value (the
    /// tiling fit is underdetermined on PTX: the 80 candidate rows are one tile family), the dense Gemv columns table
    /// (the dense sweep ties at every output count but one), the LDS-tree Gemv width and the attention key blocks.
    pub const SM_120: Self = Self {
        dense_gemv_cols: Self::PORTABLE.dense_gemv_cols,
        packed_gemv: SM_120_GEMV_DEVICE,
        lds_gemv_width: Self::PORTABLE.lds_gemv_width,
        cost: CostRatios {
            lds_read: Self::PORTABLE.cost.lds_read,
            staged_value: Self::PORTABLE.cost.staged_value,
            direct_value: SM_120_DIRECT_VALUE_COST,
            fragment_round: None,
            group_scan: None,
            gemv_dot: None,
        },
        attention_key_blocks: Self::PORTABLE.attention_key_blocks,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
    };

    /// The RTX 4090 (compute capability 8.9). Measured for the Gemv's direct value and per-dot-product cost
    /// ([`SM_89_COST_RATIOS`], POOT-1602); the packed Gemv's figures are [`SM_89_GEMV_DEVICE`]'s (POOT-1613, POOT-1614)
    /// and every other figure is [`Self::PORTABLE`]'s, spelled out as borrowed.
    pub const SM_89: Self = Self {
        dense_gemv_cols: Self::PORTABLE.dense_gemv_cols,
        packed_gemv: SM_89_GEMV_DEVICE,
        lds_gemv_width: Self::PORTABLE.lds_gemv_width,
        cost: SM_89_COST_RATIOS,
        attention_key_blocks: Self::PORTABLE.attention_key_blocks,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
    };

    /// The RTX 3090 and A5000 (compute capability 8.6). Nobody identified a cost ratio on it (POOT-1605 ran POOT-1602's
    /// row on an RTX A5000 pod: the portable ratios plan it within 1.10 of the fastest schedule, 1.09), so every
    /// ratio is [`Self::PORTABLE`]'s, spelled out as borrowed; the packed Gemv's two structural facts are the SM's and its
    /// rates priors ([`SM_86_GEMV_DEVICE`], POOT-1613).
    pub const SM_86: Self = Self {
        dense_gemv_cols: Self::PORTABLE.dense_gemv_cols,
        packed_gemv: SM_86_GEMV_DEVICE,
        lds_gemv_width: Self::PORTABLE.lds_gemv_width,
        cost: PORTABLE_COST_RATIOS,
        attention_key_blocks: Self::PORTABLE.attention_key_blocks,
        attention_split_min_steps: PORTABLE_ATTENTION_SPLIT_MIN_STEPS,
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
    /// The RTX 4090 (compute capability 8.9).
    Sm89,
    /// The RTX 3090 and A5000 (compute capability 8.6).
    Sm86,
}

impl MeasuredDevice {
    /// What was measured on this device.
    pub const fn figures(self) -> PerfCalibration {
        match self {
            Self::StrixHaloHsa => PerfCalibration::STRIX_HALO_HSA,
            Self::StrixHaloVulkan => PerfCalibration::STRIX_HALO_VULKAN,
            Self::Sm120 => PerfCalibration::SM_120,
            Self::Sm89 => PerfCalibration::SM_89,
            Self::Sm86 => PerfCalibration::SM_86,
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

    /// The RTX 4090's measured entry.
    pub const SM_89: Self = Self::Measured(MeasuredDevice::Sm89);

    /// The RTX 3090 and A5000's measured entry.
    pub const SM_86: Self = Self::Measured(MeasuredDevice::Sm86);

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
        match (major, minor) {
            (12, 0) => Self::SM_120,
            (8, 9) => Self::SM_89,
            (8, 6) => Self::SM_86,
            _ => Self::Portable,
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
            name: "PORTABLE_GEMV_DRAM_LATENCY_CYCLES",
            value: 1600.0,
            source: "prior, not measured: the POOT-1550 design model's 800 ns loaded DRAM latency (receipts/1550-design/model.py LATENCY_NS) at a ~2 GHz unit clock. A device nobody identified plans with it and the plan records that",
        },
        Provenance {
            name: "PORTABLE_GEMV_L1_LINES_PER_CYCLE",
            value: 1.0,
            source: "prior, not measured: the design model's one 128-byte line per cycle per unit (model.py L1_LINES_PER_CYCLE), which the POOT-1563 probe measures",
        },
        Provenance {
            name: "PORTABLE_GEMV_DISPATCH_CYCLES",
            value: 8000.0,
            source: "prior, not measured: the design model's 4 us fixed cost per Gemv dispatch (model.py FIXED_US) at a ~2 GHz unit clock; the POOT-1319 fit of 35 K-quant rows gave 13.9 us on sm_120 in a per-dispatch-timed window, an upper bound",
        },
        Provenance {
            name: "PORTABLE_GEMV_DEPENDENT_ISSUE_CYCLES",
            value: 4.0,
            source: "prior, not measured: a warp's dependent-ALU issue interval, about 4 cycles on Ampere and later and the same order on RDNA (Volkov, Understanding Latency Hiding on GPUs, UC Berkeley EECS-2016-143); POOT-1609 design section 6",
        },
        Provenance {
            name: "PORTABLE_GEMV_ISSUE_PER_CYCLE",
            value: 2.0,
            source: "structural: an RDNA3.5 compute unit has two SIMD32, each issuing one wave instruction per cycle (AMD RDNA3 ISA guide); the floor over the parts poot targets",
        },
        Provenance {
            name: "PORTABLE_GEMV_RESIDENT_LANES",
            value: 1024.0,
            source: "structural: an RDNA3.5 SIMD holds 16 waves of 32 lanes, two SIMDs a compute unit (AMD RDNA3 ISA guide)",
        },
        Provenance {
            name: "SM_120_GEMV_DRAM_LATENCY_CYCLES",
            value: 2070.0,
            source: "not identified, POOT-1611: log least squares over the 2440 measured geometries of 15 cells (POOT-1608/1610/1611) gives 1785 cycles, and the region within 5 % of the least log error spans 708-2833 (x4.0), flat over a 4x range, so the corpus does not identify it. 2070 (0.85 us, a loaded latency) is the point of the region with the least worst-cell regret together with the L1 rate and the dependent issue interval; the design prior 1944 chooses identically. Trades with the dependent issue interval along a ridge. Held out by cell the worst regret is 1.331 (Q4_K 1024x2560, which every refit then sends to depth 2), by format 1.465 (Q4_K out), in sample 1.183 (Q6_K 1024x2560)",
        },
        Provenance {
            name: "SM_120_GEMV_L1_LINES_PER_CYCLE",
            value: 0.77,
            source: "identified, POOT-1611: log least squares over the 2440-row corpus gives 0.756 lines per cycle, the region within 5 % of the least error spans 0.756-0.953 (x1.26), and 0.77 lies in it. Mechanism not established: the POOT-1610 probe is consistent with the L1 cost scaling with the lanes' addresses, not distinct lines, but `ncu` counters were unavailable (ERR_NVGPUCTRPERM), so the compiler splitting the column groups' loads is not excluded",
        },
        Provenance {
            name: "SM_120_GEMV_DISPATCH_CYCLES",
            value: 4450.0,
            source: "measured, not identified by any geometry sweep: the 1.83 us empty launch of the POOT-1606 probe at 2.43 GHz. The dispatch is added to every geometry of a cell, so no corpus moves an argmin with it; it matters only where a Gemv is priced against a tile",
        },
        Provenance {
            name: "SM_120_GEMV_DEPENDENT_ISSUE_CYCLES",
            value: 2.9,
            source: "identified within a ridge, POOT-1611: log least squares gives 4.85, the region within 5 % of the least error spans 1.9-6.1 (x3.2), and 2.9 is the point of it with the least worst-cell regret at the committed latency and L1 (1.183, mean 1.088; at 3.6 and above the worst is 1.331 through Q4_K 1024x2560). Trades with the latency along a ridge; the prior was 4",
        },
        Provenance {
            name: "SM_120_GEMV_ISSUE_PER_CYCLE",
            value: 4.0,
            source: "structural: four warp schedulers per SM, one warp instruction each per cycle (NVIDIA Blackwell tuning guide, compute capability 12.0)",
        },
        Provenance {
            name: "SM_120_GEMV_RESIDENT_LANES",
            value: 1536.0,
            source: "structural: 48 resident warps of 32 lanes per SM at compute capability 12.0 (CUDA programming guide, technical specifications table)",
        },
        Provenance {
            name: "SM_89_GEMV_DRAM_LATENCY_CYCLES",
            value: 2016.0,
            source: "prior, NOT identified, POOT-1613/POOT-1614: the design model's 800 ns loaded latency at the RTX 4090's 2.52 GHz boost clock. The 852-row sweep of 5 cells (ptx-sm89-packed-gemv.tsv) fits 2012 cycles but the region within 5 % of the least error spans 503-2534 (x5), so the corpus does not identify it; a pod sweep with more cells replaces it",
        },
        Provenance {
            name: "SM_89_GEMV_L1_LINES_PER_CYCLE",
            value: 0.824,
            source: "identified, POOT-1614: log least squares over the 852 rows of the 5 cells of one RTX 4090 (secure pod, ptx-sm89-packed-gemv.tsv, sha256 db5f58bb), 0.824 lines per cycle (sm_120: 0.77). Identified within the region the sweep spans (0.75-0.85 across the leave-one-cell-out refits); one part, five cells",
        },
        Provenance {
            name: "SM_89_GEMV_DISPATCH_CYCLES",
            value: 4610.0,
            source: "borrowed, not measured on sm_89: sm_120's measured empty launch of 1.83 us (POOT-1606) at the 4090's 2.52 GHz. No geometry sweep moves an argmin with it; it matters only where a Gemv is priced against a tile. An empty-launch probe on the 4090 replaces it",
        },
        Provenance {
            name: "SM_89_GEMV_DEPENDENT_ISSUE_CYCLES",
            value: 4.0,
            source: "prior, NOT identified, POOT-1613/POOT-1614: the portable prior, a warp's dependent-ALU issue interval of about 4 cycles on Ampere and later. The sm_89 sweep fits 1.08 but the region spans 0.27-4.3 (x16), so it is not identified",
        },
        Provenance {
            name: "SM_86_GEMV_DRAM_LATENCY_CYCLES",
            value: 1600.0,
            source: "prior, not measured on sm_86: the portable prior (PORTABLE_GEMV_DRAM_LATENCY_CYCLES). No packed Gemv sweep exists on an sm_86 part, and sm_89's figures are not carried over, so every sm_86 rate is a prior until a pod sweep identifies it",
        },
        Provenance {
            name: "SM_86_GEMV_L1_LINES_PER_CYCLE",
            value: 1.0,
            source: "prior, not measured on sm_86: the portable prior (PORTABLE_GEMV_L1_LINES_PER_CYCLE). sm_89's identified 0.824 is that part's and is not assumed for sm_86",
        },
        Provenance {
            name: "SM_86_GEMV_DISPATCH_CYCLES",
            value: 8000.0,
            source: "prior, not measured on sm_86: the portable prior (PORTABLE_GEMV_DISPATCH_CYCLES), the design model's 4 us at a ~2 GHz unit clock",
        },
        Provenance {
            name: "SM_86_GEMV_DEPENDENT_ISSUE_CYCLES",
            value: 4.0,
            source: "prior, not measured on sm_86: the portable prior (PORTABLE_GEMV_DEPENDENT_ISSUE_CYCLES)",
        },
        Provenance {
            name: "SM_86_89_GEMV_ISSUE_PER_CYCLE",
            value: 4.0,
            source: "structural, sm_86 and sm_89: four warp schedulers per SM, one warp instruction each per cycle (NVIDIA Ampere GA10x tuning guide, compute capability 8.6, and Ada tuning guide, 8.9); CUDA programming guide, compute capability table. It replaces the RDNA3.5 prior of 2 that these parts planned with",
        },
        Provenance {
            name: "SM_86_89_GEMV_RESIDENT_LANES",
            value: 1536.0,
            source: "structural: 48 resident warps of 32 lanes per SM at compute capabilities 8.6 and 8.9 (CUDA programming guide, technical specifications per compute capability: maximum resident warps per multiprocessor 48). It replaces the RDNA3.5 prior of 1024",
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
            name: "STRIX_HALO_HSA_FRAGMENT_ROUND_COST",
            value: 800.0,
            source: "POOT-1547 calibrate-launch fragment, gfx1151 ROCm, 611 rows (packed grouped IQ2_XS up and IQ4_NL down at 128, 512 and 2048 tokens, dense prefill and decode-head shapes; every tiling exact and narrow plus the Gemv; dispatches timed one by one; benchmarks/calibration/rocm-gfx1151.tsv): the FMA issues per K round of a grouped fragment tiling that minimise the worst per-cell regret (1.663, the dense 64-row cell the round does not touch) and then the mean (1.174 against 1.222) with the fragment multiply at the matrix unit's rate and no staging term; the interior best-fit point of the optimal cells (141 to 951), recomputed by fit_fragment_cost.py",
        },
        Provenance {
            name: "STRIX_HALO_HSA_GROUP_SCAN_COST",
            value: 61.4,
            source: "POOT-1547, the same corpus: the expert-grouping dispatch (one workgroup, 2 scans of the rows per bucket a lane owns) measured 0.0468 us per scan step over the packed rows whatever the block, which is 61.4 FMA issues at the unit the scalar fit sets (least squares, fit_fragment_cost.py)",
        },
        Provenance {
            name: "SM_120_DIRECT_VALUE_COST",
            value: 14.15,
            source: "POOT-1490 calibrate-launch indexed on one RTX PRO 6000 Blackwell Server (sm_120, secure pod, 88 cells, 176 rows): the direct values at which the model's choices are never slower than the retired 32-row switch (within 5 %) and within 5 % of the best mean regret are 11.9 to 16.8; the geometric centre. Mean regret against the faster schedule 1.056 against the retired switch's 1.177. An RTX 4090 (sm_89) gave 7.6 to 11.9, so the figure moves between parts and this one is that part's; receipts/1490-cuda-sweep/",
        },
        Provenance {
            name: "SM_89_DIRECT_VALUE_COST",
            value: 4.0,
            source: "POOT-1602 fit (fit_fragment_cost.py --device sm89) over both calibrate-launch corpora of one RTX 4090 (sm_89, secure pods): the POOT-1599 fragment corpus (551 rows: packed grouped IQ2_XS up and IQ4_NL down at 128, 512 and 2048 tokens and the dense cells, every tiling exact and narrow plus the Gemv; benchmarks/calibration/ptx-sm89.tsv) and the POOT-1490 indexed corpus (88 cells of the Gemv against the grouped tiling; ptx-sm89-indexed.tsv), jointly with the dot cost, the LDS read and staged value at the prior: the point nearest the centre of the six grid points that minimise the worst cell regret over both corpora and then the mean (1.455, a dense cell, and 1.013 against 1.035 at the prior's 7.6 with no dot term; no direct value above 5.7 is optimal, and without the dot term none lets both 128-token packed cells choose within 1.10)",
        },
        Provenance {
            name: "SM_89_GEMV_DOT_COST",
            value: 2378.0,
            source: "POOT-1602, the same fit as SM_89_DIRECT_VALUE_COST: the K-independent FMA issues of one Gemv dot product at the ridge's centre point (optimal 1682 to 3364 at direct values 3.4 to 4.8). A least-squares fit of the Gemv rows alone gives the same order relative to the direct value on this part, gfx1151 and the gfx1151 Q4_K held-out corpus (242, 204 and 257 direct values per dot product, receipts/1602-gemv-dot/)",
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
            name: "PORTABLE_ATTENTION_SPLIT_MIN_STEPS",
            value: 4.0,
            source: "prior, not yet swept: the POOT-1549 design's roofline model (receipts/1549-design/model.txt section 5) puts a clustered walk step at 200 + 2 n_rep (e_k + e_v) cycles against the combine's 4.1 us fixed launch term, so a share under 4 steps costs more in the combine than the extra workgroup hides. The W7 pod sweep {2, 4, 8} at ISL 128 and 2048 replaces it per device",
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
    /// `SM_120_GEMV_DEVICE`'s latency without its row; the value check fails.
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
        for entry in [
            PerfCalibration::PORTABLE,
            PerfCalibration::STRIX_HALO_HSA,
            PerfCalibration::STRIX_HALO_VULKAN,
            PerfCalibration::SM_120,
            PerfCalibration::SM_89,
            PerfCalibration::SM_86,
        ] {
            let PerfCalibration {
                dense_gemv_cols,
                packed_gemv: _,
                lds_gemv_width,
                cost: _,
                attention_key_blocks,
                attention_split_min_steps,
            } = entry;
            assert_provenance(
                "PORTABLE_ATTENTION_SPLIT_MIN_STEPS",
                attention_split_min_steps,
            );
            // Every measured entry spells the Strix Halo figures (SM_120 borrows the dense columns from `PORTABLE`,
            // which are the same values): one row each, checked against whichever entry carries it.
            assert_provenance("STRIX_HALO_DENSE_GEMV_COLS_WIDE", dense_gemv_cols.wide);
            assert_provenance("STRIX_HALO_DENSE_GEMV_COLS_NARROW", dense_gemv_cols.narrow);
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
        for (device, [latency, l1, dispatch, dependent, issue, resident]) in [
            (
                PerfCalibration::PORTABLE.packed_gemv,
                [
                    "PORTABLE_GEMV_DRAM_LATENCY_CYCLES",
                    "PORTABLE_GEMV_L1_LINES_PER_CYCLE",
                    "PORTABLE_GEMV_DISPATCH_CYCLES",
                    "PORTABLE_GEMV_DEPENDENT_ISSUE_CYCLES",
                    "PORTABLE_GEMV_ISSUE_PER_CYCLE",
                    "PORTABLE_GEMV_RESIDENT_LANES",
                ],
            ),
            (
                PerfCalibration::SM_120.packed_gemv,
                [
                    "SM_120_GEMV_DRAM_LATENCY_CYCLES",
                    "SM_120_GEMV_L1_LINES_PER_CYCLE",
                    "SM_120_GEMV_DISPATCH_CYCLES",
                    "SM_120_GEMV_DEPENDENT_ISSUE_CYCLES",
                    "SM_120_GEMV_ISSUE_PER_CYCLE",
                    "SM_120_GEMV_RESIDENT_LANES",
                ],
            ),
            (
                PerfCalibration::SM_89.packed_gemv,
                [
                    "SM_89_GEMV_DRAM_LATENCY_CYCLES",
                    "SM_89_GEMV_L1_LINES_PER_CYCLE",
                    "SM_89_GEMV_DISPATCH_CYCLES",
                    "SM_89_GEMV_DEPENDENT_ISSUE_CYCLES",
                    "SM_86_89_GEMV_ISSUE_PER_CYCLE",
                    "SM_86_89_GEMV_RESIDENT_LANES",
                ],
            ),
            (
                PerfCalibration::SM_86.packed_gemv,
                [
                    "SM_86_GEMV_DRAM_LATENCY_CYCLES",
                    "SM_86_GEMV_L1_LINES_PER_CYCLE",
                    "SM_86_GEMV_DISPATCH_CYCLES",
                    "SM_86_GEMV_DEPENDENT_ISSUE_CYCLES",
                    "SM_86_89_GEMV_ISSUE_PER_CYCLE",
                    "SM_86_89_GEMV_RESIDENT_LANES",
                ],
            ),
        ] {
            let GemvDevice {
                dram_latency_cycles,
                l1_lines_per_cycle,
                dispatch_cycles,
                dependent_issue_cycles,
                issue_per_cycle,
                resident_lanes,
            } = device;
            assert_provenance(latency, dram_latency_cycles);
            assert_provenance(l1, l1_lines_per_cycle);
            assert_provenance(dispatch, dispatch_cycles);
            assert_provenance(dependent, dependent_issue_cycles);
            assert_provenance(issue, issue_per_cycle);
            assert_provenance(resident, resident_lanes);
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
        assert_provenance(
            "STRIX_HALO_HSA_FRAGMENT_ROUND_COST",
            hsa.fragment_round
                .expect("gfx1151 through HSA identified the fragment round"),
        );
        assert_provenance(
            "STRIX_HALO_HSA_GROUP_SCAN_COST",
            hsa.group_scan
                .expect("gfx1151 through HSA identified the grouping scan"),
        );
        assert_eq!(
            (
                portable.fragment_round,
                portable.group_scan,
                portable.gemv_dot
            ),
            (None, None, None),
            "the prior identifies neither the fragment round, the grouping nor the dot cost"
        );
        assert_eq!(hsa.gemv_dot, None, "gfx1151 identified no dot cost");
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
        assert_eq!(sm_120.gemv_dot, None, "sm_120 identified no dot cost");
        // sm_89 identified the direct value and the dot cost together; the tile ratios are the prior's.
        let sm_89 = PerfCalibration::SM_89.cost;
        assert_provenance("SM_89_DIRECT_VALUE_COST", sm_89.direct_value);
        assert_provenance(
            "SM_89_GEMV_DOT_COST",
            sm_89.gemv_dot.expect("sm_89 identified the dot cost"),
        );
        assert_eq!(
            (
                sm_89.lds_read,
                sm_89.staged_value,
                sm_89.fragment_round,
                sm_89.group_scan
            ),
            (portable.lds_read, portable.staged_value, None, None),
            "sm_89 borrows the prior's tile ratios"
        );
        // sm_86 and sm_89 share one SM: the same two structural facts, 4 schedulers and 1536 lanes, not the RDNA3.5
        // prior's (POOT-1613). sm_86 has no packed Gemv sweep, so every one of its rates is the prior's; sm_89
        // identified the L1 rate only (POOT-1614).
        let prior = PerfCalibration::PORTABLE.packed_gemv;
        let (sm_86_gemv, sm_89_gemv) = (
            PerfCalibration::SM_86.packed_gemv,
            PerfCalibration::SM_89.packed_gemv,
        );
        for gemv in [sm_86_gemv, sm_89_gemv] {
            assert_eq!(
                (gemv.issue_per_cycle, gemv.resident_lanes),
                (4.0, 1536),
                "cc 8.6 and 8.9 have four schedulers and 48 resident warps"
            );
        }
        assert_eq!(
            GemvDevice {
                issue_per_cycle: prior.issue_per_cycle,
                resident_lanes: prior.resident_lanes,
                ..sm_86_gemv
            },
            prior,
            "every rate of sm_86 is the prior's"
        );
        assert_eq!(
            (
                sm_89_gemv.l1_lines_per_cycle,
                sm_89_gemv.dependent_issue_cycles
            ),
            (0.824, prior.dependent_issue_cycles),
            "sm_89 identified the L1 rate and left the dependent issue interval at the prior"
        );
        assert_eq!(
            PerfCalibration::SM_86.cost,
            portable,
            "sm_86 identified no cost ratio"
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
            "STRIX_HALO_LDS_GEMV_WIDTH",
            "PORTABLE_LDS_READ_COST",
            "PORTABLE_STAGED_VALUE_COST",
            "PORTABLE_DIRECT_VALUE_COST",
            "STRIX_HALO_HSA_LDS_READ_COST",
            "STRIX_HALO_HSA_STAGED_VALUE_COST",
            "STRIX_HALO_HSA_DIRECT_VALUE_COST",
            "STRIX_HALO_HSA_FRAGMENT_ROUND_COST",
            "STRIX_HALO_HSA_GROUP_SCAN_COST",
            "SM_120_DIRECT_VALUE_COST",
            "SM_89_DIRECT_VALUE_COST",
            "SM_89_GEMV_DOT_COST",
            "STRIX_HALO_ATTENTION_KEY_BLOCK_WIDEST",
            "STRIX_HALO_ATTENTION_KEY_BLOCK_NARROWEST",
            "GFX11_MATRIX_FMA_RATE",
            "SM80_MATRIX_FMA_RATE",
            "SM90_MATRIX_FMA_RATE",
            "GEFORCE_MATRIX_FMA_RATE",
            "PORTABLE_ATTENTION_SPLIT_MIN_STEPS",
            "PORTABLE_GEMV_DRAM_LATENCY_CYCLES",
            "PORTABLE_GEMV_L1_LINES_PER_CYCLE",
            "PORTABLE_GEMV_DISPATCH_CYCLES",
            "PORTABLE_GEMV_DEPENDENT_ISSUE_CYCLES",
            "PORTABLE_GEMV_ISSUE_PER_CYCLE",
            "PORTABLE_GEMV_RESIDENT_LANES",
            "SM_120_GEMV_DRAM_LATENCY_CYCLES",
            "SM_120_GEMV_L1_LINES_PER_CYCLE",
            "SM_120_GEMV_DISPATCH_CYCLES",
            "SM_120_GEMV_DEPENDENT_ISSUE_CYCLES",
            "SM_120_GEMV_ISSUE_PER_CYCLE",
            "SM_120_GEMV_RESIDENT_LANES",
            "SM_89_GEMV_DRAM_LATENCY_CYCLES",
            "SM_89_GEMV_L1_LINES_PER_CYCLE",
            "SM_89_GEMV_DISPATCH_CYCLES",
            "SM_89_GEMV_DEPENDENT_ISSUE_CYCLES",
            "SM_86_GEMV_DRAM_LATENCY_CYCLES",
            "SM_86_GEMV_L1_LINES_PER_CYCLE",
            "SM_86_GEMV_DISPATCH_CYCLES",
            "SM_86_GEMV_DEPENDENT_ISSUE_CYCLES",
            "SM_86_89_GEMV_ISSUE_PER_CYCLE",
            "SM_86_89_GEMV_RESIDENT_LANES",
        ]);
    }

    /// A device's figures follow its identity, not its runtime family: Strix Halo is measured through HSA and
    /// through Vulkan, another AMD, an Intel or an NVIDIA Vulkan device is portable, and CUDA is measured on
    /// sm_120, sm_89 and sm_86 only (sm_86 and sm_89 share their packed Gemv figures, not an entry). Mutation: key the Vulkan arm on the vendor alone; the discrete AMD card reads `Measured`.
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
        assert_eq!(
            Calibrated::of_cuda_compute_capability(8, 9),
            Calibrated::SM_89
        );
        assert_eq!(
            Calibrated::of_cuda_compute_capability(8, 6),
            Calibrated::SM_86
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

    /// The packed Gemv figures an unmeasured device plans with are the portable prior's, whichever runtime drives it, and
    /// a measured one carries its own: the launch a plan is made with names which. Mutation: make `figures` return
    /// the measured figures for `Portable`.
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
        assert_eq!(
            portable.calibrated.figures().packed_gemv,
            PORTABLE_GEMV_DEVICE
        );
        assert_eq!(sm_120.calibrated.figures().packed_gemv, SM_120_GEMV_DEVICE);
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
