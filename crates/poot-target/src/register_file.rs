//! [`RegisterFile`]: the registers a GPU gives one lane and one workgroup (POOT-1447). The planner sizes the values
//! a tiled kernel holds across its loops (accumulators, held operand fragments) against it, so a tiling the
//! compiler would have to spill is refused before any code is generated.

use crate::Queried;

/// Accumulator registers a lane may hold across a loop, as a divisor of what it keeps resident ([`RegisterFile::per_lane_of`]):
/// a quarter. Structural (POOT-1479): the other three quarters carry the step's loaded operands, the staging
/// addresses, the loop state and the room that keeps several waves resident. The gfx1151 spill rows
/// (`poot-kernelgen`'s `register_budget`) are the evidence: 64 accumulators of a 256-register lane compile to 149 VGPRs
/// with none spilled, 128 spill.
pub(crate) const ACCUMULATOR_SHARE: u32 = 4;

/// Registers a lane may hold across a loop in all (accumulators and held operands), as a divisor of what it keeps
/// resident: a half. The attention fragment rule of POOT-1447, unchanged.
pub(crate) const HELD_SHARE: u32 = 2;

/// The accumulator registers a lane holds on a device whose register file is unknown: a quarter of the 256
/// registers a lane addresses on the devices poot measured (a gfx11 lane addresses 256 VGPRs, an NVIDIA thread
/// 255). A spill is slow, not wrong, so an unknown file plans with this figure and the launch records that it did
/// ([`RegisterBudget::file`]).
pub(crate) const PORTABLE_ACCUMULATORS: u32 = 64;

/// What one lane of a workgroup may hold in registers across a loop, derived from the device's [`RegisterFile`]
/// (POOT-1479): the planner sizes a register tile against it and holds no accumulator count of its own.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegisterBudget {
    /// The device's register file, or `Unknown` where its API reports none. It is part of every plan made with the
    /// budget, so two devices that differ only here plan differently and the plan names which.
    pub file: Queried<RegisterFile>,
}

impl RegisterBudget {
    /// The f32 accumulator registers each lane of a `lanes`-lane workgroup holds across a K or key loop: a quarter
    /// of what the lane keeps resident, rounded up (a lane holds whole registers: `ceil(255 / 4)` keeps an NVIDIA
    /// thread's 8 x 8 tile, which `floor` would refuse). [`PORTABLE_ACCUMULATORS`] where the register file is
    /// unknown.
    pub fn accumulators(self, lanes: u32) -> u32 {
        match self.file.known() {
            Some(file) => file.per_lane_of(lanes).div_ceil(ACCUMULATOR_SHARE),
            None => PORTABLE_ACCUMULATORS,
        }
    }

    /// The registers each lane of a `lanes`-lane workgroup holds across the key loop in all, accumulators and held
    /// operands: half of what the lane keeps resident. `None` where the register file is unknown: a held operand's
    /// budget comes from a file the device reports (POOT-1447), so no fragment tiling that holds one is chosen.
    pub fn held(self, lanes: u32) -> Option<u32> {
        self.file
            .known()
            .map(|file| file.per_lane_of(lanes) / HELD_SHARE)
    }

    /// The registers each lane of a `lanes`-lane workgroup keeps resident (POOT-1505): the lane's whole file, or the
    /// portable lane's 256 where it is unknown. The bound a candidate's count must fit when the device's compiler
    /// decides whether it spills: no lane holds more.
    pub fn lane_registers(self, lanes: u32) -> u32 {
        self.file
            .known()
            .map_or(PORTABLE_ACCUMULATORS * ACCUMULATOR_SHARE, |file| {
                file.per_lane_of(lanes)
            })
    }

    /// The registers each lane of a `lanes`-lane workgroup holds at once across a tiled contraction's `K` loop in all,
    /// its accumulators and the round it stages (POOT-1505): the held share, half of what the lane keeps resident.
    /// The other half carries the step's operand fragments, the staging addresses and the decode's temporaries.
    /// Where the register file is unknown, half of the portable lane's (four times [`PORTABLE_ACCUMULATORS`]): a
    /// spill is slow, not wrong, as for the accumulators. Evidence: of 32 packed expert tilings compiled for gfx1151
    /// (IQ2_XS and IQ4_NL, every 16-row fragment geometry and the long tiles at both round depths), the 14 the
    /// planner's count puts within 128 spill nothing, and every one that spills is over it; 4 of the 18 over it
    /// compile just under the file, 236 to 253 VGPRs (POOT-1505).
    pub fn loop_registers(self, lanes: u32) -> u32 {
        self.held(lanes)
            .unwrap_or(PORTABLE_ACCUMULATORS * ACCUMULATOR_SHARE / HELD_SHARE)
    }
}

/// The 32-bit registers a device gives a lane and a workgroup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegisterFile {
    /// Registers one lane addresses: what a kernel holds before the compiler spills.
    pub per_lane: u32,
    /// Registers, summed over lanes, the hardware one workgroup runs on holds (CUDA's `regsPerBlock`; on AMD the
    /// SIMDs a workgroup is placed on times their file). A workgroup of `n` lanes keeps at most
    /// `per_workgroup / n` registers per lane resident.
    pub per_workgroup: u32,
}

impl RegisterFile {
    /// The registers each lane of a `lanes`-lane workgroup can hold: the addressing limit or the workgroup's share
    /// of the file, whichever is smaller.
    pub const fn per_lane_of(self, lanes: u32) -> u32 {
        let share = match self.per_workgroup.checked_div(lanes) {
            Some(share) => share,
            None => self.per_workgroup,
        };
        if share < self.per_lane {
            share
        } else {
            self.per_lane
        }
    }

    /// An AMD GPU by gfx code and wavefront size, or `None` for a part whose file is not recorded here.
    ///
    /// gfx11 (RDNA3 and 3.5) in wave32: a lane addresses 256 VGPRs. gfx1100, gfx1101 and gfx1151 have 1536 per SIMD,
    /// the other gfx11 parts 1024 (LLVM's own occupancy: a kernel at 100 VGPRs runs 12 waves per SIMD on gfx1151
    /// and gfx1100, 9 on gfx1150; POOT-1447). poot compiles in WGP mode (the
    /// gfx10+ default), where a workgroup's waves share the four SIMDs of a workgroup processor.
    ///
    /// CDNA (gfx908, gfx90a, gfx940, gfx941, gfx942) in wave64: a compute unit has four SIMDs of 512 registers per
    /// lane each (128 KiB), 512 KiB in all (the CDNA 1, 2 and 3 whitepapers); a workgroup runs on one compute unit
    /// (there is no WGP mode). Of the 512, a lane addresses 256 architectural VGPRs; the other 256 are AGPRs
    /// (`v_accvgpr_*`, the matrix cores' accumulators), which a kernel with no matrix-core instruction reaches only
    /// as spill space. So `per_lane` is the 256 VGPRs a body keeps its values in, and a 256-lane workgroup's share
    /// of the file (512) does not bind it.
    pub fn amd(gfx: &str, wave: u32) -> Option<Self> {
        const SIMDS_PER_WORKGROUP: u32 = 4;
        let gfx11 = gfx.strip_prefix("gfx").is_some_and(|n| n.starts_with("11"));
        let cdna = matches!(gfx, "gfx908" | "gfx90a" | "gfx940" | "gfx941" | "gfx942");
        let per_simd = match (gfx11, cdna, wave) {
            (true, _, 32) => match gfx {
                "gfx1100" | "gfx1101" | "gfx1151" => 1536,
                _ => 1024,
            },
            (_, true, 64) => 512,
            _ => return None,
        };
        Some(Self {
            per_lane: 256,
            per_workgroup: SIMDS_PER_WORKGROUP * per_simd * wave,
        })
    }

    /// An NVIDIA GPU: 255 registers per thread and 65 536 per block on every compute capability from 5.0 (the CUDA
    /// programming guide's technical specifications table), `None` below it.
    pub fn nvidia(compute_capability_major: u32) -> Option<Self> {
        (compute_capability_major >= 5).then_some(Self {
            per_lane: 255,
            per_workgroup: 65_536,
        })
    }

    /// An AMD GPU through `VK_AMD_shader_core_properties`: `maxVgprAllocation` registers per lane, and
    /// `vgprsPerSimd` (counted in wave64 registers, 64 lanes each) over `simdPerComputeUnit` SIMDs per workgroup.
    /// The driver picks CU or WGP mode and the extension does not say which, so the workgroup's share is the
    /// compute unit's, which holds in both.
    pub const fn amd_shader_core(
        max_vgpr_allocation: u32,
        vgprs_per_simd: u32,
        simd_per_compute_unit: u32,
    ) -> Self {
        Self {
            per_lane: max_vgpr_allocation,
            per_workgroup: vgprs_per_simd * 64 * simd_per_compute_unit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gfx1151 has the full file (WGP of 4 x 1536 x 32), gfx1150 the small one, a wave64 or non-gfx11 part is not
    /// recorded. Mutation: give every gfx11 part 1024; gfx1151's `per_workgroup` reads 131072.
    #[test]
    fn amd_register_files_follow_the_gfx_code() {
        assert_eq!(
            RegisterFile::amd("gfx1151", 32),
            Some(RegisterFile {
                per_lane: 256,
                per_workgroup: 196_608
            })
        );
        assert_eq!(
            RegisterFile::amd("gfx1150", 32).map(|f| f.per_workgroup),
            Some(131_072)
        );
        assert_eq!(RegisterFile::amd("gfx1151", 64), None);
        // A CDNA compute unit: four SIMDs of 512 registers over 64 lanes, 256 of them addressed by a lane.
        for gfx in ["gfx908", "gfx90a", "gfx940", "gfx941", "gfx942"] {
            assert_eq!(
                RegisterFile::amd(gfx, 64),
                Some(RegisterFile {
                    per_lane: 256,
                    per_workgroup: 131_072
                }),
                "{gfx}"
            );
            assert_eq!(RegisterFile::amd(gfx, 32), None, "{gfx} runs wave64");
        }
        assert_eq!(RegisterFile::amd("gfx900", 64), None);
    }

    /// The accumulator budget is a quarter of a lane's resident registers, rounded up, on every reference file: 64 on
    /// gfx1151 (ROCm and RADV) and on sm_80 (`ceil(255 / 4)`), and a wide workgroup's share lowers it. An unknown
    /// file gives the portable 64 and no held budget. Mutation: round down; sm_80 reads 63 and loses its 8 x 8
    /// tile.
    #[test]
    fn the_accumulator_budget_is_a_rounded_up_quarter_of_the_lane() {
        let budget = |file| RegisterBudget {
            file: Queried::Known(file),
        };
        let rocm = budget(RegisterFile::amd("gfx1151", 32).unwrap());
        let radv = budget(RegisterFile::amd_shader_core(256, 768, 2));
        let sm_80 = budget(RegisterFile::nvidia(8).unwrap());
        for (name, budget) in [("rocm", rocm), ("radv", radv), ("sm_80", sm_80)] {
            assert_eq!(budget.accumulators(256), 64, "{name}");
        }
        assert_eq!(sm_80.accumulators(1024), 16);
        assert_eq!(sm_80.held(256), Some(127));
        assert_eq!(rocm.held(256), Some(128));
        let unknown = RegisterBudget {
            file: Queried::Unknown,
        };
        assert_eq!(unknown.accumulators(1024), PORTABLE_ACCUMULATORS);
        assert_eq!(unknown.held(256), None);
    }

    /// A lane of a wide workgroup holds the workgroup's share of the file, a narrow one the addressing limit.
    /// Mutation: return `per_lane` alone; the 1024-thread NVIDIA block reads 255, not 64.
    #[test]
    fn a_wide_workgroup_holds_its_share_of_the_file() {
        let nvidia = RegisterFile::nvidia(8).unwrap();
        assert_eq!(nvidia.per_lane_of(1024), 64);
        assert_eq!(nvidia.per_lane_of(128), 255);
        let gfx1151 = RegisterFile::amd("gfx1151", 32).unwrap();
        assert_eq!(gfx1151.per_lane_of(512), 256);
        assert_eq!(gfx1151.per_lane_of(1024), 192);
    }
}
