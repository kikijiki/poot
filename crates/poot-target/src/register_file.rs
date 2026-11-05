//! [`RegisterFile`]: the registers a GPU gives one lane and one workgroup (POOT-1447). The planner sizes the values
//! a tiled kernel holds across its loops (accumulators, held operand fragments) against it, so a tiling the
//! compiler would have to spill is refused before any code is generated.

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
    /// and gfx1100, 9 on gfx1150; `receipts/1447-attention-q-registers/design.md`). poot compiles in WGP mode (the
    /// gfx10+ default), where a workgroup's waves share the four SIMDs of a workgroup processor.
    pub fn amd(gfx: &str, wave: u32) -> Option<Self> {
        const SIMDS_PER_WORKGROUP: u32 = 4;
        let gfx11 = gfx.strip_prefix("gfx").is_some_and(|n| n.starts_with("11"));
        if !gfx11 || wave != 32 {
            return None;
        }
        let per_simd = match gfx {
            "gfx1100" | "gfx1101" | "gfx1151" => 1536,
            _ => 1024,
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
        assert_eq!(RegisterFile::amd("gfx90a", 64), None);
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
