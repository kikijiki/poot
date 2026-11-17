//! The Intel NPU (VPU) compilation target: its typed identity and the capabilities a plan is admitted
//! against (POOT-1159, design npu-targets-1093 section 4.1).
//!
//! A VPU target is not a GPU and not an AIE: poot owns no core ISA or kernel toolchain for it. Its only
//! entry point is a vendor graph compiler (OpenVINO IR through the driver compiler), so each equation
//! lowers to a vendor opset op and a region of equations compiles to one blob. It has its own descriptor,
//! and a limit nobody measured is [`Queried::Unknown`], never a plausible number.

use crate::{Queried, StepWidths};

/// A VPU compilation target's identity. A closed enum: each variant fixes the platform and the
/// driver-compiler ABI poot lowers to, so adding one forces every match on it. A product or model name
/// never chooses a lowering; this does. A runtime device instance's identity is separate.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VpuTargetId {
    /// Intel NPU4 (Lunar Lake), driven through the `intel_vpu` kernel driver.
    Npu4,
}

impl VpuTargetId {
    /// The platform name inside the vendor compiler's options and cache identities.
    pub const fn platform(self) -> &'static str {
        match self {
            VpuTargetId::Npu4 => "NPU4000",
        }
    }

    /// The identity's canonical text, as it enters cache keys.
    pub fn canonical(&self) -> String {
        format!("vpu:platform={}", self.platform())
    }
}

/// The version of the driver compiler (`vclGetVersion`) that compiles a region. Part of every compiled
/// blob's cache key: a blob from one compiler version does not load under another.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VendorCompilerId {
    /// Compiler ABI major version.
    pub major: u32,
    /// Compiler ABI minor version.
    pub minor: u32,
}

impl VendorCompilerId {
    /// The version's canonical text, as it enters cache keys.
    pub fn canonical(&self) -> String {
        format!("vcl={}.{}", self.major, self.minor)
    }
}

/// The numerical contract a region admits F32 values under on a device that computes in F16 (POOT-1178,
/// POOT-ADR-0103): the planner converts the region's F32 inputs to F16 on entry and its F32 results back on
/// exit, the region's graph is typed over F16, and the contract bounds what that costs.
///
/// The reference is the F32 evaluation of the region (tier 2 of ADR-0101), and the bound is normwise: for every F32
/// result of a region, `max |narrowed - reference| / max |reference|`. The F16 types of the narrowed graph describe
/// the region's *interface* (what crosses its boundary), not a claim about every interior value: the interior
/// precision of a narrowed region is the vendor compiler's, at the compiler version the device rows pin
/// (`poot-npu/npu-stack.pin`). A vendor compiler may compute an interior chain wider than the graph declares, and
/// the Intel NPU's does: the first rmsnorm of Qwen2.5-0.5B's layer 3 squares a massive activation (about 848, so
/// 7.2e5) past F16's range, which a literal F16 evaluation of the graph turns into infinity (the CPU oracle's
/// evaluation of the narrowed graph does exactly that and diverges from layer 3 on), while the device stays
/// within the bound of the F32 evaluation. So the CPU evaluation of the narrowed graph is not the reference, and
/// a compiler that does not keep that interior width is outside the contract until the rows are re-validated.
///
/// The bound is a measured tolerance of the region classes the hardware rows ran, recorded here, not a proof for
/// any graph: a value with no contract is refused, never narrowed. Two figures are recorded, for two classes. The
/// contract's own `max_relative_error_ppm` is the fragment-class figure (a block's MLP and attention core). The
/// whole-step figure is wider: measured over 24 chained layers and two chunks of Qwen2.5-0.5B, the worst KV error is
/// 1.9e-2 and the logits error 8.3e-3, held to a tier 2 bound of 4e-2 by `poot-npu-rows`'s `vpu_qwen_prompt`.
///
/// Literals (POOT-1180). A finite F32 literal past FP16's range is refused (`Capability::NarrowingRange`), except as the operand of an `Add`
/// or `Sub` in a narrowed region: there a magnitude that large is a sentinel acting as an infinity (an attention
/// mask), and the contract saturates it to +-65504. A multiplicative use is still refused (`x * 1e9` is not
/// `x * 65504`), an infinite literal stays infinite and a NaN literal is refused. The one multiplicative exception
/// is a mask scale (POOT-1278): a literal factor of a value that comparisons and literals alone fix to -1, 0 or 1
/// (`(visible - 1) * 1e9`, the causal mask), whose product is exactly 0 or the sentinel, saturates like the additive
/// one. An integer converts to FP16 only when the graph declares it inside FP16's exact integers (`[-2048, 2048]`,
/// as a position declared below a short cache is): otherwise the conversion is refused.
///
/// The bound holds only for finite values the narrowed width can hold: an F32 value beyond
/// [`NarrowingContract::max_magnitude`] narrows to infinity and the error is unbounded. That range is a
/// precondition of the contract, checked where the host can: literals when the region is planned, a constant once, as it is
/// uploaded, the region's F32 inputs before each replay and its F32 results after it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NarrowingContract {
    /// The bound in parts per million of the reference result's largest magnitude.
    pub max_relative_error_ppm: u32,
    /// The largest magnitude an F32 input, literal or result may have for the bound to hold: the largest
    /// finite value of the narrower width (65504 for FP16).
    pub max_magnitude: u32,
}

impl NarrowingContract {
    /// Whether `value` is inside the range the bound holds for. NaN and infinity are not.
    pub fn admits(&self, value: f32) -> bool {
        value.abs() <= self.max_magnitude as f32
    }
}

/// What a VPU target offers a plan. Every runtime-measured bound is [`Queried`]: measured by the Level
/// Zero runtime, or `Unknown`. Admission refuses a plan whose requirement it cannot bound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VpuCaps {
    /// The target these capabilities describe.
    pub target: VpuTargetId,
    /// The driver compiler that compiles regions for this device.
    pub compiler: Queried<VendorCompilerId>,
    /// Bytes of device-visible host memory the program's arenas may total.
    pub memory_budget_bytes: Queried<u64>,
    /// The contract F32 values are narrowed to the part's native float width under; `None` when none is recorded,
    /// which makes every F32 arithmetic equation a typed refusal naming the missing contract.
    pub narrowing: Option<NarrowingContract>,
    /// The widths of a model step (rows times tokens) the device was configured to prepare: one vendor compile
    /// per width. A step of another width is refused before any compile. The operator states them when it opens
    /// the device; none is the empty set.
    pub step_widths: StepWidths,
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP16: NarrowingContract = NarrowingContract {
        max_relative_error_ppm: 5_000,
        max_magnitude: 65_504,
    };

    /// The range precondition at its edges: FP16's largest finite value is admitted, the next F32 above it, the
    /// negative twin, NaN and both infinities are not.
    /// Mutation: widen `admits` to `value.abs() <= self.max_magnitude as f32 + 1.0`; the 65504.5 row goes red.
    #[test]
    fn the_range_precondition_holds_at_the_edges_of_fp16() {
        for admitted in [0.0, -0.0, 1.0, 65_504.0, -65_504.0] {
            assert!(FP16.admits(admitted), "{admitted}");
        }
        for refused in [
            65_504.5,
            -65_504.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert!(!FP16.admits(refused), "{refused}");
        }
    }
}
