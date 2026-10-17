//! The Intel NPU (VPU) compilation target: its typed identity and the capabilities a plan is admitted
//! against (POOT-1159, design npu-targets-1093 section 4.1).
//!
//! A VPU target is not a GPU and not an AIE: poot owns no core ISA or kernel toolchain for it. Its only
//! entry point is a vendor graph compiler (OpenVINO IR through the driver compiler), so each equation
//! lowers to a vendor opset op and a region of equations compiles to one blob. It has its own descriptor,
//! and a limit nobody measured is [`Queried::Unknown`], never a plausible number.

use crate::Queried;

/// The float width a VPU target computes in natively. An equation in a wider float is a narrowing of its
/// semantics, not a schedule reassociation, so the planner computes it narrower only inside a region that
/// carries a [`NarrowingContract`], and refuses it otherwise.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VpuArithmetic {
    /// IEEE binary16 arithmetic (NPU4). F32 and BF16 arithmetic is not native.
    F16,
}

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

    /// What this part computes in natively.
    pub const fn arithmetic(self) -> VpuArithmetic {
        match self {
            VpuTargetId::Npu4 => VpuArithmetic::F16,
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
/// POOT-ADR-0103): the planner converts the region's F32 inputs to F16 on entry, computes the whole region in
/// F16 and converts its F32 results back on exit, and the contract bounds what that costs.
///
/// The bound is normwise: for every F32 result of a region, `max |narrowed - reference| / max |reference|`,
/// the reference being the same region evaluated in F32. It is a measured tolerance of the region classes the
/// hardware rows ran, recorded here, not a proof for any graph: a value with no contract is refused, never
/// narrowed.
///
/// Literals (POOT-1180). A finite F32 literal past FP16's range is refused (`Capability::NarrowingRange`), except as the operand of an `Add`
/// or `Sub` in a narrowed region: there a magnitude that large is a sentinel acting as an infinity (an attention
/// mask), and the contract saturates it to +-65504. A multiplicative use is still refused (`x * 1e9` is not
/// `x * 65504`), an infinite literal stays infinite and a NaN literal is refused.
///
/// The bound holds only for finite values the narrowed width can hold: an F32 value beyond
/// [`NarrowingContract::max_magnitude`] narrows to infinity and the error is unbounded. That range is a
/// precondition of the contract, checked where the host can: literals when the region is planned, the region's
/// F32 inputs before each replay and its F32 results after it.
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
    /// The contract F32 values are narrowed to [`VpuCaps::arithmetic`] under; `None` when none is recorded,
    /// which makes every F32 arithmetic equation a typed refusal naming the missing contract.
    pub narrowing: Option<NarrowingContract>,
}

impl VpuCaps {
    /// The float width this device computes in natively (a property of the part, not measured).
    pub const fn arithmetic(&self) -> VpuArithmetic {
        self.target.arithmetic()
    }
}
