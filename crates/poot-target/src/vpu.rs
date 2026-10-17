//! The Intel NPU (VPU) compilation target: its typed identity and the capabilities a plan is admitted
//! against (POOT-1159, design npu-targets-1093 section 4.1).
//!
//! A VPU target is not a GPU and not an AIE: poot owns no core ISA or kernel toolchain for it. Its only
//! entry point is a vendor graph compiler (OpenVINO IR through the driver compiler), so each equation
//! lowers to a vendor opset op and a region of equations compiles to one blob. It has its own descriptor,
//! and a limit nobody measured is [`Queried::Unknown`], never a plausible number.

use crate::Queried;

/// The float width a VPU target computes in natively. An equation in a wider float is a narrowing of its
/// semantics, not a schedule reassociation, so the planner refuses it instead of computing it narrower.
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
}

impl VpuCaps {
    /// The float width this device computes in natively (a property of the part, not measured).
    pub const fn arithmetic(&self) -> VpuArithmetic {
        self.target.arithmetic()
    }
}
