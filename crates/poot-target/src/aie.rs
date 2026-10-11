//! The AMD XDNA (AIE) compilation target: its typed identity and the limits a plan is admitted
//! against (POOT-664, spec 1002).
//!
//! An AIE target is not a GPU: one core runs a sequential loop, the IRON harness moves data between host
//! buffers and tile memory, and there is no thread grid, workgroup memory, barrier or subgroup. So it
//! has its own descriptor instead of borrowing the GPU-shaped [`crate::DeviceCaps`] fields, and a limit
//! nobody measured is [`Queried::Unknown`], never a plausible number.

use crate::Queried;

/// The AIE core instruction set a core object is compiled for (`llc --march=`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AieIsa {
    /// XDNA2 (Strix Point/Halo): the `aie2p` core.
    Aie2p,
}

impl AieIsa {
    /// The Peano `--march` value.
    pub const fn march(self) -> &'static str {
        match self {
            AieIsa::Aie2p => "aie2p",
        }
    }
}

/// The platform a linked artifact is built for and loaded on: the shim/array geometry the IRON link
/// recipe targets. A linked artifact for one platform does not load on another.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AiePlatform {
    /// The XDNA2 NPU (IRON's `npu2` device).
    Npu2,
}

impl AiePlatform {
    /// The platform's name inside link-recipe and cache identities.
    pub const fn name(self) -> &'static str {
        match self {
            AiePlatform::Npu2 => "npu2",
        }
    }
}

/// An AIE compilation target's identity. A closed enum: each variant fixes the ISA, the platform/link
/// ABI and its revision, so every id names a target the compiler emits for (adding one forces every
/// match on it). A product or model name never chooses a lowering; this does. A runtime device
/// instance's identity is separate.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AieTargetId {
    /// XDNA2: `aie2p` cores on the `npu2` platform.
    Xdna2,
}

impl AieTargetId {
    pub const fn isa(self) -> AieIsa {
        match self {
            AieTargetId::Xdna2 => AieIsa::Aie2p,
        }
    }

    pub const fn platform(self) -> AiePlatform {
        match self {
            AieTargetId::Xdna2 => AiePlatform::Npu2,
        }
    }

    /// Revision of the host/core binding and link-recipe ABI poot lowers to for this target. A loader
    /// refuses an artifact whose revision it does not implement.
    pub const fn abi_revision(self) -> u32 {
        match self {
            AieTargetId::Xdna2 => 1,
        }
    }

    /// The identity's canonical text, as it enters cache keys.
    pub fn canonical(&self) -> String {
        format!(
            "aie:isa={}:platform={}:abi={}",
            self.isa().march(),
            self.platform().name(),
            self.abi_revision()
        )
    }
}

/// What an AIE target offers a plan. Every bound is [`Queried`]: measured by the XDNA runtime, or
/// `Unknown`. Admission refuses a plan whose requirement it cannot bound (a limit is never guessed).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AieCaps {
    pub target: AieTargetId,
    /// Bytes of tile-local data memory one core may reserve for its FIFO buffers and scratch. Not GPU
    /// LDS: it is a different resource with a different owner.
    pub tile_local_bytes: Queried<u32>,
    /// How many host buffer objects one kernel invocation may bind.
    pub host_bindings: Queried<u32>,
    /// The largest instruction (transfer-descriptor) stream a loaded kernel may carry, in bytes.
    pub instruction_bytes: Queried<u64>,
    /// Bytes of device-visible host memory the program's buffer objects may total.
    pub memory_budget_bytes: Queried<u64>,
}

impl AieCaps {
    /// A target nobody has measured: every bound unknown, so no plan that needs one is admitted. The
    /// production value until the XDNA runtime reports (POOT-641).
    pub const fn unmeasured(target: AieTargetId) -> Self {
        Self {
            target,
            tile_local_bytes: Queried::Unknown,
            host_bindings: Queried::Unknown,
            instruction_bytes: Queried::Unknown,
            memory_budget_bytes: Queried::Unknown,
        }
    }

    /// Fixture limits for model-free tests: SYNTHETIC, not a measurement of any device. A test that
    /// shrinks one field states its own value.
    pub const fn synthetic(
        target: AieTargetId,
        tile_local_bytes: u32,
        host_bindings: u32,
        instruction_bytes: u64,
        memory_budget_bytes: u64,
    ) -> Self {
        Self {
            target,
            tile_local_bytes: Queried::Known(tile_local_bytes),
            host_bindings: Queried::Known(host_bindings),
            instruction_bytes: Queried::Known(instruction_bytes),
            memory_budget_bytes: Queried::Known(memory_budget_bytes),
        }
    }
}
