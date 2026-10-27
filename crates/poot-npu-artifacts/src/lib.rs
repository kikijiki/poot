//! The NPU-owned half of the kernel contract: what a compiled NPU region is on the host.
//!
//! `poot-codegen`'s `KernelCode` carries these types as the payload of its `Aie` and `Vpu` variants (the
//! launch carrier stays an enum, POOT-1117), but they and everything that builds or validates them belong to
//! the NPU, not to the foundation crates every GPU runtime shares:
//!
//! - [`aie`]: AMD XDNA2's linked xclbin plus instruction stream and the shape ABI that places every region
//!   value in a host arena, validated at construction.
//! - [`aie_abi`]: why an AIE region's recipe and tile bodies cannot be linked.
//! - [`vpu`]: Intel's OpenVINO IR plus the region's argument ABI.
//!
//! A leaf above `poot-target`: no compiler, planner or runtime crate is a dependency.

pub mod aie;
pub mod aie_abi;
pub mod vpu;

pub use aie::{
    AieArena, AieArtifactError, AieBinding, AieInvocation, AieLinkedArtifact, AieShapeAbi,
};
#[cfg(any(test, feature = "test-support"))]
pub use aie::{AieBindError, AieBound, check_aie_bindings};
pub use aie_abi::{AieAbiMismatch, AieLimitRefusal, RecipeFault};
pub use vpu::{VpuArtifactError, VpuBinding, VpuGraphArtifact, VpuInvocation, VpuLayer};
