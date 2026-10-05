//! wgpu/Vulkan execution contract backend: [`device::WgpuDevice`] is the
//! [`poot_executor::Device`] implementation, compiling kernel bodies for the SPIR-V/Vulkan target.
//! Fusion and dtype preparation are graph/planner transforms, not a separate wgpu op definition.

/// `WgpuDevice`: the `poot_executor::Device` implementation (Card 546a, ADR-0003).
pub mod device;

#[cfg(test)]
#[path = "tests/packed_production.rs"]
mod packed_production_tests;

#[cfg(test)]
#[path = "tests/parity.rs"]
mod parity_tests;

#[cfg(test)]
#[path = "tests/imported_kernel.rs"]
mod imported_kernel_tests;
