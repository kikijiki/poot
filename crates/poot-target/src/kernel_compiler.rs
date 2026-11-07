//! [`KernelCompiler`]: the compiler that turns a generated kernel into a device's machine code, as the runtime that
//! loads it reports it (POOT-1505). A plan made with compile feedback (whether a candidate tiling spills, asked of
//! this compiler) is a function of it, so it is part of what names the plan. It is a reported fact, not a fitted or
//! chosen value.

/// The compiler that lowers a generated kernel to the device's machine code.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum KernelCompiler {
    /// The CUDA driver's PTX JIT (poot loads PTX with `cuModuleLoadData`, no JIT options): the CUDA release the driver
    /// reports and the compute capability it compiles for.
    CudaDriver {
        /// `cuDriverGetVersion`'s value, `1000 * major + 10 * minor` (13020 for CUDA 13.2).
        release: u32,
        /// The device's compute capability, `(major, minor)`.
        compute_capability: (u32, u32),
    },
    /// The AMD-flavored clang that compiles and links the HSACO (`$POOT_AMD_CLANG`): its version, the first 64 bits
    /// of the `llvm-project` commit its `--version` names (two builds of one version number differ), and the gfx
    /// target it compiles for.
    AmdClang {
        /// `(major, minor, patch)` of `AMD clang version X.Y.Z`.
        version: (u32, u32, u32),
        /// The first 16 hex digits of the commit.
        commit: u64,
        /// The gfx target as a number: `0x1151` for gfx1151.
        gfx: u32,
    },
}

impl KernelCompiler {
    /// Whether this compiler compiles the kernels of a `backend` device: the CUDA driver's JIT PTX, AMD clang AMDGPU.
    /// A register verdict from one is about kernels only such a device runs (POOT-1505).
    pub const fn compiles_for(self, backend: crate::Backend) -> bool {
        matches!(
            (self, backend),
            (Self::CudaDriver { .. }, crate::Backend::Nvptx)
                | (Self::AmdClang { .. }, crate::Backend::AmdGcn(_))
        )
    }
}
