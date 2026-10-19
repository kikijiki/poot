//! The Intel NPU (VPU) region emitter: a typed region of lowered equations to OpenVINO IR text
//! (POOT-1159, design npu-targets-1093 sections 4.2 and 4.7).
//!
//! ```text
//! planned equations --VpuRegion--> emit_vpu --> VpuIr { xml, weights } --driver compiler--> blob
//! ```
//!
//! Intel ships no core ISA or kernel toolchain: the only entry point is a graph compiler that takes
//! OpenVINO IR. A region is one OpenVINO model of its equations, so this module is the VPU counterpart of
//! the AIE core-object path in `aie`, and it is the only place that spells opset names. The planner
//! (`poot-graph-plan`) decides support per equation and builds the `VpuRegion`; the emitter makes no
//! support decision and interprets no graph. It serializes what the region says, deterministically, so the
//! same region always yields the same text (the compiled blob is cached under that text).
//!
//! Every `VpuNode` carries the id of the poot equation it came from and names its layer after it
//! (`v<origin>`), so a layer the vendor compiler rejects maps back to the equation that produced it.
//!
//! The vendor compile, the blob cache and the Level Zero device are the NPU runtime's; nothing here needs
//! hardware.

/// The element types a VPU region carries. The device computes in `F16`; `F32` and `I32` appear only as
/// data that moves or converts (an I/O boundary, an index).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum VpuElem {
    /// IEEE binary16.
    F16,
    /// IEEE binary32.
    F32,
    /// Signed 32-bit integer.
    I32,
}

/// A static tensor type: element type and shape (the NPU compiles static shapes only).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VpuTensor {
    /// The element type.
    pub elem: VpuElem,
    /// The extents, outermost first. Empty is a scalar.
    pub shape: Vec<usize>,
}

/// A scalar constant, as the bits of its element type. A literal operand of a poot equation, or an index, lowers
/// to a rank-0 `Const` node holding one of these.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VpuScalar {
    /// An IEEE binary16 value, as its bits.
    F16(u16),
    /// An IEEE binary32 value, as its bits.
    F32(u32),
    /// A signed 32-bit integer.
    I32(i32),
}

/// The OpenVINO op one poot primitive lowers to, with the parameters the opset op takes. One variant per
/// primitive: each has exactly one lowering.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VpuOp {
    /// Elementwise `Add` with numpy broadcasting.
    Add,
    /// Elementwise `Subtract` with numpy broadcasting.
    Subtract,
    /// Elementwise `Multiply` with numpy broadcasting.
    Multiply,
    /// Elementwise `Divide` with numpy broadcasting.
    Divide,
    /// Elementwise `Maximum` with numpy broadcasting.
    Maximum,
    /// Elementwise `a >= b` as `1` or `0` in the operands' element type: `GreaterEqual`, then a `Convert` of
    /// its boolean result, which is poot's `Ge`.
    GreaterEqual,
    /// Elementwise `Exp`.
    Exp,
    /// Elementwise `Negative`.
    Negative,
    /// Elementwise `Sqrt`.
    Sqrt,
    /// `ReduceSum` over one axis.
    ReduceSum {
        /// The reduced axis.
        axis: usize,
        /// Whether the axis stays as extent 1.
        keepdim: bool,
    },
    /// `ReduceMax` over one axis.
    ReduceMax {
        /// The reduced axis.
        axis: usize,
        /// Whether the axis stays as extent 1.
        keepdim: bool,
    },
    /// Batched `MatMul`, batch dimensions broadcast (numpy).
    MatMul,
    /// `Gather` along `axis` by an index tensor.
    Gather {
        /// The axis the rows are taken along.
        axis: usize,
    },
    /// `Reshape` to the node's output shape.
    Reshape,
    /// `Transpose` by `perm` (output axis `i` is input axis `perm[i]`).
    Transpose {
        /// The permutation.
        perm: Vec<usize>,
    },
    /// numpy `Broadcast` to the node's output shape.
    Broadcast,
    /// `Slice` of `start..end` along one axis.
    Slice {
        /// The axis sliced.
        axis: usize,
        /// The first index kept.
        start: usize,
        /// One past the last index kept.
        end: usize,
    },
    /// `Concat` along `axis`.
    Concat {
        /// The concatenation axis.
        axis: usize,
    },
    /// `Convert` to the node's output element type.
    Convert,
    /// The range `0..len` as a rank-1 `Const` of the node's element type (`F32` or `I32`). No operands.
    Iota {
        /// The number of elements.
        len: usize,
    },
    /// A copy of the first operand with the second written into it from the scalar `I32` index (the third
    /// operand) along `axis`: `ScatterUpdate` at the indices `index..index + extent`. The index must select a
    /// window that fits (the CPU oracle rejects one that does not with a typed error): the device does not
    /// check it and its result for an out-of-range index is undefined.
    DynamicUpdateSlice {
        /// The axis the update is written along.
        axis: usize,
    },
    /// A rank-0 `Const`. No operands.
    Scalar(VpuScalar),
}

mod region;
pub use region::{VpuEmitError, VpuInput, VpuIr, VpuNode, VpuRegion, VpuValue, emit_vpu};
