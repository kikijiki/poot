//! poot-kernel-ir: the MIR-mirroring leaf-kernel IR. A `Body` is the slice of Rust MIR that poot lowers
//! for one `#[kernel]` function: locals, basic blocks of statements + a terminator, a workgroup shape,
//! and any workgroup-local (LDS) arrays. Anything outside the supported subset is rejected at import and
//! never represented here.
//!
//! This is the target-independent layer: no rustc or device dependency,
//! unit-tests on stable Rust. `poot-codegen` consumes it for SPIR-V, PTX, AMDGPU, and AIE core targets,
//! with named target-specific rejections where an operation is unsupported. The GPU grid is not in the IR
//! (it is a host-side launch argument); only the workgroup shape is.
//!
//! See `poot-codegen` for the data-model rationale and the per-backend lowering constraints.

pub mod fixtures;
pub mod interp;
pub mod naming;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Derives `Serialize`/`Deserialize` only when the `serde` feature is on.
macro_rules! ir_node {
    ($(#[$m:meta])* $vis:vis enum $name:ident { $($body:tt)* }) => {
        $(#[$m])*
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        $vis enum $name { $($body)* }
    };
    ($(#[$m:meta])* $vis:vis struct $name:ident { $($body:tt)* }) => {
        $(#[$m])*
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        $vis struct $name { $($body)* }
    };
}

// --- identifiers --------------------------------------------------------------------------------

ir_node! {
    /// A MIR local. `_0` is the return place, `_1..=param_count` are the params in order, the rest are
    /// temporaries.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
    pub struct Local { pub index: u32 }
}

ir_node! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
    pub struct BlockId { pub index: u32 }
}

// --- types --------------------------------------------------------------------------------------

ir_node! {
    /// The tiny type lattice poot lowers. The u8/i8/u16/i16/u64/i64 widths are out of subset and rejected at import.
    #[derive(Clone, PartialEq, Eq, Debug, Hash)]
    pub enum Ty {
        Unit,
        Bool,
        Usize,
        F16,
        /// bfloat16 (LLVM `bfloat`, spec 024). Arithmetic widens to f32 and accumulates in f32, narrowing to bf16 on store.
        BF16,
        F32,
        F64,
        I32,
        U32,
        Ref { mutable: bool, pointee: Box<Ty> },
        Slice(Box<Ty>),
        /// A fixed-size private (per-thread) array, e.g. flash attention's `o[head_dim]` accumulator.
        /// Lowers to an `alloca [len x elem]` on NVPTX; the LLVM SPIR-V backend crashes on a
        /// dynamically-indexed private array (poot-legacy spec-076), so SpirvVulkan rejects it.
        Array { elem: Box<Ty>, len: u32 },
        /// A fixed-lane SIMD vector (spec 134 P1): `lanes` is 2 or 4, `elem` a scalar numeric type.
        /// Backend-neutral (no wave width baked in). `poot-codegen` lowers it to LLVM `<lanes x elem>`;
        /// only SpirvVulkan is implemented (NVPTX/AMDGCN emit a named `Unsupported` diagnostic, see
        /// emit.rs). `Rvalue::VectorLoad`/`Statement::VectorStore` read/write a scalar-element slice
        /// `lanes` elements at a time.
        Vec { elem: Box<Ty>, lanes: u32 },
    }
}

impl Ty {
    pub fn is_float(&self) -> bool {
        match self {
            Ty::F16 | Ty::BF16 | Ty::F32 | Ty::F64 => true,
            Ty::Vec { elem, .. } => elem.is_float(),
            _ => false,
        }
    }
    pub fn is_signed_int(&self) -> bool {
        match self {
            Ty::I32 => true,
            Ty::Vec { elem, .. } => elem.is_signed_int(),
            _ => false,
        }
    }
}

// --- places, operands, constants ----------------------------------------------------------------

ir_node! {
    /// A projection step off a base local: a deref or an index by another local. `(*c)[i]` is
    /// `[Deref, Index(i)]`.
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub enum ProjectionElem {
        Deref,
        Index(Local),
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct Place {
        pub local: Local,
        pub projection: Vec<ProjectionElem>,
    }
}

impl Place {
    pub fn local(local: Local) -> Self {
        Place {
            local,
            projection: Vec::new(),
        }
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Constant {
        Bool(bool),
        Usize(u64),
        I32(i32),
        U32(u32),
        F32(f32),
        F64(f64),
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Operand {
        Copy(Place),
        Move(Place),
        Const(Constant),
    }
}

// --- operators ----------------------------------------------------------------------------------

ir_node! {
    /// Binary operators. Signedness (sdiv/udiv, etc.) is taken from the operand `Ty` at emit time.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum BinOp {
        Add, Sub, Mul, Div, Rem,
        BitAnd, BitOr, BitXor, Shl, Shr,
        Min, Max,
        Lt, Le, Gt, Ge, Eq, Ne,
    }
}

ir_node! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum UnOp {
        Neg,
        Not,
    }
}

ir_node! {
    /// Float math ops that lower to LLVM intrinsics. The transcendentals (exp/sin/cos) are split per
    /// backend in `poot-codegen` (NVPTX approx intrinsics vs SPIR-V GLSL.std.450); the IR stays abstract.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MathOp {
        Sqrt, Abs, Floor, Ceil, Trunc, Round,
        Exp, Log, Sin, Cos,
    }
}

ir_node! {
    /// Logical FP8 format carried by format-aware conversion rvalues (spec 149). Names byte semantics only;
    /// target instructions and physical packing stay below this IR.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
    pub enum Fp8Format {
        E4M3Fn,
    }
}

ir_node! {
    /// Integer bit ops on u32 (per-backend lowering split in `poot-codegen`).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum IntScalarOp {
        CountOnes,
        LeadingZeros,
        TrailingZeros,
    }
}

ir_node! {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum AtomicOp {
        Add, Min, Max, And, Or, Xor, Exchange,
    }
}

ir_node! {
    /// The scope a [`Statement::Fence`] orders memory for (spec 138). `Workgroup` makes LDS and global
    /// stores visible to the calling thread's workgroup (like `Terminator::Barrier`'s scope, without the
    /// execution sync). `Device` makes global memory visible to every workgroup (a producer CTA's plain
    /// global store is otherwise only guaranteed L1-local).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum MemoryScope {
        Workgroup,
        Device,
    }
}

ir_node! {
    /// The ordering a [`Statement::Fence`] establishes (spec 138). Only acquire/release are exposed:
    /// RMW/CAS atomics stay `monotonic` and a fence pairs with them, and Vulkan forbids
    /// `SequentiallyConsistent` (same reason `fix_barrier_semantics` strips it from the workgroup barrier).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum MemoryOrdering {
        Acquire,
        Release,
        AcqRel,
    }
}

// --- rvalues ------------------------------------------------------------------------------------

ir_node! {
    /// The instruction set (the right-hand side of an `Assign`), limited to the supported kernel subset.
    #[derive(Clone, PartialEq, Debug)]
    pub enum Rvalue {
        Use(Operand),
        BinaryOp(BinOp, Operand, Operand),
        /// A float `Add`/`Sub`/`Mul` marked to forbid fused multiply-add contraction with any op it
        /// produces or consumes (card 628; dquant.md section 7, R1). A backend compiler
        /// contracts `fmul`+`fadd`/`fsub` into an FMA by default on some paths (NVPTX's downstream `ptxas`
        /// `-fmad`, RADV/ACO's SPIR-V contraction), which changes rounding and breaks bit-exact decode
        /// against the 541 CPU oracle for the min formats (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K). Every backend's
        /// codegen honours this marker: SPIR-V decorates the result `NoContraction`, AMDGCN and NVPTX lower
        /// it as an LLVM constrained floating-point intrinsic under `strictfp` (contraction- and
        /// reassociation-immune regardless of the backend's own default). `op` is `Add`, `Sub` or `Mul`
        /// only, and the operand type must be float (`Body::verify`). Opt-in per instruction: an ordinary
        /// `Rvalue::BinaryOp` keeps today's contraction-eligible codegen (no across-the-board performance
        /// change) - only the packed-dequant decode formula (dquant.md 3.3) opts in.
        BinaryOpNoContract(BinOp, Operand, Operand),
        UnaryOp(UnOp, Operand),
        MathUnary(MathOp, Operand),
        IntScalarUnary(IntScalarOp, Operand),
        /// Slice length (`c.len()`), mirrors MIR `Rvalue::Len`.
        Len(Place),
        /// Value-converting numeric cast.
        Cast { to: Ty, operand: Operand },
        /// Bit reinterpret (`f32::from_bits`), no value conversion.
        Bitcast { to: Ty, operand: Operand },
        /// Decode the low 8 bits of a `U32` byte carrier to f32 using `format` semantics. Packed-word lane
        /// extraction happens outside this rvalue.
        Fp8Decode { format: Fp8Format, operand: Operand },
        /// Encode an f32 to the low 8 bits of a `U32` byte carrier using `format` semantics. Packed-word
        /// lane insertion happens outside this rvalue. FP8 is never a scalar `Ty`.
        Fp8Encode { format: Fp8Format, operand: Operand },
        /// Read element `idx` of workgroup-local (LDS) array `array`.
        WorkgroupLocalRead { idx: Operand, array: u8 },
        /// Atomic on workgroup-local array `array` at `idx`.
        WorkgroupLocalAtomic { idx: Operand, value: Operand, op: AtomicOp, array: u8 },
        /// Atomic on a global slice element.
        GlobalAtomic { place: Place, value: Operand, op: AtomicOp },
        /// Compare-and-swap on workgroup-local (LDS) array `array` at `idx`: replace the cell with `desired`
        /// if it holds `expected`, returning the old value (spec 138). Separate from `AtomicOp` because CAS
        /// takes two value operands. Integer-only in codegen (LLVM `cmpxchg` rejects float; Vulkan has no
        /// float `OpAtomicCompareExchange`).
        WorkgroupLocalCompareExchange {
            idx: Operand,
            expected: Operand,
            desired: Operand,
            array: u8,
        },
        /// Compare-and-swap on a global slice element: replace `place` with `desired` if it holds
        /// `expected`, returning the old value (spec 138). Integer-only in codegen (see `WorkgroupLocalCompareExchange`).
        GlobalCompareExchange {
            place: Place,
            expected: Operand,
            desired: Operand,
        },
        /// Load `lanes` contiguous elements of a scalar-element slice as one `Ty::Vec` value (spec 134,
        /// FR-001). `place` must project a slice-typed local at `[Deref, Index(i)]`; `i` is the vector-group
        /// index (elements `[i*lanes, i*lanes+lanes)`), not a scalar element index. `lanes`/`elem` come from
        /// the destination local's declared `Ty::Vec`. Alignment (FR-003) is a builder obligation; codegen
        /// only stamps `align`. SpirvVulkan-only: a body must not mix scalar and vector access to one param
        /// (the two-binding mechanism, FR-004b, is not implemented).
        VectorLoad { place: Place },
        /// Broadcast a scalar operand to every lane of the destination's declared `Ty::Vec` type.
        VectorSplat(Operand),
    }
}

// --- statements + terminators -------------------------------------------------------------------

ir_node! {
    /// Which input matrix a WMMA load fills (the `a` or `b` fragment), spec 025.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum WmmaMat {
        A,
        B,
    }
}

ir_node! {
    /// The WMMA/cooperative-matrix operand element dtype (card 154, target-neutral per card 530). Fragment
    /// packing follows the operand type, not the codegen `Target`: `F16` lowers on every implemented target
    /// (NVPTX and AMDGCN WMMA, SPIR-V cooperative matrix); `Bf16` lowers only on NVPTX/AMDGCN (RADV has no
    /// `VK_KHR_shader_bfloat16`, so SPIR-V refuses it). A dtype a target cannot lower is a typed
    /// `EmitError::Unsupported`, never a silently wrong fragment layout (see
    /// `poot_codegen::emit`'s `wmma_check_target`).
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum WmmaDtype {
        Bf16,
        F16,
    }
}

ir_node! {
    /// The `m x n x k` tile shape a matrix-fragment op operates over (card 530). Every target implemented
    /// today only lowers 16x16x16 (`M16N16K16`); the field exists so a future shape needs no new op, only a
    /// new codegen arm and a typed refusal for targets that lack it.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
    pub struct WmmaShape {
        pub m: u16,
        pub n: u16,
        pub k: u16,
    }
}

impl WmmaShape {
    pub const M16N16K16: WmmaShape = WmmaShape {
        m: 16,
        n: 16,
        k: 16,
    };
}

ir_node! {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Statement {
        Assign(Place, Rvalue),
        StorageLive(Local),
        StorageDead(Local),
        WorkgroupLocalWrite { idx: Operand, value: Operand, array: u8 },
        /// Store a `Ty::Vec` operand as `lanes` contiguous elements of a scalar-element slice at `place`'s
        /// vector-group index (counterpart of `Rvalue::VectorLoad`; same indexing and FR-004b caveat).
        /// `lanes`/`elem` come from `value`'s type.
        VectorStore { place: Place, value: Operand },
        /// Warp/subgroup-collective matrix-fragment load (spec 025/154, card 530): one target-neutral op,
        /// lowered per target by `poot_codegen::emit` (AMDGCN/NVPTX WMMA, SPIR-V cooperative matrix). `dst`
        /// is one opaque fragment handle: a `Local` that never goes through the ordinary alloca/load/store
        /// path (see `emit`'s `frag_regs`) and whose declared `Ty` is a marker only (by convention the
        /// operand scalar type, e.g. `F16`), never interpreted as that scalar's value. `tile` is the address
        /// of the tile's first element (a slice element place); `stride` the row stride in elements.
        WmmaLoad {
            which: WmmaMat,
            dtype: WmmaDtype,
            shape: WmmaShape,
            tile: Place,
            stride: u32,
            dst: Local,
        },
        /// `dst = a * b + c` (fused). `a`/`b`/`c`/`dst` are opaque fragment handles (see `WmmaLoad`); `a`/`b`
        /// carry `dtype`, `c`/`dst` are always the f32 accumulator (every target implemented today only has
        /// an f32-accumulate WMMA/coopmat instruction for `dtype`).
        WmmaMma {
            dtype: WmmaDtype,
            shape: WmmaShape,
            a: Local,
            b: Local,
            c: Local,
            dst: Local,
        },
        WmmaStore {
            dtype: WmmaDtype,
            shape: WmmaShape,
            tile: Place,
            stride: u32,
            src: Local,
        },
        /// Seed a zero-valued f32 accumulator fragment (card 154, extended to every target by card 530): an
        /// opaque fragment value has no addressable components a scalar `Assign` could zero lane by lane, so
        /// every target seeds it through one op. Lowers to `OpCompositeConstruct` on SPIR-V (extended by
        /// `SPV_KHR_cooperative_matrix` to broadcast a scalar into every component) and a zero vector/struct
        /// constant on AMDGCN/NVPTX (no device instruction; the fragment register group starts as literal
        /// zero bits).
        WmmaZero { dtype: WmmaDtype, shape: WmmaShape, dst: Local },
        /// Load an A/B fragment from a workgroup-local (LDS) `array` (base index 0) with row `stride`. LDS
        /// analog of `WmmaLoad`, used by the bf16 tensor-core dequant gemm (spec 026). NVPTX-only.
        WmmaLoadLds {
            which: WmmaMat,
            dtype: WmmaDtype,
            shape: WmmaShape,
            array: u8,
            stride: u32,
            dst: Local,
        },
        /// Store the f32 D fragment to a workgroup-local (LDS) `array` (base index 0) with row `stride`. Used
        /// by the bf16 tensor-core gemm epilogue (store to LDS, barrier, narrow-copy to bf16). NVPTX-only.
        WmmaStoreLds {
            shape: WmmaShape,
            array: u8,
            stride: u32,
            src: Local,
        },
        /// A scoped memory-ordering fence with no control-flow effect (spec 138), hence a `Statement`.
        /// Lowers to one LLVM `fence` scoped by `scope`/`ordering`; see `emit_statement`'s
        /// `Statement::Fence` arm for the per-target syncscope mapping. AieCore rejects it by name (one AIE
        /// core has nothing to fence against, like Barrier/CAS).
        Fence { scope: MemoryScope, ordering: MemoryOrdering },
    }
}

ir_node! {
    /// A dispatch-id axis. The only "call" in the subset is `thread_index*()`, hence a dedicated terminator.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum IndexAxis {
        X, Y, Z,
        LocalX, LocalY, LocalZ,
        GroupX, GroupY, GroupZ,
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct SwitchTargets {
        /// `(value, target)` pairs.
        pub branches: Vec<(u128, BlockId)>,
        pub otherwise: BlockId,
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Debug)]
    pub enum Terminator {
        Goto { target: BlockId },
        SwitchInt { discr: Operand, targets: SwitchTargets },
        /// Write the dispatch id for `dim` into `destination`, then go to `target`.
        ThreadIndexCall { destination: Place, dim: IndexAxis, target: BlockId },
        /// Workgroup barrier, then `target`.
        Barrier { target: BlockId },
        Return,
        /// An unrecoverable kernel-subset fault (a failed `Assert` or a reached `Unreachable`): the device
        /// aborts instead of continuing with whatever the kernel had computed so far (card 531c).
        /// No successors. `code` is a body-local, 1-based id naming which trap site this is (assigned in
        /// program order by the importer; 0 is reserved to mean "no fault" in the SpirvVulkan error word
        /// below, so a real code is never 0). `codegen` lowers this to a real device trap on ROCm/PTX; on
        /// SpirvVulkan, which has no compute trap instruction, it atomically writes `code` to a reserved
        /// per-dispatch error-word buffer and returns, and the executor (`poot-runtime`/`poot-gpu`) checks
        /// that word at its existing sync point and turns a nonzero value into a typed fault naming the
        /// kernel and this code (card 531c decision 3).
        Trap { code: u32 },
    }
}

// --- the body -----------------------------------------------------------------------------------

ir_node! {
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct LocalDecl {
        pub ty: Ty,
        pub mutable: bool,
    }
}

ir_node! {
    /// A workgroup-local (LDS) array. Element type is part of the declaration (an f32 tile and a u32
    /// scratch cannot share one array).
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct WorkgroupLocalDecl {
        pub elem_ty: Ty,
        pub len: u32,
    }
}

ir_node! {
    #[derive(Clone, PartialEq, Debug)]
    pub struct BasicBlock {
        pub statements: Vec<Statement>,
        pub terminator: Terminator,
    }
}

ir_node! {
    /// One lowered `#[kernel]` function.
    #[derive(Clone, PartialEq, Debug)]
    pub struct Body {
        pub name: String,
        /// Number of kernel parameters (locals `_1..=param_count`).
        pub param_count: u32,
        pub locals: Vec<LocalDecl>,
        pub blocks: Vec<BasicBlock>,
        /// Workgroup shape (x, y, z); defaults to (64, 1, 1).
        pub workgroup_size: [u32; 3],
        pub workgroup_locals: Vec<WorkgroupLocalDecl>,
    }
}

impl Body {
    /// A body with the default 64x1x1 workgroup and no LDS.
    pub fn new(
        name: impl Into<String>,
        param_count: u32,
        locals: Vec<LocalDecl>,
        blocks: Vec<BasicBlock>,
    ) -> Self {
        Body {
            name: name.into(),
            param_count,
            locals,
            blocks,
            workgroup_size: [64, 1, 1],
            workgroup_locals: Vec::new(),
        }
    }

    /// The parameter locals, in order (`_1..=param_count`).
    pub fn params(&self) -> impl Iterator<Item = Local> + '_ {
        (1..=self.param_count).map(|i| Local { index: i })
    }

    pub fn local_ty(&self, l: Local) -> &Ty {
        &self.locals[l.index as usize].ty
    }

    /// Whether any block ends in a [`Terminator::Trap`] (a failed `Assert` or a reached `Unreachable`
    /// somewhere in the body). SpirvVulkan codegen and the runtime consult this to decide whether the
    /// kernel needs the reserved per-dispatch error-word buffer (card 531c).
    pub fn has_trap(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b.terminator, Terminator::Trap { .. }))
    }
}

// --- verification -------------------------------------------------------------------------------

pub use verify::{Site, VerifyError, VerifyErrorKind};

/// [`Body::verify`]: the well-formedness contract every consumer of a [`Body`] relies on. A body that passes
/// has declared locals and workgroup arrays, existing branch targets, places of the shapes codegen lowers,
/// operands whose types agree with the rvalue and the place they feed, stores only through mutable buffer
/// bindings, and no local read on a path where it was never assigned.
mod verify {
    use super::*;
    use std::collections::HashSet;
    use std::fmt;

    /// Where in the body a violation sits.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Site {
        /// The signature: params, locals, blocks, workgroup shape.
        Signature,
        Statement {
            block: BlockId,
            index: usize,
        },
        Terminator {
            block: BlockId,
        },
    }

    #[derive(Clone, PartialEq, Debug, thiserror::Error)]
    pub enum VerifyErrorKind {
        /// The body has no blocks, so there is no entry.
        #[error("the body has no blocks")]
        NoBlocks,
        /// `param_count` needs `_0` plus one local per param.
        #[error("param_count {param_count} needs {} locals (_0 and the params), found {locals}", u64::from(*param_count) + 1)]
        ParamsExceedLocals { param_count: u32, locals: usize },
        /// A kernel param must bind a buffer: `&[T]` or `&mut [T]`.
        #[error("param _{} has type {ty:?}, expected a buffer binding (&[T] or &mut [T])", local.index)]
        ParamNotBuffer { local: Local, ty: Ty },
        #[error("workgroup size {size:?} has a zero dimension")]
        ZeroWorkgroupDim { size: [u32; 3] },
        #[error("local _{} is not declared", .0.index)]
        UndeclaredLocal(Local),
        #[error("workgroup-local array {0} is not declared")]
        UndeclaredWorkgroupArray(u8),
        #[error("block bb{} does not exist", .0.index)]
        UndefinedBlock(BlockId),
        /// A place projection codegen cannot lower: only `[]`, `[Index]` on a private array, and
        /// `[Deref, Index]` on a param buffer are places.
        #[error("malformed place {place:?}: {reason}")]
        MalformedPlace { place: Place, reason: &'static str },
        /// The local is read on a path from the entry that has not assigned it.
        #[error("local _{} is read before it is assigned on some path", .0.index)]
        UseBeforeDefinition(Local),
        #[error("{what}: expected {expected:?}, found {found:?}")]
        TypeMismatch {
            what: &'static str,
            expected: Ty,
            found: Ty,
        },
        /// The operand type is outside what the operation accepts (a float shift amount, a `Ref` cast).
        #[error("{what}: operand type {found:?} is not accepted")]
        InvalidOperandType { what: &'static str, found: Ty },
        /// A store or atomic through a `&[T]` binding (the buffer is read-only).
        #[error("write through the read-only buffer binding _{}", param.index)]
        WriteThroughSharedBuffer { param: Local },
        /// A matrix-fragment op's `shape` is not one every codegen target implements (card 530). Only `M16N16K16` is implemented today; this is the IR-level half of the check
        /// (`poot_codegen::emit`'s `wmma_check_shape` is the target-lowering half) so a malformed shape is
        /// caught before any codegen sees it, matching every other well-formedness check here.
        #[error(
            "matrix-fragment shape {shape:?} is not implemented (only WmmaShape::M16N16K16 lowers today)"
        )]
        UnsupportedWmmaShape { shape: WmmaShape },
        /// A known fragment producer has the wrong role for this consumer. `Some(A/B)` names an
        /// operand matrix; `None` names the accumulator, not an unknown role. Scalar marker types do
        /// not determine fragment roles.
        #[error(
            "fragment _{}: expected {}, found {}",
            local.index,
            fragment_role_name(*expected_matrix),
            fragment_role_name(*found_matrix)
        )]
        FragmentRoleMismatch {
            local: Local,
            expected_matrix: Option<WmmaMat>,
            found_matrix: Option<WmmaMat>,
        },
        /// One local has incompatible known fragment producers. The enclosing `Site` names the
        /// conflicting producer; `previous` names the first. `None` is the accumulator role.
        #[error(
            "fragment _{} is produced as {} at {previous} and as {} here",
            local.index,
            fragment_role_name(*previous_matrix),
            fragment_role_name(*found_matrix)
        )]
        FragmentProducerRoleConflict {
            local: Local,
            previous: Site,
            previous_matrix: Option<WmmaMat>,
            found_matrix: Option<WmmaMat>,
        },
        /// `Rvalue::BinaryOpNoContract`'s `op` is not `Add`, `Sub` or `Mul` (card 628): the no-contraction
        /// marker exists for the packed-dequant formula's multiply-add/-sub chain only, and every backend's
        /// codegen lowering (constrained intrinsic, SPIR-V decoration) assumes one of those three ops.
        #[error(
            "BinaryOpNoContract op {op:?} is not implemented (only Add, Sub and Mul carry the no-contraction marker)"
        )]
        UnsupportedNoContractOp { op: BinOp },
        /// A `SwitchInt` whose discriminant can differ between lanes of the same workgroup reaches a
        /// [`Terminator::Barrier`] before its arms are guaranteed to reconverge (card 618, hardened by
        /// its review): real hardware runs a barrier in lockstep, so a workgroup whose lanes disagree on
        /// this branch can park some lanes at `barrier` while sending others past it, or into a
        /// different one, before every lane is back at the same program point. `reconverge` is the
        /// branch's immediate post-dominator - the nearest block every arm is guaranteed to reach - or
        /// `None` when no such block exists before the body's exit (the arms never provably reunite at
        /// all, so any barrier reachable from any arm is already unsafe).
        #[error(
            "bb{} branches on a value that can differ between lanes: the barrier at bb{} is reachable \
             before the branch's arms are guaranteed to reconverge{}",
            branch.index,
            barrier.index,
            reconverge_desc(*reconverge)
        )]
        DivergentBarrierReachability {
            branch: BlockId,
            barrier: BlockId,
            reconverge: Option<BlockId>,
        },
    }

    /// Describes [`VerifyErrorKind::DivergentBarrierReachability`]'s `reconverge` field.
    fn reconverge_desc(reconverge: Option<BlockId>) -> String {
        match reconverge {
            Some(b) => format!(" (at bb{})", b.index),
            None => " (they never provably reconverge)".to_string(),
        }
    }

    /// A typed verification failure: what is wrong and where.
    #[derive(Clone, PartialEq, Debug, thiserror::Error)]
    #[error("invalid kernel body at {site}: {kind}")]
    pub struct VerifyError {
        pub site: Site,
        pub kind: VerifyErrorKind,
    }

    impl fmt::Display for Site {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Site::Signature => write!(f, "signature"),
                Site::Statement { block, index } => {
                    write!(f, "bb{} statement {index}", block.index)
                }
                Site::Terminator { block } => write!(f, "bb{} terminator", block.index),
            }
        }
    }

    type Check<T> = Result<T, VerifyErrorKind>;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FragmentRole {
        Matrix(WmmaMat),
        Accumulator,
    }

    impl FragmentRole {
        fn matrix(self) -> Option<WmmaMat> {
            match self {
                Self::Matrix(which) => Some(which),
                Self::Accumulator => None,
            }
        }
    }

    fn fragment_role_name(matrix: Option<WmmaMat>) -> &'static str {
        match matrix {
            Some(WmmaMat::A) => "matrix A",
            Some(WmmaMat::B) => "matrix B",
            None => "accumulator",
        }
    }

    fn fragment_producer(stmt: &Statement) -> Option<(Local, FragmentRole)> {
        match stmt {
            Statement::WmmaLoad { which, dst, .. } | Statement::WmmaLoadLds { which, dst, .. } => {
                Some((*dst, FragmentRole::Matrix(*which)))
            }
            Statement::WmmaZero { dst, .. } | Statement::WmmaMma { dst, .. } => {
                Some((*dst, FragmentRole::Accumulator))
            }
            _ => None,
        }
    }

    fn is_scalar(ty: &Ty) -> bool {
        matches!(
            ty,
            Ty::Bool | Ty::Usize | Ty::F16 | Ty::BF16 | Ty::F32 | Ty::F64 | Ty::I32 | Ty::U32
        )
    }

    fn is_integer(ty: &Ty) -> bool {
        matches!(ty, Ty::Usize | Ty::I32 | Ty::U32)
    }

    fn expect_ty(what: &'static str, expected: &Ty, found: &Ty) -> Check<()> {
        if expected == found {
            Ok(())
        } else {
            Err(VerifyErrorKind::TypeMismatch {
                what,
                expected: expected.clone(),
                found: found.clone(),
            })
        }
    }

    /// The buffer a global place addresses: the element type and whether stores are allowed.
    struct Element {
        ty: Ty,
        /// `Some(param)` for a buffer binding (`false` = read-only); `None` for a private array.
        binding: Option<(Local, bool)>,
    }

    impl Body {
        /// Check the body is well-formed before any backend sees it. Returns the first violation.
        ///
        /// Barrier uniformity (every thread of the workgroup reaching the same barrier) is decided by
        /// [`Self::verify_barrier_uniformity`]: a static divergence analysis independent of launch shape
        /// and input values (card 618).
        pub fn verify(&self) -> Result<(), VerifyError> {
            let at = |site| move |kind| VerifyError { site, kind };
            self.verify_signature().map_err(at(Site::Signature))?;
            for (b, block) in self.blocks.iter().enumerate() {
                let block_id = BlockId { index: b as u32 };
                for (index, stmt) in block.statements.iter().enumerate() {
                    self.verify_statement(stmt).map_err(at(Site::Statement {
                        block: block_id,
                        index,
                    }))?;
                }
                self.verify_terminator(&block.terminator)
                    .map_err(at(Site::Terminator { block: block_id }))?;
            }
            self.verify_fragment_roles()?;
            self.verify_definitions()?;
            self.verify_barrier_uniformity()
        }

        /// Collect known roles before checking consumers: CFG execution order need not match block
        /// storage order. Local indices have already passed the statement checks. This does not infer
        /// unknown roles from scalar marker types or prove collective participation; it is not a
        /// complete opaque-fragment type system.
        fn verify_fragment_roles(&self) -> Result<(), VerifyError> {
            let mut producers: Vec<Option<(FragmentRole, Site)>> = vec![None; self.locals.len()];
            for (b, block) in self.blocks.iter().enumerate() {
                for (index, stmt) in block.statements.iter().enumerate() {
                    let Some((local, role)) = fragment_producer(stmt) else {
                        continue;
                    };
                    let site = Site::Statement {
                        block: BlockId { index: b as u32 },
                        index,
                    };
                    let known = &mut producers[local.index as usize];
                    match *known {
                        Some((previous_role, previous)) if previous_role != role => {
                            return Err(VerifyError {
                                site,
                                kind: VerifyErrorKind::FragmentProducerRoleConflict {
                                    local,
                                    previous,
                                    previous_matrix: previous_role.matrix(),
                                    found_matrix: role.matrix(),
                                },
                            });
                        }
                        None => *known = Some((role, site)),
                        // Repeated same-role definitions include the ordinary in-place MMA K-loop.
                        Some(_) => {}
                    }
                }
            }
            let require = |local: Local, expected: FragmentRole| -> Check<()> {
                if let Some((found, _)) = producers[local.index as usize]
                    && found != expected
                {
                    return Err(VerifyErrorKind::FragmentRoleMismatch {
                        local,
                        expected_matrix: expected.matrix(),
                        found_matrix: found.matrix(),
                    });
                }
                Ok(())
            };
            for (b, block) in self.blocks.iter().enumerate() {
                for (index, stmt) in block.statements.iter().enumerate() {
                    let check = || -> Check<()> {
                        match stmt {
                            Statement::WmmaMma { a, b, c, .. } => {
                                require(*a, FragmentRole::Matrix(WmmaMat::A))?;
                                require(*b, FragmentRole::Matrix(WmmaMat::B))?;
                                require(*c, FragmentRole::Accumulator)
                            }
                            Statement::WmmaStore { src, .. }
                            | Statement::WmmaStoreLds { src, .. } => {
                                require(*src, FragmentRole::Accumulator)
                            }
                            _ => Ok(()),
                        }
                    };
                    check().map_err(|kind| VerifyError {
                        site: Site::Statement {
                            block: BlockId { index: b as u32 },
                            index,
                        },
                        kind,
                    })?;
                }
            }
            Ok(())
        }

        fn verify_signature(&self) -> Check<()> {
            if self.blocks.is_empty() {
                return Err(VerifyErrorKind::NoBlocks);
            }
            if self.locals.len() <= self.param_count as usize {
                return Err(VerifyErrorKind::ParamsExceedLocals {
                    param_count: self.param_count,
                    locals: self.locals.len(),
                });
            }
            if self.workgroup_size.contains(&0) {
                return Err(VerifyErrorKind::ZeroWorkgroupDim {
                    size: self.workgroup_size,
                });
            }
            for param in self.params() {
                let ty = self.local_ty(param);
                let is_buffer = matches!(
                    ty,
                    Ty::Ref { pointee, .. } if matches!(**pointee, Ty::Slice(_))
                );
                if !is_buffer {
                    return Err(VerifyErrorKind::ParamNotBuffer {
                        local: param,
                        ty: ty.clone(),
                    });
                }
            }
            Ok(())
        }

        fn declared(&self, l: Local) -> Check<&Ty> {
            self.locals
                .get(l.index as usize)
                .map(|d| &d.ty)
                .ok_or(VerifyErrorKind::UndeclaredLocal(l))
        }

        fn block_exists(&self, b: BlockId) -> Check<()> {
            if (b.index as usize) < self.blocks.len() {
                Ok(())
            } else {
                Err(VerifyErrorKind::UndefinedBlock(b))
            }
        }

        fn workgroup_array(&self, array: u8) -> Check<&WorkgroupLocalDecl> {
            self.workgroup_locals
                .get(array as usize)
                .ok_or(VerifyErrorKind::UndeclaredWorkgroupArray(array))
        }

        fn is_param(&self, l: Local) -> bool {
            l.index >= 1 && l.index <= self.param_count
        }

        /// The element a place addresses. A bare local is not an element (`None`).
        fn element(&self, place: &Place) -> Check<Option<Element>> {
            let base = self.declared(place.local)?;
            let malformed = |reason| VerifyErrorKind::MalformedPlace {
                place: place.clone(),
                reason,
            };
            let index_local =
                |i: Local| -> Check<()> { expect_ty("place index", &Ty::Usize, self.declared(i)?) };
            match place.projection.as_slice() {
                [] => Ok(None),
                [ProjectionElem::Index(i)] => {
                    index_local(*i)?;
                    match base {
                        Ty::Array { elem, .. } => Ok(Some(Element {
                            ty: (**elem).clone(),
                            binding: None,
                        })),
                        _ => Err(malformed("[Index] needs a private array local")),
                    }
                }
                [ProjectionElem::Deref, ProjectionElem::Index(i)] => {
                    index_local(*i)?;
                    if !self.is_param(place.local) {
                        return Err(malformed("[Deref, Index] needs a param buffer"));
                    }
                    match base {
                        Ty::Ref { mutable, pointee } => match &**pointee {
                            Ty::Slice(elem) => Ok(Some(Element {
                                ty: (**elem).clone(),
                                binding: Some((place.local, *mutable)),
                            })),
                            _ => Err(malformed("[Deref, Index] needs a slice reference")),
                        },
                        _ => Err(malformed("[Deref, Index] needs a slice reference")),
                    }
                }
                _ => Err(malformed("only [], [Index] and [Deref, Index] are places")),
            }
        }

        fn place_ty(&self, place: &Place) -> Check<Ty> {
            match self.element(place)? {
                Some(element) => Ok(element.ty),
                None => Ok(self.declared(place.local)?.clone()),
            }
        }

        /// The element of a place a store or atomic writes: it must be an element, and a buffer
        /// element must sit behind a `&mut` binding.
        fn writable_element(&self, place: &Place) -> Check<Element> {
            let element = self
                .element(place)?
                .ok_or_else(|| VerifyErrorKind::MalformedPlace {
                    place: place.clone(),
                    reason: "expected a buffer or array element",
                })?;
            self.require_writable(&element)?;
            Ok(element)
        }

        fn require_writable(&self, element: &Element) -> Check<()> {
            match element.binding {
                Some((param, false)) => Err(VerifyErrorKind::WriteThroughSharedBuffer { param }),
                _ => Ok(()),
            }
        }

        fn operand_ty(&self, op: &Operand) -> Check<Ty> {
            match op {
                Operand::Const(c) => Ok(match c {
                    Constant::Bool(_) => Ty::Bool,
                    Constant::Usize(_) => Ty::Usize,
                    Constant::I32(_) => Ty::I32,
                    Constant::U32(_) => Ty::U32,
                    Constant::F32(_) => Ty::F32,
                    Constant::F64(_) => Ty::F64,
                }),
                Operand::Copy(place) | Operand::Move(place) => self.place_ty(place),
            }
        }

        fn scalar_operand_ty(&self, what: &'static str, op: &Operand) -> Check<Ty> {
            let ty = self.operand_ty(op)?;
            if is_scalar(&ty) {
                Ok(ty)
            } else {
                Err(VerifyErrorKind::InvalidOperandType { what, found: ty })
            }
        }

        fn verify_statement(&self, stmt: &Statement) -> Check<()> {
            match stmt {
                Statement::Assign(place, rvalue) => {
                    let dest = self.place_ty(place)?;
                    if let Some(element) = self.element(place)? {
                        self.require_writable(&element)?;
                    }
                    let value = self.rvalue_ty(rvalue, &dest)?;
                    expect_ty("assigned value", &dest, &value)
                }
                Statement::StorageLive(l) | Statement::StorageDead(l) => {
                    self.declared(*l).map(|_| ())
                }
                Statement::WorkgroupLocalWrite { idx, value, array } => {
                    let decl = self.workgroup_array(*array)?;
                    expect_ty("workgroup array index", &Ty::Usize, &self.operand_ty(idx)?)?;
                    expect_ty(
                        "workgroup array store",
                        &decl.elem_ty,
                        &self.operand_ty(value)?,
                    )
                }
                Statement::VectorStore { place, value } => {
                    let element = self.writable_element(place)?;
                    match self.operand_ty(value)? {
                        Ty::Vec { elem, .. } => expect_ty("vector store lane", &element.ty, &elem),
                        other => Err(VerifyErrorKind::InvalidOperandType {
                            what: "vector store value",
                            found: other,
                        }),
                    }
                }
                Statement::WmmaLoad {
                    shape, tile, dst, ..
                } => {
                    self.check_wmma_shape(*shape)?;
                    self.element(tile)?;
                    self.declared_fragment(*dst)
                }
                Statement::WmmaMma {
                    shape,
                    a,
                    b,
                    c,
                    dst,
                    ..
                } => {
                    self.check_wmma_shape(*shape)?;
                    for fragment in [*a, *b, *c, *dst] {
                        self.declared_fragment(fragment)?;
                    }
                    Ok(())
                }
                Statement::WmmaStore {
                    shape, tile, src, ..
                } => {
                    self.check_wmma_shape(*shape)?;
                    self.writable_element(tile)?;
                    self.declared_fragment(*src)
                }
                Statement::WmmaZero { shape, dst, .. } => {
                    self.check_wmma_shape(*shape)?;
                    self.declared_fragment(*dst)
                }
                Statement::WmmaLoadLds {
                    shape, array, dst, ..
                } => {
                    self.check_wmma_shape(*shape)?;
                    self.workgroup_array(*array)?;
                    self.declared_fragment(*dst)
                }
                Statement::WmmaStoreLds {
                    shape, array, src, ..
                } => {
                    self.check_wmma_shape(*shape)?;
                    self.workgroup_array(*array)?;
                    self.declared_fragment(*src)
                }
                Statement::Fence { .. } => Ok(()),
            }
        }

        /// Statement-local declaration check. Known producer/consumer roles are checked separately
        /// by `verify_fragment_roles`; `Ty` remains a marker, not the fragment's type.
        fn declared_fragment(&self, l: Local) -> Check<()> {
            self.declared(l).map(|_| ())
        }

        /// Card 530: only `WmmaShape::M16N16K16` is implemented by any target's codegen
        /// today; a body naming a different shape is malformed at the IR level, the same way an
        /// out-of-range local or a type-mismatched operand is.
        fn check_wmma_shape(&self, shape: WmmaShape) -> Check<()> {
            if shape == WmmaShape::M16N16K16 {
                Ok(())
            } else {
                Err(VerifyErrorKind::UnsupportedWmmaShape { shape })
            }
        }

        /// The type an rvalue produces, after checking its operands. `dest` is the assigned place's type,
        /// which the vector rvalues take their shape from.
        fn rvalue_ty(&self, rvalue: &Rvalue, dest: &Ty) -> Check<Ty> {
            match rvalue {
                Rvalue::Use(op) => self.operand_ty(op),
                Rvalue::BinaryOp(op, a, b) => {
                    let ta = self.operand_ty(a)?;
                    let tb = self.operand_ty(b)?;
                    match op {
                        BinOp::Shl | BinOp::Shr => {
                            for (what, ty) in [("shifted value", &ta), ("shift amount", &tb)] {
                                if !is_integer(ty) {
                                    return Err(VerifyErrorKind::InvalidOperandType {
                                        what,
                                        found: ty.clone(),
                                    });
                                }
                            }
                            Ok(ta)
                        }
                        BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne => {
                            expect_ty("comparison operands", &ta, &tb)?;
                            Ok(Ty::Bool)
                        }
                        _ => {
                            expect_ty("binary operands", &ta, &tb)?;
                            Ok(ta)
                        }
                    }
                }
                Rvalue::BinaryOpNoContract(op, a, b) => {
                    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
                        return Err(VerifyErrorKind::UnsupportedNoContractOp { op: *op });
                    }
                    let ta = self.operand_ty(a)?;
                    let tb = self.operand_ty(b)?;
                    expect_ty("no-contract binary operands", &ta, &tb)?;
                    if !ta.is_float() {
                        return Err(VerifyErrorKind::InvalidOperandType {
                            what: "no-contract binary operand",
                            found: ta,
                        });
                    }
                    Ok(ta)
                }
                Rvalue::UnaryOp(_, a) => self.operand_ty(a),
                Rvalue::MathUnary(_, a) => {
                    let ty = self.operand_ty(a)?;
                    if ty.is_float() {
                        Ok(ty)
                    } else {
                        Err(VerifyErrorKind::InvalidOperandType {
                            what: "float math operand",
                            found: ty,
                        })
                    }
                }
                Rvalue::IntScalarUnary(_, a) => {
                    // The count is a 32-bit word of the operand's own signedness (`0..=32`).
                    match self.operand_ty(a)? {
                        ty @ (Ty::U32 | Ty::I32) => Ok(ty),
                        found => Err(VerifyErrorKind::InvalidOperandType {
                            what: "integer bit-op operand",
                            found,
                        }),
                    }
                }
                Rvalue::Len(place) => {
                    let ty = self.declared(place.local)?;
                    let is_buffer = self.is_param(place.local)
                        && matches!(ty, Ty::Ref { pointee, .. } if matches!(**pointee, Ty::Slice(_)));
                    // MIR spells the slice length `Len(*param)`; the bare param is accepted too.
                    let is_whole_buffer =
                        matches!(place.projection.as_slice(), [] | [ProjectionElem::Deref]);
                    if is_whole_buffer && is_buffer {
                        Ok(Ty::Usize)
                    } else {
                        Err(VerifyErrorKind::MalformedPlace {
                            place: place.clone(),
                            reason: "Len needs a buffer param, optionally dereferenced",
                        })
                    }
                }
                Rvalue::Cast { to, operand } => {
                    self.scalar_operand_ty("cast operand", operand)?;
                    if is_scalar(to) {
                        Ok(to.clone())
                    } else {
                        Err(VerifyErrorKind::InvalidOperandType {
                            what: "cast target",
                            found: to.clone(),
                        })
                    }
                }
                Rvalue::Bitcast { to, operand } => {
                    self.scalar_operand_ty("bitcast operand", operand)?;
                    if is_scalar(to) {
                        Ok(to.clone())
                    } else {
                        Err(VerifyErrorKind::InvalidOperandType {
                            what: "bitcast target",
                            found: to.clone(),
                        })
                    }
                }
                Rvalue::Fp8Decode { operand, .. } => {
                    expect_ty("fp8 decode operand", &Ty::U32, &self.operand_ty(operand)?)?;
                    Ok(Ty::F32)
                }
                Rvalue::Fp8Encode { operand, .. } => {
                    expect_ty("fp8 encode operand", &Ty::F32, &self.operand_ty(operand)?)?;
                    Ok(Ty::U32)
                }
                Rvalue::WorkgroupLocalRead { idx, array } => {
                    let decl = self.workgroup_array(*array)?;
                    expect_ty("workgroup array index", &Ty::Usize, &self.operand_ty(idx)?)?;
                    Ok(decl.elem_ty.clone())
                }
                Rvalue::WorkgroupLocalAtomic {
                    idx, value, array, ..
                } => {
                    let decl = self.workgroup_array(*array)?;
                    expect_ty("workgroup array index", &Ty::Usize, &self.operand_ty(idx)?)?;
                    expect_ty("atomic operand", &decl.elem_ty, &self.operand_ty(value)?)?;
                    Ok(decl.elem_ty.clone())
                }
                Rvalue::GlobalAtomic { place, value, .. } => {
                    let element = self.writable_element(place)?;
                    expect_ty("atomic operand", &element.ty, &self.operand_ty(value)?)?;
                    Ok(element.ty)
                }
                Rvalue::WorkgroupLocalCompareExchange {
                    idx,
                    expected,
                    desired,
                    array,
                } => {
                    let decl = self.workgroup_array(*array)?;
                    expect_ty("workgroup array index", &Ty::Usize, &self.operand_ty(idx)?)?;
                    expect_ty("expected value", &decl.elem_ty, &self.operand_ty(expected)?)?;
                    expect_ty("desired value", &decl.elem_ty, &self.operand_ty(desired)?)?;
                    Ok(decl.elem_ty.clone())
                }
                Rvalue::GlobalCompareExchange {
                    place,
                    expected,
                    desired,
                } => {
                    let element = self.writable_element(place)?;
                    expect_ty("expected value", &element.ty, &self.operand_ty(expected)?)?;
                    expect_ty("desired value", &element.ty, &self.operand_ty(desired)?)?;
                    Ok(element.ty)
                }
                Rvalue::VectorLoad { place } => {
                    let element =
                        self.element(place)?
                            .ok_or_else(|| VerifyErrorKind::MalformedPlace {
                                place: place.clone(),
                                reason: "VectorLoad needs a buffer element",
                            })?;
                    match dest {
                        Ty::Vec { elem, .. } => {
                            expect_ty("vector load lane", &element.ty, elem)?;
                            Ok(dest.clone())
                        }
                        other => Err(VerifyErrorKind::InvalidOperandType {
                            what: "vector load destination",
                            found: other.clone(),
                        }),
                    }
                }
                Rvalue::VectorSplat(op) => match dest {
                    Ty::Vec { elem, .. } => {
                        expect_ty("splat operand", elem, &self.operand_ty(op)?)?;
                        Ok(dest.clone())
                    }
                    other => Err(VerifyErrorKind::InvalidOperandType {
                        what: "splat destination",
                        found: other.clone(),
                    }),
                },
            }
        }

        fn verify_terminator(&self, term: &Terminator) -> Check<()> {
            match term {
                Terminator::Goto { target } | Terminator::Barrier { target } => {
                    self.block_exists(*target)
                }
                Terminator::SwitchInt { discr, targets } => {
                    let ty = self.operand_ty(discr)?;
                    if ty != Ty::Bool && !is_integer(&ty) {
                        return Err(VerifyErrorKind::InvalidOperandType {
                            what: "switch discriminant",
                            found: ty,
                        });
                    }
                    for (_, target) in &targets.branches {
                        self.block_exists(*target)?;
                    }
                    self.block_exists(targets.otherwise)
                }
                Terminator::ThreadIndexCall {
                    destination,
                    target,
                    ..
                } => {
                    expect_ty(
                        "thread index destination",
                        &Ty::Usize,
                        &self.place_ty(destination)?,
                    )?;
                    self.block_exists(*target)
                }
                Terminator::Return | Terminator::Trap { .. } => Ok(()),
            }
        }

        /// Every local read is assigned on every path from the entry to the read (forward must-analysis).
        /// Params are bound at entry and private arrays are filled element by element, so both count as
        /// defined from the start. Blocks the entry cannot reach are not checked.
        fn verify_definitions(&self) -> Result<(), VerifyError> {
            let n = self.locals.len();
            let mut entry_state = vec![false; n];
            for (i, decl) in self.locals.iter().enumerate() {
                entry_state[i] =
                    self.is_param(Local { index: i as u32 }) || matches!(decl.ty, Ty::Array { .. });
            }
            // `state_in[b]`: locals assigned on every path into `b`; `None` until a path reaches it.
            let mut state_in: Vec<Option<Vec<bool>>> = vec![None; self.blocks.len()];
            state_in[0] = Some(entry_state);
            let mut work = vec![0usize];
            while let Some(b) = work.pop() {
                let mut state = state_in[b].clone().expect("queued blocks have a state");
                for stmt in &self.blocks[b].statements {
                    stmt_defs(stmt, &mut |l| state[l.index as usize] = true);
                }
                terminator_defs(&self.blocks[b].terminator, &mut |l| {
                    state[l.index as usize] = true
                });
                for succ in successors(&self.blocks[b].terminator) {
                    let s = succ.index as usize;
                    let merged = match &state_in[s] {
                        None => state.clone(),
                        Some(prev) => prev.iter().zip(&state).map(|(p, q)| *p && *q).collect(),
                    };
                    if state_in[s].as_ref() != Some(&merged) {
                        state_in[s] = Some(merged);
                        work.push(s);
                    }
                }
            }
            for (b, block) in self.blocks.iter().enumerate() {
                let Some(mut state) = state_in[b].clone() else {
                    continue;
                };
                let block_id = BlockId { index: b as u32 };
                for (index, stmt) in block.statements.iter().enumerate() {
                    let mut undefined = None;
                    stmt_uses(stmt, &mut |l| {
                        if !state[l.index as usize] {
                            undefined.get_or_insert(l);
                        }
                    });
                    if let Some(l) = undefined {
                        return Err(VerifyError {
                            site: Site::Statement {
                                block: block_id,
                                index,
                            },
                            kind: VerifyErrorKind::UseBeforeDefinition(l),
                        });
                    }
                    stmt_defs(stmt, &mut |l| state[l.index as usize] = true);
                }
                let mut undefined = None;
                terminator_uses(&block.terminator, &mut |l| {
                    if !state[l.index as usize] {
                        undefined.get_or_insert(l);
                    }
                });
                if let Some(l) = undefined {
                    return Err(VerifyError {
                        site: Site::Terminator { block: block_id },
                        kind: VerifyErrorKind::UseBeforeDefinition(l),
                    });
                }
            }
            Ok(())
        }

        /// The static counterpart to `interp`'s dynamic lockstep check (card 618, hardened by its
        /// review): reject a `SwitchInt` whose discriminant can differ between lanes of one workgroup
        /// unless every [`Terminator::Barrier`] reachable from it sits at or after its immediate
        /// post-dominator - the reconvergence point every arm is guaranteed (not merely able) to reach.
        /// Runs after [`Self::verify_definitions`], so every local read here is already known to be
        /// assigned on every path that reaches it.
        fn verify_barrier_uniformity(&self) -> Result<(), VerifyError> {
            let ipdom = post_dominators(&self.blocks);
            // `state_in[b]`: for every local, the [`FlowState`] guaranteed to hold on every path into
            // `b` (the same forward must/may-analysis shape as `verify_definitions`, tracking both
            // thread-uniformity and a possibly-racing buffer write instead of definite assignment).
            let mut state_in: Vec<Option<FlowState>> = vec![None; self.blocks.len()];
            state_in[0] = Some(FlowState::entry(self.locals.len()));
            let mut work = vec![0usize];
            while let Some(b) = work.pop() {
                let mut flow = state_in[b].clone().expect("queued blocks have a state");
                flow.apply_block(&self.blocks[b]);
                for succ in successors(&self.blocks[b].terminator) {
                    let s = succ.index as usize;
                    let merged = match &state_in[s] {
                        None => flow.clone(),
                        Some(prev) => prev.merge(&flow),
                    };
                    if state_in[s].as_ref() != Some(&merged) {
                        state_in[s] = Some(merged);
                        work.push(s);
                    }
                }
            }
            for (b, block) in self.blocks.iter().enumerate() {
                let Some(mut flow) = state_in[b].clone() else {
                    continue; // unreachable from the entry block; nothing to check.
                };
                for stmt in &block.statements {
                    flow.apply_statement(stmt);
                }
                let Terminator::SwitchInt { discr, .. } = &block.terminator else {
                    continue;
                };
                if operand_uniform(discr, &flow) {
                    continue;
                }
                let block_id = BlockId { index: b as u32 };
                let reconverge = ipdom[b];
                if let Some(barrier) = self.barrier_before(block_id, reconverge) {
                    return Err(VerifyError {
                        site: Site::Terminator { block: block_id },
                        kind: VerifyErrorKind::DivergentBarrierReachability {
                            branch: block_id,
                            barrier,
                            reconverge,
                        },
                    });
                }
            }
            Ok(())
        }

        /// The first `Barrier`-terminated block reachable from `branch`'s own arms without first
        /// reaching `reconverge` (its immediate post-dominator): a barrier found here sits strictly
        /// before the point every arm is guaranteed to reach, so some lanes could stop there while
        /// others take a different arm past it, or to a different barrier, before ever reconverging -
        /// unsafe regardless of what happens at or after `reconverge` itself (even a barrier there is
        /// fine: every arm is guaranteed to reach that exact block). `reconverge: None` (no block
        /// post-dominates every arm) searches unbounded: with no guaranteed reunion at all, any
        /// reachable barrier is already unprovable.
        fn barrier_before(&self, branch: BlockId, reconverge: Option<BlockId>) -> Option<BlockId> {
            let mut visited = HashSet::new();
            let mut work = successors(&self.blocks[branch.index as usize].terminator);
            while let Some(b) = work.pop() {
                if Some(b) == reconverge || !visited.insert(b) {
                    continue;
                }
                let block = &self.blocks[b.index as usize];
                if matches!(block.terminator, Terminator::Barrier { .. }) {
                    return Some(b);
                }
                work.extend(successors(&block.terminator));
            }
            None
        }
    }

    /// A local's classification for [`Body::verify_barrier_uniformity`]'s forward dataflow: whether its
    /// value is guaranteed identical across every lane of the workgroup (`uniform`), and whether a write
    /// to it (only meaningful for a buffer param local) may have happened on some path from the entry
    /// without an intervening barrier washing it out (`racy`) - card 618: a scalar load
    /// at a uniform index is uniform only when no unsynchronized write to the same buffer can race it.
    #[derive(Clone, PartialEq, Debug)]
    struct FlowState {
        uniform: Vec<bool>,
        racy: Vec<bool>,
    }

    impl FlowState {
        /// The state at the body's entry: every local uniform (params and never-yet-defined temporaries
        /// alike - a temporary is always overwritten by a uniform-or-not statement before any real use,
        /// already guaranteed by [`Body::verify_definitions`]), nothing yet written.
        fn entry(n: usize) -> Self {
            FlowState {
                uniform: vec![true; n],
                racy: vec![false; n],
            }
        }

        /// The meet of two paths into the same block: uniform only where both agree (a local can be
        /// varying via either path), racy where either does (a write on either path can still race a
        /// read here).
        fn merge(&self, other: &FlowState) -> FlowState {
            FlowState {
                uniform: self
                    .uniform
                    .iter()
                    .zip(&other.uniform)
                    .map(|(a, b)| *a && *b)
                    .collect(),
                racy: self
                    .racy
                    .iter()
                    .zip(&other.racy)
                    .map(|(a, b)| *a || *b)
                    .collect(),
            }
        }

        /// Apply every statement of `block`, then its `ThreadIndexCall`/`Barrier` terminator effect,
        /// updating `self` in place to the state as of leaving `block` (before any successor is chosen).
        fn apply_block(&mut self, block: &BasicBlock) {
            for stmt in &block.statements {
                self.apply_statement(stmt);
            }
            apply_thread_index_uniformity(&block.terminator, &mut self.uniform);
            if matches!(block.terminator, Terminator::Barrier { .. }) {
                // Every lane's writes before the barrier are visible to every lane's reads after it
                // (the same contract `interp`'s module doc describes), so nothing is racy anymore.
                self.racy.iter_mut().for_each(|r| *r = false);
            }
        }

        /// Apply one statement: update the local it defines (if any), then record the buffer it writes
        /// (if any) - in that order, so the definition sees the racy state as of *before* this
        /// statement's own write.
        fn apply_statement(&mut self, stmt: &Statement) {
            if let Some((idx, uniform)) = stmt_uniform(stmt, self) {
                self.uniform[idx] = uniform;
            }
            if let Some(written) = stmt_writes_buffer(stmt) {
                self.racy[written.index as usize] = true;
            }
        }
    }

    /// The immediate post-dominator of every block: the nearest block every path from it - through the
    /// real control-flow graph, a `Barrier`'s own target included - is guaranteed to reach, treating
    /// every `Return` as flowing into one shared virtual exit (the standard way to make post-dominance
    /// well-defined over a CFG with more than one exit; card 618). `None` for a
    /// block that cannot reach that exit at all (every path from it loops forever): its arms' safety
    /// cannot be proven to reconverge, so [`Body::barrier_before`] then searches unbounded instead of
    /// stopping at a (non-existent) reconvergence point.
    fn post_dominators(blocks: &[BasicBlock]) -> Vec<Option<BlockId>> {
        let n = blocks.len();
        let exit = n;
        let total = n + 1;
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); total];
        for (b, block) in blocks.iter().enumerate() {
            succ[b] = match &block.terminator {
                Terminator::Return => vec![exit],
                other => successors(other)
                    .into_iter()
                    .map(|t| t.index as usize)
                    .collect(),
            };
        }
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); total];
        for (u, targets) in succ.iter().enumerate() {
            for &v in targets {
                preds[v].push(u);
            }
        }
        // Postorder DFS from `exit` walking backward (adjacency = `preds`): reversed, this gives the
        // standard dominance algorithm's traversal order, `exit` first (rpo 0 = closest to the root).
        let mut reaches_exit = vec![false; total];
        let mut postorder = Vec::with_capacity(total);
        let mut stack: Vec<(usize, usize)> = vec![(exit, 0)];
        reaches_exit[exit] = true;
        while let Some(top) = stack.last_mut() {
            let (node, idx) = *top;
            if idx < preds[node].len() {
                top.1 += 1;
                let child = preds[node][idx];
                if !reaches_exit[child] {
                    reaches_exit[child] = true;
                    stack.push((child, 0));
                }
            } else {
                postorder.push(node);
                stack.pop();
            }
        }
        let mut rpo = vec![usize::MAX; total];
        for (i, &node) in postorder.iter().rev().enumerate() {
            rpo[node] = i;
        }
        let order: Vec<usize> = postorder
            .iter()
            .rev()
            .copied()
            .filter(|&x| x != exit)
            .collect();
        let mut idom: Vec<Option<usize>> = vec![None; total];
        idom[exit] = Some(exit);
        let mut changed = true;
        while changed {
            changed = false;
            for &node in &order {
                let mut new_idom: Option<usize> = None;
                for &p in &succ[node] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => p,
                        Some(cur) => intersect(cur, p, &idom, &rpo),
                    });
                }
                if let Some(ni) = new_idom
                    && idom[node] != Some(ni)
                {
                    idom[node] = Some(ni);
                    changed = true;
                }
            }
        }
        (0..n)
            .map(|b| {
                idom[b]
                    .filter(|&d| d != exit)
                    .map(|d| BlockId { index: d as u32 })
            })
            .collect()
    }

    /// The nearest common ancestor of `a`/`b` in the (partially built) post-dominator tree: walk the
    /// finger with the larger `rpo` number up its `idom` chain until both fingers meet (Cooper, Harvey
    /// and Kennedy's `Intersect`; `rpo` numbers a node's dominators as strictly smaller, so "larger rpo"
    /// means "further from the root").
    fn intersect(mut a: usize, mut b: usize, idom: &[Option<usize>], rpo: &[usize]) -> usize {
        while a != b {
            while rpo[a] > rpo[b] {
                a = idom[a].expect("a processed node has an idom");
            }
            while rpo[b] > rpo[a] {
                b = idom[b].expect("a processed node has an idom");
            }
        }
        a
    }

    /// `Some((local index, is-uniform))` when `stmt` defines a local's value directly: a bare-local
    /// `Assign` (mirrors [`stmt_defs`]), or a matrix-fragment op's `dst` (card 618): an
    /// opaque fragment value is never meaningfully uniform, and nothing enforces that a fragment-typed
    /// local can never also be declared a plain scalar and read by a `SwitchInt` - do not rest this
    /// analysis on that being true elsewhere; mark it varying directly. `None` otherwise (a store through
    /// a place defines no local).
    fn stmt_uniform(stmt: &Statement, flow: &FlowState) -> Option<(usize, bool)> {
        match stmt {
            Statement::Assign(place, rvalue) if place.projection.is_empty() => {
                Some((place.local.index as usize, rvalue_uniform(rvalue, flow)))
            }
            Statement::WmmaLoad { dst, .. }
            | Statement::WmmaMma { dst, .. }
            | Statement::WmmaZero { dst, .. }
            | Statement::WmmaLoadLds { dst, .. } => Some((dst.index as usize, false)),
            _ => None,
        }
    }

    /// The buffer `stmt` writes to, if any (mirrors [`stmt_uses`]'s write-through-a-place case, plus the
    /// two rvalues that write as a side effect of computing their result): a plain store through a place
    /// (`place.projection` non-empty - a bare-local `Assign` defines a local instead, never a store), a
    /// vector or matrix-fragment store, or a global atomic/compare-exchange's target. Feeds
    /// [`FlowState::racy`] (card 618); a workgroup-local (LDS) write is not a buffer and
    /// is irrelevant here (an LDS read is unconditionally thread-varying regardless of racy state).
    fn stmt_writes_buffer(stmt: &Statement) -> Option<Local> {
        match stmt {
            Statement::Assign(place, rvalue) => {
                if !place.projection.is_empty() {
                    return Some(place.local);
                }
                match rvalue {
                    Rvalue::GlobalAtomic { place, .. }
                    | Rvalue::GlobalCompareExchange { place, .. } => Some(place.local),
                    _ => None,
                }
            }
            Statement::VectorStore { place, .. } => Some(place.local),
            Statement::WmmaStore { tile, .. } => Some(tile.local),
            _ => None,
        }
    }

    /// `Some(dim)`'s `ThreadIndexCall` sets its destination's uniformity: `GroupX`/`GroupY`/`GroupZ` (the
    /// workgroup index) are the same for every lane of one workgroup; every other axis (the local or
    /// global dispatch id) varies lane to lane by construction.
    fn apply_thread_index_uniformity(term: &Terminator, uniform: &mut [bool]) {
        if let Terminator::ThreadIndexCall {
            destination, dim, ..
        } = term
            && destination.projection.is_empty()
        {
            uniform[destination.local.index as usize] = matches!(
                dim,
                IndexAxis::GroupX | IndexAxis::GroupY | IndexAxis::GroupZ
            );
        }
    }

    /// An [`Rvalue`]'s result is thread-uniform iff every operand it reads is: constants and
    /// [`Rvalue::Len`] (a buffer's length, the same for every lane) are always uniform; a workgroup-local
    /// or atomic/compare-exchange read is conservatively thread-varying (its index, ordering or contents
    /// may differ per lane, and this analysis does not track them precisely enough to prove otherwise).
    fn rvalue_uniform(rvalue: &Rvalue, flow: &FlowState) -> bool {
        match rvalue {
            Rvalue::Use(op)
            | Rvalue::UnaryOp(_, op)
            | Rvalue::MathUnary(_, op)
            | Rvalue::IntScalarUnary(_, op)
            | Rvalue::VectorSplat(op) => operand_uniform(op, flow),
            Rvalue::BinaryOp(_, a, b) | Rvalue::BinaryOpNoContract(_, a, b) => {
                operand_uniform(a, flow) && operand_uniform(b, flow)
            }
            Rvalue::Cast { operand, .. }
            | Rvalue::Bitcast { operand, .. }
            | Rvalue::Fp8Decode { operand, .. }
            | Rvalue::Fp8Encode { operand, .. } => operand_uniform(operand, flow),
            Rvalue::Len(_) => true,
            Rvalue::WorkgroupLocalRead { .. }
            | Rvalue::WorkgroupLocalAtomic { .. }
            | Rvalue::GlobalAtomic { .. }
            | Rvalue::WorkgroupLocalCompareExchange { .. }
            | Rvalue::GlobalCompareExchange { .. }
            | Rvalue::VectorLoad { .. } => false,
        }
    }

    fn operand_uniform(op: &Operand, flow: &FlowState) -> bool {
        match op {
            Operand::Const(_) => true,
            Operand::Copy(place) | Operand::Move(place) => place_uniform(place, flow),
        }
    }

    /// A bare local's uniformity comes from `flow.uniform`; reading through any projection (a buffer or
    /// private array element) is thread-varying unless it is a `[Deref, Index]` element of a shared
    /// buffer (`element()`'s `binding: Some(..)`) at a thread-uniform index *and* the buffer is not
    /// `flow.racy` (card 618): with both, every lane addresses the same memory with no
    /// unsynchronized write racing the read, so every lane reads back the same value (the "scalar load"
    /// every real GPU compiler's uniformity analysis makes - `offsets[expert]` with a `GroupX`-derived
    /// `expert`, from the grouped-dequant-gemm kernel, and no store to `offsets` anywhere in the body, is
    /// exactly this). A private array's `[Index]` element is never uniform this way even at a uniform,
    /// race-free index: it is per-thread storage, so different lanes can hold different contents at the
    /// same index.
    fn place_uniform(place: &Place, flow: &FlowState) -> bool {
        match place.projection.as_slice() {
            [] => flow.uniform[place.local.index as usize],
            [ProjectionElem::Deref, ProjectionElem::Index(i)] => {
                flow.uniform[i.index as usize] && !flow.racy[place.local.index as usize]
            }
            _ => false,
        }
    }

    fn successors(term: &Terminator) -> Vec<BlockId> {
        match term {
            Terminator::Goto { target }
            | Terminator::Barrier { target }
            | Terminator::ThreadIndexCall { target, .. } => vec![*target],
            Terminator::SwitchInt { targets, .. } => targets
                .branches
                .iter()
                .map(|(_, b)| *b)
                .chain([targets.otherwise])
                .collect(),
            Terminator::Return | Terminator::Trap { .. } => Vec::new(),
        }
    }

    /// Locals a place reads: the base and every index. Writing through a projection reads them too.
    fn place_reads(place: &Place, f: &mut dyn FnMut(Local)) {
        f(place.local);
        for elem in &place.projection {
            if let ProjectionElem::Index(i) = elem {
                f(*i);
            }
        }
    }

    fn operand_reads(op: &Operand, f: &mut dyn FnMut(Local)) {
        if let Operand::Copy(p) | Operand::Move(p) = op {
            place_reads(p, f);
        }
    }

    /// The locals a place assignment defines: only a bare local, never an element.
    fn place_def(place: &Place, f: &mut dyn FnMut(Local)) {
        if place.projection.is_empty() {
            f(place.local);
        }
    }

    fn stmt_defs(stmt: &Statement, f: &mut dyn FnMut(Local)) {
        match stmt {
            Statement::Assign(place, _) => place_def(place, f),
            Statement::WmmaLoad { dst, .. }
            | Statement::WmmaMma { dst, .. }
            | Statement::WmmaZero { dst, .. }
            | Statement::WmmaLoadLds { dst, .. } => f(*dst),
            _ => {}
        }
    }

    fn terminator_defs(term: &Terminator, f: &mut dyn FnMut(Local)) {
        if let Terminator::ThreadIndexCall { destination, .. } = term {
            place_def(destination, f);
        }
    }

    fn rvalue_reads(rvalue: &Rvalue, f: &mut dyn FnMut(Local)) {
        match rvalue {
            Rvalue::Use(a)
            | Rvalue::UnaryOp(_, a)
            | Rvalue::MathUnary(_, a)
            | Rvalue::IntScalarUnary(_, a)
            | Rvalue::VectorSplat(a) => operand_reads(a, f),
            Rvalue::Cast { operand, .. }
            | Rvalue::Bitcast { operand, .. }
            | Rvalue::Fp8Decode { operand, .. }
            | Rvalue::Fp8Encode { operand, .. } => operand_reads(operand, f),
            Rvalue::BinaryOp(_, a, b) | Rvalue::BinaryOpNoContract(_, a, b) => {
                operand_reads(a, f);
                operand_reads(b, f);
            }
            Rvalue::Len(place) | Rvalue::VectorLoad { place } => place_reads(place, f),
            Rvalue::WorkgroupLocalRead { idx, .. } => operand_reads(idx, f),
            Rvalue::WorkgroupLocalAtomic { idx, value, .. } => {
                operand_reads(idx, f);
                operand_reads(value, f);
            }
            Rvalue::GlobalAtomic { place, value, .. } => {
                place_reads(place, f);
                operand_reads(value, f);
            }
            Rvalue::WorkgroupLocalCompareExchange {
                idx,
                expected,
                desired,
                ..
            } => {
                operand_reads(idx, f);
                operand_reads(expected, f);
                operand_reads(desired, f);
            }
            Rvalue::GlobalCompareExchange {
                place,
                expected,
                desired,
            } => {
                place_reads(place, f);
                operand_reads(expected, f);
                operand_reads(desired, f);
            }
        }
    }

    fn stmt_uses(stmt: &Statement, f: &mut dyn FnMut(Local)) {
        match stmt {
            Statement::Assign(place, rvalue) => {
                // Storing to a bare local reads nothing; storing to an element reads the base and index.
                if !place.projection.is_empty() {
                    place_reads(place, f);
                }
                rvalue_reads(rvalue, f);
            }
            Statement::StorageLive(_) | Statement::StorageDead(_) | Statement::Fence { .. } => {}
            Statement::WorkgroupLocalWrite { idx, value, .. } => {
                operand_reads(idx, f);
                operand_reads(value, f);
            }
            Statement::VectorStore { place, value } => {
                place_reads(place, f);
                operand_reads(value, f);
            }
            Statement::WmmaLoad { tile, .. } => place_reads(tile, f),
            Statement::WmmaMma { a, b, c, .. } => {
                f(*a);
                f(*b);
                f(*c);
            }
            Statement::WmmaStore { tile, src, .. } => {
                place_reads(tile, f);
                f(*src);
            }
            Statement::WmmaStoreLds { src, .. } => f(*src),
            Statement::WmmaZero { .. } | Statement::WmmaLoadLds { .. } => {}
        }
    }

    fn terminator_uses(term: &Terminator, f: &mut dyn FnMut(Local)) {
        match term {
            Terminator::SwitchInt { discr, .. } => operand_reads(discr, f),
            Terminator::ThreadIndexCall { destination, .. } => {
                if !destination.projection.is_empty() {
                    place_reads(destination, f);
                }
            }
            Terminator::Goto { .. }
            | Terminator::Barrier { .. }
            | Terminator::Return
            | Terminator::Trap { .. } => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build `add(a: Slice<f32>, b: Slice<f32>, c: SliceMut<f32>)` by hand and check its structure.
    fn add_body() -> Body {
        // locals: _0 unit (ret), _1 a &[f32], _2 b &[f32], _3 c &mut [f32], _4 i usize, _5 len usize,
        // _6 cmp bool, _7 sum f32.
        let slice_f32 = Ty::Ref {
            mutable: false,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
        };
        let slice_mut_f32 = Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
        };
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32.clone(),
                mutable: false,
            },
            LocalDecl {
                ty: slice_f32,
                mutable: false,
            },
            LocalDecl {
                ty: slice_mut_f32,
                mutable: true,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Bool,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::F32,
                mutable: false,
            },
        ];
        let i = Local { index: 4 };
        let len = Local { index: 5 };
        // bb0: i = thread_index(X) -> bb1
        let bb0 = BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        };
        // bb1: len = Len(c); switch (i < len) -> bb2 else bb3
        let cmp = Local { index: 6 };
        let bb1 = BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(len),
                    Rvalue::Len(Place::local(Local { index: 3 })),
                ),
                Statement::Assign(
                    Place::local(cmp),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        Operand::Copy(Place::local(i)),
                        Operand::Copy(Place::local(len)),
                    ),
                ),
            ],
            terminator: Terminator::SwitchInt {
                discr: Operand::Copy(Place::local(cmp)),
                targets: SwitchTargets {
                    branches: vec![(0, BlockId { index: 3 })],
                    otherwise: BlockId { index: 2 },
                },
            },
        };
        // bb2: c[i] = a[i] + b[i]; return
        let idx = |l: u32| Place {
            local: Local { index: l },
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(i)],
        };
        let sum = Local { index: 7 };
        let bb2 = BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(sum),
                    Rvalue::BinaryOp(BinOp::Add, Operand::Copy(idx(1)), Operand::Copy(idx(2))),
                ),
                Statement::Assign(idx(3), Rvalue::Use(Operand::Copy(Place::local(sum)))),
            ],
            terminator: Terminator::Return,
        };
        // bb3: return
        let bb3 = BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        };
        Body::new("add", 3, locals, vec![bb0, bb1, bb2, bb3])
    }

    #[test]
    fn add_body_structure() {
        let b = add_body();
        assert_eq!(b.name, "add");
        assert_eq!(b.param_count, 3);
        assert_eq!(b.params().count(), 3);
        assert_eq!(b.workgroup_size, [64, 1, 1]);
        assert!(b.workgroup_locals.is_empty());
        assert_eq!(b.blocks.len(), 4);
        // the slice-element access pattern (*c)[i].
        if let Statement::Assign(place, _) = &b.blocks[2].statements[1] {
            assert_eq!(
                place.projection,
                vec![
                    ProjectionElem::Deref,
                    ProjectionElem::Index(Local { index: 4 })
                ]
            );
        } else {
            panic!("expected an assign");
        }
    }

    #[test]
    fn ty_helpers() {
        assert!(Ty::F32.is_float());
        assert!(!Ty::U32.is_float());
        assert!(Ty::I32.is_signed_int());
    }

    /// `Ty::Vec` delegates `is_float`/`is_signed_int` to its element type, so `emit_binop` (which picks
    /// `fadd` vs `add` from them) works unchanged on vector operands (spec 134).
    #[test]
    fn vec_ty_delegates_to_elem() {
        let f32x4 = Ty::Vec {
            elem: Box::new(Ty::F32),
            lanes: 4,
        };
        assert!(f32x4.is_float());
        assert!(!f32x4.is_signed_int());
        let i32x2 = Ty::Vec {
            elem: Box::new(Ty::I32),
            lanes: 2,
        };
        assert!(!i32x2.is_float());
        assert!(i32x2.is_signed_int());
    }

    // --- Body::verify ---

    fn l(index: u32) -> Local {
        Local { index }
    }

    fn buffer(elem: Ty, mutable: bool) -> Ty {
        Ty::Ref {
            mutable,
            pointee: Box::new(Ty::Slice(Box::new(elem))),
        }
    }

    fn element(param: u32, index: u32) -> Place {
        Place {
            local: l(param),
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(index))],
        }
    }

    fn constant(c: Constant) -> Rvalue {
        Rvalue::Use(Operand::Const(c))
    }

    fn copy(local: u32) -> Operand {
        Operand::Copy(Place::local(l(local)))
    }

    fn goto(index: u32) -> Terminator {
        Terminator::Goto {
            target: BlockId { index },
        }
    }

    fn block(statements: Vec<Statement>, terminator: Terminator) -> BasicBlock {
        BasicBlock {
            statements,
            terminator,
        }
    }

    /// `_0: ()` and the given locals (`_1..=param_count` are the params), over the given blocks.
    fn body_of(param_count: u32, local_tys: Vec<Ty>, blocks: Vec<BasicBlock>) -> Body {
        let locals = std::iter::once(Ty::Unit)
            .chain(local_tys)
            .map(|ty| LocalDecl { ty, mutable: true })
            .collect();
        Body::new("k", param_count, locals, blocks)
    }

    /// One block of `statements` then `Return`, over `_1: &mut [f32]` and the extra locals.
    fn straight_line(extra: Vec<Ty>, statements: Vec<Statement>) -> Body {
        let mut tys = vec![buffer(Ty::F32, true)];
        tys.extend(extra);
        body_of(1, tys, vec![block(statements, Terminator::Return)])
    }

    fn rejection(body: &Body) -> (Site, VerifyErrorKind) {
        let err = body.verify().expect_err("the body must be rejected");
        (err.site, err.kind)
    }

    fn at_statement(block: u32, index: usize) -> Site {
        Site::Statement {
            block: BlockId { index: block },
            index,
        }
    }

    fn at_terminator(block: u32) -> Site {
        Site::Terminator {
            block: BlockId { index: block },
        }
    }

    #[test]
    fn well_formed_bodies_verify() {
        // card 671: the other fixtures this test used to cover (vadd_loop_kernel, gemv_loop_kernel,
        // square_kernel, matmul_kernel, e4m3fn_encode/decode_kernel) moved to poot-test-util, whose own
        // kernel_fixtures tests now verify them - poot-kernel-ir cannot dev-depend on poot-test-util for
        // its own unit tests (poot-test-util regular-depends on poot-kernel-ir, so the lib-test unit
        // would compile two mismatched instances of Body).
        add_body().verify().unwrap();
        let body = fixtures::add_kernel();
        body.verify()
            .unwrap_or_else(|e| panic!("fixture {} rejected: {e}", body.name));
    }

    /// Card 530: a matrix-fragment op's `shape` must be `M16N16K16` - the only shape
    /// any target's codegen implements - or `verify()` rejects it before any codegen sees it.
    #[test]
    fn wmma_shape_other_than_m16n16k16_is_rejected() {
        let bad_shape = WmmaShape {
            m: 32,
            n: 16,
            k: 16,
        };
        let body = straight_line(
            vec![],
            vec![Statement::WmmaZero {
                dtype: WmmaDtype::F16,
                shape: bad_shape,
                dst: l(1),
            }],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 0));
        assert_eq!(
            kind,
            VerifyErrorKind::UnsupportedWmmaShape { shape: bad_shape }
        );
    }

    /// R468-005: a local outside `locals` used to index-panic inside codegen.
    #[test]
    fn undeclared_local_is_rejected() {
        let body = straight_line(
            vec![],
            vec![Statement::Assign(
                Place::local(l(9)),
                constant(Constant::F32(1.0)),
            )],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 0));
        assert_eq!(kind, VerifyErrorKind::UndeclaredLocal(l(9)));
        assert_eq!(
            body.verify().unwrap_err().to_string(),
            "invalid kernel body at bb0 statement 0: local _9 is not declared"
        );
    }

    #[test]
    fn read_before_any_assignment_is_rejected() {
        // _2 and _3 are declared temporaries; _3 = _2 reads _2 before anything assigned it.
        let body = straight_line(
            vec![Ty::F32, Ty::F32],
            vec![Statement::Assign(Place::local(l(3)), Rvalue::Use(copy(2)))],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 0));
        assert_eq!(kind, VerifyErrorKind::UseBeforeDefinition(l(2)));
    }

    /// Branch on a constant, define `_2` in the `then` arm (and in the `else` arm when `both_arms`), then read
    /// `_2` at the merge. Blocks: bb0 branch, bb1 then, bb2 else, bb3 merge.
    fn diamond_reading_at_merge(both_arms: bool) -> Body {
        let define = || {
            vec![Statement::Assign(
                Place::local(l(2)),
                constant(Constant::F32(1.0)),
            )]
        };
        body_of(
            1,
            vec![buffer(Ty::F32, true), Ty::F32, Ty::F32],
            vec![
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: Operand::Const(Constant::Bool(true)),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 1 },
                        },
                    },
                ),
                block(define(), goto(3)),
                block(if both_arms { define() } else { vec![] }, goto(3)),
                block(
                    vec![Statement::Assign(Place::local(l(3)), Rvalue::Use(copy(2)))],
                    Terminator::Return,
                ),
            ],
        )
    }

    #[test]
    fn a_definition_on_only_one_path_does_not_reach_the_merge() {
        let (site, kind) = rejection(&diamond_reading_at_merge(false));
        assert_eq!(site, at_statement(3, 0));
        assert_eq!(kind, VerifyErrorKind::UseBeforeDefinition(l(2)));
        diamond_reading_at_merge(true).verify().unwrap();
    }

    #[test]
    fn a_loop_carried_definition_verifies() {
        // bb0: _2 = 0.0 -> bb1; bb1: _2 = _2 + 1.0; switch back to bb1 or exit to bb2.
        let body = body_of(
            1,
            vec![buffer(Ty::F32, true), Ty::F32],
            vec![
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        constant(Constant::F32(0.0)),
                    )],
                    goto(1),
                ),
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Add, copy(2), Operand::Const(Constant::F32(1.0))),
                    )],
                    Terminator::SwitchInt {
                        discr: Operand::Const(Constant::Bool(false)),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 1 },
                        },
                    },
                ),
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    /// R468-005: an f32 stored into a usize local crashed llc (SPIR-V) and compiled silently (NVPTX).
    #[test]
    fn f32_stored_into_a_usize_local_is_rejected() {
        let body = straight_line(
            vec![Ty::Usize],
            vec![Statement::Assign(
                Place::local(l(2)),
                constant(Constant::F32(1.0)),
            )],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 0));
        assert_eq!(
            kind,
            VerifyErrorKind::TypeMismatch {
                what: "assigned value",
                expected: Ty::Usize,
                found: Ty::F32,
            }
        );
    }

    #[test]
    fn a_buffer_element_store_checks_the_element_type() {
        // (*_1)[_2] = _3 with _3: usize into an f32 buffer.
        let body = straight_line(
            vec![Ty::Usize, Ty::Usize],
            vec![
                Statement::Assign(Place::local(l(2)), constant(Constant::Usize(0))),
                Statement::Assign(Place::local(l(3)), constant(Constant::Usize(1))),
                Statement::Assign(element(1, 2), Rvalue::Use(copy(3))),
            ],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 2));
        assert_eq!(
            kind,
            VerifyErrorKind::TypeMismatch {
                what: "assigned value",
                expected: Ty::F32,
                found: Ty::Usize,
            }
        );
    }

    #[test]
    fn binary_operands_and_comparison_results_are_typed() {
        // _2: usize = _2 + 1.0 mixes types; _3: f32 = _4 < _4 yields bool.
        let mixed = straight_line(
            vec![Ty::Usize],
            vec![
                Statement::Assign(Place::local(l(2)), constant(Constant::Usize(0))),
                Statement::Assign(
                    Place::local(l(2)),
                    Rvalue::BinaryOp(BinOp::Add, copy(2), Operand::Const(Constant::F32(1.0))),
                ),
            ],
        );
        assert_eq!(
            rejection(&mixed),
            (
                at_statement(0, 1),
                VerifyErrorKind::TypeMismatch {
                    what: "binary operands",
                    expected: Ty::Usize,
                    found: Ty::F32,
                }
            )
        );
        let compare = straight_line(
            vec![Ty::F32],
            vec![
                Statement::Assign(Place::local(l(2)), constant(Constant::F32(0.0))),
                Statement::Assign(
                    Place::local(l(2)),
                    Rvalue::BinaryOp(BinOp::Lt, copy(2), copy(2)),
                ),
            ],
        );
        assert_eq!(
            rejection(&compare),
            (
                at_statement(0, 1),
                VerifyErrorKind::TypeMismatch {
                    what: "assigned value",
                    expected: Ty::F32,
                    found: Ty::Bool,
                }
            )
        );
    }

    #[test]
    fn a_store_through_a_shared_buffer_is_rejected() {
        let body = body_of(
            1,
            vec![buffer(Ty::F32, false), Ty::Usize],
            vec![block(
                vec![
                    Statement::Assign(Place::local(l(2)), constant(Constant::Usize(0))),
                    Statement::Assign(element(1, 2), constant(Constant::F32(1.0))),
                ],
                Terminator::Return,
            )],
        );
        assert_eq!(
            rejection(&body),
            (
                at_statement(0, 1),
                VerifyErrorKind::WriteThroughSharedBuffer { param: l(1) }
            )
        );
        // An atomic is a write too.
        let atomic = body_of(
            1,
            vec![buffer(Ty::U32, false), Ty::Usize, Ty::U32],
            vec![block(
                vec![
                    Statement::Assign(Place::local(l(2)), constant(Constant::Usize(0))),
                    Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::GlobalAtomic {
                            place: element(1, 2),
                            value: Operand::Const(Constant::U32(1)),
                            op: AtomicOp::Add,
                        },
                    ),
                ],
                Terminator::Return,
            )],
        );
        assert_eq!(
            rejection(&atomic),
            (
                at_statement(0, 1),
                VerifyErrorKind::WriteThroughSharedBuffer { param: l(1) }
            )
        );
    }

    #[test]
    fn branch_targets_and_workgroup_arrays_must_exist() {
        let goto_missing = body_of(0, vec![], vec![block(vec![], goto(7))]);
        assert_eq!(
            rejection(&goto_missing),
            (
                Site::Terminator {
                    block: BlockId { index: 0 }
                },
                VerifyErrorKind::UndefinedBlock(BlockId { index: 7 })
            )
        );
        let barrier_missing = body_of(
            0,
            vec![],
            vec![block(
                vec![],
                Terminator::Barrier {
                    target: BlockId { index: 1 },
                },
            )],
        );
        assert_eq!(
            rejection(&barrier_missing).1,
            VerifyErrorKind::UndefinedBlock(BlockId { index: 1 })
        );
        let barrier_ok = body_of(
            0,
            vec![],
            vec![
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 1 },
                    },
                ),
                block(vec![], Terminator::Return),
            ],
        );
        barrier_ok.verify().unwrap();

        let lds_missing = straight_line(
            vec![Ty::F32],
            vec![Statement::Assign(
                Place::local(l(2)),
                Rvalue::WorkgroupLocalRead {
                    idx: Operand::Const(Constant::Usize(0)),
                    array: 3,
                },
            )],
        );
        assert_eq!(
            rejection(&lds_missing),
            (
                at_statement(0, 0),
                VerifyErrorKind::UndeclaredWorkgroupArray(3)
            )
        );
    }

    #[test]
    fn workgroup_array_accesses_are_typed_by_the_declaration() {
        let mut body = straight_line(
            vec![],
            vec![Statement::WorkgroupLocalWrite {
                idx: Operand::Const(Constant::Usize(0)),
                value: Operand::Const(Constant::U32(1)),
                array: 0,
            }],
        );
        body.workgroup_locals.push(WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: 8,
        });
        assert_eq!(
            rejection(&body),
            (
                at_statement(0, 0),
                VerifyErrorKind::TypeMismatch {
                    what: "workgroup array store",
                    expected: Ty::F32,
                    found: Ty::U32,
                }
            )
        );
    }

    #[test]
    fn params_must_be_buffers_and_fit_in_the_locals() {
        let scalar_param = body_of(1, vec![Ty::F32], vec![block(vec![], Terminator::Return)]);
        assert_eq!(
            rejection(&scalar_param),
            (
                Site::Signature,
                VerifyErrorKind::ParamNotBuffer {
                    local: l(1),
                    ty: Ty::F32
                }
            )
        );
        let too_few_locals = body_of(3, vec![], vec![block(vec![], Terminator::Return)]);
        assert_eq!(
            rejection(&too_few_locals),
            (
                Site::Signature,
                VerifyErrorKind::ParamsExceedLocals {
                    param_count: 3,
                    locals: 1
                }
            )
        );
        let no_blocks = body_of(0, vec![], vec![]);
        assert_eq!(
            rejection(&no_blocks),
            (Site::Signature, VerifyErrorKind::NoBlocks)
        );
        let mut flat = body_of(0, vec![], vec![block(vec![], Terminator::Return)]);
        flat.workgroup_size = [64, 0, 1];
        assert_eq!(
            rejection(&flat),
            (
                Site::Signature,
                VerifyErrorKind::ZeroWorkgroupDim { size: [64, 0, 1] }
            )
        );
    }

    #[test]
    fn places_outside_the_lowered_shapes_are_rejected() {
        // `*_1` alone is not a place codegen lowers.
        let deref_only = Place {
            local: l(1),
            projection: vec![ProjectionElem::Deref],
        };
        let body = straight_line(
            vec![Ty::F32],
            vec![Statement::Assign(
                Place::local(l(2)),
                Rvalue::Use(Operand::Copy(deref_only.clone())),
            )],
        );
        assert_eq!(
            rejection(&body),
            (
                at_statement(0, 0),
                VerifyErrorKind::MalformedPlace {
                    place: deref_only,
                    reason: "only [], [Index] and [Deref, Index] are places",
                }
            )
        );
        // `_2[_3]` indexes a local that is not a private array.
        let scalar_indexed = Place {
            local: l(2),
            projection: vec![ProjectionElem::Index(l(3))],
        };
        let body = straight_line(
            vec![Ty::F32, Ty::Usize],
            vec![
                Statement::Assign(Place::local(l(3)), constant(Constant::Usize(0))),
                Statement::Assign(
                    Place::local(l(2)),
                    Rvalue::Use(Operand::Copy(scalar_indexed.clone())),
                ),
            ],
        );
        assert_eq!(
            rejection(&body),
            (
                at_statement(0, 1),
                VerifyErrorKind::MalformedPlace {
                    place: scalar_indexed,
                    reason: "[Index] needs a private array local",
                }
            )
        );
    }

    #[test]
    fn a_thread_index_lands_in_a_usize_local() {
        let body = body_of(
            0,
            vec![Ty::U32],
            vec![
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::X,
                        target: BlockId { index: 1 },
                    },
                ),
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(
            rejection(&body),
            (
                Site::Terminator {
                    block: BlockId { index: 0 }
                },
                VerifyErrorKind::TypeMismatch {
                    what: "thread index destination",
                    expected: Ty::Usize,
                    found: Ty::U32,
                }
            )
        );
    }

    // --- barrier uniformity (card 618) ---

    /// bb0: `lane = ThreadIndexCall(dim)` -> bb1; bb1: `cmp = (lane == 0)`, `SwitchInt(cmp)` with `0 ->
    /// bb3`, otherwise `-> bb2`; bb2/bb3 each barrier into their own dead-end block (bb4/bb5). Shared by
    /// the divergent- and uniform-condition tests below; only `dim` differs.
    fn lane_split_to_two_barriers(dim: IndexAxis) -> Body {
        body_of(
            0,
            vec![Ty::Usize, Ty::Bool],
            vec![
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim,
                        target: BlockId { index: 1 },
                    },
                ),
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 5 },
                    },
                ),
                block(vec![], Terminator::Return),
                block(vec![], Terminator::Return),
            ],
        )
    }

    /// SC-001: `LocalX` (a per-lane id) feeds the branch, so lane 0 would park at the barrier in bb2 and
    /// every other lane at the barrier in bb3 - real lockstep hardware cannot honor both, the static
    /// counterpart to `interp`'s `lanes_parked_at_different_barriers_diverge`.
    #[test]
    fn thread_varying_branch_to_different_barriers_is_rejected() {
        let body = lane_split_to_two_barriers(IndexAxis::LocalX);
        assert_eq!(
            rejection(&body),
            (
                at_terminator(1),
                VerifyErrorKind::DivergentBarrierReachability {
                    branch: BlockId { index: 1 },
                    barrier: BlockId { index: 2 },
                    reconverge: None,
                }
            )
        );
    }

    /// SC-002: same shape as `thread_varying_branch_to_different_barriers_is_rejected`, but the branch
    /// reads `GroupX` (the workgroup index - Scope's "tile index every lane computes identically"):
    /// every lane of one workgroup agrees on which arm it takes, so there is no lockstep hazard.
    #[test]
    fn thread_uniform_branch_to_different_barriers_passes() {
        lane_split_to_two_barriers(IndexAxis::GroupX)
            .verify()
            .unwrap();
    }

    /// Mirrors kernelgen's LDS-tree-reduction guard (`emit_lds_tree_blocks`): the discriminant is still
    /// thread-varying (`LocalX`-derived), but the "skip" arm goes straight to the barrier block and the
    /// "combine" arm does extra work first, then `Goto`s into that very same block - both arms reach the
    /// identical `Terminator::Barrier`, so every lane still syncs together (Scope: "arms rejoin before
    /// any barrier").
    #[test]
    fn thread_varying_branch_that_rejoins_before_the_barrier_passes() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Bool],
            vec![
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(1), Operand::Const(Constant::Usize(4))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // combine (lane < 4): extra work, then into the same barrier block as the skip arm.
                block(vec![], goto(3)),
                // merge: one barrier, reached by every lane no matter which arm it took.
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    /// SC-004: `n` is read per-lane from a buffer, so lanes can disagree on the loop's trip count and
    /// reach the barrier in bb3 a different number of times - unlike a compile-time-bound loop whose
    /// barrier sits after the loop, not inside it (e.g. kernelgen's `j < K` accumulate loops).
    #[test]
    fn barrier_inside_a_loop_with_a_thread_varying_trip_count_is_rejected() {
        let body = body_of(
            1,
            vec![buffer(Ty::Usize, false), Ty::Usize, Ty::Usize, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(2)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: i = 0 -> bb2
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        constant(Constant::Usize(0)),
                    )],
                    goto(2),
                ),
                // bb2 (header): cond = i < n[lane]; false -> exit (bb4), true -> body (bb3)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(3), Operand::Copy(element(1, 2))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 4 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb3 (body): i += 1; barrier back to the header, inside the loop.
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Add, copy(3), Operand::Const(Constant::Usize(1))),
                    )],
                    Terminator::Barrier {
                        target: BlockId { index: 2 },
                    },
                ),
                // bb4: exit.
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(
            rejection(&body),
            (
                at_terminator(2),
                VerifyErrorKind::DivergentBarrierReachability {
                    branch: BlockId { index: 2 },
                    barrier: BlockId { index: 3 },
                    reconverge: Some(BlockId { index: 4 }),
                }
            )
        );
    }

    // --- card 618 review regression probes ---

    fn branch_of(kind: &VerifyErrorKind) -> BlockId {
        let VerifyErrorKind::DivergentBarrierReachability { branch, .. } = kind else {
            panic!("expected DivergentBarrierReachability, got {kind:?}");
        };
        *branch
    }

    /// Probe a: `lane == 0` (thread-varying) is nested inside a uniform `i < 10` loop;
    /// one arm barriers, the other skips straight to the rejoin block that feeds the loop back edge. The
    /// old may-reach check found the barrier arm's own block in the skip arm's reach set (reachable via
    /// the loop's back edge next iteration) and wrongly accepted this; a loop back-edge must never
    /// manufacture a rendezvous the arms are not actually guaranteed to share.
    #[test]
    fn barrier_under_a_varying_branch_nested_in_a_uniform_loop_is_rejected() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize, Ty::Bool, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: i = 0 -> bb2
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        constant(Constant::Usize(0)),
                    )],
                    goto(2),
                ),
                // bb2 (loop header, uniform): cond_loop = i < 10; false -> exit (bb7), true -> body (bb3)
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(2), Operand::Const(Constant::Usize(10))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(3),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 7 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb3 (varying): cond_lane = lane == 0; false -> skip (bb5), true -> barrier arm (bb4)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 5 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb4: barrier -> rejoin (bb6)
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 6 },
                    },
                ),
                // bb5: skip -> rejoin (bb6)
                block(vec![], goto(6)),
                // bb6 (rejoin): i += 1 -> loop header (bb2)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Add, copy(2), Operand::Const(Constant::Usize(1))),
                    )],
                    goto(2),
                ),
                // bb7: exit
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 3 });
    }

    /// Probe b: a `LocalX` branch splits
    /// into an arm gated by a uniform `GroupX` value between two *different* barriers, and a second arm
    /// that joins the first of those two barrier blocks directly. Neither the immediate post-dominator
    /// (there is none short of the body's exit) nor a pairwise-only comparison would catch every arm.
    #[test]
    fn varying_branch_whose_uniform_gated_arm_reaches_two_different_barriers_is_rejected() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize, Ty::Bool, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: g = GroupX -> bb2
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(2)),
                        dim: IndexAxis::GroupX,
                        target: BlockId { index: 2 },
                    },
                ),
                // bb2 (varying): cond_lane = lane == 0; false -> armB (bb5), true -> armA (bb3)
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(3),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 5 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb3 (armA hdr, uniform): cond_g = g == 0; false -> barrierY (bb6), true -> barrierX (bb4)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(2), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 6 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb4 (barrierX)
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 7 },
                    },
                ),
                // bb5 (armB): joins barrierX (bb4) directly
                block(vec![], goto(4)),
                // bb6 (barrierY)
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 7 },
                    },
                ),
                // bb7: return
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 2 });
    }

    /// without the old exit-collapse refinement, this body is already rejected by the
    /// must-reach check (dropping that refinement, not re-adding it under must-reach, is what the fix
    /// requires). A `LocalX` branch splits into an arm that always returns (armB) and an arm gated by a
    /// uniform value that *either* returns or barriers (armA): every lane taking armB never barriers, but
    /// a lane taking armA might, so the two arms are not provably safe together.
    #[test]
    fn varying_branch_where_one_arm_may_barrier_behind_a_uniform_gate_and_the_other_never_does_is_rejected()
     {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize, Ty::Bool, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: g = GroupX -> bb2
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(2)),
                        dim: IndexAxis::GroupX,
                        target: BlockId { index: 2 },
                    },
                ),
                // bb2 (varying): cond_lane = lane == 0; false -> armB (bb5, always returns), true -> armA (bb3)
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(3),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 5 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb3 (armA hdr, uniform): cond_g = g == 0; false -> return (bb6), true -> barrier (bb4)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(2), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 6 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb4: barrier -> bb7
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 7 },
                    },
                ),
                // bb5 (armB): return, no barrier ever
                block(vec![], Terminator::Return),
                // bb6 (armA's own return sub-path)
                block(vec![], Terminator::Return),
                // bb7
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 2 });
    }

    /// lanes split on `lane == 0` (thread-varying), rejoin, one arm stores `buf[0] =
    /// 42`, both then read `buf[0]` at the same (uniform, constant) index and branch on it. Without
    /// this precondition, the scalar-load refinement calls that read uniform (the index is
    /// uniform) and this reconverged branch passes; the store is reachable from the entry to the read
    /// without an intervening barrier, so the read is actually racy - some lanes can see the write, others
    /// the pre-write value - and the branch must be treated as thread-varying.
    #[test]
    fn a_uniform_index_load_racing_an_unsynchronized_store_is_treated_as_varying() {
        let body = body_of(
            1,
            vec![
                buffer(Ty::U32, true),
                Ty::Usize,
                Ty::Usize,
                Ty::Bool,
                Ty::U32,
                Ty::Bool,
            ],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(2)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: zero = 0; store_cond = lane == 0; false -> skip (bb3), true -> store (bb2)
                block(
                    vec![
                        Statement::Assign(Place::local(l(3)), constant(Constant::Usize(0))),
                        Statement::Assign(
                            Place::local(l(4)),
                            Rvalue::BinaryOp(
                                BinOp::Eq,
                                copy(2),
                                Operand::Const(Constant::Usize(0)),
                            ),
                        ),
                    ],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // bb2 (store arm): buf[zero] = 42 -> rejoin (bb4)
                block(
                    vec![Statement::Assign(
                        element(1, 3),
                        Rvalue::Use(Operand::Const(Constant::U32(42))),
                    )],
                    goto(4),
                ),
                // bb3 (skip arm): -> rejoin (bb4)
                block(vec![], goto(4)),
                // bb4 (rejoin): v = buf[zero]; cond = v == 42; false -> barrier (bb5), true -> return (bb6)
                block(
                    vec![
                        Statement::Assign(
                            Place::local(l(5)),
                            Rvalue::Use(Operand::Copy(element(1, 3))),
                        ),
                        Statement::Assign(
                            Place::local(l(6)),
                            Rvalue::BinaryOp(BinOp::Eq, copy(5), Operand::Const(Constant::U32(42))),
                        ),
                    ],
                    Terminator::SwitchInt {
                        discr: copy(6),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 6 })],
                            otherwise: BlockId { index: 5 },
                        },
                    },
                ),
                // bb5: barrier -> bb7
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 7 },
                    },
                ),
                // bb6: return
                block(vec![], Terminator::Return),
                // bb7: return
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 4 });
    }

    /// a matrix-fragment op's `dst` is declared as a plain `Bool` local (nothing stops
    /// this - `declared_fragment` accepts any declared local) and used directly as a `SwitchInt`
    /// discriminant feeding two different barriers. Without marking a fragment `dst` varying in
    /// `stmt_uniform`, it keeps its all-true entry uniformity forever (never reassigned) and this passes;
    /// an opaque fragment value must never be trusted as uniform just because nothing reassigned it.
    #[test]
    fn a_scalar_typed_wmma_dst_read_by_a_switch_is_treated_as_varying() {
        let body = body_of(
            0,
            vec![Ty::Bool],
            vec![
                block(
                    vec![Statement::WmmaZero {
                        dtype: WmmaDtype::F16,
                        shape: WmmaShape::M16N16K16,
                        dst: l(1),
                    }],
                    Terminator::SwitchInt {
                        discr: copy(1),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 1 },
                        },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 3 },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                block(vec![], Terminator::Return),
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 0 });
    }

    /// Positive counterpart to the loop probe above: a barrier inside a loop whose trip count is
    /// thread-uniform (a compile-time-like bound, like kernelgen's top-k/top-p bisection loops) is not
    /// even a candidate for this check - the loop header's own condition is uniform, so every lane agrees
    /// on how many times to go around and hits the same barrier the same number of times.
    #[test]
    fn a_barrier_inside_a_thread_uniform_loop_passes() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Bool],
            vec![
                // bb0: i = 0 -> bb1
                block(
                    vec![Statement::Assign(
                        Place::local(l(1)),
                        constant(Constant::Usize(0)),
                    )],
                    goto(1),
                ),
                // bb1 (header, uniform): cond = i < 10; false -> exit (bb4), true -> body (bb2)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(1), Operand::Const(Constant::Usize(10))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 4 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // bb2: barrier -> bb3
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 3 },
                    },
                ),
                // bb3: i += 1 -> loop header
                block(
                    vec![Statement::Assign(
                        Place::local(l(1)),
                        Rvalue::BinaryOp(BinOp::Add, copy(1), Operand::Const(Constant::Usize(1))),
                    )],
                    goto(1),
                ),
                // bb4: exit
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    /// Positive counterpart to the multi-barrier probes above: a `LocalX` branch's two arms reconverge at
    /// a plain (non-barrier) block, which then runs more code before eventually reaching a barrier. The
    /// barrier sits strictly *after* the reconvergence point, so every lane is already back at the same
    /// program point long before it, and is unaffected by it.
    #[test]
    fn a_reconverged_varying_branch_with_the_barrier_after_the_post_dominator_passes() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Bool, Ty::Usize],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1 (varying): cond = lane == 0; false -> armB (bb3), true -> armA (bb2)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // bb2 (armA): -> rejoin (bb4)
                block(vec![], goto(4)),
                // bb3 (armB): -> rejoin (bb4)
                block(vec![], goto(4)),
                // bb4 (reconvergence, plain block): x = 1 -> bb5
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        constant(Constant::Usize(1)),
                    )],
                    goto(5),
                ),
                // bb5: barrier, strictly after the reconvergence point -> bb6
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 6 },
                    },
                ),
                // bb6: return
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    // --- card 618 re-review (round 2) adversarial probes ---

    /// Round-2 probe 1: a locally-safe inner varying branch (its own two sub-arms rejoin cleanly at
    /// bb5, no barrier) sits inside one arm (bb2) of an outer varying branch (bb1); a barrier (bb6)
    /// follows the inner join but precedes the outer join (bb8), and the outer branch's other arm
    /// (bb7) skips straight to bb8, bypassing the barrier entirely. The outer branch's own
    /// post-dominator (bb8, not the inner branch's bb5) must catch this, undisturbed by the inner
    /// branch's independent, passing check.
    #[test]
    fn nested_varying_branches_with_a_barrier_between_the_inner_and_outer_reconvergence_is_rejected()
     {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Bool, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1 (outer varying): cond = lane == 0; false -> armB (bb7), true -> armA (bb2)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 7 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // bb2 (armA, inner varying, locally safe): cond2 = lane == 0; false -> bb4, true -> bb3
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(3),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 4 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb3 (inner true arm): -> inner join (bb5)
                block(vec![], goto(5)),
                // bb4 (inner false arm): -> inner join (bb5)
                block(vec![], goto(5)),
                // bb5 (inner join, no barrier yet): -> bb6
                block(vec![], goto(6)),
                // bb6: barrier, after the inner join but before the outer join -> bb8
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 8 },
                    },
                ),
                // bb7 (armB): skips the barrier entirely -> outer join (bb8)
                block(vec![], goto(8)),
                // bb8 (outer join): return
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 1 });
    }

    /// Round-2 probe 2: one arm returns immediately, the other barriers then returns - no shared real
    /// block before the body's exit, so the branch's immediate post-dominator is the virtual exit
    /// itself (`reconverge: None`), exercising `barrier_before`'s unbounded search path.
    #[test]
    fn varying_branch_whose_post_dominator_is_the_virtual_exit_is_rejected() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1 (varying): cond = lane == 0; false -> bb2 (return), true -> bb3 (barrier)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Eq, copy(1), Operand::Const(Constant::Usize(0))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb2: immediate return, no barrier
                block(vec![], Terminator::Return),
                // bb3: barrier -> bb4
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                // bb4: return
                block(vec![], Terminator::Return),
            ],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_terminator(1));
        assert!(
            matches!(
                &kind,
                VerifyErrorKind::DivergentBarrierReachability {
                    branch,
                    reconverge: None,
                    ..
                } if *branch == BlockId { index: 1 }
            ),
            "{kind:?}"
        );
    }

    /// Round-2 probe 3: a do-while loop - the body (and its barrier) always runs at least once, and
    /// the exit check `i < lane` (thread-varying, unlike the usual entry-gated loop) sits *after* the
    /// barrier, not before it. The post-dominator computation must still find the block after the loop
    /// as the exit check's immediate post-dominator despite the back edge, the same must-reach
    /// reasoning as the entry-gated loop probe above, applied to the opposite loop shape.
    #[test]
    fn a_do_while_loop_with_a_thread_varying_exit_check_is_rejected() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize, Ty::Usize, Ty::Bool],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: i = 0 -> bb2
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        constant(Constant::Usize(0)),
                    )],
                    goto(2),
                ),
                // bb2 (loop body): barrier, unconditionally every iteration -> bb3
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 3 },
                    },
                ),
                // bb3 (exit check, varying): cond = i < lane; false -> exit (bb5), true -> continue (bb4)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(2), copy(1)),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 5 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb4: i += 1 -> loop body (back edge)
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Add, copy(2), Operand::Const(Constant::Usize(1))),
                    )],
                    goto(2),
                ),
                // bb5 (after the loop): return
                block(vec![], Terminator::Return),
            ],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_terminator(3));
        assert!(
            matches!(
                &kind,
                VerifyErrorKind::DivergentBarrierReachability {
                    branch,
                    barrier,
                    reconverge: Some(reconverge),
                } if *branch == BlockId { index: 3 }
                    && *barrier == BlockId { index: 2 }
                    && *reconverge == BlockId { index: 5 }
            ),
            "{kind:?}"
        );
    }

    /// Round-2 probe 4: `buf[lane] = 7` is an *unconditional*, thread-varying-*indexed* store (always
    /// executed - the store itself is not gated by any branch, only its target index varies); a later
    /// uniform-indexed read `buf[0]` in the same straight-line block, with no intervening barrier,
    /// feeds a branch to two different barriers. `racy` is a per-buffer flag, not per-index, so the
    /// varying-indexed write still taints the uniform-indexed read of the same buffer.
    #[test]
    fn a_varying_indexed_store_taints_a_later_uniform_indexed_read_of_the_same_buffer() {
        let body = body_of(
            1,
            vec![
                buffer(Ty::U32, true),
                Ty::Usize,
                Ty::Usize,
                Ty::U32,
                Ty::Bool,
            ],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(2)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: buf[lane] = 7 (unconditional); zero = 0; v = buf[zero]; cond = v == 7
                block(
                    vec![
                        Statement::Assign(
                            element(1, 2),
                            Rvalue::Use(Operand::Const(Constant::U32(7))),
                        ),
                        Statement::Assign(Place::local(l(3)), constant(Constant::Usize(0))),
                        Statement::Assign(
                            Place::local(l(4)),
                            Rvalue::Use(Operand::Copy(element(1, 3))),
                        ),
                        Statement::Assign(
                            Place::local(l(5)),
                            Rvalue::BinaryOp(BinOp::Eq, copy(4), Operand::Const(Constant::U32(7))),
                        ),
                    ],
                    Terminator::SwitchInt {
                        discr: copy(5),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                // bb2: barrier -> bb4
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                // bb3: barrier -> bb5
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 5 },
                    },
                ),
                // bb4: return
                block(vec![], Terminator::Return),
                // bb5: return
                block(vec![], Terminator::Return),
            ],
        );
        assert_eq!(branch_of(&rejection(&body).1), BlockId { index: 1 });
    }

    /// Round-2 probe 5: the same uniform-loop-with-a-barrier shape as
    /// `a_barrier_inside_a_thread_uniform_loop_passes`, but the trip bound is derived from `GroupX`
    /// through an arithmetic statement (`bound = g + 4`) instead of being read directly - confirms
    /// uniformity propagates through `Rvalue::BinaryOp`, not just a bare copy of a `ThreadIndexCall`
    /// destination.
    #[test]
    fn a_barrier_inside_a_loop_bound_by_an_arithmetic_expression_of_groupx_passes() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize, Ty::Usize, Ty::Bool],
            vec![
                // bb0: g = GroupX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::GroupX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1: bound = g + 4 -> bb2
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Add, copy(1), Operand::Const(Constant::Usize(4))),
                    )],
                    goto(2),
                ),
                // bb2: i = 0 -> bb3
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        constant(Constant::Usize(0)),
                    )],
                    goto(3),
                ),
                // bb3 (header, uniform): cond = i < bound; false -> exit (bb6), true -> body (bb4)
                block(
                    vec![Statement::Assign(
                        Place::local(l(4)),
                        Rvalue::BinaryOp(BinOp::Lt, copy(3), copy(2)),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(4),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 6 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb4: barrier -> bb5
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 5 },
                    },
                ),
                // bb5: i += 1 -> loop header
                block(
                    vec![Statement::Assign(
                        Place::local(l(3)),
                        Rvalue::BinaryOp(BinOp::Add, copy(3), Operand::Const(Constant::Usize(1))),
                    )],
                    goto(3),
                ),
                // bb6: exit
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    /// Round-2 probe 6: a three-way `SwitchInt` (two explicit branch values plus `otherwise`) whose
    /// every arm rejoins at one block before the barrier - confirms the all-arm fix (round-1 finding
    /// 2) generalizes past a two-arm diamond to a genuine multi-way branch.
    #[test]
    fn a_three_way_switch_that_fully_rejoins_before_the_barrier_passes() {
        let body = body_of(
            0,
            vec![Ty::Usize, Ty::Usize],
            vec![
                // bb0: lane = LocalX -> bb1
                block(
                    vec![],
                    Terminator::ThreadIndexCall {
                        destination: Place::local(l(1)),
                        dim: IndexAxis::LocalX,
                        target: BlockId { index: 1 },
                    },
                ),
                // bb1 (3-way varying): val = lane % 3; 0 -> bb2, 1 -> bb3, otherwise -> bb4
                block(
                    vec![Statement::Assign(
                        Place::local(l(2)),
                        Rvalue::BinaryOp(BinOp::Rem, copy(1), Operand::Const(Constant::Usize(3))),
                    )],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 }), (1, BlockId { index: 3 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                // bb2: -> join (bb5)
                block(vec![], goto(5)),
                // bb3: -> join (bb5)
                block(vec![], goto(5)),
                // bb4: -> join (bb5)
                block(vec![], goto(5)),
                // bb5 (join): barrier -> bb6
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 6 },
                    },
                ),
                // bb6: return
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    /// Round-2 probe 7 (structural): a block unreachable from the entry block, itself containing a
    /// divergent-looking `SwitchInt` to two different barriers. `post_dominators` computes over every
    /// block that can reach the virtual exit regardless of forward reachability from the entry (no
    /// indexing panic), and `verify_barrier_uniformity`'s `state_in[b] == None` skip (the same pattern
    /// `verify_definitions` already uses) excludes a block no lane ever executes from the check, so
    /// `verify()` neither panics nor flags it.
    #[test]
    fn an_unreachable_block_with_a_divergent_switch_is_neither_flagged_nor_a_panic() {
        let body = body_of(
            0,
            vec![Ty::Bool],
            vec![
                // bb0 (the only block reachable from the entry): return.
                block(vec![], Terminator::Return),
                // bb1 (unreachable): a switch on a declared-but-never-assigned local, to two barriers.
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: copy(1),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                // bb2: barrier -> bb4
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                // bb3: barrier -> bb5
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 5 },
                    },
                ),
                // bb4: return
                block(vec![], Terminator::Return),
                // bb5: return
                block(vec![], Terminator::Return),
            ],
        );
        body.verify().unwrap();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trip() {
        let b = add_body();
        let json = serde_json::to_string(&b).unwrap();
        let back: Body = serde_json::from_str(&json).unwrap();
        assert_eq!(b, back);
    }

    // --- card 628: BinaryOpNoContract ------------------------------------------------------------

    // card 671: no_contract_probe_kernel moved to poot-test-util (see well_formed_bodies_verify's
    // comment); poot_test_util::kernel_fixtures's own tests cover
    // no_contract_probe_kernel(true/false).verify().

    /// `Body::verify` must reject a `BinaryOpNoContract` whose `op` is not `Add`/`Sub`/`Mul` (`Div` here),
    /// so a future producer cannot mark an op no backend's `emit_binop_no_contract` lowering handles.
    /// Mutation: this row is a positive test of that rejection, not itself mutated - dropping the check in
    /// `rvalue_ty`'s `BinaryOpNoContract` arm turns it green on the wrong body (accepted instead of
    /// rejected), which is exactly what this test exists to catch.
    #[test]
    fn binary_op_no_contract_rejects_a_non_add_sub_mul_op() {
        let body = body_of(
            0,
            vec![Ty::F32, Ty::F32],
            vec![block(
                vec![Statement::Assign(
                    Place::local(l(0)),
                    Rvalue::BinaryOpNoContract(BinOp::Div, copy(1), copy(2)),
                )],
                Terminator::Return,
            )],
        );
        assert_eq!(
            rejection(&body),
            (
                at_statement(0, 0),
                VerifyErrorKind::UnsupportedNoContractOp { op: BinOp::Div }
            )
        );
    }

    /// `Body::verify` rejects a `BinaryOpNoContract` on a non-float (integer) operand: the marker exists to
    /// keep a float multiply-add/-sub decode formula unfused, and every backend's `emit_binop_no_contract`
    /// lowering (a constrained float intrinsic, an `fadd`/`fsub`/`fmul` mnemonic) assumes a float operand.
    #[test]
    fn binary_op_no_contract_rejects_a_non_float_operand() {
        let body = body_of(
            0,
            vec![Ty::I32, Ty::I32],
            vec![block(
                vec![Statement::Assign(
                    Place::local(l(0)),
                    Rvalue::BinaryOpNoContract(BinOp::Add, copy(1), copy(2)),
                )],
                Terminator::Return,
            )],
        );
        let (site, kind) = rejection(&body);
        assert_eq!(site, at_statement(0, 0));
        assert!(matches!(
            kind,
            VerifyErrorKind::InvalidOperandType {
                what: "no-contract binary operand",
                ..
            }
        ));
    }
}
