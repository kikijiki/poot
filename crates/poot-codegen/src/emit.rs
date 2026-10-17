//! Body -> textual LLVM IR, for the SPIR-V (Vulkan resource shape) and NVPTX (raw-pointer shape)
//! targets, plus AMDGPU and AIE core (the `AieCore` branches are live: `crate::compile_aie` compiles every
//! tile body of an XDNA2 region through them, POOT-641). One target-branched walk: the block/statement/operand/binop logic is
//! shared; the leaves that differ (usize width, buffer access, thread-id intrinsics, the length source, the
//! signature, the footer) branch on `Target`. Alloca-form + plain branches throughout (llc's mem2reg/SROA does
//! SSA construction and phi insertion); only SPIR-V resource handles stay SSA-direct in the entry block,
//! because Vulkan's Logical addressing cannot store opaque handle types.

use std::collections::BTreeSet;

use poot_kernel_ir::{
    AtomicOp, BinOp, BlockId, Body, Constant, Fp8Format, IndexAxis, Local, MemoryOrdering,
    MemoryScope, Operand, Place, ProjectionElem, Rvalue, Statement, Terminator, Ty,
};

use crate::{EmitError, Target};

/// Body -> target-shaped textual LLVM IR. A `Rvalue::BinaryOpNoContract` is marked in the IR itself, per
/// target (see `Emitter::emit_binop_no_contract`), so the text alone carries every marker `compile_with`'s
/// post-processing needs.
pub fn emit_llvm_ir(body: &Body, target: Target) -> Result<String, EmitError> {
    // Every stage below indexes locals, blocks and workgroup arrays directly; a malformed body must be rejected
    // as a typed error first, not panic or reach llc.
    body.verify()?;
    let mut structured = body.clone();
    // SpirvVulkan only, and in this order: first remove every Assert/Unreachable-derived Trap branch
    // (card 531c) - Vulkan compute has no trap instruction, so a failed assert accumulates a fault bit and
    // keeps going instead of branching to a dedicated trap block (see `debranch` for why). This adds no
    // control-flow edge, so it must run before structurize; running after would leave structurize looking
    // at a CFG shape (an ordinary selection nested under another) it already handles correctly, but
    // running it first means structurize never has to reason about trap arms at all. Then structurize
    // control flow (card 100 / spec 058) so `llc`'s SPIR-V backend stops mis-placing merges (the
    // nested-selection RADV landmine). NVPTX and AIE-core tolerate arbitrary CFG and keep a real device
    // trap, so neither pass runs for them.
    let mut error_accum = None;
    if target == Target::SpirvVulkan {
        error_accum = crate::debranch::debranch_traps_for_spirv(&mut structured);
        crate::structurize::structurize(&mut structured).map_err(EmitError::Structurize)?;
    }
    let mut e = Emitter::new(&structured, target, error_accum);
    e.run()?;
    Ok(e.finish())
}

struct Emitter<'a> {
    body: &'a Body,
    target: Target,
    out: String,
    tmp: usize,
    /// declare lines to emit in the footer (deduplicated, sorted).
    declares: BTreeSet<String>,
    /// SPIR-V: whether the length buffer handle/declares are needed.
    uses_len: bool,
    /// SpirvVulkan only: the per-thread fault-accumulator local `debranch::debranch_traps_for_spirv` added
    /// (card 531c), when the body had a `Trap`. `emit_entry` binds the reserved error-word resource
    /// exactly when this is `Some`; `emit_terminator`'s `Terminator::Return` arm flushes the accumulator's
    /// current value into it there. `None` on every other target, and on SpirvVulkan when the body has no
    /// `Assert`/`Unreachable` at all.
    error_accum: Option<Local>,
    /// Nvptx/AmdGcn only (card 628): whether any `Rvalue::BinaryOpNoContract` was emitted, so
    /// `emit_signature`/`emit_footer` add the `strictfp` function attribute the constrained intrinsics
    /// (`llvm.experimental.constrained.{fadd,fsub,fmul}`) require.
    strictfp_needed: bool,
}
mod core;
mod memory;
mod targets;
mod value;
mod wmma;

enum ResultTy {
    Same,
    Bool,
}

/// The attribute-group index for `strictfp` on Nvptx/AmdGcn (card 628): neither target uses any attribute
/// group today (unlike SpirvVulkan's `#0`/`#1`, reserved for `hlsl.numthreads`/barrier convergence), so `#0`
/// is free on both. `emit_signature` references it on the `define` (and `emit_binop_no_contract` on each
/// constrained-intrinsic call) exactly when `strictfp_needed`; `emit_footer` declares it the same way.
const NO_CONTRACT_STRICTFP_ATTR: u32 = 0;

/// The type suffix LLVM mangles into an overloaded intrinsic's name (`llvm.umin.<suffix>`) for a scalar operand of
/// LLVM type `llty`.
fn intrinsic_overload(llty: &str) -> Result<&'static str, EmitError> {
    Ok(match llty {
        "i32" => "i32",
        "i64" => "i64",
        "half" => "f16",
        "bfloat" => "bf16",
        "float" => "f32",
        "double" => "f64",
        other => {
            return Err(EmitError::Unsupported(format!(
                "no overloaded-intrinsic suffix for the LLVM type `{other}`"
            )));
        }
    })
}

/// The LLVM instruction mnemonic + result-type class for a binop on a given operand type.
fn binop_inst(op: BinOp, is_float: bool, signed: bool) -> (&'static str, ResultTy) {
    use BinOp::*;
    match op {
        Add => (if is_float { "fadd" } else { "add" }, ResultTy::Same),
        Sub => (if is_float { "fsub" } else { "sub" }, ResultTy::Same),
        Mul => (if is_float { "fmul" } else { "mul" }, ResultTy::Same),
        Div => (
            if is_float {
                "fdiv"
            } else if signed {
                "sdiv"
            } else {
                "udiv"
            },
            ResultTy::Same,
        ),
        Rem => (
            if is_float {
                "frem"
            } else if signed {
                "srem"
            } else {
                "urem"
            },
            ResultTy::Same,
        ),
        BitAnd => ("and", ResultTy::Same),
        BitOr => ("or", ResultTy::Same),
        BitXor => ("xor", ResultTy::Same),
        Shl => ("shl", ResultTy::Same),
        Shr => (if signed { "ashr" } else { "lshr" }, ResultTy::Same),
        Lt => (
            if is_float {
                "fcmp olt"
            } else if signed {
                "icmp slt"
            } else {
                "icmp ult"
            },
            ResultTy::Bool,
        ),
        Le => (
            if is_float {
                "fcmp ole"
            } else if signed {
                "icmp sle"
            } else {
                "icmp ule"
            },
            ResultTy::Bool,
        ),
        Gt => (
            if is_float {
                "fcmp ogt"
            } else if signed {
                "icmp sgt"
            } else {
                "icmp ugt"
            },
            ResultTy::Bool,
        ),
        Ge => (
            if is_float {
                "fcmp oge"
            } else if signed {
                "icmp sge"
            } else {
                "icmp uge"
            },
            ResultTy::Bool,
        ),
        Eq => (
            if is_float { "fcmp oeq" } else { "icmp eq" },
            ResultTy::Bool,
        ),
        Ne => (
            if is_float { "fcmp one" } else { "icmp ne" },
            ResultTy::Bool,
        ),
        Min => ("__min", ResultTy::Same), // unsupported until needed (would be llvm.minnum / select)
        Max => ("__max", ResultTy::Same),
    }
}

/// Map an `IndexAxis` to the index (0/1/2) into `self.body.workgroup_size` and the AMDGPU intrinsic axis
/// suffix. X/LocalX/GroupX -> 0 etc. share a bucket, since the workgroup size is the same for the
/// per-wavefront, workgroup, or global coordinate on that axis.
fn axis_index(dim: IndexAxis) -> usize {
    use IndexAxis::*;
    match dim {
        X | LocalX | GroupX => 0,
        Y | LocalY | GroupY => 1,
        Z | LocalZ | GroupZ => 2,
    }
}

/// Positive finite E4M3FN value used only while generating the software encoder's exact midpoint table.
fn positive_e4m3fn_value(bits: u8) -> f32 {
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0 {
        (mantissa as f32) * (1.0 / 512.0)
    } else {
        let f32_exponent = (exponent as u32 + 120) << 23;
        let f32_mantissa = (mantissa as u32) << 20;
        f32::from_bits(f32_exponent | f32_mantissa)
    }
}

/// Format a constant operand, returning (text, llvm type). Floats use the LLVM hex-double form (exact).
fn fmt_const(c: &Constant, usize_llty: &str) -> (String, String) {
    match c {
        Constant::Bool(b) => ((if *b { "true" } else { "false" }).to_string(), "i1".into()),
        Constant::Usize(n) => (n.to_string(), usize_llty.to_string()),
        Constant::I32(n) => (n.to_string(), "i32".into()),
        Constant::U32(n) => (n.to_string(), "i32".into()),
        Constant::F32(f) => (format!("0x{:016X}", (*f as f64).to_bits()), "float".into()),
        Constant::F64(f) => (format!("0x{:016X}", f.to_bits()), "double".into()),
    }
}

fn const_ty(c: &Constant) -> Ty {
    match c {
        Constant::Bool(_) => Ty::Bool,
        Constant::Usize(_) => Ty::Usize,
        Constant::I32(_) => Ty::I32,
        Constant::U32(_) => Ty::U32,
        Constant::F32(_) => Ty::F32,
        Constant::F64(_) => Ty::F64,
    }
}

fn int_bits(llty: &str) -> u32 {
    match llty {
        "i1" => 1,
        "i64" => 64,
        _ => 32,
    }
}

/// Bit width of an LLVM float scalar type, for picking fpext vs fptrunc between float types.
fn float_bits(llty: &str) -> u32 {
    match llty {
        "half" | "bfloat" => 16,
        "double" => 64,
        _ => 32, // "float"
    }
}

fn align_of(llty: &str) -> u32 {
    // Vector type `<N x T>`: natural alignment is `N * align_of(T)` (matches clang's default vector alignment and
    // the A1 probe fixture's `align 16` for `<4 x float>`). FR-003.
    if let Some(rest) = llty.strip_prefix('<')
        && let Some((n_str, t)) = rest.trim_end_matches('>').split_once(" x ")
        && let Ok(n) = n_str.trim().parse::<u32>()
    {
        return n * align_of(t.trim());
    }
    match llty {
        "double" => 8,
        // 16-bit floats are 2-byte aligned. `bfloat` must be listed here: falling to `_ => 4` gives bf16 scalar
        // load/store `align 4` on 2-byte data (UB; the AMDGPU backend miscompiled it to 0, card 045). The WMMA path
        // reads bf16 only via the vector intrinsic and is unaffected.
        "half" | "bfloat" => 2,
        _ => 4,
    }
}

/// (intrinsic name, axis arg) for a SPIR-V dispatch-id read.
fn spirv_index_intrinsic(dim: IndexAxis) -> (&'static str, u32) {
    use IndexAxis::*;
    match dim {
        X => ("llvm.spv.thread.id.i32", 0),
        Y => ("llvm.spv.thread.id.i32", 1),
        Z => ("llvm.spv.thread.id.i32", 2),
        LocalX => ("llvm.spv.thread.id.in.group.i32", 0),
        LocalY => ("llvm.spv.thread.id.in.group.i32", 1),
        LocalZ => ("llvm.spv.thread.id.in.group.i32", 2),
        GroupX => ("llvm.spv.group.id.i32", 0),
        GroupY => ("llvm.spv.group.id.i32", 1),
        GroupZ => ("llvm.spv.group.id.i32", 2),
    }
}
