use crate::fused::{FusedScalarOp, emit_i32_shift, emit_scalar_op, is_cmp};
use crate::helpers::{
    Alloc, Layout, copy, elem, guard, ld, local, row_major_strides, slice_bf16, slice_dtype,
    slice_f32, view_eff_strides,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, IntScalarOp, Local, LocalDecl, MathOp,
    Operand, Place, Rvalue, Statement, Terminator, Ty, UnOp,
};

mod activation;
mod binary;
mod binary_scalar;
mod cast;
mod common;
mod unary;

pub use activation::*;
pub use binary::*;
pub use binary_scalar::*;
pub use cast::*;
pub use common::*;
pub use unary::*;
