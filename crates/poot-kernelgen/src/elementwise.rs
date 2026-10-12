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

pub(crate) use activation::{erf_dt, tanh_dt};
pub(crate) use binary::binary_broadcast_i32_remu_views_grid;
pub use binary::{binary, binary_broadcast_dt_views_grid, binary_broadcast_i32_geu_views_grid};
pub(crate) use binary_scalar::binary_scalar_dt_grid;
pub use binary_scalar::{
    binary_scalar, binary_scalar_i32_geu_grid, binary_scalar_i32_grid, binary_scalar_i32_remu_grid,
};
pub(crate) use cast::cast_i32_to_f32;
pub use cast::{cast_bf16_to_f32, cast_f16_to_f32, cast_f32_to_bf16, cast_f32_to_f16};
pub(crate) use common::{
    ELEMENTWISE_2D_WORKGROUP_SIZE, build_unary_dt, build_unary_dt_grid, build_unary_dt_views_grid,
};
pub(crate) use unary::{
    math_unary_dt_grid, math_unary_dt_views_grid, recip_dt_grid, recip_dt_views_grid,
    unary_dt_views_grid, unary_i32_clz_views_grid,
};
pub use unary::{unary, unary_dt_grid, unary_i32_clz_grid};
