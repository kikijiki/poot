//! Data movement: the one generator that copies elements between buffers by an index mapping
//! ([`movement_body`]), and the two index-extraction and rotation bodies that are not copies.

use crate::KernelGenError;
use crate::helpers::{
    Alloc, copy, elem, guard, ld, local, row_major_strides, slice_dtype, slice_f32,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue,
    Statement, Terminator, Ty, UnOp,
};

mod builder;
mod expert_group;
mod generate;
mod rope;
mod spec;
mod topk;

pub(crate) use builder::Cfg;
pub(crate) use expert_group::expert_group;
pub use generate::movement_body;
pub(crate) use generate::{Buf, read_lane, write_packed_word};
pub use rope::{rope, rope_dt};
pub use spec::{MovementOp, MovementSpec, Storage, UnitWindow, UpdateIndex};
pub use topk::arg_top_k_dt;
