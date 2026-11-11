//! The order-preserving key of a router score, shared by the generators that rank scores (`block_top_k` and
//! `stable_rank`), so that both read one definition of "canonical score order".

use poot_kernel_ir::{BinOp, Rvalue, Ty};

use crate::attention::local_op;
use crate::block_top_k::{select, ubin32};
use crate::data_movement::Cfg;
use crate::emit::c32;

/// The key of the F32 `value`: its canonical score's bits (`canonical_router_scores` of the graph IR: NaN and
/// `-inf` become `f32::MIN`, `+inf` becomes `f32::MAX`, and `-0.0` is `+0.0`, as the comparisons treat it),
/// flipped so that unsigned order is float order.
pub(crate) fn order_key(c: &mut Cfg, value: poot_kernel_ir::Local) -> poot_kernel_ir::Local {
    const MIN_BITS: u32 = 0xFF7F_FFFF;
    const MAX_BITS: u32 = 0x7F7F_FFFF;
    const INF_BITS: u32 = 0x7F80_0000;
    let bits = c.let_(
        Ty::U32,
        Rvalue::Bitcast {
            to: Ty::U32,
            operand: local_op(value),
        },
    );
    let magnitude = ubin32(c, BinOp::BitAnd, local_op(bits), c32(0x7FFF_FFFF));
    let sign = ubin32(c, BinOp::Shr, local_op(bits), c32(31));
    let negative = c.bin(Ty::Bool, BinOp::Ne, local_op(sign), c32(0));
    let nan = c.bin(Ty::Bool, BinOp::Gt, local_op(magnitude), c32(INF_BITS));
    let infinite = c.bin(Ty::Bool, BinOp::Eq, local_op(magnitude), c32(INF_BITS));
    let zero = c.bin(Ty::Bool, BinOp::Eq, local_op(magnitude), c32(0));
    let inf_bits = select(c, Ty::U32, negative, c32(MIN_BITS), c32(MAX_BITS));
    let bits = select(c, Ty::U32, nan, c32(MIN_BITS), local_op(bits));
    let bits = select(c, Ty::U32, infinite, local_op(inf_bits), local_op(bits));
    let bits = select(c, Ty::U32, zero, c32(0), local_op(bits));
    // Negative floats sort descending by magnitude: flip every bit; positive ones flip the sign bit.
    let sign = ubin32(c, BinOp::Shr, local_op(bits), c32(31));
    let negative = c.bin(Ty::Bool, BinOp::Ne, local_op(sign), c32(0));
    let flip = select(c, Ty::U32, negative, c32(u32::MAX), c32(0x8000_0000));
    ubin32(c, BinOp::BitXor, local_op(bits), local_op(flip))
}
