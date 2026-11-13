//! The float family ([`poot_quant::families::float`]): the code is a float of the field's format.

use poot_kernel_ir::Local;

use super::Codes;
use crate::emit::Emit;
use crate::packed::float_decode::FloatDecode;

pub(super) fn emit_value(e: &mut Emit, decode: FloatDecode, codes: Codes) -> Local {
    decode.emit(e, codes.bits)
}
