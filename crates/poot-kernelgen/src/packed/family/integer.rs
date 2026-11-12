//! The integer family ([`poot_quant::families::integer`]): the code through its encoding.

use poot_kernel_ir::Local;

use super::Codes;
use crate::emit::Emit;
use crate::packed::decode_plan::emit_encoded_number;

pub(super) fn emit_value(e: &mut Emit, codes: Codes) -> Local {
    emit_encoded_number(e, codes.field.encoding, codes.field.width(), codes.bits)
}
