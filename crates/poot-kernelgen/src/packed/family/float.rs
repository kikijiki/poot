//! The float family ([`poot_quant::families::float`]): the code is a float of the field's format.

use poot_kernel_ir::Local;
use poot_quant::format::FieldEncoding;

use super::Codes;
use crate::emit::Emit;
use crate::packed::float_decode::emit_float_decode;

pub(super) fn emit_value(e: &mut Emit, codes: Codes) -> Local {
    let FieldEncoding::Float(format) = codes.field.encoding else {
        unreachable!("format tests: a Float value map has a Float-encoded Codes field")
    };
    emit_float_decode(e, format, codes.bits)
}
