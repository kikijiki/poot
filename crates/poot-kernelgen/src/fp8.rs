//! The two FP8 value casts: `f32` to packed E4M3FN rows and back. Moving an E4M3FN payload (slice,
//! gather, concat, ...) is `crate::data_movement`, which never decodes it; these are the only bodies
//! that change a value's representation.

use poot_kernel_ir::{BinOp, Body, Fp8Format, Operand, Place, Rvalue, Ty};
use poot_quant::format::FloatFormat;

use crate::KernelGenError;
use crate::data_movement::{Buf, Cfg, Storage, read_lane, write_packed_word};
use crate::emit::select_u32;
use crate::emit::{c32, copy};
use crate::helpers::{elem, ld, local, slice_dtype};
use crate::packed::float_decode::emit_float_decode;

/// The f32 a NaN byte decodes to, as `poot_quant::scalar::e4m3fn_to_f32` states it.
const CANONICAL_NAN_BITS: u32 = 0x7fc0_0000;

fn nonempty_row(generator: &'static str, logical_row_len: usize) -> Result<(), KernelGenError> {
    if logical_row_len == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator,
            what: "logical_row_len".to_string(),
            value: 0,
            min: 1,
        });
    }
    Ok(())
}

/// Decode the normative row-padded u32 representation into contiguous f32 values. One thread handles one
/// logical output element, extracts its packed byte, and decodes it with the one E4M3FN decoder every
/// packed body uses. That decoder is exact on the finite codes; the two NaN codes (`0x7f`, `0xff`) it does
/// not state decode to the canonical NaN here, so this cast accepts every byte a payload can hold.
pub fn e4m3fn_packed_to_f32(name: &str, logical_row_len: usize) -> Result<Body, KernelGenError> {
    nonempty_row("e4m3fn_packed_to_f32", logical_row_len)?;
    let mut cfg = Cfg::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::F32, true), true),
    ]);
    let (input, output) = (local(1), local(2));
    let unit = cfg.thread_index(None);
    let len = cfg.let_(Ty::Usize, Rvalue::Len(Place::local(output)));
    let in_bounds = cfg.bin(Ty::Bool, BinOp::Lt, copy(unit), copy(len));
    cfg.if_then(in_bounds, |cfg| {
        let packed = Buf {
            param: input,
            shape: vec![logical_row_len],
        };
        let byte = read_lane(cfg, &Storage::BYTE_LANES, &packed, unit);
        let decoded = cfg.emit(|e| {
            let finite = emit_float_decode(e, FloatFormat::E4m3Fn, byte);
            let finite_bits = e.let_(
                Ty::U32,
                Rvalue::Bitcast {
                    to: Ty::U32,
                    operand: copy(finite),
                },
            );
            let magnitude = crate::emit::bin(e, Ty::U32, BinOp::BitAnd, copy(byte), c32(0x7f));
            let is_nan = crate::emit::bin(e, Ty::Bool, BinOp::Eq, copy(magnitude), c32(0x7f));
            let nan_bits = e.let_(Ty::U32, Rvalue::Use(c32(CANONICAL_NAN_BITS)));
            let bits = select_u32(e, is_nan, nan_bits, finite_bits);
            e.let_(
                Ty::F32,
                Rvalue::Bitcast {
                    to: Ty::F32,
                    operand: copy(bits),
                },
            )
        });
        cfg.assign(elem(output, unit), Rvalue::Use(copy(decoded)));
    });
    Ok(cfg.finish(name))
}

/// Encode contiguous f32 values directly into the normative row-padded u32 representation. One thread
/// owns one output word, so lane insertion has no cross-thread read/modify/write race and tail lanes stay
/// zero by construction.
pub fn f32_to_e4m3fn_packed(name: &str, logical_row_len: usize) -> Result<Body, KernelGenError> {
    nonempty_row("f32_to_e4m3fn_packed", logical_row_len)?;
    let mut cfg = Cfg::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::F32, false), false),
        ld(slice_dtype(Ty::U32, true), true),
    ]);
    let (input, output) = (local(1), local(2));
    let unit = cfg.thread_index(None);
    let len = cfg.let_(Ty::Usize, Rvalue::Len(Place::local(output)));
    let in_bounds = cfg.bin(Ty::Bool, BinOp::Lt, copy(unit), copy(len));
    cfg.if_then(in_bounds, |cfg| {
        let packed = Buf {
            param: output,
            shape: vec![logical_row_len],
        };
        write_packed_word(cfg, &packed, unit, 8, |cfg, flat| {
            let value = cfg.let_(Ty::F32, Rvalue::Use(Operand::Copy(elem(input, flat))));
            cfg.let_(
                Ty::U32,
                Rvalue::Fp8Encode {
                    format: Fp8Format::E4M3Fn,
                    operand: copy(value),
                },
            )
        });
    });
    Ok(cfg.finish(name))
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::interp::{Buffer, run};
    use poot_kernel_ir::test_support::workgroups_covering;
    use poot_quant::scalar::e4m3fn_to_f32;

    use super::*;

    /// Run `body` over `inputs` on the KIR interpreter, with `threads` threads.
    fn interpret(body: &Body, mut buffers: Vec<Buffer>, threads: usize) -> Vec<Buffer> {
        body.verify().expect("the generated body verifies");
        run(body, workgroups_covering(body, threads), &mut buffers).expect("the body runs");
        buffers
    }

    /// Pack `bytes` as rows of `row_len` into the normative layout, with `pad` in the unused lanes so a
    /// decode that read padding would show it.
    fn pack_rows(bytes: &[u8], row_len: usize, pad: u8) -> Vec<u32> {
        let words_per_row = row_len.div_ceil(4);
        let mut words = Vec::new();
        for row in bytes.chunks(row_len) {
            for word in 0..words_per_row {
                let mut value = 0u32;
                for lane in 0..4 {
                    let byte = row.get(word * 4 + lane).copied().unwrap_or(pad);
                    value |= u32::from(byte) << (8 * lane);
                }
                words.push(value);
            }
        }
        words
    }

    /// Every byte, including both NaN codes and signed zero, decodes to the scalar reference's bits for
    /// row widths that do and do not fill a word.
    #[test]
    fn unpack_matches_the_scalar_decode_for_every_byte_and_row_width() {
        let bytes: Vec<u8> = (0..=255).collect();
        for row_len in [1, 3, 4, 5, 8, 16] {
            let rows = bytes.len() / row_len * row_len;
            let bytes = &bytes[..rows];
            let words = pack_rows(bytes, row_len, 0xa5);
            let body = e4m3fn_packed_to_f32("unpack", row_len).unwrap();
            let buffers = interpret(
                &body,
                vec![
                    Buffer::from_u32s(&words),
                    Buffer::from_f32s(&vec![0.0; rows]),
                ],
                rows,
            );
            let decoded = buffers[1].to_f32s().unwrap();
            for (index, (&byte, value)) in bytes.iter().zip(&decoded).enumerate() {
                assert_eq!(
                    value.to_bits(),
                    e4m3fn_to_f32(byte).to_bits(),
                    "row_len {row_len}, element {index}, byte {byte:#04x}"
                );
            }
        }
    }

    /// Encode then unpack returns every finite code, and the encoded rows leave their padding lanes zero.
    #[test]
    fn pack_round_trips_every_finite_code_and_zeroes_padding() {
        let finite: Vec<u8> = (0..=255).filter(|byte| byte & 0x7f != 0x7f).collect();
        for row_len in [1, 3, 4, 5, 8] {
            let rows = finite.len() / row_len * row_len;
            let values: Vec<f32> = finite[..rows]
                .iter()
                .map(|&byte| e4m3fn_to_f32(byte))
                .collect();
            let words = rows.div_ceil(row_len) * row_len.div_ceil(4);
            let pack = f32_to_e4m3fn_packed("pack", row_len).unwrap();
            let packed = interpret(
                &pack,
                vec![
                    Buffer::from_f32s(&values),
                    Buffer::from_u32s(&vec![0xffff_ffff; words]),
                ],
                words,
            );
            let packed_words = packed[1].to_u32s().unwrap();
            assert_eq!(
                packed_words,
                pack_rows(&finite[..rows], row_len, 0),
                "row_len {row_len}"
            );
        }
    }
}
