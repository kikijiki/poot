//! Portable packed-row E4M3FN storage kernels (spec 149 Stage 1).

use crate::KernelGenError;
use crate::helpers::{
    Alloc, broadcast_eff_strides, copy, elem, guard, ld, local, row_major_strides, slice_dtype,
    slice_f32,
};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, Fp8Format, IndexAxis, Local, Operand, Place,
    Rvalue, Statement, Terminator, Ty,
};

fn cu(value: usize) -> Operand {
    Operand::Const(Constant::Usize(value as u64))
}

fn c32(value: u32) -> Operand {
    Operand::Const(Constant::U32(value))
}

fn goto(index: u32) -> Terminator {
    Terminator::Goto {
        target: BlockId { index },
    }
}

/// The index of `shape`'s first zero extent, if any (a packed-row generator's shapes must all be
/// nonempty: a zero-element row has no bytes to address).
fn first_zero_extent(shape: &[usize]) -> Option<usize> {
    shape.iter().position(|&extent| extent == 0)
}

/// Append an exact logical-byte read from a row-padded u32 input and return the byte local.
fn packed_byte_read(
    al: &mut Alloc,
    statements: &mut Vec<Statement>,
    input: Local,
    flat: Local,
    row_len: usize,
) -> Local {
    let words_per_row = row_len.div_ceil(4);
    let row = al.add(Ty::Usize, false);
    let col = al.add(Ty::Usize, false);
    let word_col = al.add(Ty::Usize, false);
    let row_base = al.add(Ty::Usize, false);
    let word_index = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let word = al.add(Ty::U32, false);
    let shifted = al.add(Ty::U32, false);
    let byte = al.add(Ty::U32, false);
    statements.extend([
        Statement::Assign(
            Place::local(row),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(flat)), cu(row_len)),
        ),
        Statement::Assign(
            Place::local(col),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(flat)), cu(row_len)),
        ),
        Statement::Assign(
            Place::local(word_col),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(col)), cu(4)),
        ),
        Statement::Assign(
            Place::local(row_base),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(words_per_row)),
        ),
        Statement::Assign(
            Place::local(word_index),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(row_base)),
                copy(Place::local(word_col)),
            ),
        ),
        Statement::Assign(
            Place::local(lane),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(col)), cu(4)),
        ),
        Statement::Assign(
            Place::local(shift_usize),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
        ),
        Statement::Assign(
            Place::local(shift),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(shift_usize)),
            },
        ),
        Statement::Assign(
            Place::local(word),
            Rvalue::Use(copy(elem(input, word_index))),
        ),
        Statement::Assign(
            Place::local(shifted),
            Rvalue::BinaryOp(
                BinOp::Shr,
                copy(Place::local(word)),
                copy(Place::local(shift)),
            ),
        ),
        Statement::Assign(
            Place::local(byte),
            Rvalue::BinaryOp(BinOp::BitAnd, copy(Place::local(shifted)), c32(0xff)),
        ),
    ]);
    byte
}

/// Decode the normative row-padded u32 representation into contiguous f32 values. One thread handles
/// one logical output element, extracts its packed byte, and applies the format-aware conversion.
pub fn e4m3fn_packed_to_f32(name: &str, logical_row_len: usize) -> Result<Body, KernelGenError> {
    if logical_row_len == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_packed_to_f32",
            what: "logical_row_len".to_string(),
            value: 0,
            min: 1,
        });
    }
    let words_per_row = logical_row_len.div_ceil(4);
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(true), true),
    ]);
    let (input, output) = (local(1), local(2));
    let i = al.add(Ty::Usize, false);
    let len = al.add(Ty::Usize, false);
    let in_bounds = al.add(Ty::Bool, false);
    let row = al.add(Ty::Usize, false);
    let col = al.add(Ty::Usize, false);
    let word_col = al.add(Ty::Usize, false);
    let row_base = al.add(Ty::Usize, false);
    let word_index = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let word = al.add(Ty::U32, false);
    let shifted = al.add(Ty::U32, false);
    let byte = al.add(Ty::U32, false);
    let decoded = al.add(Ty::F32, false);

    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(in_bounds),
                    Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
                ),
            ],
            terminator: guard(in_bounds, 3, 2),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(row),
                    Rvalue::BinaryOp(BinOp::Div, copy(Place::local(i)), cu(logical_row_len)),
                ),
                Statement::Assign(
                    Place::local(col),
                    Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(i)), cu(logical_row_len)),
                ),
                Statement::Assign(
                    Place::local(word_col),
                    Rvalue::BinaryOp(BinOp::Div, copy(Place::local(col)), cu(4)),
                ),
                Statement::Assign(
                    Place::local(row_base),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(words_per_row)),
                ),
                Statement::Assign(
                    Place::local(word_index),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(row_base)),
                        copy(Place::local(word_col)),
                    ),
                ),
                Statement::Assign(
                    Place::local(lane),
                    Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(col)), cu(4)),
                ),
                Statement::Assign(
                    Place::local(shift_usize),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
                ),
                Statement::Assign(
                    Place::local(shift),
                    Rvalue::Cast {
                        to: Ty::U32,
                        operand: copy(Place::local(shift_usize)),
                    },
                ),
                Statement::Assign(
                    Place::local(word),
                    Rvalue::Use(copy(elem(input, word_index))),
                ),
                Statement::Assign(
                    Place::local(shifted),
                    Rvalue::BinaryOp(
                        BinOp::Shr,
                        copy(Place::local(word)),
                        copy(Place::local(shift)),
                    ),
                ),
                Statement::Assign(
                    Place::local(byte),
                    Rvalue::BinaryOp(BinOp::BitAnd, copy(Place::local(shifted)), c32(0xff)),
                ),
                Statement::Assign(
                    Place::local(decoded),
                    Rvalue::Fp8Decode {
                        format: Fp8Format::E4M3Fn,
                        operand: copy(Place::local(byte)),
                    },
                ),
                Statement::Assign(elem(output, i), Rvalue::Use(copy(Place::local(decoded)))),
            ],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Ok(Body::new(name, 2, al.locals, blocks))
}

/// Encode contiguous f32 values directly into the normative row-padded u32 representation. One thread
/// owns one output word, so lane insertion has no cross-thread read/modify/write race and tail lanes stay
/// zero by construction.
pub fn f32_to_e4m3fn_packed(name: &str, logical_row_len: usize) -> Result<Body, KernelGenError> {
    packed_word_writer(name, None, logical_row_len, logical_row_len, None, None)
}

/// Repack a reshape between two normative row layouts while preserving the flat logical byte sequence.
/// One thread owns one output word. This is required when Reshape changes the final-axis extent because
/// source row padding is not part of the logical tensor and cannot be aliased into the new layout.
pub fn e4m3fn_repack_reshape(
    name: &str,
    input_row_len: usize,
    output_row_len: usize,
) -> Result<Body, KernelGenError> {
    packed_word_writer(
        name,
        Some(input_row_len),
        input_row_len,
        output_row_len,
        None,
        None,
    )
}

/// Transpose authoritative packed bytes without decoding them. The source terms map each row-major output
/// coordinate to its row-major input coordinate; the shared writer then extracts and repacks the raw byte.
pub fn e4m3fn_transpose_packed(
    name: &str,
    out_shape: &[usize],
    in_shape: &[usize],
    perm: &[usize],
) -> Result<Body, KernelGenError> {
    if out_shape.len() != in_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_transpose_packed",
            what: "out_shape/in_shape rank".to_string(),
            expected: in_shape.len(),
            actual: out_shape.len(),
        });
    }
    if perm.len() != in_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_transpose_packed",
            what: "perm/in_shape rank".to_string(),
            expected: in_shape.len(),
            actual: perm.len(),
        });
    }
    if let Some(i) = first_zero_extent(out_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_transpose_packed",
            what: format!("out_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    let input_row_len = in_shape.last().copied().unwrap_or(1);
    let output_row_len = out_shape.last().copied().unwrap_or(1);
    let input_strides = row_major_strides(in_shape);
    let source_terms: Vec<usize> = perm.iter().map(|&axis| input_strides[axis]).collect();
    packed_word_writer(
        name,
        Some(input_row_len),
        input_row_len,
        output_row_len,
        Some((out_shape, &source_terms, 0)),
        None,
    )
}

/// Slice authoritative packed bytes without decoding them. The source terms preserve every output
/// coordinate and the base offset applies the static range start along the sliced axis.
pub fn e4m3fn_slice_packed(
    name: &str,
    out_shape: &[usize],
    in_shape: &[usize],
    axis: usize,
    start: usize,
) -> Result<Body, KernelGenError> {
    if out_shape.len() != in_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_slice_packed",
            what: "out_shape/in_shape rank".to_string(),
            expected: in_shape.len(),
            actual: out_shape.len(),
        });
    }
    if axis >= in_shape.len() {
        return Err(KernelGenError::AxisOutOfRange {
            generator: "e4m3fn_slice_packed",
            axis,
            rank: in_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(out_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_slice_packed",
            what: format!("out_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    let input_row_len = in_shape.last().copied().unwrap_or(1);
    let output_row_len = out_shape.last().copied().unwrap_or(1);
    let input_strides = row_major_strides(in_shape);
    let source_base = start.saturating_mul(input_strides[axis]);
    packed_word_writer(
        name,
        Some(input_row_len),
        input_row_len,
        output_row_len,
        Some((out_shape, &input_strides, source_base)),
        None,
    )
}

/// Broadcast authoritative packed bytes without decoding them. Missing leading axes and aligned source
/// unit axes have stride zero; all other axes use their right-aligned row-major source stride.
pub fn e4m3fn_broadcast_packed(
    name: &str,
    out_shape: &[usize],
    in_shape: &[usize],
) -> Result<Body, KernelGenError> {
    if in_shape.len() > out_shape.len() {
        return Err(KernelGenError::ExceedsBound {
            generator: "e4m3fn_broadcast_packed",
            what: "in_shape rank".to_string(),
            value: in_shape.len(),
            bound: out_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(out_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_broadcast_packed",
            what: format!("out_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    let input_row_len = in_shape.last().copied().unwrap_or(1);
    let output_row_len = out_shape.last().copied().unwrap_or(1);
    let source_terms = broadcast_eff_strides(out_shape, in_shape);
    packed_word_writer(
        name,
        Some(input_row_len),
        input_row_len,
        output_row_len,
        Some((out_shape, &source_terms, 0)),
        None,
    )
}

/// Gather authoritative packed bytes without decoding them. The index tensor uses the graph's f32 index
/// convention; `inner`, `axis_len`, and `index_numel` collapse the general Gather coordinate mapping.
pub fn e4m3fn_gather_packed(
    name: &str,
    out_shape: &[usize],
    in_shape: &[usize],
    axis: usize,
    index_shape: &[usize],
) -> Result<Body, KernelGenError> {
    if axis >= in_shape.len() {
        return Err(KernelGenError::AxisOutOfRange {
            generator: "e4m3fn_gather_packed",
            axis,
            rank: in_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(out_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_gather_packed",
            what: format!("out_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    if in_shape[axis] == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_gather_packed",
            what: format!("in_shape[{axis}]"),
            value: 0,
            min: 1,
        });
    }
    let input_row_len = in_shape.last().copied().unwrap_or(1);
    let output_row_len = out_shape.last().copied().unwrap_or(1);
    let inner = in_shape[axis + 1..].iter().product();
    let axis_len = in_shape[axis];
    let index_numel = index_shape.iter().product();
    packed_word_writer(
        name,
        Some(input_row_len),
        input_row_len,
        output_row_len,
        None,
        Some((inner, axis_len, index_numel)),
    )
}

/// DynamicUpdateSlice with an inline static index. Each thread owns one packed output word and copies
/// authoritative bytes from operand or update without observing either input's row padding.
pub fn e4m3fn_dynamic_update_slice_packed(
    name: &str,
    operand_shape: &[usize],
    update_shape: &[usize],
    axis: usize,
    index: usize,
) -> Result<Body, KernelGenError> {
    if axis >= operand_shape.len() {
        return Err(KernelGenError::AxisOutOfRange {
            generator: "e4m3fn_dynamic_update_slice_packed",
            axis,
            rank: operand_shape.len(),
        });
    }
    if operand_shape.len() != update_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_dynamic_update_slice_packed",
            what: "operand_shape/update_shape rank".to_string(),
            expected: operand_shape.len(),
            actual: update_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(operand_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_dynamic_update_slice_packed",
            what: format!("operand_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    if let Some(i) = first_zero_extent(update_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_dynamic_update_slice_packed",
            what: format!("update_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    let end = index + update_shape[axis];
    if end > operand_shape[axis] {
        return Err(KernelGenError::ExceedsBound {
            generator: "e4m3fn_dynamic_update_slice_packed",
            what: "index + update_shape[axis]".to_string(),
            value: end,
            bound: operand_shape[axis],
        });
    }
    Ok(e4m3fn_dynamic_update_slice_impl(
        name,
        operand_shape,
        update_shape,
        axis,
        Some(index),
    ))
}

/// DynamicUpdateSlice with an empty-shape dense-f32 runtime index. The executor validates the concrete
/// index before dispatch; the kernel only converts that accepted scalar to a logical coordinate.
pub fn e4m3fn_dynamic_update_slice_dynamic_packed(
    name: &str,
    operand_shape: &[usize],
    update_shape: &[usize],
    axis: usize,
) -> Result<Body, KernelGenError> {
    if axis >= operand_shape.len() {
        return Err(KernelGenError::AxisOutOfRange {
            generator: "e4m3fn_dynamic_update_slice_dynamic_packed",
            axis,
            rank: operand_shape.len(),
        });
    }
    if operand_shape.len() != update_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_dynamic_update_slice_dynamic_packed",
            what: "operand_shape/update_shape rank".to_string(),
            expected: operand_shape.len(),
            actual: update_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(operand_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_dynamic_update_slice_dynamic_packed",
            what: format!("operand_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    if let Some(i) = first_zero_extent(update_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_dynamic_update_slice_dynamic_packed",
            what: format!("update_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    if update_shape[axis] > operand_shape[axis] {
        return Err(KernelGenError::ExceedsBound {
            generator: "e4m3fn_dynamic_update_slice_dynamic_packed",
            what: "update_shape[axis]".to_string(),
            value: update_shape[axis],
            bound: operand_shape[axis],
        });
    }
    Ok(e4m3fn_dynamic_update_slice_impl(
        name,
        operand_shape,
        update_shape,
        axis,
        None,
    ))
}

fn e4m3fn_dynamic_update_slice_impl(
    name: &str,
    operand_shape: &[usize],
    update_shape: &[usize],
    axis: usize,
    static_index: Option<usize>,
) -> Body {
    let rank = operand_shape.len();
    let operand_strides = row_major_strides(operand_shape);
    let update_strides = row_major_strides(update_shape);
    let operand_row_len = *operand_shape
        .last()
        .expect("DynamicUpdateSlice has rank >= 1");
    let update_row_len = *update_shape
        .last()
        .expect("DynamicUpdateSlice has rank >= 1");
    let output_words_per_row = operand_row_len.div_ceil(4);
    let extent = update_shape[axis];

    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::U32, false), false),
    ];
    let index_input = static_index.is_none().then(|| local(3));
    if index_input.is_some() {
        params.push(ld(slice_f32(false), false));
    }
    let output = local(params.len() as u32);
    params.push(ld(slice_dtype(Ty::U32, true), true));
    let param_count = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let operand = local(1);
    let update = local(2);
    let word_i = al.add(Ty::Usize, false);
    let output_len = al.add(Ty::Usize, false);
    let word_in_bounds = al.add(Ty::Bool, false);
    let output_row = al.add(Ty::Usize, false);
    let output_word_col = al.add(Ty::Usize, false);
    let output_base_col = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, true);
    let lane_in_bounds = al.add(Ty::Bool, false);
    let output_col = al.add(Ty::Usize, false);
    let logical_in_bounds = al.add(Ty::Bool, false);
    let flat_base = al.add(Ty::Usize, false);
    let flat = al.add(Ty::Usize, false);
    let index = al.add(Ty::Usize, false);
    let hi = al.add(Ty::Usize, false);
    let hi_last = al.add(Ty::Usize, false);
    let coords: Vec<_> = (0..rank).map(|_| al.add(Ty::Usize, false)).collect();
    let clamped_lo = al.add(Ty::Usize, false);
    let clamped_axis = al.add(Ty::Usize, false);
    let update_axis = al.add(Ty::Usize, false);
    let update_flat = al.add(Ty::Usize, true);
    let term = al.add(Ty::Usize, false);
    let after_start = al.add(Ty::Bool, false);
    let before_end = al.add(Ty::Bool, false);
    let after_u32 = al.add(Ty::U32, false);
    let before_u32 = al.add(Ty::U32, false);
    let inside = al.add(Ty::U32, false);
    let update_mask = al.add(Ty::U32, false);
    let operand_mask = al.add(Ty::U32, false);
    let masked_operand = al.add(Ty::U32, false);
    let masked_update = al.add(Ty::U32, false);
    let byte = al.add(Ty::U32, false);
    let word = al.add(Ty::U32, true);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let shifted_byte = al.add(Ty::U32, false);
    let next_word = al.add(Ty::U32, false);
    let next_lane = al.add(Ty::Usize, false);

    let mut setup = Vec::new();
    if let Some(index_value) = static_index {
        setup.push(Statement::Assign(
            Place::local(index),
            Rvalue::Use(cu(index_value)),
        ));
    } else {
        let zero = al.add(Ty::Usize, false);
        let index_f32 = al.add(Ty::F32, false);
        setup.extend([
            Statement::Assign(Place::local(zero), Rvalue::Use(cu(0))),
            Statement::Assign(
                Place::local(index_f32),
                Rvalue::Use(copy(elem(index_input.expect("runtime index input"), zero))),
            ),
            Statement::Assign(
                Place::local(index),
                Rvalue::Cast {
                    to: Ty::Usize,
                    operand: copy(Place::local(index_f32)),
                },
            ),
        ]);
    }
    setup.extend([
        Statement::Assign(
            Place::local(hi),
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(index)), cu(extent)),
        ),
        Statement::Assign(
            Place::local(hi_last),
            Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(hi)), cu(1)),
        ),
        Statement::Assign(
            Place::local(output_row),
            Rvalue::BinaryOp(
                BinOp::Div,
                copy(Place::local(word_i)),
                cu(output_words_per_row),
            ),
        ),
        Statement::Assign(
            Place::local(output_word_col),
            Rvalue::BinaryOp(
                BinOp::Rem,
                copy(Place::local(word_i)),
                cu(output_words_per_row),
            ),
        ),
        Statement::Assign(
            Place::local(output_base_col),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(output_word_col)), cu(4)),
        ),
        Statement::Assign(
            Place::local(flat_base),
            Rvalue::BinaryOp(
                BinOp::Mul,
                copy(Place::local(output_row)),
                cu(operand_row_len),
            ),
        ),
        Statement::Assign(Place::local(lane), Rvalue::Use(cu(0))),
        Statement::Assign(Place::local(word), Rvalue::Use(c32(0))),
    ]);

    let mut value_statements = Vec::new();
    for (dim, &coord) in coords.iter().enumerate() {
        let quotient = al.add(Ty::Usize, false);
        value_statements.extend([
            Statement::Assign(
                Place::local(quotient),
                Rvalue::BinaryOp(
                    BinOp::Div,
                    copy(Place::local(flat)),
                    cu(operand_strides[dim]),
                ),
            ),
            Statement::Assign(
                Place::local(coord),
                Rvalue::BinaryOp(
                    BinOp::Rem,
                    copy(Place::local(quotient)),
                    cu(operand_shape[dim]),
                ),
            ),
        ]);
    }
    value_statements.extend([
        Statement::Assign(
            Place::local(clamped_lo),
            Rvalue::BinaryOp(
                BinOp::Max,
                copy(Place::local(coords[axis])),
                copy(Place::local(index)),
            ),
        ),
        Statement::Assign(
            Place::local(clamped_axis),
            Rvalue::BinaryOp(
                BinOp::Min,
                copy(Place::local(clamped_lo)),
                copy(Place::local(hi_last)),
            ),
        ),
        Statement::Assign(
            Place::local(update_axis),
            Rvalue::BinaryOp(
                BinOp::Sub,
                copy(Place::local(clamped_axis)),
                copy(Place::local(index)),
            ),
        ),
        Statement::Assign(Place::local(update_flat), Rvalue::Use(cu(0))),
    ]);
    for (dim, &coord) in coords.iter().enumerate() {
        let source = if dim == axis { update_axis } else { coord };
        value_statements.extend([
            Statement::Assign(
                Place::local(term),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(source)),
                    cu(update_strides[dim]),
                ),
            ),
            Statement::Assign(
                Place::local(update_flat),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(update_flat)),
                    copy(Place::local(term)),
                ),
            ),
        ]);
    }
    let operand_byte = packed_byte_read(
        &mut al,
        &mut value_statements,
        operand,
        flat,
        operand_row_len,
    );
    let update_byte = packed_byte_read(
        &mut al,
        &mut value_statements,
        update,
        update_flat,
        update_row_len,
    );
    value_statements.extend([
        Statement::Assign(
            Place::local(after_start),
            Rvalue::BinaryOp(
                BinOp::Ge,
                copy(Place::local(coords[axis])),
                copy(Place::local(index)),
            ),
        ),
        Statement::Assign(
            Place::local(before_end),
            Rvalue::BinaryOp(
                BinOp::Lt,
                copy(Place::local(coords[axis])),
                copy(Place::local(hi)),
            ),
        ),
        Statement::Assign(
            Place::local(after_u32),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(after_start)),
            },
        ),
        Statement::Assign(
            Place::local(before_u32),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(before_end)),
            },
        ),
        Statement::Assign(
            Place::local(inside),
            Rvalue::BinaryOp(
                BinOp::Mul,
                copy(Place::local(after_u32)),
                copy(Place::local(before_u32)),
            ),
        ),
        Statement::Assign(
            Place::local(update_mask),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(inside)), c32(0xff)),
        ),
        Statement::Assign(
            Place::local(operand_mask),
            Rvalue::BinaryOp(BinOp::BitXor, copy(Place::local(update_mask)), c32(0xff)),
        ),
        Statement::Assign(
            Place::local(masked_operand),
            Rvalue::BinaryOp(
                BinOp::BitAnd,
                copy(Place::local(operand_byte)),
                copy(Place::local(operand_mask)),
            ),
        ),
        Statement::Assign(
            Place::local(masked_update),
            Rvalue::BinaryOp(
                BinOp::BitAnd,
                copy(Place::local(update_byte)),
                copy(Place::local(update_mask)),
            ),
        ),
        Statement::Assign(
            Place::local(byte),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(masked_operand)),
                copy(Place::local(masked_update)),
            ),
        ),
        Statement::Assign(
            Place::local(shift_usize),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
        ),
        Statement::Assign(
            Place::local(shift),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(shift_usize)),
            },
        ),
        Statement::Assign(
            Place::local(shifted_byte),
            Rvalue::BinaryOp(
                BinOp::Shl,
                copy(Place::local(byte)),
                copy(Place::local(shift)),
            ),
        ),
        Statement::Assign(
            Place::local(next_word),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(word)),
                copy(Place::local(shifted_byte)),
            ),
        ),
        Statement::Assign(
            Place::local(word),
            Rvalue::Use(copy(Place::local(next_word))),
        ),
    ]);

    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(word_i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(output_len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(word_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(word_i)),
                        copy(Place::local(output_len)),
                    ),
                ),
            ],
            terminator: guard(word_in_bounds, 8, 2),
        },
        BasicBlock {
            statements: setup,
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(lane_in_bounds),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(lane)), cu(4)),
            )],
            terminator: guard(lane_in_bounds, 7, 4),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_col),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(output_base_col)),
                        copy(Place::local(lane)),
                    ),
                ),
                Statement::Assign(
                    Place::local(logical_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(output_col)),
                        cu(operand_row_len),
                    ),
                ),
                Statement::Assign(
                    Place::local(flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(flat_base)),
                        copy(Place::local(output_col)),
                    ),
                ),
            ],
            terminator: guard(logical_in_bounds, 6, 5),
        },
        BasicBlock {
            statements: value_statements,
            terminator: goto(6),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(next_lane),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(1)),
                ),
                Statement::Assign(
                    Place::local(lane),
                    Rvalue::Use(copy(Place::local(next_lane))),
                ),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                elem(output, word_i),
                Rvalue::Use(copy(Place::local(word))),
            )],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Body::new(name, param_count, al.locals, blocks)
}

/// Scatter-update authoritative packed bytes through the graph's axis-0 inverse map. The host validates
/// every inverse entry before dispatch, so the body may clamp the `-1` keep sentinel to source row zero for
/// a safe branchless read and then mask between base and source bytes. One thread owns one output word.
pub fn e4m3fn_scatter_update_packed(
    name: &str,
    base_shape: &[usize],
    src_shape: &[usize],
) -> Result<Body, KernelGenError> {
    if base_shape.is_empty() {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_scatter_update_packed",
            what: "base_shape rank".to_string(),
            value: 0,
            min: 1,
        });
    }
    if src_shape.len() != base_shape.len() {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_scatter_update_packed",
            what: "src_shape/base_shape rank".to_string(),
            expected: base_shape.len(),
            actual: src_shape.len(),
        });
    }
    if src_shape[1..] != base_shape[1..] {
        return Err(KernelGenError::ShapeMismatch {
            generator: "e4m3fn_scatter_update_packed",
            what: "src_shape[1..] vs base_shape[1..]".to_string(),
            a: src_shape[1..].to_vec(),
            b: base_shape[1..].to_vec(),
        });
    }
    if let Some(i) = first_zero_extent(base_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_scatter_update_packed",
            what: format!("base_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    if let Some(i) = first_zero_extent(src_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_scatter_update_packed",
            what: format!("src_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }

    let rest: usize = base_shape[1..].iter().product();
    let base_row_len = *base_shape.last().expect("ScatterUpdate base has rank >= 1");
    let src_row_len = *src_shape.last().expect("ScatterUpdate src has rank >= 1");
    let base_words_per_row = base_row_len.div_ceil(4);
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_dtype(Ty::U32, false), false),
        ld(slice_f32(false), false),
        ld(slice_dtype(Ty::U32, true), true),
    ]);
    let (base, src, inverse, output) = (local(1), local(2), local(3), local(4));
    let word_i = al.add(Ty::Usize, false);
    let output_len = al.add(Ty::Usize, false);
    let word_in_bounds = al.add(Ty::Bool, false);
    let output_row = al.add(Ty::Usize, false);
    let output_word_col = al.add(Ty::Usize, false);
    let output_base_col = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, true);
    let lane_in_bounds = al.add(Ty::Bool, false);
    let output_col = al.add(Ty::Usize, false);
    let logical_in_bounds = al.add(Ty::Bool, false);
    let flat_base = al.add(Ty::Usize, false);
    let flat = al.add(Ty::Usize, false);
    let pool_row = al.add(Ty::Usize, false);
    let rest_offset = al.add(Ty::Usize, false);
    let inverse_value = al.add(Ty::F32, false);
    let clamped_inverse = al.add(Ty::F32, false);
    let source_row = al.add(Ty::Usize, false);
    let source_base = al.add(Ty::Usize, false);
    let source_flat = al.add(Ty::Usize, false);
    let keep = al.add(Ty::Bool, false);
    let update = al.add(Ty::Bool, false);
    let keep_mask = al.add(Ty::U32, false);
    let update_mask = al.add(Ty::U32, false);
    let kept_byte = al.add(Ty::U32, false);
    let updated_byte = al.add(Ty::U32, false);
    let byte = al.add(Ty::U32, false);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let shifted_byte = al.add(Ty::U32, false);
    let word = al.add(Ty::U32, true);
    let next_word = al.add(Ty::U32, false);
    let next_lane = al.add(Ty::Usize, false);

    let mut value_statements = vec![
        Statement::Assign(
            Place::local(pool_row),
            Rvalue::BinaryOp(BinOp::Div, copy(Place::local(flat)), cu(rest)),
        ),
        Statement::Assign(
            Place::local(rest_offset),
            Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(flat)), cu(rest)),
        ),
        Statement::Assign(
            Place::local(inverse_value),
            Rvalue::Use(copy(elem(inverse, pool_row))),
        ),
        Statement::Assign(
            Place::local(clamped_inverse),
            Rvalue::BinaryOp(
                BinOp::Max,
                copy(Place::local(inverse_value)),
                Operand::Const(Constant::F32(0.0)),
            ),
        ),
        Statement::Assign(
            Place::local(source_row),
            Rvalue::Cast {
                to: Ty::Usize,
                operand: copy(Place::local(clamped_inverse)),
            },
        ),
        Statement::Assign(
            Place::local(source_base),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(source_row)), cu(rest)),
        ),
        Statement::Assign(
            Place::local(source_flat),
            Rvalue::BinaryOp(
                BinOp::Add,
                copy(Place::local(source_base)),
                copy(Place::local(rest_offset)),
            ),
        ),
    ];
    let base_byte = packed_byte_read(&mut al, &mut value_statements, base, flat, base_row_len);
    let source_byte = packed_byte_read(
        &mut al,
        &mut value_statements,
        src,
        source_flat,
        src_row_len,
    );
    value_statements.extend([
        Statement::Assign(
            Place::local(keep),
            Rvalue::BinaryOp(
                BinOp::Lt,
                copy(Place::local(inverse_value)),
                Operand::Const(Constant::F32(0.0)),
            ),
        ),
        Statement::Assign(
            Place::local(update),
            Rvalue::BinaryOp(
                BinOp::Ge,
                copy(Place::local(inverse_value)),
                Operand::Const(Constant::F32(0.0)),
            ),
        ),
        Statement::Assign(
            Place::local(keep_mask),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(keep)),
            },
        ),
        Statement::Assign(
            Place::local(update_mask),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(update)),
            },
        ),
        Statement::Assign(
            Place::local(kept_byte),
            Rvalue::BinaryOp(
                BinOp::Mul,
                copy(Place::local(base_byte)),
                copy(Place::local(keep_mask)),
            ),
        ),
        Statement::Assign(
            Place::local(updated_byte),
            Rvalue::BinaryOp(
                BinOp::Mul,
                copy(Place::local(source_byte)),
                copy(Place::local(update_mask)),
            ),
        ),
        Statement::Assign(
            Place::local(byte),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(kept_byte)),
                copy(Place::local(updated_byte)),
            ),
        ),
        Statement::Assign(
            Place::local(shift_usize),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
        ),
        Statement::Assign(
            Place::local(shift),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(shift_usize)),
            },
        ),
        Statement::Assign(
            Place::local(shifted_byte),
            Rvalue::BinaryOp(
                BinOp::Shl,
                copy(Place::local(byte)),
                copy(Place::local(shift)),
            ),
        ),
        Statement::Assign(
            Place::local(next_word),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(word)),
                copy(Place::local(shifted_byte)),
            ),
        ),
        Statement::Assign(
            Place::local(word),
            Rvalue::Use(copy(Place::local(next_word))),
        ),
    ]);

    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(word_i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(output_len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(word_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(word_i)),
                        copy(Place::local(output_len)),
                    ),
                ),
            ],
            terminator: guard(word_in_bounds, 8, 2),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_row),
                    Rvalue::BinaryOp(
                        BinOp::Div,
                        copy(Place::local(word_i)),
                        cu(base_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_word_col),
                    Rvalue::BinaryOp(
                        BinOp::Rem,
                        copy(Place::local(word_i)),
                        cu(base_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_base_col),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(output_word_col)), cu(4)),
                ),
                Statement::Assign(
                    Place::local(flat_base),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(output_row)), cu(base_row_len)),
                ),
                Statement::Assign(Place::local(lane), Rvalue::Use(cu(0))),
                Statement::Assign(Place::local(word), Rvalue::Use(c32(0))),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(lane_in_bounds),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(lane)), cu(4)),
            )],
            terminator: guard(lane_in_bounds, 7, 4),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_col),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(output_base_col)),
                        copy(Place::local(lane)),
                    ),
                ),
                Statement::Assign(
                    Place::local(logical_in_bounds),
                    Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(output_col)), cu(base_row_len)),
                ),
                Statement::Assign(
                    Place::local(flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(flat_base)),
                        copy(Place::local(output_col)),
                    ),
                ),
            ],
            terminator: guard(logical_in_bounds, 6, 5),
        },
        BasicBlock {
            statements: value_statements,
            terminator: goto(6),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(next_lane),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(1)),
                ),
                Statement::Assign(
                    Place::local(lane),
                    Rvalue::Use(copy(Place::local(next_lane))),
                ),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                elem(output, word_i),
                Rvalue::Use(copy(Place::local(word))),
            )],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Ok(Body::new(name, 4, al.locals, blocks))
}

/// Concatenate two or more nonempty packed E4M3FN tensors without decoding them. One thread owns one
/// output word. Each logical lane reads every input at a clamped valid coordinate and masks in the one
/// segment that owns the output axis coordinate, avoiding an N-deep control-flow ladder.
pub fn e4m3fn_concat_packed(
    name: &str,
    out_shape: &[usize],
    axis: usize,
    in_shapes: &[&[usize]],
) -> Result<Body, KernelGenError> {
    if in_shapes.len() < 2 {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_concat_packed",
            what: "in_shapes count".to_string(),
            value: in_shapes.len(),
            min: 2,
        });
    }
    if axis >= out_shape.len() {
        return Err(KernelGenError::AxisOutOfRange {
            generator: "e4m3fn_concat_packed",
            axis,
            rank: out_shape.len(),
        });
    }
    if let Some(i) = first_zero_extent(out_shape) {
        return Err(KernelGenError::BelowMinimum {
            generator: "e4m3fn_concat_packed",
            what: format!("out_shape[{i}]"),
            value: 0,
            min: 1,
        });
    }
    for (input_index, shape) in in_shapes.iter().enumerate() {
        if shape.len() != out_shape.len() {
            return Err(KernelGenError::CountMismatch {
                generator: "e4m3fn_concat_packed",
                what: format!("in_shapes[{input_index}] rank"),
                expected: out_shape.len(),
                actual: shape.len(),
            });
        }
        if let Some(i) = first_zero_extent(shape) {
            return Err(KernelGenError::BelowMinimum {
                generator: "e4m3fn_concat_packed",
                what: format!("in_shapes[{input_index}][{i}]"),
                value: 0,
                min: 1,
            });
        }
        for (dim, (&input, &output)) in shape.iter().zip(out_shape).enumerate() {
            if dim != axis && input != output {
                return Err(KernelGenError::ShapeMismatch {
                    generator: "e4m3fn_concat_packed",
                    what: format!("in_shapes[{input_index}] vs out_shape at dim {dim}"),
                    a: shape.to_vec(),
                    b: out_shape.to_vec(),
                });
            }
        }
    }

    let rank = out_shape.len();
    let out_strides = row_major_strides(out_shape);
    let output_row_len = *out_shape.last().expect("Concat has rank >= 1");
    let output_words_per_row = output_row_len.div_ceil(4);
    let input_strides: Vec<Vec<usize>> = in_shapes
        .iter()
        .map(|shape| row_major_strides(shape))
        .collect();
    let input_row_lens: Vec<usize> = in_shapes
        .iter()
        .map(|shape| *shape.last().expect("Concat inputs have rank >= 1"))
        .collect();
    let input_words_per_row: Vec<usize> = input_row_lens
        .iter()
        .map(|row_len| row_len.div_ceil(4))
        .collect();
    let mut cumulative_axis = vec![0usize; in_shapes.len() + 1];
    for (index, shape) in in_shapes.iter().enumerate() {
        cumulative_axis[index + 1] = cumulative_axis[index] + shape[axis];
    }
    if cumulative_axis[in_shapes.len()] != out_shape[axis] {
        return Err(KernelGenError::CountMismatch {
            generator: "e4m3fn_concat_packed",
            what: "sum of in_shapes[*][axis] vs out_shape[axis]".to_string(),
            expected: out_shape[axis],
            actual: cumulative_axis[in_shapes.len()],
        });
    }

    let mut params = vec![ld(Ty::Unit, false)];
    for _ in in_shapes {
        params.push(ld(slice_dtype(Ty::U32, false), false));
    }
    params.push(ld(slice_dtype(Ty::U32, true), true));
    let mut al = Alloc::new(params);
    let inputs: Vec<_> = (0..in_shapes.len())
        .map(|index| local((index + 1) as u32))
        .collect();
    let output = local((in_shapes.len() + 1) as u32);
    let word_i = al.add(Ty::Usize, false);
    let output_len = al.add(Ty::Usize, false);
    let word_in_bounds = al.add(Ty::Bool, false);
    let output_row = al.add(Ty::Usize, false);
    let output_word_col = al.add(Ty::Usize, false);
    let output_base_col = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, true);
    let lane_in_bounds = al.add(Ty::Bool, false);
    let output_col = al.add(Ty::Usize, false);
    let logical_in_bounds = al.add(Ty::Bool, false);
    let flat_base = al.add(Ty::Usize, false);
    let flat = al.add(Ty::Usize, false);
    let coords: Vec<_> = (0..rank).map(|_| al.add(Ty::Usize, false)).collect();
    let word = al.add(Ty::U32, true);
    let byte = al.add(Ty::U32, true);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let shifted_byte = al.add(Ty::U32, false);
    let next_word = al.add(Ty::U32, false);
    let next_lane = al.add(Ty::Usize, false);

    let mut value_statements = vec![Statement::Assign(Place::local(byte), Rvalue::Use(c32(0)))];
    for (dim, &coord) in coords.iter().enumerate() {
        let quotient = al.add(Ty::Usize, false);
        value_statements.extend([
            Statement::Assign(
                Place::local(quotient),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(flat)), cu(out_strides[dim])),
            ),
            Statement::Assign(
                Place::local(coord),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(quotient)), cu(out_shape[dim])),
            ),
        ]);
    }
    for (input_index, &input) in inputs.iter().enumerate() {
        let base = al.add(Ty::Usize, false);
        let local_axis0 = al.add(Ty::Usize, false);
        let local_axis = al.add(Ty::Usize, false);
        let input_flat = al.add(Ty::Usize, true);
        let term = al.add(Ty::Usize, false);
        let input_row = al.add(Ty::Usize, false);
        let input_col = al.add(Ty::Usize, false);
        let input_word_col = al.add(Ty::Usize, false);
        let input_row_base = al.add(Ty::Usize, false);
        let input_word_index = al.add(Ty::Usize, false);
        let input_lane = al.add(Ty::Usize, false);
        let input_shift_usize = al.add(Ty::Usize, false);
        let input_shift = al.add(Ty::U32, false);
        let input_word = al.add(Ty::U32, false);
        let shifted_input = al.add(Ty::U32, false);
        let input_byte = al.add(Ty::U32, false);
        let after_start = al.add(Ty::Bool, false);
        let before_end = al.add(Ty::Bool, false);
        let after_start_u32 = al.add(Ty::U32, false);
        let before_end_u32 = al.add(Ty::U32, false);
        let mask = al.add(Ty::U32, false);
        let masked_byte = al.add(Ty::U32, false);
        let next_byte = al.add(Ty::U32, false);
        let axis_extent = in_shapes[input_index][axis];

        value_statements.extend([
            Statement::Assign(
                Place::local(base),
                Rvalue::BinaryOp(
                    BinOp::Max,
                    copy(Place::local(coords[axis])),
                    cu(cumulative_axis[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(local_axis0),
                Rvalue::BinaryOp(
                    BinOp::Sub,
                    copy(Place::local(base)),
                    cu(cumulative_axis[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(local_axis),
                Rvalue::BinaryOp(
                    BinOp::Min,
                    copy(Place::local(local_axis0)),
                    cu(axis_extent - 1),
                ),
            ),
            Statement::Assign(Place::local(input_flat), Rvalue::Use(cu(0))),
        ]);
        for (dim, &coord) in coords.iter().enumerate() {
            let source_coord = if dim == axis { local_axis } else { coord };
            value_statements.extend([
                Statement::Assign(
                    Place::local(term),
                    Rvalue::BinaryOp(
                        BinOp::Mul,
                        copy(Place::local(source_coord)),
                        cu(input_strides[input_index][dim]),
                    ),
                ),
                Statement::Assign(
                    Place::local(input_flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(input_flat)),
                        copy(Place::local(term)),
                    ),
                ),
            ]);
        }
        value_statements.extend([
            Statement::Assign(
                Place::local(input_row),
                Rvalue::BinaryOp(
                    BinOp::Div,
                    copy(Place::local(input_flat)),
                    cu(input_row_lens[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(input_col),
                Rvalue::BinaryOp(
                    BinOp::Rem,
                    copy(Place::local(input_flat)),
                    cu(input_row_lens[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(input_word_col),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(input_col)), cu(4)),
            ),
            Statement::Assign(
                Place::local(input_row_base),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(input_row)),
                    cu(input_words_per_row[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(input_word_index),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(input_row_base)),
                    copy(Place::local(input_word_col)),
                ),
            ),
            Statement::Assign(
                Place::local(input_lane),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(input_col)), cu(4)),
            ),
            Statement::Assign(
                Place::local(input_shift_usize),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(input_lane)), cu(8)),
            ),
            Statement::Assign(
                Place::local(input_shift),
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(Place::local(input_shift_usize)),
                },
            ),
            Statement::Assign(
                Place::local(input_word),
                Rvalue::Use(copy(elem(input, input_word_index))),
            ),
            Statement::Assign(
                Place::local(shifted_input),
                Rvalue::BinaryOp(
                    BinOp::Shr,
                    copy(Place::local(input_word)),
                    copy(Place::local(input_shift)),
                ),
            ),
            Statement::Assign(
                Place::local(input_byte),
                Rvalue::BinaryOp(BinOp::BitAnd, copy(Place::local(shifted_input)), c32(0xff)),
            ),
            Statement::Assign(
                Place::local(after_start),
                Rvalue::BinaryOp(
                    BinOp::Ge,
                    copy(Place::local(coords[axis])),
                    cu(cumulative_axis[input_index]),
                ),
            ),
            Statement::Assign(
                Place::local(before_end),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    copy(Place::local(coords[axis])),
                    cu(cumulative_axis[input_index + 1]),
                ),
            ),
            Statement::Assign(
                Place::local(after_start_u32),
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(Place::local(after_start)),
                },
            ),
            Statement::Assign(
                Place::local(before_end_u32),
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(Place::local(before_end)),
                },
            ),
            Statement::Assign(
                Place::local(mask),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(after_start_u32)),
                    copy(Place::local(before_end_u32)),
                ),
            ),
            Statement::Assign(
                Place::local(masked_byte),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(mask)),
                    copy(Place::local(input_byte)),
                ),
            ),
            Statement::Assign(
                Place::local(next_byte),
                Rvalue::BinaryOp(
                    BinOp::BitOr,
                    copy(Place::local(byte)),
                    copy(Place::local(masked_byte)),
                ),
            ),
            Statement::Assign(
                Place::local(byte),
                Rvalue::Use(copy(Place::local(next_byte))),
            ),
        ]);
    }
    value_statements.extend([
        Statement::Assign(
            Place::local(shift_usize),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
        ),
        Statement::Assign(
            Place::local(shift),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(shift_usize)),
            },
        ),
        Statement::Assign(
            Place::local(shifted_byte),
            Rvalue::BinaryOp(
                BinOp::Shl,
                copy(Place::local(byte)),
                copy(Place::local(shift)),
            ),
        ),
        Statement::Assign(
            Place::local(next_word),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(word)),
                copy(Place::local(shifted_byte)),
            ),
        ),
        Statement::Assign(
            Place::local(word),
            Rvalue::Use(copy(Place::local(next_word))),
        ),
    ]);

    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(word_i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(output_len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(word_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(word_i)),
                        copy(Place::local(output_len)),
                    ),
                ),
            ],
            terminator: guard(word_in_bounds, 8, 2),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_row),
                    Rvalue::BinaryOp(
                        BinOp::Div,
                        copy(Place::local(word_i)),
                        cu(output_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_word_col),
                    Rvalue::BinaryOp(
                        BinOp::Rem,
                        copy(Place::local(word_i)),
                        cu(output_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_base_col),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(output_word_col)), cu(4)),
                ),
                Statement::Assign(
                    Place::local(flat_base),
                    Rvalue::BinaryOp(
                        BinOp::Mul,
                        copy(Place::local(output_row)),
                        cu(output_row_len),
                    ),
                ),
                Statement::Assign(Place::local(lane), Rvalue::Use(cu(0))),
                Statement::Assign(Place::local(word), Rvalue::Use(c32(0))),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(lane_in_bounds),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(lane)), cu(4)),
            )],
            terminator: guard(lane_in_bounds, 7, 4),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_col),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(output_base_col)),
                        copy(Place::local(lane)),
                    ),
                ),
                Statement::Assign(
                    Place::local(logical_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(output_col)),
                        cu(output_row_len),
                    ),
                ),
                Statement::Assign(
                    Place::local(flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(flat_base)),
                        copy(Place::local(output_col)),
                    ),
                ),
            ],
            terminator: guard(logical_in_bounds, 6, 5),
        },
        BasicBlock {
            statements: value_statements,
            terminator: goto(6),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(next_lane),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(1)),
                ),
                Statement::Assign(
                    Place::local(lane),
                    Rvalue::Use(copy(Place::local(next_lane))),
                ),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                elem(output, word_i),
                Rvalue::Use(copy(Place::local(word))),
            )],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Ok(Body::new(
        name,
        (in_shapes.len() + 1) as u32,
        al.locals,
        blocks,
    ))
}

/// Build f32 encode, packed-byte reshape repacking, or packed-byte logical index remapping.
fn packed_word_writer(
    name: &str,
    packed_input_row_len: Option<usize>,
    input_row_len: usize,
    output_row_len: usize,
    packed_remap: Option<(&[usize], &[usize], usize)>,
    packed_gather: Option<(usize, usize, usize)>,
) -> Result<Body, KernelGenError> {
    if input_row_len == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator: "packed_word_writer",
            what: "input_row_len".to_string(),
            value: 0,
            min: 1,
        });
    }
    if output_row_len == 0 {
        return Err(KernelGenError::BelowMinimum {
            generator: "packed_word_writer",
            what: "output_row_len".to_string(),
            value: 0,
            min: 1,
        });
    }
    // Internal invariants between this fn's own optional arguments, guaranteed by every caller in this
    // file (never a caller-facing precondition on shape/count arguments): not a KernelGenError.
    debug_assert!(packed_remap.is_none() || packed_input_row_len.is_some());
    debug_assert!(packed_gather.is_none() || packed_input_row_len.is_some());
    debug_assert!(packed_remap.is_none() || packed_gather.is_none());
    let input_ty = if packed_input_row_len.is_some() {
        Ty::U32
    } else {
        Ty::F32
    };
    let input_words_per_row = input_row_len.div_ceil(4);
    let output_words_per_row = output_row_len.div_ceil(4);
    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_dtype(input_ty.clone(), false), false),
    ];
    let gather_index = packed_gather.map(|_| local(2));
    if packed_gather.is_some() {
        params.push(ld(slice_f32(false), false));
    }
    let output = local(params.len() as u32);
    params.push(ld(slice_dtype(Ty::U32, true), true));
    let param_count = (params.len() - 1) as u32;
    let mut al = Alloc::new(params);
    let input = local(1);
    let word_i = al.add(Ty::Usize, false);
    let output_len = al.add(Ty::Usize, false);
    let word_in_bounds = al.add(Ty::Bool, false);
    let output_row = al.add(Ty::Usize, false);
    let output_word_col = al.add(Ty::Usize, false);
    let output_base_col = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, true);
    let lane_in_bounds = al.add(Ty::Bool, false);
    let output_col = al.add(Ty::Usize, false);
    let logical_in_bounds = al.add(Ty::Bool, false);
    let flat_base = al.add(Ty::Usize, false);
    let flat = al.add(Ty::Usize, false);
    let word = al.add(Ty::U32, true);
    let byte = al.add(Ty::U32, false);
    let shift_usize = al.add(Ty::Usize, false);
    let shift = al.add(Ty::U32, false);
    let shifted_byte = al.add(Ty::U32, false);
    let next_word = al.add(Ty::U32, false);
    let next_lane = al.add(Ty::Usize, false);

    let mut value_statements = Vec::new();
    if packed_input_row_len.is_some() {
        let input_flat = al.add(Ty::Usize, false);
        if let Some((inner, axis_len, index_numel)) = packed_gather {
            let inner_index = al.add(Ty::Usize, false);
            let middle = al.add(Ty::Usize, false);
            let gather_position = al.add(Ty::Usize, false);
            let outer_index = al.add(Ty::Usize, false);
            let index_value = al.add(Ty::F32, false);
            let axis_index = al.add(Ty::Usize, false);
            let outer_base = al.add(Ty::Usize, false);
            let axis_base = al.add(Ty::Usize, false);
            let source_base = al.add(Ty::Usize, false);
            value_statements.extend([
                Statement::Assign(
                    Place::local(inner_index),
                    Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(flat)), cu(inner)),
                ),
                Statement::Assign(
                    Place::local(middle),
                    Rvalue::BinaryOp(BinOp::Div, copy(Place::local(flat)), cu(inner)),
                ),
                Statement::Assign(
                    Place::local(gather_position),
                    Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(middle)), cu(index_numel)),
                ),
                Statement::Assign(
                    Place::local(outer_index),
                    Rvalue::BinaryOp(BinOp::Div, copy(Place::local(middle)), cu(index_numel)),
                ),
                Statement::Assign(
                    Place::local(index_value),
                    Rvalue::Use(copy(elem(
                        gather_index.expect("Gather index param"),
                        gather_position,
                    ))),
                ),
                Statement::Assign(
                    Place::local(axis_index),
                    Rvalue::Cast {
                        to: Ty::Usize,
                        operand: copy(Place::local(index_value)),
                    },
                ),
                Statement::Assign(
                    Place::local(outer_base),
                    Rvalue::BinaryOp(
                        BinOp::Mul,
                        copy(Place::local(outer_index)),
                        cu(axis_len.saturating_mul(inner)),
                    ),
                ),
                Statement::Assign(
                    Place::local(axis_base),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(axis_index)), cu(inner)),
                ),
                Statement::Assign(
                    Place::local(source_base),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(outer_base)),
                        copy(Place::local(axis_base)),
                    ),
                ),
                Statement::Assign(
                    Place::local(input_flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(source_base)),
                        copy(Place::local(inner_index)),
                    ),
                ),
            ]);
        } else if let Some((out_shape, source_terms, source_base)) = packed_remap {
            debug_assert_eq!(out_shape.len(), source_terms.len());
            let output_strides = row_major_strides(out_shape);
            value_statements.push(Statement::Assign(
                Place::local(input_flat),
                Rvalue::Use(cu(source_base)),
            ));
            for ((&extent, &output_stride), &source_stride) in
                out_shape.iter().zip(&output_strides).zip(source_terms)
            {
                let quotient = al.add(Ty::Usize, false);
                let coord = al.add(Ty::Usize, false);
                let term = al.add(Ty::Usize, false);
                let next_input_flat = al.add(Ty::Usize, false);
                value_statements.extend([
                    Statement::Assign(
                        Place::local(quotient),
                        Rvalue::BinaryOp(BinOp::Div, copy(Place::local(flat)), cu(output_stride)),
                    ),
                    Statement::Assign(
                        Place::local(coord),
                        Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(quotient)), cu(extent)),
                    ),
                    Statement::Assign(
                        Place::local(term),
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(coord)), cu(source_stride)),
                    ),
                    Statement::Assign(
                        Place::local(next_input_flat),
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(input_flat)),
                            copy(Place::local(term)),
                        ),
                    ),
                    Statement::Assign(
                        Place::local(input_flat),
                        Rvalue::Use(copy(Place::local(next_input_flat))),
                    ),
                ]);
            }
        } else {
            value_statements.push(Statement::Assign(
                Place::local(input_flat),
                Rvalue::Use(copy(Place::local(flat))),
            ));
        }
        let input_row = al.add(Ty::Usize, false);
        let input_col = al.add(Ty::Usize, false);
        let input_word_col = al.add(Ty::Usize, false);
        let input_row_base = al.add(Ty::Usize, false);
        let input_word_index = al.add(Ty::Usize, false);
        let input_lane = al.add(Ty::Usize, false);
        let input_shift_usize = al.add(Ty::Usize, false);
        let input_shift = al.add(Ty::U32, false);
        let input_word = al.add(Ty::U32, false);
        let shifted_input = al.add(Ty::U32, false);
        value_statements.extend([
            Statement::Assign(
                Place::local(input_row),
                Rvalue::BinaryOp(
                    BinOp::Div,
                    copy(Place::local(input_flat)),
                    cu(input_row_len),
                ),
            ),
            Statement::Assign(
                Place::local(input_col),
                Rvalue::BinaryOp(
                    BinOp::Rem,
                    copy(Place::local(input_flat)),
                    cu(input_row_len),
                ),
            ),
            Statement::Assign(
                Place::local(input_word_col),
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(input_col)), cu(4)),
            ),
            Statement::Assign(
                Place::local(input_row_base),
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(input_row)),
                    cu(input_words_per_row),
                ),
            ),
            Statement::Assign(
                Place::local(input_word_index),
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(input_row_base)),
                    copy(Place::local(input_word_col)),
                ),
            ),
            Statement::Assign(
                Place::local(input_lane),
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(input_col)), cu(4)),
            ),
            Statement::Assign(
                Place::local(input_shift_usize),
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(input_lane)), cu(8)),
            ),
            Statement::Assign(
                Place::local(input_shift),
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(Place::local(input_shift_usize)),
                },
            ),
            Statement::Assign(
                Place::local(input_word),
                Rvalue::Use(copy(elem(input, input_word_index))),
            ),
            Statement::Assign(
                Place::local(shifted_input),
                Rvalue::BinaryOp(
                    BinOp::Shr,
                    copy(Place::local(input_word)),
                    copy(Place::local(input_shift)),
                ),
            ),
            Statement::Assign(
                Place::local(byte),
                Rvalue::BinaryOp(BinOp::BitAnd, copy(Place::local(shifted_input)), c32(0xff)),
            ),
        ]);
    } else {
        let source = al.add(Ty::F32, false);
        value_statements.extend([
            Statement::Assign(Place::local(source), Rvalue::Use(copy(elem(input, flat)))),
            Statement::Assign(
                Place::local(byte),
                Rvalue::Fp8Encode {
                    format: Fp8Format::E4M3Fn,
                    operand: copy(Place::local(source)),
                },
            ),
        ]);
    }
    value_statements.extend([
        Statement::Assign(
            Place::local(shift_usize),
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(8)),
        ),
        Statement::Assign(
            Place::local(shift),
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(shift_usize)),
            },
        ),
        Statement::Assign(
            Place::local(shifted_byte),
            Rvalue::BinaryOp(
                BinOp::Shl,
                copy(Place::local(byte)),
                copy(Place::local(shift)),
            ),
        ),
        Statement::Assign(
            Place::local(next_word),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                copy(Place::local(word)),
                copy(Place::local(shifted_byte)),
            ),
        ),
        Statement::Assign(
            Place::local(word),
            Rvalue::Use(copy(Place::local(next_word))),
        ),
    ]);

    let blocks = vec![
        BasicBlock {
            statements: vec![],
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(word_i),
                dim: IndexAxis::X,
                target: BlockId { index: 1 },
            },
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(Place::local(output_len), Rvalue::Len(Place::local(output))),
                Statement::Assign(
                    Place::local(word_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(word_i)),
                        copy(Place::local(output_len)),
                    ),
                ),
            ],
            terminator: guard(word_in_bounds, 8, 2),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_row),
                    Rvalue::BinaryOp(
                        BinOp::Div,
                        copy(Place::local(word_i)),
                        cu(output_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_word_col),
                    Rvalue::BinaryOp(
                        BinOp::Rem,
                        copy(Place::local(word_i)),
                        cu(output_words_per_row),
                    ),
                ),
                Statement::Assign(
                    Place::local(output_base_col),
                    Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(output_word_col)), cu(4)),
                ),
                Statement::Assign(
                    Place::local(flat_base),
                    Rvalue::BinaryOp(
                        BinOp::Mul,
                        copy(Place::local(output_row)),
                        cu(output_row_len),
                    ),
                ),
                Statement::Assign(Place::local(lane), Rvalue::Use(cu(0))),
                Statement::Assign(Place::local(word), Rvalue::Use(c32(0))),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                Place::local(lane_in_bounds),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(lane)), cu(4)),
            )],
            terminator: guard(lane_in_bounds, 7, 4),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(output_col),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(output_base_col)),
                        copy(Place::local(lane)),
                    ),
                ),
                Statement::Assign(
                    Place::local(logical_in_bounds),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        copy(Place::local(output_col)),
                        cu(output_row_len),
                    ),
                ),
                Statement::Assign(
                    Place::local(flat),
                    Rvalue::BinaryOp(
                        BinOp::Add,
                        copy(Place::local(flat_base)),
                        copy(Place::local(output_col)),
                    ),
                ),
            ],
            terminator: guard(logical_in_bounds, 6, 5),
        },
        BasicBlock {
            statements: value_statements,
            terminator: goto(6),
        },
        BasicBlock {
            statements: vec![
                Statement::Assign(
                    Place::local(next_lane),
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(1)),
                ),
                Statement::Assign(
                    Place::local(lane),
                    Rvalue::Use(copy(Place::local(next_lane))),
                ),
            ],
            terminator: goto(3),
        },
        BasicBlock {
            statements: vec![Statement::Assign(
                elem(output, word_i),
                Rvalue::Use(copy(Place::local(word))),
            )],
            terminator: Terminator::Return,
        },
        BasicBlock {
            statements: vec![],
            terminator: Terminator::Return,
        },
    ];
    Ok(Body::new(name, param_count, al.locals, blocks))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bodies_carry_physical_word_lengths() {
        let encode = f32_to_e4m3fn_packed("encode", 5).expect("f32_to_e4m3fn_packed precondition");
        let decode = e4m3fn_packed_to_f32("decode", 5).expect("e4m3fn_packed_to_f32 precondition");
        let repack =
            e4m3fn_repack_reshape("repack", 5, 2).expect("e4m3fn_repack_reshape precondition");
        let transpose = e4m3fn_transpose_packed("transpose", &[5, 2], &[2, 5], &[1, 0])
            .expect("e4m3fn_transpose_packed precondition");
        let slice = e4m3fn_slice_packed("slice", &[2, 3], &[2, 5], 1, 1)
            .expect("e4m3fn_slice_packed precondition");
        let broadcast = e4m3fn_broadcast_packed("broadcast", &[2, 3, 5], &[2, 1, 5])
            .expect("e4m3fn_broadcast_packed precondition");
        let concat = e4m3fn_concat_packed(
            "concat",
            &[2, 6, 3],
            1,
            &[&[2, 2, 3], &[2, 1, 3], &[2, 3, 3]],
        )
        .expect("e4m3fn_concat_packed precondition");
        let gather = e4m3fn_gather_packed("gather", &[2, 3], &[4, 3], 0, &[2])
            .expect("e4m3fn_gather_packed precondition");
        let scatter_update = e4m3fn_scatter_update_packed("scatter_update", &[5], &[3])
            .expect("e4m3fn_scatter_update_packed precondition");
        let dynamic_update =
            e4m3fn_dynamic_update_slice_packed("dynamic_update", &[2, 4, 5], &[2, 2, 5], 1, 1)
                .expect("e4m3fn_dynamic_update_slice_packed precondition");
        let dynamic_update_runtime = e4m3fn_dynamic_update_slice_dynamic_packed(
            "dynamic_update_runtime",
            &[2, 5],
            &[2, 2],
            1,
        )
        .expect("e4m3fn_dynamic_update_slice_dynamic_packed precondition");
        assert_eq!(encode.param_count, 2);
        assert_eq!(decode.param_count, 2);
        assert_eq!(repack.param_count, 2);
        assert_eq!(transpose.param_count, 2);
        assert_eq!(slice.param_count, 2);
        assert_eq!(broadcast.param_count, 2);
        assert_eq!(concat.param_count, 4);
        assert_eq!(gather.param_count, 3);
        assert_eq!(scatter_update.param_count, 4);
        assert_eq!(dynamic_update.param_count, 3);
        assert_eq!(dynamic_update_runtime.param_count, 4);
        assert!(encode.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(statement, Statement::Assign(_, Rvalue::Fp8Encode { .. }))
            })
        }));
        assert!(decode.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(statement, Statement::Assign(_, Rvalue::Fp8Decode { .. }))
            })
        }));
        assert!(!transpose.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        assert!(!slice.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        assert!(!broadcast.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        assert!(!concat.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        assert!(!gather.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        assert!(!scatter_update.blocks.iter().any(|block| {
            block.statements.iter().any(|statement| {
                matches!(
                    statement,
                    Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                )
            })
        }));
        for body in [&dynamic_update, &dynamic_update_runtime] {
            assert!(!body.blocks.iter().any(|block| {
                block.statements.iter().any(|statement| {
                    matches!(
                        statement,
                        Statement::Assign(_, Rvalue::Fp8Encode { .. } | Rvalue::Fp8Decode { .. })
                    )
                })
            }));
            assert!(body.blocks.iter().any(|block| {
                block.statements.iter().any(|statement| {
                    matches!(
                        statement,
                        Statement::Assign(_, Rvalue::BinaryOp(BinOp::BitXor, _, _))
                    )
                })
            }));
        }
    }
}
