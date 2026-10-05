use super::*;

/// Card 045: cast an f32 slice to bf16 elementwise (`out[i] = bf16(in[i])`), one thread per element, with
/// the same round-to-nearest `Rvalue::Cast` every bf16-narrowing kernel uses. Feeds bf16 activations into
/// the tensor-core matmul.
pub fn cast_f32_to_bf16(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),         // 0 ret
        ld(slice_f32(false), false), // 1 a (f32 in)
        ld(slice_bf16(true), true),  // 2 c (bf16 out, mut)
        ld(Ty::Usize, false),        // 3 i
        ld(Ty::Usize, false),        // 4 len
        ld(Ty::Bool, false),         // 5 cmp
        ld(Ty::BF16, false),         // 6 res (narrowed)
    ];
    let (a, c, i, len, cmp, res) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(res),
                Rvalue::Cast {
                    to: Ty::BF16,
                    operand: copy(elem(a, i)),
                },
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(res)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// Card 045: widen a bf16 slice to f32 elementwise (`out[i] = f32(in[i])`). Inverse of
/// [`cast_f32_to_bf16`]; lossless. One thread per element, masked.
pub fn cast_bf16_to_f32(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),          // 0 ret
        ld(slice_bf16(false), false), // 1 a (bf16 in)
        ld(slice_f32(true), true),    // 2 c (f32 out, mut)
        ld(Ty::Usize, false),         // 3 i
        ld(Ty::Usize, false),         // 4 len
        ld(Ty::Bool, false),          // 5 cmp
        ld(Ty::F32, false),           // 6 res (widened)
    ];
    let (a, c, i, len, cmp, res) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(res),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(a, i)),
                },
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(res)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// Spec 135: cast an f32 slice to f16 elementwise (`out[i] = f16(in[i])`). The f16 analog of
/// [`cast_f32_to_bf16`].
pub fn cast_f32_to_f16(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),                  // 0 ret
        ld(slice_f32(false), false),          // 1 a (f32 in)
        ld(slice_dtype(Ty::F16, true), true), // 2 c (f16 out, mut)
        ld(Ty::Usize, false),                 // 3 i
        ld(Ty::Usize, false),                 // 4 len
        ld(Ty::Bool, false),                  // 5 cmp
        ld(Ty::F16, false),                   // 6 res (narrowed)
    ];
    let (a, c, i, len, cmp, res) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(res),
                Rvalue::Cast {
                    to: Ty::F16,
                    operand: copy(elem(a, i)),
                },
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(res)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// Spec 135: widen an f16 slice to f32 elementwise (`out[i] = f32(in[i])`). Inverse of
/// [`cast_f32_to_f16`]; lossless. One thread per element, masked.
pub fn cast_f16_to_f32(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),                    // 0 ret
        ld(slice_dtype(Ty::F16, false), false), // 1 a (f16 in)
        ld(slice_f32(true), true),              // 2 c (f32 out, mut)
        ld(Ty::Usize, false),                   // 3 i
        ld(Ty::Usize, false),                   // 4 len
        ld(Ty::Bool, false),                    // 5 cmp
        ld(Ty::F32, false),                     // 6 res (widened)
    ];
    let (a, c, i, len, cmp, res) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(res),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(a, i)),
                },
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(res)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
}

/// Card 372c: widen an i32 slice to f32 elementwise (`out[i] = f32(in[i])`), one thread per element,
/// masked.
///
/// Needed by the validation witness (a packet lane must be F32) and the card 356 selector Cast. Exact for
/// `|in[i]| <= 2^24`, the bound the guard and device witness admission enforce; above that `sitofp` rounds.
///
/// The planner's Cast arm dispatches on the (input, output) dtype pair, so this body is reachable only
/// from an i32 source (a target-only dispatch emitted a bf16 kernel for cast(i32->f32)).
pub fn cast_i32_to_f32(name: &str) -> Body {
    let locals = vec![
        ld(Ty::Unit, false),                    // 0 ret
        ld(slice_dtype(Ty::I32, false), false), // 1 a (i32 in)
        ld(slice_f32(true), true),              // 2 c (f32 out, mut)
        ld(Ty::Usize, false),                   // 3 i
        ld(Ty::Usize, false),                   // 4 len
        ld(Ty::Bool, false),                    // 5 cmp
        ld(Ty::F32, false),                     // 6 res (widened)
    ];
    let (a, c, i, len, cmp, res) = (local(1), local(2), local(3), local(4), local(5), local(6));
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(i),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![
            Statement::Assign(Place::local(len), Rvalue::Len(Place::local(c))),
            Statement::Assign(
                Place::local(cmp),
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(i)), copy(Place::local(len))),
            ),
        ],
        terminator: guard(cmp, 3, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            Statement::Assign(
                Place::local(res),
                Rvalue::Cast {
                    to: Ty::F32,
                    operand: copy(elem(a, i)),
                },
            ),
            Statement::Assign(elem(c, i), Rvalue::Use(copy(Place::local(res)))),
        ],
        terminator: Terminator::Return,
    };
    let bb3 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(name, 2, locals, vec![bb0, bb1, bb2, bb3])
}
