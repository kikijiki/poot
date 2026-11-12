//! `Rvalue::SubgroupShuffleXor` fixtures: one kernel per payload type that exchanges a per-thread value across a
//! list of masks and stores what each lane received, with the output a host loop gives. The device rows
//! (`subgroup_shuffle_xor_matches_the_interpreter_on_*`) dispatch them through each runtime and the interpreter row
//! (`tests/subgroup_shuffle_fixtures.rs`) runs them on the CPU, so the expectation and both sides are one source.

use super::{ld, local};
use crate::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Operand, Place, ProjectionElem, Rvalue,
    Statement, Terminator, Ty,
};

/// The masks every shuffle fixture exchanges over, one output word per mask and thread.
pub const SHUFFLE_MASKS: [u32; 5] = [1, 2, 4, 8, 16];

/// The payload of [`SHUFFLE_U32_MARK`]'s fixture: thread `x` holds `x ^ SHUFFLE_U32_MARK`, a full 32-bit pattern so
/// a partner read from the wrong lane changes the high half too.
pub const SHUFFLE_U32_MARK: u32 = 0xdead_0000;

/// The payload of the `F32` fixture: thread `x` holds `x as f32 * SHUFFLE_F32_SCALE`, exact for every thread.
pub const SHUFFLE_F32_SCALE: f32 = 0.5;

/// One shuffle fixture: `shuffle_xor(params: out)`, thread `x` (global) stores `out[5x + k]` = the value lane
/// `LocalX ^ SHUFFLE_MASKS[k]` of its workgroup holds, for each `k`. `ty` is `Ty::U32` or `Ty::F32`; the workgroup
/// is `workgroup` lanes of X, and the caller dispatches whole workgroups (the body has no bounds guard).
pub fn subgroup_shuffle_xor_kernel(ty: Ty, workgroup: u32) -> Body {
    let (out, thread, held) = (local(1), local(2), local(3));
    let masks = SHUFFLE_MASKS.len();
    let slice = |elem: &Ty| Ty::Ref {
        mutable: true,
        pointee: Box::new(Ty::Slice(Box::new(elem.clone()))),
    };
    // _0 unit, _1 out, _2 x (usize), _3 x as `ty`, _4 the value held, _5 the output base; then per mask the
    // received value and the output index.
    let mut locals = vec![
        ld(Ty::Unit, false),
        ld(slice(&ty), true),
        ld(Ty::Usize, false),
        ld(ty.clone(), false),
        ld(ty.clone(), false),
        ld(Ty::Usize, false),
    ];
    let (value, base) = (local(4), local(5));
    let copy = |l| Operand::Copy(Place::local(l));
    let usize_const = |v: u64| Operand::Const(Constant::Usize(v));
    let held_value = match ty {
        Ty::U32 => Rvalue::BinaryOp(
            BinOp::BitXor,
            copy(held),
            Operand::Const(Constant::U32(SHUFFLE_U32_MARK)),
        ),
        Ty::F32 => Rvalue::BinaryOp(
            BinOp::Mul,
            copy(held),
            Operand::Const(Constant::F32(SHUFFLE_F32_SCALE)),
        ),
        ref other => panic!("no shuffle fixture for {other:?}"),
    };
    let mut statements = vec![
        Statement::Assign(
            Place::local(held),
            Rvalue::Cast {
                to: ty.clone(),
                operand: copy(thread),
            },
        ),
        Statement::Assign(Place::local(value), held_value),
        Statement::Assign(
            Place::local(base),
            Rvalue::BinaryOp(BinOp::Mul, copy(thread), usize_const(masks as u64)),
        ),
    ];
    for (k, &mask) in SHUFFLE_MASKS.iter().enumerate() {
        let (received, index) = (local(6 + 2 * k as u32), local(7 + 2 * k as u32));
        locals.extend([ld(ty.clone(), false), ld(Ty::Usize, false)]);
        statements.extend([
            Statement::Assign(
                Place::local(received),
                Rvalue::SubgroupShuffleXor {
                    value: copy(value),
                    mask,
                },
            ),
            Statement::Assign(
                Place::local(index),
                Rvalue::BinaryOp(BinOp::Add, copy(base), usize_const(k as u64)),
            ),
            Statement::Assign(
                Place {
                    local: out,
                    projection: vec![ProjectionElem::Deref, ProjectionElem::Index(index)],
                },
                Rvalue::Use(copy(received)),
            ),
        ]);
    }
    let mut body = Body::new(
        match ty {
            Ty::U32 => "subgroup_shuffle_xor_u32",
            _ => "subgroup_shuffle_xor_f32",
        },
        1,
        locals,
        vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(thread),
                    dim: IndexAxis::X,
                    target: BlockId { index: 1 },
                },
            },
            BasicBlock {
                statements,
                terminator: Terminator::Return,
            },
        ],
    );
    body.workgroup_size = [workgroup, 1, 1];
    body
}

/// What [`subgroup_shuffle_xor_kernel`] over `ty` stores for `threads` threads in workgroups of `workgroup`: for
/// thread `x`, mask `k`, the value of thread `x ^ SHUFFLE_MASKS[k]` (a partner in the same workgroup, since every
/// mask is below the workgroup size and the workgroup base has those bits clear). `U32` words are the values;
/// `F32` words are the bit patterns of the floats.
pub fn subgroup_shuffle_xor_expected(ty: &Ty, threads: u32) -> Vec<u32> {
    let held = |x: u32| match ty {
        Ty::U32 => x ^ SHUFFLE_U32_MARK,
        Ty::F32 => (x as f32 * SHUFFLE_F32_SCALE).to_bits(),
        other => panic!("no shuffle fixture for {other:?}"),
    };
    (0..threads)
        .flat_map(|x| SHUFFLE_MASKS.map(|mask| held(x ^ mask)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::{Buffer, run};

    const WORKGROUP: u32 = 128;
    const THREADS: u32 = 256;

    /// The device rows compare each backend with [`subgroup_shuffle_xor_expected`]; this row ties that expectation
    /// to the interpreter, so a backend that agrees with it agrees with the interpreter.
    #[test]
    fn the_interpreter_stores_what_the_host_expectation_says() {
        let words = THREADS as usize * SHUFFLE_MASKS.len();
        for ty in [Ty::U32, Ty::F32] {
            let body = subgroup_shuffle_xor_kernel(ty.clone(), WORKGROUP);
            let want = subgroup_shuffle_xor_expected(&ty, THREADS);
            let mut buffers = [match ty {
                Ty::U32 => Buffer::from_u32s(&vec![0; words]),
                _ => Buffer::from_f32s(&vec![0.0; words]),
            }];
            run(&body, [THREADS / WORKGROUP, 1, 1], &mut buffers).unwrap();
            let got: Vec<u32> = match ty {
                Ty::U32 => buffers[0].to_u32s().unwrap(),
                _ => buffers[0]
                    .to_f32s()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect(),
            };
            assert_eq!(got, want, "{ty:?}");
        }
    }
}
