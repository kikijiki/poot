//! The fragment-IR completeness probe (POOT-1432): one body that stages `A`/`B` tiles in LDS, runs a K loop of
//! `WmmaLoadLds` (at a nonzero base, in either layout) plus `WmmaMma`, and reads the accumulator back one of
//! four ways. The probe is a host-checkable contract: every backend dispatches the same body and compares it
//! with the same host reference (`FragmentProbe::reference`'s index maps), so a wrong layout, base, stride or
//! element map shows up as a wrong number, never as a plausible one.

use super::{
    BasicBlock, BinOp, BlockId, Body, Constant, FragmentMajor, FragmentRole, FragmentTy, IndexAxis,
    Local, Operand, Place, Rvalue, Statement, Terminator, Ty, WmmaDtype, WmmaMat, WmmaShape,
    WorkgroupLocalDecl, accumulator_map, copy, elem, guard, ld, local, slice_dtype, slice_f32,
};

/// How the accumulator starts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AccInit {
    /// `WmmaZero`.
    Zero,
    /// An accumulator `WmmaLoadLds` of the `c0` tile, staged in LDS in this layout.
    LoadLds(FragmentMajor),
}

/// How the accumulator reaches the output buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Readout {
    /// `WmmaStore` to the output, row-major.
    Store,
    /// `WmmaStoreLds` in this layout, a barrier, then a lane copy of the LDS tile (in LDS order) to the output.
    StoreLds(FragmentMajor),
    /// Per-lane `WmmaElemRead` of every element, written to `out[row * 16 + col]` with `(row, col)` from the
    /// declared [`accumulator_map`]. AMDGCN only.
    ElemRead,
    /// `WmmaElemRead`, multiply by `row + 1`, `WmmaElemWrite`, then `WmmaStore`: the in-place epilogue a
    /// `scale_rows` runs. AMDGCN only.
    ElemScale,
}

/// One configuration of the fragment probe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FragmentProbe {
    /// Operand dtype of the `A`/`B` fragments.
    pub dtype: WmmaDtype,
    /// Layout of the `A` tiles in memory.
    pub a_major: FragmentMajor,
    /// Layout of the `B` tiles in memory.
    pub b_major: FragmentMajor,
    /// K-loop trip count: the product accumulates `k_tiles` 16x16x16 tiles.
    pub k_tiles: u32,
    /// How the accumulator starts.
    pub acc_init: AccInit,
    /// How the accumulator is read back.
    pub readout: Readout,
}

impl FragmentProbe {
    /// Elements in one 16x16 tile.
    pub const TILE: u32 = 256;
    /// Elements before the first tile in every LDS array: every tile base is nonzero, and a multiple of 16
    /// bytes for both operand dtypes and f32.
    pub const LDS_PAD: u32 = 64;
    /// The tile edge.
    pub const EDGE: usize = 16;

    /// The probe body: params `a`, `b` (dtype slices of `k_tiles` tiles), `c0` (an f32 tile), `out` (f32,
    /// one tile). One 32-lane workgroup.
    pub fn body(&self, name: &str) -> Body {
        let shape = WmmaShape::M16N16K16;
        let dtype = self.dtype;
        let operand_ty = match dtype {
            WmmaDtype::F16 => Ty::F16,
            WmmaDtype::Bf16 => Ty::BF16,
        };
        let locals = vec![
            ld(Ty::Unit, false),                               // 0
            ld(slice_dtype(operand_ty.clone(), false), false), // 1 a
            ld(slice_dtype(operand_ty.clone(), false), false), // 2 b
            ld(slice_f32(false), false),                       // 3 c0
            ld(slice_f32(true), true),                         // 4 out
            ld(Ty::Usize, false),                              // 5 lane
            ld(Ty::Usize, true),                               // 6 idx
            ld(Ty::Usize, true),                               // 7 lds index
            ld(operand_ty, true),                              // 8 operand element
            ld(Ty::F32, true),                                 // 9 f32 element
            ld(Ty::Usize, true),                               // 10 kt
            ld(Ty::Bool, true),                                // 11 kt < k_tiles
            ld(Ty::Usize, true),                               // 12 tile base
            ld(Ty::Usize, true),                               // 13 scratch
            ld(
                Ty::Fragment(FragmentTy::matrix_major(
                    WmmaMat::A,
                    dtype,
                    shape,
                    self.a_major,
                )),
                false,
            ), // 14 af
            ld(
                Ty::Fragment(FragmentTy::matrix_major(
                    WmmaMat::B,
                    dtype,
                    shape,
                    self.b_major,
                )),
                false,
            ), // 15 bf
            ld(Ty::Fragment(FragmentTy::accumulator(dtype, shape)), false), // 16 acc
            ld(Ty::Usize, true),                               // 17 half
            ld(Ty::Usize, true),                               // 18 col
            ld(Ty::Usize, true),                               // 19 row
            ld(Ty::F32, true),                                 // 20 row as f32
            ld(Ty::F32, true),                                 // 21 element value
            ld(Ty::Usize, true),                               // 22 out index
        ];
        let (a, b, c0, out) = (local(1), local(2), local(3), local(4));
        let (lane, idx, lidx, velem, vf32) = (local(5), local(6), local(7), local(8), local(9));
        let (kt, in_range, tile_base, scratch) = (local(10), local(11), local(12), local(13));
        let (af, bf, acc) = (local(14), local(15), local(16));
        let (half, col, row, rowf, ev, oidx) = (
            local(17),
            local(18),
            local(19),
            local(20),
            local(21),
            local(22),
        );
        let cu = |x: u32| Operand::Const(Constant::Usize(u64::from(x)));
        let assign = |d: Local, rv: Rvalue| Statement::Assign(Place::local(d), rv);
        let goto = |b: u32| Terminator::Goto {
            target: BlockId { index: b },
        };
        let pad = Self::LDS_PAD;
        let ab_len = self.k_tiles * Self::TILE;

        // bb1: stage `a`, `b` (operand dtype) and `c0` (f32) in LDS at the padded base, 32 lanes striding.
        let mut stage = Vec::new();
        for (src, array, len, scalar) in [
            (a, 0u8, ab_len, velem),
            (b, 1, ab_len, velem),
            (c0, 2, Self::TILE, vf32),
        ] {
            for chunk in 0..len / 32 {
                stage.push(assign(
                    idx,
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(chunk * 32)),
                ));
                stage.push(assign(
                    lidx,
                    Rvalue::BinaryOp(BinOp::Add, copy(Place::local(idx)), cu(pad)),
                ));
                stage.push(assign(scalar, Rvalue::Use(copy(elem(src, idx)))));
                stage.push(Statement::WorkgroupLocalWrite {
                    idx: copy(Place::local(lidx)),
                    value: copy(Place::local(scalar)),
                    array,
                });
            }
        }

        // bb2: seed the accumulator, kt = 0.
        let acc_init = match self.acc_init {
            AccInit::Zero => Statement::WmmaZero {
                dtype,
                shape,
                dst: acc,
            },
            AccInit::LoadLds(major) => Statement::WmmaLoadLds {
                role: FragmentRole::Accumulator,
                dtype,
                shape,
                array: 2,
                base: cu(pad),
                major,
                stride: 16,
                dst: acc,
            },
        };

        // bb4: one K step: tile base = pad + kt * 256, load A and B from LDS, accumulate in place.
        let load =
            |which: WmmaMat, array: u8, major: FragmentMajor, dst: Local| Statement::WmmaLoadLds {
                role: FragmentRole::Matrix(which),
                dtype,
                shape,
                array,
                base: copy(Place::local(tile_base)),
                major,
                stride: 16,
                dst,
            };

        let mut blocks = vec![
            BasicBlock {
                statements: vec![],
                terminator: Terminator::ThreadIndexCall {
                    destination: Place::local(lane),
                    dim: IndexAxis::LocalX,
                    target: BlockId { index: 1 },
                },
            },
            BasicBlock {
                statements: stage,
                terminator: Terminator::Barrier {
                    target: BlockId { index: 2 },
                },
            },
            BasicBlock {
                statements: vec![acc_init, assign(kt, Rvalue::Use(cu(0)))],
                terminator: goto(3),
            },
            BasicBlock {
                statements: vec![assign(
                    in_range,
                    Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(kt)), cu(self.k_tiles)),
                )],
                terminator: guard(in_range, 5, 4),
            },
            BasicBlock {
                statements: vec![
                    assign(
                        scratch,
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(kt)), cu(Self::TILE)),
                    ),
                    assign(
                        tile_base,
                        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(scratch)), cu(pad)),
                    ),
                    load(WmmaMat::A, 0, self.a_major, af),
                    load(WmmaMat::B, 1, self.b_major, bf),
                    Statement::WmmaMma {
                        dtype,
                        shape,
                        a: af,
                        b: bf,
                        c: acc,
                        dst: acc,
                    },
                    assign(
                        kt,
                        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(kt)), cu(1)),
                    ),
                ],
                terminator: goto(3),
            },
        ];

        // bb5 (+ bb6): the readout.
        let store_global = Statement::WmmaStore {
            dtype,
            shape,
            tile: elem(out, oidx),
            stride: 16,
            src: acc,
        };
        // `(row, col)` of element `i` of this lane, from the declared accumulator map.
        let lane_parts = vec![
            assign(
                half,
                Rvalue::BinaryOp(
                    BinOp::Div,
                    copy(Place::local(lane)),
                    cu(accumulator_map::LANES_PER_ROW_SET),
                ),
            ),
            assign(
                col,
                Rvalue::BinaryOp(
                    BinOp::Rem,
                    copy(Place::local(lane)),
                    cu(accumulator_map::LANES_PER_ROW_SET),
                ),
            ),
        ];
        let element_row = |i: u32| {
            assign(
                row,
                Rvalue::BinaryOp(
                    BinOp::Add,
                    copy(Place::local(half)),
                    cu(accumulator_map::position(0, i).0),
                ),
            )
        };
        let mut exit = vec![assign(oidx, Rvalue::Use(cu(0)))];
        let mut tail = None;
        match self.readout {
            Readout::Store => exit.push(store_global),
            Readout::StoreLds(major) => {
                exit.push(Statement::WmmaStoreLds {
                    dtype,
                    shape,
                    array: 3,
                    base: cu(pad),
                    major,
                    stride: 16,
                    src: acc,
                });
                let mut copy_out = Vec::new();
                for chunk in 0..Self::TILE / 32 {
                    copy_out.push(assign(
                        idx,
                        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(lane)), cu(chunk * 32)),
                    ));
                    copy_out.push(assign(
                        lidx,
                        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(idx)), cu(pad)),
                    ));
                    copy_out.push(assign(
                        vf32,
                        Rvalue::WorkgroupLocalRead {
                            idx: copy(Place::local(lidx)),
                            array: 3,
                        },
                    ));
                    copy_out.push(Statement::Assign(
                        elem(out, idx),
                        Rvalue::Use(copy(Place::local(vf32))),
                    ));
                }
                tail = Some(BasicBlock {
                    statements: copy_out,
                    terminator: Terminator::Return,
                });
            }
            Readout::ElemRead => {
                exit.extend(lane_parts);
                for i in 0..accumulator_map::ELEMENTS_PER_LANE {
                    exit.push(Statement::WmmaElemRead {
                        src: acc,
                        idx: i as u8,
                        dst: ev,
                    });
                    exit.push(element_row(i));
                    exit.push(assign(
                        scratch,
                        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(16)),
                    ));
                    exit.push(assign(
                        oidx,
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(scratch)),
                            copy(Place::local(col)),
                        ),
                    ));
                    exit.push(Statement::Assign(
                        elem(out, oidx),
                        Rvalue::Use(copy(Place::local(ev))),
                    ));
                }
            }
            Readout::ElemScale => {
                exit.extend(lane_parts);
                for i in 0..accumulator_map::ELEMENTS_PER_LANE {
                    exit.push(Statement::WmmaElemRead {
                        src: acc,
                        idx: i as u8,
                        dst: ev,
                    });
                    exit.push(element_row(i));
                    exit.push(assign(
                        rowf,
                        Rvalue::Cast {
                            to: Ty::F32,
                            operand: copy(Place::local(row)),
                        },
                    ));
                    exit.push(assign(
                        rowf,
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            copy(Place::local(rowf)),
                            Operand::Const(Constant::F32(1.0)),
                        ),
                    ));
                    exit.push(assign(
                        ev,
                        Rvalue::BinaryOp(
                            BinOp::Mul,
                            copy(Place::local(ev)),
                            copy(Place::local(rowf)),
                        ),
                    ));
                    exit.push(Statement::WmmaElemWrite {
                        dst: acc,
                        idx: i as u8,
                        value: copy(Place::local(ev)),
                    });
                }
                exit.push(assign(oidx, Rvalue::Use(cu(0))));
                exit.push(store_global);
            }
        }
        match tail {
            Some(tail) => {
                blocks.push(BasicBlock {
                    statements: exit,
                    terminator: Terminator::Barrier {
                        target: BlockId { index: 6 },
                    },
                });
                blocks.push(tail);
            }
            None => blocks.push(BasicBlock {
                statements: exit,
                terminator: Terminator::Return,
            }),
        }

        let mut body = Body::new(name, 4, locals, blocks);
        body.workgroup_size = [32, 1, 1];
        body.workgroup_locals = vec![
            WorkgroupLocalDecl {
                elem_ty: if dtype == WmmaDtype::F16 {
                    Ty::F16
                } else {
                    Ty::BF16
                },
                len: pad + ab_len,
            },
            WorkgroupLocalDecl {
                elem_ty: if dtype == WmmaDtype::F16 {
                    Ty::F16
                } else {
                    Ty::BF16
                },
                len: pad + ab_len,
            },
            WorkgroupLocalDecl {
                elem_ty: Ty::F32,
                len: pad + Self::TILE,
            },
            WorkgroupLocalDecl {
                elem_ty: Ty::F32,
                len: pad + Self::TILE,
            },
        ];
        body
    }

    /// The buffer index of logical `A[m][k]` of K tile `tile` in the `a` parameter.
    pub fn a_index(&self, tile: usize, m: usize, k: usize) -> usize {
        tile * Self::TILE as usize + major_index(self.a_major, m, k)
    }

    /// The buffer index of logical `B[k][n]` of K tile `tile` in the `b` parameter.
    pub fn b_index(&self, tile: usize, k: usize, n: usize) -> usize {
        tile * Self::TILE as usize + major_index(self.b_major, k, n)
    }

    /// The `c0` index of logical accumulator element `(m, n)`.
    pub fn c0_index(&self, m: usize, n: usize) -> usize {
        match self.acc_init {
            AccInit::Zero => m * Self::EDGE + n,
            AccInit::LoadLds(major) => major_index(major, m, n),
        }
    }

    /// The `out` index logical accumulator element `(m, n)` lands at.
    pub fn out_index(&self, m: usize, n: usize) -> usize {
        match self.readout {
            Readout::StoreLds(major) => major_index(major, m, n),
            Readout::Store | Readout::ElemRead | Readout::ElemScale => m * Self::EDGE + n,
        }
    }

    /// The factor the readout applies to row `m`: `m + 1` for [`Readout::ElemScale`], else 1.
    pub fn row_scale(&self, m: usize) -> f32 {
        match self.readout {
            Readout::ElemScale => (m + 1) as f32,
            _ => 1.0,
        }
    }
}

/// The index of `(major_axis_index, minor_axis_index)`-ordered element `(r, c)` of a 16-wide tile: row-major
/// steps `r` by the stride, column-major steps `c`.
fn major_index(major: FragmentMajor, r: usize, c: usize) -> usize {
    match major {
        FragmentMajor::Row => r * FragmentProbe::EDGE + c,
        FragmentMajor::Col => c * FragmentProbe::EDGE + r,
    }
}
