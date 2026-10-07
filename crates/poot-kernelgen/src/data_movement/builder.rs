//! A control-flow builder for the movement bodies: straight-line statements through [`Emit`], and
//! `if`-without-`else` regions that never nest deeper than the generator writes them.

use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, IndexAxis, Local, LocalDecl, Operand, Place, Rvalue,
    Statement, Terminator, Ty,
};

use crate::elementwise::ELEMENTWISE_2D_WORKGROUP_SIZE;
use crate::emit::{Emit, bin, cu};
use crate::helpers::{Alloc, copy, guard};

pub(crate) struct Cfg {
    al: Alloc,
    blocks: Vec<BasicBlock>,
    cur: Vec<Statement>,
    param_count: u32,
}

impl Cfg {
    /// A builder over `params` (the unit local first, then the buffers).
    pub(crate) fn new(params: Vec<LocalDecl>) -> Self {
        let param_count = (params.len() - 1) as u32;
        Self {
            al: Alloc::new(params),
            blocks: Vec::new(),
            cur: Vec::new(),
            param_count,
        }
    }

    /// Run straight-line emission against the current block.
    pub(crate) fn emit<R>(&mut self, f: impl FnOnce(&mut Emit) -> R) -> R {
        let mut e = Emit::new(&mut self.al);
        let result = f(&mut e);
        self.cur.extend(e.take());
        result
    }

    pub(crate) fn let_(&mut self, ty: Ty, rvalue: Rvalue) -> Local {
        self.emit(|e| e.let_(ty, rvalue))
    }

    pub(crate) fn bin(&mut self, ty: Ty, op: BinOp, a: Operand, b: Operand) -> Local {
        self.emit(|e| bin(e, ty, op, a, b))
    }

    /// A mutable local of `ty`, assigned `rvalue` now.
    pub(crate) fn var(&mut self, ty: Ty, rvalue: Rvalue) -> Local {
        let local = self.al.add(ty, true);
        self.cur
            .push(Statement::Assign(Place::local(local), rvalue));
        local
    }

    pub(crate) fn assign(&mut self, place: Place, rvalue: Rvalue) {
        self.cur.push(Statement::Assign(place, rvalue));
    }

    pub(crate) fn usize_const(&mut self, value: usize) -> Local {
        self.let_(Ty::Usize, Rvalue::Use(cu(value)))
    }

    /// Close the current block with `terminator` and return its index.
    fn seal(&mut self, terminator: Terminator) -> usize {
        self.blocks.push(BasicBlock {
            statements: std::mem::take(&mut self.cur),
            terminator,
        });
        self.blocks.len() - 1
    }

    fn next_block(&self) -> u32 {
        self.blocks.len() as u32
    }

    /// Read the dispatch's global thread index into a fresh local; the rest of the body is the next block.
    /// With `x_groups` the launch is folded onto a 2-D grid of `x_groups` workgroups across, and the index
    /// is `(group_y * x_groups + group_x) * ELEMENTWISE_2D_WORKGROUP_SIZE + lane`.
    pub(crate) fn thread_index(&mut self, x_groups: Option<usize>) -> Local {
        let index = self.al.add(Ty::Usize, false);
        let Some(x_groups) = x_groups else {
            let target = BlockId {
                index: self.next_block() + 1,
            };
            self.seal(Terminator::ThreadIndexCall {
                destination: Place::local(index),
                dim: IndexAxis::X,
                target,
            });
            return index;
        };
        let group_x = self.index(IndexAxis::GroupX);
        let group_y = self.index(IndexAxis::GroupY);
        let lane = self.index(IndexAxis::LocalX);
        let row = self.bin(
            Ty::Usize,
            BinOp::Mul,
            copy(Place::local(group_y)),
            cu(x_groups),
        );
        let group = self.bin(
            Ty::Usize,
            BinOp::Add,
            copy(Place::local(row)),
            copy(Place::local(group_x)),
        );
        let first = self.bin(
            Ty::Usize,
            BinOp::Mul,
            copy(Place::local(group)),
            cu(ELEMENTWISE_2D_WORKGROUP_SIZE),
        );
        let global = self.bin(
            Ty::Usize,
            BinOp::Add,
            copy(Place::local(first)),
            copy(Place::local(lane)),
        );
        self.assign(Place::local(index), Rvalue::Use(copy(Place::local(global))));
        index
    }

    /// Read one launch coordinate into a fresh local; the rest of the body is the next block.
    pub(crate) fn index(&mut self, dim: IndexAxis) -> Local {
        let destination = self.al.add(Ty::Usize, false);
        let target = BlockId {
            index: self.next_block() + 1,
        };
        self.seal(Terminator::ThreadIndexCall {
            destination: Place::local(destination),
            dim,
            target,
        });
        destination
    }

    /// A workgroup barrier; every lane of the workgroup must reach it, so it belongs on a path the whole
    /// workgroup takes (not inside a loop or an `if` the lanes can leave differently).
    pub(crate) fn barrier(&mut self) {
        let target = BlockId {
            index: self.next_block() + 1,
        };
        self.seal(Terminator::Barrier { target });
    }

    /// `while cond { body }`: `cond` emits the test into the loop head and returns its bool local, so it is
    /// evaluated on every iteration; `body` runs while it holds, then control returns to the head.
    pub(crate) fn while_(
        &mut self,
        cond: impl FnOnce(&mut Self) -> Local,
        body: impl FnOnce(&mut Self),
    ) {
        let entry = self.seal(Terminator::Return);
        let head_start = self.next_block();
        self.blocks[entry].terminator = Terminator::Goto {
            target: BlockId { index: head_start },
        };
        let test = cond(self);
        let head = self.seal(Terminator::Return);
        let body_start = self.next_block();
        body(self);
        let tail = self.seal(Terminator::Goto {
            target: BlockId { index: head_start },
        });
        let exit = tail as u32 + 1;
        debug_assert!(body_start <= tail as u32);
        self.blocks[head].terminator = guard(test, exit, body_start);
    }

    /// `if cond { then }`: control rejoins after `then`, which may itself open regions.
    pub(crate) fn if_then(&mut self, cond: Local, then: impl FnOnce(&mut Self)) {
        let head = self.seal(Terminator::Return);
        let then_start = self.next_block();
        then(self);
        let tail = self.seal(Terminator::Return);
        let join = BlockId {
            index: tail as u32 + 1,
        };
        self.blocks[tail].terminator = Terminator::Goto { target: join };
        self.blocks[head].terminator = guard(cond, join.index, then_start);
    }

    /// `cond` as a bool local holding `a && b`: both operands are evaluated, so neither may fault.
    pub(crate) fn both(&mut self, a: Local, b: Local) -> Local {
        let a = self.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(a)),
            },
        );
        let b = self.let_(
            Ty::U32,
            Rvalue::Cast {
                to: Ty::U32,
                operand: copy(Place::local(b)),
            },
        );
        let both = self.bin(
            Ty::U32,
            BinOp::BitAnd,
            copy(Place::local(a)),
            copy(Place::local(b)),
        );
        self.bin(
            Ty::Bool,
            BinOp::Ne,
            copy(Place::local(both)),
            Operand::Const(poot_kernel_ir::Constant::U32(0)),
        )
    }

    /// End the body and hand back the finished kernel.
    pub(crate) fn finish(mut self, name: &str) -> Body {
        self.seal(Terminator::Return);
        Body::new(name, self.param_count, self.al.locals, self.blocks)
    }
}
