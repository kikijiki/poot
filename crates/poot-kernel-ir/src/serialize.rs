//! Grid serialization: a target-neutral [`Body`] transform that turns a barrier-free body's thread grid into
//! loops, so a target that runs one scalar thread (an AIE core) executes the body the ordinary generators
//! produced for a GPU, with no second generator.
//!
//! The contract is the one the Tier-1 tests check: for every body [`serialize_grid`] accepts,
//! `run(serialize_grid(body, grid), [1, 1, 1]) == run(body, grid / body.workgroup_size)`. It holds because
//! the accepted bodies have no thread cooperation: no barrier, no workgroup memory, no subgroup collective,
//! no atomic and no fence, so threads interact only through disjoint global stores and any serial order of
//! them is a legal execution of the grid. Everything else is a typed [`SerializeError`], never a loop that
//! changes what the body means.

use crate::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, LocalDecl, MemoryScope, Operand,
    Place, Rvalue, Statement, SwitchTargets, Terminator, Ty,
};

/// The thread extent of a launch: the number of invocations along each axis, as the launch states it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Total threads `[x, y, z]`.
    pub threads: [u32; 3],
}

/// Why a body cannot be serialized: what it does that a single thread running the grid in order would not
/// reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SerializeError {
    /// The body synchronizes a workgroup; there is no other thread to wait for.
    #[error("block bb{} ends in a workgroup barrier", block.index)]
    Barrier {
        /// The block that ends in the barrier.
        block: BlockId,
    },
    /// The body shares workgroup-local memory between threads.
    #[error("the body uses workgroup-local memory")]
    WorkgroupMemory,
    /// The body is written for a subgroup collective (a matrix fragment).
    #[error("the body uses a subgroup collective (a matrix fragment)")]
    Subgroup,
    /// The body performs an atomic whose result depends on the other threads' order.
    #[error("the body performs an atomic at {scope:?} scope")]
    Atomic {
        /// The scope the atomic orders against.
        scope: MemoryScope,
    },
    /// The body fences memory against other threads.
    #[error("the body fences memory at {scope:?} scope")]
    Fence {
        /// The scope the fence orders for.
        scope: MemoryScope,
    },
    /// An axis of the grid has no threads, so there is nothing to run.
    #[error("the grid {grid:?} has an empty axis")]
    EmptyGrid {
        /// The offending grid.
        grid: [u32; 3],
    },
    /// An axis of the grid is not a whole number of workgroups, so the dispatch the body is written for has
    /// no loop of the same extent.
    #[error("the grid {grid:?} is not a whole number of {workgroup:?} workgroups")]
    PartialWorkgroup {
        /// The offending grid.
        grid: [u32; 3],
        /// The workgroup shape of the body.
        workgroup: [u32; 3],
    },
}

/// Serialize `body` over `grid`: one loop per grid axis with more than one thread, outermost `z`, innermost
/// `x`, whose induction variables are the global thread ids. A [`Terminator::ThreadIndexCall`] becomes an
/// assignment from them (the local and group ids derive from the global id and the workgroup shape), and a
/// `Return` continues the innermost loop. The result is a one-thread body that runs once.
pub fn serialize_grid(body: &Body, grid: Grid) -> Result<Body, SerializeError> {
    reject_cooperation(body)?;
    let workgroup = body.workgroup_size;
    if grid.threads.contains(&0) {
        return Err(SerializeError::EmptyGrid { grid: grid.threads });
    }
    if grid
        .threads
        .iter()
        .zip(workgroup)
        .any(|(&threads, size)| threads % size != 0)
    {
        return Err(SerializeError::PartialWorkgroup {
            grid: grid.threads,
            workgroup,
        });
    }

    // The loops, outermost first. An axis of one thread is not a loop: its ids are the constant 0.
    let looped: Vec<usize> = (0..3)
        .rev()
        .filter(|&axis| grid.threads[axis] > 1)
        .collect();
    let depth = looped.len();
    let first_counter = body.locals.len() as u32;
    let compare = Local {
        index: first_counter + depth as u32,
    };
    let counter_of = |axis: usize| {
        looped
            .iter()
            .position(|&looped_axis| looped_axis == axis)
            .map(|level| Local {
                index: first_counter + level as u32,
            })
    };

    // Block layout: the loop scaffolding first, then the body's own blocks.
    //   0                     counter 0 starts at 0
    //   1 + level             the header of the loop at `level`
    //   1 + depth + level     entering the loop at `level`: start the next counter (not the innermost)
    //   2 * depth + level     the latch of the loop at `level`: step its counter
    //   3 * depth             the return
    let header = |level: usize| 1 + level;
    let enter = |level: usize| 1 + depth + level;
    let latch = |level: usize| 2 * depth + level;
    let exit = 3 * depth;
    let base = if depth == 0 { 0 } else { 3 * depth + 1 };
    let block = |index: usize| BlockId {
        index: index as u32,
    };
    let remap = |target: BlockId| block(base + target.index as usize);
    let usize_const = |value: u64| Operand::Const(Constant::Usize(value));

    let mut locals = body.locals.clone();
    for _ in 0..depth {
        locals.push(LocalDecl {
            ty: Ty::Usize,
            mutable: true,
        });
    }
    if depth > 0 {
        locals.push(LocalDecl {
            ty: Ty::Bool,
            mutable: true,
        });
    }

    let mut blocks = Vec::with_capacity(base + body.blocks.len());
    if depth > 0 {
        let counter = |level: usize| {
            Place::local(Local {
                index: first_counter + level as u32,
            })
        };
        let start = |level: usize| Statement::Assign(counter(level), Rvalue::Use(usize_const(0)));
        blocks.push(BasicBlock {
            statements: vec![start(0)],
            terminator: Terminator::Goto {
                target: block(header(0)),
            },
        });
        for (level, &axis) in looped.iter().enumerate() {
            let past = if level == 0 { exit } else { latch(level - 1) };
            let inside = if level + 1 == depth {
                base
            } else {
                enter(level)
            };
            blocks.push(BasicBlock {
                statements: vec![Statement::Assign(
                    Place::local(compare),
                    Rvalue::BinaryOp(
                        BinOp::Lt,
                        Operand::Copy(counter(level)),
                        usize_const(u64::from(grid.threads[axis])),
                    ),
                )],
                terminator: Terminator::SwitchInt {
                    discr: Operand::Copy(Place::local(compare)),
                    targets: SwitchTargets {
                        branches: vec![(0, block(past))],
                        otherwise: block(inside),
                    },
                },
            });
        }
        for level in 0..depth - 1 {
            blocks.push(BasicBlock {
                statements: vec![start(level + 1)],
                terminator: Terminator::Goto {
                    target: block(header(level + 1)),
                },
            });
        }
        for level in 0..depth {
            blocks.push(BasicBlock {
                statements: vec![Statement::Assign(
                    counter(level),
                    Rvalue::BinaryOp(BinOp::Add, Operand::Copy(counter(level)), usize_const(1)),
                )],
                terminator: Terminator::Goto {
                    target: block(header(level)),
                },
            });
        }
        blocks.push(BasicBlock {
            statements: Vec::new(),
            terminator: Terminator::Return,
        });
    }

    for original in &body.blocks {
        let mut statements = original.statements.clone();
        let terminator = match &original.terminator {
            Terminator::Goto { target } => Terminator::Goto {
                target: remap(*target),
            },
            Terminator::SwitchInt { discr, targets } => Terminator::SwitchInt {
                discr: discr.clone(),
                targets: SwitchTargets {
                    branches: targets
                        .branches
                        .iter()
                        .map(|&(value, target)| (value, remap(target)))
                        .collect(),
                    otherwise: remap(targets.otherwise),
                },
            },
            Terminator::ThreadIndexCall {
                destination,
                dim,
                target,
            } => {
                let (axis, kind) = axis_id(*dim);
                let global = counter_of(axis).map(|counter| Operand::Copy(Place::local(counter)));
                let value = match (kind, global) {
                    // An axis of one thread: every id on it is 0.
                    (_, None) => Rvalue::Use(usize_const(0)),
                    (IdKind::Global, Some(global)) => Rvalue::Use(global),
                    (IdKind::Local, Some(global)) => Rvalue::BinaryOp(
                        BinOp::Rem,
                        global,
                        usize_const(u64::from(workgroup[axis])),
                    ),
                    (IdKind::Group, Some(global)) => Rvalue::BinaryOp(
                        BinOp::Div,
                        global,
                        usize_const(u64::from(workgroup[axis])),
                    ),
                };
                statements.push(Statement::Assign(destination.clone(), value));
                Terminator::Goto {
                    target: remap(*target),
                }
            }
            Terminator::Return if depth > 0 => Terminator::Goto {
                target: block(latch(depth - 1)),
            },
            Terminator::Return => Terminator::Return,
            Terminator::Trap { code } => Terminator::Trap { code: *code },
            Terminator::Barrier { .. } => unreachable!("rejected by reject_cooperation"),
        };
        blocks.push(BasicBlock {
            statements,
            terminator,
        });
    }

    let mut serialized = Body::new(body.name.clone(), body.param_count, locals, blocks);
    serialized.workgroup_size = [1, 1, 1];
    Ok(serialized)
}

enum IdKind {
    Global,
    Local,
    Group,
}

/// The grid axis an index reads and which of its ids.
fn axis_id(axis: IndexAxis) -> (usize, IdKind) {
    match axis {
        IndexAxis::X => (0, IdKind::Global),
        IndexAxis::Y => (1, IdKind::Global),
        IndexAxis::Z => (2, IdKind::Global),
        IndexAxis::LocalX => (0, IdKind::Local),
        IndexAxis::LocalY => (1, IdKind::Local),
        IndexAxis::LocalZ => (2, IdKind::Local),
        IndexAxis::GroupX => (0, IdKind::Group),
        IndexAxis::GroupY => (1, IdKind::Group),
        IndexAxis::GroupZ => (2, IdKind::Group),
    }
}

/// Refuse every construct whose meaning depends on another thread of the grid.
fn reject_cooperation(body: &Body) -> Result<(), SerializeError> {
    if !body.workgroup_locals.is_empty() {
        return Err(SerializeError::WorkgroupMemory);
    }
    if body
        .locals
        .iter()
        .any(|decl| matches!(decl.ty, Ty::Fragment(_)))
    {
        return Err(SerializeError::Subgroup);
    }
    for (index, block) in body.blocks.iter().enumerate() {
        if matches!(block.terminator, Terminator::Barrier { .. }) {
            return Err(SerializeError::Barrier {
                block: BlockId {
                    index: index as u32,
                },
            });
        }
        for statement in &block.statements {
            match statement {
                Statement::WorkgroupLocalWrite { .. }
                | Statement::WmmaLoadLds { .. }
                | Statement::WmmaStoreLds { .. } => return Err(SerializeError::WorkgroupMemory),
                Statement::WmmaLoad { .. }
                | Statement::WmmaMma { .. }
                | Statement::WmmaStore { .. }
                | Statement::WmmaZero { .. } => return Err(SerializeError::Subgroup),
                Statement::Fence { scope, .. } => {
                    return Err(SerializeError::Fence { scope: *scope });
                }
                Statement::Assign(_, rvalue) => match rvalue {
                    Rvalue::WorkgroupLocalRead { .. } => {
                        return Err(SerializeError::WorkgroupMemory);
                    }
                    Rvalue::WorkgroupLocalAtomic { .. }
                    | Rvalue::WorkgroupLocalCompareExchange { .. } => {
                        return Err(SerializeError::Atomic {
                            scope: MemoryScope::Workgroup,
                        });
                    }
                    Rvalue::GlobalAtomic { .. } | Rvalue::GlobalCompareExchange { .. } => {
                        return Err(SerializeError::Atomic {
                            scope: MemoryScope::Device,
                        });
                    }
                    _ => {}
                },
                Statement::StorageLive(_)
                | Statement::StorageDead(_)
                | Statement::VectorStore { .. } => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interp::{Buffer, run};
    use crate::test_support::{wg_sum, wmma_tile};
    use crate::{AtomicOp, MemoryOrdering, ProjectionElem};

    fn l(index: u32) -> Local {
        Local { index }
    }

    fn cusize(value: u64) -> Operand {
        Operand::Const(Constant::Usize(value))
    }

    fn read(local: u32) -> Operand {
        Operand::Copy(Place::local(l(local)))
    }

    fn assign(local: u32, rvalue: Rvalue) -> Statement {
        Statement::Assign(Place::local(l(local)), rvalue)
    }

    fn out_buffer(elem: Ty) -> Ty {
        Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(elem))),
        }
    }

    fn local_decls(tys: Vec<Ty>) -> Vec<LocalDecl> {
        std::iter::once(Ty::Unit)
            .chain(tys)
            .map(|ty| LocalDecl { ty, mutable: true })
            .collect()
    }

    /// A kernel that reads all five kinds of id it can and writes
    /// `1 + group_x * 10000 + local_x * 100 + y * 10 + z` to `out[(z * threads_y + y) * threads_x + x]`. Every
    /// thread's slot and value differ, so a wrong id on any axis changes the buffer.
    fn id_probe(grid: [u32; 3], workgroup: [u32; 3]) -> Body {
        // _1 out, _2 x, _3 y, _4 z, _5 local x, _6 group x, _7 slot, _8 value, _9 scratch, _10 value as u32.
        let tys = vec![
            out_buffer(Ty::U32),
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::Usize,
            Ty::U32,
        ];
        let call = |dest: u32, dim: IndexAxis, target: u32| BasicBlock {
            statements: Vec::new(),
            terminator: Terminator::ThreadIndexCall {
                destination: Place::local(l(dest)),
                dim,
                target: BlockId { index: target },
            },
        };
        let mul = |dest: u32, a: u32, k: u64| {
            assign(dest, Rvalue::BinaryOp(BinOp::Mul, read(a), cusize(k)))
        };
        let add = |dest: u32, a: u32, b: u32| {
            assign(dest, Rvalue::BinaryOp(BinOp::Add, read(a), read(b)))
        };
        let compute = BasicBlock {
            statements: vec![
                // slot = (z * ty + y) * tx + x
                mul(9, 4, u64::from(grid[1])),
                add(9, 9, 3),
                mul(9, 9, u64::from(grid[0])),
                add(7, 9, 2),
                // value = 1 + group_x * 10000 + local_x * 100 + y * 10 + z
                mul(8, 6, 10_000),
                mul(9, 5, 100),
                add(8, 8, 9),
                mul(9, 3, 10),
                add(8, 8, 9),
                add(8, 8, 4),
                assign(8, Rvalue::BinaryOp(BinOp::Add, read(8), cusize(1))),
                assign(
                    10,
                    Rvalue::Cast {
                        to: Ty::U32,
                        operand: read(8),
                    },
                ),
                Statement::Assign(
                    Place {
                        local: l(1),
                        projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(7))],
                    },
                    Rvalue::Use(read(10)),
                ),
            ],
            terminator: Terminator::Return,
        };
        let blocks = vec![
            call(2, IndexAxis::X, 1),
            call(3, IndexAxis::Y, 2),
            call(4, IndexAxis::Z, 3),
            call(5, IndexAxis::LocalX, 4),
            call(6, IndexAxis::GroupX, 5),
            compute,
        ];
        let mut body = Body::new("id_probe", 1, local_decls(tys), blocks);
        body.workgroup_size = workgroup;
        body
    }

    /// The probe run as the grid and run serialized agree on every buffer.
    fn assert_serialization_matches_the_grid(grid: [u32; 3], workgroup: [u32; 3]) {
        let body = id_probe(grid, workgroup);
        let threads = grid.iter().map(|&n| n as usize).product::<usize>();
        let groups = [0, 1, 2].map(|axis| grid[axis] / workgroup[axis]);

        let mut want = [Buffer::from_u32s(&vec![0; threads])];
        run(&body, groups, &mut want).unwrap();
        assert!(
            want[0].to_u32s().unwrap().iter().all(|&v| v != 0),
            "the probe leaves a slot unwritten for {grid:?}/{workgroup:?}"
        );

        let serialized = serialize_grid(&body, Grid { threads: grid }).unwrap();
        assert_eq!(serialized.workgroup_size, [1, 1, 1]);
        let mut got = [Buffer::from_u32s(&vec![0; threads])];
        run(&serialized, [1, 1, 1], &mut got).unwrap();
        assert_eq!(got, want, "grid {grid:?}, workgroup {workgroup:?}");
    }

    /// Tier 1: `eval(serialize(body, grid)) == eval_grid(body, grid)` over every axis shape: a flat grid, a
    /// workgroup larger than one thread, a looped axis between unlooped ones, 2-D and 3-D grids.
    ///
    /// Mutation: make `serialize_grid` read `IdKind::Local` as the global id, or loop the axes in the wrong
    /// order for a 2-D grid (swap `grid.threads[axis]` bounds); the buffers differ.
    #[test]
    fn serializing_a_grid_matches_running_it() {
        for (grid, workgroup) in [
            ([1, 1, 1], [1, 1, 1]),
            ([7, 1, 1], [1, 1, 1]),
            ([8, 1, 1], [4, 1, 1]),
            ([1, 3, 1], [1, 1, 1]),
            ([6, 4, 1], [2, 2, 1]),
            ([4, 3, 2], [2, 1, 1]),
            ([4, 1, 3], [4, 1, 1]),
        ] {
            assert_serialization_matches_the_grid(grid, workgroup);
        }
    }

    fn refusal(body: &Body, grid: [u32; 3]) -> SerializeError {
        serialize_grid(body, Grid { threads: grid }).unwrap_err()
    }

    fn one_block(statements: Vec<Statement>, terminator: Terminator) -> Body {
        let tys = vec![out_buffer(Ty::U32), Ty::U32];
        let mut body = Body::new(
            "k",
            1,
            local_decls(tys),
            vec![
                BasicBlock {
                    statements,
                    terminator,
                },
                BasicBlock {
                    statements: Vec::new(),
                    terminator: Terminator::Return,
                },
            ],
        );
        body.workgroup_size = [1, 1, 1];
        body
    }

    #[test]
    fn a_barrier_is_refused_with_its_block() {
        let body = one_block(
            Vec::new(),
            Terminator::Barrier {
                target: BlockId { index: 1 },
            },
        );
        assert_eq!(
            refusal(&body, [4, 1, 1]),
            SerializeError::Barrier {
                block: BlockId { index: 0 }
            }
        );
    }

    #[test]
    fn workgroup_memory_is_refused() {
        assert_eq!(
            refusal(&wg_sum("k", 64), [64, 1, 1]),
            SerializeError::WorkgroupMemory
        );
    }

    #[test]
    fn a_matrix_fragment_is_refused_as_a_subgroup_collective() {
        assert_eq!(
            refusal(&wmma_tile("k"), [32, 1, 1]),
            SerializeError::Subgroup
        );
    }

    #[test]
    fn a_global_atomic_is_refused() {
        let element = Place {
            local: l(1),
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(2))],
        };
        let body = one_block(
            vec![assign(
                2,
                Rvalue::GlobalAtomic {
                    place: element,
                    value: Operand::Const(Constant::U32(1)),
                    op: AtomicOp::Add,
                },
            )],
            Terminator::Return,
        );
        assert_eq!(
            refusal(&body, [4, 1, 1]),
            SerializeError::Atomic {
                scope: MemoryScope::Device
            }
        );
    }

    #[test]
    fn a_fence_is_refused() {
        let body = one_block(
            vec![Statement::Fence {
                scope: MemoryScope::Device,
                ordering: MemoryOrdering::AcqRel,
            }],
            Terminator::Return,
        );
        assert_eq!(
            refusal(&body, [4, 1, 1]),
            SerializeError::Fence {
                scope: MemoryScope::Device
            }
        );
    }

    #[test]
    fn a_grid_the_workgroups_do_not_tile_is_refused() {
        let body = id_probe([6, 1, 1], [4, 1, 1]);
        assert_eq!(
            refusal(&body, [6, 1, 1]),
            SerializeError::PartialWorkgroup {
                grid: [6, 1, 1],
                workgroup: [4, 1, 1]
            }
        );
        assert_eq!(
            refusal(&body, [0, 1, 1]),
            SerializeError::EmptyGrid { grid: [0, 1, 1] }
        );
    }
}
