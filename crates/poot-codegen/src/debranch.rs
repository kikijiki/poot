//! SpirvVulkan-only: remove `Assert`/`Unreachable`-derived `Trap` branching before emission (card 531c,
//! R468-007). Vulkan compute SPIR-V has no trap instruction, so instead of branching to a dedicated trap
//! block, every failing `Assert` accumulates a per-thread bitmask of which trap site fired (one bit per
//! site, up to 32 distinct sites; beyond that, sites alias onto bit 31 - still detected as *a* fault,
//! just not distinguishable from each other) and execution continues; the accumulator is OR-ed into the
//! reserved per-dispatch error word right before every `Return`. This adds no new control-flow edge, so
//! `crate::structurize` (which runs after this pass) never has to reason about a selection with a trap
//! arm.
//!
//! Out-of-bounds accesses the skipped `Assert` would have guarded against are defined under Vulkan robust
//! buffer access, so continuing past a failed assert is sound: reads return an unspecified in-bounds-typed
//! value and writes are dropped or redirected, never memory-unsafe. Verified, not assumed (card 531c):
//! `wgpu-hal`'s Vulkan backend (`vulkan/adapter.rs`, `PhysicalDeviceFeatures::core`) requests
//! `robustBufferAccess` in every logical device it creates whenever the adapter advertises it, unconditionally -
//! it is not gated behind `wgpu::Features` the way most features are, so `poot-runtime` needs no opt-in for
//! it. `vulkaninfo` on this box's RADV/gfx1151 confirms the adapter advertises
//! `robustBufferAccess = true` (and `robustBufferAccess2`/`robustImageAccess2 = true`, unused here). The
//! executor discards the whole dispatch's result and raises a typed fault when it observes a nonzero error
//! word at its own sync point - never mid-dispatch, and never by inspecting the (possibly garbage) data
//! the faulting lane produced.
//!
//! ROCm and PTX are unaffected: `emit::core::emit_terminator`'s `Terminator::Trap` arm still lowers to a
//! real device trap for them, and only SpirvVulkan calls this pass (see `emit::emit_llvm_ir`).

use poot_kernel_ir::{
    BinOp, BlockId, Body, Constant, Local, LocalDecl, Operand, Place, Rvalue, Statement,
    Terminator, Ty, UnOp,
};

/// `id`'s block if it is "trap-shaped" (no statements, terminator `Trap`): its code, or `None`.
fn trap_code(body: &Body, id: BlockId) -> Option<u32> {
    let block = &body.blocks[id.index as usize];
    if !block.statements.is_empty() {
        return None;
    }
    match block.terminator {
        Terminator::Trap { code } => Some(code),
        _ => None,
    }
}

/// Rewrite every Assert-derived trap edge into branchless accumulation, and every remaining `Trap`
/// terminator (an `Unreachable` site, or an Assert's now-orphaned trap block) into an unconditional
/// accumulate-then-return. Returns the accumulator local the body now needs to have flushed into the
/// error word at every `Return` (see `emit::core::emit_terminator`'s `Terminator::Return` arm), or `None`
/// when the body has no `Trap` at all (no change).
pub(crate) fn debranch_traps_for_spirv(body: &mut Body) -> Option<Local> {
    if !body.has_trap() {
        return None;
    }
    let accum = Local {
        index: body.locals.len() as u32,
    };
    body.locals.push(LocalDecl {
        ty: Ty::U32,
        mutable: true,
    });
    // Zero-initialize at the very start of the entry block (index 0 by convention), before any use or
    // any path that could reach a use - `Body::verify`'s definite-assignment check relies on this.
    body.blocks[0].statements.insert(
        0,
        Statement::Assign(
            Place::local(accum),
            Rvalue::Use(Operand::Const(Constant::U32(0))),
        ),
    );

    // Pass 1: an Assert-derived guard is `SwitchInt { branches: [(0, X)], otherwise: Y }` where exactly
    // one of X/Y is trap-shaped. Replace the branch with the accumulate statements and a plain `Goto` to
    // the real (non-trap) target. The trap block itself is untouched here; pass 2 below cleans up every
    // remaining `Trap` terminator uniformly, whether or not anything still points to it.
    let n = body.blocks.len();
    for b in 0..n {
        let (discr, zero_branch, otherwise) = match &body.blocks[b].terminator {
            Terminator::SwitchInt { discr, targets }
                if targets.branches.len() == 1 && targets.branches[0].0 == 0 =>
            {
                (discr.clone(), targets.branches[0].1, targets.otherwise)
            }
            _ => continue,
        };
        let (trap_is_zero_branch, real_target, code) =
            match (trap_code(body, zero_branch), trap_code(body, otherwise)) {
                (Some(code), None) => (true, otherwise, code),
                (None, Some(code)) => (false, zero_branch, code),
                // Neither/both trap-shaped: an ordinary two-way `if`, not an Assert guard. Leave it for
                // `structurize` (unaffected by this pass) to handle as usual.
                _ => continue,
            };
        let shift = (code - 1).min(31);

        // failed = discr negated (trap is the "false" branch) or discr as-is (trap is "otherwise"/true).
        let failed = Local {
            index: body.locals.len() as u32,
        };
        body.locals.push(LocalDecl {
            ty: Ty::Bool,
            mutable: false,
        });
        let failed_stmt = if trap_is_zero_branch {
            Statement::Assign(Place::local(failed), Rvalue::UnaryOp(UnOp::Not, discr))
        } else {
            Statement::Assign(Place::local(failed), Rvalue::Use(discr))
        };

        let failed_u32 = Local {
            index: body.locals.len() as u32,
        };
        body.locals.push(LocalDecl {
            ty: Ty::U32,
            mutable: false,
        });
        let cast_stmt = Statement::Assign(
            Place::local(failed_u32),
            Rvalue::Cast {
                to: Ty::U32,
                operand: Operand::Copy(Place::local(failed)),
            },
        );

        let contribution = Local {
            index: body.locals.len() as u32,
        };
        body.locals.push(LocalDecl {
            ty: Ty::U32,
            mutable: false,
        });
        let shift_stmt = Statement::Assign(
            Place::local(contribution),
            Rvalue::BinaryOp(
                BinOp::Shl,
                Operand::Copy(Place::local(failed_u32)),
                Operand::Const(Constant::U32(shift)),
            ),
        );

        let or_stmt = Statement::Assign(
            Place::local(accum),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                Operand::Copy(Place::local(accum)),
                Operand::Copy(Place::local(contribution)),
            ),
        );

        let bb = &mut body.blocks[b];
        bb.statements
            .extend([failed_stmt, cast_stmt, shift_stmt, or_stmt]);
        bb.terminator = Terminator::Goto {
            target: real_target,
        };
    }

    // Pass 2: any block still terminated by `Trap` (a genuine `Unreachable` site, or now an orphaned
    // former Assert-trap target nothing points to any more) unconditionally ORs its own bit into the
    // accumulator and returns. Sound either way: unreachable, this is dead code that never runs; reached
    // (a real `Unreachable`), it records the fault exactly like a failed `Assert` does.
    for bb in &mut body.blocks {
        let Terminator::Trap { code } = bb.terminator else {
            continue;
        };
        let shift = (code - 1).min(31);
        bb.statements.push(Statement::Assign(
            Place::local(accum),
            Rvalue::BinaryOp(
                BinOp::BitOr,
                Operand::Copy(Place::local(accum)),
                Operand::Const(Constant::U32(1u32 << shift)),
            ),
        ));
        bb.terminator = Terminator::Return;
    }

    Some(accum)
}

#[cfg(test)]
mod tests {
    use poot_kernel_ir::{BasicBlock, SwitchTargets};

    use super::*;

    fn local(i: u32) -> Local {
        Local { index: i }
    }

    fn block(statements: Vec<Statement>, terminator: Terminator) -> BasicBlock {
        BasicBlock {
            statements,
            terminator,
        }
    }

    /// `_0: Unit, _1: Bool` (the "cond" a guard reads), `param_count = 0`: bodies below add whichever
    /// guard/trap/target blocks a test needs on top of this.
    fn base_locals() -> Vec<LocalDecl> {
        vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Bool,
                mutable: true,
            },
        ]
    }

    fn cond_operand() -> Operand {
        Operand::Copy(Place::local(local(1)))
    }

    /// `expected == true` shape (import.rs's Assert lowering when the MIR `expected` flag is true): the
    /// "false" (zero) branch is the trap, "otherwise" (true) is the real target - block 0 the guard, block
    /// 1 the trap (code 1), block 2 the real target (`Return`).
    fn expected_true_body() -> Body {
        Body::new(
            "k",
            0,
            base_locals(),
            vec![
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: cond_operand(),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 1 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                block(vec![], Terminator::Trap { code: 1 }),
                block(vec![], Terminator::Return),
            ],
        )
    }

    #[test]
    fn no_trap_is_a_no_op() {
        let mut b = Body::new(
            "k",
            0,
            base_locals(),
            vec![block(vec![], Terminator::Return)],
        );
        let before = b.clone();
        assert!(debranch_traps_for_spirv(&mut b).is_none());
        assert_eq!(b, before);
    }

    #[test]
    fn assert_guard_becomes_branchless_and_the_trap_becomes_return() {
        let mut b = expected_true_body();
        let accum = debranch_traps_for_spirv(&mut b).expect("body has a Trap");
        assert_eq!(accum, local(2), "accum is the first local appended");
        assert_eq!(
            b.locals.len(),
            6,
            "accum + failed + failed_u32 + contribution"
        );
        assert_eq!(b.locals[accum.index as usize].ty, Ty::U32);

        // The guard block no longer branches: it falls straight through to the real target (2).
        assert_eq!(
            b.blocks[0].terminator,
            Terminator::Goto {
                target: BlockId { index: 2 }
            }
        );
        // accum := 0 is the first statement (before any use), then the four accumulate statements.
        assert_eq!(b.blocks[0].statements.len(), 5);
        assert_eq!(
            b.blocks[0].statements[0],
            Statement::Assign(
                Place::local(accum),
                Rvalue::Use(Operand::Const(Constant::U32(0)))
            )
        );

        // The former trap block is now dead code, but still well-formed: it unconditionally records its
        // own bit and returns (never left as a bare `Trap`, which SpirvVulkan emission cannot lower).
        assert_eq!(b.blocks[1].terminator, Terminator::Return);
        assert!(
            b.blocks[1]
                .statements
                .iter()
                .any(|s| matches!(s, Statement::Assign(p, _) if *p == Place::local(accum)))
        );

        // The real target is untouched.
        assert_eq!(b.blocks[2].terminator, Terminator::Return);
        assert!(b.blocks[2].statements.is_empty());
    }

    /// The other MIR shape (`expected == false`): the trap is "otherwise" (the true branch), the real
    /// target is the "false"/zero branch. Same rewrite, mirrored - this is the case a `!discr`-vs-`discr`
    /// sign error would silently invert (a mutation-provable claim: swap which arm the pass treats as
    /// "failed" and this body's guard would fall through on success instead of on failure).
    #[test]
    fn expected_false_shape_is_handled_symmetrically() {
        let mut b = Body::new(
            "k",
            0,
            base_locals(),
            vec![
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: cond_operand(),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })], // false -> real target
                            otherwise: BlockId { index: 1 },           // true -> trap
                        },
                    },
                ),
                block(vec![], Terminator::Trap { code: 1 }),
                block(vec![], Terminator::Return),
            ],
        );
        debranch_traps_for_spirv(&mut b).expect("body has a Trap");
        assert_eq!(
            b.blocks[0].terminator,
            Terminator::Goto {
                target: BlockId { index: 2 }
            }
        );
        // `discr` feeds the cast directly (no `Not`): the "failed" bit is one iff the switch's true arm -
        // the trap - was the one that would have fired.
        let failed_stmt = &b.blocks[0].statements[1];
        assert!(matches!(
            failed_stmt,
            Statement::Assign(_, Rvalue::Use(Operand::Copy(_)))
        ));
    }

    #[test]
    fn an_unreachable_terminator_becomes_unconditional_accumulate_then_return() {
        // No SwitchInt at all: `Unreachable` substitutes `Trap` directly into its own block (import.rs's
        // design), so this body's sole block IS the trap.
        let mut b = Body::new(
            "k",
            0,
            base_locals(),
            vec![block(vec![], Terminator::Trap { code: 1 })],
        );
        let accum = debranch_traps_for_spirv(&mut b).expect("body has a Trap");
        assert_eq!(b.blocks[0].terminator, Terminator::Return);
        assert_eq!(
            b.blocks[0].statements.last(),
            Some(&Statement::Assign(
                Place::local(accum),
                Rvalue::BinaryOp(
                    BinOp::BitOr,
                    Operand::Copy(Place::local(accum)),
                    Operand::Const(Constant::U32(1)),
                ),
            ))
        );
    }

    #[test]
    fn two_asserts_in_one_thread_both_contribute_distinct_bits() {
        // block0 guard (code 1, trap=1) -> block2 guard (code 2, trap=3) -> block4 real work.
        let mut b = Body::new(
            "k",
            0,
            base_locals(),
            vec![
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: cond_operand(),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 1 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                block(vec![], Terminator::Trap { code: 1 }),
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: cond_operand(),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 4 },
                        },
                    },
                ),
                block(vec![], Terminator::Trap { code: 2 }),
                block(vec![], Terminator::Return),
            ],
        );
        debranch_traps_for_spirv(&mut b).expect("body has traps");
        // Both guards fall through, in order, to the real work block.
        assert_eq!(
            b.blocks[0].terminator,
            Terminator::Goto {
                target: BlockId { index: 2 }
            }
        );
        assert_eq!(
            b.blocks[2].terminator,
            Terminator::Goto {
                target: BlockId { index: 4 }
            }
        );
        // The second guard's OR reads the first guard's local (the BitOr's lhs is `accum`, the same
        // local both blocks share): the accumulation composes across sites in one thread.
        let second_or = b.blocks[2].statements.last().unwrap();
        let Statement::Assign(dest, Rvalue::BinaryOp(BinOp::BitOr, lhs, _)) = second_or else {
            panic!("expected the final OR statement, got {second_or:?}");
        };
        let Operand::Copy(lhs_place) = lhs else {
            panic!("expected accum to be read by Copy");
        };
        assert_eq!(dest, lhs_place, "OR accumulates in place: dest == lhs");
    }
}
