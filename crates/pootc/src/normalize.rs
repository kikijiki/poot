//! IR -> IR normalization: bridge the imported MIR shape to the shape `poot-codegen` consumes. Pure
//! `Body -> Body` with no rustc dependency (it could move to a rustc-free crate and be unit-tested off
//! the compiler).
//!
//! Two passes:
//!
//! - Reference-alias copy propagation. For `c: &mut [f32]`, `c.len()` cannot read the param directly: MIR
//!   first takes a shared reborrow (`_7 = &*c; _6 = Len(_7)`), which the importer models as
//!   `_7 = Use(Copy(*c))`. The emitter keys a slice's length off its parameter local, so the alias is
//!   copy-propagated back to its source (`Len(_7)` -> `Len(*c)`, which `emit_len` reads off the root param
//!   `c`) and the dead def is dropped. Only reference/slice-typed, single-assignment copies (the
//!   reborrows) are folded, never f32 value-loads like `_8 = a[i]`, which are unsafe to propagate across
//!   blocks.
//! - Redundant-guard elimination ([`eliminate_redundant_guards`]): `import::import_body` lowers every MIR
//!   `Assert` to a real trap edge (card 531c) instead of silently erasing it, but rustc inserts
//!   one bounds-check `Assert` per slice index regardless of an enclosing guard, so
//!   `if i < c.len() { c[i] = ... }` gets a redundant second `i < c.len()` check on the write. This pass
//!   drops a trap edge only when its condition is a `BinaryOp` provably equal - same operator, and each
//!   operand resolved through a single-assignment chain to the same root value (a `Len` of the same place,
//!   the same untouched local, or the same constant) - to a dominating `SwitchInt`'s condition on the path
//!   that reaches it (never a heuristic: an unprovable or cross-slice check, e.g. `a[i]` guarded only by
//!   `i < c.len()`, is left as a real trap). "On the path that reaches it" is checked two ways, not one:
//!   `dom[b]` always contains `b` itself, so a dominating guard whose true edge targets the redundant
//!   check directly would satisfy a naive dominance test vacuously, saying nothing about whether that
//!   redundant check has some OTHER, unguarded predecessor; `eliminate_redundant_guards` additionally
//!   requires the guard's true edge to have exactly one predecessor (the guard itself) before trusting
//!   dominance through it (card 531c test coverage surfaced this gap directly - see
//!   `leaves_a_guard_with_an_unguarded_extra_predecessor_alone`).

use std::collections::{HashMap, HashSet};

use poot_kernel_ir::{
    BinOp, Body, Constant, Operand, Place, Rvalue, Statement, SwitchTargets, Terminator, Ty,
};

/// Copy-propagate reference/slice reborrow aliases and drop redundant Assert-derived trap edges.
pub fn normalize(mut body: Body) -> Body {
    let aliases = collect_ref_aliases(&body);
    if !aliases.is_empty() {
        for bb in &mut body.blocks {
            bb.statements.retain(|s| !is_alias_def(s, &aliases));
            for s in &mut bb.statements {
                rewrite_statement(s, &aliases);
            }
            rewrite_terminator(&mut bb.terminator, &aliases);
        }
    }
    eliminate_redundant_guards(&mut body);
    body
}

// --- redundant-guard elimination -----------------------------------------------------------------

/// A condition operand's value, resolved through single-assignment locals so two syntactically different
/// locals holding the same value (e.g. two separately computed `Len(c)`s) compare equal. `None` (the
/// caller's `Option`) means "not provably any particular value" - never treated as equal to anything,
/// including itself.
#[derive(Clone, PartialEq)]
enum ValueKey {
    Const(Constant),
    Len(Place),
    /// A local proven single-valued (assigned exactly once, or a parameter never reassigned) whose
    /// definition isn't a `Len` this pass resolves further; identity by local index is safe precisely
    /// because it can hold only ever one value.
    Local(u32),
}

/// Every basic block's successor indices, ignoring edge meaning (used only for dominance).
fn block_successors(t: &Terminator) -> Vec<usize> {
    match t {
        Terminator::Goto { target } | Terminator::ThreadIndexCall { target, .. } => {
            vec![target.index as usize]
        }
        Terminator::Barrier { target } => vec![target.index as usize],
        Terminator::SwitchInt { targets, .. } => {
            let mut v: Vec<usize> = targets
                .branches
                .iter()
                .map(|(_, b)| b.index as usize)
                .collect();
            v.push(targets.otherwise.index as usize);
            v
        }
        Terminator::Return | Terminator::Trap { .. } => vec![],
    }
}

/// `dom[b]`: every block index (including `b`) that dominates `b`, by the standard iterative fixpoint
/// over predecessor sets. Kernel bodies are small (tens of blocks), so the `O(n^2)` fixpoint is cheap.
/// Also returns each block's predecessor list: dominance alone is not enough to prove a guard's true
/// edge is reached only when its condition held (see the "single predecessor" check in
/// [`eliminate_redundant_guards`], which needs it - `dom[b]` always contains `b` itself, so a naive
/// `dom[b].contains(true_a)` check is vacuously true whenever a guard's true edge targets `b` directly,
/// even if `b` has an additional, unguarded predecessor).
fn dominators(body: &Body) -> (Vec<HashSet<usize>>, Vec<Vec<usize>>) {
    let n = body.blocks.len();
    let succs: Vec<Vec<usize>> = body
        .blocks
        .iter()
        .map(|b| block_successors(&b.terminator))
        .collect();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (from, tos) in succs.iter().enumerate() {
        for &to in tos {
            preds[to].push(from);
        }
    }
    let all: HashSet<usize> = (0..n).collect();
    let mut dom: Vec<HashSet<usize>> = vec![all; n];
    if n > 0 {
        dom[0] = [0].into_iter().collect();
    }
    let mut changed = true;
    while changed {
        changed = false;
        for b in 1..n {
            if preds[b].is_empty() {
                continue; // unreachable from entry; never a dominance ancestor of anything reachable
            }
            let mut new_dom = dom[preds[b][0]].clone();
            for &p in &preds[b][1..] {
                new_dom = new_dom.intersection(&dom[p]).copied().collect();
            }
            new_dom.insert(b);
            if new_dom != dom[b] {
                dom[b] = new_dom;
                changed = true;
            }
        }
    }
    (dom, preds)
}

/// Resolve `op` to a [`ValueKey`]: a constant directly, or a bare (unprojected) local through
/// `single_valued`/`len_defs`. Any local not proven single-valued, or a projected place, resolves to
/// `None` rather than an identity that could silently alias a different value.
fn value_key(
    op: &Operand,
    single_valued: &HashSet<u32>,
    len_defs: &HashMap<u32, Place>,
) -> Option<ValueKey> {
    match op {
        Operand::Const(c) => Some(ValueKey::Const(c.clone())),
        Operand::Copy(p) | Operand::Move(p) => {
            if !p.projection.is_empty() || !single_valued.contains(&p.local.index) {
                return None;
            }
            Some(match len_defs.get(&p.local.index) {
                Some(place) => ValueKey::Len(place.clone()),
                None => ValueKey::Local(p.local.index),
            })
        }
    }
}

/// Drop a trap edge whose condition a dominating `SwitchInt` already proved true on every path reaching
/// it. See the module doc for the exact soundness condition.
fn eliminate_redundant_guards(body: &mut Body) {
    let n = body.blocks.len();
    if n == 0 {
        return;
    }
    let (dom, preds) = dominators(body);

    // Body-wide single-assignment analysis: a local is single-valued if it is a parameter never
    // reassigned, or assigned by exactly one `Statement::Assign`/`ThreadIndexCall` destination anywhere.
    let mut assign_count: HashMap<u32, u32> = HashMap::new();
    for bb in &body.blocks {
        for s in &bb.statements {
            if let Statement::Assign(p, _) = s
                && p.projection.is_empty()
            {
                *assign_count.entry(p.local.index).or_insert(0) += 1;
            }
        }
        if let Terminator::ThreadIndexCall { destination, .. } = &bb.terminator
            && destination.projection.is_empty()
        {
            *assign_count.entry(destination.local.index).or_insert(0) += 1;
        }
    }
    let mut single_valued: HashSet<u32> = (1..=body.param_count)
        .filter(|i| !assign_count.contains_key(i))
        .collect();
    single_valued.extend(
        assign_count
            .iter()
            .filter(|&(_, &c)| c == 1)
            .map(|(&l, _)| l),
    );

    let mut len_defs: HashMap<u32, Place> = HashMap::new();
    let mut binop_defs: HashMap<u32, (BinOp, Operand, Operand)> = HashMap::new();
    for bb in &body.blocks {
        for s in &bb.statements {
            let Statement::Assign(p, rv) = s else {
                continue;
            };
            if !p.projection.is_empty() || !single_valued.contains(&p.local.index) {
                continue;
            }
            match rv {
                Rvalue::Len(place) => {
                    len_defs.insert(p.local.index, place.clone());
                }
                Rvalue::BinaryOp(op, a, b) => {
                    binop_defs.insert(p.local.index, (*op, a.clone(), b.clone()));
                }
                _ => {}
            }
        }
    }

    let is_trap = |i: usize| -> bool {
        body.blocks[i].statements.is_empty()
            && matches!(body.blocks[i].terminator, Terminator::Trap { .. })
    };
    // Every `SwitchInt` in the canonical `[(0, _)], otherwise: _` shape: (discr, false_target, true_target).
    let canonical: Vec<Option<(Operand, usize, usize)>> = body
        .blocks
        .iter()
        .map(|bb| match &bb.terminator {
            Terminator::SwitchInt { discr, targets }
                if targets.branches.len() == 1 && targets.branches[0].0 == 0 =>
            {
                Some((
                    discr.clone(),
                    targets.branches[0].1.index as usize,
                    targets.otherwise.index as usize,
                ))
            }
            _ => None,
        })
        .collect();

    let resolve_cond = |discr: &Operand| -> Option<(BinOp, Operand, Operand)> {
        let (Operand::Copy(p) | Operand::Move(p)) = discr else {
            return None;
        };
        if !p.projection.is_empty() {
            return None;
        }
        binop_defs.get(&p.local.index).cloned()
    };
    let cond_eq = |a: &(BinOp, Operand, Operand), b: &(BinOp, Operand, Operand)| -> bool {
        a.0 == b.0
            && value_key(&a.1, &single_valued, &len_defs).is_some_and(|ka| {
                value_key(&b.1, &single_valued, &len_defs).is_some_and(|kb| ka == kb)
            })
            && value_key(&a.2, &single_valued, &len_defs).is_some_and(|ka| {
                value_key(&b.2, &single_valued, &len_defs).is_some_and(|kb| ka == kb)
            })
    };

    let mut rewrites: Vec<(usize, usize)> = Vec::new(); // (block, new goto target)
    for (b, entry) in canonical.iter().enumerate() {
        let Some((discr_b, false_b, true_b)) = entry else {
            continue;
        };
        // Only an Assert-derived guard (one edge a bare trap block) is a candidate to eliminate.
        let (trap_edge, success_edge) = if is_trap(*false_b) && !is_trap(*true_b) {
            (*false_b, *true_b)
        } else if is_trap(*true_b) && !is_trap(*false_b) {
            (*true_b, *false_b)
        } else {
            continue;
        };
        let _ = trap_edge;
        let Some(cond_b) = resolve_cond(discr_b) else {
            continue;
        };
        for (a, other) in canonical.iter().enumerate() {
            if a == b {
                continue;
            }
            let Some((discr_a, _, true_a)) = other else {
                continue;
            };
            // `dom[b]` always contains `b` itself, so `dom[b].contains(true_a)` alone would be vacuously
            // true whenever `a`'s true edge targets `b` directly - it says nothing about whether `b` (or,
            // for a longer chain, `true_a`) has some OTHER, unguarded predecessor that reaches it without
            // `cond_a` ever being evaluated. `true_a` must be reachable only via `a`'s true edge (a single
            // predecessor, exactly `a`) for reaching it to prove `cond_a` held; dominance of `b` by
            // `true_a` then carries that proof forward to `b` itself.
            if preds[*true_a] != [a] {
                continue;
            }
            if !dom[b].contains(true_a) {
                continue;
            }
            let Some(cond_a) = resolve_cond(discr_a) else {
                continue;
            };
            if cond_eq(&cond_a, &cond_b) {
                rewrites.push((b, success_edge));
                break;
            }
        }
    }
    for (b, target) in rewrites {
        body.blocks[b].terminator = Terminator::Goto {
            target: poot_kernel_ir::BlockId {
                index: target as u32,
            },
        };
    }
}

/// `X = Use(Copy/Move(P))` where `X` is a single-assignment, reference/slice-typed local with no LHS
/// projection. Maps `X -> P`.
fn collect_ref_aliases(body: &Body) -> HashMap<u32, Place> {
    // Count assignments per local (LHS, empty projection) so only single-assignment locals are folded.
    let mut assign_count: HashMap<u32, u32> = HashMap::new();
    for bb in &body.blocks {
        for s in &bb.statements {
            if let Statement::Assign(p, _) = s
                && p.projection.is_empty()
            {
                *assign_count.entry(p.local.index).or_insert(0) += 1;
            }
        }
    }
    let mut aliases: HashMap<u32, Place> = HashMap::new();
    for bb in &body.blocks {
        for s in &bb.statements {
            let Statement::Assign(lhs, Rvalue::Use(op)) = s else {
                continue;
            };
            if !lhs.projection.is_empty() {
                continue;
            }
            let is_ref = matches!(
                body.locals[lhs.local.index as usize].ty,
                Ty::Ref { .. } | Ty::Slice(_)
            );
            if !is_ref || assign_count.get(&lhs.local.index) != Some(&1) {
                continue;
            }
            let src = match op {
                Operand::Copy(p) | Operand::Move(p) => p.clone(),
                Operand::Const(_) => continue,
            };
            aliases.insert(lhs.local.index, src);
        }
    }
    aliases
}

fn is_alias_def(s: &Statement, aliases: &HashMap<u32, Place>) -> bool {
    matches!(s, Statement::Assign(p, _) if p.projection.is_empty() && aliases.contains_key(&p.local.index))
}

/// Resolve a place whose root may be an alias: substitute the alias source and prepend its projection,
/// transitively (bounded, aliases form a DAG).
fn rewrite_place(p: &mut Place, aliases: &HashMap<u32, Place>) {
    let mut guard = 0;
    while let Some(src) = aliases.get(&p.local.index) {
        let mut proj = src.projection.clone();
        proj.append(&mut p.projection);
        p.local = src.local;
        p.projection = proj;
        guard += 1;
        if guard > 64 {
            break; // alias cycle (shouldn't happen); stop rather than loop forever.
        }
    }
}

fn rewrite_operand(op: &mut Operand, aliases: &HashMap<u32, Place>) {
    match op {
        Operand::Copy(p) | Operand::Move(p) => rewrite_place(p, aliases),
        Operand::Const(_) => {}
    }
}

fn rewrite_rvalue(rv: &mut Rvalue, aliases: &HashMap<u32, Place>) {
    match rv {
        Rvalue::Use(op)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::MathUnary(_, op)
        | Rvalue::IntScalarUnary(_, op)
        | Rvalue::Cast { operand: op, .. }
        | Rvalue::Bitcast { operand: op, .. }
        | Rvalue::Fp8Decode { operand: op, .. }
        | Rvalue::Fp8Encode { operand: op, .. } => rewrite_operand(op, aliases),
        // Card 628: hand-built in kernelgen (like the vector ops below), never produced by MIR lowering;
        // rewritten for completeness.
        Rvalue::BinaryOp(_, a, b) | Rvalue::BinaryOpNoContract(_, a, b) => {
            rewrite_operand(a, aliases);
            rewrite_operand(b, aliases);
        }
        Rvalue::Len(p) => rewrite_place(p, aliases),
        Rvalue::WorkgroupLocalRead { idx, .. } => rewrite_operand(idx, aliases),
        Rvalue::WorkgroupLocalAtomic { idx, value, .. } => {
            rewrite_operand(idx, aliases);
            rewrite_operand(value, aliases);
        }
        Rvalue::GlobalAtomic { place, value, .. } => {
            rewrite_place(place, aliases);
            rewrite_operand(value, aliases);
        }
        Rvalue::WorkgroupLocalCompareExchange {
            idx,
            expected,
            desired,
            ..
        } => {
            rewrite_operand(idx, aliases);
            rewrite_operand(expected, aliases);
            rewrite_operand(desired, aliases);
        }
        Rvalue::GlobalCompareExchange {
            place,
            expected,
            desired,
        } => {
            rewrite_place(place, aliases);
            rewrite_operand(expected, aliases);
            rewrite_operand(desired, aliases);
        }
        // Spec 134 P1: vector ops are hand-built in kernelgen (like WMMA), never produced by MIR lowering; rewritten for completeness.
        Rvalue::VectorLoad { place } => rewrite_place(place, aliases),
        Rvalue::VectorSplat(op) => rewrite_operand(op, aliases),
    }
}

fn rewrite_statement(s: &mut Statement, aliases: &HashMap<u32, Place>) {
    match s {
        Statement::Assign(p, rv) => {
            rewrite_place(p, aliases);
            rewrite_rvalue(rv, aliases);
        }
        Statement::WorkgroupLocalWrite { idx, value, .. } => {
            rewrite_operand(idx, aliases);
            rewrite_operand(value, aliases);
        }
        Statement::VectorStore { place, value } => {
            rewrite_place(place, aliases);
            rewrite_operand(value, aliases);
        }
        Statement::StorageLive(l) | Statement::StorageDead(l) => {
            // A storage marker on an alias local is harmless; leave it.
            let _ = l;
        }
        // WMMA (spec 025) is hand-built in kernelgen, never produced by MIR lowering; the tile place is rewritten for completeness (fragment locals are not aliases).
        Statement::WmmaLoad { tile, .. } | Statement::WmmaStore { tile, .. } => {
            rewrite_place(tile, aliases)
        }
        Statement::WmmaMma { .. }
        | Statement::WmmaStoreLds { .. }
        | Statement::WmmaLoadLds { .. }
        | Statement::WmmaZero { .. } => {}
        // Spec 138 Phase 2: a Fence has no operand or place to rewrite; like WMMA/CAS it is hand-built in kernelgen.
        Statement::Fence { .. } => {}
    }
}

fn rewrite_terminator(t: &mut Terminator, aliases: &HashMap<u32, Place>) {
    match t {
        Terminator::SwitchInt { discr, targets } => {
            rewrite_operand(discr, aliases);
            let _: &mut SwitchTargets = targets;
        }
        Terminator::ThreadIndexCall { destination, .. } => rewrite_place(destination, aliases),
        Terminator::Goto { .. }
        | Terminator::Barrier { .. }
        | Terminator::Return
        | Terminator::Trap { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_kernel_ir::{BasicBlock, BlockId, Local, LocalDecl, ProjectionElem};

    fn local(i: u32) -> Local {
        Local { index: i }
    }

    fn block(statements: Vec<Statement>, terminator: Terminator) -> BasicBlock {
        BasicBlock {
            statements,
            terminator,
        }
    }

    fn bid(i: u32) -> BlockId {
        BlockId { index: i }
    }

    fn usize_local(mutable: bool) -> LocalDecl {
        LocalDecl {
            ty: Ty::Usize,
            mutable,
        }
    }

    fn bool_local() -> LocalDecl {
        LocalDecl {
            ty: Ty::Bool,
            mutable: true,
        }
    }

    fn slice_param() -> LocalDecl {
        LocalDecl {
            ty: Ty::Ref {
                mutable: false,
                pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
            },
            mutable: false,
        }
    }

    /// `i < len` as a fresh pair of statements assigning `len_dst := Len(slice)` then
    /// `cond_dst := BinaryOp(Lt, Copy(i), Copy(len_dst))`, mirroring rustc inserting an independent
    /// `Len` + comparison for each bounds check even when an outer guard already computed the same
    /// thing (the redundancy [`eliminate_redundant_guards`] is for).
    fn lt_len_guard(cond_dst: u32, len_dst: u32, i: u32, slice: u32) -> Vec<Statement> {
        vec![
            Statement::Assign(
                Place::local(local(len_dst)),
                Rvalue::Len(Place::local(local(slice))),
            ),
            Statement::Assign(
                Place::local(local(cond_dst)),
                Rvalue::BinaryOp(
                    BinOp::Lt,
                    Operand::Copy(Place::local(local(i))),
                    Operand::Copy(Place::local(local(len_dst))),
                ),
            ),
        ]
    }

    /// The canonical Assert-lowering shape (card 531c `import.rs`, `expected == true`): `cond_local`
    /// false (0) traps, true (otherwise) reaches `target`.
    fn switch_guard(cond_local: u32, trap: u32, target: u32) -> Terminator {
        Terminator::SwitchInt {
            discr: Operand::Copy(Place::local(local(cond_local))),
            targets: SwitchTargets {
                branches: vec![(0, bid(trap))],
                otherwise: bid(target),
            },
        }
    }

    /// A provably redundant guard (same slice, same operator, two independently computed `Len`s of the
    /// same place) directly inside a dominating `if i < c.len() { ... }`: block 0 is the outer guard,
    /// block 2 (reached only via block 0's true edge) recomputes the identical check before block 2's
    /// own trap. `eliminate_redundant_guards` must collapse block 2's `SwitchInt` to `Goto { target: 4 }`.
    #[test]
    fn eliminates_a_provably_redundant_guard() {
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            }, // _0
            usize_local(false), // _1 param: i
            slice_param(),      // _2 param: c
            usize_local(true),  // _3: Len(c) (outer)
            bool_local(),       // _4: i < _3 (outer cond)
            usize_local(true),  // _5: Len(c) (inner, separately computed)
            bool_local(),       // _6: i < _5 (inner cond)
        ];
        let mut body = Body::new(
            "k",
            2,
            locals,
            vec![
                block(lt_len_guard(4, 3, 1, 2), switch_guard(4, 1, 2)),
                block(vec![], Terminator::Return),
                block(lt_len_guard(6, 5, 1, 2), switch_guard(6, 3, 4)),
                block(vec![], Terminator::Trap { code: 1 }),
                block(vec![], Terminator::Return),
            ],
        );
        eliminate_redundant_guards(&mut body);
        assert_eq!(
            body.blocks[2].terminator,
            Terminator::Goto { target: bid(4) },
            "block 2's redundant guard must collapse to the success edge"
        );
        // Nothing else changes.
        assert_eq!(body.blocks[0].terminator, switch_guard(4, 1, 2));
        assert_eq!(body.blocks[3].terminator, Terminator::Trap { code: 1 });
    }

    /// Same shape as `eliminates_a_provably_redundant_guard`, but block 2's guard checks a DIFFERENT
    /// slice (`d`, param `_3`) than block 0's (`c`, param `_2`): `Len(_2) != Len(_3)` as a `ValueKey`
    /// (different `Place`), so the guard is not provably redundant and must be left as a real trap. This
    /// is the exact failure mode the module doc warns about: "two Len's of different places colliding".
    #[test]
    fn leaves_a_guard_over_a_different_slice_alone() {
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            }, // _0
            usize_local(false), // _1 param: i
            slice_param(),      // _2 param: c
            slice_param(),      // _3 param: d (a DIFFERENT slice)
            usize_local(true),  // _4: Len(c)
            bool_local(),       // _5: i < _4
            usize_local(true),  // _6: Len(d)
            bool_local(),       // _7: i < _6
        ];
        let mut body = Body::new(
            "k",
            3,
            locals,
            vec![
                block(lt_len_guard(5, 4, 1, 2), switch_guard(5, 1, 2)),
                block(vec![], Terminator::Return),
                block(lt_len_guard(7, 6, 1, 3), switch_guard(7, 3, 4)),
                block(vec![], Terminator::Trap { code: 1 }),
                block(vec![], Terminator::Return),
            ],
        );
        let before = body.blocks[2].terminator.clone();
        eliminate_redundant_guards(&mut body);
        assert_eq!(
            body.blocks[2].terminator, before,
            "a guard over a different slice must not be eliminated"
        );
    }

    /// Same guard/condition as `eliminates_a_provably_redundant_guard`, but block 2 (the inner guard) has
    /// a SECOND predecessor, block 1, which jumps straight into it without ever evaluating block 0's
    /// condition (block 1 is itself only reached via block 0's FALSE edge, i.e. exactly when `i < c.len()`
    /// is false). `dom[2]` trivially contains block 2 itself regardless of this extra edge (a block always
    /// dominates itself), so a naive `dom[b].contains(true_a)` check alone cannot see the problem; the
    /// single-predecessor check on `true_a` (`preds[true_a] == [a]`) is what must refuse this. If it did
    /// not, an out-of-range `i` reaching block 2 via block 0's false edge -> block 1 -> block 2 would skip
    /// its own bounds check entirely and fall through to the "safe" block 4 - reintroducing R468-007's
    /// silent erasure for this shape.
    #[test]
    fn leaves_a_guard_with_an_unguarded_extra_predecessor_alone() {
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            }, // _0
            usize_local(false), // _1 param: i
            slice_param(),      // _2 param: c
            usize_local(true),  // _3: Len(c) (outer)
            bool_local(),       // _4: i < _3 (outer cond)
            usize_local(true),  // _5: Len(c) (inner, separately computed)
            bool_local(),       // _6: i < _5 (inner cond)
        ];
        let mut body = Body::new(
            "k",
            2,
            locals,
            vec![
                block(lt_len_guard(4, 3, 1, 2), switch_guard(4, 1, 2)),
                block(vec![], Terminator::Goto { target: bid(2) }), // bypasses the condition entirely
                block(lt_len_guard(6, 5, 1, 2), switch_guard(6, 3, 4)),
                block(vec![], Terminator::Trap { code: 1 }),
                block(vec![], Terminator::Return),
            ],
        );
        let before = body.blocks[2].terminator.clone();
        eliminate_redundant_guards(&mut body);
        assert_eq!(
            body.blocks[2].terminator, before,
            "a guard reachable through an unguarded extra predecessor must not be eliminated"
        );
    }

    #[test]
    fn folds_ref_reborrow_alias() {
        // _1: &mut [f32] (param). _4 = &*_1 (reborrow alias), _5 = Len(_4). After normalize the alias
        // def is gone and Len reads the param place {_1,[Deref]}.
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            }, // _0
            LocalDecl {
                ty: Ty::Ref {
                    mutable: true,
                    pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
                },
                mutable: false,
            }, // _1 param
            LocalDecl {
                ty: Ty::Usize,
                mutable: true,
            }, // _2 dummy
            LocalDecl {
                ty: Ty::Usize,
                mutable: true,
            }, // _3 result len
            LocalDecl {
                ty: Ty::Ref {
                    mutable: false,
                    pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
                },
                mutable: true,
            }, // _4 alias
        ];
        let deref_of_1 = Place {
            local: local(1),
            projection: vec![ProjectionElem::Deref],
        };
        let stmts = vec![
            Statement::Assign(
                Place::local(local(4)),
                Rvalue::Use(Operand::Copy(deref_of_1.clone())),
            ),
            Statement::Assign(Place::local(local(3)), Rvalue::Len(Place::local(local(4)))),
        ];
        let body = Body::new(
            "k",
            1,
            locals,
            vec![BasicBlock {
                statements: stmts,
                terminator: Terminator::Return,
            }],
        );
        let n = normalize(body);
        let block = &n.blocks[0];
        // The alias def (_4 = ...) is removed; only the Len assign remains.
        assert_eq!(block.statements.len(), 1, "alias def should be dropped");
        match &block.statements[0] {
            Statement::Assign(p, Rvalue::Len(src)) => {
                assert_eq!(p.local, local(3));
                assert_eq!(
                    src.local,
                    local(1),
                    "Len must key off the param, not the alias"
                );
                assert_eq!(src.projection, vec![ProjectionElem::Deref]);
            }
            other => panic!("unexpected stmt: {other:?}"),
        }
    }

    #[test]
    fn leaves_value_loads_alone() {
        // _3 = a[i] is an f32 value-load, not a ref alias, so it must not be folded.
        let locals = vec![
            LocalDecl {
                ty: Ty::Unit,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Ref {
                    mutable: false,
                    pointee: Box::new(Ty::Slice(Box::new(Ty::F32))),
                },
                mutable: false,
            },
            LocalDecl {
                ty: Ty::Usize,
                mutable: false,
            },
            LocalDecl {
                ty: Ty::F32,
                mutable: true,
            },
        ];
        let elem = Place {
            local: local(1),
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(local(2))],
        };
        let stmts = vec![Statement::Assign(
            Place::local(local(3)),
            Rvalue::Use(Operand::Copy(elem)),
        )];
        let body = Body::new(
            "k",
            1,
            locals,
            vec![BasicBlock {
                statements: stmts,
                terminator: Terminator::Return,
            }],
        );
        let n = normalize(body);
        // _3 is F32, not a ref: no fold, statement preserved.
        assert_eq!(n.blocks[0].statements.len(), 1);
    }
}
