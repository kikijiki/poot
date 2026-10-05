//! Control-flow structurization on the kernel IR `Body`, run once before emission (card 100 / spec 058).
//!
//! SPIR-V requires structured control flow: every selection's merge block must be structurally dominated
//! by the selection header. poot emits LLVM IR with plain branches and lets `llc`'s SPIR-V backend rebuild
//! merge blocks, which it mis-places when two nested selections share one (the RADV "merge block not
//! structurally dominated" SIGSEGV). Spec 058 showed the fix is a CFG-topology change: give each selection
//! a dedicated merge block.
//!
//! This pass finds every 2-way selection whose join block is not dominated by the selection header (the
//! join is shared with an enclosing construct) and redirects the edges from inside the selection's region
//! that reach the shared join to a fresh block that gotos the join (`P -> M` becomes `P -> M' -> M`). It
//! only inserts goto-forwarding blocks, so it cannot change kernel results, only the CFG shape `llc`
//! structurizes. Loops are left alone (their headers have back-edges).
//!
//! Redirecting alone only works when every path from the header into the join starts inside the region.
//! `if a && b { .. } else { .. }` breaks that: MIR shares one else block between the two conditions, so
//! the inner condition's arm escapes its own dominance region and the join is still entered from outside
//! it no matter how often the edges are redirected - the pass used to append one forwarding block per
//! round and never stop (card 461, the depth-4 `&&` spin). Those region-external tails are cloned for the
//! escaping selection first ([`escape_tail`] / [`duplicate_escape_tail`]), after which the redirect
//! converges. The whole loop is bounded by a block budget derived from the body's size on entry, so an
//! input this pass cannot make sense of fails with a typed error instead of hanging the build.

use poot_kernel_ir::{BasicBlock, BlockId, Body, Terminator};

/// Successor block indices of a terminator.
fn successors(t: &Terminator) -> Vec<usize> {
    match t {
        Terminator::Goto { target }
        | Terminator::ThreadIndexCall { target, .. }
        | Terminator::Barrier { target } => vec![target.index as usize],
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

/// Replace every successor edge equal to `from` with `to` in a terminator.
fn replace_target(t: &mut Terminator, from: usize, to: usize) {
    let repl = |b: &mut BlockId| {
        if b.index as usize == from {
            b.index = to as u32;
        }
    };
    match t {
        Terminator::Goto { target }
        | Terminator::ThreadIndexCall { target, .. }
        | Terminator::Barrier { target } => repl(target),
        Terminator::SwitchInt { targets, .. } => {
            for (_, b) in targets.branches.iter_mut() {
                repl(b);
            }
            repl(&mut targets.otherwise);
        }
        Terminator::Return | Terminator::Trap { .. } => {}
    }
}

/// Reverse-postorder of the CFG reachable from `entry`, over the given successor lists.
fn reverse_postorder(n: usize, succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let mut visited = vec![false; n];
    let mut post = Vec::new();
    // Iterative DFS; stack entries are (node, next-child-index).
    let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
    visited[entry] = true;
    while let Some((node, ci)) = stack.pop() {
        if ci < succs[node].len() {
            stack.push((node, ci + 1));
            let s = succs[node][ci];
            if !visited[s] {
                visited[s] = true;
                stack.push((s, 0));
            }
        } else {
            post.push(node);
        }
    }
    post.reverse();
    post
}

/// Immediate dominators (Cooper-Harvey-Kennedy). `idom[entry] = entry`; unreachable nodes get `usize::MAX`.
fn idoms(n: usize, succs: &[Vec<usize>], entry: usize) -> Vec<usize> {
    let rpo = reverse_postorder(n, succs, entry);
    let mut rpo_num = vec![usize::MAX; n];
    for (i, &b) in rpo.iter().enumerate() {
        rpo_num[b] = i;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        if rpo_num[b] == usize::MAX {
            continue; // unreachable
        }
        for &s in ss {
            preds[s].push(b);
        }
    }
    let mut idom = vec![usize::MAX; n];
    idom[entry] = entry;
    let intersect = |mut a: usize, mut b: usize, idom: &[usize]| -> usize {
        while a != b {
            while rpo_num[a] > rpo_num[b] {
                a = idom[a];
            }
            while rpo_num[b] > rpo_num[a] {
                b = idom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in rpo.iter() {
            if b == entry {
                continue;
            }
            let mut new_idom = usize::MAX;
            for &p in &preds[b] {
                if idom[p] == usize::MAX {
                    continue;
                }
                new_idom = if new_idom == usize::MAX {
                    p
                } else {
                    intersect(p, new_idom, &idom)
                };
            }
            if new_idom != usize::MAX && idom[b] != new_idom {
                idom[b] = new_idom;
                changed = true;
            }
        }
    }
    idom
}

/// Does `a` dominate `b` (walking the idom chain from `b` up to the entry)?
fn dominates(a: usize, b: usize, idom: &[usize], entry: usize) -> bool {
    if idom[b] == usize::MAX {
        return false; // b unreachable
    }
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        if x == entry {
            return false;
        }
        x = idom[x];
    }
}

/// Post-dominators: dominators on the reverse CFG rooted at a virtual exit (index `n`) that all real
/// exit blocks (Return / no successors) flow to. Returns `ipostdom` over `0..n` (virtual exit = `n`).
fn ipostdoms(n: usize, succs: &[Vec<usize>]) -> Vec<usize> {
    let exit = n;
    let mut rsuccs: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (b, ss) in succs.iter().enumerate() {
        if ss.is_empty() {
            // virtual exit -> real exit block, so the reverse-graph root reaches every exit
            rsuccs[exit].push(b);
        }
        for &s in ss {
            rsuccs[s].push(b); // reverse edge v -> u for each forward edge u -> v
        }
    }
    idoms(n + 1, &rsuccs, exit)
}

/// Is the CFG reducible? It is iff every retreating edge (target is a DFS-stack ancestor) is a back-edge
/// (target dominates source). Structurization handles only reducible CFGs; an irreducible one (a loop with
/// two entries) cannot get valid merge/continue blocks. Kernel-subset MIR is always reducible, so this
/// guards against a construct that should not reach here, reporting it instead of miscompiling
/// (FR-005).
fn is_reducible(n: usize, succs: &[Vec<usize>], idom: &[usize]) -> bool {
    let mut color = vec![0u8; n]; // 0 white, 1 grey (on stack), 2 black
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    color[0] = 1;
    while let Some(&(node, ci)) = stack.last() {
        if ci < succs[node].len() {
            stack.last_mut().unwrap().1 += 1;
            let s = succs[node][ci];
            match color[s] {
                0 => {
                    color[s] = 1;
                    stack.push((s, 0));
                }
                1
                    // Retreating edge node -> s (s is a DFS ancestor): reducible iff s dominates node.
                    if !dominates(s, node, idom, 0) => {
                        return false;
                    }
                _ => {} // forward / cross edge - fine
            }
        } else {
            color[node] = 2;
            stack.pop();
        }
    }
    true
}

/// Rewrite every successor block of `t` with `f`.
fn map_successors(t: &mut Terminator, f: impl Fn(usize) -> usize) {
    match t {
        Terminator::Goto { target }
        | Terminator::ThreadIndexCall { target, .. }
        | Terminator::Barrier { target } => target.index = f(target.index as usize) as u32,
        Terminator::SwitchInt { targets, .. } => {
            for (_, b) in targets.branches.iter_mut() {
                b.index = f(b.index as usize) as u32;
            }
            targets.otherwise.index = f(targets.otherwise.index as usize) as u32;
        }
        Terminator::Return | Terminator::Trap { .. } => {}
    }
}

/// The blocks that carry `h`'s path to its join `m` from outside `h`'s dominance region, in ascending
/// order. Traversal starts at `h`'s successors, walks through region blocks without collecting them (a
/// tail entered from deep inside the region is still a tail), and stops at `m`, at a block that cannot
/// reach `m`, and at a block it has already seen. Region blocks are not collected: the merge-redirect
/// already owns their edges.
fn escape_tail(n: usize, succs: &[Vec<usize>], idom: &[usize], h: usize, m: usize) -> Vec<usize> {
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    let mut to_m = vec![false; n];
    to_m[m] = true;
    let mut reach = vec![m];
    while let Some(b) = reach.pop() {
        for &p in &preds[b] {
            if !to_m[p] {
                to_m[p] = true;
                reach.push(p);
            }
        }
    }

    let mut seen = vec![false; n];
    let mut tail = Vec::new();
    let mut walk = succs[h].to_vec();
    while let Some(b) = walk.pop() {
        if seen[b] || b == m || !to_m[b] {
            continue;
        }
        seen[b] = true;
        if !dominates(h, b, idom, 0) {
            tail.push(b);
        }
        walk.extend_from_slice(&succs[b]);
    }
    tail.sort_unstable();
    tail
}

/// Give `h` a private copy of the tail it shares with the rest of the CFG: `tail`'s blocks are cloned,
/// the clones' successors are remapped onto the clones, and every edge from `h`'s region into `tail`
/// (which is what kept the tail outside the region) is re-pointed at the clone. The originals keep the
/// edges from outside the region, so nothing else changes hands and the CFG stays semantically equal -
/// kernel blocks carry statements, not phis, so a clone on another path computes the same values.
fn duplicate_escape_tail(body: &mut Body, tail: &[usize], idom: &[usize], h: usize) {
    let region_end = body.blocks.len();
    let mut clone_of = vec![usize::MAX; region_end];
    for (offset, &b) in tail.iter().enumerate() {
        clone_of[b] = region_end + offset;
    }
    let enter = |block: usize| {
        if block < clone_of.len() && clone_of[block] != usize::MAX {
            clone_of[block]
        } else {
            block
        }
    };
    for &b in tail {
        let mut clone = body.blocks[b].clone();
        map_successors(&mut clone.terminator, enter);
        body.blocks.push(clone);
    }
    for b in 0..region_end {
        if dominates(h, b, idom, 0) {
            map_successors(&mut body.blocks[b].terminator, enter);
        }
    }
}

/// Structurize selections in `body` so every selection has a merge block its header dominates.
/// Idempotent; only inserts goto-forwarding blocks, plus a clone of a shared region-external tail when a
/// selection's arm escapes its own region (card 461). Returns an error naming the construct if the CFG is
/// irreducible (FR-005), or if the pass cannot converge within its block budget.
pub fn structurize(body: &mut Body) -> Result<(), String> {
    // Every iteration either stops or appends at least one block (a merge block, or a duplicated tail), so
    // bounding the block count bounds the loop. The bound is anchored to the body's size ON ENTRY: the old
    // `guard > blocks.len() * 8 + 256` compared the guard against a count that grows with it, so it could
    // never trip while the loop was still appending - which is how a `&& ... else` body spun pootc forever
    // (card 461). The coefficients are the ones the growing guard used (see
    // [[structurize-fixpoint-nonconvergence]]); hitting them means the pass made no sense of the CFG, so it
    // fails with a typed error instead of emitting a half-structured body or hanging the build.
    let block_budget = body.blocks.len() * 8 + 256;
    structurize_limited(body, block_budget)
}

/// [`structurize`] with an explicit ceiling on the body's block count, so the bound itself is testable
/// (card 461: the pass must never hang the build). `block_budget` is absolute, not relative.
fn structurize_limited(body: &mut Body, block_budget: usize) -> Result<(), String> {
    let entry_blocks = body.blocks.len();
    // Reject an irreducible CFG up front.
    {
        let n = body.blocks.len();
        let succs: Vec<Vec<usize>> = body
            .blocks
            .iter()
            .map(|b| successors(&b.terminator))
            .collect();
        let idom = idoms(n, &succs, 0);
        if !is_reducible(n, &succs, &idom) {
            return Err(
                "irreducible control flow (a loop with multiple entries) cannot be structurized for \
                 SPIR-V; the kernel subset should only produce reducible CFGs"
                    .into(),
            );
        }
    }
    loop {
        if body.blocks.len() > block_budget {
            return Err(format!(
                "selection structurization did not converge: the body grew from {entry_blocks} to {} \
                 blocks (budget {block_budget})",
                body.blocks.len()
            ));
        }
        let n = body.blocks.len();
        let succs: Vec<Vec<usize>> = body
            .blocks
            .iter()
            .map(|b| successors(&b.terminator))
            .collect();
        let idom = idoms(n, &succs, 0);
        let ipdom = ipostdoms(n, &succs);

        // Back-edge targets are loop headers; skip them as selection headers.
        let mut is_loop_header = vec![false; n];
        for (b, ss) in succs.iter().enumerate() {
            for &s in ss {
                if dominates(s, b, &idom, 0) {
                    is_loop_header[s] = true; // b->s with s dominating b is a back-edge
                }
            }
        }

        let mut fixed_one = false;
        for h in 0..n {
            if idom[h] == usize::MAX || is_loop_header[h] {
                continue;
            }
            // Only 2-way selections.
            let is_selection = matches!(&body.blocks[h].terminator, Terminator::SwitchInt { .. });
            if !is_selection {
                continue;
            }
            let m = ipdom[h];
            if m == usize::MAX || m >= n {
                continue; // both arms exit, or no real join
            }
            if dominates(h, m, &idom, 0) {
                continue; // join already privately owned by this selection - structured
            }
            // Card 461: `if a && b { .. } else { .. }` shares one else block between the two conditions,
            // so the inner header's arm escapes its own dominance region and reaches `m` from a block the
            // redirect below cannot move. Break the sharing first: duplicate that region-external tail,
            // which puts a private copy of every path into `m` under `h`. Without it the redirect appends
            // one goto-forwarding block per round and never converges (the depth-4 `&&` spin).
            let tail = escape_tail(n, &succs, &idom, h, m);
            if tail.iter().any(|&b| succs[b].contains(&m)) {
                duplicate_escape_tail(body, &tail, &idom, h);
                fixed_one = true;
                break; // recompute dominance
            }
            // The join m is shared (not dominated by h): give h a private merge m'.
            let mprime = n; // new block index (append)
            let mut redirected = false;
            // Redirect every edge into `m` from a block dominated by `h`, including `h` itself. `h` must be
            // included when one of its arms branches directly to the shared join: otherwise only the other
            // arm reconverges at `m'`, `m'` never post-dominates `h`, and the selection is re-fixed forever
            // (the tiled-GEMM cascade, update 0337).
            let preds_in_region: Vec<usize> = (0..n)
                .filter(|&p| dominates(h, p, &idom, 0))
                .filter(|&p| succs[p].contains(&m))
                .collect();
            if preds_in_region.is_empty() {
                continue; // nothing to redirect
            }
            // Append m' = goto m.
            body.blocks.push(BasicBlock {
                statements: Vec::new(),
                terminator: Terminator::Goto {
                    target: BlockId { index: m as u32 },
                },
            });
            for p in preds_in_region {
                replace_target(&mut body.blocks[p].terminator, m, mprime);
                redirected = true;
            }
            if redirected {
                fixed_one = true;
                break; // recompute dominance
            } else {
                body.blocks.pop();
            }
        }
        if !fixed_one {
            break;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_kernel_ir::{Local, Operand, Place, SwitchTargets};

    fn bid(i: u32) -> BlockId {
        BlockId { index: i }
    }
    fn goto(t: u32) -> Terminator {
        Terminator::Goto { target: bid(t) }
    }
    // 2-way selection: false-target `f` (value 0) and true-target `t` (otherwise).
    fn switch(t: u32, f: u32) -> Terminator {
        Terminator::SwitchInt {
            discr: Operand::Copy(Place::local(Local { index: 0 })),
            targets: SwitchTargets {
                branches: vec![(0, bid(f))],
                otherwise: bid(t),
            },
        }
    }
    fn body_of(terms: Vec<Terminator>) -> Body {
        Body {
            name: "t".into(),
            param_count: 0,
            locals: Vec::new(),
            blocks: terms
                .into_iter()
                .map(|terminator| BasicBlock {
                    statements: Vec::new(),
                    terminator,
                })
                .collect(),
            workgroup_size: [64, 1, 1],
            workgroup_locals: Vec::new(),
        }
    }
    fn succ_of(b: &Body, i: usize) -> Vec<usize> {
        successors(&b.blocks[i].terminator)
    }

    #[test]
    fn nested_selection_gets_a_private_merge() {
        // 0:goto1  1:sw(then=2,else=5)  2:sw(then=3,else=4)  3:goto5  4:goto5  5:ret
        // bb2's arms (3,4) share the outer merge 5, which bb2 does not dominate, so it needs a private merge.
        let mut b = body_of(vec![
            goto(1),
            switch(2, 5),
            switch(3, 4),
            goto(5),
            goto(5),
            Terminator::Return,
        ]);
        structurize(&mut b).unwrap();
        assert_eq!(b.blocks.len(), 7, "one private merge block inserted");
        // bb3 and bb4 now go to the new block (6), which gotos the original merge 5.
        let m = succ_of(&b, 3);
        assert_eq!(m, vec![6]);
        assert_eq!(succ_of(&b, 4), vec![6]);
        assert_eq!(succ_of(&b, 6), vec![5]);
        // the outer else edge 1->5 is unchanged.
        assert!(succ_of(&b, 1).contains(&5));
    }

    #[test]
    fn idempotent() {
        let mut b = body_of(vec![
            goto(1),
            switch(2, 5),
            switch(3, 4),
            goto(5),
            goto(5),
            Terminator::Return,
        ]);
        structurize(&mut b).unwrap();
        let n = b.blocks.len();
        structurize(&mut b).unwrap();
        assert_eq!(b.blocks.len(), n, "second run inserts nothing");
    }

    #[test]
    fn simple_if_else_unchanged() {
        // 0:goto1 1:sw(2,3) 2:goto4 3:goto4 4:ret  -- single selection, merge 4 dominated by 1.
        let mut b = body_of(vec![
            goto(1),
            switch(2, 3),
            goto(4),
            goto(4),
            Terminator::Return,
        ]);
        structurize(&mut b).unwrap();
        assert_eq!(b.blocks.len(), 5, "already structured; no block added");
    }

    #[test]
    fn loop_header_not_transformed() {
        // 0:goto1 1:sw(body=2,exit=3) 2:goto1(back-edge) 3:ret
        let mut b = body_of(vec![goto(1), switch(2, 3), goto(1), Terminator::Return]);
        structurize(&mut b).unwrap();
        assert_eq!(
            b.blocks.len(),
            4,
            "loop header is not treated as a selection"
        );
    }

    #[test]
    fn irreducible_cfg_is_rejected() {
        // 0:sw(1,2)  1:sw(2,3)  2:sw(1,3)  3:ret: the {1,2} loop has two entries (0->1 and 0->2) and
        // 1<->2 mutual edges, and neither dominates the other, so it is irreducible (FR-005).
        let mut b = body_of(vec![
            switch(1, 2),
            switch(2, 3),
            switch(1, 3),
            Terminator::Return,
        ]);
        let err = structurize(&mut b).unwrap_err();
        assert!(
            err.contains("irreducible"),
            "expected irreducible diagnostic, got: {err}"
        );
    }

    /// `if a && b { .. } else { .. }` the way MIR writes it (card 461): both conditions branch to the SAME
    /// else block, so the inner header's false arm leaves its own dominance region and enters the join
    /// from outside it. Without the region-external tail clone this body never converged - one merge block
    /// per round, forever, which is what spun `pootc` on the depth-4 `&&` reproducer.
    /// 0:goto1 1:sw(check b=2, else=4) 2:sw(then=3, else=4) 3:goto5 4:goto5 5:ret
    fn short_circuit_with_shared_else() -> Body {
        body_of(vec![
            goto(1),
            switch(2, 4),
            switch(3, 4),
            goto(5),
            goto(5),
            Terminator::Return,
        ])
    }

    /// Blocks reachable from the entry (every path, no loops in these fixtures).
    fn reachable(b: &Body) -> std::collections::BTreeSet<usize> {
        let n = b.blocks.len();
        let succs: Vec<Vec<usize>> = b
            .blocks
            .iter()
            .map(|block| successors(&block.terminator))
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            stack.extend(succs[node].iter().copied().filter(|&s| s < n));
        }
        seen
    }

    /// The pass's contract: every 2-way selection header owns a join block it dominates. Not implied by
    /// returning `Ok` - the fixpoint also stops when no header can be moved, which can leave one behind.
    fn assert_private_merges(b: &Body) {
        let n = b.blocks.len();
        let succs: Vec<Vec<usize>> = b
            .blocks
            .iter()
            .map(|block| successors(&block.terminator))
            .collect();
        let idom = idoms(n, &succs, 0);
        let ipdom = ipostdoms(n, &succs);
        for (h, &m) in ipdom.iter().enumerate().take(n) {
            if !matches!(&b.blocks[h].terminator, Terminator::SwitchInt { .. }) {
                continue;
            }
            if m == usize::MAX || m >= n {
                continue;
            }
            assert!(
                dominates(h, m, &idom, 0),
                "selection {h} does not dominate its join {m}\n{b:#?}"
            );
        }
    }

    #[test]
    fn shared_else_arm_of_a_short_circuit_converges() {
        let mut b = short_circuit_with_shared_else();
        let before = reachable(&b);
        structurize(&mut b).unwrap();

        assert_private_merges(&b);
        // The outer condition keeps the original else block; the inner condition is re-pointed at a clone,
        // so the shared arm is what changed hands. One clone plus one private merge: 6 -> 8 blocks.
        // (`successors` lists the 0-branch before the otherwise arm.)
        assert_eq!(succ_of(&b, 1), vec![4, 2], "outer false arm still uses 4");
        assert!(!succ_of(&b, 2).contains(&4), "inner false arm must leave 4");
        assert_eq!(b.blocks.len(), 8, "one clone of the shared else, one merge");
        // Nothing dropped: every block the input could reach is still reachable (the clone is extra).
        let after = reachable(&b);
        assert!(
            before.is_subset(&after),
            "structurizing dropped blocks: {:?}",
            before.difference(&after).collect::<Vec<_>>()
        );

        let n = b.blocks.len();
        structurize(&mut b).unwrap();
        assert_eq!(b.blocks.len(), n, "second run inserts nothing");
    }

    #[test]
    fn a_fixpoint_over_its_block_budget_is_an_error() {
        // The bound that makes the pass incapable of hanging a build: a body that still needs work once it
        // reaches the ceiling fails with the typed diagnostic instead of appending forever.
        let mut b = short_circuit_with_shared_else();
        let ceiling = b.blocks.len();
        let err = structurize_limited(&mut b, ceiling).unwrap_err();
        assert!(
            err.contains("did not converge"),
            "expected the block-budget diagnostic, got: {err}"
        );
    }
}
