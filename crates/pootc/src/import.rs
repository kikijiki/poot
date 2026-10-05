//! Stable MIR (`rustc_public`) -> poot-IR ([`poot_kernel_ir::Body`]). This is the only rustc-linked
//! piece; `poot-codegen` (Body -> LLVM -> llc -> SPIR-V/PTX) is the back half. Anything outside the
//! supported subset is rejected with a named "unsupported" error (no silent miscompile).
//!
//! Supported subset:
//! - slices, the thread-index intrinsic, indexing, arithmetic, the bounds guard, `while` loops, and named
//!   const bounds;
//! - `for k in a..b`, `for k in a..=b`, and `for x in slice` loops. [`analyze_range_loops`] detects the
//!   Range/Option/slice::Iter plumbing and rewrites it to a counter loop: inclusive runs
//!   `counter <= END`; a slice loop reuses the loop var as a `0..slice.len()` counter and rewrites `*v`
//!   element reads to `slice[counter]`;
//! - unary float-math methods (`x.sqrt()`, `x.exp()`, `x.sin()`, ...) as a `MathUnary` rvalue, binary
//!   `x.max(y)`/`x.min(y)` as a `BinOp`, and `f32::from_bits(u32)` as a `Bitcast` (an f32 packed into a
//!   u32 buffer, e.g. an attention scale in a metadata buffer). RMSNorm and stable softmax import and run
//!   on the GPU from these (see import_run.rs);
//! - workgroup-parallel kernels (spec 054): `local_index()`/`group_index()` are the within-group lane and
//!   workgroup id (`ThreadIndexCall` LocalX/GroupX), `workgroup_barrier()` a `Barrier`, and
//!   `wg_write(array,i,v)`/`wg_read(array,i)` access the workgroup-local (LDS) arrays. An imported argmax
//!   (2 LDS arrays) matches kernelgen's `argmax_lds`, a GEMV matches `gemv_lds`, and an LDS
//!   sum-reduction matches `wg_sum`. A `const WORKGROUP_SIZE: usize = N;` sets the body's lane count
//!   (default 64). Every intrinsic above (plus `atomic_add`) is a real call into the `poot_kernel_intrinsics`
//!   crate; a kernel source depends on it and never redeclares its own stub. The importer recognizes an
//!   intrinsic by the callee's resolved declaration (crate `poot_kernel_intrinsics` + item name), never by
//!   the bare callee name, so a kernel-authored helper of the same name is an ordinary (rejected) call
//!   (R468-009). `no_contract_add(a, b)` becomes `Rvalue::BinaryOpNoContract(BinOp::Add, a, b)` (card 675):
//!   the kernel source's explicit marker that codegen must not contract this add with a producing multiply
//!   into an FMA, bridging the gap Card 628's `BinaryOpNoContract` left (reachable from kernelgen's Body API,
//!   not from a plain-Rust kernel) for a tier-1 sampling primitive like the Gumbel select (ADR 0114).
//!
//! A MIR `Assert` (bounds, overflow, divide, shift) lowers to a `SwitchInt` that traps on failure instead
//! of erasing to its success target; `Unreachable` lowers directly to a trap ([`kir::Terminator::Trap`]).
//! The GPU has no panic path, so the trap is the kernel-subset's abort: on ROCm and PTX it is a real device
//! trap instruction; on SpirvVulkan, which has none, `poot-codegen` refuses a `Trap`-containing body with a
//! typed error (card 531c; R468-007). `.rev()`/`.step_by()` are rejected.

use poot_kernel_ir as kir;
use rustc_public::mir as smir;
use rustc_public::ty::{ConstantKind, FloatTy, IntTy, RigidTy, TyKind, UintTy};
use rustc_public::{CrateDef, CrateItem};

type R<T> = Result<T, String>;

fn unsupported<T>(what: impl std::fmt::Display) -> R<T> {
    Err(format!("unsupported in kernel: {what}"))
}

/// Import one `#[kernel]` item's Stable MIR into a poot [`kir::Body`].
pub fn import_item(item: &CrateItem) -> Result<kir::Body, String> {
    let name = item.name();
    let src = kir::naming::source_name_of_path(&name).unwrap_or_else(|| "kernel".to_string());
    let body = item.body().ok_or("item has no MIR body")?;
    import_body(&kir::naming::mangle(&src), &body)
}

/// Translate a kernel's Stable MIR body into poot-IR. `for` loops are detected up front and their
/// iterator plumbing rewritten to a counter loop; everything else is a block-by-block translation.
pub fn import_body(name: &str, body: &smir::Body) -> R<kir::Body> {
    let Loops {
        loops,
        support,
        new_call_blocks,
    } = analyze_range_loops(body)?;

    // Iterator-plumbing locals (Range/Option/&mut/the isize discriminant) become Unit: their
    // statements are dropped and `map_ty` would reject their types. The rest map normally.
    let mut locals: Vec<kir::LocalDecl> = body
        .locals()
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let ty = if support.contains(&i) {
                kir::Ty::Unit
            } else {
                map_ty(&l.ty)?
            };
            Ok(kir::LocalDecl {
                ty,
                mutable: matches!(l.mutability, smir::Mutability::Mut),
            })
        })
        .collect::<R<Vec<_>>>()?;
    let param_count = body.arg_locals().len() as u32;

    // One synthesized `counter < END` bool local per loop (plus a `usize` `Len(slice)` local per slice
    // loop), and a map from each loop-control block to its loop (so iter/latch/cond blocks get a
    // synthesized terminator).
    let mut cond_locals = Vec::new();
    let mut len_locals: Vec<Option<kir::Local>> = Vec::new();
    let mut block_owner: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for (li, lp) in loops.iter().enumerate() {
        cond_locals.push(kir::Local {
            index: locals.len() as u32,
        });
        locals.push(kir::LocalDecl {
            ty: kir::Ty::Bool,
            mutable: true,
        });
        // A slice loop reuses the `&T` loop var as a `usize` counter (overriding its type) and needs a `usize` local for `Len(slice)`.
        if lp.slice_local.is_some() {
            locals[lp.counter.index as usize].ty = kir::Ty::Usize;
            len_locals.push(Some(kir::Local {
                index: locals.len() as u32,
            }));
            locals.push(kir::LocalDecl {
                ty: kir::Ty::Usize,
                mutable: true,
            });
        } else {
            len_locals.push(None);
        }
        for b in [lp.iter_block, lp.latch_block, lp.cond_block] {
            block_owner.insert(b, li);
        }
    }

    // Every MIR block whose terminator is `Assert` or `Unreachable` gets a unique, 1-based trap code in
    // program order (0 means "no fault" for the SpirvVulkan error word, so a real code is never 0). An
    // `Assert` also gets its own dedicated trap block, appended after the translated MIR blocks in the
    // same order, since its own block keeps its `SwitchInt` guard; `Unreachable` substitutes `Trap`
    // directly in its own block, no extra block needed.
    let mut trap_code_of: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut assert_trap_block_of: std::collections::HashMap<usize, kir::BlockId> =
        std::collections::HashMap::new();
    let mut assert_trap_codes: Vec<u32> = Vec::new();
    let mut next_trap_code = 0u32;
    for (bidx, bb) in body.blocks.iter().enumerate() {
        match &bb.terminator.kind {
            smir::TerminatorKind::Assert { .. } => {
                next_trap_code += 1;
                trap_code_of.insert(bidx, next_trap_code);
                assert_trap_block_of.insert(
                    bidx,
                    kir::BlockId {
                        index: (body.blocks.len() + assert_trap_codes.len()) as u32,
                    },
                );
                assert_trap_codes.push(next_trap_code);
            }
            smir::TerminatorKind::Unreachable => {
                next_trap_code += 1;
                trap_code_of.insert(bidx, next_trap_code);
            }
            _ => {}
        }
    }

    let mut blocks = Vec::with_capacity(body.blocks.len());
    for (bidx, bb) in body.blocks.iter().enumerate() {
        let mut statements = Vec::new();
        for s in &bb.statements {
            // Drop iterator-plumbing statements (Range aggregate, discriminant, Some-downcast, writes to a support local) when a for-loop is present.
            if !support.is_empty() && stmt_is_iter_plumbing(&support, &s.kind) {
                continue;
            }
            if let Some(st) = map_statement(&s.kind)? {
                statements.push(st);
            }
        }
        // A loop-control block gets a synthesized terminator; a `RangeInclusive::new` block keeps its
        // statements but replaces the call (bounds already read) with a Goto to the call target;
        // everything else, including each body's back-edge Goto to its latch, translates normally.
        let override_term = block_owner
            .get(&bidx)
            .and_then(|&li| loops[li].terminator(bidx, cond_locals[li], len_locals[li]));
        let (extra, terminator) = match override_term {
            Some(t) => t,
            None => match new_call_blocks.get(&bidx) {
                Some(&target) => (
                    Vec::new(),
                    kir::Terminator::Goto {
                        target: map_block(target),
                    },
                ),
                None => map_terminator(
                    &bb.terminator.kind,
                    body,
                    param_count,
                    trap_code_of.get(&bidx).copied(),
                    assert_trap_block_of.get(&bidx).copied(),
                )?,
            },
        };
        statements.extend(extra);
        blocks.push(kir::BasicBlock {
            statements,
            terminator,
        });
    }
    for code in assert_trap_codes {
        blocks.push(kir::BasicBlock {
            statements: Vec::new(),
            terminator: kir::Terminator::Trap { code },
        });
    }
    // Slice loops: the body reads each element as `*loopvar`. With the loop var now a `usize` counter,
    // rewrite every `(*loopvar)` place to `(*slice)[loopvar]` (kir has no `Ref`-of-place).
    for lp in &loops {
        if let Some(slice) = lp.slice_local {
            rewrite_slice_reads(&mut blocks, lp.counter, slice);
        }
    }
    let mut raw = kir::Body::new(name, param_count, locals, blocks);
    // A kernel may declare its lane count with `const WORKGROUP_SIZE: usize = N;`, which sets
    // `workgroup_size` to `[N, 1, 1]` (default 64), so a workgroup-parallel kernel can match its
    // kernelgen twin (e.g. a GEMV at 128). Spec 054 FR-003.
    if let Some(wg) = workgroup_size_from_body(body) {
        raw.workgroup_size = [wg, 1, 1];
    }
    // A kernel using the LDS intrinsics gets one workgroup-local array per array id it touches (argmax
    // uses 2: values + indices), each sized to `workgroup_size[0]` lanes (emitted as `addrspace(3)`
    // globals). Spec 054. A kernel whose LDS tile exceeds the lane count (a thread-coarsened tiled GEMM
    // stages 2*ts*ts A elements with ts*ts lanes) declares `const LDS_SIZE: usize = N;` and every LDS
    // array is sized to N.
    if let Some(max_array) = max_lds_array(&raw) {
        let len = lds_size_from_body(body).unwrap_or(raw.workgroup_size[0]);
        raw.workgroup_locals = (0..=max_array)
            .map(|_| kir::WorkgroupLocalDecl {
                elem_ty: kir::Ty::F32,
                len,
            })
            .collect();
    }
    Ok(crate::normalize::normalize(raw))
}

/// The highest workgroup-local (LDS) array id any block touches, or `None` if the kernel uses no LDS. One `WorkgroupLocalDecl` per id `0..=max` is declared.
fn max_lds_array(body: &kir::Body) -> Option<u8> {
    body.blocks
        .iter()
        .flat_map(|bb| &bb.statements)
        .filter_map(|s| match s {
            kir::Statement::WorkgroupLocalWrite { array, .. } => Some(*array),
            kir::Statement::Assign(_, kir::Rvalue::WorkgroupLocalRead { array, .. }) => {
                Some(*array)
            }
            _ => None,
        })
        .max()
}

/// Read a `const WORKGROUP_SIZE: usize = N;` used by the kernel (it appears as a named Unevaluated
/// const in the MIR) and return `N`, or `None` (default 64 applies). Scans operands in `Assign` rvalues
/// and terminator call args.
fn workgroup_size_from_body(body: &smir::Body) -> Option<u32> {
    named_usize_const_from_body(body, "WORKGROUP_SIZE")
}

/// Read a `const LDS_SIZE: usize = N;` used by the kernel and return `N`, sizing every LDS array to N.
/// `None` sizes them to `workgroup_size[0]`. A coarsened tiled GEMM (A tile of 2*ts*ts elements staged by
/// ts*ts lanes) declares `LDS_SIZE = 2*ts*ts`.
fn lds_size_from_body(body: &smir::Body) -> Option<u32> {
    named_usize_const_from_body(body, "LDS_SIZE")
}

/// Read a named `const NAME: usize = N;` used by the kernel (a named Unevaluated const in the MIR, as a
/// stride, bound, or array size) and return `N`. Scans operands in `Assign` rvalues and terminator call
/// args.
fn named_usize_const_from_body(body: &smir::Body, name: &str) -> Option<u32> {
    let from_operand = |op: &smir::Operand| -> Option<u32> {
        let smir::Operand::Constant(c) = op else {
            return None;
        };
        let ConstantKind::Unevaluated(uc) = c.const_.kind() else {
            return None;
        };
        if uc.def.name().rsplit("::").next() == Some(name) {
            return c.const_.eval_target_usize().ok().map(|v| v as u32);
        }
        None
    };
    let from_rvalue = |rv: &smir::Rvalue| -> Option<u32> {
        match rv {
            smir::Rvalue::Use(op) | smir::Rvalue::UnaryOp(_, op) | smir::Rvalue::Cast(_, op, _) => {
                from_operand(op)
            }
            smir::Rvalue::BinaryOp(_, a, b) => from_operand(a).or_else(|| from_operand(b)),
            _ => None,
        }
    };
    for bb in &body.blocks {
        for s in &bb.statements {
            if let smir::StatementKind::Assign(_, rv) = &s.kind
                && let Some(n) = from_rvalue(rv)
            {
                return Some(n);
            }
        }
        if let smir::TerminatorKind::Call { args, .. } = &bb.terminator.kind {
            for a in args {
                if let Some(n) = from_operand(a) {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Root local of an operand that is a bare place (`Copy`/`Move` with no projection), as for the
/// slice-parameter and `usize`-index args of `atomic_add`. `None` for a constant or projected place (the
/// caller rejects it with a named diagnostic).
fn place_operand_local(op: &smir::Operand) -> Option<kir::Local> {
    match op {
        smir::Operand::Copy(p) | smir::Operand::Move(p) if p.projection.is_empty() => {
            Some(map_local(p.local))
        }
        _ => None,
    }
}

/// Extract a constant LDS array id (a `usize` literal in `wg_read(array, ..)`/`wg_write(array, ..)`) as the IR's `u8`; it must be a compile-time constant.
fn const_lds_array(op: &smir::Operand) -> R<u8> {
    match map_operand(op)? {
        kir::Operand::Const(kir::Constant::Usize(n)) if n < 256 => Ok(n as u8),
        _ => unsupported(
            "LDS array id (the first arg of wg_read/wg_write) must be a constant 0..255",
        ),
    }
}

/// Rewrite every `(*loopvar)` place to `(*slice)[loopvar]` (the element the slice-loop counter selects).
/// A bare `loopvar` (the counter, empty projection) is untouched; only a leading `Deref` is an element read.
fn rewrite_slice_reads(blocks: &mut [kir::BasicBlock], loopvar: kir::Local, slice: kir::Local) {
    fn fix_place(p: &mut kir::Place, loopvar: kir::Local, slice: kir::Local) {
        if p.local.index == loopvar.index
            && matches!(p.projection.first(), Some(kir::ProjectionElem::Deref))
        {
            let mut proj = vec![
                kir::ProjectionElem::Deref,
                kir::ProjectionElem::Index(loopvar),
            ];
            proj.extend(p.projection.iter().skip(1).cloned());
            p.local = slice;
            p.projection = proj;
        }
    }
    fn fix_operand(op: &mut kir::Operand, loopvar: kir::Local, slice: kir::Local) {
        if let kir::Operand::Copy(p) | kir::Operand::Move(p) = op {
            fix_place(p, loopvar, slice);
        }
    }
    fn fix_rvalue(rv: &mut kir::Rvalue, loopvar: kir::Local, slice: kir::Local) {
        match rv {
            kir::Rvalue::Use(op)
            | kir::Rvalue::UnaryOp(_, op)
            | kir::Rvalue::MathUnary(_, op)
            | kir::Rvalue::IntScalarUnary(_, op)
            | kir::Rvalue::Cast { operand: op, .. }
            | kir::Rvalue::Bitcast { operand: op, .. }
            | kir::Rvalue::Fp8Decode { operand: op, .. }
            | kir::Rvalue::Fp8Encode { operand: op, .. } => fix_operand(op, loopvar, slice),
            kir::Rvalue::BinaryOp(_, a, b) => {
                fix_operand(a, loopvar, slice);
                fix_operand(b, loopvar, slice);
            }
            kir::Rvalue::Len(p) => fix_place(p, loopvar, slice),
            _ => {}
        }
    }
    for bb in blocks.iter_mut() {
        for s in &mut bb.statements {
            if let kir::Statement::Assign(p, rv) = s {
                fix_place(p, loopvar, slice);
                fix_rvalue(rv, loopvar, slice);
            }
        }
        if let kir::Terminator::SwitchInt { discr, .. } = &mut bb.terminator {
            fix_operand(discr, loopvar, slice);
        }
    }
}

/// One `for` loop recognized from the Range-iterator MIR shape and rewritten into a counter loop. The
/// loop variable is reused as the counter, so the IR needs no Option/Range/enum support.
struct RangeLoop {
    /// Block with the `into_iter` Call, rewritten to `counter = START; goto cond`.
    iter_block: usize,
    /// The `Iterator::next` Call block (back-edge latch), rewritten to `counter += 1; goto cond`.
    latch_block: usize,
    /// The `discriminant`/`switchInt` block, rewritten to `cond = counter < END; switch`.
    cond_block: usize,
    /// The Some arm (body entry) and None arm (exit).
    body_block: usize,
    exit_block: usize,
    /// The loop variable, reused as the counter.
    counter: kir::Local,
    start: kir::Operand,
    end: kir::Operand,
    /// `for k in a..=b`: the condition is `counter <= END`, not `counter < END`.
    inclusive: bool,
    /// `for x in slice`: the slice param being iterated. `Some` makes this a slice loop: the loop var is a
    /// `0..slice.len()` counter, the bound is `Len(slice)` (materialized into `len_local`), and element
    /// reads `(*loopvar)` become `(*slice)[loopvar]`. The yielded reference is never materialized.
    slice_local: Option<kir::Local>,
}

impl RangeLoop {
    /// The synthesized (extra statements, terminator) for a loop-control block, or `None` to translate the terminator normally.
    fn terminator(
        &self,
        bidx: usize,
        cond: kir::Local,
        len: Option<kir::Local>,
    ) -> Option<(Vec<kir::Statement>, kir::Terminator)> {
        let counter_op = kir::Operand::Copy(kir::Place::local(self.counter));
        if bidx == self.iter_block {
            // Init: (slice loop only) len = Len(slice); then counter = START (0); goto cond.
            let mut init = Vec::new();
            if let (Some(slice), Some(len)) = (self.slice_local, len) {
                init.push(kir::Statement::Assign(
                    kir::Place::local(len),
                    kir::Rvalue::Len(kir::Place::local(slice)),
                ));
            }
            init.push(kir::Statement::Assign(
                kir::Place::local(self.counter),
                kir::Rvalue::Use(self.start.clone()),
            ));
            Some((
                init,
                kir::Terminator::Goto {
                    target: map_block(self.cond_block),
                },
            ))
        } else if bidx == self.latch_block {
            // Back-edge: counter += 1; goto cond.
            let inc = kir::Statement::Assign(
                kir::Place::local(self.counter),
                kir::Rvalue::BinaryOp(
                    kir::BinOp::Add,
                    counter_op,
                    kir::Operand::Const(kir::Constant::Usize(1)),
                ),
            );
            Some((
                vec![inc],
                kir::Terminator::Goto {
                    target: map_block(self.cond_block),
                },
            ))
        } else if bidx == self.cond_block {
            // cond = counter < END (exclusive, or slice `counter < len`) or counter <= END (`a..=b`); switchInt(cond) [0 -> exit, otherwise -> body].
            let cmp = if self.inclusive {
                kir::BinOp::Le
            } else {
                kir::BinOp::Lt
            };
            // A slice loop's bound is the materialized `Len(slice)`; a range loop's is its `end` operand.
            let end = match (self.slice_local, len) {
                (Some(_), Some(len)) => kir::Operand::Copy(kir::Place::local(len)),
                _ => self.end.clone(),
            };
            let set = kir::Statement::Assign(
                kir::Place::local(cond),
                kir::Rvalue::BinaryOp(cmp, counter_op, end),
            );
            Some((
                vec![set],
                kir::Terminator::SwitchInt {
                    discr: kir::Operand::Copy(kir::Place::local(cond)),
                    targets: kir::SwitchTargets {
                        branches: vec![(0, map_block(self.exit_block))],
                        otherwise: map_block(self.body_block),
                    },
                },
            ))
        } else {
            None
        }
    }
}

/// All `for` loops in a body plus the iterator-plumbing locals (Range/Option/&mut/discriminant) that map to Unit (their statements dropped).
struct Loops {
    loops: Vec<RangeLoop>,
    support: std::collections::HashSet<usize>,
    /// `RangeInclusive::new(a, b)` call blocks (`a..=b`): bounds are read from the call args and the call
    /// terminator becomes a `Goto` to its target (it is iterator plumbing like the exclusive
    /// `Range { start, end }` aggregate, but a Call, so it needs a terminator override).
    new_call_blocks: std::collections::HashMap<usize, usize>,
}

/// Detect every `for k in START..END`, `for k in START..=END`, and `for x in slice` loop and gather the
/// plumbing locals. Inclusive `a..=b` lowers to a `RangeInclusive::new(a,b)` Call (the exclusive form is a
/// `Range { start, end }` aggregate) and runs `counter <= END`; a slice loop has no range (the
/// `into_iter` arg is the slice), so its bound is `0..slice.len()` and the loop var is the index counter.
/// `.rev()`/`.step_by()` are rejected with a diagnostic rather than miscompiled.
fn analyze_range_loops(body: &smir::Body) -> R<Loops> {
    use std::collections::{HashMap, HashSet};
    let mut range_of: HashMap<usize, (kir::Operand, kir::Operand)> = HashMap::new();
    let mut into_iter_arg: HashMap<usize, usize> = HashMap::new();
    let mut into_iter_block: HashMap<usize, usize> = HashMap::new();
    let mut alias: HashMap<usize, usize> = HashMap::new();
    // `a..=b` range locals (built by `RangeInclusive::new`) and the call blocks to rewrite to `Goto`.
    let mut inclusive_ranges: HashSet<usize> = HashSet::new();
    let mut new_call_blocks: HashMap<usize, usize> = HashMap::new();
    let mut support: HashSet<usize> = body
        .locals()
        .iter()
        .enumerate()
        .filter(|(_, d)| is_iter_support_ty(&d.ty))
        .map(|(i, _)| i)
        .collect();

    // One pass: Range aggregates, discriminant dests, alias edges, into_iter calls.
    for (bidx, bb) in body.blocks.iter().enumerate() {
        for s in &bb.statements {
            if let smir::StatementKind::Assign(dest, rv) = &s.kind {
                match rv {
                    smir::Rvalue::Aggregate(smir::AggregateKind::Adt(def, ..), ops)
                        if def.name().contains("Range") && ops.len() == 2 =>
                    {
                        range_of.insert(dest.local, (map_operand(&ops[0])?, map_operand(&ops[1])?));
                    }
                    smir::Rvalue::Discriminant(_) => {
                        support.insert(dest.local);
                    }
                    smir::Rvalue::Use(smir::Operand::Copy(p) | smir::Operand::Move(p))
                    | smir::Rvalue::Ref(_, _, p)
                        if p.projection.is_empty() =>
                    {
                        alias.insert(dest.local, p.local);
                    }
                    _ => {}
                }
            }
        }
        if let smir::TerminatorKind::Call {
            func,
            args,
            destination,
            ..
        } = &bb.terminator.kind
            && callee_last_seg(func, body).as_deref() == Some("into_iter")
            && let Some(smir::Operand::Copy(p) | smir::Operand::Move(p)) = args.first()
            && p.projection.is_empty()
        {
            into_iter_arg.insert(destination.local, p.local);
            into_iter_block.insert(destination.local, bidx);
        }
        // `a..=b` lowers to a `RangeInclusive::new(a, b)` Call, not an aggregate. Recognize it by the
        // destination's RangeInclusive type, read the bounds from the args, and mark the block so its
        // call becomes a `Goto`. The exclusive `a..b` aggregate is handled above.
        if let smir::TerminatorKind::Call {
            args,
            destination,
            target: Some(target),
            ..
        } = &bb.terminator.kind
            && is_range_inclusive_ty(&body.locals()[destination.local].ty)
            && args.len() == 2
        {
            range_of.insert(
                destination.local,
                (map_operand(&args[0])?, map_operand(&args[1])?),
            );
            inclusive_ranges.insert(destination.local);
            new_call_blocks.insert(bidx, *target);
        }
    }

    // Each `Iterator::next` (Option dest) is one loop.
    let mut loops = Vec::new();
    for (bidx, bb) in body.blocks.iter().enumerate() {
        let smir::TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            ..
        } = &bb.terminator.kind
        else {
            continue;
        };
        if callee_last_seg(func, body).as_deref() != Some("next") {
            continue;
        }
        let TyKind::RigidTy(RigidTy::Adt(def, _)) = body.locals()[destination.local].ty.kind()
        else {
            continue;
        };
        if !def.name().contains("Option") {
            continue;
        }
        let latch_block = bidx;
        let cond_block = target.ok_or("for-loop: diverging Iterator::next")?;
        let smir::TerminatorKind::SwitchInt { targets, .. } =
            &body.blocks[cond_block].terminator.kind
        else {
            return unsupported("for-loop: expected switchInt after Iterator::next");
        };
        let (mut body_block, mut exit_block) = (None, None);
        for (v, t) in targets.branches() {
            match v {
                0 => exit_block = Some(t),
                1 => body_block = Some(t),
                _ => {}
            }
        }
        let (Some(body_block), Some(exit_block)) = (body_block, exit_block) else {
            return unsupported("for-loop: unexpected switchInt shape (want None=0, Some=1)");
        };
        // The counter is the `k = ((opt as Some).0)` Downcast extraction in the body block.
        let counter = body.blocks[body_block]
            .statements
            .iter()
            .find_map(|s| {
                if let smir::StatementKind::Assign(
                    dest,
                    smir::Rvalue::Use(smir::Operand::Copy(p) | smir::Operand::Move(p)),
                ) = &s.kind
                    && p.projection
                        .iter()
                        .any(|e| matches!(e, smir::ProjectionElem::Downcast(_)))
                {
                    return Some(map_local(dest.local));
                }
                None
            })
            .ok_or("for-loop: couldn't find the loop variable (Some-arm Downcast)")?;
        // Trace next(&mut iter)'s arg through the alias chain to an into_iter dest, then its Range bounds.
        let (smir::Operand::Copy(p) | smir::Operand::Move(p)) =
            args.first().ok_or("for-loop: next has no self arg")?
        else {
            return unsupported("for-loop: next self arg is not a place");
        };
        let mut cur = p.local;
        let (mut iter_block, mut range_local) = (None, None);
        for _ in 0..64 {
            if let Some(&rl) = into_iter_arg.get(&cur) {
                iter_block = into_iter_block.get(&cur).copied();
                range_local = Some(rl);
                break;
            }
            match alias.get(&cur) {
                Some(&pred) => cur = pred,
                None => break,
            }
        }
        let iter_block = iter_block.ok_or("for-loop: couldn't trace next back to its into_iter")?;
        let range_local = range_local.unwrap();
        // A range loop reads (start, end) from `range_of`; a `for x in slice` loop has no range (the
        // `into_iter` arg is the slice), so its bound is `0..slice.len()`, materialized later.
        let (start, end, slice_local) = match range_of.get(&range_local).cloned() {
            Some((start, end)) => (start, end, None),
            None if is_slice_ref_ty(&body.locals()[range_local].ty) => (
                kir::Operand::Const(kir::Constant::Usize(0)),
                kir::Operand::Const(kir::Constant::Usize(0)), // unused: a slice loop's end is Len(slice)
                Some(map_local(range_local)),
            ),
            None => {
                return unsupported(
                    "for-loop: only `for k in a..b` (exclusive), `for k in a..=b` (inclusive), and \
                     `for x in slice` are supported - `.rev()`/`.step_by()` are not in the subset yet",
                );
            }
        };
        loops.push(RangeLoop {
            iter_block,
            latch_block,
            cond_block,
            body_block,
            exit_block,
            counter,
            start,
            end,
            inclusive: inclusive_ranges.contains(&range_local),
            slice_local,
        });
    }
    Ok(Loops {
        loops,
        support,
        new_call_blocks,
    })
}

/// Is `ty` a `RangeInclusive<_>` (the `a..=b` struct built by `RangeInclusive::new`)?
fn is_range_inclusive_ty(ty: &rustc_public::ty::Ty) -> bool {
    matches!(ty.kind(), TyKind::RigidTy(RigidTy::Adt(def, _)) if def.name().contains("RangeInclusive"))
}

/// Is `ty` a `&[T]`? `for x in slice` iterates such a param.
fn is_slice_ref_ty(ty: &rustc_public::ty::Ty) -> bool {
    matches!(
        ty.kind(),
        TyKind::RigidTy(RigidTy::Ref(_, inner, _))
            if matches!(inner.kind(), TyKind::RigidTy(RigidTy::Slice(_)))
    )
}

/// Is this local's type iterator plumbing (Range/Option, or a ref to one)? Such locals map to Unit.
fn is_iter_support_ty(ty: &rustc_public::ty::Ty) -> bool {
    fn is_plumbing_adt(ty: &rustc_public::ty::Ty) -> bool {
        if let TyKind::RigidTy(RigidTy::Adt(def, _)) = ty.kind() {
            let n = def.name();
            // `slice::Iter` is the `for x in slice` iterator, Range/RangeInclusive the range iterators, Option the `next()` result.
            n.contains("Range") || n.contains("Option") || n.contains("Iter")
        } else {
            false
        }
    }
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Ref(_, inner, _)) => is_plumbing_adt(&inner),
        _ => is_plumbing_adt(ty),
    }
}

/// Iterator-plumbing statements to drop: a write to a support local, a Range aggregate, a discriminant read, or the `k = ((opt as Some).0)` Downcast extraction.
fn stmt_is_iter_plumbing(
    support: &std::collections::HashSet<usize>,
    s: &smir::StatementKind,
) -> bool {
    let smir::StatementKind::Assign(dest, rv) = s else {
        return false;
    };
    if support.contains(&dest.local) {
        return true;
    }
    match rv {
        smir::Rvalue::Aggregate(..) | smir::Rvalue::Discriminant(_) => true,
        smir::Rvalue::Use(smir::Operand::Copy(p) | smir::Operand::Move(p)) => p
            .projection
            .iter()
            .any(|e| matches!(e, smir::ProjectionElem::Downcast(_))),
        _ => false,
    }
}

fn map_ty(ty: &rustc_public::ty::Ty) -> R<kir::Ty> {
    let TyKind::RigidTy(rigid) = ty.kind() else {
        return unsupported(format!("type {:?}", ty.kind()));
    };
    match rigid {
        RigidTy::Tuple(elems) if elems.is_empty() => Ok(kir::Ty::Unit),
        RigidTy::Bool => Ok(kir::Ty::Bool),
        RigidTy::Uint(UintTy::Usize) => Ok(kir::Ty::Usize),
        RigidTy::Int(IntTy::I32) => Ok(kir::Ty::I32),
        RigidTy::Uint(UintTy::U32) => Ok(kir::Ty::U32),
        RigidTy::Float(FloatTy::F32) => Ok(kir::Ty::F32),
        RigidTy::Float(FloatTy::F64) => Ok(kir::Ty::F64),
        RigidTy::Float(FloatTy::F16) => Ok(kir::Ty::F16),
        RigidTy::Ref(_, inner, mutbl) => Ok(kir::Ty::Ref {
            mutable: matches!(mutbl, smir::Mutability::Mut),
            pointee: Box::new(map_ty(&inner)?),
        }),
        // MIR accesses slices through raw pointers (`*const`/`*mut [f32]`); treat them like a slice reference.
        RigidTy::RawPtr(inner, mutbl) => Ok(kir::Ty::Ref {
            mutable: matches!(mutbl, smir::Mutability::Mut),
            pointee: Box::new(map_ty(&inner)?),
        }),
        RigidTy::Slice(inner) => Ok(kir::Ty::Slice(Box::new(map_ty(&inner)?))),
        other => unsupported(format!("type {other:?}")),
    }
}

fn map_local(l: smir::Local) -> kir::Local {
    kir::Local { index: l as u32 }
}

fn map_block(b: usize) -> kir::BlockId {
    kir::BlockId { index: b as u32 }
}

fn map_place(p: &smir::Place) -> R<kir::Place> {
    let projection = p
        .projection
        .iter()
        .map(|elem| match elem {
            smir::ProjectionElem::Deref => Ok(kir::ProjectionElem::Deref),
            smir::ProjectionElem::Index(l) => Ok(kir::ProjectionElem::Index(map_local(*l))),
            other => unsupported(format!("place projection {other:?}")),
        })
        .collect::<R<Vec<_>>>()?;
    Ok(kir::Place {
        local: map_local(p.local),
        projection,
    })
}

fn map_operand(op: &smir::Operand) -> R<kir::Operand> {
    match op {
        smir::Operand::Copy(p) => Ok(kir::Operand::Copy(map_place(p)?)),
        smir::Operand::Move(p) => Ok(kir::Operand::Move(map_place(p)?)),
        smir::Operand::Constant(c) => map_mir_const(c).map(kir::Operand::Const),
        other => unsupported(format!("operand {other:?}")),
    }
}

fn map_mir_const(c: &smir::ConstOperand) -> R<kir::Constant> {
    let ty = c.const_.ty();
    // A named const item (`const COLS: usize = 4`) appears as an Unevaluated const. The usize case
    // (loop bounds, strides) goes through the target-usize evaluator; other types fall through to the
    // Allocated path or are rejected.
    if matches!(c.const_.kind(), ConstantKind::Unevaluated(_))
        && matches!(ty.kind(), TyKind::RigidTy(RigidTy::Uint(UintTy::Usize)))
    {
        return c
            .const_
            .eval_target_usize()
            .map(kir::Constant::Usize)
            .map_err(|e| format!("evaluate usize const: {e:?}"));
    }
    let ConstantKind::Allocated(alloc) = c.const_.kind() else {
        return unsupported(format!("non-literal constant kind {:?}", c.const_.kind()));
    };
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => alloc
            .read_bool()
            .map(kir::Constant::Bool)
            .map_err(|e| format!("bool constant read failed: {e:?}")),
        TyKind::RigidTy(RigidTy::Uint(UintTy::Usize)) => alloc
            .read_uint()
            .map(|u| kir::Constant::Usize(u as u64))
            .map_err(|e| format!("usize constant read failed: {e:?}")),
        TyKind::RigidTy(RigidTy::Uint(UintTy::U32)) => alloc
            .read_uint()
            .map(|u| kir::Constant::U32(u as u32))
            .map_err(|e| format!("u32 constant read failed: {e:?}")),
        TyKind::RigidTy(RigidTy::Int(IntTy::I32)) => alloc
            .read_int()
            .map(|i| kir::Constant::I32(i as i32))
            .map_err(|e| format!("i32 constant read failed: {e:?}")),
        TyKind::RigidTy(RigidTy::Float(FloatTy::F32)) => {
            let bytes = alloc
                .raw_bytes()
                .map_err(|e| format!("f32 raw_bytes: {e:?}"))?;
            let arr: [u8; 4] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| "f32 const not 4 bytes")?;
            Ok(kir::Constant::F32(f32::from_le_bytes(arr)))
        }
        TyKind::RigidTy(RigidTy::Float(FloatTy::F64)) => {
            let bytes = alloc
                .raw_bytes()
                .map_err(|e| format!("f64 raw_bytes: {e:?}"))?;
            let arr: [u8; 8] = bytes
                .as_slice()
                .try_into()
                .map_err(|_| "f64 const not 8 bytes")?;
            Ok(kir::Constant::F64(f64::from_le_bytes(arr)))
        }
        other => unsupported(format!("constant of type {other:?}")),
    }
}

fn map_binop(op: smir::BinOp) -> R<kir::BinOp> {
    use smir::BinOp::*;
    Ok(match op {
        Add => kir::BinOp::Add,
        Sub => kir::BinOp::Sub,
        Mul => kir::BinOp::Mul,
        Div => kir::BinOp::Div,
        Rem => kir::BinOp::Rem,
        BitAnd => kir::BinOp::BitAnd,
        BitOr => kir::BinOp::BitOr,
        BitXor => kir::BinOp::BitXor,
        Shl => kir::BinOp::Shl,
        Shr => kir::BinOp::Shr,
        Lt => kir::BinOp::Lt,
        Le => kir::BinOp::Le,
        Gt => kir::BinOp::Gt,
        Ge => kir::BinOp::Ge,
        Ne => kir::BinOp::Ne,
        Eq => kir::BinOp::Eq,
        other => return unsupported(format!("binary op {other:?}")),
    })
}

fn map_rvalue(rv: &smir::Rvalue) -> R<kir::Rvalue> {
    match rv {
        smir::Rvalue::Use(op) => Ok(kir::Rvalue::Use(map_operand(op)?)),
        smir::Rvalue::BinaryOp(op, a, b) => Ok(kir::Rvalue::BinaryOp(
            map_binop(*op)?,
            map_operand(a)?,
            map_operand(b)?,
        )),
        smir::Rvalue::Len(p) => Ok(kir::Rvalue::Len(map_place(p)?)),
        // Modern MIR lowers `slice.len()` as `PtrMetadata(slice_ptr)` (the length half of the fat pointer) rather than `Len`; same meaning here.
        smir::Rvalue::UnaryOp(smir::UnOp::PtrMetadata, op) => match op {
            smir::Operand::Copy(p) | smir::Operand::Move(p) => Ok(kir::Rvalue::Len(map_place(p)?)),
            _ => unsupported("PtrMetadata of a non-place operand"),
        },
        smir::Rvalue::UnaryOp(smir::UnOp::Neg, op) => {
            Ok(kir::Rvalue::UnaryOp(kir::UnOp::Neg, map_operand(op)?))
        }
        smir::Rvalue::UnaryOp(smir::UnOp::Not, op) => {
            Ok(kir::Rvalue::UnaryOp(kir::UnOp::Not, map_operand(op)?))
        }
        // A reborrow (`&*p`) or raw address-of of a slice aliases the same slice pointer; model it as a copy of the dereffed place (MIR inserts these around slice length/indexing).
        smir::Rvalue::Ref(_, _, p) => Ok(kir::Rvalue::Use(kir::Operand::Copy(map_place(p)?))),
        smir::Rvalue::AddressOf(_, p) => Ok(kir::Rvalue::Use(kir::Operand::Copy(map_place(p)?))),
        smir::Rvalue::Cast(
            smir::CastKind::IntToInt
            | smir::CastKind::IntToFloat
            | smir::CastKind::FloatToInt
            | smir::CastKind::FloatToFloat,
            op,
            ty,
        ) => Ok(kir::Rvalue::Cast {
            to: map_ty(ty)?,
            operand: map_operand(op)?,
        }),
        other => unsupported(format!("rvalue {other:?}")),
    }
}

/// Map a statement, or `Ok(None)` for benign MIR-only no-ops.
fn map_statement(s: &smir::StatementKind) -> R<Option<kir::Statement>> {
    match s {
        smir::StatementKind::Assign(p, rv) => {
            Ok(Some(kir::Statement::Assign(map_place(p)?, map_rvalue(rv)?)))
        }
        smir::StatementKind::StorageLive(l) => Ok(Some(kir::Statement::StorageLive(map_local(*l)))),
        smir::StatementKind::StorageDead(l) => Ok(Some(kir::Statement::StorageDead(map_local(*l)))),
        smir::StatementKind::FakeRead(..)
        | smir::StatementKind::PlaceMention(..)
        | smir::StatementKind::AscribeUserType { .. }
        | smir::StatementKind::ConstEvalCounter
        | smir::StatementKind::Nop => Ok(None),
        other => unsupported(format!("statement {other:?}")),
    }
}

fn map_terminator(
    t: &smir::TerminatorKind,
    body: &smir::Body,
    param_count: u32,
    trap_code: Option<u32>,
    assert_trap_block: Option<kir::BlockId>,
) -> R<(Vec<kir::Statement>, kir::Terminator)> {
    match t {
        smir::TerminatorKind::Goto { target } => Ok((
            vec![],
            kir::Terminator::Goto {
                target: map_block(*target),
            },
        )),
        smir::TerminatorKind::Return => Ok((vec![], kir::Terminator::Return)),
        // The GPU has no panic path: an unreachable terminator is a real trap, not a graceful return (a
        // silent `Return` here would let the kernel finish with whatever it had computed so far, exactly
        // the erasure R468-007 flags).
        smir::TerminatorKind::Unreachable => {
            let code = trap_code
                .ok_or("internal error: Unreachable terminator with no trap code assigned")?;
            Ok((vec![], kir::Terminator::Trap { code }))
        }
        smir::TerminatorKind::SwitchInt { discr, targets } => {
            let branches = targets
                .branches()
                .map(|(v, bb)| (v, map_block(bb)))
                .collect();
            Ok((
                vec![],
                kir::Terminator::SwitchInt {
                    discr: map_operand(discr)?,
                    targets: kir::SwitchTargets {
                        branches,
                        otherwise: map_block(targets.otherwise()),
                    },
                },
            ))
        }
        // A bounds/overflow/divide/shift Assert: `cond == expected` goes to `target` (success), otherwise
        // traps (R468-007; a silent Goto to `target` regardless of `cond` was the erasure this replaces).
        // Always shaped as `branches: [(0, _)], otherwise: _` (never `[(1, _)]`) so codegen's bool-guard
        // fast path (`br i1`, not the untested-on-SPIR-V/AMDGCN general N-way `switch`) always applies:
        // with `expected == true` the 0 (false) branch is the trap and `target` is `otherwise`; with
        // `expected == false` it is the other way around.
        smir::TerminatorKind::Assert {
            cond,
            expected,
            target,
            ..
        } => {
            let trap = assert_trap_block.ok_or(
                "internal error: Assert terminator with no trap block reserved for this body",
            )?;
            let target = map_block(*target);
            let (zero_branch, otherwise) = if *expected {
                (trap, target)
            } else {
                (target, trap)
            };
            Ok((
                vec![],
                kir::Terminator::SwitchInt {
                    discr: map_operand(cond)?,
                    targets: kir::SwitchTargets {
                        branches: vec![(0, zero_branch)],
                        otherwise,
                    },
                },
            ))
        }
        smir::TerminatorKind::Call {
            func,
            args,
            destination,
            target,
            ..
        } => {
            // A `thread_index*()` dispatch-axis intrinsic becomes a `ThreadIndexCall` terminator.
            if let Some(dim) = thread_index_axis(func, body) {
                let target = target.ok_or("diverging thread_index*()")?;
                return Ok((
                    vec![],
                    kir::Terminator::ThreadIndexCall {
                        destination: map_place(destination)?,
                        dim,
                        target: map_block(target),
                    },
                ));
            }
            // A unary float-math method (`x.sqrt()`, `x.exp()`, `x.sin()`, ...) becomes a `MathUnary`
            // assign plus a Goto to the success target (softmax = exp, rmsnorm = `1.0 / x.sqrt()`,
            // rope = sin/cos). Match the qualified `f32::`/`f64::` path, not the bare name: a kernel
            // helper shadowing one (e.g. a NaN-propagating `fn max`, when `llvm.maxnum` suppresses NaN)
            // would otherwise be silently replaced by the intrinsic instead of rejected as an
            // unsupported call.
            if let Some(op) = callee_full_name(func, body)
                .as_deref()
                .and_then(std_math_method_name)
                .and_then(math_op_from_name)
            {
                let target = target.ok_or("diverging math intrinsic")?;
                let arg = args.first().ok_or("math intrinsic call has no argument")?;
                return Ok((
                    vec![kir::Statement::Assign(
                        map_place(destination)?,
                        kir::Rvalue::MathUnary(op, map_operand(arg)?),
                    )],
                    kir::Terminator::Goto {
                        target: map_block(target),
                    },
                ));
            }
            // A binary-math method (`x.max(y)`, `x.min(y)`) becomes a `BinaryOp` assign + Goto. The
            // 2-arg shape (self + other) distinguishes `f32::max` from `Iterator::max` (1 self arg).
            if args.len() == 2
                && let Some(op) = callee_full_name(func, body)
                    .as_deref()
                    .and_then(std_math_method_name)
                    .and_then(binary_math_op_from_name)
            {
                let target = target.ok_or("diverging binary-math intrinsic")?;
                return Ok((
                    vec![kir::Statement::Assign(
                        map_place(destination)?,
                        kir::Rvalue::BinaryOp(op, map_operand(&args[0])?, map_operand(&args[1])?),
                    )],
                    kir::Terminator::Goto {
                        target: map_block(target),
                    },
                ));
            }
            // `f32::from_bits(u32)` becomes a `Bitcast` to f32 (reinterprets the bits, not a numeric `as`
            // cast), so a kernel can read an f32 carried in a u32 buffer, e.g. an attention `scale` in the
            // u32 `Plan::ComputeMeta` dims buffer. One arg (no `self`).
            if args.len() == 1
                && callee_full_name(func, body)
                    .as_deref()
                    .and_then(std_math_method_name)
                    == Some("from_bits")
            {
                let target = target.ok_or("diverging from_bits")?;
                return Ok((
                    vec![kir::Statement::Assign(
                        map_place(destination)?,
                        kir::Rvalue::Bitcast {
                            to: kir::Ty::F32,
                            operand: map_operand(&args[0])?,
                        },
                    )],
                    kir::Terminator::Goto {
                        target: map_block(target),
                    },
                ));
            }
            // Workgroup-parallel (LDS) intrinsics (spec 054): `workgroup_barrier()` is a `Barrier`
            // terminator; `wg_write(array, idx, val)` / `wg_read(array, idx)` access the workgroup-local
            // array `array` (a const id; argmax uses 0 for values and 1 for indices).
            match callee_full_name(func, body)
                .as_deref()
                .and_then(kernel_intrinsic_name)
            {
                Some("workgroup_barrier") => {
                    let target = target.ok_or("diverging workgroup_barrier")?;
                    return Ok((
                        vec![],
                        kir::Terminator::Barrier {
                            target: map_block(target),
                        },
                    ));
                }
                Some("wg_write") if args.len() == 3 => {
                    let target = target.ok_or("diverging wg_write")?;
                    return Ok((
                        vec![kir::Statement::WorkgroupLocalWrite {
                            array: const_lds_array(&args[0])?,
                            idx: map_operand(&args[1])?,
                            value: map_operand(&args[2])?,
                        }],
                        kir::Terminator::Goto {
                            target: map_block(target),
                        },
                    ));
                }
                Some("wg_read") if args.len() == 2 => {
                    let target = target.ok_or("diverging wg_read")?;
                    return Ok((
                        vec![kir::Statement::Assign(
                            map_place(destination)?,
                            kir::Rvalue::WorkgroupLocalRead {
                                array: const_lds_array(&args[0])?,
                                idx: map_operand(&args[1])?,
                            },
                        )],
                        kir::Terminator::Goto {
                            target: map_block(target),
                        },
                    ));
                }
                // `atomic_add(buffer, index, value)` becomes a `GlobalAtomic { op: Add }` on the storage
                // buffer element `buffer[index]`, returning the old value (the destination). It gives
                // order-independent exact reductions (a top-p / event-counter kernel sums N integer
                // contributions to exactly N regardless of interleaving; a plain `+= 1` races).
                // Integer-only: float atomic add is order-dependent. `buffer` and `index` must be places
                // (a slice param and a `usize` local); the value may be any operand. The element type is
                // read off the slice by codegen's `element_ptr`.
                // `no_contract_add(a, b)` (card 675): the kernel source's explicit no-contraction marker,
                // lowered directly to `Rvalue::BinaryOpNoContract(BinOp::Add, a, b)` (the same per-op marker
                // `poot-kernelgen`'s Body API uses for the packed-dequant decode formula, card 628). `a` and
                // `b` are ordinary operands - MIR already materialized the preceding multiply into its own
                // temp before this call, so no new local is needed here. `Body::verify` rejects a non-float
                // operand or a marker on a non-Add/Sub/Mul op; this call site only ever produces `Add`.
                Some("no_contract_add") if args.len() == 2 => {
                    let target = target.ok_or("diverging no_contract_add")?;
                    return Ok((
                        vec![kir::Statement::Assign(
                            map_place(destination)?,
                            kir::Rvalue::BinaryOpNoContract(
                                kir::BinOp::Add,
                                map_operand(&args[0])?,
                                map_operand(&args[1])?,
                            ),
                        )],
                        kir::Terminator::Goto {
                            target: map_block(target),
                        },
                    ));
                }
                Some("atomic_add") if args.len() == 3 => {
                    let target = target.ok_or("diverging atomic_add")?;
                    let buf = place_operand_local(&args[0]).ok_or(
                        "atomic_add: the buffer arg must be a slice parameter (a place, not a value)",
                    )?;
                    let idx = place_operand_local(&args[1]).ok_or(
                        "atomic_add: the index arg must be a `usize` local (bind it with `let`), \
                         not a literal",
                    )?;
                    // A parameter local (1..=param_count) holds a `&mut [T]`, so indexing it needs
                    // `[Deref, Index]`, as a direct `buf[i] = ..` reaches codegen. A reborrow temp
                    // (`&mut *buf`) aliases the dereferenced slice (`normalize` propagates it to `*param`,
                    // which already carries the Deref), so it needs only `[Index]`; a leading Deref would double it.
                    let is_param = buf.index >= 1 && buf.index <= param_count;
                    let mut projection = if is_param {
                        vec![kir::ProjectionElem::Deref]
                    } else {
                        Vec::new()
                    };
                    projection.push(kir::ProjectionElem::Index(idx));
                    return Ok((
                        vec![kir::Statement::Assign(
                            map_place(destination)?,
                            kir::Rvalue::GlobalAtomic {
                                place: kir::Place {
                                    local: buf,
                                    projection,
                                },
                                value: map_operand(&args[2])?,
                                op: kir::AtomicOp::Add,
                            },
                        )],
                        kir::Terminator::Goto {
                            target: map_block(target),
                        },
                    ));
                }
                _ => {}
            }
            let callee = callee_last_seg(func, body).unwrap_or_else(|| "<opaque>".into());
            unsupported(format!(
                "call to `{callee}` (only thread_index*()/local_index()/group_index(), unary float-math \
                 like `x.sqrt()`, `x.max()`/`x.min()`, the LDS intrinsics \
                 `workgroup_barrier()`/`wg_read()`/`wg_write()`, `atomic_add(buffer, index, value)`, and \
                 `no_contract_add(a, b)` are allowed in a kernel)"
            ))
        }
        other => unsupported(format!("terminator {other:?}")),
    }
}

/// Map a unary float-math method's last path segment (`f32::sqrt` -> "sqrt") to its `MathOp`: the
/// `core`/`std` float methods that `MathUnary` lowers. Anything else returns `None` (the call is
/// rejected).
fn math_op_from_name(name: &str) -> Option<kir::MathOp> {
    Some(match name {
        "sqrt" => kir::MathOp::Sqrt,
        "abs" => kir::MathOp::Abs,
        "floor" => kir::MathOp::Floor,
        "ceil" => kir::MathOp::Ceil,
        "trunc" => kir::MathOp::Trunc,
        "round" => kir::MathOp::Round,
        "exp" => kir::MathOp::Exp,
        "ln" => kir::MathOp::Log, // f32::ln (natural log)
        "sin" => kir::MathOp::Sin,
        "cos" => kir::MathOp::Cos,
        _ => return None,
    })
}

/// Map a binary-math method's last path segment (`f32::max` -> "max") to its `BinOp`, lowered via `llvm.maxnum`/`llvm.minnum` (float) or `llvm.smax`/`umax` (int). `None` otherwise.
fn binary_math_op_from_name(name: &str) -> Option<kir::BinOp> {
    Some(match name {
        "max" => kir::BinOp::Max,
        "min" => kir::BinOp::Min,
        _ => return None,
    })
}

/// The shared kernel-intrinsics crate's name, as it appears in a resolved callee path
/// (`poot_kernel_intrinsics::<item>`). Kernel sources are compiled with `--extern
/// poot_kernel_intrinsics=<rlib>`, so a real call to one of its functions resolves to this crate; a
/// kernel-authored helper of the same bare name (declared in the kernel's own crate, or anywhere else)
/// never does, which is the fix for R468-009 (recognition used to be the callee's last path segment,
/// so any function named e.g. `workgroup_barrier` became the intrinsic).
const INTRINSICS_CRATE: &str = "poot_kernel_intrinsics";

/// The bare intrinsic name if `full_name` is `poot_kernel_intrinsics::<name>`; `None` for anything else,
/// including a same-named item declared anywhere but that one crate.
fn kernel_intrinsic_name(full_name: &str) -> Option<&str> {
    full_name.strip_prefix(INTRINSICS_CRATE)?.strip_prefix("::")
}

/// The callee's last path segment, for *diagnostics only* (the unsupported-call message names the
/// callee). Intrinsic recognition never uses this: see [`kernel_intrinsic_name`].
fn callee_last_seg(func: &smir::Operand, body: &smir::Body) -> Option<String> {
    let ty = func.ty(body.locals()).ok()?;
    let TyKind::RigidTy(RigidTy::FnDef(def, _)) = ty.kind() else {
        return None;
    };
    def.name().rsplit("::").next().map(|s| s.to_string())
}

/// Like [`callee_last_seg`] but the full qualified path (e.g. `"f32::sqrt"`), to distinguish a real
/// `f32`/`f64` method from a kernel function with the same bare name (see [`std_math_method_name`]).
fn callee_full_name(func: &smir::Operand, body: &smir::Body) -> Option<String> {
    let ty = func.ty(body.locals()).ok()?;
    let TyKind::RigidTy(RigidTy::FnDef(def, _)) = ty.kind() else {
        return None;
    };
    Some(def.name())
}

/// If `full_name` is a real `f32`/`f64` method (path ends in `f32::<method>` or `f64::<method>`),
/// return the bare method name; otherwise `None`. A bare last-segment match would also accept a
/// kernel helper of the same name in any module.
fn std_math_method_name(full_name: &str) -> Option<&str> {
    // Path shapes seen in pootc's kernel-import tests: an f32 inherent method is
    // `std::f32::<impl f32>::sqrt`; `.min()`/`.max()` on an index type (usize/i32/...) resolve through
    // `Ord` as `std::cmp::Ord::min`, with no type name in the path. A narrower "ends with
    // `<impl f32>`" check rejected the integer index clamps that `tiled_gemm_coarsened` uses. Both
    // shapes share what a shadowing kernel function lacks: a DefId in the `std`/`core` crate.
    if !(full_name.starts_with("std::") || full_name.starts_with("core::")) {
        return None;
    }
    full_name.rsplit_once("::").map(|(_, method)| method)
}

fn thread_index_axis(func: &smir::Operand, body: &smir::Body) -> Option<kir::IndexAxis> {
    let full = callee_full_name(func, body)?;
    axis_from_name(kernel_intrinsic_name(&full)?)
}

fn axis_from_name(name: &str) -> Option<kir::IndexAxis> {
    match name.rsplit("::").next()? {
        // Global thread id (the grid index).
        "thread_index" | "thread_index_x" => Some(kir::IndexAxis::X),
        "thread_index_y" => Some(kir::IndexAxis::Y),
        "thread_index_z" => Some(kir::IndexAxis::Z),
        // Lane id within the workgroup (`0..workgroup_size_x`), used to address per-lane LDS slots. Card 044 / spec 054.
        "local_index" | "local_index_x" => Some(kir::IndexAxis::LocalX),
        "local_index_y" => Some(kir::IndexAxis::LocalY),
        "local_index_z" => Some(kir::IndexAxis::LocalZ),
        // Workgroup id (the row a per-row LDS reduction owns).
        "group_index" | "group_index_x" => Some(kir::IndexAxis::GroupX),
        "group_index_y" => Some(kir::IndexAxis::GroupY),
        "group_index_z" => Some(kir::IndexAxis::GroupZ),
        _ => None,
    }
}
