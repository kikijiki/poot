use crate::helpers::{Alloc, arr_elem, copy, elem, guard, ld, local, slice_f32};
use poot_kernel_ir::{
    BasicBlock, BinOp, BlockId, Body, Constant, IndexAxis, Local, MathOp, Operand, Place, Rvalue,
    Statement, Terminator, Ty, WorkgroupLocalDecl,
};

/// Flash attention for decode (single query token), online softmax, one thread per head. Fuses
/// `matmul(q,kᵀ) -> scale -> +mask -> softmax -> matmul(p,v)` into one kernel: thread `h` streams the `cap`
/// cached keys keeping a running max `m`, denominator `l` and private accumulator `o[d]`; the `[.,cap]` scores
/// row is never materialized. GQA via `n_rep` (`kv = h/n_rep`). Layout: `q[h*d+i]`, `k`/`v[kv*cap*d + t*d + i]`,
/// `mask[t]` (additive), `out[h*d+i]`.
/// NVPTX-only (the `o[d]` private array crashes SpirvVulkan; see `array_sum_probe`). The scalar recurrence
/// matches `flash_decode_ref` in `tests/flash_ref.rs`.
///
/// `mask_per_head` selects the mask layout: `false` is the broadcast `[1,1,1,cap]` mask (`mask[t]`), `true` is
/// the per-head `[1,Hq,1,cap]` mask ALiBi needs (`mask[h*cap + t]`). The stride is baked into the kernel.
pub fn flash_attention_decode(
    name: &str,
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
    mask_per_head: bool,
) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        ld(slice_f32(false), false), // 2 k
        ld(slice_f32(false), false), // 3 v
        ld(slice_f32(false), false), // 4 mask
        ld(slice_f32(true), true),   // 5 out
    ]);
    let (q, k, v, mask, out) = (local(1), local(2), local(3), local(4), local(5));
    let o = al.add(
        Ty::Array {
            elem: Box::new(Ty::F32),
            len: d as u32,
        },
        true,
    );
    let h = al.add(Ty::Usize, false);
    let kv = al.add(Ty::Usize, true);
    let m = al.add(Ty::F32, true);
    let l = al.add(Ty::F32, true);
    let t = al.add(Ty::Usize, true);
    let dd = al.add(Ty::Usize, true);
    let s = al.add(Ty::F32, true);
    let cmp = al.add(Ty::Bool, false);
    let m_new = al.add(Ty::F32, true);
    let corr = al.add(Ty::F32, true);
    let e = al.add(Ty::F32, true);
    let idx = al.add(Ty::Usize, true);
    let idx2 = al.add(Ty::Usize, true);
    let base = al.add(Ty::Usize, true);
    let fa = al.add(Ty::F32, false);
    let fb = al.add(Ty::F32, false);
    let fc = al.add(Ty::F32, false);
    let cnt = al.add(Ty::Usize, false);
    // Allocated only on the per-head path, and last, so every other local keeps its index.
    let maskidx = mask_per_head.then(|| al.add(Ty::Usize, true));

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };
    let asg = |dst: Local, rv: Rvalue| Statement::Assign(Place::local(dst), rv);
    let mul = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(a)), copy(Place::local(b)))
    };
    let add = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(a)), copy(Place::local(b)))
    };

    // emit `dst = kv*(cap*d) + t*d + dd` (the k/v element index) via base/idx2 scratch.
    let kv_index = |stmts: &mut Vec<Statement>, dst: Local| {
        stmts.push(asg(
            base,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(kv)), cu(cap * d)),
        ));
        stmts.push(asg(
            idx2,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(t)), cu(d)),
        ));
        stmts.push(asg(base, add(base, idx2)));
        stmts.push(asg(dst, add(base, dd)));
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(h),
            dim: IndexAxis::X,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(h)), cu(hq)),
        )],
        terminator: guard(cmp, 17, 2),
    };
    let bb2 = BasicBlock {
        statements: vec![
            asg(
                kv,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(h)), cu(n_rep)),
            ),
            asg(m, Rvalue::Use(cf(f32::NEG_INFINITY))),
            asg(l, Rvalue::Use(cf(0.0))),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(3),
    };
    // o-init loop
    let bb3 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 5, 4),
    };
    let bb4 = BasicBlock {
        statements: vec![
            Statement::Assign(arr_elem(o, dd), Rvalue::Use(cf(0.0))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(3),
    };
    let bb5 = BasicBlock {
        statements: vec![asg(t, Rvalue::Use(cu(0)))],
        terminator: goto(6),
    };
    // t loop
    let bb6 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(t)), cu(cap)),
        )],
        terminator: guard(cmp, 14, 7),
    };
    let bb7 = BasicBlock {
        statements: vec![asg(s, Rvalue::Use(cf(0.0))), asg(dd, Rvalue::Use(cu(0)))],
        terminator: goto(8),
    };
    // dot-product loop
    let bb8 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 10, 9),
    };
    let mut bb9_st = vec![
        asg(
            base,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(d)),
        ),
        asg(idx, add(base, dd)),
        asg(fa, Rvalue::Use(copy(elem(q, idx)))),
    ];
    kv_index(&mut bb9_st, idx2);
    bb9_st.extend([
        asg(fb, Rvalue::Use(copy(elem(k, idx2)))),
        asg(fc, mul(fa, fb)),
        asg(s, add(s, fc)),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb9 = BasicBlock {
        statements: bb9_st,
        terminator: goto(8),
    };
    // online-softmax update; the mask read is `mask[t]` (broadcast) or `mask[h*cap + t]` (per-head, ALiBi).
    let mask_read = match maskidx {
        None => vec![asg(fa, Rvalue::Use(copy(elem(mask, t))))],
        Some(mi) => vec![
            asg(
                mi,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(cap)),
            ),
            asg(mi, add(mi, t)),
            asg(fa, Rvalue::Use(copy(elem(mask, mi)))),
        ],
    };
    let mut bb10_st = vec![asg(
        s,
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(s)), cf(scale)),
    )];
    bb10_st.extend(mask_read);
    bb10_st.extend([
        asg(s, add(s, fa)),
        asg(
            m_new,
            Rvalue::BinaryOp(BinOp::Max, copy(Place::local(m)), copy(Place::local(s))),
        ),
        // card 674: `m_new == -inf` means every position seen so far (including this one) is masked -
        // this lane has contributed nothing yet. `exp(m - m_new)`/`exp(s - m_new)` would then be
        // `exp(-inf - -inf) = exp(NaN) = NaN` (IEEE754 inf-inf), poisoning `l`/`o` even though the
        // correct contribution here is exactly zero (matching the CPU oracle, `ops::attention::flash_decode`,
        // which never hits this subtraction because it takes one pass over the whole row instead of a
        // running max). Route around the NaN: an empty running state contributes `corr=e=0` directly.
        asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(m_new)), cf(f32::NEG_INFINITY)),
        ),
    ]);
    let bb10 = BasicBlock {
        statements: bb10_st,
        terminator: guard(cmp, 19, 18),
    };
    // bb18 (m_new == -inf): nothing seen yet; this position contributes zero weight.
    let bb18 = BasicBlock {
        statements: vec![
            asg(corr, Rvalue::Use(cf(0.0))),
            asg(e, Rvalue::Use(cf(0.0))),
        ],
        terminator: goto(20),
    };
    // bb19 (m_new finite): the ordinary online-softmax rescale.
    let bb19 = BasicBlock {
        statements: vec![
            asg(
                fa,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(m)), copy(Place::local(m_new))),
            ),
            asg(corr, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fa)))),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(s)), copy(Place::local(m_new))),
            ),
            asg(e, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fb)))),
        ],
        terminator: goto(20),
    };
    // bb20: join - fold corr/e into l regardless of which branch produced them.
    let bb20 = BasicBlock {
        statements: vec![
            asg(l, mul(l, corr)),
            asg(l, add(l, e)),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(11),
    };
    // o-accumulation loop
    let bb11 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 13, 12),
    };
    let mut bb12_st = vec![
        asg(fa, Rvalue::Use(copy(arr_elem(o, dd)))),
        asg(fa, mul(fa, corr)),
    ];
    kv_index(&mut bb12_st, idx2);
    bb12_st.extend([
        asg(fb, Rvalue::Use(copy(elem(v, idx2)))),
        asg(fb, mul(e, fb)),
        asg(fc, add(fa, fb)),
        Statement::Assign(arr_elem(o, dd), Rvalue::Use(copy(Place::local(fc)))),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb12 = BasicBlock {
        statements: bb12_st,
        terminator: goto(11),
    };
    let bb13 = BasicBlock {
        statements: vec![
            asg(m, Rvalue::Use(copy(Place::local(m_new)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(t)), cu(1)),
            ),
            asg(t, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(6),
    };
    // epilogue: out[h*d+dd] = o[dd]/l
    let bb14 = BasicBlock {
        statements: vec![asg(dd, Rvalue::Use(cu(0)))],
        terminator: goto(15),
    };
    let bb15 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 17, 16),
    };
    let bb16 = BasicBlock {
        statements: vec![
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(d)),
            ),
            asg(idx, add(base, dd)),
            asg(fa, Rvalue::Use(copy(arr_elem(o, dd)))),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(fa)), copy(Place::local(l))),
            ),
            Statement::Assign(elem(out, idx), Rvalue::Use(copy(Place::local(fb)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(15),
    };
    let bb17 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    Body::new(
        name,
        5,
        al.locals,
        vec![
            bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11, bb12, bb13, bb14, bb15,
            bb16, bb17, bb18, bb19, bb20,
        ],
    )
}

/// Synthesizer flash-attention decode generator: the online softmax of [`flash_attention_decode`], but
/// one workgroup of `w` lanes per (batch,head) instead of one thread per head. Each lane strides over
/// `t in lane, lane+w, lane+2w, ..` keeping its own `(m, l, o[D])`; an LDS epilogue merges the `w` partials with
/// the online-softmax combine rule (`m = max(m1,m2)`, `l = l1*exp(m1-m) + l2*exp(m2-m)`,
/// `o = o1*exp(m1-m) + o2*exp(m2-m)`) before the final `o/l` divide. `o[D]` lives in LDS (array 0, `w*D` wide:
/// lane `i`'s partial is at `[i*D .. i*D+D)`) because a dynamically-indexed private array crashes the SPIR-V
/// backend (spec 061); `m`/`l` partials live in LDS arrays 1/2 (`w` wide each). The per-lane loops are
/// barrier-free (disjoint LDS regions); the single barrier is on the straight-line path after them, reached by
/// every lane exactly once (same convergent shape as `gemv_lds`'s reduction, safe on RADV and ROCm). After it
/// only lane 0 continues: it folds the `w` partials into its own `[0..D)` slot (cost `O(D)` per lane), divides by
/// the merged `l` and stores `out`. GQA via `kv = h / n_rep`.
/// Params: `_1 q` \[B,Hq,1,D\], `_2 k`/`_3 v` \[B,Hkv,cap,D\], `_4 mask` \[B,Hm,1,cap\] (additive), `out` \[B,Hq,1,D\].
/// Grid: `B*Hq` workgroups of `w` lanes (`dispatch` with `[B*Hq*w,1,1]` threads, wg `[w,1,1]`).
///
/// `mask_per_head` selects `Hm`: `false` is the broadcast `[B,1,1,cap]` mask (`maskbase = bi*cap`), `true` is
/// the per-head `[B,Hq,1,cap]` mask ALiBi needs (`maskbase = bi*Hq*cap + h*cap`). Strides are baked.
#[allow(clippy::too_many_arguments)]
pub fn flash_region_decode(
    name: &str,
    bsz: usize,
    hq: usize,
    n_rep: usize,
    cap: usize,
    d: usize,
    scale: f32,
    w: usize,
    mask_per_head: bool,
) -> Body {
    let hkv = hq / n_rep;
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        ld(slice_f32(false), false), // 2 k
        ld(slice_f32(false), false), // 3 v
        ld(slice_f32(false), false), // 4 mask
        ld(slice_f32(true), true),   // 5 out
    ]);
    let (q, k, v, mask, out) = (local(1), local(2), local(3), local(4), local(5));
    // One workgroup per (batch row, query head): gi = GroupX in 0..B*Hq; bi = gi/Hq, h = gi%Hq.
    // q\[B,Hq,1,D\], k/v\[B,Hkv,cap,D\], mask[B,1,1,cap]. `lane` = LocalX in 0..w, this lane's slice of the KV loop.
    let gi = al.add(Ty::Usize, false);
    let lane = al.add(Ty::Usize, false);
    let bi = al.add(Ty::Usize, false);
    let h = al.add(Ty::Usize, false);
    let qbase = al.add(Ty::Usize, false);
    let kvbase = al.add(Ty::Usize, false);
    let maskbase = al.add(Ty::Usize, false);
    let loff = al.add(Ty::Usize, false); // lane*D - this lane's base offset into the o-partials LDS array
    let kv = al.add(Ty::Usize, true);
    let m = al.add(Ty::F32, true);
    let l = al.add(Ty::F32, true);
    let t = al.add(Ty::Usize, true);
    let dd = al.add(Ty::Usize, true);
    let s = al.add(Ty::F32, true);
    let cmp = al.add(Ty::Bool, false);
    let m_new = al.add(Ty::F32, true);
    let corr = al.add(Ty::F32, true);
    let e = al.add(Ty::F32, true);
    let idx = al.add(Ty::Usize, true);
    let idx2 = al.add(Ty::Usize, true);
    let base = al.add(Ty::Usize, true);
    let fa = al.add(Ty::F32, false);
    let fb = al.add(Ty::F32, false);
    let fc = al.add(Ty::F32, false);
    let cnt = al.add(Ty::Usize, false);
    let oidx = al.add(Ty::Usize, true); // scratch: this lane's (or, post-barrier, the merge's) o-array index
    // post-barrier merge state (lane 0 only): rm/rl are the running (m,l), seeded from lane 0's own partial
    // (which doubles as the running `o` accumulator at LDS array-0 slot [0..D)).
    let zero_u = al.add(Ty::Usize, false);
    let rm = al.add(Ty::F32, true);
    let rl = al.add(Ty::F32, true);
    let ii = al.add(Ty::Usize, true); // the other-lane index being folded in, 1..w
    let coff = al.add(Ty::Usize, true); // ii*D - the lane-`ii` partial's base offset
    let mi_l = al.add(Ty::F32, false);
    let li_l = al.add(Ty::F32, false);
    let corr_a = al.add(Ty::F32, false); // exp(rm_old - m_merged): rescale for the running accumulator
    let corr_b = al.add(Ty::F32, false); // exp(m_i - m_merged): rescale for lane ii's partial

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };
    let asg = |dst: Local, rv: Rvalue| Statement::Assign(Place::local(dst), rv);
    let mul = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(a)), copy(Place::local(b)))
    };
    let add = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(a)), copy(Place::local(b)))
    };
    // LDS read/write of the o-partials array (array 0, `w*D` wide; lane 0's `[0..D)` slot is also the
    // post-barrier running-merge accumulator).
    let o_read = |idx_l: Local| -> Rvalue {
        Rvalue::WorkgroupLocalRead {
            idx: copy(Place::local(idx_l)),
            array: 0,
        }
    };
    let o_write = |idx_l: Local, val_l: Local| -> Statement {
        Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(idx_l)),
            value: copy(Place::local(val_l)),
            array: 0,
        }
    };
    // LDS read/write of the per-lane m/l partials (arrays 1/2, `w` wide each).
    let m_read = |idx_l: Local| -> Rvalue {
        Rvalue::WorkgroupLocalRead {
            idx: copy(Place::local(idx_l)),
            array: 1,
        }
    };
    let m_write = |idx_l: Local, val_l: Local| -> Statement {
        Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(idx_l)),
            value: copy(Place::local(val_l)),
            array: 1,
        }
    };
    let l_read = |idx_l: Local| -> Rvalue {
        Rvalue::WorkgroupLocalRead {
            idx: copy(Place::local(idx_l)),
            array: 2,
        }
    };
    let l_write = |idx_l: Local, val_l: Local| -> Statement {
        Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(idx_l)),
            value: copy(Place::local(val_l)),
            array: 2,
        }
    };
    // emit `dst = kvbase + t*d + dd` (the k/v element index, kvbase = (bi*Hkv+kv)*cap*d) via base/idx2.
    let kv_index = |stmts: &mut Vec<Statement>, dst: Local| {
        stmts.push(asg(
            idx2,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(t)), cu(d)),
        ));
        stmts.push(asg(base, add(kvbase, idx2)));
        stmts.push(asg(dst, add(base, dd)));
    };

    // bb0: gi = GroupX (one workgroup per (batch,head)).
    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gi),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(gi)), cu(bsz * hq)),
        )],
        terminator: guard(cmp, 26, 2),
    };
    // bb2: lane = LocalX (this workgroup's `w` lanes cooperate on gi's KV reduction).
    let bb2 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(lane),
            dim: IndexAxis::LocalX,
            target: BlockId { index: 3 },
        },
    };
    // bi = gi/Hq; h = gi%Hq; kv = h/n_rep; qbase = (bi*Hq+h)*D; kvbase = (bi*Hkv+kv)*cap*D; maskbase = bi*cap;
    // loff = lane*D.
    let mut bb3 = BasicBlock {
        statements: vec![
            asg(
                bi,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(gi)), cu(hq)),
            ),
            asg(
                h,
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(gi)), cu(hq)),
            ),
            asg(
                kv,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(h)), cu(n_rep)),
            ),
            // qbase = (bi*Hq + h) * D
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(bi)), cu(hq)),
            ),
            asg(base, add(base, h)),
            asg(
                qbase,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(base)), cu(d)),
            ),
            // kvbase = (bi*Hkv + kv) * cap * D
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(bi)), cu(hkv)),
            ),
            asg(base, add(base, kv)),
            asg(
                kvbase,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(base)), cu(cap * d)),
            ),
            // maskbase = bi*cap (broadcast); the per-head form adds `+ h*cap` with a bi*Hq*cap batch stride, appended
            // just below so the broadcast statement list is unchanged.
            asg(
                maskbase,
                Rvalue::BinaryOp(
                    BinOp::Mul,
                    copy(Place::local(bi)),
                    cu(if mask_per_head { hq * cap } else { cap }),
                ),
            ),
        ],
        terminator: goto(4),
    };
    if mask_per_head {
        bb3.statements.extend([
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(cap)),
            ),
            asg(maskbase, add(maskbase, base)),
        ]);
    }
    bb3.statements.extend([
        asg(
            loff,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(lane)), cu(d)),
        ),
        asg(m, Rvalue::Use(cf(f32::NEG_INFINITY))),
        asg(l, Rvalue::Use(cf(0.0))),
        asg(dd, Rvalue::Use(cu(0))),
    ]);
    // o-init loop: o[loff+dd] = 0 (this lane's slice of the o-partials array)
    let bb4 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 6, 5),
    };
    let bb5 = BasicBlock {
        statements: vec![
            asg(oidx, add(loff, dd)),
            asg(fa, Rvalue::Use(cf(0.0))),
            o_write(oidx, fa),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(4),
    };
    // t (KV) loop: this lane's strided slice, t = lane, lane+w, lane+2w, ..
    let bb6 = BasicBlock {
        statements: vec![asg(t, Rvalue::Use(copy(Place::local(lane))))],
        terminator: goto(7),
    };
    let bb7 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(t)), cu(cap)),
        )],
        terminator: guard(cmp, 15, 8),
    };
    let bb8 = BasicBlock {
        statements: vec![asg(s, Rvalue::Use(cf(0.0))), asg(dd, Rvalue::Use(cu(0)))],
        terminator: goto(9),
    };
    // dot-product loop: s += q[h*d+dd] * k[kv,t,dd]
    let bb9 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 11, 10),
    };
    let mut bb10_st = vec![
        asg(idx, add(qbase, dd)),
        asg(fa, Rvalue::Use(copy(elem(q, idx)))),
    ];
    kv_index(&mut bb10_st, idx2);
    bb10_st.extend([
        asg(fb, Rvalue::Use(copy(elem(k, idx2)))),
        asg(fc, mul(fa, fb)),
        asg(s, add(s, fc)),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb10 = BasicBlock {
        statements: bb10_st,
        terminator: goto(9),
    };
    // online-softmax update (this lane's local recurrence): s = s*scale + mask[maskbase + t];
    // m_new=max(m,s); corr=exp(m-m_new); e=exp(s-m_new); l = l*corr + e.
    //
    // card 674: when `m_new == -inf` this lane has seen nothing but masked positions so far
    // (including this one) - `exp(m - m_new)`/`exp(s - m_new)` would be `exp(-inf - -inf) = NaN`.
    // Route around it: an empty running state contributes `corr=e=0` (matches the CPU oracle,
    // `ops::attention::flash_decode`, which never subtracts two infinities because it takes one pass
    // over the whole row instead of a running max).
    let bb11 = BasicBlock {
        statements: vec![
            asg(
                s,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(s)), cf(scale)),
            ),
            asg(idx, add(maskbase, t)),
            asg(fa, Rvalue::Use(copy(elem(mask, idx)))),
            asg(s, add(s, fa)),
            asg(
                m_new,
                Rvalue::BinaryOp(BinOp::Max, copy(Place::local(m)), copy(Place::local(s))),
            ),
            asg(
                cmp,
                Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(m_new)), cf(f32::NEG_INFINITY)),
            ),
        ],
        terminator: guard(cmp, 28, 27),
    };
    // bb27 (m_new == -inf): this position contributes zero weight.
    let bb27 = BasicBlock {
        statements: vec![
            asg(corr, Rvalue::Use(cf(0.0))),
            asg(e, Rvalue::Use(cf(0.0))),
        ],
        terminator: goto(29),
    };
    // bb28 (m_new finite): the ordinary online-softmax rescale.
    let bb28 = BasicBlock {
        statements: vec![
            asg(
                fa,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(m)), copy(Place::local(m_new))),
            ),
            asg(corr, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fa)))),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(s)), copy(Place::local(m_new))),
            ),
            asg(e, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fb)))),
        ],
        terminator: goto(29),
    };
    // bb29: join - fold corr/e into l regardless of which branch produced them.
    let bb29 = BasicBlock {
        statements: vec![
            asg(l, mul(l, corr)),
            asg(l, add(l, e)),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(12),
    };
    // o-accumulation loop: o[loff+dd] = o[loff+dd]*corr + e * v[kv,t,dd]
    let bb12 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 14, 13),
    };
    let mut bb13_st = vec![
        asg(oidx, add(loff, dd)),
        asg(fa, o_read(oidx)),
        asg(fa, mul(fa, corr)),
    ];
    kv_index(&mut bb13_st, idx2);
    bb13_st.extend([
        asg(fb, Rvalue::Use(copy(elem(v, idx2)))),
        asg(fb, mul(e, fb)),
        asg(fc, add(fa, fb)),
        o_write(oidx, fc),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb13 = BasicBlock {
        statements: bb13_st,
        terminator: goto(12),
    };
    let bb14 = BasicBlock {
        statements: vec![
            asg(m, Rvalue::Use(copy(Place::local(m_new)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(t)), cu(w)),
            ),
            asg(t, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(7),
    };
    // t-loop exhausted: flush this lane's final (m,l) into the LDS partials, then the one barrier. Every lane
    // reaches it exactly once (uniform control flow), and it is not inside a loop.
    let bb15 = BasicBlock {
        statements: vec![m_write(lane, m), l_write(lane, l)],
        terminator: Terminator::Barrier {
            target: BlockId { index: 16 },
        },
    };
    // lane != 0 is done (its partial has been merged by lane 0); lane 0 runs the serial fold.
    let bb16 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(lane)), cu(0)),
        )],
        terminator: guard(cmp, 26, 17),
    };
    // seed the running merge state from lane 0's own partial (already sitting at o-array slot [0..D)).
    let bb17 = BasicBlock {
        statements: vec![
            asg(zero_u, Rvalue::Use(cu(0))),
            asg(rm, m_read(zero_u)),
            asg(rl, l_read(zero_u)),
            asg(ii, Rvalue::Use(cu(1))),
        ],
        terminator: goto(18),
    };
    // combine loop header: fold in lane `ii`'s partial, ii = 1..w.
    let bb18 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(ii)), cu(w)),
        )],
        terminator: guard(cmp, 23, 19),
    };
    // merge scalars: m_i=m_read(ii); l_i=l_read(ii); m_merged=max(rm,m_i); corr_a=exp(rm-m_merged);
    // corr_b=exp(m_i-m_merged); rl = rl*corr_a + l_i*corr_b; coff = ii*D.
    //
    // card 674: `rm`/`mi_l` are both `-inf` exactly when neither the running accumulator nor lane
    // `ii`'s partial has seen an unmasked position yet - `exp(rm - m_merged)`/`exp(mi_l - m_merged)`
    // would be `exp(-inf - -inf) = NaN`. Both sides already carry `l=0`/`o=0` in that state (by
    // induction from this same guard at the per-lane update above), so `corr_a=corr_b=0` merges them
    // into "still empty" without ever subtracting two infinities.
    let bb19 = BasicBlock {
        statements: vec![
            asg(mi_l, m_read(ii)),
            asg(li_l, l_read(ii)),
            asg(
                m_new,
                Rvalue::BinaryOp(BinOp::Max, copy(Place::local(rm)), copy(Place::local(mi_l))),
            ),
            asg(
                cmp,
                Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(m_new)), cf(f32::NEG_INFINITY)),
            ),
        ],
        terminator: guard(cmp, 31, 30),
    };
    // bb30 (m_merged == -inf): neither side has seen an unmasked position; merge stays empty.
    let bb30 = BasicBlock {
        statements: vec![
            asg(corr_a, Rvalue::Use(cf(0.0))),
            asg(corr_b, Rvalue::Use(cf(0.0))),
        ],
        terminator: goto(32),
    };
    // bb31 (m_merged finite): the ordinary online-softmax merge rescale.
    let bb31 = BasicBlock {
        statements: vec![
            asg(
                fa,
                Rvalue::BinaryOp(
                    BinOp::Sub,
                    copy(Place::local(rm)),
                    copy(Place::local(m_new)),
                ),
            ),
            asg(
                corr_a,
                Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fa))),
            ),
            asg(
                fb,
                Rvalue::BinaryOp(
                    BinOp::Sub,
                    copy(Place::local(mi_l)),
                    copy(Place::local(m_new)),
                ),
            ),
            asg(
                corr_b,
                Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fb))),
            ),
        ],
        terminator: goto(32),
    };
    // bb32: join - fold the merge scalars into rl regardless of which branch produced them.
    let bb32 = BasicBlock {
        statements: vec![
            asg(rl, mul(rl, corr_a)),
            asg(fc, mul(li_l, corr_b)),
            asg(rl, add(rl, fc)),
            asg(
                coff,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(ii)), cu(d)),
            ),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(20),
    };
    // nested o-merge loop: running_o[dd] (o-array slot [0..D), the accumulator) =
    // running_o[dd]*corr_a + o_partial[ii][dd]*corr_b.
    let bb20 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 22, 21),
    };
    let bb21 = BasicBlock {
        statements: vec![
            asg(fa, o_read(dd)), // running accumulator (lane 0's own slot, index 0*D+dd = dd)
            asg(fa, mul(fa, corr_a)),
            asg(oidx, add(coff, dd)),
            asg(fb, o_read(oidx)), // lane ii's partial
            asg(fb, mul(fb, corr_b)),
            asg(fc, add(fa, fb)),
            o_write(dd, fc),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(20),
    };
    let bb22 = BasicBlock {
        statements: vec![
            asg(rm, Rvalue::Use(copy(Place::local(m_new)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(ii)), cu(1)),
            ),
            asg(ii, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(18),
    };
    // epilogue: out[qbase+dd] = o[dd] / rl (the fully-merged running accumulator / combined denominator).
    let bb23 = BasicBlock {
        statements: vec![
            asg(l, Rvalue::Use(copy(Place::local(rl)))),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(24),
    };
    let bb24 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 26, 25),
    };
    let bb25 = BasicBlock {
        statements: vec![
            asg(idx, add(qbase, dd)),
            asg(fa, o_read(dd)),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(fa)), copy(Place::local(l))),
            ),
            Statement::Assign(elem(out, idx), Rvalue::Use(copy(Place::local(fb)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(24),
    };
    let bb26 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(
        name,
        5,
        al.locals,
        vec![
            bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11, bb12, bb13, bb14, bb15,
            bb16, bb17, bb18, bb19, bb20, bb21, bb22, bb23, bb24, bb25, bb26, bb27, bb28, bb29,
            bb30, bb31, bb32,
        ],
    );
    body.workgroup_size = [w as u32, 1, 1];
    body.workgroup_locals = vec![
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: (w * d) as u32,
        },
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: w as u32,
        },
        WorkgroupLocalDecl {
            elem_ty: Ty::F32,
            len: w as u32,
        },
    ];
    body
}

/// Synthesizer flash prefill generator: the L-queries analog of [`flash_region_decode`]. Every query row
/// attends to the whole-prompt KV with a masked online softmax, `o[D]` in LDS. One workgroup per (query head,
/// query row) (`workgroup_size = 1`); the flat `(head, row)` index is laid over a 2-D workgroup grid
/// `gi = GroupY*x_groups + GroupX` so `Hq*L` can exceed the wgpu gridDim.x cap at long context. RADV-safe (LDS,
/// no barrier). Never materializes the `[1,Hq,L,L]` scores (peak attention memory O(L*D)).
/// Params: `_1 q` \[Hq*L*D\], `_2 k`/`_3 v` \[Hkv*L*D\], `_4 mask` \[Hm*L*L\] (causal, additive), `out` \[Hq*L*D\].
/// Grid: `Hq*L` workgroups (1 thread each) over the 2-D `[x_groups, y_groups]` layout. GQA via `kv = h / n_rep`.
///
/// `mask_per_head` selects `Hm`: `false` is the broadcast `[1,1,L,L]` mask (`maskbase = row*L`), `true` is
/// the per-head `[1,Hq,L,L]` mask ALiBi needs (`maskbase = h*L*L + row*L`). The stride is baked.
#[allow(clippy::too_many_arguments)]
pub fn flash_region_prefill(
    name: &str,
    hq: usize,
    l: usize,
    d: usize,
    n_rep: usize,
    scale: f32,
    x_groups: usize,
    mask_per_head: bool,
) -> Body {
    let mut al = Alloc::new(vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        ld(slice_f32(false), false), // 2 k
        ld(slice_f32(false), false), // 3 v
        ld(slice_f32(false), false), // 4 mask
        ld(slice_f32(true), true),   // 5 out
    ]);
    let (q, k, v, mask, out) = (local(1), local(2), local(3), local(4), local(5));
    // o[D] in LDS array 0 (this (head,row)'s region).
    let gx = al.add(Ty::Usize, false);
    let gy = al.add(Ty::Usize, false);
    let gi = al.add(Ty::Usize, true);
    let h = al.add(Ty::Usize, false);
    let row = al.add(Ty::Usize, false);
    let kv = al.add(Ty::Usize, true);
    let qbase = al.add(Ty::Usize, false);
    let maskbase = al.add(Ty::Usize, false);
    let m = al.add(Ty::F32, true);
    let ll = al.add(Ty::F32, true);
    let t = al.add(Ty::Usize, true);
    let dd = al.add(Ty::Usize, true);
    let s = al.add(Ty::F32, true);
    let cmp = al.add(Ty::Bool, false);
    let m_new = al.add(Ty::F32, true);
    let corr = al.add(Ty::F32, true);
    let e = al.add(Ty::F32, true);
    let idx = al.add(Ty::Usize, true);
    let idx2 = al.add(Ty::Usize, true);
    let base = al.add(Ty::Usize, true);
    let fa = al.add(Ty::F32, false);
    let fb = al.add(Ty::F32, false);
    let fc = al.add(Ty::F32, false);
    let cnt = al.add(Ty::Usize, false);

    let cu = |x: usize| Operand::Const(Constant::Usize(x as u64));
    let cf = |x: f32| Operand::Const(Constant::F32(x));
    let goto = |b: u32| Terminator::Goto {
        target: BlockId { index: b },
    };
    let asg = |dst: Local, rv: Rvalue| Statement::Assign(Place::local(dst), rv);
    let mul = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(a)), copy(Place::local(b)))
    };
    let add = |a: Local, b: Local| {
        Rvalue::BinaryOp(BinOp::Add, copy(Place::local(a)), copy(Place::local(b)))
    };
    let o_read = |idx_l: Local| -> Rvalue {
        Rvalue::WorkgroupLocalRead {
            idx: copy(Place::local(idx_l)),
            array: 0,
        }
    };
    let o_write = |idx_l: Local, val_l: Local| -> Statement {
        Statement::WorkgroupLocalWrite {
            idx: copy(Place::local(idx_l)),
            value: copy(Place::local(val_l)),
            array: 0,
        }
    };
    // emit `dst = kv*(L*d) + t*d + dd` (the k/v element index).
    let kv_index = |stmts: &mut Vec<Statement>, dst: Local| {
        stmts.push(asg(
            base,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(kv)), cu(l * d)),
        ));
        stmts.push(asg(
            idx2,
            Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(t)), cu(d)),
        ));
        stmts.push(asg(base, add(base, idx2)));
        stmts.push(asg(dst, add(base, dd)));
    };

    let bb0 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gx),
            dim: IndexAxis::GroupX,
            target: BlockId { index: 1 },
        },
    };
    let bb1 = BasicBlock {
        statements: vec![],
        terminator: Terminator::ThreadIndexCall {
            destination: Place::local(gy),
            dim: IndexAxis::GroupY,
            target: BlockId { index: 2 },
        },
    };
    // gi = gy*x_groups + gx (the flat head*L + row index, laid over the 2-D grid).
    let bb2 = BasicBlock {
        statements: vec![
            asg(
                gi,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(gy)), cu(x_groups)),
            ),
            asg(gi, add(gi, gx)),
            asg(
                cmp,
                Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(gi)), cu(hq * l)),
            ),
        ],
        terminator: guard(cmp, 18, 3),
    };
    // h = gi/L; row = gi%L; kv = h/n_rep; qbase = (h*L+row)*D; maskbase = row*L; init m/ll.
    let mut bb3 = BasicBlock {
        statements: vec![
            asg(
                h,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(gi)), cu(l)),
            ),
            asg(
                row,
                Rvalue::BinaryOp(BinOp::Rem, copy(Place::local(gi)), cu(l)),
            ),
            asg(
                kv,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(h)), cu(n_rep)),
            ),
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(l)),
            ),
            asg(base, add(base, row)),
            asg(
                qbase,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(base)), cu(d)),
            ),
            asg(
                maskbase,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(row)), cu(l)),
            ),
        ],
        terminator: goto(4),
    };
    // per-head ALiBi mask: maskbase += h*L*L, appended so the broadcast statement list above is unchanged.
    if mask_per_head {
        bb3.statements.extend([
            asg(
                base,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(h)), cu(l * l)),
            ),
            asg(maskbase, add(maskbase, base)),
        ]);
    }
    bb3.statements.extend([
        asg(m, Rvalue::Use(cf(f32::NEG_INFINITY))),
        asg(ll, Rvalue::Use(cf(0.0))),
        asg(dd, Rvalue::Use(cu(0))),
    ]);
    // o-init loop
    let bb4 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 6, 5),
    };
    let bb5 = BasicBlock {
        statements: vec![
            asg(fa, Rvalue::Use(cf(0.0))),
            o_write(dd, fa),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(4),
    };
    let bb6 = BasicBlock {
        statements: vec![asg(t, Rvalue::Use(cu(0)))],
        terminator: goto(7),
    };
    // t (KV) loop over L
    let bb7 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(t)), cu(l)),
        )],
        terminator: guard(cmp, 15, 8),
    };
    let bb8 = BasicBlock {
        statements: vec![asg(s, Rvalue::Use(cf(0.0))), asg(dd, Rvalue::Use(cu(0)))],
        terminator: goto(9),
    };
    // dot loop: s += q[qbase+dd] * k[kv,t,dd]
    let bb9 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 11, 10),
    };
    let mut bb10_st = vec![
        asg(idx, add(qbase, dd)),
        asg(fa, Rvalue::Use(copy(elem(q, idx)))),
    ];
    kv_index(&mut bb10_st, idx2);
    bb10_st.extend([
        asg(fb, Rvalue::Use(copy(elem(k, idx2)))),
        asg(fc, mul(fa, fb)),
        asg(s, add(s, fc)),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb10 = BasicBlock {
        statements: bb10_st,
        terminator: goto(9),
    };
    // online softmax: s = s*scale + mask[row*L + t]; m_new=max(m,s); corr=exp(m-m_new); e=exp(s-m_new);
    // ll = ll*corr + e.
    //
    // card 674: when `m_new == -inf`, every position seen so far (including this one) is masked -
    // `exp(m - m_new)`/`exp(s - m_new)` would be `exp(-inf - -inf) = NaN`. Route around it: an empty
    // running state contributes `corr=e=0` (matches the CPU oracle, `ops::attention::flash_prefill`,
    // which never subtracts two infinities because it takes one pass over the whole row instead of a
    // running max).
    let bb11 = BasicBlock {
        statements: vec![
            asg(
                s,
                Rvalue::BinaryOp(BinOp::Mul, copy(Place::local(s)), cf(scale)),
            ),
            asg(idx, add(maskbase, t)),
            asg(fa, Rvalue::Use(copy(elem(mask, idx)))),
            asg(s, add(s, fa)),
            asg(
                m_new,
                Rvalue::BinaryOp(BinOp::Max, copy(Place::local(m)), copy(Place::local(s))),
            ),
            asg(
                cmp,
                Rvalue::BinaryOp(BinOp::Eq, copy(Place::local(m_new)), cf(f32::NEG_INFINITY)),
            ),
        ],
        terminator: guard(cmp, 20, 19),
    };
    // bb19 (m_new == -inf): this position contributes zero weight.
    let bb19 = BasicBlock {
        statements: vec![
            asg(corr, Rvalue::Use(cf(0.0))),
            asg(e, Rvalue::Use(cf(0.0))),
        ],
        terminator: goto(21),
    };
    // bb20 (m_new finite): the ordinary online-softmax rescale.
    let bb20 = BasicBlock {
        statements: vec![
            asg(
                fa,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(m)), copy(Place::local(m_new))),
            ),
            asg(corr, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fa)))),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Sub, copy(Place::local(s)), copy(Place::local(m_new))),
            ),
            asg(e, Rvalue::MathUnary(MathOp::Exp, copy(Place::local(fb)))),
        ],
        terminator: goto(21),
    };
    // bb21: join - fold corr/e into ll regardless of which branch produced them.
    let bb21 = BasicBlock {
        statements: vec![
            asg(ll, mul(ll, corr)),
            asg(ll, add(ll, e)),
            asg(dd, Rvalue::Use(cu(0))),
        ],
        terminator: goto(12),
    };
    // o-accum loop: o[dd] = o[dd]*corr + e * v[kv,t,dd]
    let bb12 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 14, 13),
    };
    let mut bb13_st = vec![asg(fa, o_read(dd)), asg(fa, mul(fa, corr))];
    kv_index(&mut bb13_st, idx2);
    bb13_st.extend([
        asg(fb, Rvalue::Use(copy(elem(v, idx2)))),
        asg(fb, mul(e, fb)),
        asg(fc, add(fa, fb)),
        o_write(dd, fc),
        asg(
            cnt,
            Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
        ),
        asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
    ]);
    let bb13 = BasicBlock {
        statements: bb13_st,
        terminator: goto(12),
    };
    let bb14 = BasicBlock {
        statements: vec![
            asg(m, Rvalue::Use(copy(Place::local(m_new)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(t)), cu(1)),
            ),
            asg(t, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(7),
    };
    // epilogue: out[qbase+dd] = o[dd] / ll
    let bb15 = BasicBlock {
        statements: vec![asg(dd, Rvalue::Use(cu(0)))],
        terminator: goto(16),
    };
    let bb16 = BasicBlock {
        statements: vec![asg(
            cmp,
            Rvalue::BinaryOp(BinOp::Lt, copy(Place::local(dd)), cu(d)),
        )],
        terminator: guard(cmp, 18, 17),
    };
    let bb17 = BasicBlock {
        statements: vec![
            asg(idx, add(qbase, dd)),
            asg(fa, o_read(dd)),
            asg(
                fb,
                Rvalue::BinaryOp(BinOp::Div, copy(Place::local(fa)), copy(Place::local(ll))),
            ),
            Statement::Assign(elem(out, idx), Rvalue::Use(copy(Place::local(fb)))),
            asg(
                cnt,
                Rvalue::BinaryOp(BinOp::Add, copy(Place::local(dd)), cu(1)),
            ),
            asg(dd, Rvalue::Use(copy(Place::local(cnt)))),
        ],
        terminator: goto(16),
    };
    let bb18 = BasicBlock {
        statements: vec![],
        terminator: Terminator::Return,
    };
    let mut body = Body::new(
        name,
        5,
        al.locals,
        vec![
            bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9, bb10, bb11, bb12, bb13, bb14, bb15,
            bb16, bb17, bb18, bb19, bb20, bb21,
        ],
    );
    body.workgroup_size = [1, 1, 1];
    body.workgroup_locals = vec![WorkgroupLocalDecl {
        elem_ty: Ty::F32,
        len: d as u32,
    }];
    body
}
