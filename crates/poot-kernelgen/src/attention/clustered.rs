//! The clustered key walk: the body of the [`AttentionSchedule::Streaming`] and [`AttentionSchedule::SplitKv`]
//! schedules (decode attention, and the block-selected attention).
//!
//! One workgroup serves a key/value head's whole group of `n_rep` query heads at one query row (and, under the
//! split schedule, one share of the row's keys), so every key and value element is loaded once per workgroup
//! instead of once per head. The `lanes` threads form `clusters = lanes / G` clusters of `G =`
//! [`KeyWalk::cluster`] lanes, and a cluster walks one key per step: lane `j` of a cluster owns the contiguous
//! `run = `[`KeyWalk::run`] channels `[j * run, (j + 1) * run)` of the key and of the value, so a cluster reads one
//! contiguous key row and one contiguous value row. A lane whose run starts past a head's width holds zeros.
//!
//! Everything a lane carries through the walk is a register: its `n_rep` queries' runs, and per head the running
//! maximum `m`, the running denominator `l` and the `run` accumulators of `o`. All are constant-indexed locals, so
//! no workgroup memory and no barrier is touched inside the key loop. A key's score is each lane's partial dot
//! product over its run, then an xor butterfly over the cluster's `G` lanes (masks `G / 2, .., 1`): in each round
//! every lane adds its partner's value to its own, and IEEE addition commutes, so after the last round every lane of
//! the cluster holds the bit-identical score. The online softmax of the head then runs redundantly on every lane of
//! the cluster and stays consistent without a broadcast.
//!
//! The butterfly is a subgroup collective, so every lane of an aligned 32-lane group must reach it together: the
//! key loop's trip count is the same for every lane of the workgroup, and a cluster whose key is out of the
//! visited range loads nothing and scores `-inf` rather than leaving the loop (its few address reads, of a slot
//! map or a mask word, use the range's first key).
//!
//! One workgroup-memory pass ends the walk. Each lane publishes its accumulator runs, and the cluster its
//! `(m, l)`, into a `[cluster][head][dv + 2]` slab; one barrier; then the lanes, over `(head, channel)`, merge the
//! clusters in cluster order with the `(m, l)` algebra and write the normalized output row (streaming), or the
//! unnormalized partial state a split-KV stage one hands to its combine.

use poot_kernel_ir::{BinOp, Body, Constant, IndexAxis, Local, Operand, Place, Rvalue, Ty};

use super::{
    AttentionSchedule, AttentionSink, AttentionSpec, KeyBound, KeyLayout, KeyWalk, KvLane,
    MaskKind, count, count_to, exp, f32_array, f32c, fbin, local_op, score_of, ubin, uc,
};
use crate::KernelGenError;
use crate::data_movement::Cfg;
use crate::emit::cu;
use crate::helpers::{copy, elem, ld, slice_dtype, slice_f32};

/// Where a row's keys and values start in their operands.
#[derive(Clone, Copy)]
enum KeyRows {
    /// Dense operands: the element offsets of the row's `(batch, head)` plane in `k`, `k_shared` and `v`.
    Dense { k: Local, shared: Local, v: Local },
    /// Pools read through the block table parameter `read`, whose words for this batch row start at
    /// `read_base`, each covering `block` positions.
    Paged {
        read: Local,
        read_base: Local,
        block: usize,
    },
}

/// A key a block-selected row considers: its position, and whether the row reads it.
struct Candidate {
    key: Local,
    valid: Local,
}

/// Candidate `candidate` of query `query` (an index into `ids` and `pos`) among the `(k + 1) * ratio` a
/// block-selected row considers: the positions of its `k` selected blocks in order, then the positions of its
/// own unfinished block. `(k, ratio, lk)` are the selection width, the block size and the stored key
/// extent. A position is read when it is real (`id >= 0`, or inside the unfinished block), at or before
/// the query, and inside the stored keys: whatever `ids` holds, no address leaves the key buffers.
fn block_candidate(
    c: &mut Cfg,
    ids: Local,
    pos: Local,
    query: Local,
    (k, ratio, lk): (usize, usize, usize),
    candidate: Local,
) -> Candidate {
    let i32c = |value: i32| Operand::Const(Constant::I32(value));
    let widen = |c: &mut Cfg, word: Local| {
        c.let_(
            Ty::Usize,
            Rvalue::Cast {
                to: Ty::Usize,
                operand: local_op(word),
            },
        )
    };
    let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(pos, query))));
    let at = c.bin(Ty::I32, BinOp::Max, local_op(raw), i32c(0));
    let position = widen(c, at);
    let next = uc(c, BinOp::Add, position, 1);
    let complete = uc(c, BinOp::Div, next, ratio);
    let tail_start = uc(c, BinOp::Mul, complete, ratio);

    let slot = uc(c, BinOp::Div, candidate, ratio);
    let offset = uc(c, BinOp::Rem, candidate, ratio);
    let in_ids = c.bin(Ty::Bool, BinOp::Lt, local_op(slot), cu(k));
    // Read a real slot of `ids` even for a tail candidate: the value is discarded there.
    let last_slot = c.usize_const(k - 1);
    let slot_read =
        crate::block_top_k::select(c, Ty::Usize, in_ids, local_op(slot), local_op(last_slot));
    let id_at = {
        let row = uc(c, BinOp::Mul, query, k);
        ubin(c, BinOp::Add, row, slot_read)
    };
    let id = c.let_(Ty::I32, Rvalue::Use(copy(elem(ids, id_at))));
    let id_real = c.bin(Ty::Bool, BinOp::Ge, local_op(id), i32c(0));
    let id_clamped = c.bin(Ty::I32, BinOp::Max, local_op(id), i32c(0));
    let block = widen(c, id_clamped);
    let block_start = uc(c, BinOp::Mul, block, ratio);
    let selected_key = ubin(c, BinOp::Add, block_start, offset);
    let tail_key = ubin(c, BinOp::Add, tail_start, offset);
    let key = crate::block_top_k::select(
        c,
        Ty::Usize,
        in_ids,
        local_op(selected_key),
        local_op(tail_key),
    );
    let tail_real = c.bin(Ty::Bool, BinOp::Le, local_op(tail_key), local_op(position));
    let real =
        crate::block_top_k::select(c, Ty::Bool, in_ids, local_op(id_real), local_op(tail_real));
    let causal = c.bin(Ty::Bool, BinOp::Le, local_op(key), local_op(position));
    let stored = c.bin(Ty::Bool, BinOp::Lt, local_op(key), cu(lk));
    let readable = c.both(real, causal);
    let valid = c.both(readable, stored);
    Candidate { key, valid }
}

/// The grid of a split-KV schedule ([`AttentionSchedule::SplitKv`]) the partition is cut against.
#[derive(Clone, Copy)]
struct SplitGrid {
    splits: usize,
    min_units: usize,
}

/// A split workgroup's place in its row's partition: whether it has a share of the visited range to walk, and
/// how many shares the row has.
#[derive(Clone, Copy)]
struct Share {
    live: Local,
    active: Local,
}

/// Cut the `n = hi - lo` steps of a split-KV row's visited range into `active` shares, returning `(active,
/// per)`: `per` steps in each share but the last, which may be shorter.
///
/// `active = max(1, min(splits, ceil(n / min_units)))` shares, each `ceil(ceil(n / clusters) / active)` whole
/// steps of `clusters` keys, so the shares are balanced to the granularity the cost has. The last share may
/// be shorter, so `active` is then recounted as `ceil(n / per)`: no share is empty. The result depends on
/// `(n, splits, min_units, clusters)` alone, never on the cache capacity.
fn split_shares(
    c: &mut Cfg,
    lo: Local,
    hi: Local,
    grid: SplitGrid,
    clusters: usize,
) -> (Local, Local) {
    let n = steps_between(c, lo, hi);
    let wanted = {
        let padded = uc(c, BinOp::Add, n, grid.min_units - 1);
        uc(c, BinOp::Div, padded, grid.min_units)
    };
    let shares = c.var(Ty::Usize, Rvalue::Use(local_op(wanted)));
    raise_to(c, shares, 1);
    cap_at(c, shares, grid.splits);
    let cluster_steps = {
        let padded = uc(c, BinOp::Add, n, clusters - 1);
        uc(c, BinOp::Div, padded, clusters)
    };
    let steps_per = {
        let padded = ubin(c, BinOp::Add, cluster_steps, shares);
        let padded = uc(c, BinOp::Sub, padded, 1);
        let steps = ubin(c, BinOp::Div, padded, shares);
        c.var(Ty::Usize, Rvalue::Use(local_op(steps)))
    };
    // An empty range has no steps; its one share still steps by a cluster step.
    raise_to(c, steps_per, 1);
    let per = uc(c, BinOp::Mul, steps_per, clusters);
    let active = {
        let padded = ubin(c, BinOp::Add, n, per);
        let padded = uc(c, BinOp::Sub, padded, 1);
        let shares = ubin(c, BinOp::Div, padded, per);
        c.var(Ty::Usize, Rvalue::Use(local_op(shares)))
    };
    raise_to(c, active, 1);
    (active, per)
}

/// `hi - lo`, or zero when the range is empty. A window can start past a short live length: the range is then
/// empty, not negative.
fn steps_between(c: &mut Cfg, lo: Local, hi: Local) -> Local {
    let n = c.var(Ty::Usize, Rvalue::Use(cu(0)));
    let ahead = c.bin(Ty::Bool, BinOp::Gt, local_op(hi), local_op(lo));
    c.if_then(ahead, |c| {
        let steps = ubin(c, BinOp::Sub, hi, lo);
        c.assign(Place::local(n), Rvalue::Use(local_op(steps)));
    });
    n
}

/// `var <- max(var, floor)`.
fn raise_to(c: &mut Cfg, var: Local, floor: usize) {
    let below = c.bin(Ty::Bool, BinOp::Lt, local_op(var), cu(floor));
    c.if_then(below, |c| {
        c.assign(Place::local(var), Rvalue::Use(cu(floor)));
    });
}

/// `var <- min(var, ceiling)`.
fn cap_at(c: &mut Cfg, var: Local, ceiling: usize) {
    let above = c.bin(Ty::Bool, BinOp::Gt, local_op(var), cu(ceiling));
    c.if_then(above, |c| {
        c.assign(Place::local(var), Rvalue::Use(cu(ceiling)));
    });
}

/// A fresh mutable f32 register holding zero.
fn zero(c: &mut Cfg) -> Local {
    c.var(Ty::F32, Rvalue::Use(f32c(0.0)))
}

/// `dst <- src`.
fn set(c: &mut Cfg, dst: Local, src: Local) {
    c.assign(Place::local(dst), Rvalue::Use(local_op(src)));
}

/// Load `run.len()` consecutive elements of `buffer` starting at `start` into the registers `run`.
fn load_run(c: &mut Cfg, kv: KvLane, buffer: Local, start: Local, run: &[Local]) {
    for (channel, &dst) in run.iter().enumerate() {
        let at = uc(c, BinOp::Add, start, channel);
        let value = c.emit(|e| kv.read(e, buffer, at));
        set(c, dst, value);
    }
}

/// One query head of the workgroup's group: its place in the operands and its registers.
struct Head {
    /// The head index `h`.
    index: Local,
    /// The flat `(batch, head, query row)` row, which the output and the partial state are indexed by.
    row: Local,
    /// Where the head's additive mask row starts, for the masks that have a tensor.
    mask_base: Option<Local>,
    /// The lane's run of this head's query.
    q: Vec<Local>,
    /// The running maximum.
    m: Local,
    /// The running denominator.
    l: Local,
    /// The lane's run of the running weighted value sum.
    o: Vec<Local>,
}

/// Build the clustered attention kernel `spec` describes over its workgroups `rows` (all of them, except for a
/// streaming schedule split into runs of rows). The launch is `rows.len()` workgroups of `lanes` threads laid over
/// a 2-D grid `x_groups` workgroups across: `GroupY * x_groups + GroupX` is the workgroup's place in the run,
/// `gi = rows.start +` that place is the flat `(batch, key/value head, query row)` row-group index, and a
/// workgroup past the run (the fold's remainder) returns.
///
/// Parameters: `q`, `k`, `v`, `mask`, then `k_shared` for a split key or `read` for a paged one, then `len` for a
/// live bound, then `sinks` for a sink, then `out` (the partial-state scratch under
/// [`AttentionSchedule::SplitKv`]: the `x_groups`/`gi` index then runs over `(row group, split)`,
/// `gi = group * splits + split`).
pub(super) fn clustered_body(
    name: &str,
    spec: &AttentionSpec,
    rows: std::ops::Range<usize>,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    let AttentionSpec {
        batch: _,
        hq,
        n_rep,
        lq,
        lk,
        dk,
        dv,
        key,
        score,
        mask: mask_kind,
        bound,
        sink,
        schedule,
        kv,
        precision: _,
    } = *spec;
    let (lanes, walk, grid) = match schedule {
        AttentionSchedule::Streaming { lanes, walk } => (lanes, walk, None),
        AttentionSchedule::SplitKv {
            lanes,
            splits,
            min_units,
            walk,
        } => (lanes, walk, Some(SplitGrid { splits, min_units })),
        AttentionSchedule::Tiled { .. } => unreachable!("the tiled schedule has its own body"),
    };
    let KeyWalk { cluster, run } = walk;
    let clusters = lanes / cluster;
    let splits = grid.map(|grid| grid.splits);
    let hkv = hq / n_rep;
    let mask_heads = match mask_kind {
        MaskKind::Broadcast | MaskKind::Causal { .. } | MaskKind::Blocks { .. } => 1,
        MaskKind::PerHead => hq,
    };

    let mut params = vec![
        ld(Ty::Unit, false),
        ld(slice_f32(false), false), // 1 q
        kv.param(),                  // 2 k
        kv.param(),                  // 3 v
    ];
    // 4 the additive mask, the query positions, or the selected block ids and the query positions (5)
    match mask_kind {
        MaskKind::Broadcast | MaskKind::PerHead => params.push(ld(slice_f32(false), false)),
        MaskKind::Causal { .. } => params.push(ld(slice_dtype(Ty::I32, false), false)),
        MaskKind::Blocks { .. } => {
            params.push(ld(slice_dtype(Ty::I32, false), false));
            params.push(ld(slice_dtype(Ty::I32, false), false));
        }
    }
    let mut next_param = |decl| {
        params.push(decl);
        Local {
            index: params.len() as u32 - 1,
        }
    };
    let k_shared = match key {
        KeyLayout::Single | KeyLayout::Paged { .. } => None,
        KeyLayout::Split { .. } => Some(next_param(kv.param())),
    };
    let read = match key {
        KeyLayout::Single | KeyLayout::Split { .. } => None,
        KeyLayout::Paged { block } => {
            Some((next_param(ld(slice_dtype(Ty::I32, false), false)), block))
        }
    };
    let len = match bound {
        KeyBound::Static => None,
        KeyBound::Live => Some(next_param(ld(slice_dtype(Ty::I32, false), false))),
    };
    let sinks = match sink {
        AttentionSink::Absent => None,
        AttentionSink::PerHead => Some(next_param(ld(slice_f32(false), false))),
    };
    let out = next_param(ld(slice_f32(true), true));
    let (q, k, v, mask) = (
        Local { index: 1 },
        Local { index: 2 },
        Local { index: 3 },
        Local { index: 4 },
    );
    let mut c = Cfg::new(params);
    let shared_dim = match key {
        KeyLayout::Single | KeyLayout::Paged { .. } => 0,
        KeyLayout::Split { shared_dim } => shared_dim,
    };
    let head_dim = dk - shared_dim;
    // One `[cluster][head][o (dv), m, l]` slab per cluster and head for the end combine.
    let slab = dv + 2;

    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let lane = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let place = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(place), cu(rows.len()));

    c.if_then(in_range, |c| {
        let gi = uc(c, BinOp::Add, place, rows.start);
        // unit = the row group the workgroup serves, `(b * hkv + kv_head) * lq + i`. Under the split schedule
        // gi = unit * splits + split.
        let (unit, split) = match splits {
            None => (gi, None),
            Some(splits) => (
                uc(c, BinOp::Div, gi, splits),
                Some(uc(c, BinOp::Rem, gi, splits)),
            ),
        };
        let i = uc(c, BinOp::Rem, unit, lq);
        let by_group = uc(c, BinOp::Div, unit, lq);
        let kv_head = uc(c, BinOp::Rem, by_group, hkv);
        let b = uc(c, BinOp::Div, by_group, hkv);
        let first_head = uc(c, BinOp::Mul, kv_head, n_rep);
        let kv_row = {
            let by_batch = uc(c, BinOp::Mul, b, hkv);
            ubin(c, BinOp::Add, by_batch, kv_head)
        };
        // The query's index among the `B * Lq` of the selection operands.
        let qrow = {
            let by_batch = uc(c, BinOp::Mul, b, lq);
            ubin(c, BinOp::Add, by_batch, i)
        };

        // The lane's place in its cluster, and the runs it owns.
        let cluster_id = uc(c, BinOp::Div, lane, cluster);
        let j = uc(c, BinOp::Rem, lane, cluster);
        let key_start = uc(c, BinOp::Mul, j, run);
        let value_start = uc(c, BinOp::Mul, j, run);
        // A lane past the head width (the cluster is wider than the head needs) holds zeros.
        let key_live = c.bin(Ty::Bool, BinOp::Lt, local_op(key_start), cu(dk));
        let value_live = c.bin(Ty::Bool, BinOp::Lt, local_op(value_start), cu(dv));

        // The group's heads, `h = kv_head * n_rep + r`: their rows follow each other `lq` apart.
        let heads: Vec<Head> = (0..n_rep)
            .map(|r| {
                let index = uc(c, BinOp::Add, first_head, r);
                let row = {
                    let by_batch = uc(c, BinOp::Mul, b, hq);
                    let by_head = ubin(c, BinOp::Add, by_batch, index);
                    let by_row = uc(c, BinOp::Mul, by_head, lq);
                    ubin(c, BinOp::Add, by_row, i)
                };
                // Where this row's additive mask starts; the other masks have no tensor to index.
                let mask_base = match mask_kind {
                    MaskKind::Broadcast | MaskKind::PerHead => {
                        let plane = match mask_kind {
                            MaskKind::PerHead => {
                                let by_batch = uc(c, BinOp::Mul, b, mask_heads);
                                ubin(c, BinOp::Add, by_batch, index)
                            }
                            _ => uc(c, BinOp::Mul, b, mask_heads),
                        };
                        let plane_start = uc(c, BinOp::Mul, plane, lq * lk);
                        let row_start = uc(c, BinOp::Mul, i, lk);
                        Some(ubin(c, BinOp::Add, plane_start, row_start))
                    }
                    MaskKind::Causal { .. } | MaskKind::Blocks { .. } => None,
                };
                let q_run: Vec<Local> = (0..run).map(|_| zero(c)).collect();
                c.if_then(key_live, |c| {
                    let at = {
                        let base = uc(c, BinOp::Mul, row, dk);
                        ubin(c, BinOp::Add, base, key_start)
                    };
                    for (channel, &dst) in q_run.iter().enumerate() {
                        let qi = uc(c, BinOp::Add, at, channel);
                        let value = c.let_(Ty::F32, Rvalue::Use(copy(elem(q, qi))));
                        set(c, dst, value);
                    }
                });
                Head {
                    index,
                    row,
                    mask_base,
                    q: q_run,
                    m: c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY))),
                    l: zero(c),
                    o: (0..run).map(|_| zero(c)).collect(),
                }
            })
            .collect();

        // A dense key's rows start at its `(batch, head)` plane; a paged key has no plane: its block table starts
        // at the batch row and the pool row of key `t` comes from the slot its block names.
        let rows_of = match read {
            None => KeyRows::Dense {
                k: uc(c, BinOp::Mul, kv_row, lk * head_dim),
                shared: uc(c, BinOp::Mul, b, lk * shared_dim),
                v: uc(c, BinOp::Mul, kv_row, lk * dv),
            },
            Some((read, block)) => KeyRows::Paged {
                read,
                read_base: uc(c, BinOp::Mul, b, lk / block),
                block,
            },
        };

        // The keys this row visits: all `lk`, or the first `len[b]` clamped into `0..=lk`. A negative length
        // is a zero bound (the signed compare comes before the unsigned cast), a length past `lk` is `lk`.
        let key_end = match len {
            None => cu(spec.candidates()),
            Some(len) => {
                let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(len, b))));
                let end = c.var(Ty::Usize, Rvalue::Use(cu(0)));
                let non_negative = c.bin(
                    Ty::Bool,
                    BinOp::Ge,
                    local_op(raw),
                    Operand::Const(Constant::I32(0)),
                );
                c.if_then(non_negative, |c| {
                    let widened = c.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: local_op(raw),
                        },
                    );
                    c.assign(Place::local(end), Rvalue::Use(local_op(widened)));
                });
                let past_end = c.bin(Ty::Bool, BinOp::Gt, local_op(end), cu(lk));
                c.if_then(past_end, |c| {
                    c.assign(Place::local(end), Rvalue::Use(cu(lk)));
                });
                local_op(end)
            }
        };
        // The range `[lo, hi)` of steps the row visits. A causal row walks only the keys it can see: it ends
        // at its position and starts at its window's first key.
        let (lo, hi) = match mask_kind {
            MaskKind::Causal { window } => {
                let raw = c.let_(Ty::I32, Rvalue::Use(copy(elem(mask, qrow))));
                // The position as an index; a negative one (a padded query) is 0.
                let position = c.var(Ty::Usize, Rvalue::Use(cu(0)));
                let non_negative = c.bin(
                    Ty::Bool,
                    BinOp::Ge,
                    local_op(raw),
                    Operand::Const(Constant::I32(0)),
                );
                c.if_then(non_negative, |c| {
                    let widened = c.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: local_op(raw),
                        },
                    );
                    c.assign(Place::local(position), Rvalue::Use(local_op(widened)));
                });
                let visible_end = uc(c, BinOp::Add, position, 1);
                let hi = c.var(Ty::Usize, Rvalue::Use(key_end));
                let shorter = c.bin(Ty::Bool, BinOp::Lt, local_op(visible_end), local_op(hi));
                c.if_then(shorter, |c| {
                    c.assign(Place::local(hi), Rvalue::Use(local_op(visible_end)));
                });
                // The window's first key is `visible_end - window`, or 0.
                let lo = c.var(Ty::Usize, Rvalue::Use(cu(0)));
                if let Some(window) = window {
                    let past_window = c.bin(Ty::Bool, BinOp::Gt, local_op(visible_end), cu(window));
                    c.if_then(past_window, |c| {
                        let window_start = uc(c, BinOp::Sub, visible_end, window);
                        c.assign(Place::local(lo), Rvalue::Use(local_op(window_start)));
                    });
                }
                (lo, hi)
            }
            MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Blocks { .. } => (
                c.var(Ty::Usize, Rvalue::Use(cu(0))),
                c.var(Ty::Usize, Rvalue::Use(key_end)),
            ),
        };
        // One workgroup's keys: the whole range, or its split's share of it. A split with no share (`active`
        // shares cover the range) returns without a key, a partial or a barrier.
        let (begin, end, share) = match (grid, split) {
            (Some(grid), Some(split)) => {
                let (active, per) = split_shares(c, lo, hi, grid, clusters);
                let live = c.bin(Ty::Bool, BinOp::Lt, local_op(split), local_op(active));
                let skipped = ubin(c, BinOp::Mul, split, per);
                let begin = ubin(c, BinOp::Add, lo, skipped);
                let share_end = ubin(c, BinOp::Add, begin, per);
                let end = c.var(Ty::Usize, Rvalue::Use(local_op(hi)));
                let shorter = c.bin(Ty::Bool, BinOp::Lt, local_op(share_end), local_op(end));
                c.if_then(shorter, |c| {
                    c.assign(Place::local(end), Rvalue::Use(local_op(share_end)));
                });
                (begin, end, Some(Share { live, active }))
            }
            _ => (lo, hi, None),
        };

        let walk_keys = |c: &mut Cfg| {
            // The sink is the row's initial state, held by exactly one cluster of one workgroup of the row.
            if let Some(sinks) = sinks {
                let holds_sink = {
                    let first_cluster = c.bin(Ty::Bool, BinOp::Eq, local_op(cluster_id), cu(0));
                    match split {
                        None => first_cluster,
                        Some(split) => {
                            let first_split = c.bin(Ty::Bool, BinOp::Eq, local_op(split), cu(0));
                            c.both(first_cluster, first_split)
                        }
                    }
                };
                c.if_then(holds_sink, |c| {
                    for head in &heads {
                        let logit = c.let_(Ty::F32, Rvalue::Use(copy(elem(sinks, head.index))));
                        set(c, head.m, logit);
                        c.assign(Place::local(head.l), Rvalue::Use(f32c(1.0)));
                    }
                });
            }

            // The trip count is the same for every lane of the workgroup: the steps of `clusters` keys the
            // range takes.
            let span = steps_between(c, begin, end);
            let steps = {
                let padded = uc(c, BinOp::Add, span, clusters - 1);
                uc(c, BinOp::Div, padded, clusters)
            };
            count_to(c, cu(0), local_op(steps), 1, |c, step| {
                // This cluster's key of the step: position `at` of the range, readable when it is inside it.
                let at = {
                    let by_step = uc(c, BinOp::Mul, step, clusters);
                    let offset = ubin(c, BinOp::Add, by_step, cluster_id);
                    ubin(c, BinOp::Add, begin, offset)
                };
                let in_walk = c.bin(Ty::Bool, BinOp::Lt, local_op(at), local_op(end));
                let (valid, t) = match mask_kind {
                    MaskKind::Blocks { ratio, k, .. } => {
                        let ids = Local { index: 4 };
                        let pos = Local { index: 5 };
                        // A cluster past the range resolves the range's first candidate, whose key is discarded.
                        let candidate = crate::block_top_k::select(
                            c,
                            Ty::Usize,
                            in_walk,
                            local_op(at),
                            local_op(begin),
                        );
                        let head = block_candidate(c, ids, pos, qrow, (k, ratio, lk), candidate);
                        let valid = c.both(in_walk, head.valid);
                        let t = crate::block_top_k::select(
                            c,
                            Ty::Usize,
                            valid,
                            local_op(head.key),
                            cu(0),
                        );
                        (valid, t)
                    }
                    MaskKind::Broadcast | MaskKind::PerHead | MaskKind::Causal { .. } => {
                        // The range's first key is in the buffers whenever the range has a step.
                        let t = crate::block_top_k::select(
                            c,
                            Ty::Usize,
                            in_walk,
                            local_op(at),
                            local_op(begin),
                        );
                        (in_walk, t)
                    }
                };

                // The row of the key and of the value in their operands: dense rows follow the plane, a paged
                // key's operands share the pool row `slot * hkv + kv_head`.
                let (key_row, value_row) = match rows_of {
                    KeyRows::Dense { k, v, .. } => {
                        let key = uc(c, BinOp::Mul, t, head_dim);
                        let value = uc(c, BinOp::Mul, t, dv);
                        (ubin(c, BinOp::Add, k, key), ubin(c, BinOp::Add, v, value))
                    }
                    KeyRows::Paged {
                        read,
                        read_base,
                        block,
                    } => {
                        let entry = uc(c, BinOp::Div, t, block);
                        let at = ubin(c, BinOp::Add, read_base, entry);
                        let word = c.let_(Ty::I32, Rvalue::Use(copy(elem(read, at))));
                        let block_id = c.let_(
                            Ty::Usize,
                            Rvalue::Cast {
                                to: Ty::Usize,
                                operand: local_op(word),
                            },
                        );
                        let first = uc(c, BinOp::Mul, block_id, block);
                        let within = uc(c, BinOp::Rem, t, block);
                        let slot = ubin(c, BinOp::Add, first, within);
                        let by_slot = uc(c, BinOp::Mul, slot, hkv);
                        let pool_row = ubin(c, BinOp::Add, by_slot, kv_head);
                        (
                            uc(c, BinOp::Mul, pool_row, head_dim),
                            uc(c, BinOp::Mul, pool_row, dv),
                        )
                    }
                };

                // The key's and the value's runs, loaded once for every head of the group, by the lanes whose
                // cluster has a key this step and whose run is inside the head: a cluster past the range
                // reads nothing, and its zeros score `-inf` below. A split key's run lies in one of its two
                // parts (`validate` takes the run width dividing both).
                let key_run: Vec<Local> = (0..run).map(|_| zero(c)).collect();
                let value_run: Vec<Local> = (0..run).map(|_| zero(c)).collect();
                let load_key = c.both(valid, key_live);
                match (k_shared, rows_of) {
                    (Some(k_shared), KeyRows::Dense { shared, .. }) => {
                        let in_own = c.bin(Ty::Bool, BinOp::Lt, local_op(key_start), cu(head_dim));
                        let load_own = c.both(load_key, in_own);
                        c.if_then(load_own, |c| {
                            let start = ubin(c, BinOp::Add, key_row, key_start);
                            load_run(c, kv, k, start, &key_run);
                        });
                        let past_own =
                            c.bin(Ty::Bool, BinOp::Ge, local_op(key_start), cu(head_dim));
                        let load_shared = c.both(load_key, past_own);
                        c.if_then(load_shared, |c| {
                            let shared_row = {
                                let by_key = uc(c, BinOp::Mul, t, shared_dim);
                                ubin(c, BinOp::Add, shared, by_key)
                            };
                            let within = uc(c, BinOp::Sub, key_start, head_dim);
                            let start = ubin(c, BinOp::Add, shared_row, within);
                            load_run(c, kv, k_shared, start, &key_run);
                        });
                    }
                    _ => c.if_then(load_key, |c| {
                        let start = ubin(c, BinOp::Add, key_row, key_start);
                        load_run(c, kv, k, start, &key_run);
                    }),
                }
                let load_value = c.both(valid, value_live);
                c.if_then(load_value, |c| {
                    let start = ubin(c, BinOp::Add, value_row, value_start);
                    load_run(c, kv, v, start, &value_run);
                });

                // Each head's partial dot product over the lane's run, c ascending.
                let parts: Vec<Local> = heads
                    .iter()
                    .map(|head| {
                        let part = zero(c);
                        for (&qv, &kk) in head.q.iter().zip(&key_run) {
                            let product = fbin(c, BinOp::Mul, qv, kk);
                            let sum = fbin(c, BinOp::Add, part, product);
                            set(c, part, sum);
                        }
                        part
                    })
                    .collect();
                // The xor butterfly over the cluster, every head interleaved: after the last round all `G`
                // lanes hold the same bits.
                let mut distance = cluster / 2;
                while distance > 0 {
                    for &part in &parts {
                        let partner = c.let_(
                            Ty::F32,
                            Rvalue::SubgroupShuffleXor {
                                value: local_op(part),
                                mask: distance as u32,
                            },
                        );
                        let sum = fbin(c, BinOp::Add, part, partner);
                        set(c, part, sum);
                    }
                    distance /= 2;
                }

                for (head, &dot) in heads.iter().zip(&parts) {
                    let s = score_of(c, score, dot);
                    // A block-selected or causal row visits only keys it can see, so there is no mask to add.
                    let s = match (mask_kind, head.mask_base) {
                        (MaskKind::Broadcast | MaskKind::PerHead, Some(mask_base)) => {
                            let mi = ubin(c, BinOp::Add, mask_base, t);
                            let mask_value = c.let_(Ty::F32, Rvalue::Use(copy(elem(mask, mi))));
                            fbin(c, BinOp::Add, s, mask_value)
                        }
                        _ => s,
                    };
                    // A cluster whose key is past the range weighs nothing.
                    let s = crate::block_top_k::select(
                        c,
                        Ty::F32,
                        valid,
                        local_op(s),
                        f32c(f32::NEG_INFINITY),
                    );

                    // The online-softmax step. `m_new == -inf` means every position seen so far, this one
                    // included, is masked: the head has contributed nothing, and `exp(-inf - -inf)` is NaN.
                    let m_new = fbin(c, BinOp::Max, head.m, s);
                    let corr = zero(c);
                    let weight = zero(c);
                    let seen = c.bin(
                        Ty::Bool,
                        BinOp::Ne,
                        local_op(m_new),
                        f32c(f32::NEG_INFINITY),
                    );
                    c.if_then(seen, |c| {
                        let behind = fbin(c, BinOp::Sub, head.m, m_new);
                        let rescale = exp(c, behind);
                        set(c, corr, rescale);
                        let below = fbin(c, BinOp::Sub, s, m_new);
                        let own = exp(c, below);
                        set(c, weight, own);
                    });
                    let decayed = fbin(c, BinOp::Mul, head.l, corr);
                    let l_next = fbin(c, BinOp::Add, decayed, weight);
                    set(c, head.l, l_next);
                    for (&o, &vv) in head.o.iter().zip(&value_run) {
                        let rescaled = fbin(c, BinOp::Mul, o, corr);
                        let product = fbin(c, BinOp::Mul, weight, vv);
                        let sum = fbin(c, BinOp::Add, rescaled, product);
                        set(c, o, sum);
                    }
                    set(c, head.m, m_new);
                }
            });

            // Publish the lane's accumulator runs, and the cluster's `(m, l)` (the same on every lane of it),
            // into the cluster's slab of each head. Then the one barrier: every lane reaches it exactly once.
            let slabs: Vec<Local> = (0..n_rep)
                .map(|r| {
                    let by_cluster = uc(c, BinOp::Mul, cluster_id, n_rep);
                    let by_head = uc(c, BinOp::Add, by_cluster, r);
                    uc(c, BinOp::Mul, by_head, slab)
                })
                .collect();
            c.if_then(value_live, |c| {
                for (head, &slab_start) in heads.iter().zip(&slabs) {
                    let start = ubin(c, BinOp::Add, slab_start, value_start);
                    for (channel, &o) in head.o.iter().enumerate() {
                        let at = uc(c, BinOp::Add, start, channel);
                        super::lds_write(c, 0, at, local_op(o));
                    }
                }
            });
            let first_lane = c.bin(Ty::Bool, BinOp::Eq, local_op(j), cu(0));
            c.if_then(first_lane, |c| {
                for (head, &slab_start) in heads.iter().zip(&slabs) {
                    let m_at = uc(c, BinOp::Add, slab_start, dv);
                    super::lds_write(c, 0, m_at, local_op(head.m));
                    let l_at = uc(c, BinOp::Add, slab_start, dv + 1);
                    super::lds_write(c, 0, l_at, local_op(head.l));
                }
            });
            c.barrier();

            // The merge of the clusters, over `(head, channel)` items strided across the lanes. `big_m` is the
            // maximum of the clusters' maxima; a cluster whose maximum is `-inf` saw no unmasked key and weighs
            // zero, so no `-inf - -inf` is formed.
            let first_row = heads[0].row;
            count(c, local_op(lane), n_rep * dv, lanes, |c, item| {
                let r = uc(c, BinOp::Div, item, dv);
                let channel = uc(c, BinOp::Rem, item, dv);
                let slab_of = |c: &mut Cfg, cl: Local| {
                    let by_cluster = uc(c, BinOp::Mul, cl, n_rep);
                    let by_head = ubin(c, BinOp::Add, by_cluster, r);
                    uc(c, BinOp::Mul, by_head, slab)
                };
                let weight_of = |c: &mut Cfg, cl: Local, big_m: Local| {
                    let weight = zero(c);
                    let at = {
                        let start = slab_of(c, cl);
                        uc(c, BinOp::Add, start, dv)
                    };
                    let m_cl = super::lds_read(c, 0, at);
                    let seen = c.bin(Ty::Bool, BinOp::Ne, local_op(m_cl), f32c(f32::NEG_INFINITY));
                    c.if_then(seen, |c| {
                        let behind = fbin(c, BinOp::Sub, m_cl, big_m);
                        let w = exp(c, behind);
                        set(c, weight, w);
                    });
                    weight
                };
                let big_m = c.var(Ty::F32, Rvalue::Use(f32c(f32::NEG_INFINITY)));
                count(c, cu(0), clusters, 1, |c, cl| {
                    let at = {
                        let start = slab_of(c, cl);
                        uc(c, BinOp::Add, start, dv)
                    };
                    let m_cl = super::lds_read(c, 0, at);
                    let next = fbin(c, BinOp::Max, big_m, m_cl);
                    set(c, big_m, next);
                });
                let denominator = zero(c);
                let acc = zero(c);
                count(c, cu(0), clusters, 1, |c, cl| {
                    let w = weight_of(c, cl, big_m);
                    let start = slab_of(c, cl);
                    let l_at = uc(c, BinOp::Add, start, dv + 1);
                    let l_cl = super::lds_read(c, 0, l_at);
                    let part = fbin(c, BinOp::Mul, l_cl, w);
                    let next = fbin(c, BinOp::Add, denominator, part);
                    set(c, denominator, next);
                    let o_at = ubin(c, BinOp::Add, start, channel);
                    let o_cl = super::lds_read(c, 0, o_at);
                    let part = fbin(c, BinOp::Mul, o_cl, w);
                    let next = fbin(c, BinOp::Add, acc, part);
                    set(c, acc, next);
                });
                // The item's row: the head `r` rows of the group after its first.
                let row = {
                    let by_head = uc(c, BinOp::Mul, r, lq);
                    ubin(c, BinOp::Add, first_row, by_head)
                };
                // The streaming schedule writes the normalized row; a split writes its unnormalized state.
                let out_base = match split {
                    None => uc(c, BinOp::Mul, row, dv),
                    Some(split) => {
                        let partial = {
                            let first = uc(
                                c,
                                BinOp::Mul,
                                row,
                                splits.expect("a split schedule has splits"),
                            );
                            ubin(c, BinOp::Add, first, split)
                        };
                        uc(c, BinOp::Mul, partial, slab)
                    }
                };
                let result = match split {
                    None => fbin(c, BinOp::Div, acc, denominator),
                    Some(_) => acc,
                };
                let oi = ubin(c, BinOp::Add, out_base, channel);
                c.assign(elem(out, oi), Rvalue::Use(local_op(result)));
                if let (Some(split), Some(grid), Some(Share { active, .. })) = (split, grid, share)
                {
                    let first_channel = c.bin(Ty::Bool, BinOp::Eq, local_op(channel), cu(0));
                    c.if_then(first_channel, |c| {
                        let m_at = uc(c, BinOp::Add, out_base, dv);
                        c.assign(elem(out, m_at), Rvalue::Use(local_op(big_m)));
                        let l_at = uc(c, BinOp::Add, out_base, dv + 1);
                        c.assign(elem(out, l_at), Rvalue::Use(local_op(denominator)));
                        // Split 0 publishes how many splits the combine reads, in the row's word after the
                        // partials.
                        let first_split = c.bin(Ty::Bool, BinOp::Eq, local_op(split), cu(0));
                        c.if_then(first_split, |c| {
                            let count_at = uc(c, BinOp::Add, row, spec.rows() * grid.splits * slab);
                            let published = c.let_(
                                Ty::F32,
                                Rvalue::Cast {
                                    to: Ty::F32,
                                    operand: local_op(active),
                                },
                            );
                            c.assign(elem(out, count_at), Rvalue::Use(local_op(published)));
                        });
                    });
                }
            });
        };
        match share {
            None => walk_keys(c),
            Some(Share { live, .. }) => c.if_then(live, walk_keys),
        }
    });

    let mut body = c.finish(name);
    body.workgroup_size = [lanes as u32, 1, 1];
    body.workgroup_locals = vec![f32_array(clusters * n_rep * slab)];
    Ok(body)
}
