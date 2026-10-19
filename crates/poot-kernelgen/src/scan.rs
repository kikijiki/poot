//! The scan generator: a streaming recurrence over the sequence axis (POOT-634).
//!
//! One workgroup per `(head, column block)`; the head's `[rows, width]` state column `j` lives in workgroup
//! memory. Both recurrences update a state column from that column alone, so columns share no state. By
//! default one lane owns a column and walks its `rows` rows serially, with no barrier; with
//! [`ScanSpec::row_split`] `R > 1`, `R` lanes share a column, each walking every `R`-th row, and meet at one
//! barrier per dot product to sum their partials (see [`ScanSpec::row_split`]). The recurrences:
//!
//! - [`ScanKernel::DeltaRule`] (`rows == width == D`): `S[:,j] <- a S[:,j]`, then
//!   `S[:,j] += k * beta * (v[j] - k . S[:,j])` and `o[j] = q . S[:,j] / sqrt(D)`, `a = exp(g)`.
//! - [`ScanKernel::Ssd`] (`rows == N`, `width == P`): `h[:,j] <- exp(delta A) h[:,j] + delta B x[j]` and
//!   `y[j] = C . h[:,j] + D x[j]`.
//!
//! The sequence loop is a runtime `while`, so the body is the same size for every length. One launch covers the
//! whole sequence.
//!
//! The result is the packed `[1, H, len + rows, width]` tensor of `OpKind::Scan` (see `poot_graph_ir::ScanLayout`):
//! rows `..len` the sequence output, rows `len..` the final state. Parameters: the family's operands in
//! `OpKind::Scan` order, then `out`.

use poot_kernel_ir::{BinOp, Body, IndexAxis, MathOp, Operand, Place, Rvalue, Ty};

use crate::KernelGenError;
use crate::attention::{
    count, count_to, f32_array, f32c, fbin, lds_read, lds_write, local_op, ubin, uc,
};
use crate::data_movement::Cfg;
use crate::emit::cu;
use crate::helpers::{copy, elem, ld, slice_f32};

/// Which recurrence a scan kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanKernel {
    /// The delta-rule recurrence, `rows == width == D`, over six operands.
    DeltaRule,
    /// The SSD (Mamba-2) recurrence, `rows == N`, `width == P`, over seven operands.
    Ssd,
}

impl ScanKernel {
    fn operands(self) -> usize {
        match self {
            Self::DeltaRule => 6,
            Self::Ssd => 7,
        }
    }
}

/// What the planner asks the scan generator for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanSpec {
    /// Which recurrence the kernel runs.
    pub kernel: ScanKernel,
    /// Heads, each with its own independent state.
    pub heads: usize,
    /// The sequence length (the packed result's layout).
    pub len: usize,
    /// The state's rows (`D` for the delta rule, `N` for SSD).
    pub rows: usize,
    /// The state's columns and the output width (`D`, or `P`).
    pub width: usize,
    /// The workgroup width in lanes. `row_split` consecutive lanes share one state column, so a workgroup
    /// covers `cols / row_split` state columns.
    pub cols: usize,
    /// Lanes that split one state column's rows between them (a power of two dividing `rows` and `cols`). One
    /// lane walks all `rows` rows of its column serially; `row_split` lanes each walk `rows / row_split` of them
    /// and combine the two per-token dot products through workgroup memory, two barriers per token for the delta
    /// rule and one for SSD. The split reassociates each dot product, so its result differs from the
    /// unsplit one in f32 rounding only.
    pub row_split: usize,
}

impl ScanSpec {
    /// State columns one workgroup owns.
    pub fn state_cols(&self) -> usize {
        self.cols / self.row_split
    }

    /// Workgroups along the state-column axis per head: the columns divided into workgroup-sized blocks.
    pub fn col_blocks(&self) -> usize {
        self.width.div_ceil(self.state_cols())
    }

    /// Workgroups the launch needs: one per head per column block.
    pub fn workgroups(&self) -> usize {
        self.heads * self.col_blocks()
    }

    /// Static workgroup memory the state columns and the cross-lane partials need, in bytes.
    pub fn lds_bytes(&self) -> u64 {
        let scratch = self.scratch_floats();
        ((self.rows * self.state_cols() + scratch) * 4) as u64
    }

    /// Floats of cross-lane partial-sum scratch: none unsplit; split, one `[row_split, state_cols]` plane
    /// per reduction the kernel makes per token (two for the delta rule, two alternating for SSD).
    fn scratch_floats(&self) -> usize {
        if self.row_split == 1 {
            0
        } else {
            2 * self.row_split * self.state_cols()
        }
    }
}

const GENERATOR: &str = "scan_body";
const STATE: u8 = 0;

fn below_one(what: &str, value: usize) -> KernelGenError {
    KernelGenError::BelowMinimum {
        generator: GENERATOR,
        what: what.to_string(),
        value,
        min: 1,
    }
}

fn validate(spec: &ScanSpec) -> Result<(), KernelGenError> {
    for (what, value) in [
        ("heads", spec.heads),
        ("len", spec.len),
        ("rows", spec.rows),
        ("width", spec.width),
        ("cols", spec.cols),
        ("row_split", spec.row_split),
    ] {
        if value == 0 {
            return Err(below_one(what, value));
        }
    }
    for (dim, value) in [("rows", spec.rows), ("cols", spec.cols)] {
        if !value.is_multiple_of(spec.row_split) {
            return Err(KernelGenError::NotDivisible {
                generator: GENERATOR,
                dim: dim.to_string(),
                value,
                divisor: spec.row_split,
            });
        }
    }
    if spec.kernel == ScanKernel::DeltaRule && spec.rows != spec.width {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "delta-rule state rows".to_string(),
            value: spec.rows,
            bound: spec.width,
        });
    }
    Ok(())
}

/// The partial-sum scratch array, present only when `row_split > 1`: two planes of `[row_split, state_cols]`.
const PLANES: u8 = 1;

/// Where one lane sits in its workgroup: its state column, and the first of the rows it walks (then every
/// `row_split`-th).
#[derive(Clone, Copy)]
struct Lane {
    /// The column within the workgroup's `state_cols` (the LDS column).
    col_in_block: poot_kernel_ir::Local,
    /// The first row of this lane's share; `None` unsplit, where the lane walks every row from 0.
    first_row: Option<poot_kernel_ir::Local>,
}

/// Build the scan kernel `spec` describes. The launch is `spec.workgroups()` workgroups of `cols` lanes
/// laid over a 2-D grid `x_groups` workgroups across: `gi = GroupY * x_groups + GroupX` is the flat
/// `(head, column block)` index, and a workgroup past `workgroups()` returns.
///
/// Every lane of a workgroup runs the whole token loop, so the per-token barriers of a split schedule sit
/// on a path all lanes take: a lane whose column is past `width` (the last block of an uneven width) reads
/// a clamped column and only its stores are masked.
pub(crate) fn scan_body(
    name: &str,
    spec: &ScanSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    validate(spec)?;
    let ScanSpec {
        kernel,
        rows,
        width,
        row_split,
        ..
    } = *spec;
    let state_cols = spec.state_cols();
    let blocks = spec.col_blocks();
    let mut params = vec![ld(Ty::Unit, false)];
    for _ in 0..kernel.operands() {
        params.push(ld(slice_f32(false), false));
    }
    params.push(ld(slice_f32(true), true));
    let input = |i: usize| poot_kernel_ir::Local {
        index: i as u32 + 1,
    };
    let out = input(kernel.operands());
    let state_in = input(kernel.operands() - 1);
    let mut c = Cfg::new(params);

    let group_x = c.index(IndexAxis::GroupX);
    let group_y = c.index(IndexAxis::GroupY);
    let local = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let gi = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(gi), cu(spec.workgroups()));
    let operands = Operands {
        inputs: (0..kernel.operands() - 1).map(input).collect(),
        state_in,
        out,
    };
    c.if_then(in_range, |c| {
        let col_in_block = if row_split == 1 {
            local
        } else {
            uc(c, BinOp::Rem, local, state_cols)
        };
        let h = uc(c, BinOp::Div, gi, blocks);
        let block = uc(c, BinOp::Rem, gi, blocks);
        let block_start = uc(c, BinOp::Mul, block, state_cols);
        let col = ubin(c, BinOp::Add, block_start, col_in_block);
        let live = c.bin(Ty::Bool, BinOp::Lt, local_op(col), cu(width));
        if row_split == 1 {
            // One lane per column and no barrier: a lane past the width has nothing to do at all.
            c.if_then(live, |c| {
                let lane = Lane {
                    col_in_block,
                    first_row: None,
                };
                scan_column(c, spec, &operands, lane, h, col, None);
            });
        } else {
            // Every lane reaches the barriers, so a lane past the width reads a clamped column and only its
            // stores are masked.
            let first_row = uc(c, BinOp::Div, local, state_cols);
            let lane = Lane {
                col_in_block,
                first_row: Some(first_row),
            };
            let clamped = c.bin(Ty::Usize, BinOp::Min, local_op(col), cu(width - 1));
            scan_column(c, spec, &operands, lane, h, clamped, Some(live));
        }
    });
    let mut body = c.finish(name);
    body.workgroup_size = [spec.cols as u32, 1, 1];
    body.workgroup_locals = vec![f32_array(rows * state_cols)];
    if row_split > 1 {
        body.workgroup_locals.push(f32_array(spec.scratch_floats()));
    }
    Ok(body)
}

/// The kernel's parameters: the family's operands, the carried-state operand (the last of them) and `out`.
struct Operands {
    inputs: Vec<poot_kernel_ir::Local>,
    state_in: poot_kernel_ir::Local,
    out: poot_kernel_ir::Local,
}

/// Run `f` under `guard` when there is one: the stores of a split launch are masked for lanes past the width.
fn guarded(c: &mut Cfg, guard: Option<poot_kernel_ir::Local>, f: impl FnOnce(&mut Cfg)) {
    match guard {
        Some(live) => c.if_then(live, f),
        None => f(c),
    }
}

/// One lane's whole run over its share of column `col` of head `h`: load the state, step every token,
/// publish the state. `masked` is the guard the stores sit under when a split launch needs every lane on
/// the barrier path; the column's first lane alone writes the sequence output.
fn scan_column(
    c: &mut Cfg,
    spec: &ScanSpec,
    ops: &Operands,
    lane: Lane,
    h: poot_kernel_ir::Local,
    col: poot_kernel_ir::Local,
    masked: Option<poot_kernel_ir::Local>,
) {
    let ScanSpec {
        kernel,
        len,
        rows,
        width,
        ..
    } = *spec;
    let Operands {
        inputs,
        state_in,
        out,
    } = ops;
    let (state_in, out) = (*state_in, *out);
    let input = |i: usize| inputs[i];
    // The packed result's per-head origin and the carried state's rows within it.
    let out_base = uc(c, BinOp::Mul, h, (len + rows) * width);
    let state_at = |c: &mut Cfg, r: poot_kernel_ir::Local| {
        let row = uc(c, BinOp::Add, r, len);
        let row = uc(c, BinOp::Mul, row, width);
        let at = ubin(c, BinOp::Add, out_base, row);
        ubin(c, BinOp::Add, at, col)
    };
    // Load this lane's share of the state column.
    my_rows(c, spec, lane, |c, r| {
        let head = uc(c, BinOp::Mul, h, rows * width);
        let row = uc(c, BinOp::Mul, r, width);
        let at = ubin(c, BinOp::Add, head, row);
        let at = ubin(c, BinOp::Add, at, col);
        let value = c.let_(Ty::F32, Rvalue::Use(copy(elem(state_in, at))));
        let slot = state_slot(c, spec, r, lane);
        lds_write(c, STATE, slot, local_op(value));
    });
    count_to(c, cu(0), cu(len), 1, |c, t| {
        let pos = {
            let head = uc(c, BinOp::Mul, h, len);
            ubin(c, BinOp::Add, head, t)
        };
        let o_acc = match kernel {
            ScanKernel::DeltaRule => delta_rule_step(
                c,
                spec,
                [input(0), input(1), input(2), input(3), input(4)],
                pos,
                col,
                lane,
            ),
            ScanKernel::Ssd => ssd_step(
                c,
                spec,
                [input(0), input(1), input(2), input(3), input(4), input(5)],
                pos,
                h,
                col,
                lane,
            ),
        };
        // Every lane of a column holds the reduced value; the column's first lane writes it.
        guarded(c, masked, |c| {
            let write = |c: &mut Cfg| {
                let oi = uc(c, BinOp::Mul, t, width);
                let oi = ubin(c, BinOp::Add, out_base, oi);
                let oi = ubin(c, BinOp::Add, oi, col);
                c.assign(elem(out, oi), Rvalue::Use(local_op(o_acc)));
            };
            match lane.first_row {
                None => write(c),
                Some(first_row) => {
                    let first = c.bin(Ty::Bool, BinOp::Eq, local_op(first_row), cu(0));
                    c.if_then(first, write);
                }
            }
        });
    });
    // Publish the state column into the packed result's trailing rows.
    guarded(c, masked, |c| {
        my_rows(c, spec, lane, |c, r| {
            let slot = state_slot(c, spec, r, lane);
            let s = lds_read(c, STATE, slot);
            let at = state_at(c, r);
            c.assign(elem(out, at), Rvalue::Use(local_op(s)));
        });
    });
}

/// `body(r)` for every state row `r` this lane walks, in increasing order: `rows / row_split` of them.
fn my_rows(
    c: &mut Cfg,
    spec: &ScanSpec,
    lane: Lane,
    body: impl FnOnce(&mut Cfg, poot_kernel_ir::Local),
) {
    let split = spec.row_split;
    count(c, cu(0), spec.rows / split, 1, |c, i| {
        let r = match lane.first_row {
            None => i,
            Some(first_row) => {
                let by = uc(c, BinOp::Mul, i, split);
                ubin(c, BinOp::Add, by, first_row)
            }
        };
        body(c, r);
    });
}

/// This lane's slot for row `r` of its column in the state array: row-major `[rows, state_cols]`.
fn state_slot(
    c: &mut Cfg,
    spec: &ScanSpec,
    r: poot_kernel_ir::Local,
    lane: Lane,
) -> poot_kernel_ir::Local {
    let slot = uc(c, BinOp::Mul, r, spec.state_cols());
    ubin(c, BinOp::Add, slot, lane.col_in_block)
}

/// The sum of `partial` over the `row_split` lanes of this lane's column, in lane order, returned to every one
/// of them. `plane` is the float offset of the `[row_split, state_cols]` plane the partials go through. Each
/// lane writes its partial, all reach the barrier, then each lane reads the column's `row_split` values: the
/// barrier is on the straight-line path of the token loop. Unsplit, the partial already is the sum.
fn reduce_rows(
    c: &mut Cfg,
    spec: &ScanSpec,
    lane: Lane,
    plane: Operand,
    partial: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    if spec.row_split == 1 {
        return partial;
    }
    let state_cols = spec.state_cols();
    let first_row = lane.first_row.expect("a split lane has a first row");
    let own = uc(c, BinOp::Mul, first_row, state_cols);
    let own = ubin(c, BinOp::Add, own, lane.col_in_block);
    let own = c.bin(Ty::Usize, BinOp::Add, plane.clone(), local_op(own));
    lds_write(c, PLANES, own, local_op(partial));
    c.barrier();
    let sum = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    count(c, cu(0), spec.row_split, 1, |c, other| {
        let at = uc(c, BinOp::Mul, other, state_cols);
        let at = ubin(c, BinOp::Add, at, lane.col_in_block);
        let at = c.bin(Ty::Usize, BinOp::Add, plane, local_op(at));
        let part = lds_read(c, PLANES, at);
        let next = fbin(c, BinOp::Add, sum, part);
        c.assign(Place::local(sum), Rvalue::Use(local_op(next)));
    });
    sum
}

/// One delta-rule token for this lane's column; returns `o[col]` (on every lane of the column). `ops` is
/// `[q, k, v, g, beta]`. The two reductions go through the two planes: a lane that is past the first barrier
/// has seen every lane's first-plane write, so it may not be overwritten before it reaches the second.
fn delta_rule_step(
    c: &mut Cfg,
    spec: &ScanSpec,
    ops: [poot_kernel_ir::Local; 5],
    pos: poot_kernel_ir::Local,
    col: poot_kernel_ir::Local,
    lane: Lane,
) -> poot_kernel_ir::Local {
    let [q, k, v, g, beta] = ops;
    let d = spec.rows;
    let plane = spec.row_split * spec.state_cols();
    let row = uc(c, BinOp::Mul, pos, d);
    let g_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(g, pos))));
    let decay = c.let_(Ty::F32, Rvalue::MathUnary(MathOp::Exp, local_op(g_t)));
    let beta_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(beta, pos))));
    // kv = k . (decay * S[:,col])
    let kv_part = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    my_rows(c, spec, lane, |c, r| {
        let slot = state_slot(c, spec, r, lane);
        let s = lds_read(c, STATE, slot);
        let s = fbin(c, BinOp::Mul, s, decay);
        let ki = ubin(c, BinOp::Add, row, r);
        let kr = c.let_(Ty::F32, Rvalue::Use(copy(elem(k, ki))));
        let term = fbin(c, BinOp::Mul, kr, s);
        let sum = fbin(c, BinOp::Add, kv_part, term);
        c.assign(Place::local(kv_part), Rvalue::Use(local_op(sum)));
    });
    let kv = reduce_rows(c, spec, lane, cu(0), kv_part);
    let vi = ubin(c, BinOp::Add, row, col);
    let v_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(v, vi))));
    let residual = fbin(c, BinOp::Sub, v_t, kv);
    let delta = fbin(c, BinOp::Mul, beta_t, residual);
    // S[r] <- decay * S[r] + k[r] * delta; o = q_scaled . S_new
    let scale = 1.0 / (d as f32).sqrt();
    let o_part = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    my_rows(c, spec, lane, |c, r| {
        let slot = state_slot(c, spec, r, lane);
        let s = lds_read(c, STATE, slot);
        let s = fbin(c, BinOp::Mul, s, decay);
        let ki = ubin(c, BinOp::Add, row, r);
        let kr = c.let_(Ty::F32, Rvalue::Use(copy(elem(k, ki))));
        let bump = fbin(c, BinOp::Mul, kr, delta);
        let s_new = fbin(c, BinOp::Add, s, bump);
        lds_write(c, STATE, slot, local_op(s_new));
        let qr = c.let_(Ty::F32, Rvalue::Use(copy(elem(q, ki))));
        let qs = c.bin(Ty::F32, BinOp::Mul, local_op(qr), f32c(scale));
        let term = fbin(c, BinOp::Mul, qs, s_new);
        let sum = fbin(c, BinOp::Add, o_part, term);
        c.assign(Place::local(o_part), Rvalue::Use(local_op(sum)));
    });
    reduce_rows(c, spec, lane, cu(plane), o_part)
}

/// One SSD token for this lane's column; returns `y[col]` (on every lane of the column). `ops` is
/// `[x, b, c, delta, a, d]`. One reduction per token: consecutive tokens alternate planes, so a lane that
/// runs a token ahead writes the plane its slower neighbours are not reading, and a lane two tokens ahead
/// cannot exist (it would have to pass the barrier of the token between).
#[allow(clippy::too_many_arguments)]
fn ssd_step(
    c: &mut Cfg,
    spec: &ScanSpec,
    ops: [poot_kernel_ir::Local; 6],
    pos: poot_kernel_ir::Local,
    head: poot_kernel_ir::Local,
    col: poot_kernel_ir::Local,
    lane: Lane,
) -> poot_kernel_ir::Local {
    let [x, b_in, c_in, delta, a, d_skip] = ops;
    let (n, p) = (spec.rows, spec.width);
    let x_row = uc(c, BinOp::Mul, pos, p);
    let xi = ubin(c, BinOp::Add, x_row, col);
    let x_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(x, xi))));
    let bc_row = uc(c, BinOp::Mul, pos, n);
    let delta_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(delta, pos))));
    let a_h = c.let_(Ty::F32, Rvalue::Use(copy(elem(a, head))));
    let gate = fbin(c, BinOp::Mul, delta_t, a_h);
    let abar = c.let_(Ty::F32, Rvalue::MathUnary(MathOp::Exp, local_op(gate)));
    let dx = fbin(c, BinOp::Mul, delta_t, x_t);
    let y_part = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    my_rows(c, spec, lane, |c, r| {
        let slot = state_slot(c, spec, r, lane);
        let s = lds_read(c, STATE, slot);
        let s = fbin(c, BinOp::Mul, s, abar);
        let bi = ubin(c, BinOp::Add, bc_row, r);
        let br = c.let_(Ty::F32, Rvalue::Use(copy(elem(b_in, bi))));
        let bump = fbin(c, BinOp::Mul, br, dx);
        let s_new = fbin(c, BinOp::Add, s, bump);
        lds_write(c, STATE, slot, local_op(s_new));
        let cr = c.let_(Ty::F32, Rvalue::Use(copy(elem(c_in, bi))));
        let term = fbin(c, BinOp::Mul, cr, s_new);
        let sum = fbin(c, BinOp::Add, y_part, term);
        c.assign(Place::local(y_part), Rvalue::Use(local_op(sum)));
    });
    let y_acc = if spec.row_split == 1 {
        y_part
    } else {
        let parity = uc(c, BinOp::Rem, pos, 2);
        let plane = uc(c, BinOp::Mul, parity, spec.row_split * spec.state_cols());
        reduce_rows(c, spec, lane, local_op(plane), y_part)
    };
    let d_h = c.let_(Ty::F32, Rvalue::Use(copy(elem(d_skip, head))));
    let skip = fbin(c, BinOp::Mul, d_h, x_t);
    fbin(c, BinOp::Add, y_acc, skip)
}
