//! The scan generator: a streaming recurrence over the sequence axis (POOT-634).
//!
//! One workgroup per `(head, column block)`; a lane owns one column `j` of the head's `[rows, width]` state,
//! held in workgroup memory (`rows * cols` floats). Both recurrences update a state column from that column
//! alone, so lanes share no state and the body needs no barrier:
//!
//! - [`ScanKernel::DeltaRule`] (`rows == width == D`): `S[:,j] <- a S[:,j]`, then
//!   `S[:,j] += k * beta * (v[j] - k . S[:,j])` and `o[j] = q . S[:,j] / sqrt(D)`, `a = exp(g)`.
//! - [`ScanKernel::Ssd`] (`rows == N`, `width == P`): `h[:,j] <- exp(delta A) h[:,j] + delta B x[j]` and
//!   `y[j] = C . h[:,j] + D x[j]`.
//!
//! The sequence loop is a runtime `while`, so the body is the same size for every length. A launch covers the
//! token range of its [`ScanPhase`]; a longer sequence is several launches, each reading the carried state the
//! previous one left in the packed result and writing it back, so one dispatch's serial work stays bounded.
//!
//! The result is the packed `[1, H, len + rows, width]` tensor of `OpKind::Scan` (see `poot_graph_ir::ScanLayout`):
//! rows `..len` the sequence output, rows `len..` the final state. Parameters: the family's operands in
//! `OpKind::Scan` order, then `out`.

use poot_kernel_ir::{BinOp, Body, IndexAxis, MathOp, Place, Rvalue, Ty};

use crate::KernelGenError;
use crate::attention::{
    count, count_to, f32_array, f32c, fbin, lds_read, lds_write, local_op, ubin, uc,
};
use crate::data_movement::Cfg;
use crate::emit::cu;
use crate::helpers::{copy, elem, ld, slice_f32};
use crate::request::SerialWork;

/// Which recurrence a scan kernel runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanKernel {
    DeltaRule,
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

/// The token range one launch covers: `count` tokens from `start`. The first phase (`start == 0`) reads the
/// carried state from the state operand; every later one reads it from the packed result's state rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScanPhase {
    pub start: usize,
    pub count: usize,
}

/// What the planner asks the scan generator for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScanSpec {
    pub kernel: ScanKernel,
    pub heads: usize,
    /// The whole sequence length (the packed result's layout), of which `phase` runs a part.
    pub len: usize,
    /// The state's rows (`D` for the delta rule, `N` for SSD).
    pub rows: usize,
    /// The state's columns and the output width (`D`, or `P`).
    pub width: usize,
    /// State columns per workgroup (the workgroup width).
    pub cols: usize,
    pub phase: ScanPhase,
}

impl ScanSpec {
    pub fn col_blocks(&self) -> usize {
        self.width.div_ceil(self.cols)
    }

    pub fn workgroups(&self) -> usize {
        self.heads * self.col_blocks()
    }

    /// Static workgroup memory the state columns need.
    pub fn lds_bytes(&self) -> u64 {
        (self.rows * self.cols * 4) as u64
    }

    /// The serial work of this launch: every lane steps once per token of its phase, over about three
    /// passes of the state column.
    pub(crate) fn serial_work(&self) -> SerialWork {
        SerialWork {
            threads: (self.workgroups() * self.cols) as u64,
            steps_per_thread: (self.phase.count * 3 * self.rows) as u64,
        }
    }

    /// The most tokens one launch may cover so its serial work stays within `budget`, at least one.
    pub fn tokens_within(&self, budget: u64) -> usize {
        let per_token = SerialWork {
            threads: (self.workgroups() * self.cols) as u64,
            steps_per_thread: (3 * self.rows) as u64,
        }
        .total()
        .max(1);
        usize::try_from(budget / per_token)
            .unwrap_or(usize::MAX)
            .max(1)
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
        ("phase.count", spec.phase.count),
    ] {
        if value == 0 {
            return Err(below_one(what, value));
        }
    }
    let end = spec.phase.start.saturating_add(spec.phase.count);
    if end > spec.len {
        return Err(KernelGenError::ExceedsBound {
            generator: GENERATOR,
            what: "phase end".to_string(),
            value: end,
            bound: spec.len,
        });
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

/// Build the scan kernel `spec` describes. The launch is `spec.workgroups()` workgroups of `cols` threads
/// laid over a 2-D grid `x_groups` workgroups across: `gi = GroupY * x_groups + GroupX` is the flat
/// `(head, column block)` index, and a workgroup past `workgroups()` returns.
pub(crate) fn scan_body(
    name: &str,
    spec: &ScanSpec,
    x_groups: usize,
) -> Result<Body, KernelGenError> {
    validate(spec)?;
    let ScanSpec {
        kernel,
        len,
        rows,
        width,
        cols,
        phase,
        ..
    } = *spec;
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
    let lane = c.index(IndexAxis::LocalX);
    let row_offset = uc(&mut c, BinOp::Mul, group_y, x_groups);
    let gi = ubin(&mut c, BinOp::Add, row_offset, group_x);
    let in_range = c.bin(Ty::Bool, BinOp::Lt, local_op(gi), cu(spec.workgroups()));
    c.if_then(in_range, |c| {
        let h = uc(c, BinOp::Div, gi, blocks);
        let block = uc(c, BinOp::Rem, gi, blocks);
        let block_start = uc(c, BinOp::Mul, block, cols);
        let col = ubin(c, BinOp::Add, block_start, lane);
        let live = c.bin(Ty::Bool, BinOp::Lt, local_op(col), cu(width));
        c.if_then(live, |c| {
            // The packed result's per-head origin and the carried state's rows within it.
            let out_base = uc(c, BinOp::Mul, h, (len + rows) * width);
            let state_at = |c: &mut Cfg, r: poot_kernel_ir::Local| {
                let row = uc(c, BinOp::Add, r, len);
                let row = uc(c, BinOp::Mul, row, width);
                let at = ubin(c, BinOp::Add, out_base, row);
                ubin(c, BinOp::Add, at, col)
            };
            // Load this lane's state column.
            count(c, cu(0), rows, 1, |c, r| {
                let value = if phase.start == 0 {
                    let head = uc(c, BinOp::Mul, h, rows * width);
                    let row = uc(c, BinOp::Mul, r, width);
                    let at = ubin(c, BinOp::Add, head, row);
                    let at = ubin(c, BinOp::Add, at, col);
                    c.let_(Ty::F32, Rvalue::Use(copy(elem(state_in, at))))
                } else {
                    let at = state_at(c, r);
                    c.let_(Ty::F32, Rvalue::Use(copy(elem(out, at))))
                };
                let slot = uc(c, BinOp::Mul, r, cols);
                let slot = ubin(c, BinOp::Add, slot, lane);
                lds_write(c, STATE, slot, local_op(value));
            });
            count_to(
                c,
                cu(phase.start),
                cu(phase.start + phase.count),
                1,
                |c, t| {
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
                    let oi = uc(c, BinOp::Mul, t, width);
                    let oi = ubin(c, BinOp::Add, out_base, oi);
                    let oi = ubin(c, BinOp::Add, oi, col);
                    c.assign(elem(out, oi), Rvalue::Use(local_op(o_acc)));
                },
            );
            // Publish the state column into the packed result's trailing rows.
            count(c, cu(0), rows, 1, |c, r| {
                let slot = uc(c, BinOp::Mul, r, cols);
                let slot = ubin(c, BinOp::Add, slot, lane);
                let s = lds_read(c, STATE, slot);
                let at = state_at(c, r);
                c.assign(elem(out, at), Rvalue::Use(local_op(s)));
            });
        });
    });
    let mut body = c.finish(name);
    body.workgroup_size = [cols as u32, 1, 1];
    body.workgroup_locals = vec![f32_array(rows * cols)];
    Ok(body)
}

fn state_slot(
    c: &mut Cfg,
    r: poot_kernel_ir::Local,
    cols: usize,
    lane: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let slot = uc(c, BinOp::Mul, r, cols);
    ubin(c, BinOp::Add, slot, lane)
}

/// One delta-rule token for this lane's column; returns `o[col]`. `ops` is `[q, k, v, g, beta]`.
fn delta_rule_step(
    c: &mut Cfg,
    spec: &ScanSpec,
    ops: [poot_kernel_ir::Local; 5],
    pos: poot_kernel_ir::Local,
    col: poot_kernel_ir::Local,
    lane: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let [q, k, v, g, beta] = ops;
    let (d, cols) = (spec.rows, spec.cols);
    let row = uc(c, BinOp::Mul, pos, d);
    let g_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(g, pos))));
    let decay = c.let_(Ty::F32, Rvalue::MathUnary(MathOp::Exp, local_op(g_t)));
    let beta_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(beta, pos))));
    // kv = k . (decay * S[:,col])
    let kv = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    count(c, cu(0), d, 1, |c, r| {
        let slot = state_slot(c, r, cols, lane);
        let s = lds_read(c, STATE, slot);
        let s = fbin(c, BinOp::Mul, s, decay);
        let ki = ubin(c, BinOp::Add, row, r);
        let kr = c.let_(Ty::F32, Rvalue::Use(copy(elem(k, ki))));
        let term = fbin(c, BinOp::Mul, kr, s);
        let sum = fbin(c, BinOp::Add, kv, term);
        c.assign(Place::local(kv), Rvalue::Use(local_op(sum)));
    });
    let vi = ubin(c, BinOp::Add, row, col);
    let v_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(v, vi))));
    let residual = fbin(c, BinOp::Sub, v_t, kv);
    let delta = fbin(c, BinOp::Mul, beta_t, residual);
    // S[r] <- decay * S[r] + k[r] * delta; o = q_scaled . S_new
    let scale = 1.0 / (d as f32).sqrt();
    let o_acc = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    count(c, cu(0), d, 1, |c, r| {
        let slot = state_slot(c, r, cols, lane);
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
        let sum = fbin(c, BinOp::Add, o_acc, term);
        c.assign(Place::local(o_acc), Rvalue::Use(local_op(sum)));
    });
    o_acc
}

/// One SSD token for this lane's column; returns `y[col]`. `ops` is `[x, b, c, delta, a, d]`.
#[allow(clippy::too_many_arguments)]
fn ssd_step(
    c: &mut Cfg,
    spec: &ScanSpec,
    ops: [poot_kernel_ir::Local; 6],
    pos: poot_kernel_ir::Local,
    head: poot_kernel_ir::Local,
    col: poot_kernel_ir::Local,
    lane: poot_kernel_ir::Local,
) -> poot_kernel_ir::Local {
    let [x, b_in, c_in, delta, a, d_skip] = ops;
    let (n, p, cols) = (spec.rows, spec.width, spec.cols);
    let x_row = uc(c, BinOp::Mul, pos, p);
    let xi = ubin(c, BinOp::Add, x_row, col);
    let x_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(x, xi))));
    let bc_row = uc(c, BinOp::Mul, pos, n);
    let delta_t = c.let_(Ty::F32, Rvalue::Use(copy(elem(delta, pos))));
    let a_h = c.let_(Ty::F32, Rvalue::Use(copy(elem(a, head))));
    let gate = fbin(c, BinOp::Mul, delta_t, a_h);
    let abar = c.let_(Ty::F32, Rvalue::MathUnary(MathOp::Exp, local_op(gate)));
    let dx = fbin(c, BinOp::Mul, delta_t, x_t);
    let y_acc = c.var(Ty::F32, Rvalue::Use(f32c(0.0)));
    count(c, cu(0), n, 1, |c, r| {
        let slot = state_slot(c, r, cols, lane);
        let s = lds_read(c, STATE, slot);
        let s = fbin(c, BinOp::Mul, s, abar);
        let bi = ubin(c, BinOp::Add, bc_row, r);
        let br = c.let_(Ty::F32, Rvalue::Use(copy(elem(b_in, bi))));
        let bump = fbin(c, BinOp::Mul, br, dx);
        let s_new = fbin(c, BinOp::Add, s, bump);
        lds_write(c, STATE, slot, local_op(s_new));
        let cr = c.let_(Ty::F32, Rvalue::Use(copy(elem(c_in, bi))));
        let term = fbin(c, BinOp::Mul, cr, s_new);
        let sum = fbin(c, BinOp::Add, y_acc, term);
        c.assign(Place::local(y_acc), Rvalue::Use(local_op(sum)));
    });
    let d_h = c.let_(Ty::F32, Rvalue::Use(copy(elem(d_skip, head))));
    let skip = fbin(c, BinOp::Mul, d_h, x_t);
    fbin(c, BinOp::Add, y_acc, skip)
}
