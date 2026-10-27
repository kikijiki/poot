//! The one data-movement generator: one thread owns one output buffer unit (an element, or a `u32` word of
//! packed lanes), computes the logical input element each of its lanes copies, and stores the unit. The
//! storage decides how a lane is read and written; the op decides which element a lane copies. An in-place
//! write (a row scatter, a donated update's window) owns only the units it writes.

use poot_kernel_ir::{BinOp, Body, Local, LocalDecl, Operand, Place, Rvalue, Ty};

use super::builder::Cfg;
use super::spec::{MovementOp, MovementSpec, Storage, UpdateIndex, rows_of};
use crate::KernelGenError;
use crate::emit::{c32, copy, cu};
use crate::helpers::{broadcast_eff_strides, elem, ld, local, row_major_strides, slice_dtype};

/// A kernel buffer and the logical shape it holds.
pub(crate) struct Buf {
    pub(crate) param: Local,
    pub(crate) shape: Vec<usize>,
}

/// Build the kernel `spec` names. The launch is [`MovementSpec::threads`] threads; `x_groups` is the fold
/// the launch uses past the device's x-grid cap ([`crate::Fold`]), `None` for a flat launch.
pub fn movement_body(
    name: &str,
    spec: &MovementSpec,
    x_groups: Option<usize>,
) -> Result<Body, KernelGenError> {
    spec.validate()?;
    let storage = &spec.storage;
    let unit_ty = storage.unit_ty();
    let slice = |ty: &Ty, mutable| ld(slice_dtype(ty.clone(), mutable), mutable);

    // Parameters: the thread unit, the data inputs, an index buffer where the op reads one, the output.
    let mut params: Vec<LocalDecl> = vec![ld(Ty::Unit, false)];
    let mut data_shapes: Vec<Vec<usize>> = Vec::new();
    let mut reads_index = false;
    match &spec.op {
        MovementOp::Transpose { in_shape, .. }
        | MovementOp::Slice { in_shape, .. }
        | MovementOp::Broadcast { in_shape }
        | MovementOp::Reshape { in_shape } => data_shapes.push(in_shape.clone()),
        MovementOp::Gather { in_shape, .. } => {
            data_shapes.push(in_shape.clone());
            reads_index = true;
        }
        MovementOp::Concat { in_shapes, .. } => data_shapes.extend(in_shapes.iter().cloned()),
        MovementOp::UpdateSlice {
            update_shape,
            index,
            ..
        } => {
            data_shapes.push(spec.out_shape.clone());
            data_shapes.push(update_shape.clone());
            reads_index = matches!(index, UpdateIndex::Runtime);
        }
        // The output buffer already holds the operand: the kernel reads only the update (and its index).
        MovementOp::WriteSlice {
            update_shape,
            index,
            ..
        } => {
            data_shapes.push(update_shape.clone());
            reads_index = matches!(index, UpdateIndex::Runtime);
        }
        // The rows are written in place into the output buffer, which already holds the base: the
        // kernel reads the source rows and the slot map, never the base.
        MovementOp::ScatterRows { src_shape, .. } => {
            data_shapes.push(src_shape.clone());
            reads_index = true;
        }
    }
    // The input an update or a row scatter writes from: its element type is `written` when the write converts.
    let written_input = match &spec.op {
        MovementOp::UpdateSlice { .. } => Some(1),
        MovementOp::WriteSlice { .. } | MovementOp::ScatterRows { .. } => Some(0),
        _ => None,
    };
    let mut inputs = Vec::new();
    for (position, shape) in data_shapes.into_iter().enumerate() {
        let ty = match &spec.written {
            Some(written) if written_input == Some(position) => written,
            _ => &unit_ty,
        };
        params.push(slice(ty, false));
        inputs.push(Buf {
            param: local(params.len() as u32 - 1),
            shape,
        });
    }
    // The index buffer follows the data inputs and precedes the output: the order is always data inputs,
    // index, output.
    let index = reads_index.then(|| {
        params.push(slice(&Ty::I32, false));
        local(params.len() as u32 - 1)
    });
    params.push(slice(&unit_ty, true));
    let out = Buf {
        param: local(params.len() as u32 - 1),
        shape: spec.out_shape.clone(),
    };

    let mut cfg = Cfg::new(params);
    let thread = cfg.thread_index(x_groups);
    match &spec.op {
        MovementOp::ScatterRows { src_shape, window } => {
            // One thread owns one source unit (an element, or a packed `u32` word): it copies the unit to
            // the same offset of the pool slot its source row names, or nothing for a padding row. A unit
            // of the output is written by at most one thread because the map holds each slot once.
            let units = storage.units(src_shape);
            let per_slot = storage.units(&spec.out_shape[1..]);
            let slots = index.expect("a scatter reads its slot map");
            let unit = match window {
                Some(window) if window.offset != 0 => {
                    cfg.bin(Ty::Usize, BinOp::Add, copy(thread), cu(window.offset))
                }
                _ => thread,
            };
            let in_bounds = cfg.bin(Ty::Bool, BinOp::Lt, copy(unit), cu(units));
            cfg.if_then(in_bounds, |cfg| {
                let row = cfg.bin(Ty::Usize, BinOp::Div, copy(unit), cu(per_slot));
                let column = cfg.bin(Ty::Usize, BinOp::Rem, copy(unit), cu(per_slot));
                let slot = cfg.let_(Ty::I32, Rvalue::Use(Operand::Copy(elem(slots, row))));
                let writes = cfg.bin(
                    Ty::Bool,
                    BinOp::Ge,
                    copy(slot),
                    Operand::Const(poot_kernel_ir::Constant::I32(0)),
                );
                cfg.if_then(writes, |cfg| {
                    let slot = cfg.let_(
                        Ty::Usize,
                        Rvalue::Cast {
                            to: Ty::Usize,
                            operand: copy(slot),
                        },
                    );
                    let first = cfg.bin(Ty::Usize, BinOp::Mul, copy(slot), cu(per_slot));
                    let destination = cfg.bin(Ty::Usize, BinOp::Add, copy(first), copy(column));
                    let value = read_written(cfg, spec, inputs[0].param, unit);
                    cfg.assign(elem(out.param, destination), Rvalue::Use(copy(value)));
                });
            });
        }
        MovementOp::WriteSlice {
            update_shape,
            axis,
            index: update_index,
        } => {
            let window = Window {
                spec,
                storage,
                out: &out,
                update: &inputs[0],
                axis: *axis,
            };
            let in_bounds = cfg.bin(Ty::Bool, BinOp::Lt, copy(thread), cu(spec.threads()));
            cfg.if_then(in_bounds, |cfg| {
                let start = update_start(cfg, update_index, index);
                if spec.updates_inside_words(*axis) {
                    let extent = update_shape[*axis];
                    let span = spec.row_window_words(extent);
                    window.write_row_word(cfg, thread, start, extent, span);
                } else {
                    window.write_unit(cfg, thread, start);
                }
            });
        }
        _ => {
            let units = storage.units(&spec.out_shape);
            let unit = thread;
            let in_bounds = cfg.bin(Ty::Bool, BinOp::Lt, copy(unit), cu(units));
            cfg.if_then(in_bounds, |cfg| {
                let source = Source {
                    spec,
                    inputs: &inputs,
                    index,
                };
                match storage {
                    Storage::Elements { .. } => {
                        let value = source.value(cfg, unit);
                        cfg.assign(elem(out.param, unit), Rvalue::Use(copy(value)));
                    }
                    Storage::PackedRows { lane_bits } => {
                        write_packed_word(cfg, &out, unit, *lane_bits, |cfg, flat| {
                            source.value(cfg, flat)
                        });
                    }
                }
            });
        }
    }
    Ok(cfg.finish(name))
}

/// Compose the word of packed lanes thread `unit` owns and store it. `value(flat)` emits the lane value
/// (in the low `lane_bits` of a `u32`) of the logical output element `flat`. A lane past the row's last
/// element (the row's padding) is left zero and never evaluates `value`.
pub(crate) fn write_packed_word(
    cfg: &mut Cfg,
    out: &Buf,
    unit: Local,
    lane_bits: u32,
    mut value: impl FnMut(&mut Cfg, Local) -> Local,
) {
    let lanes = (32 / lane_bits) as usize;
    let (_, row_len) = rows_of(&out.shape);
    let words_per_row = row_len.div_ceil(lanes);
    let row = cfg.bin(Ty::Usize, BinOp::Div, copy(unit), cu(words_per_row));
    let word_col = cfg.bin(Ty::Usize, BinOp::Rem, copy(unit), cu(words_per_row));
    let first_col = cfg.bin(Ty::Usize, BinOp::Mul, copy(word_col), cu(lanes));
    let row_base = cfg.bin(Ty::Usize, BinOp::Mul, copy(row), cu(row_len));
    let word = cfg.var(Ty::U32, Rvalue::Use(c32(0)));
    for lane in 0..lanes {
        let mut place = |cfg: &mut Cfg| {
            let col = cfg.bin(Ty::Usize, BinOp::Add, copy(first_col), cu(lane));
            let flat = cfg.bin(Ty::Usize, BinOp::Add, copy(row_base), copy(col));
            let lane_value = value(cfg, flat);
            let moved = if lane == 0 {
                lane_value
            } else {
                cfg.bin(
                    Ty::U32,
                    BinOp::Shl,
                    copy(lane_value),
                    c32(lane as u32 * lane_bits),
                )
            };
            let combined = cfg.bin(Ty::U32, BinOp::BitOr, copy(word), copy(moved));
            cfg.assign(Place::local(word), Rvalue::Use(copy(combined)));
        };
        // Lane 0 always holds a row element; a later lane may be the row's padding.
        if lane > 0 && !row_len.is_multiple_of(lanes) {
            let col = cfg.bin(Ty::Usize, BinOp::Add, copy(first_col), cu(lane));
            let in_row = cfg.bin(Ty::Bool, BinOp::Lt, copy(col), cu(row_len));
            cfg.if_then(in_row, place);
        } else {
            place(cfg);
        }
    }
    cfg.assign(elem(out.param, unit), Rvalue::Use(copy(word)));
}

/// How one output lane finds its source: the logical flat index of the output element it writes in, the
/// value it copies out.
struct Source<'a> {
    spec: &'a MovementSpec,
    inputs: &'a [Buf],
    index: Option<Local>,
}

impl Source<'_> {
    /// The value of the logical output element `flat`: the bits its source element holds (a carrier-typed
    /// local for element storage, the lane's value in the low bits of a `u32` for packed rows).
    fn value(&self, cfg: &mut Cfg, flat: Local) -> Local {
        let storage = &self.spec.storage;
        let out_shape = &self.spec.out_shape;
        match &self.spec.op {
            MovementOp::Transpose { in_shape, perm } => {
                let in_strides = row_major_strides(in_shape);
                let terms: Vec<usize> = perm.iter().map(|&axis| in_strides[axis]).collect();
                let src = remap_index(cfg, out_shape, &terms, 0, flat);
                read_lane(cfg, storage, &self.inputs[0], src)
            }
            MovementOp::Slice {
                in_shape,
                axis,
                start,
            } => {
                let in_strides = row_major_strides(in_shape);
                let src = remap_index(cfg, out_shape, &in_strides, start * in_strides[*axis], flat);
                read_lane(cfg, storage, &self.inputs[0], src)
            }
            MovementOp::Broadcast { in_shape } => {
                let terms = broadcast_eff_strides(out_shape, in_shape);
                let src = remap_index(cfg, out_shape, &terms, 0, flat);
                read_lane(cfg, storage, &self.inputs[0], src)
            }
            MovementOp::Reshape { .. } => read_lane(cfg, storage, &self.inputs[0], flat),
            MovementOp::Gather {
                in_shape,
                axis,
                index_numel,
                ..
            } => {
                let index = self.index.expect("a gather reads an index buffer");
                let inner: usize = in_shape[axis + 1..].iter().product();
                let outer_extent: usize = in_shape[..*axis].iter().product();
                let axis_len = in_shape[*axis];
                let mid = if inner == 1 {
                    flat
                } else {
                    cfg.bin(Ty::Usize, BinOp::Div, copy(flat), cu(inner))
                };
                let position = if outer_extent == 1 {
                    mid
                } else {
                    cfg.bin(Ty::Usize, BinOp::Rem, copy(mid), cu(*index_numel))
                };
                let picked = read_index(cfg, index, position);
                let mut src = if inner == 1 {
                    picked
                } else {
                    cfg.bin(Ty::Usize, BinOp::Mul, copy(picked), cu(inner))
                };
                if outer_extent != 1 {
                    let outer = cfg.bin(Ty::Usize, BinOp::Div, copy(mid), cu(*index_numel));
                    let outer_base =
                        cfg.bin(Ty::Usize, BinOp::Mul, copy(outer), cu(axis_len * inner));
                    src = cfg.bin(Ty::Usize, BinOp::Add, copy(outer_base), copy(src));
                }
                if inner != 1 {
                    let inner_i = cfg.bin(Ty::Usize, BinOp::Rem, copy(flat), cu(inner));
                    src = cfg.bin(Ty::Usize, BinOp::Add, copy(src), copy(inner_i));
                }
                read_lane(cfg, storage, &self.inputs[0], src)
            }
            MovementOp::Concat { axis, in_shapes } => {
                let axis_split = AxisSplit::of(out_shape, *axis);
                let (outer, a, inner_i) = axis_split.coords(cfg, flat);
                // Input 0 is read at a clamped coordinate so every thread has a valid first value; each
                // later input overwrites it for the coordinates it owns. The selection is a flat run of
                // single `if` regions, never an N-deep ladder.
                let first_extent = in_shapes[0][*axis];
                let clamped = cfg.bin(Ty::Usize, BinOp::Min, copy(a), cu(first_extent - 1));
                let first_src = axis_split.source(cfg, outer, clamped, inner_i, first_extent);
                let first = read_lane(cfg, storage, &self.inputs[0], first_src);
                let value = cfg.var(storage.unit_ty(), Rvalue::Use(copy(first)));
                let mut start = first_extent;
                for (k, shape) in in_shapes.iter().enumerate().skip(1) {
                    let extent = shape[*axis];
                    let at_least = cfg.bin(Ty::Bool, BinOp::Ge, copy(a), cu(start));
                    let below = cfg.bin(Ty::Bool, BinOp::Lt, copy(a), cu(start + extent));
                    let owns = cfg.both(at_least, below);
                    cfg.if_then(owns, |cfg| {
                        let within = cfg.bin(Ty::Usize, BinOp::Sub, copy(a), cu(start));
                        let src = axis_split.source(cfg, outer, within, inner_i, extent);
                        let moved = read_lane(cfg, storage, &self.inputs[k], src);
                        cfg.assign(Place::local(value), Rvalue::Use(copy(moved)));
                    });
                    start += extent;
                }
                value
            }
            MovementOp::UpdateSlice {
                update_shape,
                axis,
                index: update_index,
            } => {
                let axis_split = AxisSplit::of(out_shape, *axis);
                let (outer, a, inner_i) = axis_split.coords(cfg, flat);
                let extent = update_shape[*axis];
                let kept = read_lane(cfg, storage, &self.inputs[0], flat);
                let value = cfg.var(storage.unit_ty(), Rvalue::Use(copy(kept)));
                let start = update_start(cfg, update_index, self.index);
                let end = cfg.bin(Ty::Usize, BinOp::Add, copy(start), cu(extent));
                let at_least = cfg.bin(Ty::Bool, BinOp::Ge, copy(a), copy(start));
                let below = cfg.bin(Ty::Bool, BinOp::Lt, copy(a), copy(end));
                let inside = cfg.both(at_least, below);
                cfg.if_then(inside, |cfg| {
                    let within = cfg.bin(Ty::Usize, BinOp::Sub, copy(a), copy(start));
                    let src = axis_split.source(cfg, outer, within, inner_i, extent);
                    let written = match self.spec.written {
                        Some(_) => read_written(cfg, self.spec, self.inputs[1].param, src),
                        None => read_lane(cfg, storage, &self.inputs[1], src),
                    };
                    cfg.assign(Place::local(value), Rvalue::Use(copy(written)));
                });
                value
            }
            MovementOp::ScatterRows { .. } | MovementOp::WriteSlice { .. } => {
                unreachable!(
                    "an in-place write is planned by the units it writes, not by output lane"
                )
            }
        }
    }
}

/// Unit `unit` of the buffer `written` an update or a row scatter writes from, as the output's carrier: the
/// value itself, or (when `spec.written` names the buffer's element type) that value converted, the narrowing
/// store of f32 values into a two-byte buffer.
fn read_written(cfg: &mut Cfg, spec: &MovementSpec, written: Local, unit: Local) -> Local {
    let carrier = spec.storage.unit_ty();
    let Some(source) = &spec.written else {
        return cfg.let_(carrier, Rvalue::Use(Operand::Copy(elem(written, unit))));
    };
    let value = cfg.let_(
        source.clone(),
        Rvalue::Use(Operand::Copy(elem(written, unit))),
    );
    cfg.let_(
        carrier.clone(),
        Rvalue::Cast {
            to: carrier,
            operand: copy(value),
        },
    )
}

/// Where an update starts on its axis: the baked position, or the first element of the `I32` index buffer.
fn update_start(cfg: &mut Cfg, update_index: &UpdateIndex, index: Option<Local>) -> Local {
    match update_index {
        UpdateIndex::Static(start) => cfg.usize_const(*start),
        UpdateIndex::Runtime => {
            let index = index.expect("a runtime update reads an index");
            let first = cfg.usize_const(0);
            read_index(cfg, index, first)
        }
    }
}

/// An in-place update's window of the output buffer, which already holds the operand: the update is
/// copied into it and nothing else of the output is read or written.
struct Window<'a> {
    spec: &'a MovementSpec,
    storage: &'a Storage,
    out: &'a Buf,
    update: &'a Buf,
    axis: usize,
}

impl Window<'_> {
    /// The buffer-unit shape of a tensor of logical `shape`: the shape itself, with the last axis counted in
    /// words for packed lanes.
    fn unit_shape(&self, shape: &[usize]) -> Vec<usize> {
        let mut units = shape.to_vec();
        if let Some(row_len) = units.last_mut() {
            *row_len = row_len.div_ceil(self.storage.lanes());
        }
        units
    }

    /// Copy update unit `unit` to its place in the output, `start` positions along the axis. Off the row axis
    /// of packed lanes the update's units are whole output units: the copy moves a unit's bits unchanged.
    fn write_unit(&self, cfg: &mut Cfg, unit: Local, start: Local) {
        let update_units = self.unit_shape(&self.update.shape);
        let out_extent = self.out.shape[self.axis];
        let split = AxisSplit::of(&update_units, self.axis);
        let (outer, within, inner_i) = split.coords(cfg, unit);
        let at = cfg.bin(Ty::Usize, BinOp::Add, copy(within), copy(start));
        let inside = cfg.bin(Ty::Bool, BinOp::Lt, copy(at), cu(out_extent));
        cfg.if_then(inside, |cfg| {
            let destination = split.source(cfg, outer, at, inner_i, out_extent);
            let value = read_written(cfg, self.spec, self.update.param, unit);
            cfg.assign(elem(self.out.param, destination), Rvalue::Use(copy(value)));
        });
    }

    /// Write word `thread` of the window of `extent` lanes along the row axis of packed lanes: thread
    /// `row * span + j` owns word `start / lanes + j` of output row `row`. The window can start and end
    /// inside a word, so the word keeps the lanes the output already holds outside it. One thread owns each
    /// word, so no two write one.
    fn write_row_word(
        &self,
        cfg: &mut Cfg,
        thread: Local,
        start: Local,
        extent: usize,
        span: usize,
    ) {
        let Storage::PackedRows { lane_bits } = *self.storage else {
            unreachable!("only packed lanes share a word along the row axis")
        };
        let lanes = self.storage.lanes();
        let (_, row_len) = rows_of(&self.out.shape);
        let words_per_row = row_len.div_ceil(lanes);
        let row = cfg.bin(Ty::Usize, BinOp::Div, copy(thread), cu(span));
        let j = cfg.bin(Ty::Usize, BinOp::Rem, copy(thread), cu(span));
        let first = cfg.bin(Ty::Usize, BinOp::Div, copy(start), cu(lanes));
        let word_col = cfg.bin(Ty::Usize, BinOp::Add, copy(first), copy(j));
        let end = cfg.bin(Ty::Usize, BinOp::Add, copy(start), cu(extent));
        let first_lane = cfg.bin(Ty::Usize, BinOp::Mul, copy(word_col), cu(lanes));
        let in_row = cfg.bin(Ty::Bool, BinOp::Lt, copy(word_col), cu(words_per_row));
        let before_end = cfg.bin(Ty::Bool, BinOp::Lt, copy(first_lane), copy(end));
        let overlaps = cfg.both(in_row, before_end);
        cfg.if_then(overlaps, |cfg| {
            let row_base = cfg.bin(Ty::Usize, BinOp::Mul, copy(row), cu(words_per_row));
            let word_index = cfg.bin(Ty::Usize, BinOp::Add, copy(row_base), copy(word_col));
            let held = cfg.let_(
                Ty::U32,
                Rvalue::Use(Operand::Copy(elem(self.out.param, word_index))),
            );
            let word = cfg.var(Ty::U32, Rvalue::Use(copy(held)));
            let update_row = cfg.bin(Ty::Usize, BinOp::Mul, copy(row), cu(extent));
            for lane in 0..lanes {
                let col = cfg.bin(Ty::Usize, BinOp::Add, copy(first_lane), cu(lane));
                let at_least = cfg.bin(Ty::Bool, BinOp::Ge, copy(col), copy(start));
                let below_end = cfg.bin(Ty::Bool, BinOp::Lt, copy(col), copy(end));
                let in_window = cfg.both(at_least, below_end);
                let in_row = cfg.bin(Ty::Bool, BinOp::Lt, copy(col), cu(row_len));
                let writes = cfg.both(in_window, in_row);
                cfg.if_then(writes, |cfg| {
                    let within = cfg.bin(Ty::Usize, BinOp::Sub, copy(col), copy(start));
                    let flat = cfg.bin(Ty::Usize, BinOp::Add, copy(update_row), copy(within));
                    let value = read_lane(cfg, self.storage, self.update, flat);
                    let shift = lane as u32 * lane_bits;
                    let mask = ((1u32 << lane_bits) - 1) << shift;
                    let kept = cfg.bin(Ty::U32, BinOp::BitAnd, copy(word), c32(!mask));
                    let moved = cfg.bin(Ty::U32, BinOp::Shl, copy(value), c32(shift));
                    let combined = cfg.bin(Ty::U32, BinOp::BitOr, copy(kept), copy(moved));
                    cfg.assign(Place::local(word), Rvalue::Use(copy(combined)));
                });
            }
            cfg.assign(elem(self.out.param, word_index), Rvalue::Use(copy(word)));
        });
    }
}

/// An output shape split around `axis` into `(outer, axis coordinate, inner)`: the decomposition a concat
/// and an update-slice share.
struct AxisSplit {
    inner: usize,
    extent: usize,
}

impl AxisSplit {
    fn of(shape: &[usize], axis: usize) -> Self {
        Self {
            inner: shape[axis + 1..].iter().product(),
            extent: shape[axis],
        }
    }

    /// `(outer, axis coordinate, inner coordinate)` of the output element `flat`.
    fn coords(&self, cfg: &mut Cfg, flat: Local) -> (Local, Local, Local) {
        let inner_i = cfg.bin(Ty::Usize, BinOp::Rem, copy(flat), cu(self.inner));
        let above = cfg.bin(Ty::Usize, BinOp::Div, copy(flat), cu(self.inner));
        let a = cfg.bin(Ty::Usize, BinOp::Rem, copy(above), cu(self.extent));
        let outer = cfg.bin(Ty::Usize, BinOp::Div, copy(above), cu(self.extent));
        (outer, a, inner_i)
    }

    /// The flat index, in an input whose extent on the axis is `extent`, of `(outer, at, inner_i)`.
    fn source(
        &self,
        cfg: &mut Cfg,
        outer: Local,
        at: Local,
        inner_i: Local,
        extent: usize,
    ) -> Local {
        let rows = cfg.bin(Ty::Usize, BinOp::Mul, copy(outer), cu(extent));
        let row = cfg.bin(Ty::Usize, BinOp::Add, copy(rows), copy(at));
        let base = cfg.bin(Ty::Usize, BinOp::Mul, copy(row), cu(self.inner));
        cfg.bin(Ty::Usize, BinOp::Add, copy(base), copy(inner_i))
    }
}

/// `base + sum_d coordinate_d(flat) * terms[d]`: the input flat index of an output element whose source
/// moves by `terms[d]` per step along output axis `d`. An axis with a zero term or a unit extent adds
/// nothing.
fn remap_index(
    cfg: &mut Cfg,
    out_shape: &[usize],
    terms: &[usize],
    base: usize,
    flat: Local,
) -> Local {
    let out_strides = row_major_strides(out_shape);
    let mut src = cfg.usize_const(base);
    for (d, &term) in terms.iter().enumerate() {
        if term == 0 || out_shape[d] == 1 {
            continue;
        }
        let above = cfg.bin(Ty::Usize, BinOp::Div, copy(flat), cu(out_strides[d]));
        let coordinate = cfg.bin(Ty::Usize, BinOp::Rem, copy(above), cu(out_shape[d]));
        let step = cfg.bin(Ty::Usize, BinOp::Mul, copy(coordinate), cu(term));
        src = cfg.bin(Ty::Usize, BinOp::Add, copy(src), copy(step));
    }
    src
}

/// Element `position` of an `I32` index buffer, as a `usize`.
fn read_index(cfg: &mut Cfg, buffer: Local, position: Local) -> Local {
    let value = cfg.let_(Ty::I32, Rvalue::Use(Operand::Copy(elem(buffer, position))));
    cfg.let_(
        Ty::Usize,
        Rvalue::Cast {
            to: Ty::Usize,
            operand: copy(value),
        },
    )
}

/// The bits of logical element `flat` of `buf`: its buffer element for element storage, the lane's value
/// in the low bits of a `u32` for packed rows (the row's padding is never read).
pub(crate) fn read_lane(cfg: &mut Cfg, storage: &Storage, buf: &Buf, flat: Local) -> Local {
    match storage {
        Storage::Elements { carrier } => cfg.let_(
            carrier.clone(),
            Rvalue::Use(Operand::Copy(elem(buf.param, flat))),
        ),
        Storage::PackedRows { lane_bits } => {
            let lanes = storage.lanes();
            let (_, row_len) = rows_of(&buf.shape);
            let words_per_row = row_len.div_ceil(lanes);
            let row = cfg.bin(Ty::Usize, BinOp::Div, copy(flat), cu(row_len));
            let col = cfg.bin(Ty::Usize, BinOp::Rem, copy(flat), cu(row_len));
            let word_col = cfg.bin(Ty::Usize, BinOp::Div, copy(col), cu(lanes));
            let row_base = cfg.bin(Ty::Usize, BinOp::Mul, copy(row), cu(words_per_row));
            let word_index = cfg.bin(Ty::Usize, BinOp::Add, copy(row_base), copy(word_col));
            let lane = cfg.bin(Ty::Usize, BinOp::Rem, copy(col), cu(lanes));
            let shift_bits = cfg.bin(Ty::Usize, BinOp::Mul, copy(lane), cu(*lane_bits as usize));
            let shift = cfg.let_(
                Ty::U32,
                Rvalue::Cast {
                    to: Ty::U32,
                    operand: copy(shift_bits),
                },
            );
            let word = cfg.let_(
                Ty::U32,
                Rvalue::Use(Operand::Copy(elem(buf.param, word_index))),
            );
            let shifted = cfg.bin(Ty::U32, BinOp::Shr, copy(word), copy(shift));
            let mask = (1u32 << lane_bits) - 1;
            cfg.bin(Ty::U32, BinOp::BitAnd, copy(shifted), c32(mask))
        }
    }
}
